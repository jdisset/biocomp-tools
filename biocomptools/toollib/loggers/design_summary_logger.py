"""Design summary logger: saves metrics and generates plots via batched evaluation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING
from pydantic import ConfigDict
import csv

from biocomp.design import get_topk_replicate_network_pairs
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.design_results import (
    DesignResultsManager,
    compute_design_metrics,
    NREMetrics,
)
from biocomptools.toollib.design_eval import is_valid_network
from biocomptools.toollib.design_pipeline import (
    CommitRequest,
    build_design_result,
    evaluate_design_inputs,
    invoke_design_summary_plot,
    make_design_input,
    precommit_pairs,
    resolve_commit_requests,
    save_network_recipe_yaml,
)
from biocomptools.toollib.design_data import prepare_target_data
from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from biocomptools.toollib.design_eval import EvaluatedDesign

logger = get_logger(__name__)


class DesignSummaryLogger(Logger):
    """Logger that saves design metrics and generates summary plots via batched evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_period: int = 500
    log_at_end: bool = True
    topk_per_target: int = 3
    output_formats: list[str] = ["png"]
    grid_resolution: tuple[int, int] = (48, 48)
    max_evals: int = 100000
    eval_seed: int = 42

    output_dir: str | None = None
    model: Any | None = None
    targets: list[Any] | None = None
    dmanager: Any | None = None
    design_conf: Any | None = None

    _results_manager: DesignResultsManager | None = None
    _loss_history: list[float] = []
    _step_count: int = 0
    _all_metrics: list[dict] = []
    _cached_stack: Any | None = None
    _run_name: str = ""

    def initialize(self, training_program=None):
        if training_program:
            if hasattr(training_program, 'design_conf'):
                self.design_conf = training_program.design_conf
        if self.output_dir:
            design_dir = Path(self.output_dir) / 'design'
            self._results_manager = DesignResultsManager(design_dir)
            self._all_metrics = []
            logger.info(f"DesignSummaryLogger initialized: {design_dir}")

    def _get_or_build_stack(self, stack: Any = None) -> Any:
        if stack is not None:
            self._cached_stack = stack
            return stack
        if self._cached_stack is not None:
            return self._cached_stack
        assert self.dmanager is not None, "dmanager required to build stack"
        assert self.model is not None, "model required to build stack"
        auto_lock = (
            getattr(self.design_conf, 'auto_lock_topology_tus', True)
            if self.design_conf
            else True
        )
        self._cached_stack = self.dmanager.build_stack(
            self.model, auto_lock_topology_tus=auto_lock
        )
        return self._cached_stack

    def _generate_summaries(
        self, step: int, params: Any, stack: Any, all_losses: np.ndarray, is_final: bool = False
    ):
        stack = self._get_or_build_stack(stack)
        assert self._results_manager is not None, "results manager required for summaries"
        assert stack is not None, "stack required for summaries"

        targets = self.targets or (self.dmanager.targets if self.dmanager else [])
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)
        n_targets = all_losses.shape[-2] if all_losses.ndim >= 2 else 1
        assert self.dmanager is not None, "dmanager required for SSOT top-k selector"
        assert self.design_conf is not None, "design_conf required for SSOT top-k selector"
        losses_for_topk = jnp.asarray(all_losses)
        topk_by_target = get_topk_replicate_network_pairs(
            losses=losses_for_topk,
            dmanager=self.dmanager,
            dconf=self.design_conf,
            k=self.topk_per_target,
        )

        # collect all candidates first for batching
        all_candidates = []
        for target_id in range(min(n_targets, len(targets))):
            target = targets[target_id]
            target_name = getattr(target, 'name', f'target_{target_id}')
            assert target_id < len(topk_by_target), (
                f"target index {target_id} out of bounds for topk results {len(topk_by_target)}"
            )
            candidates = topk_by_target[target_id]
            self._results_manager.save_rankings(
                target_name, candidates, step=None if is_final else step
            )

            for rank, (rep_id, net_id, loss) in enumerate(candidates, 1):
                all_candidates.append(
                    {
                        'target': target,
                        'target_id': target_id,
                        'target_name': target_name,
                        'rep_id': rep_id,
                        'net_id': net_id,
                        'loss': loss,
                        'rank': rank,
                        'step': step,
                        'is_final': is_final,
                        'params': params,
                        'stack': stack,
                    }
                )

        # batch process all candidates
        self._batch_process_candidates(all_candidates)

        if is_final:
            self._generate_comparison_outputs()

    def _batch_process_candidates(self, candidates: list[dict]):
        """Batch process all design candidates with single evaluation call."""
        assert candidates, "at least one candidate required"
        assert self.model is not None, "model required for batch candidate processing"

        params = candidates[0]['params']
        stack = candidates[0]['stack']
        assert all(c['params'] is params for c in candidates), "all candidates must share same params"
        assert all(c['stack'] is stack for c in candidates), "all candidates must share same stack"

        pairs = {(c['rep_id'], c['target_id']) for c in candidates}
        commit_cache, commit_failures = precommit_pairs(params, stack, pairs, fail_fast=True)
        assert not commit_failures, f"commit failures not allowed: {commit_failures}"

        commit_requests = [
            CommitRequest(
                rep_id=c['rep_id'],
                target_id=c['target_id'],
                net_id=c['net_id'],
                context={'candidate': c},
            )
            for c in candidates
        ]
        commit_results = resolve_commit_requests(commit_requests, commit_cache)
        committed_info = []
        for result in commit_results:
            info = result.request.context['candidate']
            assert result.error is None, (
                f"commit resolution failed for {info['target_name']} rank {info['rank']}: {result.error}"
            )
            network = result.network
            assert network is not None, "resolved committed network is None"
            committed_info.append(
                {
                    **info,
                    'network': network,
                    'valid': bool(is_valid_network(network)),
                }
            )

        # build DesignInput list for valid networks
        design_inputs = []
        input_to_candidate = {}
        assert self._results_manager is not None, "results manager required for rank output"
        for i, info in enumerate(committed_info):
            if not info['valid']:
                continue
            rank_dir = self._results_manager.get_rank_dir(
                info['target_name'], info['rank'], step=None if info['is_final'] else info['step']
            )
            inp = make_design_input(
                network=info['network'],
                target=info['target'],
                target_name=info['target_name'],
                rank=info['rank'],
                replicate=info['rep_id'],
                net_id=info['net_id'],
                loss=info['loss'],
                run_name=self._run_name,
                design_dir=str(rank_dir),
            )
            design_inputs.append(inp)
            input_to_candidate[len(design_inputs) - 1] = (i, info, rank_dir)

        assert design_inputs, "no valid networks to evaluate"

        # batch evaluate all designs
        evaluated = evaluate_design_inputs(self.model, design_inputs, max_evals=self.max_evals)

        # process results
        for idx, ev in enumerate(evaluated):
            i, info, rank_dir = input_to_candidate[idx]
            self._save_single_result(ev, info, rank_dir)

    def _save_single_result(self, ev: EvaluatedDesign, info: dict, rank_dir: Path):
        """Save a single evaluated design result."""
        if not ev.is_valid:
            return

        inp = ev.input
        network = inp.network
        target = inp.target

        # get prediction data for metrics
        td = prepare_target_data(target, max_samples=self.max_evals, seed=self.eval_seed)
        y_true = td.Y if td.Y is not None else np.zeros(td.n_samples)
        yval = ev.pred_data.yval
        y_pred = np.asarray(yval) if yval is not None and len(yval) > 0 else np.zeros_like(y_true)

        # compute fingerprint for committed network
        from biocomp.fingerprint import compute_fingerprint
        from biocomptools.modelmodel import NetworkModel

        assert self.model is not None, "model required for fingerprint computation"
        network_model = NetworkModel(model=self.model, network=network)
        fingerprint = compute_fingerprint(network_model)
        logger.debug(f"  [{inp.target_name} rank {inp.rank}] Fingerprint: {fingerprint}")

        # create NRE metrics
        nre_metrics = (
            NREMetrics(
                design_nre=ev.design_nre,
                baseline_nre=ev.baseline_nre,
            )
            if ev.design_nre is not None
            else None
        )

        if ev.design_nre is not None:
            baseline_str = f" (baseline: {ev.baseline_nre:.2f})" if ev.baseline_nre else ""
            logger.info(
                f"  [{inp.target_name} rank {inp.rank}] Design NRE: {ev.design_nre:.2f}{baseline_str}"
            )

        # compute and save metrics
        metrics = compute_design_metrics(
            y_true,
            y_pred,
            inp.loss,
            inp.target_name,
            inp.scaffold_network_name,
            inp.replicate,
            info['net_id'],
            inp.rank,
            info['step'],
            nre_metrics=nre_metrics,
            fingerprint=fingerprint,
        )
        metrics.to_json(rank_dir / 'metrics.json')
        self._all_metrics.append(metrics.to_dict())

        # save recipe
        self._save_recipe(network, rank_dir / 'recipe.yaml')

        # save evaluation data
        self._save_evaluation_data(
            rank_dir, td.X, y_true, y_pred, inp.target_name, inp.scaffold_network_name, info['step']
        )

        # generate plot via PlotJob
        result = build_design_result(ev, model=self.model, fingerprint=fingerprint)
        invoke_design_summary_plot(result, output_dir=rank_dir)

    def _save_recipe(self, network: Any, output_path: Path):
        save_network_recipe_yaml(network, output_path)

    def _save_evaluation_data(
        self,
        rank_dir: Path,
        x_data: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
        network_name: str,
        step: int,
    ):
        import json

        (rank_dir / 'evaluation_data.json').write_text(
            json.dumps(
                {
                    'x': x_data.tolist(),
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'target_name': target_name,
                    'network_name': network_name,
                    'step': step,
                    'grid_resolution': list(self.grid_resolution),
                    'max_evals': self.max_evals,
                },
                indent=2,
            )
        )
        np.savez_compressed(
            rank_dir / 'evaluation_data.npz', x=x_data, y_true=y_true, y_pred=y_pred
        )

    def _generate_comparison_outputs(self):
        if self._results_manager is None or not self._all_metrics:
            return
        comp_dir = self._results_manager.get_comparison_dir()
        csv_path = comp_dir / 'metrics_table.csv'
        if self._all_metrics:
            flat = [
                {
                    'target': m['target_name'],
                    'network': m['network_name'],
                    'rank': m['rank'],
                    'replicate': m['replicate_id'],
                    'loss': m['loss']['total'],
                    'rmse': m['regression']['rmse'],
                    'mae': m['regression']['mae'],
                    'r2': m['regression']['r2'],
                    'pearson_r': m['regression']['pearson_r'],
                }
                for m in self._all_metrics
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat[0].keys())
                writer.writeheader()
                writer.writerows(flat)
            logger.info(f"Saved metrics table to {csv_path}")

        import json

        if self._loss_history:
            (comp_dir / 'loss_history.json').write_text(json.dumps(self._loss_history, indent=2))

    def get_callbacks(self, training_program=None) -> list[tuple[int, Callable]]:
        def periodic_callback(step, training_config, step_history=None, stack=None, **kwargs):
            self._step_count = step
            if step_history and 'loss' in step_history:
                lv = step_history['loss']
                self._loss_history.append(
                    lv.item()
                    if hasattr(lv, 'item')
                    else float(np.mean(lv))
                    if isinstance(lv, np.ndarray)
                    else lv
                )
            if step_history is None:
                return
            all_losses, params = step_history.get('all_losses'), step_history.get('latest_params')
            if all_losses is None or params is None:
                return
            self._generate_summaries(step, params, stack, all_losses, is_final=False)

        callbacks = [(self.log_period, periodic_callback)]
        if self.log_at_end:

            def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
                if step_history is None:
                    return
                all_losses, params = (
                    step_history.get('all_losses'),
                    step_history.get('latest_params'),
                )
                if all_losses is None or params is None:
                    return
                self._generate_summaries(step, params, stack, all_losses, is_final=True)

            callbacks.append((-1, final_callback))
        return callbacks

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        return {
            'summaries_generated': self._step_count,
            'loss_history_length': len(self._loss_history),
        }

    def finalize(self):
        logger.info(f"DesignSummaryLogger finalized at step {self._step_count}")
