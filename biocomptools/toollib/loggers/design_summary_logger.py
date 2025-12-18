"""Design summary logger: saves metrics and delegates visualization to PlotJob."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any
from pydantic import ConfigDict
import csv

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.design_results import (
    DesignResultsManager,
    compute_design_metrics,
    compute_nre_for_network,
    NREMetrics,
)
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def make_lattice_grid(res: Tuple[int, int], xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, res[0]), np.linspace(ymin, ymax, res[1]))
    return np.column_stack([xx.ravel(), yy.ravel()])


class DesignSummaryLogger(Logger):
    """Logger that saves design metrics and generates summary plots via PlotJob."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_period: int = 500
    log_at_end: bool = True
    topk_per_target: int = 3
    output_formats: List[str] = ["png"]
    grid_resolution: Tuple[int, int] = (48, 48)
    max_evals: int = 100000
    eval_seed: int = 42

    output_dir: Optional[str] = None
    model: Optional[Any] = None
    targets: Optional[List[Any]] = None
    dmanager: Optional[Any] = None

    _results_manager: Optional[DesignResultsManager] = None
    _loss_history: List[float] = []
    _step_count: int = 0
    _all_metrics: List[dict] = []
    _cached_stack: Optional[Any] = None

    def initialize(self, training_program=None):
        if self.output_dir:
            design_dir = Path(self.output_dir) / 'design'
            self._results_manager = DesignResultsManager(design_dir)
            self._all_metrics = []
            logger.info(f"DesignSummaryLogger initialized: {design_dir}")

    def _get_top_candidates(
        self, all_losses: np.ndarray, target_id: int, n: int
    ) -> List[Tuple[int, int, float]]:
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 2:
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)

        n_networks = all_losses.shape[-1]
        target_losses = all_losses[:, target_id, :].ravel()
        top_indices = np.argsort(target_losses)[:n]

        return [
            (int(i // n_networks), int(i % n_networks), float(target_losses[i]))
            for i in top_indices
        ]

    def _get_or_build_stack(self, stack: Any = None) -> Any:
        if stack is not None:
            self._cached_stack = stack
            return stack
        if self._cached_stack is not None:
            return self._cached_stack
        if self.dmanager is not None and self.model is not None:
            try:
                self._cached_stack = self.dmanager.build_stack(self.model)
                logger.info("Built stack from dmanager and model")
                return self._cached_stack
            except Exception as e:
                logger.warning(f"Failed to build stack: {e}")
        return None

    def _generate_summaries(
        self, step: int, params: Any, stack: Any, all_losses: np.ndarray, is_final: bool = False
    ):
        stack = self._get_or_build_stack(stack)
        if self._results_manager is None:
            logger.warning("Results manager not initialized")
            return
        if stack is None:
            logger.warning("No stack available for summary generation")
            return

        targets = self.targets or (self.dmanager.targets if self.dmanager else [])
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)
        n_targets = all_losses.shape[-2] if all_losses.ndim >= 2 else 1

        for target_id in range(min(n_targets, len(targets))):
            target = targets[target_id]
            target_name = getattr(target, 'name', f'target_{target_id}')
            candidates = self._get_top_candidates(all_losses, target_id, self.topk_per_target)
            self._results_manager.save_rankings(
                target_name, candidates, step=None if is_final else step
            )

            for rank, (rep_id, net_id, loss) in enumerate(candidates, 1):
                try:
                    self._process_single_design(
                        params, stack, target, target_id, rep_id, net_id, loss, rank, step, is_final
                    )
                except Exception as e:
                    logger.error(f"Failed rank {rank} for {target_name}: {e}")
                    logger.exception(e)

        if is_final:
            self._generate_comparison_outputs()

    def _process_single_design(
        self,
        params: Any,
        stack: Any,
        target: Any,
        target_id: int,
        rep_id: int,
        net_id: int,
        loss: float,
        rank: int,
        step: int,
        is_final: bool,
    ):
        target_name = getattr(target, 'name', f'target_{target_id}')
        rank_dir = self._results_manager.get_rank_dir(
            target_name, rank, step=None if is_final else step
        )

        import jax

        try:
            specific_params = jax.tree.map(lambda x: x[rep_id, target_id], params)
            committed = stack.commit(specific_params)
            network = committed[net_id] if net_id < len(committed) else committed[0]
        except Exception as e:
            logger.warning(f"Failed to commit network: {e}")
            logger.exception(e)
            network = stack.networks[net_id] if net_id < len(stack.networks) else stack.networks[0]

        try:
            x_data, y_true, y_pred = self._get_evaluation_data(target, network, target_id)
        except Exception as e:
            logger.warning(f"Failed to get evaluation data: {e}")
            n = self.grid_resolution[0] * self.grid_resolution[1]
            x_data, y_true, y_pred = (
                make_lattice_grid(self.grid_resolution),
                np.random.rand(n),
                np.random.rand(n),
            )

        network_name = getattr(network, 'name', f'network_{net_id}')
        nre_metrics = None
        from biocomp.design import DataTarget

        if isinstance(target, DataTarget):
            design_nre, design_nrmse, data_nrmse = compute_nre_for_network(
                target, network, self.model, max_evals=self.max_evals
            )
            baseline_nre, baseline_nrmse = (
                (None, None)
                if target.original_network is None
                else compute_nre_for_network(
                    target, target.original_network, self.model, max_evals=self.max_evals
                )[:2]
            )
            nre_metrics = NREMetrics(
                design_nre=design_nre,
                baseline_nre=baseline_nre,
                design_nrmse=design_nrmse,
                baseline_nrmse=baseline_nrmse,
                data_nrmse=data_nrmse,
            )
            if design_nre is not None:
                logger.info(
                    f"  [{target_name} rank {rank}] Design NRE: {design_nre:.2f}"
                    + (f" (baseline: {baseline_nre:.2f})" if baseline_nre else "")
                )

        metrics = compute_design_metrics(
            y_true,
            y_pred,
            loss,
            target_name,
            network_name,
            rep_id,
            net_id,
            rank,
            step,
            nre_metrics=nre_metrics,
        )
        metrics.to_json(rank_dir / 'metrics.json')
        self._all_metrics.append(metrics.to_dict())
        self._save_recipe(network, rank_dir / 'recipe.yaml')
        self._save_evaluation_data(
            rank_dir, x_data, y_true, y_pred, target_name, network_name, step
        )

        # delegate visualization to PlotJob
        self._generate_design_plot(target, network, loss, rank, rep_id, network_name, rank_dir)

    def _generate_design_plot(
        self,
        target: Any,
        network: Any,
        loss: float,
        rank: int,
        rep_id: int,
        network_name: str,
        output_dir: Path,
    ):
        """Delegate plot generation to PlotJob with the standard design summary template."""
        try:
            from biocomptools.plot import PlotJob
            from biocomptools.toollib.figuremakers.designutils import DesignResult
            from biocomptools.toollib.hashutils import pronounceable_hash48

            target_name = getattr(target, 'name', 'unknown')
            recipe_hash = pronounceable_hash48(
                f"{target_name}_{rank}_{rep_id}_{network_name}".encode('utf-8')
            )

            result = DesignResult(
                network=network,
                target=target,
                target_name=target_name,
                rank=rank,
                replicate=rep_id,
                scaffold_network_name=network_name,
                loss=loss,
                recipe_hash=recipe_hash,
                run_name=getattr(self, '_run_name', ''),
                model=self.model,
            )
            PlotJob.invoke(
                'biocomp-jobs/plot/auto_figures/autofig_design_summary.yaml',
                result=result,
                output_dir=str(output_dir),
            )
            logger.debug(f"Generated design summary plot at {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate design plot: {e}")
            logger.debug("Full traceback:", exc_info=True)

    def _save_recipe(self, network: Any, output_path: Path):
        try:
            import dracon

            with open(output_path, 'w') as f:
                f.write(dracon.dump(network.to_recipe()))
        except Exception as e:
            logger.warning(f"Failed to save recipe: {e}")
            output_path.write_text(
                f"# Recipe extraction failed: {e}\nnetwork_name: {getattr(network, 'name', 'unknown')}\n"
            )

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

    def _get_evaluation_data(
        self, target: Any, network: Any, target_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if hasattr(target, 'get_lattice'):
            X, Y_true = target.get_lattice(self.grid_resolution)
        elif hasattr(target, 'get_samples'):
            X, Y_true = target.get_samples(
                n=self.grid_resolution[0] * self.grid_resolution[1], grid=self.grid_resolution
            )
        else:
            X, Y_true = (
                make_lattice_grid(self.grid_resolution),
                np.zeros(self.grid_resolution[0] * self.grid_resolution[1]),
            )
        X, Y_true = np.asarray(X), np.asarray(Y_true).squeeze()

        if self.model is not None:
            try:
                from biocomptools.modelmodel import NetworkModel
                from biocomptools.toollib.networkprediction import NetworkPrediction

                predictor = NetworkPrediction(
                    predict_at=[X],
                    network_model=NetworkModel(network=[network], model=self.model),
                    max_evals=self.max_evals,
                    z_value='uniform',
                    verbose=False,
                    enable_gridstats=False,
                    already_latent=True,
                )
                Y_pred = np.asarray(predictor.get_data()[0].y).squeeze()
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                Y_pred = np.zeros_like(Y_true)
        else:
            Y_pred = np.zeros_like(Y_true)
        return X, Y_true, Y_pred

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

        # save loss history as json
        import json

        if self._loss_history:
            (comp_dir / 'loss_history.json').write_text(json.dumps(self._loss_history, indent=2))

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
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
            try:
                self._generate_summaries(step, params, stack, all_losses, is_final=False)
            except Exception as e:
                logger.error(f"Summary generation failed at step {step}: {e}")
                logger.exception(e)

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
                    logger.warning(
                        f"Missing data for final summary: all_losses={all_losses is not None}, params={params is not None}"
                    )
                    return
                try:
                    self._generate_summaries(step, params, stack, all_losses, is_final=True)
                except Exception as e:
                    logger.error(f"Final summary generation failed: {e}")
                    logger.exception(e)

            callbacks.append((-1, final_callback))
        return callbacks

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[dict]:
        return {
            'summaries_generated': self._step_count,
            'loss_history_length': len(self._loss_history),
        }

    def finalize(self):
        logger.info(f"DesignSummaryLogger finalized at step {self._step_count}")
