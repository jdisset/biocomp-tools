from biocomptools.optimtools import (
    BaseOptimizationProgram,
    run_optimization_program,
    Logger,
)
from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.trainutils import make_json_ready
from biocomptools.logging_config import get_logger
from biocomptools.toollib.design_pipeline import precommit_pairs
from biocomptools.toollib.design_pipeline import (
    build_design_result,
    evaluate_design_inputs,
    invoke_design_summary_plot,
    make_design_input,
    serialize_network_recipe,
)

from biocomp.design import (
    start,
    DesignManager,
    DesignConfig,
    sample_for_evaluation,
    evaluate_design,
    get_topk_replicate_network_pairs,
    compute_baseline_loss,
    set_design_debug_output_dir,
)
from biocomp.design_targets import (
    SVGTarget,
    DataTarget,
    TargetUnion,
    SamplingConfigUnion,
    UniformSampling,
)
from biocomp.tracing import save_debug_state, is_design_debug_enabled
from biocomp.paramintrospect import format_committed_network_params_rich
from biocomp.network import Network, recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.jaxutils import tree_to_np, tree_get

from dracon.commandline import Arg, dracon_program
from biocomptools.optimtools import DEFAULT_TYPES
from biocomptools.toollib.common import config
from biocomptools.write_policy import DESIGN_DEFAULT, WritePolicy
import asyncio

import sys
import numpy as np
import jax
from pathlib import Path
from typing import Annotated, Optional
from pydantic import Field
import pickle
from datetime import datetime

logger = get_logger(__name__)


def _target_name(t) -> str:
    if isinstance(t, DataTarget):
        return t.name or "data_target"
    return t.name or Path(t.path).stem


def _create_swapped_network(network: Network) -> Network | None:
    """Create a copy of the network with swapped input_order (x<->y axes)."""
    input_order = network.metadata.get("input_order")
    if input_order is None or len(input_order) != 2:
        return None

    swapped = network.model_copy(deep=True)
    swapped_order = [input_order[1], input_order[0]]
    swapped.apply_input_order(swapped_order)
    swapped.name = f"{network.name}_swapped"

    if "axis_mapping" in swapped.metadata:
        old_mapping = swapped.metadata["axis_mapping"]
        swapped.metadata["axis_mapping"] = {
            k: ("y" if v == "x" else "x") for k, v in old_mapping.items()
        }

    return swapped


@dracon_program(
    name='biocomp-design',
    description='Run design optimization for biocomp models.',
    context_types=DEFAULT_TYPES,
    context={'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve()},
)
class DesignProgram(BaseOptimizationProgram):
    write_policy: WritePolicy | None = Field(default_factory=lambda: DESIGN_DEFAULT)

    design_conf: Annotated[DesignConfig, Arg(help='Design optimization config')] = Field(
        default_factory=lambda: DesignConfig()
    )

    targets: Annotated[
        list[TargetUnion] | TargetUnion, Arg(help='Design targets (SVG files or DataTarget)')
    ] = Field(default_factory=list)

    networks: Annotated[Optional[list[Network] | Network], Arg(help='Networks to optimize')] = None
    scaffolds: Annotated[
        Optional[list[Recipe] | Recipe],
        Arg(help='Base recipes to optimize (converted to networks)'),
    ] = None
    sampling: Annotated[SamplingConfigUnion, Arg(help='Sampling strategy configuration')] = Field(
        default_factory=UniformSampling
    )
    network_subset_size: Annotated[Optional[int], Arg(help='Limit networks to first N')] = None
    model_name: Annotated[
        Optional[str], Arg(help='Model path or signature (simpler alternative to model_selector)')
    ] = None
    model_selector: Annotated[
        Optional[ModelSelector], Arg(help='Model selector for complex queries')
    ] = None

    experiment_name: str = 'default_design_xp'

    n_eval_samples: Annotated[int, Arg(help='Number of samples for evaluation')] = 10000
    eval_seed: Annotated[int, Arg(help='Random seed for evaluation')] = 42
    max_eval_chunk_size: Annotated[int, Arg(help='Max samples per evaluation chunk')] = 64
    max_eval_loss_size: Annotated[int, Arg(help='Max samples per loss evaluation chunk')] = 64

    save_evaluation_data: Annotated[bool, Arg(help='Save evaluation data for later analysis')] = (
        False
    )

    topk_n: Annotated[int, Arg(help='Number of top designs to keep per target')] = 10

    plot_results: Annotated[bool, Arg(help='Generate result plots')] = True
    plot_n_samples: Annotated[int, Arg(help='Max samples to plot')] = 5000
    skip_evaluation: Annotated[
        bool, Arg(help='Skip post-optimization evaluation (useful for DataTarget)')
    ] = False
    show_difference_plots: Annotated[bool, Arg(help='Show difference plots')] = False
    lock_ratios: Annotated[
        bool, Arg(help='Lock ratios to recipe-specified values (for zero-freedom baseline tests)')
    ] = False
    save_commit_debug: Annotated[
        bool, Arg(help='Save networks and params before/after commit for debugging')
    ] = False
    swap_axes_duplicate: Annotated[
        bool, Arg(help='Duplicate each network with swapped x/y input_order for both orientations')
    ] = False

    def model_post_init(self, __context):
        self._model: BiocompModel | None = None
        self._dmanager: DesignManager | None = None
        self._effective_dconf: DesignConfig | None = None
        self._design_id: str | None = None
        super().model_post_init(__context)

    @property
    def design_id(self) -> str:
        if self._design_id is None:
            self._design_id = self.unique_id
        return self._design_id

    def get_output_subdir(self) -> str:
        return ''

    def _targets_list(self) -> list[TargetUnion]:
        targets = self.targets
        if isinstance(targets, list):
            return targets
        return [targets]

    def initialize_context(self):
        logger.info("Initializing design context...")

        if self.model_name and self.model_name.strip():
            model_path = Path(self.model_name)
            if model_path.exists() and model_path.suffix == '.pickle':
                logger.info(f"Loading model directly from path: {model_path}")
                self._model = BiocompModel.load(model_path)
            else:
                logger.debug(f"Using model_name for DB lookup: {self.model_name}")
                effective_selector = ModelSelector(name=self.model_name)
                with self.db_session as session:
                    selected_model = effective_selector.get_model(session)
                    logger.info(
                        f"Loading model: {selected_model.name if selected_model else 'None'}"
                    )
                    self._model = selected_model.load() if selected_model else None
                    session.expunge_all()
                    session.close()
        elif self.model_selector:
            logger.debug(f"Using model_selector: {self.model_selector}")
            effective_selector = self.model_selector
            with self.db_session as session:
                selected_model = effective_selector.get_model(session)
                logger.info(f"Loading model: {selected_model.name if selected_model else 'None'}")
                self._model = selected_model.load() if selected_model else None
                session.expunge_all()
                session.close()
        else:
            logger.warning("No model specified - model_name or model_selector required")
            return

        if self._model is None:
            raise ValueError(
                f"Could not load model: model_name={self.model_name}, model_selector={self.model_selector}"
            )

        logger.info(f"Successfully loaded model with signature: {self._model.signature}")

        if isinstance(self.networks, Network):
            self.networks = [self.networks]
        if isinstance(self.scaffolds, Recipe):
            self.scaffolds = [self.scaffolds]

        networks = []
        if self.networks is not None:
            networks.extend(self.networks)
            logger.debug(f"Using {len(self.networks)} directly specified networks")

        if self.scaffolds is not None:
            for scaffold in self.scaffolds:
                if not scaffold.has_input_order():
                    logger.warning(
                        f"Scaffold '{scaffold.name}' has no input_order - may cause axis alignment issues"
                    )
                scaffold_networks = recipe_to_networks(scaffold, invert=True, inversion_mode="main")
                for net in scaffold_networks:
                    if scaffold.has_input_order() and not net.has_input_order():
                        logger.error(f"input_order lost: {scaffold.name} -> {net.name}")
                networks.extend(scaffold_networks)
            logger.debug(f"Generated {len(networks)} networks from {len(self.scaffolds)} scaffolds")

        if not networks:
            raise ValueError(
                "No networks or scaffolds specified. Use --networks or --scaffolds in config."
            )

        if self.network_subset_size is not None:
            networks = networks[: self.network_subset_size]
            logger.info(f"Limited networks to first {self.network_subset_size}")

        if self.swap_axes_duplicate:
            swapped_networks = []
            for net in networks:
                swapped = _create_swapped_network(net)
                if swapped is not None:
                    swapped_networks.append(swapped)
            networks.extend(swapped_networks)
            logger.info(
                f"Added {len(swapped_networks)} swapped-axes network variants (total: {len(networks)})"
            )

        if isinstance(self.targets, (SVGTarget, DataTarget)):
            self.targets = [self.targets]
        logger.info(
            f"Creating DesignManager with {len(self.targets)} targets, {len(networks)} networks, "
            f"{self.sampling.strategy} sampling, TU masking mode={self.design_conf.tu_masking.mode.value}"
        )
        self._dmanager = DesignManager(
            targets=self.targets,
            networks=networks,
            sampling=self.sampling,
            enable_tu_masking=self.design_conf.enable_tu_masking,
        )
        if self.design_conf.enable_tu_masking:
            n_networks = len(self._dmanager.networks)
            mode = self.design_conf.tu_masking.mode.value
            logger.info(f"TU masking ({mode}): {self._dmanager.n_tus} TUs × {n_networks} networks")

    def _get_logger_context(self) -> dict:
        context = super()._get_logger_context()
        context.update(
            {
                'design_conf': self.design_conf,
                'targets': self.targets,
                'networks': self.networks,
                'dmanager': self._dmanager,
                'model': self._model,
                'output_dir': str(self._save_dir),
            }
        )
        return context

    def _save_design_networks(self):
        """Save networks early for replay/diagnostic loggers to access."""
        if self._dmanager is None:
            return
        networks_file = self._save_dir / 'design_networks.pickle'
        networks_data = {
            'networks': self._dmanager.networks,
            'network_names': [n.name for n in self._dmanager.networks],
            'target_names': [_target_name(t) for t in self._targets_list()],
            'n_targets': self._dmanager.n_targets,
        }
        with open(networks_file, 'wb') as f:
            pickle.dump(networks_data, f)
        logger.debug(f"Saved {len(self._dmanager.networks)} networks to {networks_file}")

    def enrich_metadata(self):
        if self._dmanager is None:
            return

        networks = self._dmanager.networks
        design_info = {
            "n_targets": self._dmanager.n_targets,
            "target_names": [_target_name(t) for t in self._targets_list()],
            "n_networks": len(networks),
            "network_names": [n.name for n in networks],
            "n_replicates": self.design_conf.n_replicates,
            "n_epochs": self.design_conf.n_epochs,
            "batch_size": self.design_conf.batch_size,
            "n_batches_per_epoch": self.design_conf.n_batches_per_epoch,
            "sampling_strategy": self.sampling.strategy,
            "sampling_config": self.sampling.model_dump(),
        }

        self._metadata.update(
            {
                'design_id': self.design_id,
                'run_name': self._run_name,
                'experiment_name': self.experiment_name,
                'model_signature': self._model.signature if self._model else 'unknown',
                'model_name': self.model_name,
                'model_selector': self.model_selector.model_dump() if self.model_selector else None,
                'design_info': design_info,
                'design_conf': self.design_conf.model_dump(),
                'targets': [
                    t.model_dump() if hasattr(t, "model_dump") else {'name': _target_name(t)}
                    for t in self._targets_list()
                ],
                'networks': [n.name for n in networks],
                'evaluation': {
                    'n_eval_samples': self.n_eval_samples,
                    'eval_seed': self.eval_seed,
                    'topk_n': self.topk_n,
                },
                'final_model_dump': self._modeldump,
            }
        )

    async def execute_optimization(self, dispatch):
        logger.info("Starting design optimization...")
        assert self._dmanager is not None

        # Save networks early for replay/diagnostic loggers
        self._save_design_networks()

        # Set debug output directory for design module
        set_design_debug_output_dir(str(self._save_dir))
        if is_design_debug_enabled():
            logger.info(
                f"Design debug enabled - saving debug dumps to {self._save_dir}/_debug_dumps/"
            )

        target_names = [_target_name(t) for t in self._targets_list()]
        logger.info(f"Optimizing for {self._dmanager.n_targets} targets: {', '.join(target_names)}")

        network_names: list[str] = [str(n.name) for n in self._dmanager.networks[:5]]
        if len(self._dmanager.networks) > 5:
            network_names.append(f"... and {len(self._dmanager.networks) - 5} more")
        logger.info(f"Using {len(self._dmanager.networks)} networks: {', '.join(network_names)}")
        logger.info("Optimization parameters:")
        logger.info(f"  - Replicates: {self.design_conf.n_replicates}")
        logger.info(f"  - Epochs: {self.design_conf.n_epochs}")
        logger.info(f"  - Batch size: {self.design_conf.batch_size}")
        logger.info(f"  - Batches per epoch: {self.design_conf.n_batches_per_epoch}")
        logger.debug(
            f"  - Learning rate: {self.design_conf.optimizer_stack[1].kwargs.get('learning_rate', 'N/A')}"
        )
        logger.debug(
            f"  - Gradient clip norm: {self.design_conf.optimizer_stack[0].kwargs.get('max_norm', 'N/A')}"
        )

        assert self._model is not None

        if self.design_conf.hard_pruning_enabled:
            from biocomp.design_pruning import run_with_hard_pruning

            final_params, loss_history, step_history, self._dmanager, self._effective_dconf = (
                run_with_hard_pruning(
                    self._dmanager, self.design_conf, self._model,
                    dispatch=dispatch, lock_ratios=self.lock_ratios,
                )
            )
        else:
            final_params, loss_history, step_history = start(
                dmanager=self._dmanager,
                dconf=self.design_conf,
                model=self._model,
                dispatch=dispatch,
                lock_ratios=self.lock_ratios,
            )
            self._effective_dconf = self.design_conf

        logger.info(
            f"Optimization completed. Final loss: {loss_history[-1]:.4f}"
            if loss_history
            else "Optimization completed. No loss history"
        )

        if not self.skip_evaluation:
            await self._evaluate_and_save_results(final_params, loss_history)
        else:
            logger.info("Skipping post-optimization evaluation (skip_evaluation=True)")

        return final_params, loss_history, step_history

    async def _evaluate_and_save_results(self, final_params, loss_history):
        import time

        logger.info("=" * 60)
        logger.info("POST-OPTIMIZATION EVALUATION")
        logger.info("=" * 60)

        assert self._dmanager is not None
        assert self._model is not None

        eval_dconf = self._effective_dconf or self.design_conf
        eval_key = jax.random.key(self.eval_seed)

        t0 = time.perf_counter()
        logger.info(f"[1/3] Sampling {self.n_eval_samples} evaluation points...")
        xraw, yraw = sample_for_evaluation(
            dmanager=self._dmanager,
            dconf=eval_dconf,
            final_params=final_params,
            n_eval_samples=self.n_eval_samples,
            key=eval_key,
        )
        logger.info(
            f"  -> Sampled in {time.perf_counter() - t0:.2f}s (shapes: x={xraw.shape}, y={yraw.shape})"
        )

        t1 = time.perf_counter()
        logger.info(
            f"[2/3] Running forward pass evaluation (chunk_size={self.max_eval_chunk_size})..."
        )
        yhatdep, losses = evaluate_design(
            dmanager=self._dmanager,
            dconf=eval_dconf,
            model=self._model,
            final_params=final_params,
            xraw=xraw,
            yraw=yraw,
            key=eval_key,
            max_eval_size=self.max_eval_chunk_size,
            max_loss_size=self.max_eval_loss_size,
            store_predictions=self.plot_results,
        )
        logger.info(f"  -> Evaluated in {time.perf_counter() - t1:.2f}s")
        logger.info(
            f"  -> Losses: min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}"
        )

        t2 = time.perf_counter()
        logger.info(f"[3/3] Finding top {self.topk_n} designs...")

        # Request 3x candidates to handle empty recipes after commit
        # (some designs may become empty due to aggressive TU pruning)
        topk_candidate_pool = 3
        topk = get_topk_replicate_network_pairs(
            losses=losses,
            dmanager=self._dmanager,
            dconf=eval_dconf,
            k=self.topk_n * topk_candidate_pool,
        )
        logger.info(f"  -> Top-k found in {time.perf_counter() - t2:.2f}s")

        t3 = time.perf_counter()
        baseline_results = compute_baseline_loss(
            dmanager=self._dmanager,
            model=self._model,
            n_samples=self.n_eval_samples,
            seed=self.eval_seed,
        )
        logger.info(f"  -> Baseline computed in {time.perf_counter() - t3:.2f}s")

        logger.info("=" * 60)
        logger.info("RESULTS: Best replicate/network pairs for each target:")
        for tid, target in enumerate(self._dmanager.targets):
            target_name = _target_name(target)
            rep_id, net_id, loss_val = topk[tid][0]
            network_name = self._dmanager.networks[net_id].name
            baseline_info = baseline_results.get(target_name, {})
            if baseline_info.get('has_original_network'):
                baseline_loss = baseline_info['model_prediction_loss']
                improvement = (
                    (baseline_loss - loss_val) / baseline_loss * 100 if baseline_loss > 0 else 0
                )
                logger.info(
                    f"  {target_name}: Rep {rep_id}, Net '{network_name}' "
                    f"(loss={loss_val:.4f}, baseline={baseline_loss:.4f}, improvement={improvement:+.1f}%)"
                )
            else:
                logger.info(
                    f"  {target_name}: Rep {rep_id}, Net '{network_name}' (loss={loss_val:.4f})"
                )

        self._evaluation_results = (final_params, loss_history, topk, losses, xraw, yraw, yhatdep)

        targets = self._targets_list()
        self._metadata['evaluation_results'] = {
            'losses_shape': losses.shape,
            'baseline_results': baseline_results,
            'topk_results': [
                {
                    'target': _target_name(targets[tid]),
                    'baseline': baseline_results.get(_target_name(targets[tid]), {}),
                    'best_designs': [
                        {
                            'replicate_id': rep_id,
                            'network_id': net_id,
                            'network_name': self._dmanager.networks[net_id].name,
                            'loss': float(loss_val),
                        }
                        for rep_id, net_id, loss_val in target_topk
                    ],
                }
                for tid, target_topk in enumerate(topk)
            ],
        }

    def save_outputs(self, final_params, loss_history, step_history=None):
        save_dir = self._save_dir / self.get_output_subdir()
        logger.info(f"Saving outputs to {save_dir}")

        if step_history and len(step_history) > 0:
            final_step = step_history[-1] if isinstance(step_history, list) else step_history
            if "pareto_front" in final_step and "pareto_fitness" in final_step:
                pareto_front = np.asarray(final_step["pareto_front"])
                pareto_fitness = np.asarray(final_step["pareto_fitness"])
                np.savez_compressed(
                    save_dir / "pareto_front.npz",
                    population=pareto_front,
                    fitness=pareto_fitness,
                )
                import json

                pareto_summary = {
                    "n_solutions": int(len(pareto_fitness)),
                    "min_loss": float(np.min(pareto_fitness[:, 0])),
                    "max_loss": float(np.max(pareto_fitness[:, 0])),
                    "min_tu_count": float(np.min(pareto_fitness[:, 1])),
                    "max_tu_count": float(np.max(pareto_fitness[:, 1])),
                }
                (save_dir / "pareto_summary.json").write_text(json.dumps(pareto_summary, indent=2))
                logger.info(
                    f"Saved pareto front: {len(pareto_fitness)} solutions, "
                    f"loss range [{pareto_summary['min_loss']:.4f}, {pareto_summary['max_loss']:.4f}]"
                )

        if hasattr(self, '_evaluation_results'):
            final_params, loss_history, topk, losses, xraw, yraw, yhatdep = self._evaluation_results

            params_file = save_dir / 'final_params.pickle'
            with open(params_file, 'wb') as f:
                pickle.dump(tree_to_np(final_params), f)
            logger.debug(f"Saved final parameters to {params_file}")

            if loss_history:
                all_losses = np.array(loss_history)
                np.save(save_dir / 'loss_history.npy', all_losses)
            else:
                logger.warning("No loss history to save (0 training steps)")

            topk_file = save_dir / 'topk_results.pickle'
            with open(topk_file, 'wb') as f:
                pickle.dump(topk, f)
            logger.debug(f"Saved top-k results to {topk_file}")

            np.save(save_dir / 'evaluation_losses.npy', losses)

            eval_data = {
                'xraw': xraw,
                'yraw': yraw,
                'yhatdep': yhatdep,
            }
            if self.save_evaluation_data:
                eval_file = save_dir / 'evaluation_data.pickle'
                with open(eval_file, 'wb') as f:
                    pickle.dump(eval_data, f)
                logger.debug(f"Saved evaluation data to {eval_file}")

            self._save_best_designs_summary(save_dir, topk)

        logger_metrics = [
            m
            for m in (
                lg.get_metrics(replicate=None) for lg in self.loggers if isinstance(lg, Logger)
            )
            if m
        ]
        if logger_metrics:
            self._metadata['logger_metrics_all_replicates'] = make_json_ready(logger_metrics)

        if loss_history:
            # design loss_history elements have shape (n_replicates, n_targets, ...)
            # plot_loss expects list of (n_replicates, n_steps) - average over targets
            processed = []
            for lh in loss_history:
                arr = np.array(lh)
                if arr.ndim == 0:
                    # scalar -> (1, 1)
                    processed.append(arr.reshape(1, 1))
                elif arr.ndim == 1:
                    # (n_replicates,) -> (n_replicates, 1)
                    processed.append(arr.reshape(-1, 1))
                elif arr.ndim == 2:
                    # (n_replicates, n_targets) -> average over targets -> (n_replicates, 1)
                    processed.append(np.nanmean(arr, axis=1, keepdims=True))
                else:
                    # ndim >= 3: (n_replicates, n_targets, ...) -> average over targets
                    processed.append(np.nanmean(arr, axis=1))
            if processed:
                self.save_loss_plot(processed, save_dir)
        logger.debug("Saving metadata...")
        self.save_metadata(save_dir)
        logger.info("All outputs saved successfully.")

    def _save_best_designs_summary(self, save_dir, topk):
        summary_file = save_dir / 'best_designs_summary.txt'
        design_results_dir = save_dir / 'design_results'
        design_results_dir.mkdir(exist_ok=True)

        final_params = self._evaluation_results[0] if hasattr(self, '_evaluation_results') else None
        assert final_params is not None, "final_params required to save designs"
        assert self._model is not None, "model required to save designs"
        assert self._dmanager is not None, "design manager required to save designs"

        n_targets = len(self._targets_list())
        n_networks = len(self._dmanager.networks)
        n_replicates = (self._effective_dconf or self.design_conf).n_replicates

        assert len(topk) == n_targets, f"topk length {len(topk)} != n_targets {n_targets}"

        import time

        stack = self._dmanager.build_stack(
            self._model,
            unlock_ratios=False,
            auto_lock_topology_tus=self.design_conf.auto_lock_topology_tus,
        )

        run_datetime = datetime.now().isoformat()
        run_name = self._run_name
        all_design_results = []

        # Pre-compute all unique (rep_id, tid) pairs needed from topk
        needed_pairs: set[tuple[int, int]] = set()
        for tid in range(n_targets):
            for rep_id, _net_id, _loss_val in topk[tid]:
                needed_pairs.add((rep_id, tid))

        # Pre-commit all needed parameter sets (expensive operation, do once per pair)
        t0 = time.perf_counter()
        logger.info(f"Pre-committing {len(needed_pairs)} unique (replicate, target) pairs...")

        # setup commit debug directory if enabled
        commit_debug_dir = None
        if self.save_commit_debug:
            commit_debug_dir = save_dir / "commit_debug"
            commit_debug_dir.mkdir(exist_ok=True)
            logger.info(f"Saving commit debug data to {commit_debug_dir}")

        def _on_before(rep_id, tid, bparams, commit_stack):
            if commit_debug_dir is None:
                return
            pre_commit_data = {
                'rep_id': rep_id,
                'tid': tid,
                'params': tree_to_np(bparams),
                'networks': [n.model_copy(deep=True) for n in commit_stack.networks],
                'tu_id_to_idx': commit_stack.tu_id_to_idx,
            }
            pre_path = commit_debug_dir / f"pre_commit_rep{rep_id}_tid{tid}.pickle"
            with open(pre_path, 'wb') as f:
                pickle.dump(pre_commit_data, f)

        def _on_after(rep_id, tid, _bparams, _commit_stack, committed_networks):
            if commit_debug_dir is None:
                return
            post_commit_data = {
                'rep_id': rep_id,
                'tid': tid,
                'committed_networks': committed_networks,
            }
            post_path = commit_debug_dir / f"post_commit_rep{rep_id}_tid{tid}.pickle"
            with open(post_path, 'wb') as f:
                pickle.dump(post_commit_data, f)

        def _on_error(rep_id, tid, _bparams, _commit_stack, err):
            logger.error(f"Commit failed for (rep={rep_id}, target={tid}): {type(err).__name__}: {err}")

        commit_cache, commit_failures = precommit_pairs(
            final_params,
            stack,
            needed_pairs,
            fail_fast=True,
            on_before=_on_before,
            on_after=_on_after,
            on_error=_on_error,
        )
        assert not commit_failures, f"commit failures not allowed: {commit_failures}"
        logger.info(f"  -> Commits completed in {time.perf_counter() - t0:.2f}s")

        with open(summary_file, 'w') as f:
            f.write("BEST DESIGN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Date: {run_datetime}\n")
            f.write(f"Model: {self._model.signature}\n\n")

            best_per_target = {}

            for tid, target in enumerate(self.targets):
                target_name = _target_name(target)
                f.write(f"Target: {target_name}\n")
                f.write("-" * 30 + "\n")

                target_results_dir = design_results_dir / target_name
                target_results_dir.mkdir(exist_ok=True)

                valid_rank = 0
                for rep_id, net_id, loss_val in topk[tid]:
                    # Stop when we have enough valid designs
                    if valid_rank >= self.topk_n:
                        break

                    assert 0 <= rep_id < n_replicates, (
                        f"rep_id {rep_id} out of bounds [0, {n_replicates}) for target {tid}"
                    )
                    assert 0 <= net_id < n_networks, (
                        f"net_id {net_id} out of bounds [0, {n_networks}) for target {tid}"
                    )

                    bparams = tree_get(final_params, (rep_id, tid))

                    # Use pre-computed commit result (all pairs committed upfront)
                    committed_networks = commit_cache[(rep_id, tid)]
                    cnet = committed_networks[net_id]
                    assert cnet is not None, f"committed network {net_id} is None"

                    recipe = cnet.to_recipe(auto_name_from_l1=True)

                    if not recipe.content:
                        network_name = self._dmanager.networks[net_id].name
                        raise ValueError(
                            f"Empty recipe after commit for target '{target_name}' "
                            f"(rep={rep_id}, net={net_id}, scaffold='{network_name}', "
                            f"loss={loss_val:.6f})"
                        )

                    valid_rank += 1
                    rank = valid_rank  # Use 1-indexed rank for output

                    network_name = self._dmanager.networks[net_id].name
                    f.write(f"  Rank {rank}: Replicate {rep_id}, Network '{network_name}'\n")
                    f.write(f"           Loss: {loss_val:.6f}\n")

                    recipe_metadata = {
                        'target_name': target_name,
                        'datetime': run_datetime,
                        'run_name': run_name,
                        'loss': float(loss_val),
                        'rank': rank,
                        'replicate': rep_id,
                        'scaffold_network_name': network_name,
                        'scaffold_network_id': net_id,
                        'model_signature': self._model.signature,
                    }

                    _, full_recipe_yaml, recipe_hash = serialize_network_recipe(
                        cnet,
                        auto_name_from_l1=True,
                        metadata=recipe_metadata,
                    )
                    design_dir = target_results_dir / f"design_{rank:02d}"
                    design_dir.mkdir(exist_ok=True)
                    recipe_filename = design_dir / f"{recipe_hash}.yaml"
                    with open(recipe_filename, 'w') as rf:
                        rf.write(full_recipe_yaml)
                    network_pickle = design_dir / f"{recipe_hash}.pickle"
                    with open(network_pickle, 'wb') as npf:
                        pickle.dump(cnet, npf)

                    target_input_names = getattr(target, 'input_names', None)
                    scaffold_input_proteins = cnet.get_inverted_input_proteins()

                    design_info = {
                        'rank': rank,
                        'replicate': rep_id,
                        'network_name': network_name,
                        'network_id': net_id,
                        'network': cnet,
                        'loss': float(loss_val),
                        'params': bparams,
                        'recipe_hash': recipe_hash,
                        'recipe_path': str(recipe_filename),
                        'design_dir': str(design_dir),
                        'target_name': target_name,
                        'target': target,
                        'target_id': tid,
                        'target_input_names': target_input_names,
                        'scaffold_input_proteins': scaffold_input_proteins,
                    }

                    all_design_results.append(design_info)
                    if rank == 1:
                        best_per_target[target_name] = design_info

                    f.write(f"           Recipe: {recipe_hash}.yaml\n")
                    f.write(f"           Path: {design_dir.relative_to(save_dir)}/\n")
                    logger.debug(f"Saved design {target_name} rank {rank}: {recipe_hash}")

                f.write("\n")

        if best_per_target:
            best_designs_file = save_dir / 'best_designs.pickle'
            with open(best_designs_file, 'wb') as f:
                for target_data in best_per_target.values():
                    target_data['params'] = tree_to_np(target_data['params'])
                pickle.dump(best_per_target, f)
            logger.info(f"Saved best designs data to {best_designs_file}")

        if self.plot_results and all_design_results:
            self._generate_design_diagnostic_plots(save_dir, all_design_results)

        if all_design_results:
            self._print_rich_design_summary(all_design_results, stack=stack)

        logger.info(f"Saved best designs summary to {summary_file}")
        if design_results_dir.exists() and any(design_results_dir.iterdir()):
            logger.info(f"Saved design results to {design_results_dir}")

    def _print_rich_design_summary(
        self, all_design_results: list[dict], top_n: int = 3, stack=None
    ):
        """Print design summary using same format as DesignHeatmapLogger."""
        from rich.console import Console
        from rich.panel import Panel
        from biocomp.designutils import side_by_side_txt_plot
        from biocomp.fingerprint import compute_fingerprint, FINGERPRINT_SEED
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction
        from collections import defaultdict

        console = Console()

        by_target = defaultdict(list)
        for d in all_design_results:
            by_target[d['target_name']].append(d)

        for target_name in by_target:
            by_target[target_name].sort(key=lambda x: x['loss'])

        loss_weights = {}
        if self.design_conf and hasattr(self.design_conf, 'loss_function'):
            lf = self.design_conf.loss_function
            if hasattr(lf, 'kwargs') and lf.kwargs:
                for k, v in lf.kwargs.items():
                    if k.startswith('w_'):
                        loss_weights[k] = v

        assert self._dmanager is not None, "design manager required for rich summary"
        res = self._dmanager.grid_resolution
        assert res is not None, "grid resolution required for rich summary"
        xres, yres = res
        display_width, display_height = 40, 20

        console.print()
        console.rule(
            "[bold cyan]DESIGN RESULTS SUMMARY (recomputed from committed networks)[/bold cyan]",
            style="cyan",
        )

        for target_name, designs in by_target.items():
            target = designs[0].get('target') if designs else None
            assert target is not None, f"missing target for group '{target_name}'"
            assert self._model is not None, "model required for rich summary"

            console.print()
            console.print(
                Panel(f"[bold white]{target_name}[/bold white]", style="blue", expand=False)
            )

            valid_designs = [
                (i, d) for i, d in enumerate(designs[:top_n]) if d.get('network') is not None
            ]
            assert valid_designs, f"no valid networks for target '{target_name}'"

            networks = [d['network'] for _, d in valid_designs]
            nm = NetworkModel(model=self._model, network=networks)
            X_lat, Y_target = target.get_lattice(resolution=res, seed=0)
            Y_target_grid = np.asarray(Y_target).reshape(yres, xres)

            pred = NetworkPrediction(
                predict_at=[X_lat] * len(networks),
                network_model=nm,
                already_latent=True,
                z_value=0.0,
                disable_variational=True,
                skip_input_reorder=True,
                seed=FINGERPRINT_SEED,
            )
            data_list = pred.get_data(rescale_latent=False)

            for net_idx, ((orig_rank, design), data) in enumerate(
                zip(valid_designs, data_list, strict=True)
            ):
                network_name = design.get('network_name', f"Net {orig_rank}")
                loss = design.get('loss', float('nan'))
                rep_id = design.get('replicate', 0)

                Y_pred_grid = np.asarray(data.y).reshape(yres, xres)
                fingerprint = compute_fingerprint(nm, network_idx=net_idx)
                fp_str = f" │ FP: {fingerprint}"

                width = display_width * 2 + 13
                n_shown = len(valid_designs)
                header = f" Rep {rep_id} {network_name} (rank {orig_rank + 1}/{n_shown}, loss={loss:.4f}){fp_str}"
                console.print(f"{'─' * width}")
                console.print(header)
                console.print(f"{'─' * width}")

                txt_output, metrics = side_by_side_txt_plot(
                    Y_target_grid,
                    Y_pred_grid,
                    height=display_height,
                    width=display_width,
                    loss_weights=loss_weights,
                    title_target="TARGET",
                    title_prediction="PREDICTION",
                    shared_colorbar=False,
                    show_axes=True,
                    compute_metrics=True,
                )
                console.print(txt_output)

                pred_range = metrics.get('pred_range', (0, 0))
                corr = metrics.get('correlation', 0.0)
                console.print(
                    f"Pred: [{pred_range[0]:.2f}, {pred_range[1]:.2f}] │ Corr: {corr:.4f}"
                )

                committed_net = design.get('network')
                if committed_net is not None:
                    console.print("")
                    bparams = design.get('params')
                    net_id = design.get('network_id', 0)
                    assert stack is not None, "stack required for committed network display"
                    assert bparams is not None, "params required for committed network display"
                    format_committed_network_params_rich(
                        committed_net, stack, bparams, net_id, console
                    )

                console.print()

        console.rule(style="cyan")

    def _generate_design_diagnostic_plots(self, save_dir, design_results: list[dict]):
        """Generate diagnostic plots using batched evaluation."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_results = len(design_results)
        assert self._model is not None, "model required for diagnostic plots"
        assert n_results > 0, "at least one design result required"
        run_name = self._run_name
        assert run_name is not None, "run_name required for diagnostic plots"
        logger.info(f"Generating diagnostic plots for {n_results} designs...")

        # convert dicts to DesignInput objects
        inputs = [
            make_design_input(
                network=r['network'],
                target=r['target'],
                target_name=r['target_name'],
                rank=r['rank'],
                replicate=r['replicate'],
                net_id=r['network_id'],
                loss=r['loss'],
                recipe_hash=r['recipe_hash'],
                run_name=run_name,
                design_dir=r['design_dir'],
            )
            for r in design_results
        ]

        # batch evaluate all designs
        evaluated = evaluate_design_inputs(self._model, inputs, max_evals=50000)

        invalid = [ev.input.recipe_hash for ev in evaluated if not ev.is_valid]
        assert not invalid, f"invalid design results not allowed: {invalid}"

        def _generate_single_plot(ev) -> str:
            """Worker function to generate a single design plot."""
            inp = ev.input
            design_dir = Path(inp.design_dir)

            result = build_design_result(ev, model=self._model)

            if is_design_debug_enabled():
                original_input_proteins = None
                if hasattr(inp.target, 'original_network') and inp.target.original_network:
                    original_input_proteins = inp.target.original_network.get_inverted_input_proteins()
                save_debug_state(
                    f"design_result_rank{inp.rank}",
                    {
                        'target_X': getattr(inp.target, 'X', None),
                        'target_Y': getattr(inp.target, 'Y', None),
                        'pred_X': ev.pred_data.xval,
                        'pred_Y': ev.pred_data.yval,
                    },
                    {
                        'target_name': inp.target_name,
                        'rank': inp.rank,
                        'replicate': inp.replicate,
                        'loss': inp.loss,
                        'recipe_hash': inp.recipe_hash,
                        'target_input_names': getattr(inp.target, 'input_names', None),
                        'network_name': getattr(inp.network, 'name', None),
                        'network_inputs': inp.network.get_inverted_input_proteins(),
                        'original_network_inputs': original_input_proteins,
                    },
                    output_dir=str(design_dir),
                    mode="design",
                )

            invoke_design_summary_plot(result, output_dir=design_dir)
            return inp.recipe_hash

        max_workers = min(8, len(evaluated))
        assert max_workers > 0, "no evaluated designs for plotting"
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generate_single_plot, ev): ev for ev in evaluated}
            for future in as_completed(futures):
                recipe_hash = future.result()
                completed += 1
                logger.debug(f"Generated summary plot for {recipe_hash}")

        logger.info(f"Plot generation complete: {completed} succeeded")


async def main_async():
    await run_optimization_program(
        DesignProgram,
        'biocomp-design',
        'Run design optimization for biocomp models.',
        sys.argv[1:],
    )


def main():
    from biocomptools.logging_config import setup_logging

    setup_logging()
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
