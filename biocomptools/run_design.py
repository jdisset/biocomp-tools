from biocomptools.optimtools import (
    BaseOptimizationProgram,
    run_optimization_program,
    Logger,
)
from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.trainutils import make_json_ready
from biocomptools.logging_config import get_logger
from biocomptools.toollib.hashutils import pronounceable_hash48
from biocomptools.toollib.design_eval import DesignEvaluator, DesignInput

from biocomp.design import (
    start,
    DesignManager,
    DesignConfig,
    Target,
    DataTarget,
    TargetUnion,
    sample_for_evaluation,
    evaluate_design,
    get_topk_replicate_network_pairs,
    SamplingConfigUnion,
    UniformSampling,
    compute_baseline_loss,
    set_design_debug_output_dir,
)
from biocomp.designdebug import save_debug_state, is_design_debug_enabled
from biocomp.network import Network, recipe_to_networks
from biocomp.graphengine import GraphState
from biocomp.recipe import Recipe
from biocomp.jaxutils import tree_to_np, tree_get

from dracon.commandline import Arg, dracon_program
from biocomptools.optimtools import DEFAULT_TYPES, make_context_from_types
from biocomptools.toollib.common import config
import dracon
import asyncio

import sys
import traceback
import numpy as np
import jax
import yaml
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


def _safe_get_input_proteins(network) -> list | None:
    """Safely get inverted input proteins, returning None on failure."""
    if not network or not hasattr(network, 'get_inverted_input_proteins'):
        return None
    try:
        return network.get_inverted_input_proteins()
    except (AssertionError, AttributeError):
        return None


@dracon_program(
    name='biocomp-design',
    description='Run design optimization for biocomp models.',
    context_types=DEFAULT_TYPES,
    context={'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve()},
)
class DesignProgram(BaseOptimizationProgram):
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
    disable_tu_masking: Annotated[
        bool, Arg(help='Disable Hard Concrete TU masking (use binary mask system instead)')
    ] = True

    def model_post_init(self, __context):
        self._model = None
        self._dmanager = None
        self._design_id = None
        super().model_post_init(__context)

    @property
    def design_id(self) -> str:
        if self._design_id is None:
            self._design_id = self.unique_id
        return self._design_id

    def get_output_subdir(self) -> str:
        return ''

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

        if isinstance(self.targets, (Target, DataTarget)):
            self.targets = [self.targets]
        logger.info(
            f"Creating DesignManager with {len(self.targets)} targets, {len(networks)} networks, "
            f"{self.sampling.strategy} sampling, TU masking={not self.disable_tu_masking}"
        )
        self._dmanager = DesignManager(
            targets=self.targets,
            networks=networks,
            sampling=self.sampling,
            enable_tu_masking=not self.disable_tu_masking,
        )
        if not self.disable_tu_masking:
            n_networks = len(self._dmanager.networks)
            logger.info(f"TU masking enabled: {self._dmanager.n_tus} TUs × {n_networks} networks")

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
        try:
            networks_data = {
                'networks': self._dmanager.networks,
                'network_names': [n.name for n in self._dmanager.networks],
                'target_names': [_target_name(t) for t in self.targets],
                'n_targets': self._dmanager.n_targets,
            }
            with open(networks_file, 'wb') as f:
                pickle.dump(networks_data, f)
            logger.debug(f"Saved {len(self._dmanager.networks)} networks to {networks_file}")
        except Exception as e:
            logger.warning(f"Failed to save design networks: {e}")

    def enrich_metadata(self):
        if self._dmanager is None:
            return

        networks = self._dmanager.networks
        design_info = {
            "n_targets": self._dmanager.n_targets,
            "target_names": [_target_name(t) for t in self.targets],
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
                'targets': [t.model_dump() for t in self.targets],
                'networks': [n.name for n in networks],
                'evaluation': {
                    'n_eval_samples': self.n_eval_samples,
                    'eval_seed': self.eval_seed,
                    'topk_n': self.topk_n,
                },
                'final_model_dump': self._modeldump,
            }
        )

    async def execute_optimization(self, logger_callbacks, async_handler):
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

        target_names = [_target_name(t) for t in self.targets]
        logger.info(f"Optimizing for {self._dmanager.n_targets} targets: {', '.join(target_names)}")

        network_names = [n.name for n in self._dmanager.networks[:5]]
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
        logger.debug(f"Starting optimization with {len(logger_callbacks)} logger callbacks")

        final_params, loss_history, step_history = start(
            dmanager=self._dmanager,
            dconf=self.design_conf,
            model=self._model,
            loggers=logger_callbacks,
            async_handler=async_handler,
        )

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

        eval_key = jax.random.key(self.eval_seed)

        t0 = time.perf_counter()
        logger.info(f"[1/3] Sampling {self.n_eval_samples} evaluation points...")
        xraw, yraw = sample_for_evaluation(
            dmanager=self._dmanager,
            dconf=self.design_conf,
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
            dconf=self.design_conf,
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
            dconf=self.design_conf,
            k=self.topk_n * topk_candidate_pool,
        )
        logger.info(f"  -> Top-k found in {time.perf_counter() - t2:.2f}s")

        t3 = time.perf_counter()
        try:
            baseline_results = compute_baseline_loss(
                dmanager=self._dmanager,
                model=self._model,
                n_samples=self.n_eval_samples,
                seed=self.eval_seed,
            )
            logger.info(f"  -> Baseline computed in {time.perf_counter() - t3:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to compute baseline loss: {e}")
            baseline_results = {}

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

        self._metadata['evaluation_results'] = {
            'losses_shape': losses.shape,
            'baseline_results': baseline_results,
            'topk_results': [
                {
                    'target': _target_name(self.targets[tid]),
                    'baseline': baseline_results.get(_target_name(self.targets[tid]), {}),
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
            try:
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
            except Exception as e:
                logger.warning(f"Failed to save loss plot: {e}")
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

        n_targets = len(self.targets)
        n_networks = len(self._dmanager.networks)
        n_replicates = self.design_conf.n_replicates

        assert len(topk) == n_targets, f"topk length {len(topk)} != n_targets {n_targets}"

        import biocomp.compute as cmp
        import time

        stack = cmp.ComputeStack(networks=self._dmanager.networks)
        stack.build(self._model.compute_config, enable_tu_masking=True)

        run_datetime = datetime.now().isoformat()
        run_name = self._run_name
        all_design_results = []

        # Pre-compute all unique (rep_id, tid) pairs needed from topk
        needed_pairs: set[tuple[int, int]] = set()
        for tid in range(n_targets):
            for rep_id, net_id, loss_val in topk[tid]:
                needed_pairs.add((rep_id, tid))

        # Pre-commit all needed parameter sets (expensive operation, do once per pair)
        commit_cache: dict[tuple[int, int], list] = {}
        commit_failures: list[tuple[int, int, str]] = []
        t0 = time.perf_counter()
        logger.info(f"Pre-committing {len(needed_pairs)} unique (replicate, target) pairs...")
        for i, (rep_id, tid) in enumerate(sorted(needed_pairs)):
            bparams = tree_get(final_params, (rep_id, tid))
            try:
                commit_cache[(rep_id, tid)] = stack.commit(bparams)
            except Exception as e:
                # Log error but don't crash - return empty network list
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning(f"Commit failed for (rep={rep_id}, target={tid}): {error_msg}")
                commit_failures.append((rep_id, tid, error_msg))
                # Create empty network list so downstream code can handle gracefully
                commit_cache[(rep_id, tid)] = [
                    Network(compute_graph=GraphState(nodes={}, edges={})) for _ in range(n_networks)
                ]
        if commit_failures:
            logger.warning(f"  -> {len(commit_failures)} commits failed (will be skipped)")
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

                # Track valid designs (non-empty recipes) and skip empty ones
                valid_rank = 0
                empty_count = 0
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

                    try:
                        # Use pre-computed commit result (all pairs committed upfront)
                        committed_networks = commit_cache[(rep_id, tid)]
                        cnet = committed_networks[net_id]
                        assert cnet is not None, f"committed network {net_id} is None"

                        recipe = cnet.to_recipe()

                        # Skip empty recipes (all TUs were pruned)
                        if not recipe.content:
                            empty_count += 1
                            network_name = self._dmanager.networks[net_id].name
                            logger.warning(
                                f"Skipping empty recipe for target '{target_name}' "
                                f"(rep={rep_id}, net={net_id}, scaffold='{network_name}', "
                                f"loss={loss_val:.6f}) - all TUs pruned after commit"
                            )
                            continue

                        valid_rank += 1
                        rank = valid_rank  # Use 1-indexed rank for output

                        network_name = self._dmanager.networks[net_id].name
                        f.write(f"  Rank {rank}: Replicate {rep_id}, Network '{network_name}'\n")
                        f.write(f"           Loss: {loss_val:.6f}\n")

                        recipe_yaml = dracon.dump(recipe)
                        recipe_hash = pronounceable_hash48(recipe_yaml.encode('utf-8'))
                        recipe_metadata = {
                            'target_name': target_name,
                            'datetime': run_datetime,
                            'run_name': run_name,
                            'loss': float(loss_val),
                            'rank': rank,
                            'replicate': rep_id,
                            'scaffold_network_name': network_name,
                            'scaffold_network_id': net_id,
                            'recipe_hash': recipe_hash,
                            'model_signature': self._model.signature,
                        }

                        design_dir = target_results_dir / f"design_{rank:02d}"
                        design_dir.mkdir(exist_ok=True)
                        recipe_filename = design_dir / f"{recipe_hash}.yaml"
                        metadata_yaml = yaml.dump(
                            {'_metadata': recipe_metadata}, default_flow_style=False
                        )
                        with open(recipe_filename, 'w') as rf:
                            rf.write(metadata_yaml + '\n' + recipe_yaml)
                        network_pickle = design_dir / f"{recipe_hash}.pickle"
                        with open(network_pickle, 'wb') as npf:
                            pickle.dump(cnet, npf)

                        target_input_names = getattr(target, 'input_names', None)
                        scaffold_input_proteins = None
                        try:
                            scaffold_input_proteins = cnet.get_inverted_input_proteins()
                        except Exception:
                            pass

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

                    except Exception as e:
                        logger.warning(
                            f"Failed to process design for target '{target_name}' "
                            f"(rep={rep_id}, net={net_id}): {e}"
                        )

                # Log summary of empty recipes for this target
                if empty_count > 0:
                    logger.info(
                        f"Target '{target_name}': skipped {empty_count} empty recipes, "
                        f"saved {valid_rank} valid designs"
                    )

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
            self._print_rich_design_summary(all_design_results)

        logger.info(f"Saved best designs summary to {summary_file}")
        if design_results_dir.exists() and any(design_results_dir.iterdir()):
            logger.info(f"Saved design results to {design_results_dir}")

    def _print_rich_design_summary(self, all_design_results: list[dict], top_n: int = 3):
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        from biocomp.plotting.ascii_heatmap import heatmap
        from biocomptools.toollib.networkprediction import NetworkPrediction
        from biocomptools.modelmodel import NetworkModel
        from collections import defaultdict

        console = Console()

        by_target = defaultdict(list)
        for d in all_design_results:
            by_target[d['target_name']].append(d)

        for target_name in by_target:
            by_target[target_name].sort(key=lambda x: x['loss'])

        console.print()
        console.rule("[bold cyan]DESIGN RESULTS SUMMARY[/bold cyan]", style="cyan")

        for target_name, designs in by_target.items():
            target = designs[0].get('target') if designs else None
            res = self._dmanager.grid_resolution if self._dmanager else (32, 32)
            xres, yres = res

            console.print()
            console.print(Panel(f"[bold white]{target_name}[/bold white]", style="blue", expand=False))

            if target and self._model:
                try:
                    X_lat, Y_target = target.get_lattice(resolution=res, seed=0)
                    Y_target_grid = np.flipud(np.asarray(Y_target).reshape(yres, xres))
                    vmin_t, vmax_t = float(Y_target_grid.min()), float(Y_target_grid.max())
                    target_hm = heatmap(Y_target_grid, vmin=vmin_t, vmax=vmax_t, xres=36, yres=12, show_colorbar=False)
                    console.print("[dim]TARGET:[/dim]")
                    console.print(target_hm)
                    console.print(f"[dim]{vmin_t:.2f} ░▒▓█ {vmax_t:.2f}[/dim]")
                except Exception as e:
                    console.print(f"[red]Failed to render target: {e}[/red]")

            table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            table.add_column("Rank", style="dim", width=4)
            table.add_column("Network", style="cyan", width=30)
            table.add_column("Loss", justify="right", width=10)
            table.add_column("Corr", justify="right", width=8)
            table.add_column("MSE", justify="right", width=10)
            table.add_column("Recipe", style="dim", width=20)

            for i, design in enumerate(designs[:top_n]):
                network = design.get('network')
                loss = design.get('loss', float('nan'))
                recipe_hash = design.get('recipe_hash', 'unknown')[:18]

                if network is None or target is None or self._model is None:
                    table.add_row(str(i + 1), design.get('network_name', '?'), f"{loss:.6f}", "N/A", "N/A", recipe_hash)
                    continue

                try:
                    nm = NetworkModel(model=self._model, network=network)
                    X_lat, Y_target = target.get_lattice(resolution=res, seed=0)
                    pred = NetworkPrediction(predict_at=[X_lat], network_model=nm, already_latent=True)
                    data = pred.get_data(rescale_latent=False)[0]
                    Y_pred = np.flipud(np.asarray(data.y).reshape(yres, xres))
                    Y_target_grid = np.flipud(np.asarray(Y_target).reshape(yres, xres))

                    corr = np.corrcoef(Y_target_grid.ravel(), Y_pred.ravel())[0, 1]
                    mse = np.mean((Y_pred - Y_target_grid) ** 2)

                    table.add_row(
                        str(i + 1),
                        design.get('network_name', '?')[:28],
                        f"{loss:.6f}",
                        f"{corr:.4f}",
                        f"{mse:.6f}",
                        recipe_hash,
                    )

                    if i == 0:
                        vmin = min(Y_pred.min(), Y_target_grid.min())
                        vmax = max(Y_pred.max(), Y_target_grid.max())
                        pred_hm = heatmap(Y_pred, vmin=vmin, vmax=vmax, xres=36, yres=12, show_colorbar=False)
                        console.print()
                        console.print("[green]BEST PREDICTION:[/green]")
                        console.print(pred_hm)
                        console.print(f"[dim]{vmin:.2f} ░▒▓█ {vmax:.2f}[/dim]")

                except Exception as e:
                    table.add_row(str(i + 1), design.get('network_name', '?'), f"{loss:.6f}", "err", "err", recipe_hash)
                    logger.debug(f"Failed prediction for {design.get('network_name')}: {e}")

            console.print()
            console.print(table)

        console.rule(style="cyan")

    def _generate_design_diagnostic_plots(self, save_dir, design_results: list[dict]):
        """Generate diagnostic plots using batched evaluation."""
        from biocomptools.plot import PlotJob
        from biocomptools.toollib.figuremakers.designutils import DesignResult
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_results = len(design_results)
        logger.info(f"Generating diagnostic plots for {n_results} designs...")

        # convert dicts to DesignInput objects
        inputs = [
            DesignInput(
                network=r['network'],
                target=r['target'],
                target_name=r['target_name'],
                rank=r['rank'],
                replicate=r['replicate'],
                scaffold_network_name=r['network_name'],
                loss=r['loss'],
                recipe_hash=r['recipe_hash'],
                run_name=self._run_name,
                design_dir=r['design_dir'],
            )
            for r in design_results
        ]

        # batch evaluate all designs
        evaluator = DesignEvaluator(self._model, max_evals=50000)
        evaluated = evaluator.evaluate_designs(inputs)

        n_valid = sum(1 for ev in evaluated if ev.is_valid)
        if n_valid < n_results:
            logger.warning(
                f"Skipped {n_results - n_valid} design(s) with invalid network structure. "
                f"Proceeding with {n_valid} valid designs."
            )

        def _generate_single_plot(ev) -> tuple[str, Exception | None]:
            """Worker function to generate a single design plot."""
            try:
                if not ev.is_valid:
                    return ev.input.recipe_hash, None

                inp = ev.input
                design_dir = Path(inp.design_dir)

                result = DesignResult(
                    network=inp.network,
                    target=inp.target,
                    target_name=inp.target_name,
                    rank=inp.rank,
                    replicate=inp.replicate,
                    scaffold_network_name=inp.scaffold_network_name,
                    loss=inp.loss,
                    recipe_hash=inp.recipe_hash,
                    run_name=inp.run_name,
                    model=self._model,
                    gt_data=ev.gt_data,
                    pred_data=ev.pred_data,
                    lattice_data=ev.lattice_data,
                    lattice_grid=ev.lattice_grid,
                    lattice_extent=ev.lattice_extent,
                    lattice_resolution=ev.lattice_resolution,
                    design_nre=ev.design_nre,
                    baseline_nre=ev.baseline_nre,
                )

                if is_design_debug_enabled():
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
                            'network_inputs': _safe_get_input_proteins(inp.network),
                            'original_network_inputs': (
                                _safe_get_input_proteins(inp.target.original_network)
                                if hasattr(inp.target, 'original_network')
                                and inp.target.original_network
                                else None
                            ),
                            'design_nre': ev.design_nre,
                            'baseline_nre': ev.baseline_nre,
                        },
                        output_dir=str(design_dir),
                        mode="design",
                    )

                PlotJob.invoke(
                    'biocomp-jobs/plot/auto_figures/autofig_design_summary.yaml',
                    result=result,
                    output_dir=str(design_dir),
                )
                return inp.recipe_hash, None
            except Exception as e:
                return ev.input.recipe_hash, e

        max_workers = min(8, len(evaluated))
        completed, failed = 0, 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generate_single_plot, ev): ev for ev in evaluated}
            for future in as_completed(futures):
                recipe_hash, error = future.result()
                if error is None:
                    completed += 1
                    logger.debug(f"Generated summary plot for {recipe_hash}")
                else:
                    failed += 1
                    logger.warning(f"Failed to generate plot for {recipe_hash}: {error}")
                    logger.debug(traceback.format_exc())

        logger.info(f"Plot generation complete: {completed} succeeded, {failed} failed")


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
