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
)
from biocomp.network import Network, recipe_to_networks
from biocomp.recipe import Recipe
from biocomp.jaxutils import tree_to_np, tree_get

from dracon.commandline import Arg
import dracon
import asyncio

import sys
import numpy as np
import jax
import yaml
from pathlib import Path
from typing import Annotated, Optional
from pydantic import Field
import pickle
from datetime import datetime

logger = get_logger(__name__)


class DesignProgram(BaseOptimizationProgram):
    design_conf: Annotated[DesignConfig, Arg(help='Design optimization config')] = Field(
        default_factory=lambda: DesignConfig()
    )

    targets: Annotated[
        list[TargetUnion] | TargetUnion, Arg(help='Design targets (SVG files or DataTarget)')
    ] = Field(default_factory=list)

    networks: Annotated[Optional[list[Network] | Network], Arg(help='Networks to optimize')] = None
    # Scaffolds: generic circuit topologies with unlocked parameters, optimized during design
    scaffolds: Annotated[
        Optional[list[Recipe] | Recipe],
        Arg(help='Base recipes to optimize (converted to networks)'),
    ] = None

    # Sampling configuration
    sampling: Annotated[SamplingConfigUnion, Arg(help='Sampling strategy configuration')] = Field(
        default_factory=UniformSampling
    )

    # for quick use
    network_subset_size: Annotated[Optional[int], Arg(help='Limit networks to first N')] = None

    # Model can be specified by path/name directly, or via model_selector for complex queries
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

    topk_n: Annotated[int, Arg(help='Number of top designs to keep per target')] = 64

    plot_results: Annotated[bool, Arg(help='Generate result plots')] = True
    plot_n_samples: Annotated[int, Arg(help='Max samples to plot')] = 5000
    skip_evaluation: Annotated[
        bool, Arg(help='Skip post-optimization evaluation (useful for DataTarget)')
    ] = False
    show_difference_plots: Annotated[bool, Arg(help='Show difference plots')] = False

    # TU masking options
    enable_tu_masking: Annotated[bool, Arg(help='Enable Hard Concrete TU masking for architecture search')] = False

    def model_post_init(self, __context):
        # Initialize private attributes
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
        return ''  # no subdirectory - outputs go directly in run folder

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

        # Build networks from scaffolds or use directly specified networks
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
                scaffold_networks = recipe_to_networks(scaffold, invert=True, inversion_mode="main")
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
            f"{self.sampling.strategy} sampling, TU masking={self.enable_tu_masking}"
        )
        self._dmanager = DesignManager(
            targets=self.targets, networks=networks, sampling=self.sampling,
            enable_tu_masking=self.enable_tu_masking,
        )
        if self.enable_tu_masking:
            logger.info(f"TU masking enabled: {self._dmanager.n_tus} TUs identified")

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

    def enrich_metadata(self):
        # Skip if not initialized yet
        if self._dmanager is None:
            return

        # Use the networks from dmanager which has the resolved networks
        networks = self._dmanager.networks

        def get_target_name(t):
            if isinstance(t, DataTarget):
                return t.name or "data_target"
            else:
                return t.name or Path(t.path).stem

        design_info = {
            "n_targets": self._dmanager.n_targets,
            "target_names": [get_target_name(t) for t in self.targets],
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

        # Log target details
        def get_target_name(t):
            if isinstance(t, DataTarget):
                return t.name or "data_target"
            else:
                return t.name or Path(t.path).stem

        target_names = [get_target_name(t) for t in self.targets]
        logger.info(f"Optimizing for {self._dmanager.n_targets} targets: {', '.join(target_names)}")

        # Log network details
        network_names = [n.name for n in self._dmanager.networks[:5]]  # Show first 5
        if len(self._dmanager.networks) > 5:
            network_names.append(f"... and {len(self._dmanager.networks) - 5} more")
        logger.info(f"Using {len(self._dmanager.networks)} networks: {', '.join(network_names)}")

        # Log optimization parameters
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
            f"Optimization completed. Final loss: {loss_history[-1]:.4f}" if loss_history else "Optimization completed. No loss history"
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
        logger.info(f"  -> Sampled in {time.perf_counter() - t0:.2f}s (shapes: x={xraw.shape}, y={yraw.shape})")

        t1 = time.perf_counter()
        logger.info(f"[2/3] Running forward pass evaluation (chunk_size={self.max_eval_chunk_size})...")
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
        logger.info(f"  -> Losses: min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}")

        t2 = time.perf_counter()
        logger.info(f"[3/3] Finding top {self.topk_n} designs...")

        topk = get_topk_replicate_network_pairs(
            losses=losses,
            dmanager=self._dmanager,
            dconf=self.design_conf,
            k=self.topk_n,
        )
        logger.info(f"  -> Top-k found in {time.perf_counter() - t2:.2f}s")

        def get_target_name(t):
            if isinstance(t, DataTarget):
                return t.name or "data_target"
            else:
                return t.name or Path(t.path).stem

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
            target_name = get_target_name(target)
            rep_id, net_id, loss_val = topk[tid][0]
            network_name = self._dmanager.networks[net_id].name

            # Show baseline comparison if available
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
                    'target': get_target_name(self.targets[tid]),
                    'baseline': baseline_results.get(get_target_name(self.targets[tid]), {}),
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
                    if arr.ndim == 2:
                        # (n_replicates, n_targets) -> average over targets -> (n_replicates, 1)
                        processed.append(np.nanmean(arr, axis=1, keepdims=True))
                    elif arr.ndim >= 3:
                        # (n_replicates, n_targets, ...) -> average over targets
                        processed.append(np.nanmean(arr, axis=1))
                    else:
                        processed.append(arr.reshape(-1, 1) if arr.ndim == 1 else arr)
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
        assert final_params is not None, "Final params required to save designs"
        assert self._model is not None, "Model required to save designs"
        assert self._dmanager is not None, "Design manager required to save designs"

        import biocomp.compute as cmp

        stack = cmp.ComputeStack(networks=self._dmanager.networks)
        stack.build(self._model.compute_config)

        run_datetime = datetime.now().isoformat()
        run_name = self._run_name

        def get_target_name(t):
            if isinstance(t, DataTarget):
                return t.name or "data_target"
            return t.name or Path(t.path).stem

        all_design_results = []  # collect for diagnostic plots

        with open(summary_file, 'w') as f:
            f.write("BEST DESIGN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Date: {run_datetime}\n")
            f.write(f"Model: {self._model.signature}\n\n")

            best_per_target = {}

            for tid, target in enumerate(self.targets):
                target_name = get_target_name(target)
                f.write(f"Target: {target_name}\n")
                f.write("-" * 30 + "\n")

                target_results_dir = design_results_dir / target_name
                target_results_dir.mkdir(exist_ok=True)

                for rank, (rep_id, net_id, loss_val) in enumerate(topk[tid], 1):
                    network_name = self._dmanager.networks[net_id].name
                    f.write(f"  Rank {rank}: Replicate {rep_id}, Network '{network_name}'\n")
                    f.write(f"           Loss: {loss_val:.6f}\n")

                    bparams = tree_get(final_params, (rep_id, tid))

                    try:
                        committed_networks = stack.commit(bparams)
                        cnet = committed_networks[net_id]
                        assert cnet is not None, f"Committed network {net_id} is None"

                        recipe = cnet.to_recipe()
                        recipe_yaml = dracon.dump(recipe)

                        # hash recipe content for unique identifier
                        recipe_hash = pronounceable_hash48(recipe_yaml.encode('utf-8'))

                        # rich metadata for recipe
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

                        # directory: design_results/{target}/design_{rank:02d}/
                        design_dir = target_results_dir / f"design_{rank:02d}"
                        design_dir.mkdir(exist_ok=True)

                        # save recipe as {hash}.yaml
                        recipe_filename = design_dir / f"{recipe_hash}.yaml"
                        metadata_yaml = yaml.dump(
                            {'_metadata': recipe_metadata}, default_flow_style=False
                        )
                        with open(recipe_filename, 'w') as rf:
                            rf.write(metadata_yaml + '\n' + recipe_yaml)

                        # save network pickle
                        network_pickle = design_dir / f"{recipe_hash}.pickle"
                        with open(network_pickle, 'wb') as npf:
                            pickle.dump(cnet, npf)

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
                        }

                        all_design_results.append(design_info)
                        if rank == 1:
                            best_per_target[target_name] = design_info

                        f.write(f"           Recipe: {recipe_hash}.yaml\n")
                        f.write(f"           Path: {design_dir.relative_to(save_dir)}/\n")
                        logger.debug(f"Saved design {target_name} rank {rank}: {recipe_hash}")

                    except Exception as e:
                        logger.warning(f"Failed to save design {target_name} rank {rank}: {e}")
                        f.write(f"           Recipe: Failed ({e})\n")

                f.write("\n")

        # save best_per_target dict for later use
        if best_per_target:
            best_designs_file = save_dir / 'best_designs.pickle'
            with open(best_designs_file, 'wb') as f:
                for target_data in best_per_target.values():
                    target_data['params'] = tree_to_np(target_data['params'])
                pickle.dump(best_per_target, f)
            logger.info(f"Saved best designs data to {best_designs_file}")

        # generate diagnostic summary plots
        if self.plot_results and all_design_results:
            self._generate_design_diagnostic_plots(save_dir, all_design_results)

        logger.info(f"Saved best designs summary to {summary_file}")
        if design_results_dir.exists() and any(design_results_dir.iterdir()):
            logger.info(f"Saved design results to {design_results_dir}")

    def _generate_design_diagnostic_plots(self, save_dir, design_results: list[dict]):
        """Generate diagnostic summary plots for each design result."""
        from biocomptools.plot import PlotJob
        from biocomptools.toollib.figuremakers.designutils import DesignResult

        logger.info(f"Generating diagnostic plots for {len(design_results)} designs...")

        for r in design_results:
            try:
                design_dir = Path(r['design_dir'])
                result = DesignResult(
                    network=r['network'],
                    target=r['target'],
                    target_name=r['target_name'],
                    rank=r['rank'],
                    replicate=r['replicate'],
                    scaffold_network_name=r['network_name'],
                    loss=r['loss'],
                    recipe_hash=r['recipe_hash'],
                    run_name=self._run_name,
                    model=self._model,
                )
                PlotJob.invoke(
                    'biocomp-jobs/plot/auto_figures/autofig_design_summary.yaml',
                    result=result,
                    output_dir=str(design_dir),
                )
                logger.debug(f"Generated summary plot: {design_dir / r['recipe_hash']}_summary.pdf")

            except Exception as e:
                logger.warning(f"Failed to generate plot for {r.get('recipe_hash', '?')}: {e}")
                import traceback
                logger.debug(traceback.format_exc())


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
