from biocomptools.optimtools import (
    BaseOptimizationProgram,
    run_optimization_program,
    Logger,
)
from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.trainutils import make_json_ready
from biocomptools.logging_config import get_logger

from biocomp.design import (
    start,
    DesignManager,
    DesignConfig,
    Target,
    sample_for_evaluation,
    evaluate_design,
    get_topk_replicate_network_pairs,
    plot_design_results,
    distance_loss,
)
from biocomp.old_network.network import Network
from biocomp.jaxutils import tree_to_np, tree_get
from biocomptools.configs.designs.networks import ALL_NETWORKS, TWO_AND_ONE_NETWORKS

from dracon.commandline import Arg
import asyncio

import sys
import numpy as np
import jax
from pathlib import Path
from typing import Annotated, Optional, Literal, Any
from pydantic import Field
import pickle
from functools import partial

logger = get_logger(__name__)

DEFAULT_NETWORKS = [TWO_AND_ONE_NETWORKS[0]]
DEFAULT_NETWORKS = ALL_NETWORKS


class DesignProgram(BaseOptimizationProgram):
    design_conf: Annotated[DesignConfig, Arg(help='Design optimization config')] = Field(
        default_factory=lambda: DesignConfig()
    )

    targets: Annotated[list[Target] | Target, Arg(help='Design targets (SVG files)')] = Field(
        default_factory=list
    )

    networks: Annotated[Optional[list[Network] | Network], Arg(help='Networks to optimize')] = None

    # Network subset configuration
    network_subset_size: Annotated[Optional[int], Arg(help='Limit networks to first N')] = None

    model_selector: Annotated[ModelSelector, Arg(help='Model selector for pre-trained model')] = (
        Field(default_factory=ModelSelector)
    )

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
    show_difference_plots: Annotated[bool, Arg(help='Show difference plots')] = False

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
        return 'design'

    def initialize_context(self):
        logger.info("Initializing design context...")
        logger.debug(f"Model selector: {self.model_selector}")

        with self.db_session as session:
            selected_model = self.model_selector.get_model(session)
            logger.info(f"Loading model: {selected_model.name if selected_model else 'None'}")
            self._model = selected_model.load() if selected_model else None
            session.expunge_all()
            session.close()

        if self._model is None:
            raise ValueError(f"Could not load model using selector: {self.model_selector}")

        logger.info(f"Successfully loaded model with signature: {self._model.signature}")

        # Use default networks if none specified
        if isinstance(self.networks, Network):
            self.networks = [self.networks]

        if self.networks is not None:
            networks = self.networks
            logger.debug(f"Using {len(networks)} user-specified networks")
        else:
            networks = DEFAULT_NETWORKS
            logger.debug(f"Using {len(networks)} default networks")
            # Apply subset if specified
            if self.network_subset_size is not None:
                networks = networks[: self.network_subset_size]
                logger.info(f"Limited networks to first {self.network_subset_size}")

        if isinstance(self.targets, Target):
            self.targets = [self.targets]
        logger.info(
            f"Creating DesignManager with {len(self.targets)} targets and {len(networks)} networks"
        )
        self._dmanager = DesignManager(targets=self.targets, networks=networks)

    def _get_logger_context(self) -> dict:
        context = super()._get_logger_context()
        context.update(
            {
                'design_conf': self.design_conf,
                'targets': self.targets,
                'networks': self.networks,
                'dmanager': self._dmanager,
            }
        )
        return context

    def enrich_metadata(self):
        # Skip if not initialized yet
        if self._dmanager is None:
            return

        # Use the networks from dmanager which has the resolved networks
        networks = self._dmanager.networks

        design_info = {
            "n_targets": self._dmanager.n_targets,
            "target_names": [t.name or Path(t.path).stem for t in self.targets],
            "n_networks": len(networks),
            "network_names": [n.name for n in networks],
            "n_replicates": self.design_conf.n_replicates,
            "n_epochs": self.design_conf.n_epochs,
            "batch_size": self.design_conf.batch_size,
            "n_batches_per_epoch": self.design_conf.n_batches_per_epoch,
        }

        self._metadata.update(
            {
                'design_id': self.design_id,
                'run_name': self._run_name,
                'experiment_name': self.experiment_name,
                'model_signature': self._model.signature if self._model else 'unknown',
                'model_selector': self.model_selector.model_dump(),
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
        target_names = [t.name or Path(t.path).stem for t in self.targets]
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
            f"Optimization completed. Final loss shape: {loss_history[-1].shape if loss_history else 'No loss history'}"
        )

        await self._evaluate_and_save_results(final_params, loss_history)

        return final_params, loss_history, step_history

    async def _evaluate_and_save_results(self, final_params, loss_history):
        logger.info("=" * 60)
        logger.info("Evaluating design performance...")
        logger.debug(
            f"Evaluation parameters: n_samples={self.n_eval_samples}, seed={self.eval_seed}"
        )

        assert self._dmanager is not None
        assert self._model is not None

        eval_key = jax.random.key(self.eval_seed)
        logger.debug("Sampling evaluation data...")

        xraw, yraw = sample_for_evaluation(
            dmanager=self._dmanager,
            dconf=self.design_conf,
            final_params=final_params,
            n_eval_samples=self.n_eval_samples,
            key=eval_key,
        )

        logger.debug(f"Sampled evaluation data shapes: xraw={xraw.shape}, yraw={yraw.shape}")

        logger.debug(f"Running design evaluation with max chunk size={self.max_eval_chunk_size}...")

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
        )

        logger.debug(f"Evaluation complete. Losses shape: {losses.shape}")
        logger.debug(
            f"Loss statistics: min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}"
        )

        logger.debug(f"Finding top {self.topk_n} designs for each target...")

        topk = get_topk_replicate_network_pairs(
            losses=losses,
            dmanager=self._dmanager,
            dconf=self.design_conf,
            k=self.topk_n,
        )

        logger.info("=" * 60)
        logger.info("RESULTS: Best replicate/network pairs for each target:")
        for tid, target in enumerate(self._dmanager.targets):
            target_name = target.name or Path(target.path).stem
            rep_id, net_id, loss_val = topk[tid][0]
            network_name = self._dmanager.networks[net_id].name
            logger.info(
                f"  {target_name}: Rep {rep_id}, Net '{network_name}' (loss={loss_val:.4f})"
            )

        self._evaluation_results = (final_params, loss_history, topk, losses, xraw, yraw, yhatdep)

        self._metadata['evaluation_results'] = {
            'losses_shape': losses.shape,
            'topk_results': [
                {
                    'target': self.targets[tid].name or Path(self.targets[tid].path).stem,
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
                all_losses = np.concatenate(loss_history, axis=-1)
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

            if self.plot_results:
                logger.info("Generating visualization plots...")
                plot_dir = save_dir / 'plots'
                plot_dir.mkdir(exist_ok=True)
                logger.debug(
                    f"Plotting with n_samples={self.plot_n_samples}, show_difference={self.show_difference_plots}, plot_top_k={self.topk_n}"
                )

                assert self._dmanager is not None

                # Create target-specific plot subdirectories and plot each target separately
                for tid, target in enumerate(self._dmanager.targets):
                    target_name = target.name or Path(target.path).stem
                    target_plot_dir = plot_dir / target_name
                    target_plot_dir.mkdir(exist_ok=True)

                    # Create single-target topk list for this target only
                    single_target_topk = [topk[tid]]  # wrap in list to maintain expected structure

                    # Plot all top-k results for this target
                    plot_design_results(
                        dmanager=DesignManager(targets=[target], networks=self._dmanager.networks),
                        dconf=self.design_conf,
                        xraw=xraw[:, :, :, tid : tid + 1, :],  # slice to keep only this target
                        yraw=yraw[:, :, :, tid : tid + 1, :],  # slice to keep only this target
                        yhatdep=yhatdep[:, :, tid : tid + 1, :] if yhatdep is not None else None,
                        topk=single_target_topk,
                        n_eval_samples=self.plot_n_samples,
                        save_dir=target_plot_dir,
                        show_difference=self.show_difference_plots,
                        plot_top_k=self.topk_n,
                    )

                total_plots = len(self.targets) * self.topk_n
                logger.info(
                    f"Generated {total_plots} plots organized by target, saved to {plot_dir}"
                )

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
            # Convert loss_history to the expected format
            if isinstance(loss_history, list) and len(loss_history) > 0:
                # Each element in loss_history is shape (n_replicates, batches_per_step)
                # plot_loss expects a list of arrays to concatenate along axis=1
                if len(loss_history) == 1:
                    # Single step, wrap in a list
                    self.save_loss_plot([loss_history[0]], save_dir)
                else:
                    # Multiple steps, let plot_loss concatenate them
                    self.save_loss_plot(loss_history, save_dir)
            elif isinstance(loss_history, np.ndarray):
                # Wrap single array in a list for plot_loss
                self.save_loss_plot([loss_history], save_dir)
        logger.debug("Saving metadata...")
        self.save_metadata(save_dir)
        logger.info("All outputs saved successfully.")

    def _save_best_designs_summary(self, save_dir, topk):
        summary_file = save_dir / 'best_designs_summary.txt'
        recipes_dir = save_dir / 'best_recipes'
        recipes_dir.mkdir(exist_ok=True)

        # Get final params and stack from evaluation results
        final_params = self._evaluation_results[0] if hasattr(self, '_evaluation_results') else None

        # Build the compute stack once
        if final_params is not None and self._model is not None:
            import biocomp.compute as cmp

            stack = cmp.ComputeStack(networks=self._dmanager.networks)
            stack.build(self._model.compute_config)
        else:
            stack = None
            logger.warning("Cannot commit networks: final_params or model not available")

        with open(summary_file, 'w') as f:
            f.write("BEST DESIGN SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            best_per_target = {}

            for tid, target in enumerate(self.targets):
                target_name = target.name or Path(target.path).stem
                f.write(f"Target: {target_name}\n")
                f.write("-" * 30 + "\n")

                # Create target-specific recipe subfolder
                target_recipes_dir = recipes_dir / target_name
                target_recipes_dir.mkdir(exist_ok=True)

                # Store all designs for this target (not just best)
                target_designs = []

                for rank, (rep_id, net_id, loss_val) in enumerate(topk[tid], 1):
                    network_name = self._dmanager.networks[net_id].name
                    f.write(f"  Rank {rank}: Replicate {rep_id}, Network '{network_name}'\n")
                    f.write(f"           Loss: {loss_val:.6f}\n")

                    # Save all top-k network recipes (not just rank 1)
                    if final_params is not None and stack is not None:
                        # Get params for this replicate and target
                        bparams = tree_get(final_params, (rep_id, tid))

                        # Commit the networks to get post-processed versions
                        try:
                            committed_networks = stack.commit(bparams)
                            cnet = committed_networks[net_id]

                            # Store design info
                            design_info = {
                                'rank': rank,
                                'replicate': rep_id,
                                'network_name': network_name,
                                'network_id': net_id,
                                'network': cnet,
                                'loss': float(loss_val),
                                'params': bparams,
                            }
                            target_designs.append(design_info)

                            # Save best design to best_per_target dict
                            if rank == 1:
                                best_per_target[target_name] = design_info

                            # Generate pretty recipe
                            network_recipe = cnet.to_pretty_recipe()

                            # Save recipe to file with rank prefix
                            recipe_filename = (
                                target_recipes_dir
                                / f"rank{rank:02d}_{target_name}_rep{rep_id}_net{net_id}.txt"
                            )
                            with open(recipe_filename, 'w') as rf:
                                rf.write(f"# Design for target: {target_name}\n")
                                rf.write(f"# Rank: {rank}\n")
                                rf.write(f"# Network: {network_name}\n")
                                rf.write(f"# Replicate: {rep_id}\n")
                                rf.write(f"# Loss: {loss_val:.6f}\n")
                                rf.write("#" + "=" * 59 + "\n\n")
                                rf.write(network_recipe)

                            f.write(
                                f"           Recipe saved: {target_name}/{recipe_filename.name}\n"
                            )
                            logger.debug(
                                f"Saved recipe for {target_name} rank {rank} to {recipe_filename}"
                            )
                            # also save network as pickle
                            network_pickle_file = (
                                target_recipes_dir
                                / f"rank{rank:02d}_{target_name}_rep{rep_id}_net{net_id}.pickle"
                            )
                            with open(network_pickle_file, 'wb') as npf:
                                pickle.dump(cnet, npf)

                        except Exception as e:
                            logger.warning(
                                f"Failed to commit/save recipe for {target_name} rank {rank}: {e}"
                            )
                            f.write(f"           Recipe: Failed to generate ({e})\n")

                f.write("\n")

        # Save the best_per_target dict as pickle for later use
        if best_per_target:
            best_designs_file = save_dir / 'best_designs.pickle'
            with open(best_designs_file, 'wb') as f:
                # Convert JAX arrays to numpy before saving
                for target_data in best_per_target.values():
                    target_data['params'] = tree_to_np(target_data['params'])
                pickle.dump(best_per_target, f)
            logger.info(f"Saved best designs data to {best_designs_file}")

        logger.info(f"Saved best designs summary to {summary_file}")
        if recipes_dir.exists() and any(recipes_dir.iterdir()):
            logger.info(f"Saved network recipes to {recipes_dir}")


async def main_async():
    await run_optimization_program(
        DesignProgram,
        'biocomp-design',
        'Run design optimization for biocomp models.',
        sys.argv[1:],
    )


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
