"""
Logger for visualizing design optimization results with circuit diagrams.

Generates 4-panel figures showing:
1. Network compute diagram
2. Genetic circuit schematic
3. Design target
4. Predicted output

For top-n candidates at each step.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any, Union
from pydantic import Field, ConfigDict
from copy import deepcopy

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def render_target_heatmap(
    ax: plt.Axes,
    target: Any,
    resolution: Tuple[int, int] = (48, 48),
    title: Optional[str] = None,
    cmap: str = 'viridis',
):
    """Render a design target as a 2D heatmap."""
    from biocomp.design import Target, DataTarget

    if hasattr(target, 'get_lattice'):
        # DataTarget
        X, Y = target.get_lattice(resolution)
        Y = np.asarray(Y).squeeze()
        if Y.ndim == 1:
            Y = Y.reshape(resolution)
    elif hasattr(target, 'get_samples'):
        # Target (SVG-based)
        X, Y = target.get_samples(n=resolution[0] * resolution[1], grid=resolution)
        Y = np.asarray(Y).squeeze()
        if Y.ndim == 1:
            Y = Y.reshape(resolution)
    else:
        ax.text(0.5, 0.5, 'Unknown target type', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title or 'Design Target')
        return

    # Ensure Y is 2D
    if Y.ndim > 2:
        Y = Y.squeeze()
    if Y.ndim == 1:
        Y = Y.reshape(resolution)

    # Plot as heatmap
    im = ax.imshow(Y, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or 'Design Target')


def render_prediction_heatmap(
    ax: plt.Axes,
    model: Any,
    network: Any,
    target: Any,
    resolution: Tuple[int, int] = (48, 48),
    title: Optional[str] = None,
    cmap: str = 'viridis',
    n_z_samples: int = 1000,
    use_latent_space: bool = True,
):
    """Render model prediction on a grid as a 2D heatmap."""
    import jax.numpy as jnp
    import jax

    # Get the xlim/ylim from target
    if hasattr(target, 'xlim') and target.xlim is not None:
        xlim = target.xlim
    else:
        xlim = (0.0, 0.5)

    if hasattr(target, 'ylim') and target.ylim is not None:
        ylim = target.ylim
    else:
        ylim = (0.0, 0.5)

    # Create grid in latent space (same as target)
    x = np.linspace(xlim[0], xlim[1], resolution[0])
    y = np.linspace(ylim[0], ylim[1], resolution[1])
    xx, yy = np.meshgrid(x, y)
    grid_points_latent = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    n_grid_points = resolution[0] * resolution[1]

    # Convert to raw space for prediction if needed
    if use_latent_space and hasattr(model, 'rescaler') and model.rescaler is not None:
        # Try inv() first (CompressedSymLogRescaler), then bwd() (DataRescaler)
        if hasattr(model.rescaler, 'inv'):
            grid_points = model.rescaler.inv(grid_points_latent)
        elif hasattr(model.rescaler, 'bwd'):
            grid_points = model.rescaler.bwd(grid_points_latent)
        else:
            grid_points = grid_points_latent
    else:
        grid_points = grid_points_latent

    # Run prediction
    try:
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction

        network_model = NetworkModel(network=[network], model=model)
        predictor = NetworkPrediction(
            predict_at=[grid_points],
            network_model=network_model,
            max_evals=n_z_samples,
            z_value='uniform',
            verbose=False,
        )
        pred_data = predictor.get_data()[0]
        pred_y = np.asarray(pred_data.y).squeeze()

        # Convert prediction to latent space for comparison with target
        if use_latent_space and hasattr(model, 'rescaler') and model.rescaler is not None:
            pred_y = model.rescaler.fwd(pred_y)

        # Handle different output shapes
        if pred_y.ndim == 2:
            pred_y = pred_y[:, 0]  # take first output

        # Ensure we have right number of points
        if pred_y.shape[0] != n_grid_points:
            logger.warning(f"Prediction shape {pred_y.shape} doesn't match grid {resolution}")
            pred_grid = pred_y[:min(n_grid_points, pred_y.shape[0])].reshape(-1)
            # Pad or truncate
            if len(pred_grid) < n_grid_points:
                pred_grid = np.pad(pred_grid, (0, n_grid_points - len(pred_grid)), mode='edge')
            pred_grid = pred_grid[:n_grid_points].reshape(resolution)
        else:
            pred_grid = pred_y.reshape(resolution)

    except Exception as e:
        logger.warning(f"Failed to get prediction: {e}")
        ax.text(0.5, 0.5, f'Prediction failed:\n{str(e)[:50]}', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.set_title(title or 'Predicted Output')
        return

    # Plot as heatmap
    im = ax.imshow(pred_grid, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title or 'Predicted Output')


class DesignVisualizationLogger(Logger):
    """
    Logger that creates 4-panel visualization of design candidates.

    For each target and top-n candidates, generates:
    - Network compute diagram
    - Genetic circuit schematic
    - Design target heatmap
    - Model prediction heatmap
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    output_dir: Optional[str] = None
    top_n: int = 3  # number of top candidates per target
    grid_resolution: Tuple[int, int] = (48, 48)
    dpi: int = 150
    figsize: Tuple[float, float] = (20, 5)

    # These must be injected by the caller
    model: Optional[Any] = None
    targets: Optional[List[Any]] = None
    dmanager: Optional[Any] = None

    # Internal state
    _step_count: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step_count = 0

    def _get_top_candidates(
        self,
        all_losses: np.ndarray,
        target_id: int,
        n: int,
    ) -> List[Tuple[int, int, float]]:
        """Get top n candidates for a target as (replicate_id, network_id, loss)."""
        all_losses = np.asarray(all_losses)

        # Handle different shapes
        if all_losses.ndim == 2:
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)

        n_replicates, n_targets, n_networks = all_losses.shape
        target_losses = all_losses[:, target_id, :]  # (n_replicates, n_networks)

        # Get top n indices
        flat_losses = target_losses.ravel()
        top_indices = np.argsort(flat_losses)[:n]

        candidates = []
        for flat_idx in top_indices:
            rep_id = flat_idx // n_networks
            net_id = flat_idx % n_networks
            loss = flat_losses[flat_idx]
            candidates.append((int(rep_id), int(net_id), float(loss)))

        return candidates

    def _render_single_candidate(
        self,
        params: Any,
        stack: Any,
        target: Any,
        target_id: int,
        rep_id: int,
        net_id: int,
        loss: float,
        step: int,
        output_path: Path,
    ):
        """Render 4-panel visualization for a single candidate."""
        from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax
        from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

        fig, axes = plt.subplots(1, 4, figsize=self.figsize)

        # Get the specific params for this replicate and target
        try:
            # params shape: (n_replicates, n_targets, ...)
            # We need to extract [rep_id, target_id]
            specific_params = params.slice_along_dims([rep_id, target_id])
        except Exception as e:
            logger.warning(f"Failed to slice params: {e}, trying direct access")
            specific_params = params

        # Commit to get the network with resolved parameters
        try:
            committed_networks = stack.commit(specific_params)
            network = committed_networks[net_id] if net_id < len(committed_networks) else committed_networks[0]
        except Exception as e:
            logger.warning(f"Failed to commit network: {e}")
            network = stack.networks[net_id] if net_id < len(stack.networks) else stack.networks[0]

        # 1. Network diagram
        try:
            render_diagram_to_ax(
                network=network,
                ax=axes[0],
                simplified=True,
                title=f"Network Diagram",
            )
        except Exception as e:
            logger.warning(f"Failed to render diagram: {e}")
            axes[0].text(0.5, 0.5, f'Diagram failed:\n{str(e)[:40]}', ha='center', va='center',
                        transform=axes[0].transAxes, fontsize=8)
            axes[0].set_title("Network Diagram")

        # 2. Genetic circuit
        try:
            render_circuit_to_ax(
                network=network,
                ax=axes[1],
                hide_marker_tus=True,
                title=f"Genetic Circuit",
            )
        except Exception as e:
            logger.warning(f"Failed to render circuit: {e}")
            axes[1].text(0.5, 0.5, f'Circuit failed:\n{str(e)[:40]}', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=8)
            axes[1].set_title("Genetic Circuit")

        # 3. Design target
        try:
            render_target_heatmap(
                ax=axes[2],
                target=target,
                resolution=self.grid_resolution,
                title="Design Target",
            )
        except Exception as e:
            logger.warning(f"Failed to render target: {e}")
            axes[2].text(0.5, 0.5, f'Target failed:\n{str(e)[:40]}', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=8)
            axes[2].set_title("Design Target")

        # 4. Predicted output
        if self.model is not None:
            try:
                render_prediction_heatmap(
                    ax=axes[3],
                    model=self.model,
                    network=network,
                    target=target,
                    resolution=self.grid_resolution,
                    title=f"Prediction (loss={loss:.4f})",
                )
            except Exception as e:
                logger.warning(f"Failed to render prediction: {e}")
                axes[3].text(0.5, 0.5, f'Prediction failed:\n{str(e)[:40]}', ha='center', va='center',
                            transform=axes[3].transAxes, fontsize=8)
                axes[3].set_title(f"Prediction (loss={loss:.4f})")
        else:
            axes[3].text(0.5, 0.5, 'Model not provided', ha='center', va='center',
                        transform=axes[3].transAxes)
            axes[3].set_title(f"Prediction (loss={loss:.4f})")

        # Add suptitle
        target_name = getattr(target, 'name', f'Target {target_id}')
        fig.suptitle(
            f"Step {step} | {target_name} | Rep={rep_id} Net={net_id} | Loss={loss:.6f}",
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.debug(f"Saved visualization to {output_path}")

    def _generate_visualizations(
        self,
        step: int,
        params: Any,
        stack: Any,
        all_losses: np.ndarray,
    ):
        """Generate visualizations for all targets and top candidates."""
        if self.output_dir is None:
            return

        output_path = Path(self.output_dir) / 'visualizations'
        output_path.mkdir(parents=True, exist_ok=True)

        targets = self.targets or []
        if self.dmanager is not None and not targets:
            targets = self.dmanager.targets

        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)

        n_targets = all_losses.shape[-2] if all_losses.ndim >= 2 else 1

        for target_id in range(min(n_targets, len(targets))):
            target = targets[target_id]
            candidates = self._get_top_candidates(all_losses, target_id, self.top_n)

            for rank, (rep_id, net_id, loss) in enumerate(candidates):
                filename = f"step{step:06d}_target{target_id}_rank{rank}_r{rep_id}n{net_id}.png"
                try:
                    self._render_single_candidate(
                        params=params,
                        stack=stack,
                        target=target,
                        target_id=target_id,
                        rep_id=rep_id,
                        net_id=net_id,
                        loss=loss,
                        step=step,
                        output_path=output_path / filename,
                    )
                except Exception as e:
                    logger.error(f"Failed to render candidate {rank} for target {target_id}: {e}")

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
        """Return callbacks for the training loop."""

        def periodic_callback(
            step, training_config, step_history=None, stack=None, **kwargs
        ):
            self._step_count = step

            if step_history is None:
                return

            all_losses = step_history.get('all_losses')
            params = step_history.get('latest_params')

            if all_losses is None or params is None or stack is None:
                logger.warning(f"Missing data for visualization at step {step}")
                return

            try:
                self._generate_visualizations(step, params, stack, all_losses)
            except Exception as e:
                logger.error(f"Visualization generation failed at step {step}: {e}")
                logger.exception(e)

        callbacks = []
        if isinstance(self.periods, int):
            callbacks.append((self.periods, periodic_callback))
        else:
            for period in self.periods:
                callbacks.append((period, periodic_callback))

        return callbacks

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[dict]:
        """Return current metrics (none for this logger)."""
        return {'visualizations_generated': self._step_count}

    def finalize(self):
        """Cleanup after training ends."""
        pass
