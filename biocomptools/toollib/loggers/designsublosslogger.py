"""Design Subloss Logger: Comprehensive debugging visualization for design optimization.

Produces detailed diagnostic plots at each logged step:
- Single Y_true vs Y_pred comparison (not repeated per loss)
- Per-pixel error maps showing where predictions fail
- Per-pixel loss contribution for each loss component
- Regularization term breakdown and impact
- Network variation analysis across scaffolds
- Comprehensive statistics panel
"""

import numpy as np
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.utils import extract_design_step_metrics
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def _get_target_extents(target) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get X/Y coordinate extents from target, handling both old and new attribute names."""
    # Try new names first
    x_extent = getattr(target, 'latent_x', None)
    y_extent = getattr(target, 'latent_y', None)
    # Fall back to legacy names
    if x_extent is None:
        x_extent = getattr(target, 'lattice_x_extent', (0.0, 1.0))
    if y_extent is None:
        y_extent = getattr(target, 'lattice_y_extent', (0.0, 1.0))
    return x_extent, y_extent


class DesignSublossLogger(Logger):
    """Comprehensive debugging logger for design optimization.

    Generates detailed diagnostic plots showing:
    - Y_true vs Y_pred (single comparison, not per-loss)
    - Per-pixel squared error map
    - Per-pixel contribution to each loss component
    - Spatial correlation analysis
    - Regularization impact breakdown
    - Network scaffold comparison
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: str | None = None
    save_pickle: bool = Field(default=True, description="Save raw subloss data as pickle")
    generate_plots: bool = Field(default=True, description="Generate diagnostic plots")
    plot_period: int = Field(default=100, description="How often to generate plots")
    max_networks_to_plot: int = Field(default=4, description="Max networks to show detailed plots")

    _save_dir: Path | None = None
    _dmanager: Any = None
    _design_config: Any = None
    _grid_resolution: tuple[int, int] | None = None
    _history: list[dict] = []
    _step_count: int = 0
    _total_steps: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._step_count = 0
        self._save_dir = None
        self._dmanager = None
        self._design_config = None
        self._grid_resolution = None
        self._total_steps = 0

    def initialize(self, training_program=None):
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / 'subloss_logs'
        else:
            self._save_dir = Path('subloss_logs')

        self._save_dir.mkdir(parents=True, exist_ok=True)

        if training_program:
            if hasattr(training_program, '_dmanager'):
                self._dmanager = training_program._dmanager
                if hasattr(self._dmanager, 'grid_resolution') and self._dmanager.grid_resolution:
                    self._grid_resolution = self._dmanager.grid_resolution
            if hasattr(training_program, 'design_conf'):
                self._design_config = training_program.design_conf
                if hasattr(self._design_config, 'n_epochs') and hasattr(
                    self._design_config, 'n_batches_per_epoch'
                ):
                    self._total_steps = (
                        self._design_config.n_epochs * self._design_config.n_batches_per_epoch
                    )

        logger.info(
            f"DesignSublossLogger initialized: {self._save_dir}, "
            f"resolution={self._grid_resolution}, total_steps={self._total_steps}"
        )

    def _generate_lattice_coordinates(self, target) -> np.ndarray:
        """Generate lattice X coordinates using target's actual extents."""
        if self._grid_resolution is None:
            return None

        xres, yres = self._grid_resolution
        x_extent, y_extent = _get_target_extents(target)

        x_lin = np.linspace(x_extent[0], x_extent[1], xres)
        y_lin = np.linspace(y_extent[0], y_extent[1], yres)
        xx, yy = np.meshgrid(x_lin, y_lin)
        return np.stack([xx.ravel(), yy.ravel()], axis=-1)

    def _extract_subloss_data(self, step: int, step_history: dict) -> dict:
        """Extract comprehensive metrics from step_history."""
        data = extract_design_step_metrics(step_history)
        data["step"] = step
        data["progress"] = step / max(self._total_steps, 1)

        # Subloss-specific: extended all_losses stats
        all_losses = step_history.get("all_losses")
        if all_losses is not None:
            arr = np.asarray(all_losses)
            data["all_losses_shape"] = arr.shape
            data["all_losses_std"] = float(np.nanstd(arr))
            if arr.ndim >= 4:
                per_net = arr[0, 0, 0, :] if arr.ndim == 4 else arr[0, 0, 0, 0, :]
                data["per_network_losses"] = per_net.tolist()

        # Subloss-specific: prediction stats
        yhatdep = step_history.get("yhatdep")
        if yhatdep is not None:
            arr = np.asarray(yhatdep)
            data["yhatdep_shape"] = arr.shape
            data["yhatdep_mean"] = float(np.nanmean(arr))
            data["yhatdep_min"] = float(np.nanmin(arr))
            data["yhatdep_max"] = float(np.nanmax(arr))
            data["yhatdep_std"] = float(np.nanstd(arr))
            data["yhatdep_nan_count"] = int(np.sum(np.isnan(arr)))

        return data

    def _generate_diagnostic_plot(
        self,
        step: int,
        step_history: dict,
        data: dict,
        output_path: Path,
    ):
        """Generate comprehensive diagnostic plot for debugging."""
        if self._grid_resolution is None or self._dmanager is None:
            return

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            logger.warning("matplotlib not available")
            return

        yhatdep = step_history.get("yhatdep")
        if yhatdep is None:
            return

        yhatdep = np.asarray(yhatdep)
        xres, yres = self._grid_resolution

        # Handle multi-dimensional yhatdep
        if yhatdep.ndim == 5:
            yhatdep = yhatdep[0, 0]  # first replicate, first batch
        elif yhatdep.ndim == 4:
            yhatdep = yhatdep[0]
        elif yhatdep.ndim != 3:
            return

        batch_size, n_targets, n_networks = yhatdep.shape
        if batch_size != xres * yres:
            return

        targets = self._dmanager.targets if self._dmanager else []

        # Get loss weights from design config
        loss_weights = {}
        if self._design_config and hasattr(self._design_config, 'loss_function'):
            lf = self._design_config.loss_function
            if hasattr(lf, 'kwargs'):
                for k in ['w_sinkhorn', 'w_lncc', 'w_mse', 'w_spectral']:
                    if k in lf.kwargs:
                        loss_weights[k.replace('w_', '')] = lf.kwargs[k]

        for tid in range(min(n_targets, 1)):  # Focus on first target
            target = targets[tid] if tid < len(targets) else None
            if target is None:
                logger.warning(f"Target {tid} is None, skipping plot")
                continue

            target_name = getattr(target, 'name', f'target_{tid}')
            x_extent, y_extent = _get_target_extents(target)

            # Get Y_true from target
            Y_true = None
            Y_true_grid = None
            try:
                _, Y_grid = target.get_lattice(resolution=(xres, yres), seed=0)
                Y_true = Y_grid.ravel()
                Y_true_grid = Y_grid.reshape(yres, xres)
            except Exception as e:
                logger.warning(f"Failed to get target lattice for {target_name}: {e}")
                # Try to get Y_true from step_history if available
                y_true_key = f"y_true_{tid}"
                if y_true_key in step_history:
                    Y_true = np.asarray(step_history[y_true_key]).ravel()
                    Y_true_grid = Y_true.reshape(yres, xres)
                    logger.info(f"Using Y_true from step_history key {y_true_key}")
                else:
                    logger.warning(
                        f"No Y_true available for target {tid}, will show predictions only"
                    )

            # Select networks to plot (best, worst, median by loss)
            if n_networks > self.max_networks_to_plot:
                all_losses_arr = step_history.get("all_losses")
                if all_losses_arr is not None:
                    arr = np.asarray(all_losses_arr)
                    if arr.ndim >= 4:
                        net_losses = arr[0, 0, 0, :] if arr.ndim == 4 else arr[0, 0, 0, 0, :]
                        sorted_idx = np.argsort(net_losses)
                        # Best, median, worst, and one more
                        net_indices = [
                            sorted_idx[0],  # best
                            sorted_idx[len(sorted_idx) // 2],  # median
                            sorted_idx[-1],  # worst
                        ]
                        if len(sorted_idx) > 3:
                            net_indices.append(sorted_idx[len(sorted_idx) // 4])
                        net_indices = sorted(set(net_indices))[: self.max_networks_to_plot]
                    else:
                        net_indices = list(range(min(n_networks, self.max_networks_to_plot)))
                else:
                    net_indices = list(range(min(n_networks, self.max_networks_to_plot)))
            else:
                net_indices = list(range(n_networks))

            # Create figure with GridSpec for flexible layout
            # Row 0: Header stats
            # Row 1: Y_true, Y_pred per network, Error maps
            # Row 2: Per-loss contribution maps
            # Row 3: Network comparison stats
            n_nets = len(net_indices)
            fig = plt.figure(figsize=(4 * (n_nets + 2), 16))
            gs = GridSpec(4, n_nets + 2, figure=fig, height_ratios=[0.8, 1, 1, 0.8])

            # === ROW 0: Header with comprehensive stats ===
            ax_header = fig.add_subplot(gs[0, :])
            ax_header.axis('off')

            progress_pct = data.get('progress', 0) * 100
            total_loss = data.get('loss', float('nan'))
            penalty_frac = data.get('penalty_fraction', 0) * 100

            header_text = (
                f"STEP {step}/{self._total_steps} ({progress_pct:.1f}%) | "
                f"Target: {target_name} | Grid: {xres}×{yres} | "
                f"X∈[{x_extent[0]:.2f},{x_extent[1]:.2f}] Y∈[{y_extent[0]:.2f},{y_extent[1]:.2f}]\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"TOTAL LOSS: {total_loss:.6f} | "
                f"Penalty contribution: {penalty_frac:.1f}%\n"
            )

            # Add subloss breakdown
            subloss_parts = []
            for key in ['sinkhorn', 'lncc', 'mse', 'spectral']:
                raw_val = data.get(f'subloss_{key}', float('nan'))
                weighted_val = data.get(f'subloss_{key}_weighted', float('nan'))
                weight = loss_weights.get(key, 0)
                if not np.isnan(raw_val) and weight > 0:
                    subloss_parts.append(f"{key}: {raw_val:.4f}×{weight}={weighted_val:.4f}")
            header_text += "SUBLOSSES: " + " | ".join(subloss_parts) + "\n"

            # Add regularization breakdown
            reg_parts = []
            for pname in ['l0_penalty', 'spread_penalty', 'coupling_penalty', 'ern_tying_penalty']:
                val = data.get(pname, 0)
                if val > 1e-8:
                    reg_parts.append(f"{pname.replace('_penalty', '')}: {val:.6f}")
            if reg_parts:
                header_text += "REGULARIZATION: " + " | ".join(reg_parts) + "\n"
            else:
                header_text += "REGULARIZATION: none active\n"

            # TU masking stats
            tu_enabled = data.get('tu_enabled_count', 0)
            tu_total = data.get('tu_total_count', 1)
            tu_prob = data.get('tu_mean_prob', float('nan'))
            header_text += (
                f"TU MASKING: {tu_enabled:.0f}/{tu_total:.0f} enabled "
                f"({100 * tu_enabled / max(tu_total, 1):.1f}%) | mean_prob={tu_prob:.3f}\n"
            )

            # Prediction stats
            yhat_mean = data.get('yhatdep_mean', float('nan'))
            yhat_std = data.get('yhatdep_std', float('nan'))
            yhat_min = data.get('yhatdep_min', float('nan'))
            yhat_max = data.get('yhatdep_max', float('nan'))
            header_text += (
                f"PREDICTIONS: mean={yhat_mean:.4f} std={yhat_std:.4f} "
                f"range=[{yhat_min:.4f}, {yhat_max:.4f}]\n"
            )

            # Target stats for comparison
            if Y_true is not None:
                y_true_mean = np.mean(Y_true)
                y_true_std = np.std(Y_true)
                header_text += (
                    f"TARGET: mean={y_true_mean:.4f} std={y_true_std:.4f} "
                    f"range=[{Y_true.min():.4f}, {Y_true.max():.4f}]"
                )
            else:
                y_true_mean = float('nan')
                y_true_std = float('nan')
                header_text += "TARGET: [DATA UNAVAILABLE - check target.get_lattice()]"

            ax_header.text(
                0.02,
                0.95,
                header_text,
                transform=ax_header.transAxes,
                fontsize=9,
                fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            )

            # === ROW 1: Y_true and Y_pred per network ===
            # First column: Y_true (ground truth)
            ax_true = fig.add_subplot(gs[1, 0])
            if Y_true_grid is not None:
                im = ax_true.imshow(
                    Y_true_grid,
                    cmap='viridis',
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    aspect='equal',
                )
                ax_true.set_title(f"Y_TRUE\nmean={y_true_mean:.3f}", fontsize=9)
                plt.colorbar(im, ax=ax_true, fraction=0.046)
            else:
                ax_true.text(
                    0.5,
                    0.5,
                    "TARGET DATA\nUNAVAILABLE",
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='red',
                    transform=ax_true.transAxes,
                )
                ax_true.set_title("Y_TRUE\n[ERROR]", fontsize=9, color='red')
            ax_true.set_xlabel('X₀')
            ax_true.set_ylabel('X₁')
            ax_true.set_xlim(x_extent)
            ax_true.set_ylim(y_extent)

            # Network predictions and error maps
            for i, nid in enumerate(net_indices):
                Y_pred = yhatdep[:, tid, nid].reshape(yres, xres)

                # Compute per-pixel stats if Y_true available
                if Y_true_grid is not None:
                    corr = np.corrcoef(Y_true_grid.ravel(), Y_pred.ravel())[0, 1]
                else:
                    corr = float('nan')

                # Y_pred
                ax_pred = fig.add_subplot(gs[1, i + 1])
                im = ax_pred.imshow(
                    Y_pred,
                    cmap='viridis',
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    aspect='equal',
                )
                ax_pred.set_title(
                    f"NET {nid}: Y_PRED\nmean={Y_pred.mean():.3f} corr={corr:.3f}", fontsize=9
                )
                ax_pred.set_xlabel('X₀')
                plt.colorbar(im, ax=ax_pred, fraction=0.046)

            # Last column: Best network's error map
            best_nid = net_indices[0]
            Y_pred_best = yhatdep[:, tid, best_nid].reshape(yres, xres)

            ax_err = fig.add_subplot(gs[1, -1])
            if Y_true_grid is not None:
                error_best = np.abs(Y_pred_best - Y_true_grid)
                im = ax_err.imshow(
                    error_best,
                    cmap='Reds',
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    aspect='equal',
                )
                ax_err.set_title(
                    f"BEST NET {best_nid}: |ERROR|\n"
                    f"mean={error_best.mean():.4f} max={error_best.max():.4f}",
                    fontsize=9,
                )
                plt.colorbar(im, ax=ax_err, fraction=0.046)
            else:
                ax_err.text(
                    0.5,
                    0.5,
                    "NO TARGET\nFOR ERROR",
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='red',
                    transform=ax_err.transAxes,
                )
                ax_err.set_title(f"NET {best_nid}: ERROR\n[UNAVAILABLE]", fontsize=9, color='red')
            ax_err.set_xlabel('X₀')
            ax_err.set_xlim(x_extent)
            ax_err.set_ylim(y_extent)

            # === ROW 2: Per-pixel loss contribution maps ===
            ax_title = fig.add_subplot(gs[2, 0])
            ax_title.axis('off')
            if Y_true_grid is not None:
                ax_title.text(
                    0.5,
                    0.5,
                    "PER-PIXEL\nLOSS\nCONTRIBUTION\n(darker=worse)",
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                )
            else:
                ax_title.text(
                    0.5,
                    0.5,
                    "PER-PIXEL\nPREDICTION\nVARIATION\n(no target)",
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='orange',
                )

            # MSE contribution map (or variance map if no target)
            for i, nid in enumerate(net_indices):
                Y_pred = yhatdep[:, tid, nid].reshape(yres, xres)
                ax_mse = fig.add_subplot(gs[2, i + 1])

                if Y_true_grid is not None:
                    sq_error = (Y_pred - Y_true_grid) ** 2
                    im = ax_mse.imshow(
                        sq_error,
                        cmap='hot',
                        origin='lower',
                        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                        aspect='equal',
                    )
                    mse_val = np.mean(sq_error)
                    ax_mse.set_title(f"NET {nid}: SQUARED ERROR\nMSE={mse_val:.4f}", fontsize=9)
                else:
                    # Show prediction magnitude instead
                    im = ax_mse.imshow(
                        Y_pred,
                        cmap='hot',
                        origin='lower',
                        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                        aspect='equal',
                    )
                    ax_mse.set_title(f"NET {nid}: Y_PRED MAG\nmean={Y_pred.mean():.4f}", fontsize=9)
                ax_mse.set_xlabel('X₀')
                plt.colorbar(im, ax=ax_mse, fraction=0.046)

            # Signed error for best network (or gradient map if no target)
            ax_signed = fig.add_subplot(gs[2, -1])
            Y_pred_best = yhatdep[:, tid, best_nid].reshape(yres, xres)

            if Y_true_grid is not None:
                signed_error = Y_pred_best - Y_true_grid
                vmax = max(abs(signed_error.min()), abs(signed_error.max()), 1e-6)
                im = ax_signed.imshow(
                    signed_error,
                    cmap='RdBu_r',
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    aspect='equal',
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax_signed.set_title(
                    f"BEST NET {best_nid}: SIGNED ERROR\n(blue=under, red=over)", fontsize=9
                )
            else:
                # Show spatial gradient of predictions instead
                grad_y, grad_x = np.gradient(Y_pred_best)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                im = ax_signed.imshow(
                    grad_mag,
                    cmap='viridis',
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    aspect='equal',
                )
                ax_signed.set_title(
                    f"NET {best_nid}: GRADIENT MAG\n(spatial variation)", fontsize=9
                )
            ax_signed.set_xlabel('X₀')
            plt.colorbar(im, ax=ax_signed, fraction=0.046)

            # === ROW 3: Network comparison and stats ===
            # Loss distribution across networks
            ax_dist = fig.add_subplot(gs[3, : n_nets // 2 + 1])
            all_losses_arr = step_history.get("all_losses")
            if all_losses_arr is not None:
                arr = np.asarray(all_losses_arr)
                if arr.ndim >= 4:
                    net_losses = arr[0, 0, 0, :] if arr.ndim == 4 else arr[0, 0, 0, 0, :]
                    ax_dist.bar(range(len(net_losses)), net_losses, alpha=0.7)
                    ax_dist.axhline(
                        np.mean(net_losses),
                        color='r',
                        linestyle='--',
                        label=f'mean={np.mean(net_losses):.4f}',
                    )
                    ax_dist.set_xlabel('Network Index')
                    ax_dist.set_ylabel('Loss')
                    ax_dist.set_title('Loss per Network (lower=better)')
                    ax_dist.legend()
                    # Highlight selected networks
                    for nid in net_indices:
                        ax_dist.bar(nid, net_losses[nid], color='red', alpha=0.8)

            # Correlation/std scatter across networks
            ax_corr = fig.add_subplot(gs[3, n_nets // 2 + 1 :])
            stds = []
            means = []
            corrs = []
            for nid in range(n_networks):
                Y_pred = yhatdep[:, tid, nid].ravel()
                stds.append(np.std(Y_pred))
                means.append(np.mean(Y_pred))
                if Y_true is not None:
                    corrs.append(np.corrcoef(Y_true, Y_pred)[0, 1])

            stds = np.array(stds)
            means = np.array(means)

            if Y_true is not None and len(corrs) > 0:
                corrs = np.array(corrs)
                ax_corr.scatter(stds, corrs, alpha=0.6, s=30)
                ax_corr.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_corr.set_xlabel('Prediction Std (variation)')
                ax_corr.set_ylabel('Correlation with Target')
                valid_corrs = corrs[~np.isnan(corrs)]
                if len(valid_corrs) > 0:
                    best_idx = np.nanargmax(corrs)
                    ax_corr.set_title(
                        f'Network Quality: std vs corr\n'
                        f'Best corr={valid_corrs.max():.3f} (net {best_idx})'
                    )
                else:
                    ax_corr.set_title('Network Quality: std vs corr\n(all correlations NaN)')
                # Highlight selected networks
                for nid in net_indices:
                    ax_corr.scatter([stds[nid]], [corrs[nid]], color='red', s=100, marker='*')
            else:
                # No target - show mean vs std instead
                ax_corr.scatter(stds, means, alpha=0.6, s=30)
                ax_corr.set_xlabel('Prediction Std (variation)')
                ax_corr.set_ylabel('Prediction Mean')
                ax_corr.set_title('Network Predictions: std vs mean\n(no target for correlation)')
                for nid in net_indices:
                    ax_corr.scatter([stds[nid]], [means[nid]], color='red', s=100, marker='*')

            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved diagnostic plot to {output_path}")

    def _generate_text_report(self, step: int, data: dict) -> str:
        """Generate detailed text report."""
        lines = [
            "=" * 80,
            f"DESIGN SUBLOSS REPORT - Step {step}",
            f"Progress: {data.get('progress', 0) * 100:.1f}%",
            "=" * 80,
            "",
            "LOSS BREAKDOWN:",
            "-" * 40,
        ]

        total = data.get('loss', 0)
        lines.append(f"  Total Loss: {total:.6f}")

        # Sublosses
        lines.append("")
        lines.append("  Subloss Components:")
        for key in ['sinkhorn', 'lncc', 'mse', 'spectral']:
            raw = data.get(f'subloss_{key}', float('nan'))
            weighted = data.get(f'subloss_{key}_weighted', float('nan'))
            if not np.isnan(raw):
                pct = 100 * weighted / total if total > 0 else 0
                lines.append(
                    f"    {key:>10}: {raw:.6f} (weighted: {weighted:.6f}, {pct:.1f}% of total)"
                )

        # Penalties
        lines.append("")
        lines.append("  Regularization Penalties:")
        for pname in ['l0_penalty', 'spread_penalty', 'coupling_penalty', 'ern_tying_penalty']:
            val = data.get(pname, 0)
            pct = 100 * val / total if total > 0 else 0
            lines.append(
                f"    {pname.replace('_penalty', ''):>12}: {val:.6f} ({pct:.1f}% of total)"
            )

        lines.append("")
        lines.append(
            f"  Total Penalty: {data.get('total_penalty', 0):.6f} ({data.get('penalty_fraction', 0) * 100:.1f}% of total)"
        )

        # Predictions
        lines.extend(["", "PREDICTIONS:", "-" * 40])
        lines.append(f"  Shape: {data.get('yhatdep_shape', 'N/A')}")
        lines.append(f"  Mean: {data.get('yhatdep_mean', float('nan')):.6f}")
        lines.append(f"  Std: {data.get('yhatdep_std', float('nan')):.6f}")
        lines.append(
            f"  Range: [{data.get('yhatdep_min', float('nan')):.6f}, {data.get('yhatdep_max', float('nan')):.6f}]"
        )
        lines.append(f"  NaN count: {data.get('yhatdep_nan_count', 0)}")

        # TU masking
        lines.extend(["", "TU MASKING:", "-" * 40])
        lines.append(
            f"  Enabled: {data.get('tu_enabled_count', 'N/A')}/{data.get('tu_total_count', 'N/A')}"
        )
        lines.append(f"  Mean Probability: {data.get('tu_mean_prob', float('nan')):.6f}")

        # Ratios
        lines.extend(["", "RATIO STATS:", "-" * 40])
        lines.append(f"  Mean: {data.get('ratio_mean', float('nan')):.6f}")
        lines.append(f"  Std: {data.get('ratio_std', float('nan')):.6f}")
        lines.append(
            f"  Range: [{data.get('ratio_min', float('nan')):.6f}, {data.get('ratio_max', float('nan')):.6f}]"
        )

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def _save_step_data(self, step: int, data: dict, step_history: dict):
        if not self.save_pickle:
            return

        import dill as pickle

        step_dir = self._save_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save extracted data
        with open(step_dir / "subloss_data.pickle", 'wb') as f:
            pickle.dump(data, f)

        # Save raw arrays
        raw_data = {}
        for key in ["sublosses", "all_losses", "yhatdep", "tu_stats", "ratio_stats"]:
            if key in step_history:
                val = step_history[key]
                if hasattr(val, '__array__'):
                    raw_data[key] = np.asarray(val)
                elif isinstance(val, dict):
                    raw_data[key] = {
                        k: np.asarray(v) if hasattr(v, '__array__') else v for k, v in val.items()
                    }
                else:
                    raw_data[key] = val

        if raw_data:
            with open(step_dir / "raw_step_history.pickle", 'wb') as f:
                pickle.dump(raw_data, f)

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        self._step_count = step
        batch = view.latest()
        if batch is None:
            return
        step_history = view.to_step_history()
        data = self._extract_subloss_data(step, step_history)
        self._history.append(data)

        if self._save_dir and step % self.plot_period == 0:
            step_dir = self._save_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            report = self._generate_text_report(step, data)
            (step_dir / "subloss_report.txt").write_text(report)
            self._save_step_data(step, data, step_history)

            if self.generate_plots:
                self._generate_diagnostic_plot(
                    step, step_history, data, step_dir / "diagnostic.png"
                )

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        self._step_count = step
        batch = view.latest()
        if batch is None:
            return
        step_history = view.to_step_history()
        data = self._extract_subloss_data(step, step_history)
        self._history.append(data)

        if self._save_dir:
            final_dir = self._save_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            report = self._generate_text_report(step, data)
            (final_dir / "subloss_report.txt").write_text(report)
            self._save_step_data(step, data, step_history)

            if self.generate_plots:
                self._generate_diagnostic_plot(
                    step, step_history, data, final_dir / "diagnostic.png"
                )

            if self.save_pickle:
                import dill as pickle

                with open(self._save_dir / "subloss_history.pickle", 'wb') as f:
                    pickle.dump(self._history, f)

            self._generate_trajectory_plots()

    def _generate_trajectory_plots(self):
        if not self._history or not self.generate_plots:
            return

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps = [h["step"] for h in self._history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss with subloss stacking
        ax = axes[0, 0]
        total = [h.get("loss", float('nan')) for h in self._history]
        ax.plot(steps, total, 'k-', linewidth=2, label='Total')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Weighted sublosses (stacked area)
        ax = axes[0, 1]
        weighted_keys = [
            'subloss_sinkhorn_weighted',
            'subloss_lncc_weighted',
            'subloss_mse_weighted',
        ]
        bottoms = np.zeros(len(self._history))
        for key in weighted_keys:
            vals = np.array([h.get(key, 0) for h in self._history])
            ax.fill_between(
                steps,
                bottoms,
                bottoms + vals,
                alpha=0.7,
                label=key.replace('subloss_', '').replace('_weighted', ''),
            )
            bottoms += vals
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss Contribution')
        ax.set_title('Subloss Breakdown (Stacked)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Prediction evolution
        ax = axes[1, 0]
        yhat_mean = [h.get("yhatdep_mean", float('nan')) for h in self._history]
        yhat_std = [h.get("yhatdep_std", float('nan')) for h in self._history]
        ax.fill_between(
            steps,
            np.array(yhat_mean) - np.array(yhat_std),
            np.array(yhat_mean) + np.array(yhat_std),
            alpha=0.3,
            label='±1 std',
        )
        ax.plot(steps, yhat_mean, 'b-', linewidth=1.5, label='mean')
        ax.set_xlabel('Step')
        ax.set_ylabel('Prediction Value')
        ax.set_title('Prediction Statistics Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # TU masking evolution
        ax = axes[1, 1]
        tu_enabled = [h.get("tu_enabled_count", 0) for h in self._history]
        tu_total = [h.get("tu_total_count", 1) for h in self._history]
        tu_pct = [100 * e / max(t, 1) for e, t in zip(tu_enabled, tu_total, strict=True)]
        ax.plot(steps, tu_pct, 'g-', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('% TU Enabled')
        ax.set_title('TU Pruning Progress')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._save_dir / "trajectories.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        if not self._history:
            return None
        return {
            "steps_tracked": len(self._history),
            "final_loss": self._history[-1].get("loss") if self._history else None,
        }

    def finalize(self):
        logger.info(f"DesignSublossLogger finalized with {len(self._history)} entries")
