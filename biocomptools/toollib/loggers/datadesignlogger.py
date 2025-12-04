"""
Logger for data-driven design optimization.

Tracks metrics specific to design runs where experimental data is used as targets,
enabling comparison between model prediction quality and optimized design quality.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any
from pydantic import Field, ConfigDict

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DataDesignLogger(Logger):
    """
    Logger for data-driven design optimization runs.

    Tracks:
    - Design loss over time (per target, per replicate, per network)
    - Comparison with baseline model prediction loss
    - Summary statistics and plots
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    output_dir: Optional[str] = None
    baseline_r2: Optional[float] = None  # baseline model R² on this data
    baseline_rmse: Optional[float] = None  # baseline model RMSE
    baseline_loss: Optional[float] = None  # baseline loss (same metric as design loss)
    top_k: int = 5  # number of top designs to track
    save_interval: int = 100  # save plots every N steps

    # Internal state (not validated)
    _loss_history: List[Tuple[int, np.ndarray]] = []
    _best_loss_per_target: Dict[int, float] = {}
    _best_config_per_target: Dict[int, Tuple[int, int]] = {}  # target_id -> (rep_id, net_id)
    _final_params: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss_history = []
        self._best_loss_per_target = {}
        self._best_config_per_target = {}
        self._final_params = None
        self._step_count = 0

    def _update_history(self, step: int, all_losses: np.ndarray):
        """Update loss history with new step data."""
        all_losses = np.asarray(all_losses)

        # Handle different shapes
        if all_losses.ndim == 2:
            # (n_targets, n_networks)
            all_losses = all_losses[None, :, :]  # add replicate dim
        elif all_losses.ndim == 4:
            # (n_replicates, n_batches, n_targets, n_networks) -> average over batches
            all_losses = np.mean(all_losses, axis=1)

        # Shape should now be (n_replicates, n_targets, n_networks)
        self._loss_history.append((step, all_losses.copy()))

        # Update best configurations per target
        n_replicates, n_targets, n_networks = all_losses.shape
        for target_id in range(n_targets):
            target_losses = all_losses[:, target_id, :]  # (n_replicates, n_networks)
            min_loss = float(np.min(target_losses))

            if target_id not in self._best_loss_per_target or min_loss < self._best_loss_per_target[target_id]:
                self._best_loss_per_target[target_id] = min_loss
                flat_idx = np.argmin(target_losses)
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks
                self._best_config_per_target[target_id] = (int(rep_id), int(net_id))

    def _plot_loss_curves(self, save_path: Path):
        """Plot loss curves over training."""
        if len(self._loss_history) < 2:
            return

        steps = [s for s, _ in self._loss_history]
        # Get shape from first entry
        _, first_losses = self._loss_history[0]
        n_replicates, n_targets, n_networks = first_losses.shape

        # Create figure with subplots for each target
        n_cols = min(2, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

        colors = plt.cm.tab10(np.linspace(0, 1, min(10, self.top_k)))

        for target_id in range(n_targets):
            ax = axes[target_id // n_cols, target_id % n_cols]

            # Get final step losses for ranking
            _, final_losses = self._loss_history[-1]
            target_final = final_losses[:, target_id, :]  # (n_replicates, n_networks)
            flat_final = target_final.reshape(-1)
            top_k_indices = np.argsort(flat_final)[:self.top_k]

            for rank, flat_idx in enumerate(top_k_indices):
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks

                # Extract history for this (rep, net) pair
                history = [losses[rep_id, target_id, net_id] for _, losses in self._loss_history]

                ax.semilogy(
                    steps, history,
                    color=colors[rank],
                    label=f'R{rep_id}N{net_id} (final={flat_final[flat_idx]:.4f})',
                    alpha=0.8,
                    linewidth=1.5
                )

            # Add baseline if provided
            if self.baseline_loss is not None:
                ax.axhline(
                    y=self.baseline_loss, color='red', linestyle='--',
                    label=f'Baseline: {self.baseline_loss:.4f}', linewidth=2
                )

            ax.set_xlabel('Step')
            ax.set_ylabel('Loss (log scale)')
            ax.set_title(f'Target {target_id}: Top {self.top_k} Designs')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_targets, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved loss curves to {save_path}")

    def _plot_comparison_summary(self, save_path: Path):
        """Plot summary comparing baseline vs designed solution."""
        if not self._loss_history:
            return

        _, final_losses = self._loss_history[-1]
        n_replicates, n_targets, n_networks = final_losses.shape

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Loss improvement over time (mean across all)
        ax1 = axes[0]
        steps = [s for s, _ in self._loss_history]
        mean_losses = [np.mean(losses) for _, losses in self._loss_history]
        min_losses = [np.min(losses) for _, losses in self._loss_history]

        ax1.semilogy(steps, mean_losses, 'b-', label='Mean loss', linewidth=2)
        ax1.semilogy(steps, min_losses, 'g-', label='Best loss', linewidth=2)

        if self.baseline_loss is not None:
            ax1.axhline(y=self.baseline_loss, color='red', linestyle='--',
                       label=f'Baseline: {self.baseline_loss:.4f}', linewidth=2)

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('Design Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Final loss distribution
        ax2 = axes[1]
        all_final = final_losses.reshape(-1)
        ax2.hist(all_final, bins=30, alpha=0.7, color='blue', edgecolor='black')

        if self.baseline_loss is not None:
            ax2.axvline(x=self.baseline_loss, color='red', linestyle='--',
                       label=f'Baseline: {self.baseline_loss:.4f}', linewidth=2)

        best_final = np.min(all_final)
        ax2.axvline(x=best_final, color='green', linestyle='-',
                   label=f'Best: {best_final:.4f}', linewidth=2)

        ax2.set_xlabel('Loss')
        ax2.set_ylabel('Count')
        ax2.set_title('Final Loss Distribution')
        ax2.legend()

        # 3. Improvement metrics bar chart
        ax3 = axes[2]
        metrics = []
        labels = []

        if self.baseline_loss is not None and self.baseline_loss > 0:
            improvement_pct = (1 - best_final / self.baseline_loss) * 100
            metrics.append(improvement_pct)
            labels.append('Loss Improvement (%)')

        if self.baseline_r2 is not None:
            metrics.append(self.baseline_r2 * 100)
            labels.append('Baseline R² (%)')

        initial_loss = np.mean([losses for _, losses in self._loss_history[:1]][0]) if self._loss_history else 0
        if initial_loss > 0:
            optim_improvement = (1 - best_final / initial_loss) * 100
            metrics.append(optim_improvement)
            labels.append('Optim Improvement (%)')

        if metrics:
            bars = ax3.bar(labels, metrics, color=['green', 'blue', 'orange'][:len(metrics)])
            ax3.set_ylabel('Percentage')
            ax3.set_title('Summary Metrics')

            for bar, val in zip(bars, metrics):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No baseline provided', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Summary Metrics (baseline required)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved comparison summary to {save_path}")

    def _plot_final_summary(self, save_path: Path):
        """Plot comprehensive final summary."""
        if not self._loss_history:
            return

        _, final_losses = self._loss_history[-1]
        _, initial_losses = self._loss_history[0]
        n_replicates, n_targets, n_networks = final_losses.shape

        fig = plt.figure(figsize=(16, 10))

        # Top row: per-target best losses
        ax1 = fig.add_subplot(2, 2, 1)
        target_ids = list(range(n_targets))
        best_losses = [self._best_loss_per_target.get(t, np.nan) for t in target_ids]
        initial_best = [np.min(initial_losses[:, t, :]) for t in target_ids]

        x = np.arange(len(target_ids))
        width = 0.35

        ax1.bar(x - width/2, initial_best, width, label='Initial', color='lightcoral')
        ax1.bar(x + width/2, best_losses, width, label='Final', color='lightgreen')

        if self.baseline_loss is not None:
            ax1.axhline(y=self.baseline_loss, color='red', linestyle='--',
                       label=f'Baseline', linewidth=2)

        ax1.set_xlabel('Target')
        ax1.set_ylabel('Loss')
        ax1.set_title('Best Loss per Target: Initial vs Final')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'T{t}' for t in target_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Top right: Heatmap of final losses
        ax2 = fig.add_subplot(2, 2, 2)
        # Average over replicates for visualization
        avg_final = np.mean(final_losses, axis=0)  # (n_targets, n_networks)
        im = ax2.imshow(avg_final, aspect='auto', cmap='viridis_r')
        ax2.set_xlabel('Network')
        ax2.set_ylabel('Target')
        ax2.set_title('Average Final Loss (across replicates)')
        plt.colorbar(im, ax=ax2)

        # Bottom left: Loss trajectory
        ax3 = fig.add_subplot(2, 2, 3)
        steps = [s for s, _ in self._loss_history]
        mean_losses = [np.mean(losses) for _, losses in self._loss_history]
        std_losses = [np.std(losses) for _, losses in self._loss_history]

        ax3.fill_between(
            steps,
            np.array(mean_losses) - np.array(std_losses),
            np.array(mean_losses) + np.array(std_losses),
            alpha=0.3, color='blue'
        )
        ax3.plot(steps, mean_losses, 'b-', linewidth=2, label='Mean ± Std')

        if self.baseline_loss is not None:
            ax3.axhline(y=self.baseline_loss, color='red', linestyle='--',
                       label=f'Baseline: {self.baseline_loss:.4f}', linewidth=2)

        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Mean Loss Trajectory')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom right: Text summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        summary_text = "DESIGN OPTIMIZATION SUMMARY\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += f"Total Steps: {self._step_count}\n"
        summary_text += f"Replicates: {n_replicates}\n"
        summary_text += f"Targets: {n_targets}\n"
        summary_text += f"Networks: {n_networks}\n\n"

        summary_text += f"Initial Mean Loss: {np.mean(initial_losses):.6f}\n"
        summary_text += f"Final Mean Loss: {np.mean(final_losses):.6f}\n"
        summary_text += f"Final Best Loss: {np.min(final_losses):.6f}\n\n"

        if self.baseline_loss is not None:
            improvement = (1 - np.min(final_losses) / self.baseline_loss) * 100
            summary_text += f"Baseline Loss: {self.baseline_loss:.6f}\n"
            summary_text += f"Improvement vs Baseline: {improvement:.1f}%\n\n"

        if self.baseline_r2 is not None:
            summary_text += f"Baseline R²: {self.baseline_r2:.4f}\n"
        if self.baseline_rmse is not None:
            summary_text += f"Baseline RMSE: {self.baseline_rmse:.6f}\n"

        summary_text += "\nBest Configuration per Target:\n"
        for t in target_ids:
            if t in self._best_config_per_target:
                rep, net = self._best_config_per_target[t]
                loss = self._best_loss_per_target.get(t, np.nan)
                summary_text += f"  Target {t}: Rep={rep}, Net={net}, Loss={loss:.6f}\n"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved final summary to {save_path}")

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
        """Return callbacks for the training loop."""

        def periodic_callback(
            step, training_config, step_history=None, stack=None, **kwargs
        ):
            self._step_count = step

            if step_history is None:
                return

            all_losses = step_history.get('all_losses')
            if all_losses is None:
                return

            self._update_history(step, all_losses)
            self._final_params = step_history.get('latest_params')

            # Periodically save plots
            if self.output_dir and step % self.save_interval == 0:
                output_path = Path(self.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self._plot_loss_curves(output_path / f'loss_curves_step{step:06d}.png')

        def end_callback(
            step, training_config, step_history=None, stack=None, **kwargs
        ):
            self._step_count = step

            if step_history is not None:
                all_losses = step_history.get('all_losses')
                if all_losses is not None:
                    self._update_history(step, all_losses)
                self._final_params = step_history.get('latest_params')

            if self.output_dir:
                output_path = Path(self.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                self._plot_loss_curves(output_path / 'final_loss_curves.png')
                self._plot_comparison_summary(output_path / 'comparison_summary.png')
                self._plot_final_summary(output_path / 'final_summary.png')

        callbacks = []
        if isinstance(self.periods, int):
            callbacks.append((self.periods, periodic_callback))
        else:
            for period in self.periods:
                callbacks.append((period, periodic_callback))

        # End callback (period=-1)
        callbacks.append((-1, end_callback))

        return callbacks

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return current metrics."""
        if not self._loss_history:
            return None

        _, final_losses = self._loss_history[-1]

        metrics = {
            'final_mean_loss': float(np.mean(final_losses)),
            'final_best_loss': float(np.min(final_losses)),
            'total_steps': self._step_count,
            'best_configs': dict(self._best_config_per_target),
            'best_losses': dict(self._best_loss_per_target),
        }

        if self.baseline_loss is not None:
            metrics['baseline_loss'] = self.baseline_loss
            metrics['improvement_vs_baseline'] = (1 - np.min(final_losses) / self.baseline_loss) * 100

        if self.baseline_r2 is not None:
            metrics['baseline_r2'] = self.baseline_r2

        if self.baseline_rmse is not None:
            metrics['baseline_rmse'] = self.baseline_rmse

        return metrics

    def finalize(self):
        """Cleanup after training ends."""
        pass
