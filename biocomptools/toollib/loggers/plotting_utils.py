## {{{                          --     imports     --

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from biocomptools.trainutils import ffill
from biocomptools.toollib.loggers.paramgradlogger import get_plot_rows_and_columns
from biocomptools.toollib.loggers.metrics_models import LoggerMetricsHistory
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class MetricsPlotter:
    """Shared plotting functionality for validation and training metrics."""
    
    def __init__(self, logger_name: str, plot_save_dir: Optional[Path] = None, 
                 plot_dpi: int = 200, plot_training_losses: bool = False):
        self.logger_name = logger_name
        self.plot_save_dir = plot_save_dir
        self.plot_dpi = plot_dpi
        self.plot_training_losses = plot_training_losses
        
        if self.plot_save_dir:
            self.plot_save_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_metrics_history(self, metrics_history: LoggerMetricsHistory, current_step: int) -> None:
        """Plot complete metrics history."""
        if not metrics_history.history or self.plot_save_dir is None:
            return
        
        import time
        plot_start_time = time.time()
        
        # Extract data for plotting
        steps = [h.step for h in metrics_history.history]
        
        # Determine number of replicates
        n_replicates = len(metrics_history.history[0].metrics) if metrics_history.history else 1
        
        # Get all network names from per_networkdatapair data
        all_net_names = set()
        for step_metrics in metrics_history.history:
            for rep_metrics in step_metrics.metrics:
                for net_metrics in rep_metrics.per_networkdatapair:
                    all_net_names.add(net_metrics.network_name)
        
        all_net_names = sorted(all_net_names)
        n_nets = len(all_net_names)
        
        # Build loss data array: (n_nets + 1, n_replicates, n_steps)
        loss_data = np.full((n_nets + 1, n_replicates, len(steps)), np.nan)
        
        for s_idx, step_metrics in enumerate(metrics_history.history):
            for r_idx, rep_metrics in enumerate(step_metrics.metrics):
                # Overall RMSE
                loss_data[0, r_idx, s_idx] = rep_metrics.overall_RMSE
                
                # Per-network RMSE
                net_rmse_dict = {net.network_name: net.RMSE for net in rep_metrics.per_networkdatapair}
                for n_idx, net_name in enumerate(all_net_names):
                    if net_name in net_rmse_dict:
                        loss_data[n_idx + 1, r_idx, s_idx] = net_rmse_dict[net_name]
        
        # Create plot layout
        n_rows, n_cols = get_plot_rows_and_columns(n_nets, ideal_ratio=1.0)
        fig = plt.figure(figsize=(5 * n_cols, 2.5 * (n_rows + 2)), dpi=self.plot_dpi)
        gs = fig.add_gridspec(n_rows + 2, n_cols, hspace=0.5, wspace=0.5)
        fig.suptitle(f'{self.logger_name.title()} Metrics History - Step {current_step}', fontsize=14)
        
        # Main plot (overall RMSE)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        self._plot_single_metric(
            ax_main,
            loss_data[0],
            steps,
            'Overall Average RMSE',
            n_replicates,
            is_main=True,
            training_history=metrics_history.history,
        )
        
        # Individual network plots
        axes_flat = [fig.add_subplot(gs[r + 2, c]) for r in range(n_rows) for c in range(n_cols)]
        for i, ax in enumerate(axes_flat):
            if i >= n_nets:
                ax.set_visible(False)
                continue
            self._plot_single_metric(ax, loss_data[i + 1], steps, all_net_names[i], n_replicates)
        
        # Save plot
        plot_filename = f"{self.logger_name}_{current_step:05d}.png"
        fig.savefig(self.plot_save_dir / plot_filename)
        plt.close(fig)
        
        plot_time = time.time() - plot_start_time
        logger.info(f"MetricsPlotter: Plot generation completed in {plot_time:.2f} seconds")
    
    def _plot_single_metric(self, ax, data, steps, title, n_replicates, is_main=False, training_history=None):
        """Plot a single metric with optional training loss overlay."""
        filled_data = ffill(data)
        cmap = plt.cm.get_cmap('tab10')
        plt.minorticks_off()
        ax.minorticks_off()
        
        # Plot validation/training metric lines (colored)
        for i in range(n_replicates):
            ax.plot(
                steps,
                data[i, :],
                color=cmap(i % 10),
                linestyle='-',
                linewidth=1.5 if is_main else 0.5,
                alpha=1.0,
                label=f'Rep {i}' if is_main else None,
            )
        
        # Add training loss line for main plot (grey)
        if is_main and training_history is not None and self.plot_training_losses:
            training_losses = []
            train_steps = []
            for h in training_history:
                if h.training_loss is not None:
                    train_steps.append(h.step)
                    # Average training loss across replicates
                    loss_array = h.training_loss
                    if hasattr(loss_array, 'mean'):
                        avg_loss = float(loss_array.mean())
                    else:
                        avg_loss = float(np.mean(loss_array))
                    training_losses.append(avg_loss)
            
            if training_losses:
                ax.plot(
                    train_steps,
                    training_losses,
                    color='grey',
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.7,
                    label='Training Loss',
                )
        
        ax.set_title(title, fontsize=11 if is_main else 7)
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('RMSE (log scale)')
        
        if is_main and (n_replicates > 1 or (training_history is not None and self.plot_training_losses)):
            ax.legend(fontsize='small')
    
    def create_final_plot(self, metrics_history: LoggerMetricsHistory) -> None:
        """Create a single comprehensive plot at the end of training."""
        if not metrics_history.history or self.plot_save_dir is None:
            return
        
        final_step = metrics_history.history[-1].step
        self.plot_metrics_history(metrics_history, final_step)
        
        # Also create a special "final" plot
        final_plot_name = f"{self.logger_name}_final.png"
        if (self.plot_save_dir / f"{self.logger_name}_{final_step:05d}.png").exists():
            import shutil
            shutil.copy2(
                self.plot_save_dir / f"{self.logger_name}_{final_step:05d}.png",
                self.plot_save_dir / final_plot_name
            )
        
        logger.info(f"MetricsPlotter: Final plot saved as {final_plot_name}")