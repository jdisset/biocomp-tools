## {{{                          --     imports     --

from biocomptools.toollib.loggers.base_metrics_logger import BaseMetricsLogger
from biocomptools.toollib.loggers.metrics_models import ReplicateMetrics, NetworkDataPairMetrics
from biocomptools.logging_config import get_logger
from biocomptools.run_training import TrainingProgram
from biocomp.datautils import DataManager
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
import time

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class DetailedTrainingStatsLogger(BaseMetricsLogger):
    """
    Logs detailed training statistics during training, computing RMSE and MSE
    for both overall training set and per-networkdatapair if y and yhat are available
    in the step history.

    Requirements:
    - training_config.keep_in_history must include both "y" and "yhat"

    Usage:
        logger = DetailedTrainingStatsLogger(periods=10)  # Log every 10 steps
        training_program.loggers.append(logger)
    """

    # Additional fields specific to this logger
    _data_manager: Optional[DataManager] = None

    def initialize(self, training_program):
        """Initialize from training program."""
        super().initialize(training_program)

        if training_program:
            self._data_manager = training_program._training_dman

    def _compute_metrics(self, step_data: Dict[str, Any]) -> List[ReplicateMetrics]:
        """
        Compute RMSE and MSE metrics for training data from step_data.

        Args:
            step_data: Dictionary containing 'y' and 'yhat' from step_history

        Returns:
            List of ReplicateMetrics objects, one per replicate
        """
        import numpy as np

        # Check if y and yhat are available
        if 'y' not in step_data or 'yhat' not in step_data:
            return []

        y = step_data['y']
        yhat = step_data['yhat']

        metrics_list = []
        n_replicates = y.shape[0]

        for rep_idx in range(n_replicates):
            # Flatten batches_per_step and batch_size dimensions
            y_rep = y[rep_idx].reshape(
                -1, y.shape[-1]
            )  # (batches_per_step * batch_size, n_outputs)
            yhat_rep = yhat[rep_idx].reshape(
                -1, yhat.shape[-1]
            )  # (batches_per_step * batch_size, n_outputs)

            # compute overall metrics
            mse = float(np.mean((y_rep - yhat_rep) ** 2))
            rmse = float(np.sqrt(mse))
            
            # Debug logging to trace data values
            logger.info(f"Rep {rep_idx}: DetailedTraining - y_rep stats: mean={y_rep.mean():.6f}, std={y_rep.std():.6f}, shape={y_rep.shape}")
            logger.info(f"Rep {rep_idx}: DetailedTraining - yhat_rep stats: mean={yhat_rep.mean():.6f}, std={yhat_rep.std():.6f}")
            if y_rep.shape[0] >= 5:
                logger.info(f"Rep {rep_idx}: DetailedTraining - First 5 y values: {y_rep[:5, 0]}")
                logger.info(f"Rep {rep_idx}: DetailedTraining - First 5 yhat values: {yhat_rep[:5, 0]}")

            # Debug: compare with sublosses RMSE if available
            if 'sublosses' in step_data and step_data['sublosses'] is not None:
                if 'rmse' in step_data['sublosses']:
                    # step_data['sublosses']['rmse'] has shape (n_replicates, batches_per_step)
                    # Each element is the RMSE computed for one batch for one replicate
                    subloss_rmse_values = step_data['sublosses']['rmse']

                    # Extract RMSE values for this specific replicate
                    replicate_rmse_values = subloss_rmse_values[
                        rep_idx
                    ]  # Shape: (batches_per_step,)
                    avg_subloss_rmse = float(np.mean(replicate_rmse_values))
                    logger.info(
                        f"Rep {rep_idx}: Training Logger RMSE = {rmse:.6f}, Avg Sublosses RMSE = {avg_subloss_rmse:.6f}, Ratio = {avg_subloss_rmse / rmse:.2f}"
                    )
                    logger.info(
                        f"Rep {rep_idx}: Full step_data shapes - y: {y.shape}, yhat: {yhat.shape}"
                    )
                    logger.info(f"Rep {rep_idx}: Sublosses RMSE shape: {subloss_rmse_values.shape}")
                    logger.info(
                        f"Rep {rep_idx}: This replicate's RMSE values: {replicate_rmse_values}"
                    )

                    # Now compute RMSE per batch manually to verify
                    batches_per_step = y.shape[
                        1
                    ]  # y has shape (n_replicates, n_batches_per_step, batch_size, n_outputs)

                    # Compute RMSE per batch like the loss function does
                    per_batch_rmses = []
                    for batch_idx in range(batches_per_step):
                        y_batch = y[rep_idx, batch_idx]  # Shape: [batch_size, n_outputs]
                        yhat_batch = yhat[rep_idx, batch_idx]  # Shape: [batch_size, n_outputs]
                        batch_mse = np.mean((y_batch - yhat_batch) ** 2)
                        batch_rmse = np.sqrt(batch_mse)
                        per_batch_rmses.append(batch_rmse)

                    avg_per_batch_rmse = np.mean(per_batch_rmses)
                    logger.info(
                        f"Rep {rep_idx}: Manually computed avg per-batch RMSE = {avg_per_batch_rmse:.6f}"
                    )
                    logger.info(
                        f"Rep {rep_idx}: Individual batch RMSEs: {[f'{r:.6f}' for r in per_batch_rmses]}"
                    )
                    logger.info(
                        f"Rep {rep_idx}: Manual vs Sublosses RMSE ratio = {avg_per_batch_rmse / avg_subloss_rmse:.6f}"
                    )

                    # The key question: why is avg_per_batch_rmse different from the training logger rmse?
                    # Training logger: sqrt(mean(all_errors^2))
                    # Per-batch average: mean(sqrt(mean(batch_errors^2)))
                    # These are mathematically different due to Jensen's inequality!

            # compute per-networkdatapair metrics if data manager available
            per_network_list = []
            if self._data_manager is not None:
                try:
                    per_network_list = self._compute_per_network_metrics(
                        y_rep, yhat_rep, self._data_manager, rep_idx
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to compute per-network metrics for replicate {rep_idx}: {e}"
                    )
                    logger.exception(e)

            # Extract sublosses for this replicate if available
            sublosses_data = None
            if 'sublosses' in step_data and step_data['sublosses'] is not None:
                try:
                    # sublosses is shape (n_replicates, n_batches_per_step) for each loss component
                    sublosses_rep = {}
                    for loss_name, loss_values in step_data['sublosses'].items():
                        if hasattr(loss_values, 'shape') and len(loss_values.shape) >= 1:
                            # Average over batches for this replicate
                            avg_loss = float(np.mean(loss_values[rep_idx]))
                            sublosses_rep[loss_name] = avg_loss
                    sublosses_data = sublosses_rep if sublosses_rep else None
                except Exception as e:
                    logger.warning(f"Failed to extract sublosses for replicate {rep_idx}: {e}")

            replicate_metrics = ReplicateMetrics(
                replicate=rep_idx,
                overall_RMSE=rmse,
                overall_MSE=mse,
                n_samples=int(y_rep.shape[0]),
                per_networkdatapair=per_network_list,
                sublosses=sublosses_data,
            )

            metrics_list.append(replicate_metrics)

        return metrics_list

    def _compute_per_network_metrics(
        self, y: np.ndarray, yhat: np.ndarray, data_manager, replicate: int
    ) -> List[NetworkDataPairMetrics]:
        """Compute RMSE and MSE for each network-datapair"""
        per_network_metrics = []

        try:
            networks = data_manager.get_networks()
            n_outputs = [net.get_nb_outputs() for net in networks]
            slice_at_y = np.cumsum(n_outputs)[:-1]
            per_net_y = np.split(y, slice_at_y, axis=1)
            per_net_yhat = np.split(yhat, slice_at_y, axis=1)

            # Compute metrics for each network
            for i, (net_y, net_yhat, network) in enumerate(zip(per_net_y, per_net_yhat, networks)):
                if net_y.size > 0:  # Only compute if there's data
                    mse = np.mean((net_y - net_yhat) ** 2)
                    rmse = np.sqrt(mse)
                    network_name = network.name
                    
                    # Debug logging for per-network
                    if i == 0:  # Log first network
                        logger.info(f"DetailedTraining Per-Network {network_name} - net_y stats: mean={net_y.mean():.6f}, std={net_y.std():.6f}, shape={net_y.shape}")
                        logger.info(f"DetailedTraining Per-Network {network_name} - net_yhat stats: mean={net_yhat.mean():.6f}, std={net_yhat.std():.6f}")
                        if net_y.shape[0] >= 5:
                            logger.info(f"DetailedTraining Per-Network {network_name} - First 5 y: {net_y[:5, 0]}")
                            logger.info(f"DetailedTraining Per-Network {network_name} - First 5 yhat: {net_yhat[:5, 0]}")
                    networkdatapair = {
                        'network_name': network_name,
                        'network_hash': getattr(network, 'hash', ''),
                        'n_inputs': network.get_nb_inputs(),
                        'n_outputs': network.get_nb_outputs(),
                    }
                    per_network_metrics.append(
                        NetworkDataPairMetrics(
                            network_name=network_name,
                            networkdatapair=networkdatapair,
                            RMSE=float(rmse),
                            MSE=float(mse),
                            n_samples=int(net_y.shape[0]),
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to compute per-network metrics: {e}")
            logger.exception(e)
            raise e

        return per_network_metrics

    def _print_metrics(self, step: int, metrics: List[ReplicateMetrics]):
        """Print training statistics in a formatted table."""
        if not metrics:
            return

        table = Table(title=f"Training Statistics - Step {step}")
        table.add_column("Replicate", style="cyan", justify="right")
        table.add_column("Overall RMSE", style="green", justify="right")
        table.add_column("Overall MSE", style="yellow", justify="right")
        table.add_column("# Networks", style="blue", justify="right")

        for replicate_metrics in metrics:
            n_networks = len(replicate_metrics.per_networkdatapair)
            table.add_row(
                str(replicate_metrics.replicate),
                f"{replicate_metrics.overall_RMSE:.6f}",
                f"{replicate_metrics.overall_MSE:.6f}",
                str(n_networks) if n_networks > 0 else "N/A",
            )

        self._console.print(table)

        # also print per-network details for first replicate if available
        if metrics and metrics[0].per_networkdatapair:
            per_net = metrics[0].per_networkdatapair
            net_table = Table(title=f"Per-Network Statistics (Replicate 0) - Step {step}")
            net_table.add_column("Network", style="cyan")
            net_table.add_column("RMSE", style="green", justify="right")
            net_table.add_column("MSE", style="yellow", justify="right")
            net_table.add_column("Samples", style="blue", justify="right")

            for net_metrics in per_net:
                net_table.add_row(
                    net_metrics.network_name,
                    f"{net_metrics.RMSE:.6f}",
                    f"{net_metrics.MSE:.6f}",
                    str(net_metrics.n_samples),
                )

            self._console.print(net_table)

        # Print sublosses if available
        if metrics and metrics[0].sublosses:
            sublosses_table = Table(title=f"Sublosses - Step {step}")
            sublosses_table.add_column("Replicate", style="cyan", justify="right")

            # Add columns for each subloss type
            loss_names = list(metrics[0].sublosses.keys())
            for loss_name in loss_names:
                sublosses_table.add_column(loss_name, style="green", justify="right")

            for replicate_metrics in metrics:
                if replicate_metrics.sublosses:
                    row = [str(replicate_metrics.replicate)]
                    for loss_name in loss_names:
                        value = replicate_metrics.sublosses.get(loss_name, 0.0)
                        row.append(f"{value:.6f}")
                    sublosses_table.add_row(*row)

            self._console.print(sublosses_table)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        self.initialize(training_program)

        def log_training_stats(step, training_config, step_history=None, **kwargs):
            if step_history is None:
                return

            if 'y' not in step_history or 'yhat' not in step_history:
                if step > 0:
                    logger.debug(
                        f"DetailedTrainingStatsLogger {self.name}: y/yhat not available at step {step}"
                    )
                return
            logger.info(
                f"DetailedTrainingStatsLogger {self.name}: Computing training statistics at step {step}"
            )

            # Use the base class method to handle metrics computation and plotting
            training_loss = step_history.get('loss')
            self._log_metrics_step(step, step_history, training_loss)

        return [(self.periods, log_training_stats)]

    def finalize(self):
        """Create video from training metrics plots and call parent finalize."""
        # Call parent finalize first
        super().finalize()

        if self._plot_save_dir is None:
            return

        from biocomptools.toollib.video_utils import create_video_from_plots

        logger.info("DetailedTrainingStatsLogger: Creating video from training plots...")

        video_path = self._plot_save_dir / "training_metrics_video.mp4"
        video_created = create_video_from_plots(
            plot_dir=self._plot_save_dir,
            output_path=video_path,
            plot_pattern="*step_*.png",  # Catch all step-based plots
        )

        if video_created:
            logger.info(
                f"DetailedTrainingStatsLogger: Created training metrics video: {video_path}"
            )
        else:
            logger.debug(
                "DetailedTrainingStatsLogger: No video created (insufficient plots or errors)"
            )
