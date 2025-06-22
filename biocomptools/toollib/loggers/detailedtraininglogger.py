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
            y_rep = y[rep_idx].reshape(-1, y.shape[-1])  # (batches_per_step * batch_size, n_outputs)
            yhat_rep = yhat[rep_idx].reshape(-1, yhat.shape[-1])  # (batches_per_step * batch_size, n_outputs)

            # compute overall metrics
            mse = float(np.mean((y_rep - yhat_rep) ** 2))
            rmse = float(np.sqrt(mse))

            # compute per-networkdatapair metrics if data manager available
            per_network_list = []
            if self._data_manager is not None:
                try:
                    per_network_list = self._compute_per_network_metrics(y_rep, yhat_rep)
                except Exception as e:
                    logger.warning(f"Failed to compute per-network metrics: {e}")

            replicate_metrics = ReplicateMetrics(
                replicate=rep_idx,
                overall_RMSE=rmse,
                overall_MSE=mse,
                n_samples=int(y_rep.shape[0]),
                per_networkdatapair=per_network_list
            )
            
            metrics_list.append(replicate_metrics)

        return metrics_list

    def _compute_per_network_metrics(self, y_batch, yhat_batch) -> List[NetworkDataPairMetrics]:
        """
        Compute per-network metrics by splitting batch according to data manager's network structure.

        Args:
            y_batch: Ground truth for one replicate, shape (total_batch_size, n_outputs)
            yhat_batch: Predictions for one replicate, shape (total_batch_size, n_outputs)

        Returns:
            List of NetworkDataPairMetrics objects
        """
        import numpy as np

        per_network_metrics = []

        # get network information from data manager
        networks = self._data_manager.get_networks()
        network_sizes = [len(self._data_manager.get_network_data(net)) for net in networks]
        total_network_data = sum(network_sizes)

        if total_network_data == 0:
            return per_network_metrics

        # calculate how batch samples map to networks (assuming proportional sampling)
        batch_size = y_batch.shape[0]
        network_batch_sizes = []
        cumulative_size = 0

        for i, net_size in enumerate(network_sizes):
            if i == len(network_sizes) - 1:
                # last network gets remaining samples
                net_batch_size = batch_size - cumulative_size
            else:
                net_batch_size = int((net_size / total_network_data) * batch_size)
            network_batch_sizes.append(net_batch_size)
            cumulative_size += net_batch_size

        # split batch and compute metrics per network
        start_idx = 0
        for net_idx, (network, net_batch_size) in enumerate(zip(networks, network_batch_sizes)):
            if net_batch_size == 0:
                continue

            end_idx = start_idx + net_batch_size
            y_net = y_batch[start_idx:end_idx]
            yhat_net = yhat_batch[start_idx:end_idx]

            # compute metrics for this network
            mse = float(np.mean((y_net - yhat_net) ** 2))
            rmse = float(np.sqrt(mse))

            # get networkdatapair info
            networkdatapair_info = {
                'network_name': network.name,
                'datafile_path': getattr(network, 'datafile_path', 'unknown'),
            }

            net_metrics = NetworkDataPairMetrics(
                network_name=network.name,
                networkdatapair=networkdatapair_info,
                MSE=mse,
                RMSE=rmse,
                n_samples=int(net_batch_size)
            )

            per_network_metrics.append(net_metrics)
            start_idx = end_idx

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

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        self.initialize(training_program)

        def log_training_stats(step, training_config, step_history=None, **kwargs):
            if step_history is None:
                return

            # check if y and yhat are available in step history
            if 'y' not in step_history or 'yhat' not in step_history:
                if step == 0:
                    logger.debug(
                        f"DetailedTrainingStatsLogger {self.name}: y/yhat not available at step 0 (expected)"
                    )
                else:
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
