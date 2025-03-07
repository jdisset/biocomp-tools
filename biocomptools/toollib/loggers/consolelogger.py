## {{{                          --     imports     --

from dracon.deferred import DeferredNode
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable, Union, Annotated, Literal, TypeVar
from biocomptools.toollib.loggers.logger import Logger

from biocomptools.logging_config import get_logger
logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class EnhancedConsoleLogger(Logger):
    """Logs and visualizes the training loss to console with historical tracking"""

    plot_height: int = 22
    plot_width: int = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: Dict[int, Dict] = {}
        self._best_mean_loss = float('inf')

    def _print_step_stats(self, step: int, losses: np.ndarray):
        """Print detailed statistics for current step"""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        losses = np.asarray(losses)
        table = Table(title=f"Step {step} Statistics")
        table.add_column("Replicate", justify="right", style="cyan")
        table.add_column("Avg Loss", justify="right", style="green")
        table.add_column("Min Loss", justify="right", style="blue")
        table.add_column("Max Loss", justify="right", style="red")

        for i, loss in enumerate(losses):
            avg_loss = np.mean(loss)
            min_loss = np.min(loss)
            max_loss = np.max(loss)
            table.add_row(str(i), f"{avg_loss:.4f}", f"{min_loss:.4f}", f"{max_loss:.4f}")

        console.print(table)

    def _update_history(self, step: int, losses: np.ndarray):
        """Update history with new loss values"""
        losses = np.asarray(losses)
        mean_losses = np.mean(losses, axis=1)
        best_mean = float(np.mean(mean_losses))

        self._history[step] = {
            'losses': losses,
            'mean_per_replicate': mean_losses,
            'best_mean': best_mean,
        }

        if best_mean < self._best_mean_loss:
            self._best_mean_loss = best_mean

    def _plot_loss_history(self):
        """Plot the loss history in the console using log scale"""
        import plotext as plt

        if not self._history:
            return

        steps = np.cumsum([v['losses'].shape[1] for _, v in self._history.items()])
        best_means = [float(self._history[k]['best_mean']) for k in self._history]

        plt.clf()
        plt.theme("matrix")
        plt.plot_size(self.plot_width, self.plot_height)
        plt.plot(steps, best_means, marker="braille")
        plt.yscale("log")
        plt.title(f"Training Loss (current best: {self._best_mean_loss:.4f})")
        plt.xlabel("Batch")
        plt.ylabel("Loss (log scale)")
        plt.show()

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_loss(step, training_config, step_history=None, **kwargs):
            if step_history is not None:
                losses = step_history.get('loss')
                if losses is not None:
                    self._update_history(step, losses)
                    self._print_step_stats(step, losses)
                    self._plot_loss_history()

        return [(self.periods, log_loss)]


class ConsoleLogger(Logger):
    """Logs the training loss to console"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_loss(step, training_config, step_history=None, **kwargs):
            # we will show the avg, min and max loss of the current step for each replicate:
            if step_history is not None:
                losses = step_history.get('loss')
                # shape is (n_reps, n_batches)
                if losses is not None:
                    for i, loss in enumerate(losses):
                        avg_loss = np.mean(loss)
                        min_loss = np.min(loss)
                        max_loss = np.max(loss)
                        logger.debug(
                            f"Step {step}, Replicate {i}: Avg loss: {avg_loss:.4f}, Min loss: {min_loss:.4f}, Max loss: {max_loss:.4f}"
                        )

        return [(self.periods, log_loss)]
