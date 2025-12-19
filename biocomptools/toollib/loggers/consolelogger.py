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
    top_k_designs: int = 5  # number of top designs to show per target in design mode
    async_ok: bool = False  # fast console logger doesn't need async

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: Dict[int, Dict] = {}
        self._replicate_histories: Dict[int, List] = {}  # replicate_id -> list of all losses
        self._global_step_counter = 0
        self._best_mean_loss = float('inf')
        self._design_mode = False
        self._design_losses_history = []  # list of (n_replicates, n_targets, n_networks) arrays per batch
        self._latest_step_avg_losses = None  # stores step-averaged losses for consistent ranking

    def _print_step_stats(self, step: int, losses: np.ndarray, all_losses: np.ndarray = None):
        """Print detailed statistics for current step"""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        losses = np.asarray(losses)

        # check if we're in design mode by looking for all_losses
        if all_losses is not None:
            self._print_design_stats(step, all_losses, console, self.top_k_designs)
        else:
            # original training mode stats
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

    def _print_design_stats(self, step: int, all_losses: np.ndarray, console, top_k: int = 5):
        """Print design mode statistics showing top-k losses per target (based on step average)"""
        from rich.table import Table

        all_losses = np.asarray(all_losses)

        # handle different possible shapes of all_losses
        if all_losses.ndim == 2:
            # single replicate case: (n_targets, n_networks)
            n_targets, n_networks = all_losses.shape
            n_replicates = 1
            all_losses = all_losses[None, :, :]  # add replicate dimension
        elif all_losses.ndim == 3:
            # (n_replicates, n_targets, n_networks)
            n_replicates, n_targets, n_networks = all_losses.shape
        elif all_losses.ndim == 4:
            # (n_replicates, n_batches_per_step, n_targets, n_networks)
            n_replicates, n_batches, n_targets, n_networks = all_losses.shape
            # average over batches to get (n_replicates, n_targets, n_networks) for ranking
            all_losses = np.mean(all_losses, axis=1)
        else:
            console.print(f"[red]Unexpected all_losses shape: {all_losses.shape}[/red]")
            return

        # Store the step-averaged losses for consistent ranking
        self._latest_step_avg_losses = all_losses

        table = Table(title=f"Step {step} Design Statistics (Top {top_k} per Target)")
        table.add_column("Target", justify="right", style="cyan")
        table.add_column("Rep", justify="right", style="yellow")
        table.add_column("Net", justify="right", style="magenta")
        table.add_column("Loss (avg)", justify="right", style="green")

        for target_id in range(n_targets):
            # get losses for this target across all replicates and networks
            target_losses = all_losses[:, target_id, :]  # shape: (n_replicates, n_networks)
            flat_losses = target_losses.reshape(-1)  # flatten to (n_replicates * n_networks)

            # get top-k indices (lowest losses)
            top_k_actual = min(top_k, len(flat_losses))
            top_indices = np.argsort(flat_losses)[:top_k_actual]

            for rank, flat_idx in enumerate(top_indices):
                # convert flat index back to (replicate, network)
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks
                loss_val = flat_losses[flat_idx]

                # style the best result differently
                loss_style = "bright_green" if rank == 0 else "green"
                table.add_row(
                    str(target_id) if rank == 0 else "",  # only show target_id for first row
                    str(rep_id),
                    str(net_id),
                    f"[{loss_style}]{loss_val:.4f}[/{loss_style}]",
                )

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

        # Update per-replicate full history
        for replicate_id, replicate_losses in enumerate(losses):
            if replicate_id not in self._replicate_histories:
                self._replicate_histories[replicate_id] = []
            self._replicate_histories[replicate_id].extend(replicate_losses)

        if best_mean < self._best_mean_loss:
            self._best_mean_loss = best_mean

    def _update_design_history(self, all_losses: np.ndarray, step: int):
        """Update design mode loss history - track step-averaged losses"""
        all_losses = np.asarray(all_losses)

        # handle different shapes to get (n_replicates, n_targets, n_networks)
        if all_losses.ndim == 2:
            # (n_targets, n_networks) - add replicate dimension
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 3:
            # (n_replicates, n_targets, n_networks) - already correct
            pass
        elif all_losses.ndim == 4:
            # (n_replicates, n_batches, n_targets, n_networks) - average over batches
            all_losses = np.mean(all_losses, axis=1)
        else:
            return  # unexpected shape

        # store step-averaged losses with step number
        self._design_losses_history.append((step, all_losses))

    def _plot_design_loss_history(self):
        """Plot design mode loss history - one plot per target showing top-k (replicate, network) pairs"""
        import plotext as plt

        if not self._design_losses_history:
            return

        # Use the latest step-averaged losses for ranking
        if hasattr(self, '_latest_step_avg_losses'):
            ranking_losses = self._latest_step_avg_losses
        else:
            # fallback: use the latest step losses
            _, ranking_losses = self._design_losses_history[-1]
        
        # shape should be (n_replicates, n_targets, n_networks)
        n_replicates, n_targets, n_networks = ranking_losses.shape

        # determine plot layout
        n_cols = min(2, n_targets)  # max 2 columns
        n_rows = (n_targets + n_cols - 1) // n_cols  # ceiling division

        # create subplots for each target
        plt.clf()
        plt.theme("matrix")
        plt.subplots(n_rows, n_cols)
        plt.plot_size(self.plot_width, self.plot_height * n_rows)

        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'orange']

        for target_id in range(n_targets):
            plt.subplot(target_id // n_cols + 1, target_id % n_cols + 1)

            # get the step-averaged losses for this target to identify top-k (rep, net) pairs
            ranking_target_losses = ranking_losses[:, target_id, :]  # shape: (n_replicates, n_networks)
            flat_losses = ranking_target_losses.reshape(-1)
            
            # get top-k (replicate, network) pairs based on step average
            top_k_actual = min(self.top_k_designs, len(flat_losses))
            top_k_flat_indices = np.argsort(flat_losses)[:top_k_actual]
            
            # plot each top-k (rep, net) pair over time
            all_positive = True
            for rank, flat_idx in enumerate(top_k_flat_indices):
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks

                # extract history for this specific (rep, net) pair across all logged steps
                step_numbers = []
                pair_history = []
                for step, step_losses in self._design_losses_history:
                    step_numbers.append(step)
                    val = step_losses[rep_id, target_id, net_id]
                    pair_history.append(val)
                    if val <= 0:
                        all_positive = False

                color = colors[rank % len(colors)]
                step_avg_loss = ranking_target_losses[rep_id, net_id]
                plt.plot(
                    step_numbers,
                    pair_history,
                    marker="braille",
                    color=color,
                    label=f"R{rep_id}N{net_id} ({step_avg_loss:.3f})",
                )

            if all_positive:
                plt.yscale("log")
            plt.title(f"Target {target_id}: Top {self.top_k_designs} (Rep,Net) Pairs")
            plt.xlabel("Step")
            plt.ylabel("Loss (log)")

        plt.show()

    def _plot_loss_history(self):
        """Plot the loss history in the console using log scale - one line per replicate"""
        import plotext as plt

        if not self._replicate_histories:
            return

        plt.clf()
        plt.theme("matrix")
        plt.plot_size(self.plot_width, self.plot_height)

        # Plot each replicate as a separate line
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white']
        all_positive = True
        for replicate_id, loss_history in self._replicate_histories.items():
            if loss_history:  # only plot if we have data
                x_values = list(range(len(loss_history)))
                color = colors[replicate_id % len(colors)]
                plt.plot(
                    x_values,
                    loss_history,
                    marker="braille",
                    color=color,
                    label=f"Rep {replicate_id}",
                )
                if any(v <= 0 for v in loss_history):
                    all_positive = False

        if all_positive:
            plt.yscale("log")
        plt.title(f"Training Loss per Replicate (best avg: {self._best_mean_loss:.4f})")
        plt.xlabel("Batch")
        plt.ylabel("Loss (log scale)")
        plt.show()

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_loss(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is not None:
                losses = step_history.get('loss')
                if losses is not None:
                    # check for design mode by looking for all_losses in step_history
                    all_losses = step_history.get('all_losses')

                    # track design mode and update appropriate history
                    if all_losses is not None:
                        self._design_mode = True
                        self._update_design_history(all_losses, step)

                    self._update_history(step, losses)
                    self._print_step_stats(step, losses, all_losses)

                    # plot appropriate history based on mode
                    if self._design_mode:
                        self._plot_design_loss_history()
                    else:
                        self._plot_loss_history()

        return [(self.periods, log_loss)]


class ConsoleLogger(Logger):
    """Logs the training loss to console"""

    async_ok: bool = False  # fast console logger doesn't need async

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_loss(step, training_config, step_history=None, stack=None, **kwargs):
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
