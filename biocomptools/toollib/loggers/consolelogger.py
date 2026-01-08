## {{{                          --     imports     --

from dracon.deferred import DeferredNode
import numpy as np
from typing import List, Tuple, Callable, Literal, TypeVar
from pydantic import PrivateAttr
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class EnhancedConsoleLogger(Logger):
    """Logs and visualizes training loss with historical tracking.

    Uses the new pattern with HistoryView to access loss history from ALL steps,
    including steps where this logger wasn't called.
    """

    # new pattern attributes
    frequency: int = 10
    history_mode: Literal["window", "since_last", "all"] = "all"
    history_window: int | None = None
    required_metrics: list[str] = []
    required_arrays: list[str] = ["loss", "all_losses"]

    # display config
    plot_height: int = 22
    plot_width: int = 100
    top_k_designs: int = 5
    async_ok: bool = False

    # internal state
    _design_mode: bool = PrivateAttr(default=False)
    _best_mean_loss: float = PrivateAttr(default=float('inf'))

    def on_batch(self, view: HistoryView, context: LoggerContext):
        if view.n_batches == 0:
            return

        latest = view.latest()
        if latest is None:
            return

        current_losses = latest.arrays.get('loss')
        if current_losses is None:
            return

        all_losses = latest.arrays.get('all_losses')
        if all_losses is not None:
            self._design_mode = True

        self._print_step_stats(context.current_step, current_losses, all_losses)

        mean_loss = float(np.mean(current_losses))
        if mean_loss < self._best_mean_loss:
            self._best_mean_loss = mean_loss

        if self._design_mode:
            self._plot_design_loss_history_from_view(view)
        else:
            self._plot_loss_history_from_view(view)

    def _print_step_stats(
        self, step: int, losses: np.ndarray, all_losses: np.ndarray | None = None
    ):
        from rich.console import Console
        from rich.table import Table

        console = Console()
        losses = np.asarray(losses)

        if all_losses is not None:
            self._print_design_stats(step, all_losses, console, self.top_k_designs)
        else:
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
        from rich.table import Table

        all_losses = np.asarray(all_losses)

        if all_losses.ndim == 2:
            n_targets, n_networks = all_losses.shape
            n_replicates = 1
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 3:
            n_replicates, n_targets, n_networks = all_losses.shape
        elif all_losses.ndim == 4:
            n_replicates, n_batches, n_targets, n_networks = all_losses.shape
            all_losses = np.mean(all_losses, axis=1)
        else:
            console.print(f"[red]Unexpected all_losses shape: {all_losses.shape}[/red]")
            return

        table = Table(title=f"Step {step} Design Statistics (Top {top_k} per Target)")
        table.add_column("Target", justify="right", style="cyan")
        table.add_column("Rep", justify="right", style="yellow")
        table.add_column("Net", justify="right", style="magenta")
        table.add_column("Loss (avg)", justify="right", style="green")

        for target_id in range(n_targets):
            target_losses = all_losses[:, target_id, :]
            flat_losses = target_losses.reshape(-1)
            top_k_actual = min(top_k, len(flat_losses))
            top_indices = np.argsort(flat_losses)[:top_k_actual]

            for rank, flat_idx in enumerate(top_indices):
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks
                loss_val = flat_losses[flat_idx]
                loss_style = "bright_green" if rank == 0 else "green"
                table.add_row(
                    str(target_id) if rank == 0 else "",
                    str(rep_id),
                    str(net_id),
                    f"[{loss_style}]{loss_val:.4f}[/{loss_style}]",
                )

        console.print(table)

    def _plot_loss_history_from_view(self, view: HistoryView):
        import plotext as plt

        if view.n_batches == 0:
            return

        plt.clf()
        plt.theme("matrix")
        plt.plot_size(self.plot_width, self.plot_height)

        replicate_histories: dict[int, list[float]] = {}

        for batch in view.iter_batches():
            loss_arr = batch.arrays.get('loss')
            if loss_arr is None:
                continue
            loss_arr = np.asarray(loss_arr)

            if loss_arr.ndim == 1:
                for rep_id, val in enumerate(loss_arr):
                    replicate_histories.setdefault(rep_id, []).append(float(val))
            elif loss_arr.ndim == 2:
                for rep_id, rep_losses in enumerate(loss_arr):
                    replicate_histories.setdefault(rep_id, []).extend(rep_losses.tolist())

        if not replicate_histories:
            return

        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white']
        all_positive = True

        for rep_id, losses in replicate_histories.items():
            if losses:
                x_values = list(range(len(losses)))
                color = colors[rep_id % len(colors)]
                plt.plot(x_values, losses, marker="braille", color=color, label=f"Rep {rep_id}")
                if any(v <= 0 for v in losses):
                    all_positive = False

        if all_positive:
            plt.yscale("log")
        plt.title(f"Training Loss per Replicate (best avg: {self._best_mean_loss:.4f})")
        plt.xlabel("Batch")
        plt.ylabel("Loss (log scale)")
        plt.show()

    def _plot_design_loss_history_from_view(self, view: HistoryView):
        import plotext as plt

        if view.n_batches == 0:
            return

        # collect design losses history from view
        design_history: list[tuple[int, np.ndarray]] = []
        for batch in view.iter_batches():
            all_losses = batch.arrays.get('all_losses')
            if all_losses is None:
                continue
            all_losses = np.asarray(all_losses)
            # normalize to (n_replicates, n_targets, n_networks)
            if all_losses.ndim == 2:
                all_losses = all_losses[None, :, :]
            elif all_losses.ndim == 4:
                all_losses = np.mean(all_losses, axis=1)
            elif all_losses.ndim != 3:
                continue
            design_history.append((batch.step_index, all_losses))

        if not design_history:
            return

        # use latest for ranking
        _, ranking_losses = design_history[-1]
        n_replicates, n_targets, n_networks = ranking_losses.shape

        n_cols = min(2, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        plt.clf()
        plt.theme("matrix")
        plt.subplots(n_rows, n_cols)
        plt.plot_size(self.plot_width, self.plot_height * n_rows)

        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'orange']

        for target_id in range(n_targets):
            plt.subplot(target_id // n_cols + 1, target_id % n_cols + 1)

            ranking_target_losses = ranking_losses[:, target_id, :]
            flat_losses = ranking_target_losses.reshape(-1)
            top_k_actual = min(self.top_k_designs, len(flat_losses))
            top_k_flat_indices = np.argsort(flat_losses)[:top_k_actual]

            all_positive = True
            for rank, flat_idx in enumerate(top_k_flat_indices):
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks

                step_numbers = []
                pair_history = []
                for step, step_losses in design_history:
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


class ConsoleLogger(Logger):
    """Logs the training loss to console"""

    async_ok: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_loss(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is not None:
                losses = step_history.get('loss')
                if losses is not None:
                    for i, loss in enumerate(losses):
                        avg_loss = np.mean(loss)
                        min_loss = np.min(loss)
                        max_loss = np.max(loss)
                        logger.debug(
                            f"Step {step}, Replicate {i}: Avg loss: {avg_loss:.4f}, Min loss: {min_loss:.4f}, Max loss: {max_loss:.4f}"
                        )

        return [(self.periods, log_loss)]
