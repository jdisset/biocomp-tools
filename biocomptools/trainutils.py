## {{{                          --     imports     --

from pathlib import Path
from datetime import datetime
import re
import dracon as dr
from dracon.deferred import DeferredNode
import numpy as np
import logging
from scipy.ndimage import gaussian_filter1d
from labellines import labelLine, labelLines
import matplotlib.pyplot as plt
from numpy import ndarray as ndArray
from typing import Dict, List, Optional, Tuple, Callable, Union, Annotated, Literal, TypeVar
from pydantic import BaseModel, ConfigDict
from biocomptools.plot import plot_extra_context
from biocomptools.plot import PlotJob

logger = logging.getLogger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     Loggers     --


class Logger(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    periods: Union[int, List[int]] = 1  # Number of steps between logs or list of periods

    def initialize(self, training_program):
        """Optional initialization before training starts."""
        pass

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Return a list of (period, callback_function) tuples for the training loop."""
        raise NotImplementedError

    def finalize(self):
        """Optional cleanup after training ends."""
        pass


T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class PlotLogger(Logger):
    jobs: List[DeferredNode[PlotJob]] = []

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def plot_callback(step, training_config, step_history=None, **kwargs):
            logger.debug(f"\n==== PlotLogger callback at step {step} ====")
            best_model = None
            if step_history is not None:
                losses = step_history.get('loss')
                params = step_history.get('latest_params')
                best_model = training_program.get_best_model(params, losses)

                if best_model is not None:
                    logger.debug(f"Got best model with signature: {best_model.signature()}")

            logging.info(f'Plotting at step {step}')

            for job in self.jobs:
                j = job.copy()
                jstr = dr.node_repr(j, context_paths=['**.d', '**.all_predicted_data'])
                try:
                    if best_model is not None:
                        constructed_job = j.construct(
                            context={
                                **plot_extra_context,
                                'training_program': training_program,
                                'best_model': best_model,
                                'step': step,
                            },
                            # deferred_paths=['/**.figures.*'],
                        )
                        if not isinstance(constructed_job, PlotJob):
                            constructed_job = PlotJob(**constructed_job)
                        constructed_job.run()
                    else:
                        logging.info('Skipping prediction plots - no best model available yet')
                except Exception as e:
                    logging.error(f'Error plotting job: {e}')
                    continue

        return [(self.periods, plot_callback)]


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


##────────────────────────────────────────────────────────────────────────────}}}

## {{{              --     saving and plotting best model     --


def make_unique_dir(directory: Path | str, prefix: str = '', suffix: str = ''):
    """
    Generate a unique name for a new directory inside the given directory.
    """
    directory = Path(directory)
    datestr = datetime.now().strftime('%Y%m%d')

    directory.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(f'^{re.escape(prefix)}{datestr}-(\d+){re.escape(suffix)}$')

    # Find the highest existing number for today
    max_number = -1
    for existing_dir in directory.iterdir():
        match = pattern.match(existing_dir.name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    # Start trying from the next number
    start_number = max_number + 1

    while True:
        candidate_name = f'{prefix}{datestr}-{start_number:03d}{suffix}'
        dir_path = directory / candidate_name

        try:
            # Try to create the directory - this is atomic on most filesystems
            dir_path.mkdir()
            return dir_path

        except FileExistsError:
            # If we hit a collision just try the next number
            start_number += 1


def generate_unique_funny_name(directory: Path | str, prefix: str = '', suffix: str = '') -> str:
    """Generate a unique name for a file or directory in the given directory."""
    funny_words = dr.load('pkg:biocomptools:configs/funnywords.yaml')
    directory = Path(directory)
    adj = np.random.choice(funny_words['adjectives'])
    noun = np.random.choice(funny_words['nouns'])
    name = f'{prefix}{adj}-{noun}{suffix}'
    # add a number to the name if it already exists
    i = 1
    while (directory / name).exists():
        name = f'{prefix}{adj}-{noun}-{i}{suffix}'
        i += 1
    return name


def get_best_smoothed_loss_id(all_losses: ndArray, sigma: float = 12.0) -> Tuple[int, np.ndarray]:
    all_losses = np.asarray(all_losses)
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma, mode='nearest')
    # endval = smoothed_losses[:, -1]
    # instead, take the mean of the last third of the unsmoothed losses
    endval = np.mean(all_losses[:, -int(all_losses.shape[1] / 3) :], axis=1)
    endval[np.isnan(endval)] = np.inf
    best_loss_id = int(np.argmin(endval))
    return best_loss_id, smoothed_losses


def ffill(arr, mask=None):
    if mask is None:
        mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def plot_loss(loss_history: List[np.ndarray]):
    all_losses = np.hstack(loss_history)

    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])

    nan_mask = np.isnan(all_losses)
    filled_losses = ffill(all_losses)
    best_loss_id, smoothed_losses = get_best_smoothed_loss_id(filled_losses)

    yrange = np.nanmax(all_losses) - np.nanmin(all_losses)

    # plot non-nan values as blue solid lines
    colormap = plt.cm.get_cmap('tab10')
    lines = []
    for i in range(all_losses.shape[0]):
        non_nan_indices = ~nan_mask[i]
        l = ax.plot(
            np.arange(all_losses.shape[1])[non_nan_indices],
            all_losses[i, non_nan_indices],
            color='#AAA',
            linestyle='-',
            linewidth=1,
            alpha=0.5,
        )
        lines.append(l)

        nan_boundaries = np.where(np.diff(non_nan_indices))[0]
        # plot red cross
        for boundary in nan_boundaries:
            ax.plot(
                boundary,
                all_losses[i, boundary],
                'x',
                linewidth=2,
                color='red',
                alpha=0.5,
                markersize=5,
            )
            offsetx = 0.01 * all_losses.shape[1]
            offsety = 0.00 * yrange
            ax.text(
                boundary + offsetx,
                all_losses[i, boundary] + offsety,
                f'rep {i}',
                fontsize=7,
                color='red',
                ha='left',
                va='center',
            )

        valid_propotion = non_nan_indices.sum() / all_losses.shape[1]

        if valid_propotion > 0.2:
            ax.plot(
                np.arange(all_losses.shape[1])[non_nan_indices],
                smoothed_losses[i, non_nan_indices],
                linewidth=1,
                label=f'rep {i}',
                color=colormap(i % 20),
            )

    ax.set_title(
        f'Loss history. Best loss with replicate {best_loss_id}, ~ {smoothed_losses[best_loss_id, -1]:.4f}'
    )

    try:
        labelLines(ax.get_lines(), zorder=2.5)
    except Exception as e:
        pass

    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax


def add_metadata(fig, ax, metadata: dict, run_name: str):
    """Add metadata to the figure in a clean, formatted way"""
    fig.suptitle(f'Run "{run_name}"')

    ax_meta = fig.add_subplot(fig.add_gridspec(1, 2, width_ratios=[3, 1])[1])
    ax_meta.set_axis_off()

    meta_text = '\n'.join(f'{k}: {v}' for k, v in metadata.items())
    ax_meta.text(0, 1, meta_text, va='top', ha='left', fontsize=8)

    plt.tight_layout()

    return fig


##────────────────────────────────────────────────────────────────────────────}}}
