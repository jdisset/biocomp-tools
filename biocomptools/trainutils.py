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
