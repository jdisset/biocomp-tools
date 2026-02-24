## {{{                          --     imports     --

from pathlib import Path
from datetime import datetime
import re
from dracon.deferred import DeferredNode
import numpy as np
import logging
from scipy.ndimage import gaussian_filter1d
from labellines import labelLines
import matplotlib.pyplot as plt
from numpy import ndarray as ndArray
from typing import Tuple
from pydantic import BaseModel

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

    pattern = re.compile(rf'^{re.escape(prefix)}{datestr}-(\d+){re.escape(suffix)}$')

    max_number = -1
    for existing_dir in directory.iterdir():
        match = pattern.match(existing_dir.name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    start_number = max_number + 1

    while True:
        candidate_name = f'{prefix}{datestr}-{suffix}-{start_number:03d}'
        dir_path = directory / candidate_name

        try:
            dir_path.mkdir(parents=True, exist_ok=False)
            return dir_path

        except FileExistsError:
            # If we hit a collision just try the next number
            start_number += 1


def ffill(arr, mask=None):
    if mask is None:
        mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def _losses_array_to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        if arr.shape[1] <= arr.shape[0]:
            return np.moveaxis(arr, 1, 0)
        return arr

    # common training shape: (n_steps, n_replicates, batches_per_step, ...)
    arr = np.moveaxis(arr, 1, 0)
    return arr.reshape(arr.shape[0], -1)


def losses_to_2d(all_losses) -> np.ndarray:
    arr = np.asarray(all_losses)
    if arr.dtype != object:
        return _losses_array_to_2d(arr)

    flat_chunks = []
    for item in all_losses:
        item_2d = _losses_array_to_2d(np.asarray(item))
        if item_2d.size:
            flat_chunks.append(item_2d)
    if not flat_chunks:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(flat_chunks, axis=1)


def get_latest_avg_loss(all_losses, replicate_id: int, window: int = 64) -> float:
    """Calculates the average loss over the last `window` for a specific replicate."""
    losses_array = losses_to_2d(all_losses)
    if losses_array.size == 0:
        return np.nan

    if replicate_id >= losses_array.shape[0]:
        return np.nan

    replicate_losses = losses_array[replicate_id]
    avg_window = min(window, len(replicate_losses))
    latest_window = replicate_losses[-avg_window:]

    return float(np.nanmean(latest_window))


def get_best_smoothed_loss_replicate_id(
    all_losses,
    sigma: float = 12.0,
    max_window: int = 64,
) -> Tuple[int, np.ndarray, float]:
    """Determines the best replicate based on the average loss in the final window."""
    losses_array = losses_to_2d(all_losses)
    if losses_array.size == 0:
        return -1, np.array([]), np.inf

    n_replicates = losses_array.shape[0]

    end_vals = np.array(
        [get_latest_avg_loss(all_losses, i, window=max_window) for i in range(n_replicates)]
    )

    if np.all(np.isnan(end_vals)):
        return -1, np.array([]), np.inf

    best_replicate_id = int(np.nanargmin(end_vals))
    best_loss_value = end_vals[best_replicate_id]

    smoothed_losses = gaussian_filter1d(losses_array, sigma=sigma, axis=1, mode='nearest')

    return best_replicate_id, smoothed_losses, best_loss_value


def plot_loss(all_losses):
    losses_array = losses_to_2d(all_losses)  # (n_replicates, total_steps)
    if losses_array.size == 0:
        losses_array = np.zeros((1, 1), dtype=float)

    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])

    nan_mask = np.isnan(losses_array)
    ffill(losses_array)
    best_loss_id, smoothed_losses, _ = get_best_smoothed_loss_replicate_id(all_losses)
    if smoothed_losses.ndim != 2 or smoothed_losses.shape != losses_array.shape:
        smoothed_losses = losses_array.copy()

    yrange = np.nanmax(losses_array) - np.nanmin(losses_array)

    # plot non-nan values as blue solid lines
    colormap = plt.cm.get_cmap('tab10')
    lines = []
    for i in range(losses_array.shape[0]):
        non_nan_indices = ~nan_mask[i]
        line = ax.plot(
            np.arange(losses_array.shape[1])[non_nan_indices],
            losses_array[i, non_nan_indices],
            color='#AAA',
            linestyle='-',
            linewidth=1,
            alpha=0.5,
        )
        lines.append(line)

        nan_boundaries = np.where(np.diff(non_nan_indices))[0]
        # plot red cross
        for boundary in nan_boundaries:
            ax.plot(
                boundary,
                losses_array[i, boundary],
                'x',
                linewidth=2,
                color='red',
                alpha=0.5,
                markersize=5,
            )
            offsetx = 0.01 * losses_array.shape[1]
            offsety = 0.00 * yrange
            ax.text(
                boundary + offsetx,
                losses_array[i, boundary] + offsety,
                f'rep {i}',
                fontsize=7,
                color='red',
                ha='left',
                va='center',
            )

        valid_propotion = non_nan_indices.sum() / losses_array.shape[1]

        if valid_propotion > 0.2:
            ax.plot(
                np.arange(losses_array.shape[1])[non_nan_indices],
                smoothed_losses[i, non_nan_indices],
                linewidth=1,
                label=f'rep {i}',
                color=colormap(i % 20),
            )

    if best_loss_id >= 0 and best_loss_id < smoothed_losses.shape[0]:
        best_loss_str = f'Best loss with replicate {best_loss_id}, ~ {smoothed_losses[best_loss_id, -1]:.4f}'
    else:
        best_loss_str = 'Best loss unavailable'
    ax.set_title(f'Loss history. {best_loss_str}')

    try:
        labelLines(ax.get_lines(), zorder=2.5)
    except Exception:
        pass

    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax


def print_matadata(fig, ax, metadata: dict, run_name: str):
    """Add metadata to the figure in a clean, formatted way"""
    fig.suptitle(f'Run "{run_name}"')

    ax_meta = fig.add_subplot(fig.add_gridspec(1, 2, width_ratios=[3, 1])[1])
    ax_meta.set_axis_off()

    meta_text = '\n'.join(f'{k}: {v}' for k, v in metadata.items())
    ax_meta.text(0, 1, meta_text, va='top', ha='left', fontsize=8)

    plt.tight_layout()

    return fig


def make_json_ready(obj):
    """Roundtrip to json to iron out any weakref/unpickleable issues with DeferredNodes"""
    import json
    from dracon.dracontainer import Mapping, Sequence
    import numpy as np

    def sanitize_keys(o):
        """Recursively sanitize dict keys (convert tuples to strings)."""
        if isinstance(o, dict):
            return {str(k) if isinstance(k, tuple) else k: sanitize_keys(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [sanitize_keys(i) for i in o]
        return o

    def convert(o):
        if isinstance(o, DeferredNode):
            return {f'{o.value.tag}': 'deferred'}
        elif isinstance(o, BaseModel):
            return o.model_dump()
        elif isinstance(o, Mapping):
            return {str(k) if isinstance(k, tuple) else k: v for k, v in o.items()}
        elif isinstance(o, Sequence):
            return [i for i in o]
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.integer, np.floating)):
            return o.item()
        else:
            logger.debug(f"Unhandled type during json serialization: {type(o)}")
            return str(type(o))

    # First sanitize keys, then dump
    sanitized = sanitize_keys(obj)
    dmp = json.dumps(sanitized, default=convert)

    return json.loads(dmp)


##────────────────────────────────────────────────────────────────────────────}}}
