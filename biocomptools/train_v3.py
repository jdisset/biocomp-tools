## {{{                          --     imports     --
import pickle
import numpy as np
import biocomp.compute as cmp
from typing import List, Tuple
from scipy.ndimage import gaussian_filter1d
import dracon as dr
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from labellines import labelLine, labelLines
from biocomptools.modelmodel import BiocompModel, get_shared_params, get_nonshared_params
import biocomp.utils as ut
import pydantic
from pathlib import Path

import base64
import zlib
from pydantic import BaseModel, field_serializer, field_validator

ndArray = np.ndarray | jnp.ndarray

##────────────────────────────────────────────────────────────────────────────}}}

## {{{              --     saving and plotting best model     --


def get_best_smoothed_loss_id(all_losses: ndArray, sigma: float = 12.0) -> Tuple[int, np.ndarray]:
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma, mode='nearest')
    endval = smoothed_losses[:, -1]
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
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    nan_mask = np.isnan(all_losses)
    filled_losses = ffill(all_losses)
    best_loss_id, smoothed_losses = get_best_smoothed_loss_id(filled_losses)

    yrange = np.nanmax(all_losses) - np.nanmin(all_losses)

    # Plot non-nan values as blue solid lines
    for i in range(all_losses.shape[0]):
        non_nan_indices = ~nan_mask[i]
        ax.plot(
            np.arange(all_losses.shape[1])[non_nan_indices],
            all_losses[i, non_nan_indices],
            color='#AAA',
            linestyle='-',
            linewidth=1,
            alpha=0.5,
        )

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
            )
        # only plot smoothed for the valid

    try:
        labelLines(ax.get_lines(), zorder=2.5)
    except Exception as e:
        pass

    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}


# open ./training_output/training/all_models.pickle
trainingdir = Path('~/Code/Weiss/trainingtest/training_output/training').expanduser().resolve()
with open(trainingdir / 'all_models.pickle', 'rb') as f:
    all_models = pickle.load(f)


# open ./training_output/training/loss_history.npy
loss_history = np.load(trainingdir / 'loss_history.npy')
plot_loss(loss_history)
best_model_id, smoothed_losses = get_best_smoothed_loss_id(loss_history)
best_model_id
best_params = get_shared_params(ut.tree_get(all_models, best_model_id))
smoothed_losses

##


compute_conf = dr.load(
    'file:training_output/training/training_program_dump@training_conf.compute_config'
)

model = BiocompModel(compute_config=compute_conf, shared_params=best_params)

model.model_dump()

dr.dump(model)
