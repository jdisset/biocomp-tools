# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""SSOT for the lattice kernel noise-floor estimator.

The model-side eRMSE floor (`networkprediction`) and the replicate-side
rep-noise floor (`replicate_metrics`) are the *same* estimator: kNN-smooth
(x, y) onto a hypercube lattice, then linearly interpolate that surface to
query points. Defining it once here lets both pipelines compare on one ruler.

Params come from `biocomp.metric_utils.DEFAULT_GRIDSTATS_PARAMS`."""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray as NdArray
from scipy.interpolate import RegularGridInterpolator

from jeanplot.splat import SplatField


def make_hypercube(ndim: int, res: int = 100, xmin: float = 0, xmax: float = 1) -> NdArray:
    """Hypercube lattice with ``indexing='ij'`` so the flattened grid reshapes
    naturally to ``(res, ..., res, n_outs)`` for ``RegularGridInterpolator``.
    """
    assert ndim > 0 and res > 0
    grid = np.meshgrid(
        *[np.linspace(xmin, xmax, res) for _ in range(ndim)],
        indexing='ij',
    )
    return np.vstack([g.ravel() for g in grid]).T


def kernel_lattice_interp(
    grid_mean_latent: NdArray,
    params: Dict[str, Any],
    ndim: int,
) -> RegularGridInterpolator:
    """Linear interpolator over a flattened (res**ndim, n_outs) grid-mean
    array, using the same lattice as ``make_hypercube``. SSOT for evaluating
    kernel-smoother predictions at arbitrary points.
    """
    res = int(params['hypercube_res'])
    arr = np.asarray(grid_mean_latent)
    n_outs = arr.shape[-1] if arr.ndim > 1 else 1
    grid_axes = tuple(
        np.linspace(params['hypercube_min'], params['hypercube_max'], res)
        for _ in range(ndim)
    )
    return RegularGridInterpolator(
        grid_axes, arr.reshape(*([res] * ndim), n_outs),
        method='linear', bounds_error=False, fill_value=np.nan,
    )


def fit_lattice(latent_x: NdArray, latent_y: NdArray | None, params: Dict[str, Any]) -> SplatField:
    """Fixed-sigma splat smoother on the cube-view lattice. SSOT geometry for
    gt and yhat (same x + params -> same normaliser, so means are comparable)."""
    d = latent_x.shape[1]
    return SplatField.fit(
        latent_x, latent_y,
        bounds=[(params['hypercube_min'], params['hypercube_max'])] * d,
        resolution=params['hypercube_res'],
        radius=params['radius'],
        sigma_in_radius=params.get('sigma_in_radius', 3.0),
        min_points=params['min_points'],
        stats=['mean', 'std'],
    )


def _kernel_smoother_lattice(
    latent_x: NdArray,
    latent_y: NdArray,
    params: Dict[str, Any],
) -> Tuple[NdArray, NdArray, NdArray, SplatField]:
    """Splat-smoothed mean/stdev/n_eff on the lattice (flattened to match
    ``make_hypercube``). The returned field is the SSOT geometry."""
    field = fit_lattice(latent_x, latent_y, params)
    n = field.n_outs
    mean = field.lattice('mean').reshape(-1, n)
    stdev = field.lattice('std').reshape(-1, n)
    n_eff = np.nan_to_num(field._n_eff).reshape(-1, 1)
    return mean, stdev, n_eff, field


def lattice_kernel_predict(
    x_train: NdArray,
    y_train: NdArray,
    x_query: NdArray,
    params: Dict[str, Any],
) -> NdArray:
    """Model-side kernel noise-floor surface fit on (x_train, y_train),
    evaluated at x_query. Dedupes identical x rows (the lattice smoother gains
    nothing from exact duplicates), matching `_compute_data_kernel_state`.
    Returns ``(len(x_query), n_outs)``; NaN where the lattice extrapolates."""
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    x_query = np.asarray(x_query, dtype=np.float32)
    n_outs = y_train.shape[1]

    finite = np.all(np.isfinite(x_train), axis=1) & np.all(np.isfinite(y_train), axis=1)
    x_train, y_train = x_train[finite], y_train[finite]
    if x_train.shape[0] == 0:
        return np.full((len(x_query), n_outs), np.nan, dtype=np.float32)

    _, uniq = np.unique(x_train, axis=0, return_index=True)
    x_train, y_train = x_train[uniq], y_train[uniq]
    field = fit_lattice(x_train, y_train, params)
    return np.asarray(field.at(x_query, 'mean')).reshape(len(x_query), n_outs)
