# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from pydantic.functional_validators import BeforeValidator
from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, List, Union, Dict, Annotated, Literal, Tuple, TypeAlias, Callable
import numpy as np
import jax.numpy as jnp
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
from biocomp.datautils import IdentityRescaler
from biocomptools.toollib.common import make_pretty_input_names
from biocomptools.modelmodel import NetworkModel, BiocompModel, NodeSpec
from biocomptools.logging_config import get_logger
import biocomp.parameters as pr
from biocomptools.toollib.datasources import DataSource
from biocomptools.toollib.types import InputOrderElement
from jeanplot.knn import get_gaussian_weighted_knn
from jeanplot.plots.smooth_kernel import build_tree, knn_stats
from biocomp.datautils import density_balanced_indices
from scipy.interpolate import RegularGridInterpolator
from biocomp.metric_utils import (
    grid_mse as compute_grid_mse,
    grid_r_squared as compute_grid_r_squared,
    grid_snr as compute_grid_snr,
    grid_kl_divergence as compute_grid_kl,
    compute_nrmse,
    compute_nrmse_pointwise,
    GridStatsFields,
)
from pathlib import Path
import os
import time
import json as _json
import xxhash

# from concurrent.futures import ProcessPoolExecutor as PoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed

from biocomp.utils import get_cache
from dracon.progress import each, step

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


def _cache_dir(env_var: str, default_subdir: str) -> Path:
    return Path(os.environ.get(env_var, os.path.expanduser(f"~/.cache/{default_subdir}")))


def _env_flag(env_var: str, default: bool) -> bool:
    val = os.environ.get(env_var)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


_DATA_KERNEL_CACHE_DIR = _cache_dir("BIOCOMP_DATA_KERNEL_CACHE_DIR", "biocomp_data_kernel")
_DATA_KERNEL_CACHE_DISABLED = _env_flag("BIOCOMP_DATA_KERNEL_CACHE_DISABLE", False)
_PREDICTION_CACHE_DIR = _cache_dir("BIOCOMP_PREDICTION_CACHE_DIR", "biocomp_predictions")
_PREDICTION_CACHE_ENABLED = _env_flag("BIOCOMP_PREDICTION_CACHE_ENABLE", False)


def _dks_save(data: Dict[str, Any], path: Path) -> None:
    arr_kwargs: Dict[str, Any] = {}
    scalar_kwargs: Dict[str, Any] = {}
    for k, v in data.items():
        if k == 'iw':
            idx, w = v
            arr_kwargs['iw_idx'] = np.ascontiguousarray(idx, dtype=np.int32)
            arr_kwargs['iw_w'] = np.ascontiguousarray(w, dtype=np.float32)
        elif isinstance(v, np.ndarray):
            arr_kwargs[k] = v
        else:
            scalar_kwargs[k] = v
    arr_kwargs['__scalars__'] = np.array(_json.dumps(scalar_kwargs))
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arr_kwargs)
    tmp.replace(path)


def _dks_load(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        with np.load(fh, allow_pickle=False) as f:
            out: Dict[str, Any] = {k: f[k] for k in f.files if k not in ('__scalars__', 'iw_idx', 'iw_w')}
            out['iw'] = (f['iw_idx'], f['iw_w'])
            out.update(_json.loads(str(f['__scalars__'])))
    return out


_DKS_SERIALIZER = {"save": _dks_save, "load": _dks_load}


class PredictionSamplingConfig(BaseModel):
    """Controls how NetworkPrediction samples the stochastic layers.

    Grouped because these knobs move together - a `distribution` preset
    flips several at once, and every prediction-bearing caller accepts
    the full set. Kept as one sub-model so a 10th knob is one edit.
    """
    model_config = ConfigDict(extra='forbid')

    z_value: Union[Literal['uniform', 'normal'], float] = 'uniform'
    z_normal_mean: float = 0.5
    z_normal_std: float = 0.2
    z_normal_clip: bool = True
    disable_variational: bool = True
    max_evals: int = 250_000
    seed: int = 0
    device: Literal['cpu', 'gpu'] = 'gpu'
    input_order: Optional[list] = None


def _stats_task_dump(
    *,
    task: Dict[str, Any],
    idx: int,
    n_networks: int,
    yhats: List[NdArray] | None,
    xvals: List[NdArray] | None,
    input_order: Any,
    collection_points: Any,
) -> Dict[str, Any]:
    return {
        'n_networks': n_networks,
        'shapes_yhats': [yh.shape for yh in yhats] if yhats is not None else None,
        'shapes_x': [x.shape for x in xvals] if xvals is not None else None,
        'input_order': input_order,
        'collection_points': collection_points,
        'task_i': idx,
        'task': task,
    }


def reconstruct_from_flat(flat_values, shapes):
    """reconstruct a list of arrays from flat values and shapes"""
    result = []
    offset = 0
    for shape in shapes:
        n = np.prod(shape)
        result.append(flat_values[:, offset : offset + n].reshape(-1, *shape))
        offset += n
    return result


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


def _knn_mean_var_neff(tree, grid, y, k, min_points, max_radius, sigma_in_radius=3.0):
    """Combined KNN query + adaptive Gaussian weights + mean/var computation.

    Fused version of get_gaussian_weighted_knn(adaptive_sigma=True, normed_w=False)
    followed by normalization and get_knn_mean_and_variance.
    Returns (mean, variance, n_eff) without intermediate weight arrays.

    Skips rows below ``min_points`` valid neighbors (NaN in any case) - large win
    on 3D lattices where most cells are empty.
    """
    eps = 1e-12
    from jeanplot.knn.tree import KNN_WORKERS, _query
    n_grid = grid.shape[0]
    distances, indices = _query(tree, grid, k=k, workers=KNN_WORKERS)
    finite_mask = np.isfinite(distances)
    max_finite = np.where(finite_mask, distances, -np.inf).max(axis=1)

    if max_radius is not None:
        valid_mask = finite_mask & (distances <= max_radius)
    else:
        valid_mask = finite_mask
    nb_points = valid_mask.sum(axis=1)

    valid_rows = nb_points >= min_points
    n_outs = y.shape[1] if y.ndim > 1 else 1

    if not valid_rows.any():
        # Everything is too_few -> all NaN
        nan_mat = np.full((n_grid, n_outs), np.nan)
        nan_neff = np.full((n_grid, 1), np.nan)
        return nan_mat, nan_mat.copy(), nan_neff

    if valid_rows.all():
        sub_distances = distances
        sub_indices = indices
        sub_valid_mask = valid_mask
        sub_max_finite = max_finite
    else:
        sub_distances = distances[valid_rows]
        sub_indices = indices[valid_rows]
        sub_valid_mask = valid_mask[valid_rows]
        sub_max_finite = max_finite[valid_rows]

    sigma = (sub_max_finite / sigma_in_radius).reshape(-1, 1) + eps

    inv_sigma = 1.0 / sigma
    Z = sub_distances * inv_sigma
    Z *= Z
    Z *= -0.5
    np.exp(Z, out=Z)
    Z[~sub_valid_mask] = 0.0

    n_eff_v = Z.sum(axis=1, keepdims=True)

    indices_c = sub_indices
    if not sub_valid_mask.all():
        indices_c = sub_indices.copy()
        indices_c[~sub_valid_mask] = 0

    y_nei = y[indices_c]
    z = Z[..., None]
    sum_y = (z * y_nei).sum(axis=1)
    sum_y2 = (z * y_nei * y_nei).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_v = sum_y / n_eff_v
        m2 = sum_y2 / n_eff_v
        variance_num_v = m2 - mean_v * mean_v
        # DoF correction: 1 - sum(W^2) = 1 - sum(Z^2)/n_eff^2
        z2sum = (Z * Z).sum(axis=1, keepdims=True)
        denom = np.maximum(1.0 - z2sum / (n_eff_v * n_eff_v), eps)
        variance_v = variance_num_v / denom

    if valid_rows.all():
        return mean_v, variance_v, n_eff_v

    mean = np.full((n_grid, n_outs), np.nan)
    variance = np.full((n_grid, n_outs), np.nan)
    n_eff = np.full((n_grid, 1), np.nan)
    mean[valid_rows] = mean_v
    variance[valid_rows] = variance_v
    n_eff[valid_rows] = n_eff_v
    return mean, variance, n_eff


def kernel_lattice_interp(
    grid_mean_latent: NdArray,
    params: Dict[str, Any],
    ndim: int,
) -> RegularGridInterpolator:
    """Linear interpolator over a flattened (res**ndim, n_outs) grid-mean
    array, using the same lattice as ``make_hypercube``. SSOT for
    evaluating kernel-smoother predictions at arbitrary points after
    ``_calculate_grid_stats`` has produced the lattice means.
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


def _kernel_smoother_lattice(
    latent_x: NdArray,
    latent_y: NdArray,
    grid: NdArray,
    params: Dict[str, Any],
) -> Tuple[NdArray, NdArray, NdArray, Tuple[NdArray, NdArray]]:
    """Adaptive-sigma Gaussian-kNN smoother evaluated on the cube-view
    lattice. Returns ``(mean, stdev, n_eff, (indices, norm_weights))``;
    the ``iw`` tuple is reused for further ``knn_stats`` calls on
    aligned y arrays.
    """
    tree = build_tree(latent_x)
    indices, raw_weights = get_gaussian_weighted_knn(
        grid, tree=tree,
        k=params['k'], min_points=params['min_points'],
        adaptive_sigma=True, max_radius=params['radius'],
        normed_w=False,
    )
    n_eff = np.nansum(raw_weights, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_weights = raw_weights / n_eff
    _, stdev, mean = knn_stats(
        grid, latent_y, iw=(indices, norm_weights),
        stats=['iw', 'std', 'mean'],
        k=params['k'], min_points=params['min_points'],
    )
    return mean, stdev, n_eff, (indices, norm_weights)


def validate_predict_at(v: Any) -> List[NdArray]:
    """convert single numpy array to list of arrays"""
    if isinstance(v, NdArray):
        return [np.asarray(v, dtype=np.float32)]
    if isinstance(v, list) and all(isinstance(x, NdArray) for x in v):
        return [np.asarray(x, dtype=np.float32) for x in v]
    return v


def validate_ground_truth(v: Any) -> Optional[List[Optional[NdArray]]]:
    """convert single numpy array to list of arrays or none to list of none"""
    if v is None:
        logger.warning("ground_truth is None, will not compare predictions to ground truth")
        return None
    if isinstance(v, NdArray):
        logger.warning(f"ground_truth is a single array of shape {v.shape}, converting to list")
        aslist = [np.asarray(v, dtype=np.float32)]
        logger.info(f"ground_truth converted to list of shapes {[x.shape for x in aslist]}")
    return v


def validate_input_order(v: Any) -> Optional[List[List[InputOrderElement]]]:
    """Convert flat list of ints/strings to list of lists for broadcasting."""
    if v is None:
        return None
    if isinstance(v, list) and all(isinstance(x, (int, str)) for x in v):
        return [v]
    return v


# static function for parallel processing
def _calculate_single_network_stats(
    network_idx: int,
    yhat: NdArray,
    gt: Optional[NdArray],
    x: NdArray,
    dependent_output_pos: Union[int, List[int]],
    nb_points_in_eval: int,
    rescaler,
    gridstats_params: Dict[str, Any],
    network_info: Dict[str, Any],
    enable_gridstats: bool = True,
) -> Dict[str, Any]:
    """Per-network statistics; safe to run in worker threads."""
    latent_yhats = np.asarray(rescaler.fwd(yhat), dtype=np.float32)
    if isinstance(dependent_output_pos, int):
        dependent_output_pos = [dependent_output_pos]
    latent_yhat = latent_yhats[:, dependent_output_pos]

    base = {
        'xp_name': network_info.get('xp_name'),
        'recipe_name': network_info.get('recipe_name'),
        'network_name': network_info.get('network_name', f"Network_{network_idx}"),
        'eval_npoints': nb_points_in_eval,
        'samples': yhat.shape[0],
        'mse': None,
        'rmse': None,
    }
    if 'extra_prediction_info' in network_info:
        base['extra_prediction_info'] = network_info['extra_prediction_info']

    if latent_yhat.size == 0:
        return {
            **base,
            'latent_mean': np.nan,
            'latent_std': np.nan,
            'latent_min': np.nan,
            'latent_max': np.nan,
            'error': 'Empty output array - invalid dependent_output_pos selection',
        }

    network_stats = {
        **base,
        'latent_mean': float(latent_yhat.mean()),
        'latent_std': float(latent_yhat.std()),
        'latent_min': float(latent_yhat.min()),
        'latent_max': float(latent_yhat.max()),
    }

    if gt is not None:
        latent_x = rescaler.fwd(x)
        latent_gt = np.asarray(rescaler.fwd(gt), dtype=np.float32)
        # Reconcile gt vs yhat column counts: force_single_output may have
        # collapsed gt to 1 column, or yhat may have extra columns to slice.
        if latent_gt.shape[1] > 1:
            latent_gt = latent_gt[:, dependent_output_pos]
        elif latent_yhat.shape[1] > 1 and latent_gt.shape[1] == 1:
            latent_yhat = latent_yhat[:, :1]
        assert latent_gt.shape[1] == latent_yhat.shape[1], (
            f"shape mismatch after reconcile: gt={latent_gt.shape}, yhat={latent_yhat.shape}"
        )

        mse = float(np.mean((latent_yhat - latent_gt) ** 2))
        network_stats['mse'] = mse
        network_stats['rmse'] = float(np.sqrt(mse))

        if enable_gridstats:
            grid_stats, _grid_cache = _calculate_grid_stats(
                latent_yhat, latent_gt, latent_x, gridstats_params
            )
            network_stats.update(grid_stats)

    return network_stats


def _data_kernel_signature(latent_x: NdArray, latent_gt: NdArray, params: Dict[str, Any]) -> str:
    """Stable hash over the model-independent inputs to grid stats compute.

    Uses shape + ~1024 decimated samples + a sorted, JSON-encoded subset
    of params. Float arrays are normalized to contiguous float32 so cache
    hits aren't broken by view/dtype noise. Bump the leading version tag
    when changing what's cached.
    """
    h = xxhash.xxh128()
    h.update(b"v2.")
    x = np.ascontiguousarray(latent_x, dtype=np.float32)
    g = np.ascontiguousarray(latent_gt, dtype=np.float32)
    h.update(repr(x.shape).encode())
    h.update(repr(g.shape).encode())
    n = max(1, len(x) // 1024)
    h.update(np.ascontiguousarray(x[::n]).tobytes())
    h.update(np.ascontiguousarray(g[::n]).tobytes())
    p_subset = {
        k: v for k, v in params.items()
        if isinstance(v, (int, float, str, bool, list, tuple))
    }
    h.update(_json.dumps(p_subset, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _compute_data_kernel_state(
    latent_x: NdArray,
    latent_gt: NdArray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Pure model-independent state for grid stats. Cacheable per (data, params).

    Captures everything in ``_calculate_grid_stats`` that depends only on
    ``(latent_x, latent_gt, params)`` - the kernel smoother lattice, the
    kernel interpolator outputs, the density-balanced subsample, and the
    split-half data nRMSE. The model-dependent residuals are computed
    fresh per call from these cached arrays.
    """
    latent_gt_2d = latent_gt.reshape(-1, 1) if latent_gt.ndim == 1 else latent_gt

    valid_x_mask = np.all(np.isfinite(latent_x), axis=1)
    full_x = np.asarray(latent_x[valid_x_mask])
    full_gt = np.asarray(latent_gt_2d[valid_x_mask])
    _, unique_idx = np.unique(full_x, axis=0, return_index=True)
    deduped_x = full_x[unique_idx]
    deduped_gt = full_gt[unique_idx]

    d = full_x.shape[1]
    grid = make_hypercube(
        d,
        res=params['hypercube_res'],
        xmin=params['hypercube_min'],
        xmax=params['hypercube_max'],
    )

    gt_mean, gt_stdev, n_eff, iw = _kernel_smoother_lattice(
        deduped_x, deduped_gt, grid, params,
    )

    mu_interp = kernel_lattice_interp(gt_mean, params, d)
    sigma_interp = kernel_lattice_interp(gt_stdev, params, d)
    kernel_pred_full = np.asarray(mu_interp(full_x))
    kernel_resid = full_gt - kernel_pred_full

    sub_idx_in_valid = density_balanced_indices(
        full_x,
        n_samples=int(params.get('subsample_n', 5000)),
        knn_k=int(params.get('subsample_knn_k', 64)),
        density_threshold_quantile=float(params.get('subsample_density_quantile', 0.025)),
    )
    valid_to_original = np.where(valid_x_mask)[0]

    if sub_idx_in_valid.size:
        kr = kernel_resid[sub_idx_in_valid]
        sub_finite_mask = np.all(np.isfinite(kr), axis=1)
    else:
        sub_finite_mask = np.zeros(0, dtype=bool)

    if sub_idx_in_valid.size and sub_finite_mask.any():
        x_sub = full_x[sub_idx_in_valid][sub_finite_mask]
        sigma_sub = np.asarray(sigma_interp(x_sub))
    else:
        sigma_sub = np.empty((0, full_gt.shape[1]), dtype=np.float32)

    global_var = float(np.var(deduped_gt)) if deduped_gt.size else 1.0
    global_range = float(np.ptp(deduped_gt)) if deduped_gt.size else 1.0

    return {
        'valid_x_mask': valid_x_mask,
        'unique_idx': unique_idx,
        'full_x': full_x,
        'full_gt': full_gt,
        'd': d,
        'grid': grid,
        'gt_mean': gt_mean,
        'gt_stdev': gt_stdev,
        'n_eff': n_eff,
        'iw': iw,
        'kernel_pred_full': kernel_pred_full,
        'sub_idx_in_valid': sub_idx_in_valid,
        'sub_finite_mask': sub_finite_mask,
        'sigma_sub': sigma_sub,
        'valid_to_original': valid_to_original,
        'global_var': global_var,
        'global_range': global_range,
    }


def _calculate_grid_stats(
    latent_yhat: NdArray,
    latent_gt: NdArray,
    latent_x: NdArray,
    params: Dict[str, Any],
) -> tuple[Dict[str, Any], NdArray]:
    """Lattice-kernel grid stats + density-balanced kernel/model RMSE.

    Returns ``(stats, grid)`` so the caller can reuse the lattice for
    follow-on computations (e.g. split-half nRMSE). The lattice itself is
    built on de-duplicated data so the adaptive-sigma kernel can't divide
    by zero; the density-balanced subsample for ``kernel_rmse_latent`` /
    ``model_rmse_latent`` runs on the original (non-deduped) data.

    Model-independent state (kernel smoother, interpolators, subsample,
    split-half nRMSE) is cached on disk keyed by ``(latent_x, latent_gt,
    params)`` so that running this for the same network across many model
    evaluations only pays for the heavy data-side compute once.
    """
    latent_yhat = latent_yhat.reshape(-1, 1) if latent_yhat.ndim == 1 else latent_yhat
    latent_gt = latent_gt.reshape(-1, 1) if latent_gt.ndim == 1 else latent_gt

    if _DATA_KERNEL_CACHE_DISABLED:
        ds = _compute_data_kernel_state(latent_x, latent_gt, params)
    else:
        sig = _data_kernel_signature(latent_x, latent_gt, params)
        ds = get_cache(
            gen_f=lambda: _compute_data_kernel_state(latent_x, latent_gt, params),
            signature=sig,
            cache_location=_DATA_KERNEL_CACHE_DIR,
            serializer=_DKS_SERIALIZER,
        )

    valid_x_mask = ds['valid_x_mask']
    unique_idx = ds['unique_idx']
    grid = ds['grid']
    gt_mean = ds['gt_mean']
    gt_stdev = ds['gt_stdev']
    n_eff = ds['n_eff']
    iw = ds['iw']
    full_gt = ds['full_gt']
    kernel_pred_full = ds['kernel_pred_full']
    sub_idx_in_valid = ds['sub_idx_in_valid']
    sub_finite_mask = ds['sub_finite_mask']
    sigma_sub = ds['sigma_sub']
    valid_to_original = ds['valid_to_original']
    global_var = ds['global_var']
    global_range = ds['global_range']

    full_yhat = np.asarray(latent_yhat[valid_x_mask])
    deduped_yhat = full_yhat[unique_idx]

    yhat_stdev, yhat_mean = knn_stats(
        grid, deduped_yhat, iw=iw,
        stats=['std', 'mean'],
        k=params['k'], min_points=params['min_points'],
    )

    sq_error = (yhat_mean - gt_mean) ** 2
    local_var = gt_stdev ** 2

    mse_val = compute_grid_mse(yhat_mean, gt_mean)
    rmse_val = float(np.sqrt(mse_val))
    r_squared_val = compute_grid_r_squared(yhat_mean, gt_mean)
    snr_val = compute_grid_snr(gt_mean, local_var)
    kl_mean, kl_similarity = compute_grid_kl(yhat_mean, yhat_stdev, gt_mean, gt_stdev)
    nrmse_val = compute_nrmse(
        sq_error, local_var, n_eff, global_var, global_range, gt_mean,
        k=params['k'],
    )

    nan = float('nan')
    if sub_idx_in_valid.size and sub_finite_mask.any():
        sub_idx_finite = sub_idx_in_valid[sub_finite_mask]
        gt_sub = full_gt[sub_idx_in_valid][sub_finite_mask]
        yhat_sub = full_yhat[sub_idx_in_valid][sub_finite_mask]
        kernel_pred_sub = kernel_pred_full[sub_idx_in_valid][sub_finite_mask]

        kernel_resid_sub = gt_sub - kernel_pred_sub
        model_resid_sub = gt_sub - yhat_sub

        var_gt = float(np.var(gt_sub))
        mse_kernel = float(np.mean(kernel_resid_sub ** 2))
        mse_model = float(np.mean(model_resid_sub ** 2))
        kernel_rmse_latent = float(np.sqrt(mse_kernel))
        model_rmse_latent = float(np.sqrt(mse_model))
        ratio_rmse = (
            model_rmse_latent / kernel_rmse_latent
            if kernel_rmse_latent > 0 else nan
        )
        # Excess error over the kernel-smoother noise floor - single-pair,
        # noise-invariant scalars for cross-network model comparison.
        excess_rmse_latent = model_rmse_latent - kernel_rmse_latent
        bias_mag_latent = float(np.sqrt(max(0.0, mse_model - mse_kernel)))
        if var_gt > 0:
            kernel_r_squared_latent = 1.0 - mse_kernel / var_gt
            model_r_squared_latent = 1.0 - mse_model / var_gt
        else:
            kernel_r_squared_latent = model_r_squared_latent = nan
        ratio_r_squared = (
            model_r_squared_latent / kernel_r_squared_latent
            if (np.isfinite(kernel_r_squared_latent) and kernel_r_squared_latent != 0)
            else nan
        )

        sub_range = float(np.ptp(gt_sub)) if gt_sub.size else 1.0
        kernel_nrmse_local = compute_nrmse_pointwise(
            gt_sub, kernel_pred_sub, sigma_sub,
            gt_mean_local=kernel_pred_sub, global_range=sub_range,
        )
        model_nrmse_local = compute_nrmse_pointwise(
            gt_sub, yhat_sub, sigma_sub,
            gt_mean_local=kernel_pred_sub, global_range=sub_range,
        )
        kratio = (
            model_nrmse_local / kernel_nrmse_local
            if (np.isfinite(kernel_nrmse_local) and kernel_nrmse_local > 0)
            else nan
        )
        subsample_indices = valid_to_original[sub_idx_finite].astype(np.intp)
    else:
        kernel_rmse_latent = model_rmse_latent = ratio_rmse = nan
        kernel_r_squared_latent = model_r_squared_latent = ratio_r_squared = nan
        kernel_nrmse_local = model_nrmse_local = kratio = nan
        excess_rmse_latent = bias_mag_latent = nan
        subsample_indices = np.array([], dtype=np.intp)

    return {
        'grid_gt_var': float(np.nanvar(gt_mean)),
        'grid_mse': mse_val,
        'grid_rmse': rmse_val,
        'grid_nrmse': nrmse_val,
        'grid_snr': snr_val,
        'grid_kl': kl_mean,
        'grid_kl_similarity': kl_similarity,
        'grid_r_squared': r_squared_val,
        'kernel_rmse_latent': kernel_rmse_latent,
        'model_rmse_latent': model_rmse_latent,
        'ratio_rmse': ratio_rmse,
        'excess_rmse_latent': excess_rmse_latent,
        'bias_mag_latent': bias_mag_latent,
        'kernel_r_squared_latent': kernel_r_squared_latent,
        'model_r_squared_latent': model_r_squared_latent,
        'ratio_r_squared': ratio_r_squared,
        'kernel_nrmse_local': kernel_nrmse_local,
        'model_nrmse_local': model_nrmse_local,
        'kratio': kratio,
        'n_subsample_used': int(subsample_indices.size),
        'subsample_indices': subsample_indices,
        'grid_xy_latent': np.asarray(grid),
        'grid_gt_mean_latent': np.asarray(gt_mean),
        'grid_gt_stdev_latent': np.asarray(gt_stdev),
        'grid_yhat_mean_latent': np.asarray(yhat_mean),
        'grid_yhat_stdev_latent': np.asarray(yhat_stdev),
        'grid_n_eff': np.asarray(n_eff).ravel(),
    }, grid


class NetworkPrediction(GridStatsFields, DataSource):
    """Performs predictions using a NetworkModel and prepares data for plotting.

    IMPORTANT - Space Handling (Latent vs Raw):
    ==========================================
    The biocomp model operates in "latent space" (normalized 0-1 range). Raw fluorescence
    data (values in millions) must be rescaled before prediction.

    There are two modes controlled by `already_latent`:

    1. already_latent=False (DEFAULT):
       - predict_at: expects RAW space (original fluorescence values, e.g., 1e6)
       - Internally calls NetworkModel.predict_unscaled() which:
         * Converts X to latent via rescaler.fwd(X)
         * Runs model prediction
         * Converts Y back to raw via rescaler.inv(Y)
       - get_data() returns: (X_raw, Y_raw)

    2. already_latent=True:
       - predict_at: expects LATENT space (0-1 normalized values)
       - Internally calls NetworkModel.predict() directly (no rescaling)
       - get_data() returns: (X_latent, Y_latent)
       - get_data(rescale_latent=True) returns: (X_raw, Y_raw)

    Common mistake: Passing latent data without already_latent=True will cause
    double-rescaling (latent values treated as raw, scaled again), producing garbage.

    Example usage:
        # From raw experimental data (DBSource, etc.)
        pred = NetworkPrediction(predict_at=[X_raw], network_model=nm)
        data = pred.get_data()  # Returns (X_raw, Y_raw)

        # From design targets (DataTarget.X is already latent)
        pred = NetworkPrediction(predict_at=[X_latent], network_model=nm, already_latent=True)
        data = pred.get_data(rescale_latent=True)  # Returns (X_raw, Y_raw)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    predict_at: Annotated[Union[NdArray, List[NdArray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Annotated[
        Optional[List[List[InputOrderElement]]], BeforeValidator(validate_input_order)
    ] = None
    ground_truth: Annotated[
        Optional[Union[NdArray, List[Optional[NdArray]]]],
        BeforeValidator(validate_ground_truth),
    ] = None
    # Optional per-network column-protein identity. When supplied, verified
    # against `network.get_inverted_input_proteins()` at construction time -
    # this is the boundary check that catches X-column scrambling
    # (see bugs/eval-x-axis-permutation-iRFP720.md). `None` entries (or a
    # `None` list) skip the check; producers that know the column convention
    # should always supply it.
    predict_at_column_proteins: Optional[List[Optional[List[str]]]] = None

    per_prediction_info: Optional[list[Dict[str, Any]]] = None  # metadata for each prediction

    collection_points: Optional[List[NodeSpec]] = None  # collection points to examine inner nodes

    seed: int = 0
    max_evals: int = 300000
    use_output_as_input: bool = False
    z_value: Union[Literal['uniform', 'normal'], float] = 'uniform'
    z_normal_mean: float = 0.5
    z_normal_std: float = 0.2
    z_normal_clip: bool = True
    disable_variational: bool = True
    device: Literal['cpu', 'gpu'] = 'cpu'  # device preference for predictions

    # Optional grouped sampling config. When supplied, its fields win over the
    # scalar fields above (transition state - scalar kwargs stay as-is for
    # backward compat until every call site is migrated).
    sampling: Optional[PredictionSamplingConfig] = None

    save_csv_to: Optional[str] = None  # save prediction statistics to a CSV file

    already_latent: bool = (
        False  # Set True if predict_at is in latent space (0-1). See class docstring.
    )

    enable_gridstats: bool = True
    # gridstats_* fields inherited from GridStatsFields mixin

    shuffle_inputs: bool = False  # shuffle inputs before prediction (False preserves spatial structure)

    skip_input_reorder: bool = False  # skip input column reordering (for design visualization)

    n_stats_workers: int = 8  # number of workers for parallel processing of statistics

    # When True, _run_predict_and_stats only runs predict; stats are computed
    # lazily per-network on first `get_network_stats(network_idx)` access.
    # Disables the predict+stats blob cache (predict-only caching is a future
    # optimisation; predict alone is cheap).
    defer_stats_compute: bool = False

    verbose: bool = False  # print prediction statistics to the console

    _yhats: Optional[List[NdArray]] = None
    _x: Optional[List[NdArray]] = None
    _gtruths: Optional[List[Optional[NdArray]]] = None
    _aligned_x: Optional[List[NdArray]] = None
    _aligned_ground_truth: Optional[List[Optional[NdArray]]] = None
    _network_stats: Optional[List[Dict[str, Any]]] = None

    def model_post_init(self, *args, **kwargs):
        """initialize the model after validation"""
        super().model_post_init(*args, **kwargs)

        # If a grouped sampling config is supplied, unpack its fields onto the
        # scalar fields. Preserves backward compat for call sites that still
        # pass the 9 scalar kwargs directly.
        if self.sampling is not None:
            for _f in (
                'z_value', 'z_normal_mean', 'z_normal_std', 'z_normal_clip',
                'disable_variational', 'max_evals', 'seed', 'device', 'input_order',
            ):
                setattr(self, _f, getattr(self.sampling, _f))

        logger.debug(
            f"predict_at: {len(self.predict_at)} arrays, shapes: {[x.shape for x in self.predict_at]}"
        )

        assert isinstance(self.network_model.network, list)

        # validate number of networks matches input data
        if len(self.predict_at) != len(self.network_model.network):
            raise ValueError(
                f"number of predict_at arrays ({len(self.predict_at)}) "
                f"does not match number of networks ({len(self.network_model.network)})"
            )

        # validate and fix input dimensions for each network
        fixed_predict_at = []
        for i, (x, net) in enumerate(zip(self.predict_at, self.network_model.network, strict=True)):
            expected_inputs = net.nb_inputs
            actual_inputs = x.shape[1] if x.ndim > 1 else 1
            if actual_inputs > expected_inputs:
                raise ValueError(
                    f"Network {i} ({net.name}): predict_at has {actual_inputs} columns but "
                    f"network expects {expected_inputs} inputs. "
                    "Explicitly provide correctly shaped inputs instead of relying on truncation."
                )
            elif actual_inputs < expected_inputs:
                raise ValueError(
                    f"Network {i} ({net.name}): predict_at has {actual_inputs} columns but "
                    f"network expects {expected_inputs} inputs."
                )
            else:
                fixed_predict_at.append(x)
        self.predict_at = fixed_predict_at

        # Boundary check: verify each predict_at array's columns match the
        # network's wired input protein order. This is the structural guard
        # for the X-column scrambling bug class. Skipped per-entry when the
        # caller can't supply column_proteins (e.g. design-space inputs with
        # placeholder X1/X2 labels).
        if self.predict_at_column_proteins is not None:
            assert len(self.predict_at_column_proteins) == len(self.predict_at), (
                f"predict_at_column_proteins has {len(self.predict_at_column_proteins)} "
                f"entries but predict_at has {len(self.predict_at)}"
            )
            for i, (cp, net) in enumerate(
                zip(self.predict_at_column_proteins, self.network_model.network, strict=True)
            ):
                if cp is None:
                    continue
                expected = net.get_inverted_input_proteins()
                assert list(cp) == list(expected), (
                    f"Network {i} ({getattr(net, 'name', '?')}): predict_at columns are "
                    f"aligned to {list(cp)} but network expects {list(expected)}. "
                    f"This is the X-column misalignment bug class - do not silence this "
                    f"assertion; fix the producer of predict_at."
                )

        if self.ground_truth is None:
            self.ground_truth = [None] * len(self.predict_at)
        else:
            if len(self.ground_truth) != len(self.predict_at):
                raise ValueError(
                    f"number of ground truth arrays ({len(self.ground_truth)}) "
                    f"does not match number of predict_at arrays ({len(self.predict_at)})"
                )

        # shuffle inputs with fixed random seed
        if self.shuffle_inputs:
            self._shuffle_inputs()

        # prepare aligned inputs
        self._aligned_x, self._aligned_ground_truth = self._prepare_inputs()
        logger.debug(f"aligned_x shapes: {[x.shape for x in self._aligned_x]}")
        logger.debug(
            f"aligned_ground_truth shapes: {[None if gt is None else gt.shape for gt in self._aligned_ground_truth]}"
        )

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkPrediction':
        """create a new networkprediction with a different model"""
        logger.debug(f"creating new networkprediction with model {model.signature=}")

        # create new instance with updated network model
        new = self.model_copy(update={'network_model': self.network_model.with_model(model)})
        new._yhats = None  # clear the cache
        new._network_stats = None  # clear the stats cache

        return new

    def with_csv_output_path(self, path: str) -> 'NetworkPrediction':
        """set path to save prediction statistics to a CSV file"""
        return self.model_copy(update={'save_csv_to': path})

    def with_device(self, device: Literal['cpu', 'gpu']) -> 'NetworkPrediction':
        """set device preference for predictions"""
        return self.model_copy(update={'device': device})

    def _shuffle_inputs(self):
        """shuffle inputs and ground truth with the same random order"""
        # set random seed for reproducibility
        rng = np.random.RandomState(self.seed)

        new_predict_at = []
        new_gt = []
        for x, gt in zip(self.predict_at, self.ground_truth, strict=True):
            order = rng.permutation(len(x))
            new_predict_at.append(x[order])
            new_gt.append(gt[order] if gt is not None else None)
        self.predict_at = new_predict_at
        self.ground_truth = new_gt

    def _prepare_inputs(self) -> Tuple[List[NdArray], List[Optional[NdArray]]]:
        """prepare inputs by padding or truncating to the same length"""
        max_prediction_length = max(len(x) for x in self.predict_at)
        logger.debug(f"max_prediction_length across networks: {max_prediction_length}")

        effective_max_evals = min(
            self.max_evals if self.max_evals > 0 else max_prediction_length, max_prediction_length
        )
        logger.debug(f"effective_max_evals: {effective_max_evals}")

        aligned_predict_at = []
        aligned_ground_truth = []

        for i, (x, gt, _network) in enumerate(
            zip(self.predict_at, self.ground_truth, self.network_model.network, strict=True)
        ):
            if len(x) < effective_max_evals:
                zeros = np.zeros((effective_max_evals - len(x), x.shape[1]))
                padded_x = np.vstack([x, zeros])
                aligned_predict_at.append(padded_x)

                if gt is not None:
                    gtzeros = np.zeros((effective_max_evals - len(x), gt.shape[1]))
                    padded_gt = np.vstack([gt, gtzeros])
                    aligned_ground_truth.append(padded_gt)
                else:
                    aligned_ground_truth.append(None)
            else:
                logger.debug(
                    f"truncating prediction queries for network {i} from {len(x)} to {effective_max_evals} points"
                )
                truncated_x = x[:effective_max_evals]
                aligned_predict_at.append(truncated_x)

                if gt is not None:
                    truncated_gt = gt[:effective_max_evals]
                    aligned_ground_truth.append(truncated_gt)
                else:
                    aligned_ground_truth.append(None)

        return aligned_predict_at, aligned_ground_truth

    def _prediction_cache_signature(self) -> Optional[str]:
        h = xxhash.xxh128()
        h.update(b"v1.")
        h.update(str(self.network_model.signature).encode())
        for net in self.network_model.network:
            try:
                h.update(repr(net.to_recipe()).encode())
            except Exception:
                return None
        assert isinstance(self._aligned_x, list)
        for x in self._aligned_x:
            a = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
        params = {
            'seed': self.seed,
            'z_value': str(self.z_value),
            'z_normal_mean': float(self.z_normal_mean),
            'z_normal_std': float(self.z_normal_std),
            'z_normal_clip': bool(self.z_normal_clip),
            'disable_variational': bool(self.disable_variational),
            'already_latent': bool(self.already_latent),
            'device': str(self.device),
        }
        h.update(_json.dumps(params, sort_keys=True).encode())
        return h.hexdigest()

    def compute_all_network_predictions(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ):
        start_time = time.time()
        assert isinstance(self._aligned_x, list)
        stacked_x = np.column_stack(self._aligned_x)
        effective_max_evals = len(self._aligned_x[0])

        predict_f = (
            self.network_model.predict
            if self.already_latent
            else self.network_model.predict_unscaled
        )

        if self.verbose:
            logger.debug(f"computing predictions with model {self.network_model.signature}")
            logger.debug(
                f"prediction params: seed={self.seed}, disable_variational={self.disable_variational}, "
                f"z_value={self.z_value}, z_normal_mean={self.z_normal_mean}, "
                f"z_normal_std={self.z_normal_std}, z_normal_clip={self.z_normal_clip}, "
                f"device={self.device}"
            )
            logger.debug(f"effective_max_evals: {effective_max_evals}")

        collect_in_idx, collect_out_idx = [], []
        self._input_shapes, self._output_shapes = [], []
        if self.collection_points:
            for cp in self.collection_points:
                (in_idx, input_shapes), (out_idx, output_shapes) = (
                    self.network_model.get_node_indices(cp.network_id, cp.node_id)
                )
                collect_in_idx.append(in_idx)
                collect_out_idx.append(out_idx)
                self._input_shapes.append(input_shapes)
                self._output_shapes.append(output_shapes)

        def _run_predict_and_stats():
            with step("predict"):
                yhats, (cin, cout) = predict_f(
                    stacked_x,
                    key=self.seed,
                    z_value=self.z_value,
                    z_normal_mean=self.z_normal_mean,
                    z_normal_std=self.z_normal_std,
                    z_normal_clip=self.z_normal_clip,
                    disable_variational=self.disable_variational,
                    with_shared_params=with_shared_params,
                    with_local_params=with_local_params,
                    collect_in_indices=np.concatenate(collect_in_idx).flatten() if collect_in_idx else None,
                    collect_out_indices=np.concatenate(collect_out_idx).flatten() if collect_out_idx else None,
                    device=self.device,
                )
            outs = self.network_model.split_outputs_per_network(np.asarray(yhats), effective_max_evals)
            self._process_prediction_results(outs, effective_max_evals)
            self._collected_in = np.asarray(cin) if cin is not None else None
            self._collected_out = np.asarray(cout) if cout is not None else None
            if self.defer_stats_compute:
                self._network_stats = [None] * len(self.network_model.network)
            else:
                with step("stats"):
                    self._network_stats = self._calculate_all_network_stats()
            return {
                'yhats': self._yhats,
                'gtruths': self._gtruths,
                'x': self._x,
                'stats': self._network_stats,
                'collected_in': self._collected_in,
                'collected_out': self._collected_out,
            }

        cacheable = (
            _PREDICTION_CACHE_ENABLED
            and with_shared_params is None
            and with_local_params is None
            and not self.collection_points
            and not self.defer_stats_compute
        )
        sig = self._prediction_cache_signature() if cacheable else None

        if sig is None:
            _run_predict_and_stats()
        else:
            blob = get_cache(
                gen_f=_run_predict_and_stats,
                signature=sig,
                cache_location=_PREDICTION_CACHE_DIR,
            )
            self._yhats = list(blob['yhats'])
            self._gtruths = list(blob['gtruths'])
            self._x = list(blob['x'])
            self._network_stats = blob['stats']
            self._collected_in = blob['collected_in']
            self._collected_out = blob['collected_out']

        logger.info(f"Network predictions+stats completed in {time.time() - start_time:.2f} seconds")

    def _process_prediction_results(self, network_outputs: List[NdArray], max_evals: int):
        """process and store prediction results"""
        self._x = []
        self._yhats = []
        self._gtruths = []
        assert isinstance(self.ground_truth, list)
        assert isinstance(self._aligned_ground_truth, list)
        assert isinstance(self._aligned_x, list)

        for i, (network_output, aligned_x) in enumerate(
            zip(network_outputs, self._aligned_x, strict=True)
        ):
            effective_max_evals = min(max_evals, len(aligned_x))

            # store prediction results
            truncated_output = network_output[:effective_max_evals]
            self._yhats.append(np.asarray(truncated_output, dtype=np.float32))

            # store ground truth if available
            if self.ground_truth[i] is not None:
                truncated_gt = self._aligned_ground_truth[i][:effective_max_evals]
                self._gtruths.append(np.asarray(truncated_gt, dtype=np.float32))
            else:
                self._gtruths.append(None)

            # store inputs in NETWORK order (not display order)
            # extract_lazy_plot_data_from_network will apply input_order to convert to display
            truncated_x = aligned_x[:effective_max_evals]
            self._x.append(np.asarray(truncated_x, dtype=np.float32))

        # validate results
        assert len(self._x) == len(self.predict_at)
        assert len(self._x) == len(self.network_model.network)

    def _create_xy_function(
        self,
        network_idx: int,
        rescale_latent: bool = False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> Callable[[PlotData], Tuple[NdArray, NdArray]]:
        """create a function to get x and y data for a specific network"""

        def get_xy(pdata: PlotData) -> Tuple[NdArray, NdArray]:
            if self.verbose:
                logger.debug(
                    f"getting xy for network {network_idx} with model {self.network_model.signature}"
                )

            # compute predictions if not already computed
            if (
                not hasattr(self, '_yhats')
                or self._yhats is None
                or with_shared_params is not None
                or with_local_params is not None
            ):
                self.compute_all_network_predictions(
                    with_shared_params=with_shared_params,
                    with_local_params=with_local_params,
                )
                if self.save_csv_to:
                    self.save_csv()

            assert isinstance(self._x, list)
            assert isinstance(self._yhats, list)
            assert isinstance(self._gtruths, list)

            x = self._x[network_idx]
            yhat = self._yhats[network_idx]
            gt = self._gtruths[network_idx]

            if self.verbose:
                logger.debug(
                    f"x shape: {x.shape}, yhat shape: {yhat.shape}, gt: {None if gt is None else gt.shape}"
                )

            assert isinstance(self.network_model.network, list)

            network = self.network_model.network[network_idx]

            # get output position for this network (can be int for single output or list for multiple outputs)
            _, dependent_output_pos, _, _ = get_reordered_protein_names(network)
            if self.verbose:
                logger.info(f"dep_output_pos: {dependent_output_pos}")

            stat = self.get_network_stats(network_idx=network_idx)
            # Drop array-valued grid overlays from the embedded plot metadata -
            # they're consumed by data-holder helpers via `get_network_stats()`
            # directly and don't belong in the JSON-bound figure subject blob.
            pdata.metadata['prediction_stats'] = {
                k: v for k, v in stat.items()
                if not isinstance(v, np.ndarray)
            }

            # apply inverse rescaling if requested and data is already latent
            if self.already_latent and rescale_latent:
                x = self.network_model.model.rescaler.inv(x)
                yhat = self.network_model.model.rescaler.inv(yhat)
                if self.verbose:
                    logger.debug("applied inverse rescaling to x and yhat")

            if self.use_output_as_input:
                if self.verbose:
                    logger.debug("using output as input, returning yhat as both x and y")
                return yhat, yhat
            else:
                if self.verbose:
                    logger.debug("using original input, returning x and yhat")
                return x, yhat

        return get_xy

    def _log_stats(
        self,
        network_idx: int,
        prediction_stats: Dict[str, Any],
        latent_gt: NdArray,
        latent_yhat: NdArray,
        latent_x: NdArray,
    ):
        if self.verbose:
            n_samples = 5

            assert len(latent_gt) == len(latent_yhat) == len(latent_x)
            sample_ids = np.random.choice(len(latent_gt), n_samples, replace=False)
            gt_sample = latent_gt[sample_ids]
            yhat_sample = latent_yhat[sample_ids]
            latentx_sample = latent_x[sample_ids]
            se_sample = (yhat_sample - gt_sample) ** 2

            original_yhat = self.network_model.model.rescaler.inv(yhat_sample)
            original_gt = self.network_model.model.rescaler.inv(gt_sample)
            unscaledx_sample = self.network_model.model.rescaler.inv(latentx_sample)
            unscaledx_sample = [tuple(np.round(x, 3) for x in xs) for xs in unscaledx_sample]
            latentx_sample = [tuple(np.round(x, 3) for x in xs) for xs in latentx_sample]
            import pandas as pd

            df = pd.DataFrame(
                {
                    'unscaled X': unscaledx_sample,
                    'unscaled gt': original_gt.tolist(),
                    'unscaled yhat': original_yhat.tolist(),
                    'latent X': latentx_sample,
                    'latent gt': gt_sample.tolist(),
                    'latent yhat': yhat_sample.tolist(),
                    'l2 error': se_sample.tolist(),
                }
            )

            logger.info(
                f"""network {network_idx} evaluated over {prediction_stats['samples']} samples:
                    - mse: {prediction_stats['mse']:.3f}
                    - rmse: {prediction_stats['rmse']:.3f}
                    - mean: {prediction_stats['latent_mean']:.3f}
                    - std: {prediction_stats['latent_std']:.3f}
                    - min: {prediction_stats['latent_min']:.3f}
                    - max: {prediction_stats['latent_max']:.3f}
                    """
            )
            logger.info(f"Random samples:\n{df.round(3).to_string()}")

    def _build_stats_task(self, i: int) -> Dict[str, Any]:
        network = self.network_model.network[i]
        network_name = getattr(network, 'name', f"Network_{i}")
        nb_points_in_eval = len(self.predict_at[i])
        _, dependent_output_pos, _, _ = get_reordered_protein_names(network)
        all_outputs = set(network.get_output_proteins())
        input_proteins = set(network.get_inverted_input_proteins())
        n_dependent_outputs = len(all_outputs - input_proteins)
        network_info = {
            'network_name': network_name,
            'n_dependent_outputs': n_dependent_outputs,
        }
        if self.per_prediction_info is not None and i < len(self.per_prediction_info):
            network_info['extra_prediction_info'] = self.per_prediction_info[i]
        return {
            'network_idx': i,
            'yhat': self._yhats[i],
            'gt': self._gtruths[i],
            'x': self._x[i],
            'dependent_output_pos': dependent_output_pos,
            'nb_points_in_eval': nb_points_in_eval,
            'rescaler': self.network_model.model.rescaler
            if not self.already_latent
            else IdentityRescaler(),
            'gridstats_params': self.get_gridstats_params(),
            'network_info': network_info,
            'enable_gridstats': self.enable_gridstats,
        }

    def _compute_single_network_stats(self, i: int) -> Dict[str, Any]:
        """Compute stats for one network. Used by both eager (parallel) and
        lazy (per-figure-worker) paths."""
        return _calculate_single_network_stats(**self._build_stats_task(i))

    def _calculate_all_network_stats(self) -> List[Dict[str, Any]]:
        tasks = [self._build_stats_task(i) for i in range(len(self.network_model.network))]

        all_stats: List[Dict[str, Any] | None] = [None] * len(self.network_model.network)

        # Bench-mode hook: dump the inputs to _calculate_single_network_stats and
        # short-circuit. Not part of the production path - only fires when
        # BIOCOMP_BENCH_CAPTURE is set in the environment.
        _bench_capture = os.environ.get("BIOCOMP_BENCH_CAPTURE")
        if _bench_capture:
            import pickle
            cache_data = {
                "yhats": [t["yhat"] for t in tasks],
                "gtruths": [t["gt"] for t in tasks],
                "x": [t["x"] for t in tasks],
                "rescaler": tasks[0]["rescaler"] if tasks else None,
                "networks": list(self.network_model.network),
                "names": [t["network_info"].get("network_name", f"net_{i}")
                          for i, t in enumerate(tasks)],
            }
            with open(_bench_capture, "wb") as f:
                pickle.dump(cache_data, f)
            logger.info(f"BIOCOMP_BENCH_CAPTURE: dumped stats inputs to {_bench_capture}")
            # do not exit; let stats run so we still get correct artifacts

        def process_and_log_stats(idx, task, result):
            all_stats[idx] = result
            if self.verbose and task.get('gt') is not None:
                rescaler = task['rescaler']
                latent_gt = rescaler.fwd(task['gt'])
                latent_yhat = rescaler.fwd(task['yhat'])
                if latent_gt.shape < latent_yhat.shape:
                    logger.debug(
                        f"ground truth shape {latent_gt.shape} is smaller than yhat shape {latent_yhat.shape}, "
                        "Assuming gt is only the dependent outputs."
                    )
                    output_pos = task['dependent_output_pos']
                    if isinstance(output_pos, int):
                        output_pos = [output_pos]
                    latent_yhat = latent_yhat[:, output_pos]
                latent_x = rescaler.fwd(task['x'])
                self._log_stats(idx, result, latent_gt, latent_yhat, latent_x)

        n_total = len(tasks)
        if self.n_stats_workers > 1 and len(tasks) > 1:
            logger.info(
                f"Calculating network stats in parallel with {self.n_stats_workers} workers."
            )
            with PoolExecutor(max_workers=self.n_stats_workers) as executor:
                future_to_idx = {
                    executor.submit(_calculate_single_network_stats, **task): task['network_idx']
                    for task in tasks
                }
                for future in each("network", as_completed(future_to_idx), total=n_total):
                    idx = future_to_idx[future]
                    task = tasks[idx]
                    try:
                        result = future.result()
                    except Exception as e:
                        net_name = task['network_info'].get('network_name', f"Network_{idx}")
                        raise RuntimeError(
                            f"Error calculating stats for network {net_name}: {e}; "
                            f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                        ) from e
                    process_and_log_stats(idx, task, result)

        else:
            if len(tasks) > 0:
                logger.info(
                    "Calculating network stats sequentially using the unified static method."
                )

            for task in each("network", tasks, total=n_total):
                idx = task['network_idx']
                try:
                    result = _calculate_single_network_stats(**task)
                except Exception as e:
                    net_name = task['network_info'].get('network_name', f"Network_{idx}")
                    raise RuntimeError(
                        f"Error calculating stats for network {net_name}: {e}; "
                        f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                    ) from e
                process_and_log_stats(idx, task, result)
        assert all(s is not None for s in all_stats), "all network stats must be populated"
        return [s for s in all_stats if s is not None]

    def compute_stats_from_outputs(self, network_outputs: List[NdArray]) -> List[Dict[str, Any]]:
        """Compute stats from externally-provided network outputs (e.g., from batched predictions).

        This is the canonical stats computation - uses same logic as _calculate_all_network_stats.
        Useful for hyperopt batched validation where outputs come from vmapped predictions.
        """
        from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed

        rescaler = (
            self.network_model.model.rescaler if not self.already_latent else IdentityRescaler()
        )
        gridstats_params = self.get_gridstats_params()

        # Build tasks
        tasks = []
        for i, network in enumerate(self.network_model.network):
            if i >= len(network_outputs):
                break
            yhat = network_outputs[i]
            gt = (
                self._aligned_ground_truth[i]
                if self._aligned_ground_truth and i < len(self._aligned_ground_truth)
                else None
            )
            x = self._aligned_x[i] if self._aligned_x and i < len(self._aligned_x) else None

            _, dependent_output_pos, _, _ = get_reordered_protein_names(network)
            network_name = getattr(network, 'name', f"Network_{i}")
            nb_points_in_eval = len(self.predict_at[i]) if self.predict_at else yhat.shape[0]

            all_outputs = set(network.get_output_proteins())
            input_proteins = set(network.get_inverted_input_proteins())
            n_dependent_outputs = len(all_outputs - input_proteins)

            network_info = {
                'network_name': network_name,
                'n_dependent_outputs': n_dependent_outputs,
            }
            if self.per_prediction_info is not None and i < len(self.per_prediction_info):
                network_info['extra_prediction_info'] = self.per_prediction_info[i]

            tasks.append(
                {
                    'network_idx': i,
                    'yhat': yhat,
                    'gt': gt,
                    'x': x,
                    'dependent_output_pos': dependent_output_pos,
                    'nb_points_in_eval': nb_points_in_eval,
                    'rescaler': rescaler,
                    'gridstats_params': gridstats_params,
                    'network_info': network_info,
                    'enable_gridstats': self.enable_gridstats,
                }
            )

        # Process tasks (parallel if n_stats_workers > 1)
        all_stats: List[Dict[str, Any] | None] = [None] * len(tasks)
        show_progress = self.verbose and len(tasks) > 3

        if self.n_stats_workers > 1 and len(tasks) > 1:
            from tqdm import tqdm

            with PoolExecutor(max_workers=self.n_stats_workers) as executor:
                future_to_idx = {
                    executor.submit(_calculate_single_network_stats, **task): task['network_idx']
                    for task in tasks
                }
                futures_iter = as_completed(future_to_idx)
                if show_progress:
                    futures_iter = tqdm(
                        futures_iter, total=len(tasks), desc="Stats", leave=False, ncols=80
                    )
                for future in futures_iter:
                    idx = future_to_idx[future]
                    task = tasks[idx]
                    try:
                        all_stats[idx] = future.result()
                    except Exception as e:
                        net_name = task['network_info'].get('network_name', f'Network_{idx}')
                        raise RuntimeError(
                            f"Stats computation failed for network {net_name}: {e}; "
                            f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                        ) from e
        else:
            from tqdm import tqdm

            task_iter = tasks
            if show_progress:
                task_iter = tqdm(tasks, desc="Stats", leave=False, ncols=80)
            for task in task_iter:
                idx = task['network_idx']
                try:
                    all_stats[idx] = _calculate_single_network_stats(**task)
                except Exception as e:
                    net_name = task['network_info'].get('network_name', f'Network_{idx}')
                    raise RuntimeError(
                        f"Stats computation failed for network {net_name}: {e}; "
                        f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                    ) from e
        assert all(s is not None for s in all_stats), "all output stats must be populated"
        return [s for s in all_stats if s is not None]

    def get_network_stats(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
        network_idx: Optional[int] = None,
    ):
        """Get network stats. Without ``network_idx`` returns the full list
        (computing all missing entries). With ``network_idx`` returns just one
        network's stats, computing it lazily if missing - used by per-figure
        workers to avoid eager all-up-front compute."""
        if (
            not hasattr(self, '_yhats')
            or self._yhats is None
            or with_shared_params
            or with_local_params
        ):
            self.compute_all_network_predictions(
                with_shared_params=with_shared_params,
                with_local_params=with_local_params,
            )

        if not hasattr(self, '_network_stats') or self._network_stats is None:
            self._network_stats = [None] * len(self.network_model.network)

        if network_idx is not None:
            if self._network_stats[network_idx] is None:
                self._network_stats[network_idx] = self._compute_single_network_stats(network_idx)
            return self._network_stats[network_idx]

        if any(s is None for s in self._network_stats):
            missing = [i for i, s in enumerate(self._network_stats) if s is None]
            if self.n_stats_workers > 1 and len(missing) > 1:
                from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed
                with PoolExecutor(max_workers=self.n_stats_workers) as ex:
                    futs = {ex.submit(self._compute_single_network_stats, i): i for i in missing}
                    for fut in as_completed(futs):
                        self._network_stats[futs[fut]] = fut.result()
            else:
                for i in missing:
                    self._network_stats[i] = self._compute_single_network_stats(i)

        return self._network_stats

    def get_shared_params(self) -> pr.ParameterTree:
        """get shared parameters from the network model"""
        return self.network_model.model.shared_params

    def get_local_params(self) -> pr.ParameterTree:
        """get local parameters from the network model"""
        # ensure the network model is built and parameters are initialized
        if (
            not hasattr(self.network_model, '_local_params')
            or self.network_model._local_params is None
        ):
            self.network_model.update_params()
        return self.network_model._local_params

    def save_csv(self):
        """save prediction statistics to a CSV file"""
        import pandas as pd

        stats = self.get_network_stats()
        # Drop array-valued fields (grid-mean / grid-weight overlays) - the
        # CSV is for scalar metrics; arrays are consumed by plot helpers.
        stats = [
            {k: v for k, v in s.items() if not isinstance(v, np.ndarray)}
            for s in stats
        ]
        df = pd.DataFrame(stats)

        # update df with content of self.extra_metadata
        for key, value in self.metadata.items():
            df[key] = value

        if self.verbose:
            logger.debug(f"prediction statistics DataFrame:\n{df.to_string()}")

        assert self.save_csv_to is not None, "save_csv_to must be set before saving CSV"
        save_to = Path(self.save_csv_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
        logger.info(f"saved prediction statistics to {save_to}")

    def _normalize_input_order(self) -> List[Optional[List[InputOrderElement]]]:
        """normalize input_order to ensure it's consistent across networks"""
        assert self.network_model is not None
        assert isinstance(self.network_model.network, list)

        if self.input_order is None:
            logger.debug(
                f"input_order is None, using None for all {len(self.network_model.network)} networks"
            )
            return [None] * len(self.network_model.network)

        if isinstance(self.input_order, list):
            # single input order for all networks (ints or protein names)
            if all(isinstance(x, (int, str)) for x in self.input_order):
                logger.debug(f"using same input_order {self.input_order} for all networks")
                return [self.input_order] * len(self.network_model.network)

            if len(self.input_order) != len(self.network_model.network):
                # if it's a list of a single element, use it for all networks
                if len(self.input_order) == 1 and isinstance(self.input_order[0], list):
                    logger.debug(
                        f"input_order is a single list, using it for all {len(self.network_model.network)} networks"
                    )
                    return [self.input_order[0]] * len(self.network_model.network)
                raise ValueError(
                    f"input order list has {len(self.input_order)} items but there are "
                    f"{len(self.network_model.network)} networks"
                )
            return self.input_order

        raise ValueError(f"unexpected input_order type: {type(self.input_order)}")

    def get_data_lazy(
        self,
        rescale_latent=False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[LazyPlotData]:
        """get plot data in lazy evaluation mode"""
        logger.debug(f"getting data lazily for model {self.network_model.signature}")

        # handle collection points if provided
        if self.collection_points is not None and len(self.collection_points) > 0:
            return self._get_collection_data_lazy(with_shared_params, with_local_params)

        plot_data_list = []
        if self.skip_input_reorder:
            input_order = [list(range(net.nb_inputs)) for net in self.network_model.network]
            logger.debug(f"skip_input_reorder=True, using identity input_order: {input_order}")
        else:
            input_order = self._normalize_input_order()
            logger.debug(f"normalized input_order: {input_order}")

        for i, network in each("extract", list(enumerate(self.network_model.network))):
            metadata = self._create_network_metadata(i, network)
            plot_data = self._extract_plot_data(
                i,
                network,
                input_order[i],
                metadata,
                rescale_latent=rescale_latent,
                with_shared_params=with_shared_params,
                with_local_params=with_local_params,
            )

            network_info = metadata['network_info']
            pretty_inputs = make_pretty_input_names(
                network_info['cotx'],
                plot_data.input_names,
            )
            plot_data.metadata['pretty_inputs'] = pretty_inputs

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} plot data objects")
        return plot_data_list

    def _get_collection_data_lazy(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[LazyPlotData]:
        """get plot data for collection points in lazy evaluation mode"""
        logger.debug(f"getting collection data lazily for model {self.network_model.signature}")

        assert self.collection_points is not None

        plot_data_list = []

        # compute index offsets for each collection point if needed
        if (
            not hasattr(self, '_collected_in')
            or self._collected_in is None
            or with_shared_params is not None
            or with_local_params is not None
        ):
            self.compute_all_network_predictions(
                with_shared_params=with_shared_params,
                with_local_params=with_local_params,
            )

        assert self._collected_in is not None
        assert len(self.collection_points) == len(self._input_shapes)
        assert self._collected_out is not None
        assert len(self.collection_points) == len(self._output_shapes)

        in_sizes, out_sizes = [], []
        for inshapes, outshapes in zip(self._input_shapes, self._output_shapes, strict=True):
            in_sizes.append(np.sum([np.prod(s) for s in inshapes]))
            out_sizes.append(np.sum([np.prod(s) for s in outshapes]))
        in_offsets = np.cumsum([0] + in_sizes[:-1])
        out_offsets = np.cumsum([0] + out_sizes[:-1])

        collect_in_idx = [in_offsets[i] + np.arange(in_sizes[i]) for i in range(len(in_sizes))]
        collect_out_idx = [out_offsets[i] + np.arange(out_sizes[i]) for i in range(len(out_sizes))]

        # create plot data for each collection point
        for i, collection_point in enumerate(self.collection_points):

            def get_xy_fn(
                pdata: PlotData,
                collection_point_nodespec=collection_point,
                input_shapes=self._input_shapes[i],
                output_shapes=self._output_shapes[i],
                i=i,
            ) -> Tuple[NdArray, NdArray]:
                if (
                    not hasattr(self, '_collected_in')
                    or self._collected_in is None
                    or with_shared_params is not None
                    or with_local_params is not None
                ):
                    self.compute_all_network_predictions(
                        with_shared_params=with_shared_params,
                        with_local_params=with_local_params,
                    )
                assert self._collected_in is not None
                assert self._collected_out is not None

                flatx = self._collected_in[:, collect_in_idx[i]]
                flaty = self._collected_out[:, collect_out_idx[i]]

                pdata.metadata['collection_point_nodespec'] = collection_point_nodespec
                pdata.metadata['input_shapes'] = input_shapes
                pdata.metadata['output_shapes'] = output_shapes

                if self.verbose:
                    logger.debug(
                        f"collection point {i}: {collection_point_nodespec}\n"
                        f"  in_shape={self._collected_in.shape}, out_shape={self._collected_out.shape}\n"
                        f"  in_idx={collect_in_idx[i]}, out_idx={collect_out_idx[i]}\n"
                        f"  node_input_shapes={input_shapes}, node_output_shapes={output_shapes}\n"
                        f"  flatx={flatx.shape}, flaty={flaty.shape}"
                    )

                return flatx, flaty

            output_name = f"Node {collection_point.network_id}:{collection_point.node_id}"

            input_names = [f"In {i}" for i in range(len(self._input_shapes[i]))]

            metadata = self.metadata.copy()

            metadata.update(
                {
                    'datasource_type': 'collection',
                    'seed': self.seed,
                    'model_signature': self.network_model.model.signature,
                    'collection_point_index': i,
                    'network_id': collection_point.network_id,
                    'node_id': collection_point.node_id,
                }
            )

            plot_data = LazyPlotData(
                get_xy=get_xy_fn,
                input_names=input_names,
                output_name=output_name,
                metadata=metadata,
                disable_check_shapes=True,
            )

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} collection plot data objects")
        return plot_data_list

    def _create_network_metadata(self, network_idx: int, network) -> Dict[str, Any]:
        """create metadata dictionary for a network"""
        metadata = self.metadata.copy()

        from biocomp.network import generate_network_info

        network_info = generate_network_info(network)

        metadata.update(
            {
                'datasource_type': 'prediction',
                'seed': self.seed,
                'network': network.model_dump(),
                'model_signature': self.network_model.model.signature,
                'network_name': getattr(network, 'name', f"Network_{network_idx}"),
                'network_info': network_info,
                'network_index': network_idx,
                'prediction_settings': {
                    'use_output_as_input': self.use_output_as_input,
                    'z_value': self.z_value,
                    'disable_variational': self.disable_variational,
                    'max_evals': self.max_evals,
                    'n_predictions': len(self.predict_at[network_idx]),
                },
            }
        )

        if self.per_prediction_info is not None:
            assert len(self.per_prediction_info) >= network_idx
            metadata['extra_prediction_info'] = self.per_prediction_info[network_idx]
            if 'network_name' in metadata['extra_prediction_info']:
                assert (
                    metadata['extra_prediction_info']['network_name'] == metadata['network_name']
                ), (
                    f"network name mismatch: {metadata['extra_prediction_info']['network_name']} != {metadata['network_name']}"
                )

        return metadata

    def _extract_plot_data(
        self,
        network_idx: int,
        network,
        input_order: Optional[List[InputOrderElement]],
        metadata: Dict[str, Any],
        rescale_latent: bool = False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> LazyPlotData:
        """extract plot data from a network"""
        import biocomp.plotutils as pu

        # create function to get data for this network
        get_xy_fn = self._create_xy_function(
            network_idx,
            rescale_latent=rescale_latent,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
        )

        # extract plot data
        plot_data = pu.extract_lazy_plot_data_from_network(
            network,
            get_xy_fn,
            input_order=input_order,
            metadata=metadata,
        )

        logger.debug(
            f"extracted plot data for network {network_idx}: {plot_data.input_names=}, {plot_data.output_name=}"
        )

        return plot_data

    def get_data(
        self,
        rescale_latent=False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[PlotData]:
        """Get fully-evaluated PlotData for each network.

        Args:
            rescale_latent: Only applies when already_latent=True. If True, converts
                output (X, Y) from latent space to raw space using rescaler.inv().
                Has no effect when already_latent=False (output is already raw).
            with_shared_params: Optional override for model shared parameters.
            with_local_params: Optional override for model local parameters.

        Returns:
            List of PlotData objects, one per network. Output space depends on settings:
            - already_latent=False: (X_raw, Y_raw)
            - already_latent=True, rescale_latent=False: (X_latent, Y_latent)
            - already_latent=True, rescale_latent=True: (X_raw, Y_raw)
        """
        lazy_data = self.get_data_lazy(
            rescale_latent=rescale_latent,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
        )

        # evaluate all plot data
        for data in lazy_data:
            data.set_xy()

        return lazy_data

    def save_results(
        self,
        output_dir: Path | str,
        prediction_data: list[PlotData] | None = None,
        ground_truth: list[PlotData] | None = None,
    ) -> list[Path]:
        """Save per-network prediction results as Parquet files with metadata.

        Saves processed PlotData (from get_data()) which has correct column semantics.
        Ground truth saved separately (may differ in row count).

        Args:
            output_dir: Directory to save Parquet files.
            ground_truth: Optional ground truth PlotData list (for embedding in metadata).

        Returns:
            List of saved file paths.
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prediction_data is None:
            prediction_data = self.get_data()

        stats = self.get_network_stats() if self._yhats is not None else [{}] * len(prediction_data)
        saved = []

        for i, pred in enumerate(prediction_data):
            x_pred = np.asarray(pred.x)
            y_pred = np.asarray(pred.y)

            pred_cols = {f"x_{j}": x_pred[:, j] for j in range(x_pred.shape[1])}
            pred_cols.update({f"y_pred_{j}": y_pred[:, j] for j in range(y_pred.shape[1])})
            df = pd.DataFrame(pred_cols)
            table = pa.Table.from_pandas(df)

            network_name = pred.metadata.get('network_name', f'network_{i}')
            meta = {
                'network_name': network_name,
                'input_names': pred.input_names,
                'output_name': pred.output_name,
                'model_signature': self.network_model.signature,
                'stats': stats[i] if i < len(stats) else {},
            }
            import json
            existing_meta = table.schema.metadata or {}
            existing_meta[b'biocomp'] = json.dumps(meta, default=str).encode()
            table = table.replace_schema_metadata(existing_meta)

            safe_name = network_name.replace('/', '_').replace(' ', '_')
            fpath = output_dir / f"{safe_name}.parquet"
            pq.write_table(table, fpath)
            saved.append(fpath)

            if ground_truth is not None and i < len(ground_truth):
                gt = ground_truth[i]
                gt_x, gt_y = np.asarray(gt.x), np.asarray(gt.y)
                gt_cols = {f"x_{j}": gt_x[:, j] for j in range(gt_x.shape[1])}
                gt_cols.update({f"y_gt_{j}": gt_y[:, j] for j in range(gt_y.shape[1])})
                gt_table = pa.Table.from_pandas(pd.DataFrame(gt_cols))
                gt_meta_dict = {**meta, 'data_type': 'ground_truth', 'input_names': gt.input_names, 'output_name': gt.output_name}
                gt_existing = gt_table.schema.metadata or {}
                gt_existing[b'biocomp'] = json.dumps(gt_meta_dict, default=str).encode()
                gt_table = gt_table.replace_schema_metadata(gt_existing)
                pq.write_table(gt_table, output_dir / f"{safe_name}_gt.parquet")

        logger.info(f"Saved {len(saved)} prediction results to {output_dir}")
        return saved

    @staticmethod
    def load_results(output_dir: Path | str) -> list[tuple[PlotData, PlotData]]:
        """Load saved prediction results as (ground_truth, prediction) PlotData pairs.

        No model or GPU needed - pure file I/O. Ready for the auto-dispatch plot system.

        Returns:
            List of (gt_plotdata, pred_plotdata) tuples, one per network.
        """
        import pyarrow.parquet as pq
        import json

        output_dir = Path(output_dir)
        pairs = []

        for fpath in sorted(output_dir.glob("*.parquet")):
            if fpath.stem.endswith('_gt'):
                continue  # skip GT files, loaded alongside pred files

            table = pq.read_table(fpath)
            meta_bytes = table.schema.metadata.get(b'biocomp')
            meta = json.loads(meta_bytes.decode()) if meta_bytes else {}

            df = table.to_pandas()
            x_cols = sorted([c for c in df.columns if c.startswith('x_')])
            y_pred_cols = sorted([c for c in df.columns if c.startswith('y_pred_')])

            x_pred = df[x_cols].to_numpy()
            y_pred = df[y_pred_cols].to_numpy()

            n_outputs = len(y_pred_cols)
            input_names = meta.get('input_names', [f'x_{i}' for i in range(len(x_cols))])
            output_name = meta.get('output_name', 'output')
            if isinstance(output_name, str) and n_outputs > 1:
                output_name = [output_name] + [f'output_{j}' for j in range(1, n_outputs)]
            stats = meta.get('stats', {})
            network_name = meta.get('network_name', fpath.stem)

            pred_metadata = {
                'network_name': network_name,
                'model_signature': meta.get('model_signature', ''),
                'prediction_stats': stats,
            }
            pred_data = PlotData(
                xval=x_pred, yval=y_pred,
                input_names=input_names, output_name=output_name,
                metadata=pred_metadata,
            )

            # Load separate GT file if it exists
            gt_fpath = fpath.parent / f"{fpath.stem}_gt.parquet"
            if gt_fpath.exists():
                gt_df = pq.read_table(gt_fpath).to_pandas()
                gt_x_cols = sorted([c for c in gt_df.columns if c.startswith('x_')])
                gt_y_cols = sorted([c for c in gt_df.columns if c.startswith('y_gt_')])
                gt_data = PlotData(
                    xval=gt_df[gt_x_cols].to_numpy(),
                    yval=gt_df[gt_y_cols].to_numpy(),
                    input_names=input_names, output_name=output_name,
                    metadata={'network_name': network_name},
                )
            else:
                gt_data = PlotData(
                    xval=x_pred, yval=np.full_like(y_pred, np.nan),
                    input_names=input_names, output_name=output_name,
                    metadata={'network_name': network_name},
                )

            pairs.append((gt_data, pred_data))

        return pairs


# 7.11 s
