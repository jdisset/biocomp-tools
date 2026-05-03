"""Tests for the grid-stats helpers: kernel lattice interp, smoother, and the
density-balanced subsample exposed in `_calculate_grid_stats`."""

from __future__ import annotations

import numpy as np
import pytest

from biocomptools.toollib.networkprediction import (
    _calculate_grid_stats,
    _kernel_smoother_lattice,
    kernel_lattice_interp,
    make_hypercube,
)


@pytest.fixture
def synthetic_2d():
    """Smooth 2D regression problem with mild observation noise."""
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.uniform(0.0, 0.7, (n, 2)).astype(np.float32)
    y_true = (np.sin(5 * x[:, 0]) + np.cos(5 * x[:, 1])).reshape(-1, 1).astype(np.float32) * 0.3 + 0.4
    yhat = (y_true + rng.normal(0, 0.05, y_true.shape).astype(np.float32))
    gt = (y_true + rng.normal(0, 0.04, y_true.shape).astype(np.float32))
    return x, gt, yhat


@pytest.fixture
def gridstats_params():
    return dict(
        hypercube_res=24, hypercube_min=0.0, hypercube_max=0.7,
        k=64, radius=0.1, min_points=10,
        subsample_n=2000, subsample_knn_k=32, subsample_density_quantile=0.025,
    )


def test_make_hypercube_ij_indexing_reshapes_naturally():
    """Flatten → reshape with ``indexing='ij'`` recovers axis structure."""
    grid = make_hypercube(2, res=5, xmin=0.0, xmax=1.0)
    assert grid.shape == (25, 2)
    reshaped = grid.reshape(5, 5, 2)
    # Along axis 0: x-coord changes, y-coord constant.
    np.testing.assert_allclose(reshaped[:, 0, 0], np.linspace(0, 1, 5))
    np.testing.assert_allclose(reshaped[0, :, 1], np.linspace(0, 1, 5))


def test_kernel_lattice_interp_recovers_lattice_values():
    """Linear interpolator must return exact lattice values at lattice points."""
    res = 8
    grid = make_hypercube(2, res=res, xmin=0.0, xmax=0.7)
    n_outs = 1
    grid_means = (grid[:, 0:1] + 2 * grid[:, 1:2]).astype(np.float64).reshape(-1, n_outs)
    interp = kernel_lattice_interp(
        grid_means,
        params=dict(hypercube_res=res, hypercube_min=0.0, hypercube_max=0.7),
        ndim=2,
    )
    out = np.asarray(interp(grid))
    np.testing.assert_allclose(out.ravel(), grid_means.ravel(), rtol=1e-6)


def test_kernel_lattice_interp_extrapolates_to_nan():
    res = 6
    grid_means = np.zeros(res * res * 1).reshape(-1, 1)
    interp = kernel_lattice_interp(
        grid_means,
        params=dict(hypercube_res=res, hypercube_min=0.0, hypercube_max=0.7),
        ndim=2,
    )
    out = np.asarray(interp(np.array([[1.5, 1.5]])))
    assert np.isnan(out).all()


def test_kernel_smoother_lattice_returns_consistent_shapes(synthetic_2d, gridstats_params):
    x, gt, _ = synthetic_2d
    grid = make_hypercube(2, res=gridstats_params['hypercube_res'],
                            xmin=0.0, xmax=0.7)
    mean, stdev, n_eff, iw = _kernel_smoother_lattice(x, gt, grid, gridstats_params)
    n_grid = grid.shape[0]
    assert mean.shape == (n_grid, 1)
    assert stdev.shape == (n_grid, 1)
    assert n_eff.shape == (n_grid, 1)
    indices, norm_weights = iw
    assert indices.shape[0] == n_grid
    assert norm_weights.shape == indices.shape


def test_calculate_grid_stats_returns_finite_metrics(synthetic_2d, gridstats_params):
    x, gt, yhat = synthetic_2d
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    for key in (
        'grid_nrmse', 'grid_rmse', 'grid_snr',
        'kernel_rmse_latent', 'model_rmse_latent', 'ratio_rmse',
    ):
        assert np.isfinite(stats[key]), f"{key} should be finite, got {stats[key]}"
    assert stats['kernel_rmse_latent'] > 0
    assert stats['model_rmse_latent'] > 0
    # In this synthetic setup the model adds extra noise on top of gt, so
    # the model should be no better than the kernel.
    assert stats['ratio_rmse'] >= 1.0 - 1e-6


def test_calculate_grid_stats_subsample_indices_in_range(synthetic_2d, gridstats_params):
    x, gt, yhat = synthetic_2d
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    idx = stats['subsample_indices']
    assert idx.dtype == np.intp
    assert idx.size == stats['n_subsample_used']
    assert idx.size > 0
    assert (idx >= 0).all() and (idx < x.shape[0]).all()


def test_calculate_grid_stats_indices_skip_invalid_x(gridstats_params):
    """Rows with non-finite X must never appear in subsample_indices."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 0.7, (n, 2)).astype(np.float32)
    x[:50] = np.nan  # invalidate first 50 rows
    gt = rng.uniform(0, 1, (n, 1)).astype(np.float32)
    yhat = gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    idx = stats['subsample_indices']
    assert idx.size > 0
    assert (idx >= 50).all(), "subsample_indices should skip the invalid prefix"


def test_calculate_grid_stats_subsample_size_matches_param(synthetic_2d, gridstats_params):
    """When all data is finite, subsample size matches the configured n."""
    x, gt, yhat = synthetic_2d
    params = {**gridstats_params, 'subsample_n': 1500}
    stats = _calculate_grid_stats(yhat, gt, x, params)
    assert stats['n_subsample_used'] == 1500


def test_calculate_grid_stats_excess_zero_for_perfect_model(gridstats_params):
    """A model that exactly matches gt must have model_rmse == 0 (excess ≈ 0)."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0, 0.7, (n, 2)).astype(np.float32)
    gt = rng.uniform(0, 1, (n, 1)).astype(np.float32)
    yhat = gt.copy()  # perfect model
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    assert stats['model_rmse_latent'] == pytest.approx(0.0, abs=1e-7)
    assert stats['ratio_rmse'] == pytest.approx(0.0, abs=1e-7)
    assert stats['model_r_squared_latent'] == pytest.approx(1.0, abs=1e-6)
    assert stats['model_nrmse_local'] == pytest.approx(0.0, abs=1e-6)
    assert stats['kratio'] == pytest.approx(0.0, abs=1e-6)


def test_calculate_grid_stats_r_squared_consistent_with_rmse(synthetic_2d, gridstats_params):
    """R² and RMSE must be consistent: R² = 1 − RMSE² / var(gt)."""
    x, gt, yhat = synthetic_2d
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    sub = stats['subsample_indices']
    assert sub.size > 0

    # Reconstruct expected R² from RMSE + variance over the subsample.
    valid_mask = np.all(np.isfinite(x), axis=1)
    gt_sub = gt[valid_mask][np.searchsorted(np.where(valid_mask)[0], sub)]
    var_gt_sub = float(np.var(gt_sub))
    expected_kernel_r2 = 1.0 - stats['kernel_rmse_latent'] ** 2 / var_gt_sub
    expected_model_r2 = 1.0 - stats['model_rmse_latent'] ** 2 / var_gt_sub
    assert stats['kernel_r_squared_latent'] == pytest.approx(expected_kernel_r2, rel=1e-3)
    assert stats['model_r_squared_latent'] == pytest.approx(expected_model_r2, rel=1e-3)


def test_calculate_grid_stats_kernel_at_least_as_good_as_model(synthetic_2d, gridstats_params):
    """Kernel-smoother is the optimal nonparametric → kernel_r² ≥ model_r²."""
    x, gt, yhat = synthetic_2d
    stats = _calculate_grid_stats(yhat, gt, x, gridstats_params)
    assert stats['kernel_r_squared_latent'] >= stats['model_r_squared_latent'] - 1e-6
    assert stats['ratio_rmse'] >= 1.0 - 1e-6
    assert stats['kratio'] >= 1.0 - 1e-6
