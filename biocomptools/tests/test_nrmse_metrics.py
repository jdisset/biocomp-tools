"""
Tests for the normalized RMSE (nRMSE) and SNR metrics in networkprediction.py.

These tests verify that the nRMSE metric:
1. Equals ~0 when prediction matches ground truth
2. Scales properly with prediction error relative to local variance
3. Is invariant to global noise scale (fair cross-experiment comparison)
4. Doesn't explode for flat-line data (robust epsilon)
"""

import numpy as np
import pytest

from biocomptools.toollib.networkprediction import _calculate_grid_stats


@pytest.fixture
def default_params():
    return {
        'hypercube_res': 20,
        'hypercube_min': 0.0,
        'hypercube_max': 1.0,
        'k': 50,
        'radius': 0.15,
        'min_points': 10,
    }


def test_nrmse_perfect_prediction(default_params):
    """nRMSE should be ~0 when prediction equals ground truth."""
    np.random.seed(42)
    n = 2000
    x = np.random.rand(n, 2)
    gt = np.sin(2 * np.pi * x[:, 0:1]) + 0.1 * np.random.randn(n, 1)
    yhat = gt.copy()  # Perfect prediction

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    assert stats['grid_nrmse'] < 0.1  # Near zero (not exact due to KNN smoothing)
    assert stats['grid_rmse'] < 0.01
    assert np.isfinite(stats['grid_snr'])


def test_nrmse_scales_with_prediction_bias(default_params):
    """nRMSE should increase when prediction has systematic bias relative to noise."""
    np.random.seed(42)
    n = 5000
    x = np.random.rand(n, 2)

    # Ground truth with known local variance
    gt_mean = np.sin(2 * np.pi * x[:, 0:1])
    noise_std = 0.1
    gt = gt_mean + np.random.randn(n, 1) * noise_std

    # Small bias prediction (bias = 0.5 * noise)
    yhat_small_bias = gt_mean + 0.05  # constant bias of half noise_std

    # Large bias prediction (bias = 2 * noise)
    yhat_large_bias = gt_mean + 0.2  # constant bias of 2x noise_std

    stats_small = _calculate_grid_stats(yhat_small_bias, gt, x, default_params)
    stats_large = _calculate_grid_stats(yhat_large_bias, gt, x, default_params)

    # Larger bias should give larger nRMSE
    assert stats_large['grid_nrmse'] > stats_small['grid_nrmse']
    # Both should be finite and positive
    assert stats_small['grid_nrmse'] > 0
    assert np.isfinite(stats_large['grid_nrmse'])


def test_nrmse_invariant_to_noise_scale(default_params):
    """nRMSE should be similar for experiments with different noise levels when predicting local mean."""
    np.random.seed(42)
    n = 5000
    x = np.random.rand(n, 2)
    gt_mean = np.sin(2 * np.pi * x[:, 0:1])

    # Experiment A: low noise
    noise_a = 0.01
    gt_a = gt_mean + np.random.randn(n, 1) * noise_a
    yhat_a = gt_mean  # Perfect mean prediction

    # Experiment B: high noise (10x)
    noise_b = 0.1
    np.random.seed(43)  # different seed for different noise realization
    gt_b = gt_mean + np.random.randn(n, 1) * noise_b
    yhat_b = gt_mean  # Perfect mean prediction

    stats_a = _calculate_grid_stats(yhat_a, gt_a, x, default_params)
    stats_b = _calculate_grid_stats(yhat_b, gt_b, x, default_params)

    # grid_rmse differs significantly (noise difference)
    assert stats_b['grid_rmse'] > 3 * stats_a['grid_rmse']

    # nRMSE should be similar (both predict local mean well)
    # Allow for some variance due to sampling
    assert abs(stats_a['grid_nrmse'] - stats_b['grid_nrmse']) < 0.5


def test_robust_epsilon_prevents_explosion(default_params):
    """nRMSE should not explode for flat-line data (zero local variance)."""
    n = 1000
    x = np.random.rand(n, 2)

    # Constant ground truth (zero variance)
    gt = np.ones((n, 1)) * 0.5  # Use latent space range [0,1]
    yhat = np.ones((n, 1)) * 0.5001  # Tiny error (0.1% of scale)

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    # Should not be infinite or huge
    # With robust epsilon of 0.01 (1% of [0,1] scale), error of 0.0001
    # gives nRMSE = 0.0001/0.01 = 0.01
    assert np.isfinite(stats['grid_nrmse'])
    assert stats['grid_nrmse'] < 1.0  # Should be small, not explosive


def test_snr_reasonable_range(default_params):
    """SNR should be in reasonable dB range for typical data."""
    np.random.seed(42)
    n = 5000
    x = np.random.rand(n, 2)
    gt_mean = np.sin(2 * np.pi * x[:, 0:1])
    gt = gt_mean + np.random.randn(n, 1) * 0.1
    yhat = gt_mean

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    # SNR should be positive (signal > noise) and finite
    assert np.isfinite(stats['grid_snr'])
    assert -20 < stats['grid_snr'] < 40  # Reasonable dB range


def test_snr_increases_with_signal(default_params):
    """SNR should increase when signal variance increases relative to noise."""
    np.random.seed(42)
    n = 5000
    x = np.random.rand(n, 2)
    noise_std = 0.1

    # Low signal amplitude
    gt_low = 0.1 * np.sin(2 * np.pi * x[:, 0:1]) + np.random.randn(n, 1) * noise_std
    yhat_low = 0.1 * np.sin(2 * np.pi * x[:, 0:1])

    # High signal amplitude
    gt_high = 1.0 * np.sin(2 * np.pi * x[:, 0:1]) + np.random.randn(n, 1) * noise_std
    yhat_high = 1.0 * np.sin(2 * np.pi * x[:, 0:1])

    stats_low = _calculate_grid_stats(yhat_low, gt_low, x, default_params)
    stats_high = _calculate_grid_stats(yhat_high, gt_high, x, default_params)

    # Higher signal amplitude should give higher SNR
    assert stats_high['grid_snr'] > stats_low['grid_snr']


def test_nrmse_worse_than_mean(default_params):
    """nRMSE should be > 1 when model is worse than predicting local mean."""
    np.random.seed(42)
    n = 3000
    x = np.random.rand(n, 2)

    # Ground truth with clear pattern
    gt_mean = np.sin(2 * np.pi * x[:, 0:1])
    noise_std = 0.05
    gt = gt_mean + np.random.randn(n, 1) * noise_std

    # Bad prediction - predicts global mean (ignores pattern)
    yhat = np.ones((n, 1)) * np.mean(gt)

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    # Model ignoring pattern should have nRMSE > 1
    assert stats['grid_nrmse'] > 1.0


def test_all_metrics_present(default_params):
    """All expected metrics should be present in the output."""
    np.random.seed(42)
    n = 1000
    x = np.random.rand(n, 2)
    gt = np.random.randn(n, 1)
    yhat = np.random.randn(n, 1)

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    # Check all expected keys are present
    expected_keys = [
        'grid_gt_var',
        'grid_mse',
        'grid_rmse',
        'grid_nrmse',
        'grid_snr',
        'grid_kl',
        'grid_kl_similarity',
        'grid_r_squared',
    ]
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"
        assert np.isfinite(stats[key]), f"Non-finite value for {key}: {stats[key]}"


def test_1d_inputs(default_params):
    """Should handle 1D input arrays correctly."""
    np.random.seed(42)
    n = 1000
    x = np.random.rand(n, 1)
    gt = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n, 1)
    yhat = np.sin(2 * np.pi * x)

    params_1d = {**default_params, 'hypercube_res': 30}  # 1D needs fewer points
    stats = _calculate_grid_stats(yhat, gt, x, params_1d)

    assert np.isfinite(stats['grid_nrmse'])
    assert np.isfinite(stats['grid_snr'])


def test_3d_inputs(default_params):
    """Should handle 3D input arrays correctly."""
    np.random.seed(42)
    n = 3000  # Need more points for 3D
    x = np.random.rand(n, 3)
    gt = np.sin(2 * np.pi * x[:, 0:1]) + 0.1 * np.random.randn(n, 1)
    yhat = np.sin(2 * np.pi * x[:, 0:1])

    params_3d = {**default_params, 'hypercube_res': 10}  # Lower res for 3D
    stats = _calculate_grid_stats(yhat, gt, x, params_3d)

    assert np.isfinite(stats['grid_nrmse'])
    assert np.isfinite(stats['grid_snr'])


def test_nan_handling(default_params):
    """Should handle NaN values in inputs gracefully."""
    np.random.seed(42)
    n = 1000
    x = np.random.rand(n, 2)
    gt = np.sin(2 * np.pi * x[:, 0:1]) + 0.1 * np.random.randn(n, 1)
    yhat = np.sin(2 * np.pi * x[:, 0:1])

    # Introduce some NaNs in x
    x[0, 0] = np.nan
    x[10, 1] = np.nan

    stats = _calculate_grid_stats(yhat, gt, x, default_params)

    # Should still produce finite results
    assert np.isfinite(stats['grid_nrmse'])
    assert np.isfinite(stats['grid_snr'])
