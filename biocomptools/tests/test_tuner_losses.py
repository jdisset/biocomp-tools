"""Test that tuner and design mode compute identical losses."""

import jax
import jax.numpy as jnp
import pytest

from biocomp.designloss import GridLossResult, compute_grid_losses


@pytest.fixture
def sample_grids():
    """Generate sample prediction and target grids."""
    key = jax.random.key(42)
    Y_target = jax.random.uniform(key, (32, 32))
    k1, k2 = jax.random.split(key)
    Y_pred = Y_target + 0.1 * jax.random.normal(k2, (32, 32))
    return Y_pred, Y_target


def test_compute_grid_losses_returns_correct_type(sample_grids):
    """Test that compute_grid_losses returns a GridLossResult."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target)
    assert isinstance(result, GridLossResult)


def test_compute_grid_losses_basic(sample_grids):
    """Test basic loss computation has valid values."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target)

    assert result.total >= 0
    assert result.sinkhorn >= 0
    assert result.lncc >= 0
    assert result.mse >= 0


def test_compute_grid_losses_weights(sample_grids):
    """Test that weights affect loss computation correctly."""
    Y_pred, Y_target = sample_grids

    result_sinkhorn = compute_grid_losses(Y_pred, Y_target, w_sinkhorn=1.0, w_lncc=0.0, w_mse=0.0)
    result_lncc = compute_grid_losses(Y_pred, Y_target, w_sinkhorn=0.0, w_lncc=1.0, w_mse=0.0)

    assert abs(result_sinkhorn.total - result_sinkhorn.sinkhorn) < 1e-5
    assert abs(result_lncc.total - result_lncc.lncc) < 1e-5


def test_compute_grid_losses_zero_weight_returns_zero():
    """Test that zero-weighted losses are actually zero."""
    key = jax.random.key(123)
    Y_pred = jax.random.uniform(key, (16, 16))
    Y_target = jax.random.uniform(jax.random.split(key)[0], (16, 16))

    result = compute_grid_losses(
        Y_pred, Y_target, w_sinkhorn=0.0, w_lncc=0.0, w_mse=0.0, w_spectral=0.0
    )

    assert result.sinkhorn == 0.0
    assert result.lncc == 0.0
    assert result.mse == 0.0
    assert result.spectral == 0.0
    assert result.total == 0.0


def test_compute_grid_losses_identical_grids():
    """Test that identical grids have low loss."""
    key = jax.random.key(456)
    Y = jax.random.uniform(key, (32, 32))

    result = compute_grid_losses(Y, Y)

    assert result.mse < 1e-10
    assert result.lncc < 0.01


def test_compute_grid_losses_to_dict(sample_grids):
    """Test that to_dict returns expected keys."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target)
    d = result.to_dict()

    assert "total" in d
    assert "sinkhorn" in d
    assert "lncc" in d
    assert "mse" in d
    assert "spectral" in d


def test_compute_grid_losses_with_contributions(sample_grids):
    """Test that contributions are computed when requested."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target, return_contributions=True)

    assert result.lncc_contrib is not None
    assert result.lncc_contrib.shape == Y_pred.shape


def test_compute_grid_losses_shape_mismatch_raises():
    """Test that shape mismatch raises assertion error."""
    Y_pred = jnp.zeros((32, 32))
    Y_target = jnp.zeros((16, 16))

    with pytest.raises(AssertionError, match="Shape mismatch"):
        compute_grid_losses(Y_pred, Y_target)


def test_compute_grid_losses_wrong_dim_raises():
    """Test that non-2D input raises assertion error."""
    Y = jnp.zeros((32,))

    with pytest.raises(AssertionError, match="Expected 2D"):
        compute_grid_losses(Y, Y)


def test_compute_grid_losses_mse_nonzero_when_weighted(sample_grids):
    """Test that MSE is non-zero when w_mse > 0 and grids differ."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target, w_mse=1.0)
    assert result.mse > 0, f"MSE should be > 0 for different grids, got {result.mse}"
    expected_mse = float(jnp.mean((Y_pred - Y_target) ** 2))
    assert abs(result.mse - expected_mse) < 1e-5, f"MSE mismatch: {result.mse} vs {expected_mse}"


def test_compute_grid_losses_simse_nonzero_when_weighted(sample_grids):
    """Test that SIMSE is non-zero when w_simse > 0 and grids differ."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target, w_simse=1.0)
    assert result.simse > 0, f"SIMSE should be > 0 for different grids, got {result.simse}"


def test_compute_grid_losses_to_dict_includes_simse(sample_grids):
    """Test that to_dict includes simse."""
    Y_pred, Y_target = sample_grids
    result = compute_grid_losses(Y_pred, Y_target, w_simse=1.0)
    d = result.to_dict()
    assert "simse" in d
    assert d["simse"] == result.simse
