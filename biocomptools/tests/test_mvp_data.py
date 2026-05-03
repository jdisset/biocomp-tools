"""Tests for MeasuredVsPredictedData data holder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData


def _make_mock_prediction(
    n_points: int = 200,
    n_outputs: int = 1,
    seed: int = 0,
    stats_list: list | None = None,
):
    """Create a mock NetworkPrediction-like object with _yhats and _gtruths.

    ``stats_list`` is what ``pred.get_network_stats()`` returns. Pass a list of
    dicts to exercise the ``subsample_indices`` consumption path.
    """
    rng = np.random.default_rng(seed)
    pred = MagicMock()
    pred._yhats = [rng.uniform(0, 1, (n_points, n_outputs)).astype(np.float32)]
    pred._gtruths = [rng.uniform(0, 1, (n_points, n_outputs)).astype(np.float32)]

    network = MagicMock()
    network.get_inverted_input_proteins.return_value = ["input1"]
    network.get_output_proteins.return_value = ["output1"]
    pred.network_model = MagicMock()
    pred.network_model.network = [network]
    pred.get_network_stats = MagicMock(return_value=stats_list or [])
    return pred


def test_raw_mode_shapes():
    pred = _make_mock_prediction(n_points=100)
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (100,)
    assert data.predicted.shape == (100,)


def test_raw_mode_subsampling():
    pred = _make_mock_prediction(n_points=1000)
    data = MeasuredVsPredictedData(
        predictions=[pred],
        resample_per_experiment=200,
        dependent_output_only=False,
    )
    assert data.measured.shape == (200,)
    assert data.predicted.shape == (200,)


def test_multiple_predictions():
    pred1 = _make_mock_prediction(n_points=100, seed=0)
    pred2 = _make_mock_prediction(n_points=150, seed=1)
    data = MeasuredVsPredictedData(predictions=[pred1, pred2], dependent_output_only=False)
    assert data.measured.shape == (250,)


def test_multi_output_flatten():
    pred = _make_mock_prediction(n_points=50, n_outputs=3)
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (150,)  # 50 * 3


def test_no_ground_truth_skipped():
    pred = MagicMock()
    pred._yhats = [np.ones((10, 1), dtype=np.float32)]
    pred._gtruths = [None]
    network = MagicMock()
    pred.network_model = MagicMock()
    pred.network_model.network = [network]

    pred2 = _make_mock_prediction(n_points=20)
    data = MeasuredVsPredictedData(predictions=[pred, pred2], dependent_output_only=False)
    assert data.measured.shape == (20,)


def test_compute_triggered_if_yhats_none():
    pred = _make_mock_prediction(n_points=50)
    pred._yhats = None
    pred.compute_all_network_predictions = MagicMock(
        side_effect=lambda: setattr(pred, "_yhats", [np.ones((50, 1), dtype=np.float32)])
    )
    MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    pred.compute_all_network_predictions.assert_called_once()


def test_nan_filtered():
    pred = _make_mock_prediction(n_points=10)
    pred._gtruths[0][0, 0] = np.nan
    pred._yhats[0][1, 0] = np.nan
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (8,)
    assert np.all(np.isfinite(data.measured))
    assert np.all(np.isfinite(data.predicted))


# ── subsample_indices SSOT integration ────────────────────────────────────


def test_mvp_uses_subsample_indices_from_stats():
    """Cloud must equal gt[subsample_indices].ravel() (SSOT with the metric)."""
    pred = _make_mock_prediction(n_points=300)
    sub_idx = np.array([5, 17, 42, 99, 200, 250], dtype=np.intp)
    pred.get_network_stats.return_value = [{'subsample_indices': sub_idx}]

    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    expected_m = pred._gtruths[0][sub_idx].ravel()
    expected_p = pred._yhats[0][sub_idx].ravel()
    np.testing.assert_array_equal(data.measured, expected_m)
    np.testing.assert_array_equal(data.predicted, expected_p)


def test_mvp_subsample_indices_with_duplicates_keep_multiplicity():
    """With-replacement indices appear N times in the cloud (matching the metric)."""
    pred = _make_mock_prediction(n_points=50)
    sub_idx = np.array([3, 3, 3, 7, 7], dtype=np.intp)
    pred.get_network_stats.return_value = [{'subsample_indices': sub_idx}]
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (5,)
    np.testing.assert_array_equal(data.measured, pred._gtruths[0][sub_idx].ravel())


def test_mvp_subsample_indices_multi_output_ravel_order():
    """For multi-output: cloud == gt[idx, :].ravel() in row-major order."""
    pred = _make_mock_prediction(n_points=20, n_outputs=3)
    sub_idx = np.array([0, 5, 10], dtype=np.intp)
    pred.get_network_stats.return_value = [{'subsample_indices': sub_idx}]
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (9,)  # 3 rows × 3 outs
    np.testing.assert_array_equal(data.measured, pred._gtruths[0][sub_idx].ravel())


def test_mvp_falls_back_when_stats_empty():
    """No stats → original random-sampling path (existing behavior)."""
    pred = _make_mock_prediction(n_points=1000)
    pred.get_network_stats.return_value = []
    data = MeasuredVsPredictedData(
        predictions=[pred],
        resample_per_experiment=200,
        dependent_output_only=False,
    )
    assert data.measured.shape == (200,)


def test_mvp_falls_back_when_subsample_indices_missing():
    """Stats present but no subsample_indices → fallback to random sampling."""
    pred = _make_mock_prediction(n_points=1000)
    pred.get_network_stats.return_value = [{'grid_nrmse': 0.1}]  # no subsample_indices
    data = MeasuredVsPredictedData(
        predictions=[pred],
        resample_per_experiment=300,
        dependent_output_only=False,
    )
    assert data.measured.shape == (300,)


def test_mvp_subsample_indices_filters_nan_after_subsample():
    """NaN at sub-rows are filtered, others retained."""
    pred = _make_mock_prediction(n_points=20)
    pred._gtruths[0][3, 0] = np.nan
    sub_idx = np.array([1, 3, 5, 7], dtype=np.intp)
    pred.get_network_stats.return_value = [{'subsample_indices': sub_idx}]
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)
    assert data.measured.shape == (3,)  # row 3 dropped
    assert np.all(np.isfinite(data.measured))


def test_mvp_uses_real_calculate_grid_stats_indices():
    """End-to-end: feed `_calculate_grid_stats` output dict straight into the MVP
    holder; the resulting cloud must equal gt[indices]/yhat[indices] (post-NaN-filter)."""
    from biocomptools.toollib.networkprediction import _calculate_grid_stats

    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 0.7, (n, 2)).astype(np.float32)
    gt = rng.uniform(0, 1, (n, 1)).astype(np.float32)
    yhat = gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)
    params = dict(
        hypercube_res=16, hypercube_min=0.0, hypercube_max=0.7,
        k=32, radius=0.1, min_points=10,
        subsample_n=1000, subsample_knn_k=32, subsample_density_quantile=0.025,
    )
    stats = _calculate_grid_stats(yhat, gt, x, params)
    sub_idx = stats['subsample_indices']
    assert sub_idx.size == 1000

    pred = _make_mock_prediction(n_points=n, stats_list=[stats])
    pred._gtruths[0] = gt
    pred._yhats[0] = yhat
    data = MeasuredVsPredictedData(predictions=[pred], dependent_output_only=False)

    expected = gt[sub_idx].ravel()
    expected_p = yhat[sub_idx].ravel()
    np.testing.assert_array_equal(data.measured, expected)
    np.testing.assert_array_equal(data.predicted, expected_p)
