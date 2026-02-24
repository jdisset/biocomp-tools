"""Tests for MeasuredVsPredictedData data holder."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData


def _make_mock_prediction(n_points: int = 200, n_outputs: int = 1, seed: int = 0):
    """Create a mock NetworkPrediction-like object with _yhats and _gtruths."""
    rng = np.random.default_rng(seed)
    pred = MagicMock()
    pred._yhats = [rng.uniform(0, 1, (n_points, n_outputs)).astype(np.float32)]
    pred._gtruths = [rng.uniform(0, 1, (n_points, n_outputs)).astype(np.float32)]

    network = MagicMock()
    network.get_inverted_input_proteins.return_value = ["input1"]
    network.get_output_proteins.return_value = ["output1"]
    pred.network_model = MagicMock()
    pred.network_model.network = [network]
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
