"""Tests for DesignHeatmapLogger, specifically the fresh prediction fix."""

import numpy as np


class TestDesignHeatmapLoggerConfig:
    """Tests for DesignHeatmapLogger configuration."""

    def test_default_use_fresh_predictions(self):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger()
        assert logger.use_fresh_predictions is True

    def test_disable_fresh_predictions(self):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(use_fresh_predictions=False)
        assert logger.use_fresh_predictions is False

    def test_default_disallow_stale_final_fallback(self):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger()
        assert logger.allow_stale_final_fallback is False

    def test_compute_fresh_prediction_returns_none_without_model(self):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger()
        result = logger._compute_fresh_prediction(params={}, rep_idx=0, target_idx=0, network_idx=0)
        assert result is None

    def test_compute_fresh_prediction_returns_none_without_dmanager(self):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger()
        logger._model = object()
        result = logger._compute_fresh_prediction(params={}, rep_idx=0, target_idx=0, network_idx=0)
        assert result is None


class TestHelperFunctions:
    """Tests for helper functions in designheatmaplogger module."""

    def test_to_scalar_with_none(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _to_scalar

        assert _to_scalar(None) == 0.0

    def test_to_scalar_with_scalar(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _to_scalar

        assert _to_scalar(3.14) == 3.14

    def test_to_scalar_with_single_element_array(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _to_scalar

        assert _to_scalar(np.array([5.0])) == 5.0

    def test_to_scalar_with_multi_element_array(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _to_scalar

        result = _to_scalar(np.array([1.0, 2.0, 3.0]))
        assert result == 2.0

    def test_extract_at_indices_scalar(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _extract_at_indices

        result = _extract_at_indices(np.array(0.5), rid=0, tid=0, nid=0)
        assert result == 0.5

    def test_extract_at_indices_1d(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _extract_at_indices

        result = _extract_at_indices(np.array([0.1, 0.2, 0.3]), rid=0, tid=0, nid=1)
        assert result == 0.2

    def test_extract_at_indices_2d(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _extract_at_indices

        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = _extract_at_indices(arr, rid=0, tid=1, nid=0)
        assert result == 0.3

    def test_extract_at_indices_3d(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _extract_at_indices

        arr = np.ones((2, 3, 4)) * np.arange(24).reshape(2, 3, 4)
        result = _extract_at_indices(arr, rid=1, tid=2, nid=3)
        expected = arr[1, 2, 3]
        assert result == expected

    def test_extract_at_indices_none(self):
        from biocomptools.toollib.loggers.designheatmaplogger import _extract_at_indices

        result = _extract_at_indices(None, rid=0, tid=0, nid=0)
        assert result == 0.0

    def test_extract_single_dependent_prediction_1d_passthrough(self):
        from biocomptools.toollib.loggers.designheatmaplogger import (
            _extract_single_dependent_prediction,
        )

        y_pred = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = _extract_single_dependent_prediction(y_pred, dependent_mask=None)
        np.testing.assert_allclose(out, y_pred)

    def test_extract_single_dependent_prediction_multichannel_uses_dep_mask(self):
        from biocomptools.toollib.loggers.designheatmaplogger import (
            _extract_single_dependent_prediction,
        )

        y_pred = np.array(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
                [3.0, 30.0, 300.0, 3000.0],
            ],
            dtype=np.float32,
        )
        out = _extract_single_dependent_prediction(
            y_pred,
            dependent_mask=np.array([False, False, False, True]),
        )
        np.testing.assert_allclose(out, np.array([1000.0, 2000.0, 3000.0], dtype=np.float32))

    def test_extract_single_dependent_prediction_raises_on_mask_length_mismatch(self):
        import pytest
        from biocomptools.toollib.loggers.designheatmaplogger import (
            _extract_single_dependent_prediction,
        )

        y_pred = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="output dim mismatch"):
            _extract_single_dependent_prediction(
                y_pred,
                dependent_mask=np.array([True], dtype=bool),
            )

    def test_extract_single_dependent_prediction_raises_on_multiple_dependents(self):
        import pytest
        from biocomptools.toollib.loggers.designheatmaplogger import (
            _extract_single_dependent_prediction,
        )

        y_pred = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected exactly one dependent output"):
            _extract_single_dependent_prediction(
                y_pred,
                dependent_mask=np.array([True, False, True, False], dtype=bool),
            )


class TestFinalRenderBehavior:
    def test_final_render_forces_fresh_prediction_and_updates_header_loss(self, monkeypatch):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(
            use_fresh_predictions=False,
            show_local_params=False,
            show_ratio_stats=False,
        )
        logger._grid_resolution = (2, 2)
        logger._cached_target_grid = np.zeros((2, 2), dtype=np.float32)
        logger._network_names = ["net0"]
        logger._total_steps = 4
        logger._n_targets = 1

        calls = {"fresh": 0}

        def _fake_fresh(*_args, **_kwargs):
            calls["fresh"] += 1
            return np.ones((2, 2), dtype=np.float32) * 0.5

        def _fake_side_by_side(*_args, **_kwargs):
            return "PLOT", {
                "weighted_total": 0.1234,
                "pred_range": (0.5, 0.5),
                "correlation": 0.0,
                "contrast_target_gap": 0.30,
                "contrast_pred_gap": 0.04,
                "pred_q95_q05_gap": 0.05,
            }

        monkeypatch.setattr(logger, "_compute_fresh_prediction", _fake_fresh)
        monkeypatch.setattr(
            "biocomptools.toollib.loggers.designheatmaplogger.side_by_side_txt_plot",
            _fake_side_by_side,
        )

        yhatdep = np.zeros((1, 4, 1, 1), dtype=np.float32)
        step_history = {
            "latest_params": {"dummy": 1},
            "all_losses": np.array([[[0.9]]], dtype=np.float32),
        }

        final_lines = logger._render_single_design(
            step=4,
            step_history=step_history,
            yhatdep=yhatdep,
            tid=0,
            rid=0,
            nid=0,
            rank=0,
            loss=0.9,
            Y_target_grid=np.zeros((2, 2), dtype=np.float32),
            n_total_designs=1,
            is_final=True,
            stack=None,
        )
        assert calls["fresh"] == 1
        assert any("loss=0.1234" in line for line in final_lines)
        assert any("Contrast:" in line for line in final_lines)

        non_final_lines = logger._render_single_design(
            step=3,
            step_history=step_history,
            yhatdep=yhatdep,
            tid=0,
            rid=0,
            nid=0,
            rank=0,
            loss=0.9,
            Y_target_grid=np.zeros((2, 2), dtype=np.float32),
            n_total_designs=1,
            is_final=False,
            stack=None,
        )
        assert calls["fresh"] == 1
        assert any("loss=0.9000" in line for line in non_final_lines)

    def test_render_includes_contrast_diagnostics_when_available(self, monkeypatch):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(show_local_params=False, show_ratio_stats=False)
        logger._grid_resolution = (2, 2)
        logger._cached_target_grid = np.zeros((2, 2), dtype=np.float32)
        logger._network_names = ["net0"]
        logger._total_steps = 4
        logger._n_targets = 1

        monkeypatch.setattr(
            "biocomptools.toollib.loggers.designheatmaplogger.side_by_side_txt_plot",
            lambda *_args, **_kwargs: (
                "PLOT",
                {
                    "weighted_total": 0.1111,
                    "pred_range": (0.0, 0.1),
                    "correlation": 0.5,
                    "contrast_target_gap": 0.30,
                    "contrast_pred_gap": 0.04,
                    "pred_q95_q05_gap": 0.05,
                },
            ),
        )

        lines = logger._render_single_design(
            step=4,
            step_history={
                "latest_params": {"dummy": 1},
                "all_losses": np.array([[[0.9]]], dtype=np.float32),
            },
            yhatdep=np.zeros((1, 4, 1, 1), dtype=np.float32),
            tid=0,
            rid=0,
            nid=0,
            rank=0,
            loss=0.9,
            Y_target_grid=np.zeros((2, 2), dtype=np.float32),
            n_total_designs=1,
            is_final=False,
            stack=None,
        )
        assert any("Contrast: target_gap=0.3000" in line for line in lines)

    def test_final_render_refuses_stale_fallback_when_fresh_prediction_fails(self, monkeypatch):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(
            use_fresh_predictions=False,
            show_local_params=False,
            show_ratio_stats=False,
        )
        logger._grid_resolution = (2, 2)
        logger._cached_target_grid = np.zeros((2, 2), dtype=np.float32)
        logger._network_names = ["net0"]
        logger._total_steps = 4
        logger._n_targets = 1

        monkeypatch.setattr(logger, "_compute_fresh_prediction", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            "biocomptools.toollib.loggers.designheatmaplogger.side_by_side_txt_plot",
            lambda *_args, **_kwargs: (
                "PLOT",
                {"weighted_total": 0.5, "pred_range": (0.0, 1.0), "correlation": 0.0},
            ),
        )

        lines = logger._render_single_design(
            step=4,
            step_history={
                "latest_params": {"dummy": 1},
                "all_losses": np.array([[[0.9]]], dtype=np.float32),
            },
            yhatdep=np.zeros((1, 4, 1, 1), dtype=np.float32),
            tid=0,
            rid=0,
            nid=0,
            rank=0,
            loss=0.9,
            Y_target_grid=np.zeros((2, 2), dtype=np.float32),
            n_total_designs=1,
            is_final=True,
            stack=None,
        )
        assert any("FINAL_RESULT_UNAVAILABLE" in line for line in lines)

    def test_final_render_allows_stale_fallback_when_explicitly_enabled(self, monkeypatch):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(
            use_fresh_predictions=False,
            allow_stale_final_fallback=True,
            show_local_params=False,
            show_ratio_stats=False,
        )
        logger._grid_resolution = (2, 2)
        logger._cached_target_grid = np.zeros((2, 2), dtype=np.float32)
        logger._network_names = ["net0"]
        logger._total_steps = 4
        logger._n_targets = 1

        monkeypatch.setattr(logger, "_compute_fresh_prediction", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            "biocomptools.toollib.loggers.designheatmaplogger.side_by_side_txt_plot",
            lambda *_args, **_kwargs: (
                "PLOT",
                {"weighted_total": 0.5, "pred_range": (0.0, 1.0), "correlation": 0.0},
            ),
        )

        lines = logger._render_single_design(
            step=4,
            step_history={
                "latest_params": {"dummy": 1},
                "all_losses": np.array([[[0.9]]], dtype=np.float32),
            },
            yhatdep=np.zeros((1, 4, 1, 1), dtype=np.float32),
            tid=0,
            rid=0,
            nid=0,
            rank=0,
            loss=0.9,
            Y_target_grid=np.zeros((2, 2), dtype=np.float32),
            n_total_designs=1,
            is_final=True,
            stack=None,
        )
        assert any("FINAL_RESULT_FALLBACK" in line for line in lines)


class TestFreshPredictionMultiOutput:
    def test_compute_fresh_prediction_extracts_single_dependent_channel(self, monkeypatch):
        from types import SimpleNamespace
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger()
        logger._model = object()
        logger._grid_resolution = (32, 32)

        class _Target:
            def get_lattice(self, resolution, seed):
                xres, yres = resolution
                n = xres * yres
                x = np.zeros((n, 2), dtype=np.float32)
                y = np.zeros((n,), dtype=np.float32)
                return x, y

        logger._dmanager = SimpleNamespace(targets=[_Target()])

        class _CommittedNetwork:
            def __init__(self):
                self.compute_graph = SimpleNamespace(nodes={"n0": object()})

            def get_dependent_output_mask(self):
                return np.array([False, False, False, True], dtype=bool)

        monkeypatch.setattr(
            logger,
            "_get_committed_networks",
            lambda *_args, **_kwargs: [_CommittedNetwork()],
        )

        class _FakeNetworkModel:
            def __init__(self, model, network):
                self.model = model
                self.network = network

            def predict(self, X_lat, **_kwargs):
                n = X_lat.shape[0]
                y = np.zeros((n, 4), dtype=np.float32)
                y[:, 0] = 1.0
                y[:, 1] = 2.0
                y[:, 2] = 3.0
                y[:, 3] = np.arange(n, dtype=np.float32)
                return y, None

        monkeypatch.setattr("biocomptools.modelmodel.NetworkModel", _FakeNetworkModel)

        out = logger._compute_fresh_prediction(
            params={"dummy": 1},
            rep_idx=0,
            target_idx=0,
            network_idx=0,
            stack=object(),
        )

        assert out is not None
        assert out.shape == (32, 32)
        assert logger._last_fresh_prediction_error is None
        np.testing.assert_allclose(out.ravel(), np.arange(1024, dtype=np.float32))


class TestTargetGridOrientation:
    def test_render_heatmaps_uses_cached_target_grid_without_extra_flip(self, monkeypatch):
        from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger

        logger = DesignHeatmapLogger(show_local_params=False, show_ratio_stats=False, top_k=1)
        logger._grid_resolution = (2, 2)
        logger._dmanager = object()
        logger._cached_target_grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        logger._network_names = ["net0"]
        logger._total_steps = 1
        logger._n_targets = 1
        logger.target_idx = 0

        captured = {}

        def _fake_render_single_design(
            step,
            step_history,
            yhatdep,
            tid,
            rid,
            nid,
            rank,
            loss,
            Y_target_grid,
            n_total_designs,
            is_final=False,
            stack=None,
        ):
            captured["target"] = np.asarray(Y_target_grid)
            return ["ok"]

        monkeypatch.setattr(logger, "_render_single_design", _fake_render_single_design)

        step_history = {
            "latest_params": {"dummy": 1},
            "yhatdep": np.zeros((1, 4, 1, 1), dtype=np.float32),
            "all_losses": np.array([[[0.1]]], dtype=np.float32),
        }
        out = logger._render_heatmaps(step=1, step_history=step_history, is_final=True, stack=None)

        assert out is not None
        assert "target" in captured
        np.testing.assert_array_equal(captured["target"], logger._cached_target_grid)
