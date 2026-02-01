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
