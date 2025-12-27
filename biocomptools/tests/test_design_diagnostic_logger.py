"""Tests for DesignDiagnosticLogger."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "diagnostics"


class TestHelperFunctions:
    """Tests for helper functions in designdiagnosticlogger module."""

    def test_to_scalar_with_none(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import _to_scalar
        assert np.isnan(_to_scalar(None))

    def test_to_scalar_with_scalar(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import _to_scalar
        assert _to_scalar(3.14) == 3.14

    def test_to_scalar_with_single_element_array(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import _to_scalar
        assert _to_scalar(np.array([5.0])) == 5.0

    def test_to_scalar_with_multi_element_array(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import _to_scalar
        result = _to_scalar(np.array([1.0, 2.0, 3.0]))
        assert result == 2.0  # mean

    def test_unroll_dict_flat(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_dict
        d = {"a": 1.0, "b": 2.0}
        result = unroll_dict(d)
        assert result == {"a": 1.0, "b": 2.0}

    def test_unroll_dict_nested(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_dict
        d = {"outer": {"inner": 3.0}}
        result = unroll_dict(d)
        assert result == {"outer.inner": 3.0}

    def test_unroll_dict_with_prefix(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_dict
        d = {"a": 1.0}
        result = unroll_dict(d, "prefix")
        assert result == {"prefix.a": 1.0}

    def test_unroll_dict_with_array(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_dict
        d = {"arr": np.array([1.0, 2.0])}
        result = unroll_dict(d)
        assert result == {"arr.0": 1.0, "arr.1": 2.0}

    def test_prepare_particle_data_basic(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import prepare_particle_data
        history = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        keys = ["a", "b"]
        data, names, derivatives = prepare_particle_data(history, keys)

        assert data.shape == (2, 2)
        assert names == ["a", "b"]
        assert derivatives is None
        assert data[0, 0] == 1.0
        assert data[1, 1] == 4.0

    def test_prepare_particle_data_with_missing_keys(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import prepare_particle_data
        history = [{"a": 1.0}, {"a": 2.0}]
        keys = ["a", "missing"]
        data, names, _ = prepare_particle_data(history, keys)

        assert data.shape == (2, 2)
        assert data[0, 0] == 1.0
        assert np.isnan(data[1, 0])

    def test_prepare_particle_data_with_derivatives(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import prepare_particle_data
        history = [{"a": 1.0}]
        deriv_history = [{"a": 0.5}]
        keys = ["a"]
        _, _, derivatives = prepare_particle_data(history, keys, deriv_history)

        assert derivatives is not None
        assert derivatives[0] == 0.5


class TestDesignDiagnosticLogger:
    """Tests for the DesignDiagnosticLogger class."""

    def test_import(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        assert DesignDiagnosticLogger is not None

    def test_create_logger(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(
            output_dir=str(tmp_output_dir),
            periods=5,
            max_history_len=10,
        )
        assert logger.periods == 5
        assert logger.max_history_len == 10

    def test_initialize_creates_directory(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir))
        logger.initialize()
        assert tmp_output_dir.exists()

    def test_get_callbacks_returns_two(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir))
        logger.initialize()
        callbacks = logger.get_callbacks()
        assert len(callbacks) == 2
        assert callbacks[0][0] == logger.periods  # periodic
        assert callbacks[1][0] == -1  # final

    def test_extract_metrics(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir))
        logger._total_steps = 100

        # all_losses: 3D array (replicate, target, network) -> arr[0, target_id, network_id]
        # per_network metrics: 4D arrays (replicate, batch, target, network) -> arr[0, 0, target_id, network_id]
        step_history = {
            "loss": 0.5,
            "all_losses": np.array([[[0.4, 0.5, 0.6]]]),  # (1, 1, 3) -> network_loss = arr[0, 0, 0] = 0.4
            "sublosses": {
                "sinkhorn_per_network": np.array([[[[0.1, 0.2, 0.3]]]]),  # (1, 1, 1, 3)
                "lncc_per_network": np.array([[[[0.15, 0.25, 0.35]]]]),
            },
            "tu_stats": {
                "enabled_count_per_network": np.array([[[[3.0, 4.0, 5.0]]]]),
                "mean_prob_per_network": np.array([[[[0.5, 0.6, 0.7]]]]),
            },
            "l0_penalty_per_network": np.array([[[[0.01, 0.02, 0.03]]]]),
            "tucount_penalty": 0.05,
        }

        metrics = logger._extract_metrics(50, step_history, target_id=0, network_id=0)

        assert metrics["step"] == 50
        assert metrics["progress"] == 0.5
        assert metrics["loss"] == 0.5
        assert metrics["network_loss"] == 0.4  # from all_losses[0, 0, 0]
        assert metrics["sinkhorn"] == 0.1  # from sublosses.sinkhorn_per_network[0, 0, 0, 0]
        assert metrics["tu_enabled_count"] == 3.0  # from tu_stats.enabled_count_per_network[0, 0, 0, 0]
        assert metrics["l0_penalty"] == 0.01  # from l0_penalty_per_network[0, 0, 0, 0]
        assert metrics["tucount_penalty"] == 0.05  # global penalty

    def test_append_to_history(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir), max_history_len=3)
        logger.initialize()

        for i in range(5):
            logger._append_to_history(0, 0, {"step": i})

        assert len(logger._history[(0, 0)]) == 3
        assert logger._history[(0, 0)][0]["step"] == 2  # oldest kept

    def test_get_metrics_empty(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir))
        assert logger.get_metrics() is None

    def test_get_metrics_with_data(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
        logger = DesignDiagnosticLogger(output_dir=str(tmp_output_dir))
        logger._history = {(0, 0): [{"step": 1}, {"step": 2}]}

        metrics = logger.get_metrics()
        assert metrics["targets_networks_tracked"] == 1
        assert metrics["total_entries"] == 2


class TestUnrollParams:
    """Tests for parameter unrolling functions."""

    def test_unroll_params_with_none(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_params
        result = unroll_params(None, 0, 0, 0)
        assert result == {}

    def test_unroll_grads_with_none(self):
        from biocomptools.toollib.loggers.designdiagnosticlogger import unroll_grads
        result = unroll_grads(None, 0, 0, 0)
        assert result == {}


class TestIntegration:
    """Integration tests for the logger with mock step_history."""

    def test_periodic_callback_with_valid_data(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger

        logger = DesignDiagnosticLogger(
            output_dir=str(tmp_output_dir),
            periods=1,
            generate_plots=False,  # skip plotting for speed
        )
        logger.initialize()
        logger._grid_resolution = (4, 4)  # 16 points
        logger._total_steps = 10

        callbacks = logger.get_callbacks()
        periodic_cb = callbacks[0][1]

        # Mock step_history with correct shapes
        step_history = {
            "loss": 0.5,
            "all_losses": np.array([[[0.4, 0.5]]]),  # (1, 1, 2)
            "yhatdep": np.random.rand(16, 1, 2),  # (batch, targets, networks)
            "sublosses": {"sinkhorn": 0.1, "lncc": 0.2},
            "tu_stats": {"enabled_count": 3, "total_count": 5},
            "ratio_stats": {"min": 0.1, "max": 0.9},
        }

        # Call callback
        periodic_cb(step=1, training_config=None, step_history=step_history)

        # Check history was updated
        assert (0, 0) in logger._history
        assert len(logger._history[(0, 0)]) == 1

    def test_finalize_saves_history(self, tmp_output_dir):
        from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger

        logger = DesignDiagnosticLogger(
            output_dir=str(tmp_output_dir),
            generate_plots=False,
            save_history=True,
        )
        logger.initialize()
        logger._history = {(0, 0): [{"step": 1}]}

        logger.finalize()

        # finalize just logs, final_callback saves the pickle
        callbacks = logger.get_callbacks()
        final_cb = callbacks[1][1]

        step_history = {
            "loss": 0.3,
            "yhatdep": np.random.rand(16, 1, 1),
        }
        logger._grid_resolution = (4, 4)
        final_cb(step=10, training_config=None, step_history=step_history)

        final_dir = tmp_output_dir / "final"
        assert final_dir.exists()
        assert (final_dir / "full_history.pickle").exists()
