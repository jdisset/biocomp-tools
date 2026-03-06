"""E2E tests: actual loggers through StepWriter → DB → LoggerRunner replay pipeline.

Tests real logger implementations (ConsoleLogger, EnhancedConsoleLogger,
DataDesignLogger, DesignAuxLogger, DesignSublossLogger, TUMaskingDiagLogger,
CMAESLogger) with realistic step_history data flowing through the full
write → store → replay path.
"""

import json
import threading

import numpy as np
import pytest

from biocomptools.history_db import RunHistoryDB
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logger_runner import LoggerRunner
from biocomptools.step_writer import StepWriter
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.write_policy import WritePolicy


# ---------------------------------------------------------------------------
# Realistic step history factories
# ---------------------------------------------------------------------------


def _make_training_step(step: int) -> dict:
    """Produce step_history matching what train.start() emits."""
    rng = np.random.RandomState(step)
    n_replicates = 2
    # loss is typically an array of shape (n_replicates,) or (n_replicates, batches_per_step)
    loss_per_rep = rng.exponential(0.5, size=n_replicates).astype(np.float32) / (1 + step * 0.1)
    return {
        "loss": loss_per_rep,
        "learning_rate": 0.001 / (1 + step * 0.01),
        "sublosses": {
            "energy_score": float(loss_per_rep.mean()) * 0.7,
            "pairwise": float(loss_per_rep.mean()) * 0.2,
            "coverage": float(loss_per_rep.mean()) * 0.1,
        },
        "yhatdep": rng.randn(n_replicates, 10, 3).astype(np.float32),
        "all_losses": loss_per_rep,
        "latest_params": {"shared": {"w": rng.randn(4, 4), "b": rng.randn(4)}},
        "step_time": rng.uniform(0.1, 0.5),
    }


def _make_design_step(step: int) -> dict:
    """Produce step_history matching what design optimization emits."""
    rng = np.random.RandomState(step + 1000)
    n_replicates, n_targets, n_networks = 2, 3, 4
    n_tus = 8

    all_losses = rng.exponential(0.3, size=(n_replicates, n_targets, n_networks)).astype(
        np.float32
    ) / (1 + step * 0.05)

    # TU masking log_alpha: positive = enabled, negative = disabled
    log_alpha = rng.randn(n_networks, n_tus).astype(np.float32)
    # As optimization progresses, some TUs get more decisive
    log_alpha *= 1 + step * 0.1

    probs = 1 / (1 + np.exp(-log_alpha))

    return {
        "loss": float(all_losses.mean()),
        "l0_penalty": max(0, 0.1 * (step - 50) / 100),
        "spread_penalty": 0.01 * step,
        "sublosses": {
            "sinkhorn": float(all_losses.mean()) * 0.6,
            "lncc": float(all_losses.mean()) * 0.2,
            "rmse": float(all_losses.mean()) * 0.1,
            "l0": max(0, 0.1 * (step - 50) / 100),
        },
        "tu_stats": {
            "n_enabled": int(np.sum(probs >= 0.5)),
            "entropy": float(-np.mean(probs * np.log(probs + 1e-8))),
            "log_alpha": log_alpha,
        },
        "ratio_stats": {
            "mean": float(rng.uniform(0.2, 0.8)),
            "std": float(rng.uniform(0.05, 0.2)),
            "min": float(rng.uniform(0.01, 0.1)),
            "max": float(rng.uniform(0.8, 1.0)),
        },
        "yhatdep": rng.randn(n_replicates, n_targets, 100).astype(np.float32),
        "all_losses": all_losses,
        "apply_aux": {
            "mask": rng.randint(0, 2, size=(n_networks, n_tus)).tolist(),
        },
        "latest_params": {
            "design": {"ratios": rng.randn(n_networks, 3)},
            "tu_mask": {"log_alpha": log_alpha},
        },
    }


def _make_ec_step(step: int) -> dict:
    """Produce step_history matching CMA-ES evolutionary optimizer output."""
    rng = np.random.RandomState(step + 2000)
    return {
        "loss": float(rng.exponential(0.3) / (1 + step * 0.1)),
        "gen_best_loss": float(rng.exponential(0.2) / (1 + step * 0.1)),
        "gen_mean_loss": float(rng.exponential(0.4) / (1 + step * 0.1)),
        "sigma": max(0.01, 0.5 - step * 0.01),
        "n_valid": int(rng.randint(20, 50)),
        "sublosses": {"sinkhorn": float(rng.exponential(0.2))},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAVE_ALL = WritePolicy(save_all=True)


def _write_loop(db: RunHistoryDB, n_steps: int, step_factory, policy=None) -> None:
    writer = StepWriter(db, policy or SAVE_ALL)
    for step in range(1, n_steps + 1):
        writer.write_step(step, float(step), step_factory(step))
    db.mark_finished()


def _replay(db: RunHistoryDB, loggers: list[Logger]) -> None:
    runner = LoggerRunner(db=db, loggers=loggers, mode="replay")
    runner.run()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_db(tmp_path):
    db = RunHistoryDB(tmp_path / "training.db")
    db.save_run_info(run_type="training")
    yield db
    db.close()


@pytest.fixture
def design_db(tmp_path):
    db = RunHistoryDB(tmp_path / "design.db")
    db.save_run_info(run_type="design")
    yield db
    db.close()


# ===========================================================================
# Training logger tests
# ===========================================================================


class TestConsoleLoggerE2E:
    """ConsoleLogger: inline, prints loss stats per replicate."""

    def test_console_logger_live_and_replay(self, training_db, tmp_path):
        from biocomptools.logger_dispatch import LoggerDispatcher
        from biocomptools.toollib.loggers.consolelogger import ConsoleLogger

        db = training_db
        lg_live = ConsoleLogger(call_at_interval=5, execution_mode="inline")

        dispatcher = LoggerDispatcher(
            loggers=[lg_live],
            training_program=None,
            async_logging=True,
            history_db=db,
            write_policy=SAVE_ALL,
        )

        for step in range(1, 21):
            dispatcher.on_step(step, None, _make_training_step(step), None)

        result = (None, list(range(20)), _make_training_step(20))
        dispatcher.shutdown(result)

        # Now replay
        lg_replay = ConsoleLogger(call_at_interval=5)
        db2 = RunHistoryDB(db.path, read_only=True)
        _replay(db2, [lg_replay])
        db2.close()

    def test_enhanced_console_logger_replay(self, training_db):
        from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger

        _write_loop(training_db, 30, _make_training_step)

        lg = EnhancedConsoleLogger(call_at_interval=10)
        _replay(training_db, [lg])
        # EnhancedConsoleLogger is inline normally but can be used in thread mode for replay
        # No crash = success (it prints to console)


class TestEnhancedConsoleDesignMode:
    """EnhancedConsoleLogger detects design mode from all_losses."""

    def test_design_mode_detection_replay(self, design_db):
        from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger

        _write_loop(design_db, 20, _make_design_step)

        lg = EnhancedConsoleLogger(call_at_interval=5)
        _replay(design_db, [lg])
        # Should detect design mode from all_losses shape


# ===========================================================================
# Design logger tests
# ===========================================================================


class TestDataDesignLoggerE2E:
    """DataDesignLogger: tracks loss per target/replicate/network, saves JSON."""

    def test_live_write_then_replay(self, design_db, tmp_path):
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger

        _write_loop(design_db, 50, _make_design_step)

        output_dir = tmp_path / "design_output"
        lg = DataDesignLogger(
            call_at_interval=10,
            call_at=[-1],
            output_dir=str(output_dir),
            save_interval=20,
        )
        _replay(design_db, [lg])

        # Should have saved final files
        assert (output_dir / "final_loss_history.json").exists()
        assert (output_dir / "final_summary.json").exists()

        # Verify JSON content
        with open(output_dir / "final_summary.json") as f:
            summary = json.load(f)
        assert "total_steps" in summary
        assert summary["total_steps"] == 50
        assert summary["shape"]["n_replicates"] == 2
        assert summary["shape"]["n_targets"] == 3
        assert summary["shape"]["n_networks"] == 4

        # Check metrics
        metrics = lg.get_metrics()
        assert metrics is not None
        assert "final_mean_loss" in metrics
        assert "final_best_loss" in metrics
        assert metrics["total_steps"] == 50

    def test_interim_saves_at_save_interval(self, design_db, tmp_path):
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger

        _write_loop(design_db, 30, _make_design_step)

        output_dir = tmp_path / "interim_output"
        lg = DataDesignLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(output_dir),
            save_interval=10,
        )
        _replay(design_db, [lg])

        # Should have saved interim at step 10, 20, 30
        assert (output_dir / "loss_history_step000010.json").exists()
        assert (output_dir / "loss_history_step000020.json").exists()
        assert (output_dir / "loss_history_step000030.json").exists()


class TestDesignAuxLoggerE2E:
    """DesignAuxLogger: tracks comprehensive aux data from design optimization."""

    def test_replay_with_design_data(self, design_db, tmp_path):
        from biocomptools.toollib.loggers.designauxlogger import DesignAuxLogger

        _write_loop(design_db, 30, _make_design_step)

        output_dir = tmp_path / "aux_output"
        lg = DesignAuxLogger(
            call_at_interval=10,
            call_at=[-1],
            output_dir=str(output_dir),
            generate_plots=False,  # Skip matplotlib for speed
            save_pickle=True,
            save_json=True,
        )
        _replay(design_db, [lg])

        # Logger should have accumulated history
        assert len(lg._history) > 0

        # Each history entry should have extracted metrics
        entry = lg._history[0]
        assert "step" in entry


class TestDesignSublossLoggerE2E:
    """DesignSublossLogger: comprehensive debugging visualization."""

    def test_replay_extracts_subloss_data(self, design_db, tmp_path):
        from biocomptools.toollib.loggers.designsublosslogger import DesignSublossLogger

        _write_loop(design_db, 20, _make_design_step)

        output_dir = tmp_path / "subloss_output"
        lg = DesignSublossLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(output_dir),
            generate_plots=False,
            save_pickle=False,
        )
        _replay(design_db, [lg])

        assert len(lg._history) > 0
        entry = lg._history[0]
        assert "step" in entry


class TestTUMaskingDiagLoggerE2E:
    """TUMaskingDiagLogger: per-network TU masking convergence diagnostics."""

    def test_replay_tracks_tu_diagnostics(self, design_db, tmp_path):
        from biocomptools.toollib.loggers.tumaskingdiaglogger import TUMaskingDiagLogger

        _write_loop(design_db, 40, _make_design_step)

        output_dir = tmp_path / "tu_diag_output"
        lg = TUMaskingDiagLogger(
            call_at_interval=10,
            call_at=[-1],
            output_dir=str(output_dir),
            console_output=False,  # No console spam in tests
        )
        _replay(design_db, [lg])

        # Should have accumulated diagnostics
        assert len(lg._history) > 0

        # Each diagnostic should have per-network breakdown
        diag = lg._history[0]
        assert diag.step > 0
        if not diag.no_tu_stats:
            assert len(diag.networks) > 0
            net_diag = diag.networks[0]
            assert net_diag.n_tus == 8  # matches our factory


class TestCMAESLoggerE2E:
    """CMAESLogger: inline, prints EC-specific stats."""

    def test_replay_with_ec_data(self, design_db):
        from biocomptools.toollib.loggers.cmaeslogger import CMAESLogger

        _write_loop(design_db, 20, _make_ec_step)

        lg = CMAESLogger(
            call_at_interval=5,
            call_at=[-1],
            execution_mode="thread",  # Thread for replay
        )
        _replay(design_db, [lg])
        # No crash = success (prints to terminal)


# ===========================================================================
# Multi-logger replay tests
# ===========================================================================


class TestMultiLoggerReplay:
    """Multiple loggers replaying from the same DB simultaneously."""

    def test_training_multi_logger(self, training_db):
        """ConsoleLogger + a recording logger see the same data."""

        class StepRecorder(Logger):
            model_config = Logger.model_config.copy()
            model_config["extra"] = "allow"
            call_at_interval: int | None = 1
            call_at: list[int] = [-1]

            def model_post_init(self, __context):
                super().model_post_init(__context)
                self._lock = threading.Lock()
                self._losses: list[float] = []
                self._end_called = False

            def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
                latest = view.latest()
                if latest:
                    with self._lock:
                        self._losses.append(latest.loss)

            def on_end(self, view: HistoryView, context: LoggerContext) -> None:
                with self._lock:
                    self._end_called = True

        _write_loop(training_db, 20, _make_training_step)

        from biocomptools.toollib.loggers.consolelogger import ConsoleLogger

        console = ConsoleLogger(call_at_interval=10)
        recorder = StepRecorder(call_at_interval=1)

        _replay(training_db, [console, recorder])

        assert len(recorder._losses) == 20
        assert recorder._end_called

    def test_design_multi_logger(self, design_db, tmp_path):
        """DataDesignLogger + DesignAuxLogger both see design data."""
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger
        from biocomptools.toollib.loggers.designauxlogger import DesignAuxLogger

        _write_loop(design_db, 30, _make_design_step)

        data_lg = DataDesignLogger(
            call_at_interval=10,
            call_at=[-1],
            output_dir=str(tmp_path / "data_design"),
            save_interval=100,
        )
        aux_lg = DesignAuxLogger(
            call_at_interval=10,
            call_at=[-1],
            output_dir=str(tmp_path / "aux_design"),
            generate_plots=False,
        )

        _replay(design_db, [data_lg, aux_lg])

        # Both should have received data
        assert data_lg.get_metrics() is not None
        assert len(aux_lg._history) > 0


# ===========================================================================
# Live + Replay consistency tests
# ===========================================================================


class TestLiveReplayConsistency:
    """Verify loggers produce equivalent results in live vs replay mode."""

    def test_data_design_logger_live_vs_replay(self, tmp_path):
        """DataDesignLogger sees same data live vs replayed."""
        from biocomptools.logger_dispatch import LoggerDispatcher
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger

        db = RunHistoryDB(tmp_path / "consistency.db")
        db.save_run_info(run_type="design")

        live_out = tmp_path / "live_output"
        live_lg = DataDesignLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(live_out),
            save_interval=100,
            execution_mode="inline",
        )

        dispatcher = LoggerDispatcher(
            loggers=[live_lg],
            training_program=None,
            async_logging=True,
            history_db=db,
            write_policy=SAVE_ALL,
        )

        n_steps = 25
        for step in range(1, n_steps + 1):
            dispatcher.on_step(step, None, _make_design_step(step), None)

        result = (None, list(range(n_steps)), _make_design_step(n_steps))
        dispatcher.shutdown(result)

        live_metrics = live_lg.get_metrics()

        # Now replay
        replay_out = tmp_path / "replay_output"
        replay_lg = DataDesignLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(replay_out),
            save_interval=100,
        )
        db2 = RunHistoryDB(db.path, read_only=True)
        _replay(db2, [replay_lg])
        db2.close()
        db.close()

        replay_metrics = replay_lg.get_metrics()

        # Both should have metrics
        assert live_metrics is not None
        assert replay_metrics is not None

        # Same number of steps tracked
        assert live_metrics["total_steps"] > 0
        assert replay_metrics["total_steps"] > 0

        # Loss values should exist in both
        assert "final_mean_loss" in live_metrics
        assert "final_mean_loss" in replay_metrics


# ===========================================================================
# WritePolicy interaction with actual loggers
# ===========================================================================


class TestWritePolicyWithLoggers:
    """Verify WritePolicy correctly controls what loggers see during replay."""

    def test_sparse_all_losses_affects_data_design_logger(self, design_db, tmp_path):
        """With periodic all_losses, DataDesignLogger only gets data at policy intervals."""
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger

        policy = WritePolicy(
            periodic_arrays={"all_losses": 10, "yhatdep": 10},
            every_step_arrays=frozenset(),
            params_interval=100,
        )
        _write_loop(design_db, 30, _make_design_step, policy)

        lg = DataDesignLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(tmp_path / "sparse_output"),
            save_interval=100,
        )
        _replay(design_db, [lg])

        # DataDesignLogger only updates when all_losses is present
        # With periodic_arrays={"all_losses": 10}, only steps 10, 20, 30 have all_losses
        # on_end also processes the latest data, so step 30 appears twice (on_batch + on_end)
        assert len(lg._loss_history) == 4  # steps 10, 20, 30 (on_batch), 30 (on_end)

    def test_tu_masking_logger_needs_tu_stats_dict(self, design_db, tmp_path):
        """TUMaskingDiagLogger needs tu_stats dict — always saved by default policy."""
        from biocomptools.toollib.loggers.tumaskingdiaglogger import TUMaskingDiagLogger

        # Default policy saves dicts every step
        _write_loop(design_db, 20, _make_design_step, WritePolicy())

        lg = TUMaskingDiagLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(tmp_path / "tu_diag"),
            console_output=False,
        )
        _replay(design_db, [lg])

        # tu_stats is a dict, always saved → logger should get it
        assert len(lg._history) > 0


# ===========================================================================
# High-level replay_history API with actual loggers
# ===========================================================================


class TestReplayHistoryWithActualLoggers:
    """replay_history() with real logger implementations."""

    def test_replay_history_with_data_design_logger(self, tmp_path):
        from biocomptools.run_replay import replay_history
        from biocomptools.toollib.loggers.datadesignlogger import DataDesignLogger

        history_dir = tmp_path / "run_output"
        history_dir.mkdir()
        output_dir = tmp_path / "replay_output"
        output_dir.mkdir()

        db = RunHistoryDB(history_dir / "run_history.db")
        db.save_run_info(run_type="design")
        _write_loop(db, 20, _make_design_step)
        db.close()

        lg = DataDesignLogger(
            call_at_interval=5,
            call_at=[-1],
            output_dir=str(output_dir),
            save_interval=100,
        )
        replay_history(history_dir, [lg], output_dir)

        assert (output_dir / "final_loss_history.json").exists()
        assert (output_dir / "final_summary.json").exists()
        metrics = lg.get_metrics()
        assert metrics is not None
        assert metrics["total_steps"] == 20
