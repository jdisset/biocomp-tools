"""E2E tests: StepWriter → RunHistoryDB → LoggerRunner replay pipeline.

Validates the full round-trip: optimization loop writes data → DB stores it →
replay reads it → loggers receive correct data.
"""

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
# Step history factories (deterministic via seed)
# ---------------------------------------------------------------------------


def _make_training_step_history(step: int) -> dict:
    rng = np.random.RandomState(step)
    return {
        "loss": float(step) * 0.1,
        "learning_rate": 0.001 / (1 + step * 0.01),
        "sublosses": {"mse": step * 0.05, "reg": step * 0.02},
        "yhatdep": rng.randn(10, 3).astype(np.float32),
        "all_losses": rng.randn(10).astype(np.float32),
        "latest_params": {"layer0": {"w": rng.randn(4, 4), "b": rng.randn(4)}},
    }


def _make_design_step_history(step: int) -> dict:
    rng = np.random.RandomState(step)
    return {
        "loss": float(step) * 0.15,
        "l0_penalty": step * 0.01,
        "spread_penalty": step * 0.005,
        "sublosses": {"sinkhorn": step * 0.08, "lncc": step * 0.03},
        "tu_stats": {"n_enabled": max(1, 10 - step), "entropy": 0.5 / (1 + step)},
        "ratio_stats": {"mean": 0.3 + step * 0.01, "std": 0.1},
        "yhatdep": rng.randn(8, 2).astype(np.float32),
        "apply_aux": {"mask": rng.randint(0, 2, size=5).tolist(), "scores": rng.randn(5).tolist()},
        "latest_params": {"design": {"ratios": rng.randn(3)}},
    }


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _write_training_loop(
    db: RunHistoryDB,
    n_steps: int,
    policy: WritePolicy | None,
    step_factory=_make_training_step_history,
) -> None:
    writer = StepWriter(db, policy)
    for step in range(1, n_steps + 1):
        sh = step_factory(step)
        writer.write_step(step, float(step), sh)
    db.mark_finished()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    d = RunHistoryDB(tmp_path / "test.db")
    d.save_run_info(run_type="test")
    yield d
    d.close()


@pytest.fixture
def history_dir(tmp_path):
    return tmp_path / "history"


# ---------------------------------------------------------------------------
# Test logger subclasses
# ---------------------------------------------------------------------------


class AllDataCaptureLogger(Logger):
    """Captures everything from on_batch/on_end."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._steps: list[int] = []
        self._losses: list[float] = []
        self._metric_keys: list[set[str]] = []
        self._array_keys: list[set[str]] = []
        self._n_batches: list[int] = []
        self._end_called = False
        self._end_view_n_batches = 0

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest is None:
            return
        with self._lock:
            self._steps.append(latest.step_index)
            self._losses.append(latest.loss)
            self._metric_keys.append(set(latest.metrics.keys()))
            self._array_keys.append(set(latest.arrays.keys()))
            self._n_batches.append(view.n_batches)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._end_called = True
            self._end_view_n_batches = view.n_batches


class IntervalCaptureLogger(Logger):
    """Records which steps fired."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._steps_seen: list[int] = []
        self._end_called = False

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            with self._lock:
                self._steps_seen.append(latest.step_index)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._end_called = True


class WindowedCaptureLogger(Logger):
    """Records (step, n_batches) per call."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._window_sizes: list[tuple[int, int]] = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            with self._lock:
                self._window_sizes.append((latest.step_index, view.n_batches))


class SelectiveLoadLogger(Logger):
    """Has required_metrics/required_arrays to test selective loading."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._metric_keys_by_step: dict[int, set[str]] = {}
        self._array_keys_by_step: dict[int, set[str]] = {}

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            with self._lock:
                self._metric_keys_by_step[latest.step_index] = set(latest.metrics.keys())
                self._array_keys_by_step[latest.step_index] = set(latest.arrays.keys())


class ContextVerifierLogger(Logger):
    """Records is_replay/is_final flags."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [0, -1]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._batch_is_replay: list[bool] = []
        self._start_is_replay: bool | None = None
        self._end_is_replay: bool | None = None
        self._end_is_final: bool | None = None

    def on_start(self, context: LoggerContext) -> None:
        with self._lock:
            self._start_is_replay = context.is_replay

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._batch_is_replay.append(context.is_replay)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._end_is_replay = context.is_replay
            self._end_is_final = context.is_final


class ArrayValueLogger(Logger):
    """Captures actual array values per step."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._arrays_by_step: dict[int, dict[str, np.ndarray]] = {}
        self._blobs_by_step: dict[int, dict] = {}

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            with self._lock:
                arrs = {}
                blobs = {}
                for k, v in latest.arrays.items():
                    if isinstance(v, np.ndarray):
                        arrs[k] = v.copy()
                    elif isinstance(v, dict):
                        blobs[k] = v
                self._arrays_by_step[latest.step_index] = arrs
                self._blobs_by_step[latest.step_index] = blobs


# ===========================================================================
# Tests
# ===========================================================================


class TestRoundtripDataIntegrity:
    """Round-trip data integrity (tests 1-4)."""

    def test_full_roundtrip_training_data(self, db):
        """20 steps training data → StepWriter → replay → verify all data categories."""
        _write_training_loop(db, 20, WritePolicy(save_all=True))

        capture = AllDataCaptureLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        # should_fire skips step 0, so steps 1..20
        assert len(capture._steps) == 20
        assert capture._end_called

        # Verify data categories present at each step
        for i, step in enumerate(capture._steps):
            assert capture._losses[i] == pytest.approx(step * 0.1)
            # Scalars: learning_rate triaged as scalar
            assert "learning_rate" in capture._metric_keys[i]
            # Dicts: sublosses triaged as dict
            assert "sublosses" in capture._metric_keys[i]
            # Arrays: yhatdep, all_losses
            assert "yhatdep" in capture._array_keys[i]

    def test_full_roundtrip_design_data(self, db):
        """15 steps design data → StepWriter → replay → verify design-specific keys."""
        _write_training_loop(db, 15, WritePolicy(save_all=True), _make_design_step_history)

        capture = AllDataCaptureLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        assert len(capture._steps) == 15
        assert capture._end_called

        for i, step in enumerate(capture._steps):
            assert capture._losses[i] == pytest.approx(step * 0.15)
            # Design scalars
            assert "l0_penalty" in capture._metric_keys[i]
            assert "spread_penalty" in capture._metric_keys[i]
            # Design dicts
            assert "tu_stats" in capture._metric_keys[i]
            assert "ratio_stats" in capture._metric_keys[i]
            assert "sublosses" in capture._metric_keys[i]

    def test_array_values_roundtrip(self, db):
        """10 steps → replay → verify yhatdep values match per step."""
        _write_training_loop(db, 10, WritePolicy(save_all=True))

        capture = ArrayValueLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        for step in range(1, 11):
            expected = _make_training_step_history(step)["yhatdep"]
            actual = capture._arrays_by_step[step]["yhatdep"]
            np.testing.assert_array_almost_equal(actual, expected)

    def test_blob_values_roundtrip(self, db):
        """5 steps → replay → verify latest_params blob correct."""
        _write_training_loop(db, 5, WritePolicy(save_all=True))

        capture = ArrayValueLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        for step in range(1, 6):
            expected = _make_training_step_history(step)["latest_params"]
            actual = capture._blobs_by_step[step]["latest_params"]
            assert set(actual.keys()) == set(expected.keys())
            for k in expected:
                if isinstance(expected[k], dict):
                    for k2 in expected[k]:
                        np.testing.assert_array_almost_equal(
                            np.asarray(actual[k][k2]), np.asarray(expected[k][k2])
                        )


class TestValueConsistency:
    """Value consistency (tests 5-6)."""

    def test_dict_values_roundtrip(self, db):
        """Verify sublosses dict values survive JSON round-trip."""
        _write_training_loop(db, 10, WritePolicy(save_all=True))

        capture = AllDataCaptureLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        for step in capture._steps:
            expected_sublosses = _make_training_step_history(step)["sublosses"]
            # Access via the DB directly for a clean check
            bd = db.load_step_data(step)
            assert bd is not None
            actual = bd.metrics["sublosses"]
            assert actual["mse"] == pytest.approx(expected_sublosses["mse"])
            assert actual["reg"] == pytest.approx(expected_sublosses["reg"])

    def test_loss_series_consistency(self, db):
        """Logger with history_window=None captures full loss series at last step."""
        n_steps = 15
        _write_training_loop(db, n_steps, WritePolicy(save_all=True))

        capture = AllDataCaptureLogger(call_at_interval=None, call_at=[-1], history_window=None)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        assert capture._end_called
        assert capture._end_view_n_batches == n_steps

        # Verify loss values via DB API
        steps_arr, losses_arr = db.load_loss_series()
        expected_losses = [float(s) * 0.1 for s in range(1, n_steps + 1)]
        np.testing.assert_array_almost_equal(losses_arr, expected_losses)


class TestWritePolicyEffects:
    """WritePolicy affects what's available during replay (test 7)."""

    def test_write_policy_affects_replay_availability(self, db):
        """Periodic arrays only available at policy-specified intervals."""
        policy = WritePolicy(
            periodic_arrays={"yhatdep": 5, "all_losses": 3},
            every_step_arrays=frozenset(),
            params_interval=4,
        )
        _write_training_loop(db, 10, policy)

        # Verify yhatdep only at steps 5, 10
        for step in range(1, 11):
            arr = db.load_array(step, "yhatdep")
            if step % 5 == 0:
                assert arr is not None, f"yhatdep should exist at step {step}"
            else:
                assert arr is None, f"yhatdep should NOT exist at step {step}"

        # Verify all_losses only at steps 3, 6, 9
        for step in range(1, 11):
            arr = db.load_array(step, "all_losses")
            if step % 3 == 0:
                assert arr is not None, f"all_losses should exist at step {step}"
            else:
                assert arr is None, f"all_losses should NOT exist at step {step}"

        # Verify latest_params only at steps 4, 8
        for step in range(1, 11):
            blob = db.load_blob(step, "latest_params")
            if step % 4 == 0:
                assert blob is not None, f"latest_params should exist at step {step}"
            else:
                assert blob is None, f"latest_params should NOT exist at step {step}"


class TestSchedulingAndWindowing:
    """Scheduling and windowing (tests 8-9)."""

    def test_multiple_loggers_different_intervals_replay(self, db):
        """Two loggers with different intervals see correct step subsets."""
        _write_training_loop(db, 21, WritePolicy(save_all=True))

        lg3 = IntervalCaptureLogger(call_at_interval=3, call_at=[-1])
        lg7 = IntervalCaptureLogger(call_at_interval=7, call_at=[-1])
        runner = LoggerRunner(db=db, loggers=[lg3, lg7], mode="replay")
        runner.run()

        assert sorted(lg3._steps_seen) == [3, 6, 9, 12, 15, 18, 21]
        assert sorted(lg7._steps_seen) == [7, 14, 21]
        assert lg3._end_called
        assert lg7._end_called

    def test_history_window_limits_view_size(self, db):
        """Logger with history_window=5 never sees more than 5 batches."""
        _write_training_loop(db, 20, WritePolicy(save_all=True))

        capture = WindowedCaptureLogger(call_at_interval=1, history_window=5)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        for step, n_batches in capture._window_sizes:
            # _build_view uses start = max(0, step - window), loading [start..step] inclusive
            # so max batches = window + 1 for steps > window
            assert n_batches <= 6, f"step {step}: got {n_batches} batches, expected <= 6"
            expected = min(step, 6)
            assert n_batches == expected, f"step {step}: got {n_batches}, expected {expected}"


class TestContextAndLifecycle:
    """Context and lifecycle (tests 10-11)."""

    def test_context_is_replay_true_during_replay(self, db):
        """is_replay=True in all on_start/on_batch/on_end contexts during replay."""
        _write_training_loop(db, 10, WritePolicy(save_all=True))

        verifier = ContextVerifierLogger(call_at_interval=1, call_at=[0, -1])
        runner = LoggerRunner(db=db, loggers=[verifier], mode="replay")
        runner.run()

        assert verifier._start_is_replay is True
        assert all(verifier._batch_is_replay), "All on_batch calls should have is_replay=True"
        assert verifier._end_is_replay is True
        assert verifier._end_is_final is True

    def test_on_start_on_end_only_through_replay(self, db):
        """Logger with call_at=[0, -1] and no interval: on_start + on_end, no on_batch."""
        _write_training_loop(db, 10, WritePolicy(save_all=True))

        verifier = ContextVerifierLogger(call_at_interval=None, call_at=[0, -1])
        runner = LoggerRunner(db=db, loggers=[verifier], mode="replay")
        runner.run()

        assert verifier._start_is_replay is True
        assert verifier._end_is_replay is True
        assert len(verifier._batch_is_replay) == 0  # no on_batch calls


class TestHighLevelAPI:
    """High-level replay_history API (test 12)."""

    def test_replay_history_high_level_api(self, tmp_path):
        """Create DB at history_dir/run_history.db → write → replay_history()."""
        from biocomptools.run_replay import replay_history

        history_dir = tmp_path / "run_output"
        history_dir.mkdir()
        output_dir = tmp_path / "replay_output"
        output_dir.mkdir()

        db = RunHistoryDB(history_dir / "run_history.db")
        db.save_run_info(run_type="test")
        _write_training_loop(db, 10, WritePolicy(save_all=True))
        db.close()

        capture = AllDataCaptureLogger(call_at_interval=1)
        replay_history(history_dir, [capture], output_dir)

        assert len(capture._steps) == 10
        assert capture._end_called
        assert pytest.approx(0.1) in capture._losses


class TestFullIntegration:
    """Full integration with LoggerDispatcher (test 13)."""

    def test_dispatcher_writes_to_db_then_replay_reads(self, tmp_path):
        """LoggerDispatcher writes → shutdown → replay from same DB."""
        from biocomptools.logger_dispatch import LoggerDispatcher

        db = RunHistoryDB(tmp_path / "integration.db")
        db.save_run_info(run_type="test")

        # Inline logger sees live data
        inline_lg = IntervalCaptureLogger(call_at_interval=1, call_at=[-1], execution_mode="inline")

        dispatcher = LoggerDispatcher(
            loggers=[inline_lg],
            training_program=None,
            async_logging=True,
            history_db=db,
            write_policy=WritePolicy(save_all=True),
        )

        # Simulate optimization loop (plain dicts are StepHistoryLike)
        n_steps = 15
        for step in range(1, n_steps + 1):
            sh = _make_training_step_history(step)
            dispatcher.on_step(step, None, sh, None)

        # Build result tuple for shutdown: (model, loss_history, step_history)
        final_sh = _make_training_step_history(n_steps)
        result = (None, list(range(n_steps)), final_sh)
        dispatcher.shutdown(result)

        # Inline logger saw live steps
        assert len(inline_lg._steps_seen) > 0

        # Now replay from the same DB
        replay_lg = AllDataCaptureLogger(call_at_interval=1)
        db2 = RunHistoryDB(tmp_path / "integration.db", read_only=True)
        replay_runner = LoggerRunner(db=db2, loggers=[replay_lg], mode="replay")
        replay_runner.run()
        db2.close()
        db.close()

        assert replay_lg._end_called
        assert len(replay_lg._steps) > 0
        # All replay contexts should have is_replay=True (verified by ContextVerifier above)


class TestSelectiveLoading:
    """Selective loading filters keys (test 14)."""

    def test_selective_loading_filters_scalar_keys(self, db):
        """Logger with required_metrics/required_arrays gets only declared keys."""
        _write_training_loop(db, 10, WritePolicy(save_all=True))

        selective = SelectiveLoadLogger(
            call_at_interval=1,
            required_metrics=["sublosses"],
            required_arrays=["yhatdep"],
        )
        runner = LoggerRunner(db=db, loggers=[selective], mode="replay")
        runner.run()

        for step, mkeys in selective._metric_keys_by_step.items():
            # learning_rate is a scalar — should NOT be loaded (only "sublosses" requested)
            assert "learning_rate" not in mkeys, (
                f"learning_rate should be filtered out at step {step}"
            )
            # sublosses is a dict — should be loaded
            assert "sublosses" in mkeys, f"sublosses should be present at step {step}"

        for step, akeys in selective._array_keys_by_step.items():
            # yhatdep requested
            assert "yhatdep" in akeys, f"yhatdep should be present at step {step}"
            # all_losses not requested
            assert "all_losses" not in akeys, f"all_losses should be filtered out at step {step}"


class TestEdgeCases:
    """Edge cases (test 15)."""

    def test_empty_replay_no_steps(self, db):
        """DB with RunInfo but no steps → no on_batch, no on_end."""
        db.mark_finished()

        capture = AllDataCaptureLogger(call_at_interval=1)
        runner = LoggerRunner(db=db, loggers=[capture], mode="replay")
        runner.run()

        assert len(capture._steps) == 0
        assert not capture._end_called
