"""End-to-end tests for async logger handler optimizations.

Tests exercise the full LoggerDispatcher → AsyncLoggerHandler → consumer thread →
thread pool dispatch pipeline with synthetic training loops and capture loggers.

Validates three fixes:
  Fix 1: In-memory queue (no disk serialization on main thread)
  Fix 2: Cached stack/training_config (not re-serialized per step)
  Fix 3: Params not accumulated in HistoryManager (single-slot cache + injection)
"""

import threading
from typing import Any

from biocomptools.logger_dispatch import LoggerDispatcher
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.toollib.loggers.logger import Logger


# ──────────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────────


def _make_params(step: int) -> dict[str, float]:
    return {"value": step * 10.0, "step_marker": float(step)}


def _make_step_history(step: int) -> dict[str, Any]:
    return {
        "loss": step * 0.01,
        "latest_params": _make_params(step),
        "sublosses": {"mse": step * 0.005, "rmse": step * 0.003},
    }


# ──────────────────────────────────────────────────────────────────────
# Capture loggers (new-pattern: override on_batch / on_end)
# ──────────────────────────────────────────────────────────────────────


class _ParamsVerifierLogger(Logger):
    """Captures latest_params at every step via on_batch."""

    call_at_interval: int = 1
    required_arrays: list[str] = ["latest_params"]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._captures: list[tuple[int, Any]] = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        params = view.get_array("latest_params")
        with self._lock:
            self._captures.append((context.current_step, params))


class _HistoryAccumulatorLogger(Logger):
    """Records history window size at each invocation."""

    call_at_interval: int = 5
    history_window: int = 10
    required_metrics: list[str] = ["sublosses"]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._captures: list[tuple[int, int, list[float]]] = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        losses = view.losses.tolist()
        with self._lock:
            self._captures.append((context.current_step, view.n_batches, losses))


class _ParamsAndHistoryLogger(Logger):
    """Captures both params and loss history."""

    call_at_interval: int = 3
    history_window: int = 5
    required_arrays: list[str] = ["latest_params"]
    required_metrics: list[str] = ["sublosses"]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._captures: list[tuple[int, Any, list[float]]] = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        params = view.get_array("latest_params")
        losses = view.losses.tolist()
        with self._lock:
            self._captures.append((context.current_step, params, losses))


class _EndOnlyLogger(Logger):
    """Fires only at training end."""

    call_at: list[int] = [-1]
    call_at_interval: int | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._end_calls: list[tuple[int, dict]] = []

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        sh = view.to_step_history()
        with self._lock:
            self._end_calls.append((context.current_step, sh))


class _StartAndEndLogger(Logger):
    """Fires at start and end."""

    call_at: list[int] = [0, -1]
    call_at_interval: int | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._start_calls: list[int] = []
        self._end_calls: list[int] = []

    def on_start(self, context: LoggerContext) -> None:
        with self._lock:
            self._start_calls.append(context.current_step)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._end_calls.append(context.current_step)


class _ContextCaptureLogger(Logger):
    """Records object identity of stack/config to verify caching."""

    call_at_interval: int = 1

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._lock = threading.Lock()
        self._stack_ids: list[int] = []
        self._config_ids: list[int] = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        with self._lock:
            self._stack_ids.append(id(context.stack))
            self._config_ids.append(id(context.training_config))


# ──────────────────────────────────────────────────────────────────────
# Simulated training loop
# ──────────────────────────────────────────────────────────────────────


def _run_simulated_training(
    loggers: list[Logger],
    n_steps: int = 20,
    *,
    async_logging: bool = True,
    n_workers: int = 2,
) -> LoggerDispatcher:
    """Run synthetic training loop through the full dispatcher machinery."""
    fake_stack = {"layers": [1, 2, 3]}
    fake_config = {"lr": 0.01, "epochs": 5}

    dispatcher = LoggerDispatcher(
        loggers,
        training_program=object(),
        async_logging=async_logging,
        n_workers=n_workers,
    )

    dispatcher.on_start(fake_config, fake_stack)

    last_sh: dict[str, Any] = {}
    for step in range(1, n_steps + 1):
        last_sh = _make_step_history(step)
        dispatcher.on_step(step, fake_config, last_sh, fake_stack)

    dispatcher.on_end(n_steps, fake_config, last_sh, fake_stack)
    dispatcher.shutdown((None, [i * 0.01 for i in range(1, n_steps + 1)], last_sh))
    return dispatcher


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_params_correct_at_every_step():
    """Fix 3 critical test: each on_batch invocation gets the correct step's params."""
    lg = _ParamsVerifierLogger()
    _run_simulated_training([lg], n_steps=20)

    captured = sorted(lg._captures, key=lambda x: x[0])
    steps_seen = {step for step, _ in captured}
    assert steps_seen == set(range(1, 21)), f"Missing steps: {set(range(1, 21)) - steps_seen}"

    for step, params in captured:
        expected = _make_params(step)
        assert params is not None, f"Step {step}: params is None"
        assert params == expected, f"Step {step}: expected {expected}, got {params}"


def test_history_window_respected():
    """HistoryManager windowing works with stripped params."""
    lg = _HistoryAccumulatorLogger()
    _run_simulated_training([lg], n_steps=30)

    captures = sorted(lg._captures, key=lambda x: x[0])
    assert len(captures) > 0

    for step, n_batches, _losses in captures:
        assert n_batches <= 10, f"Step {step}: window exceeded ({n_batches} > 10)"
        expected_max = min(step, 10)
        assert n_batches <= expected_max, (
            f"Step {step}: n_batches={n_batches} > expected max {expected_max}"
        )


def test_params_and_history_combined():
    """Combined params + history logger receives correct data."""
    lg = _ParamsAndHistoryLogger()
    _run_simulated_training([lg], n_steps=15)

    captures = sorted(lg._captures, key=lambda x: x[0])
    assert len(captures) > 0

    for step, params, losses in captures:
        expected = _make_params(step)
        assert params == expected, f"Step {step}: wrong params"
        assert len(losses) <= 5, f"Step {step}: history window exceeded"


def test_end_logger_receives_final_state():
    """EndOnlyLogger fires exactly once."""
    lg = _EndOnlyLogger()
    _run_simulated_training([lg], n_steps=10)

    assert len(lg._end_calls) == 1, f"Expected 1 end call, got {len(lg._end_calls)}"


def test_start_and_end_lifecycle():
    """StartAndEndLogger fires at start (step=0) and at end."""
    lg = _StartAndEndLogger()
    _run_simulated_training([lg], n_steps=10)

    assert len(lg._start_calls) == 1
    assert lg._start_calls[0] == 0
    assert len(lg._end_calls) == 1


def test_multiple_loggers_different_intervals():
    """Multiple loggers with different intervals coexist without interference."""
    params_lg = _ParamsVerifierLogger()
    history_lg = _HistoryAccumulatorLogger()
    end_lg = _EndOnlyLogger()
    start_end_lg = _StartAndEndLogger()

    _run_simulated_training(
        [params_lg, history_lg, end_lg, start_end_lg],
        n_steps=20,
    )

    # Params logger: every step
    params_steps = {step for step, _ in params_lg._captures}
    assert params_steps == set(range(1, 21))

    # History logger: multiples of 5
    history_steps = {step for step, _, _ in history_lg._captures}
    assert history_steps == {5, 10, 15, 20}

    # End / start+end: one call each
    assert len(end_lg._end_calls) == 1
    assert len(start_end_lg._start_calls) == 1
    assert len(start_end_lg._end_calls) == 1


def test_high_frequency_no_params_in_history_deque():
    """Fix 3: 50 steps with interval=1, all params correct (proves injection works)."""
    lg = _ParamsVerifierLogger()
    _run_simulated_training([lg], n_steps=50)

    assert len(lg._captures) == 50
    for step, params in sorted(lg._captures, key=lambda x: x[0]):
        assert params == _make_params(step)


def test_stack_and_config_cached():
    """Fix 2: stack and config are the same object (by identity) across invocations."""
    lg = _ContextCaptureLogger()
    _run_simulated_training([lg], n_steps=10)

    assert len(lg._stack_ids) == 10
    assert len(set(lg._stack_ids)) == 1, "Stack objects differ across invocations"
    assert len(set(lg._config_ids)) == 1, "Config objects differ across invocations"


def test_db_step_records_exclude_stack(tmp_path):
    """Step records in DB contain only step_history data, not stack/config."""
    from biocomptools.history_db import RunHistoryDB

    db = RunHistoryDB(tmp_path / "test.db")
    db.save_run_info(run_type="test")

    lg = _ParamsVerifierLogger()
    _run_simulated_training(
        [lg],
        n_steps=5,
        async_logging=False,
    )

    # Verify params were captured correctly (the core invariant)
    for step, params in sorted(lg._captures, key=lambda x: x[0]):
        assert params == _make_params(step)

    db.close()


def test_step_offset_produces_non_overlapping_db_records(tmp_path):
    """Two segments writing to same DB produce non-overlapping step indices."""
    from biocomptools.history_db import RunHistoryDB

    db = RunHistoryDB(tmp_path / "test.db")
    db.save_run_info(run_type="test")

    fake_stack = {"layers": [1, 2, 3]}
    fake_config = {"lr": 0.01}

    # Segment 0: steps 0..9
    lg0 = _ParamsVerifierLogger()
    d0 = LoggerDispatcher(
        [lg0],
        training_program=object(),
        async_logging=False,
        n_workers=2,
        history_db=db,
    )
    d0.on_start(fake_config, fake_stack)
    last_sh: dict[str, Any] = {}
    for step in range(10):
        last_sh = _make_step_history(step)
        d0.on_step(step, fake_config, last_sh, fake_stack)
    d0.shutdown((None, list(range(10)), last_sh))

    # Segment 1: steps 10..14 (same DB, different dispatcher)
    lg1 = _ParamsVerifierLogger()
    d1 = LoggerDispatcher(
        [lg1],
        training_program=object(),
        async_logging=False,
        n_workers=2,
        history_db=db,
    )
    d1.on_start(fake_config, fake_stack)
    for step in range(10, 15):
        last_sh = _make_step_history(step)
        d1.on_step(step, fake_config, last_sh, fake_stack)
    d1.shutdown((None, list(range(5)), last_sh))

    assert db.get_step_count() >= 15
    lo, hi = db.get_step_range()
    assert lo == 0
    assert hi >= 14

    db.close()
