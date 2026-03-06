"""End-to-end tests for async logger handler optimizations.

Tests exercise the full LoggerDispatcher → AsyncLoggerHandler → consumer thread →
thread pool dispatch pipeline with synthetic training loops and capture loggers.

Validates three fixes:
  Fix 1: In-memory queue (no disk serialization on main thread)
  Fix 2: Cached stack/training_config (not re-serialized per step)
  Fix 3: Params not accumulated in HistoryManager (single-slot cache + injection)
"""

from __future__ import annotations

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
    async_ok: bool = True

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
    async_ok: bool = True

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
    async_ok: bool = True

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
    async_ok: bool = True

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
    async_ok: bool = True

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
    async_ok: bool = True

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
    keep_history_on_disk: bool = False,
    save_all_steps: bool = False,
    n_workers: int = 2,
    async_store_location: str | None = None,
) -> LoggerDispatcher:
    """Run synthetic training loop through the full dispatcher machinery."""
    fake_stack = {"layers": [1, 2, 3]}
    fake_config = {"lr": 0.01, "epochs": 5}

    dispatcher = LoggerDispatcher(
        loggers,
        training_program=object(),
        async_logging=async_logging,
        n_workers=n_workers,
        keep_history_on_disk=keep_history_on_disk,
        save_all_steps=save_all_steps,
        async_store_location=async_store_location,
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


def test_disk_save_excludes_stack(tmp_path):
    """Fix 2: saved step files omit stack and training_config."""
    import dill

    lg = _ParamsVerifierLogger()
    store_dir = str(tmp_path / "step_history")

    _run_simulated_training(
        [lg],
        n_steps=5,
        keep_history_on_disk=True,
        async_store_location=store_dir,
    )

    pkl_files = sorted((tmp_path / "step_history").glob("step_*.pkl"))
    assert len(pkl_files) > 0, f"No step files found in {store_dir}"

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = dill.load(f)
        assert "stack" not in data, f"{pkl_file.name} contains 'stack'"
        assert "training_config" not in data, f"{pkl_file.name} contains 'training_config'"
        assert "step" in data
        assert "timestamp" in data


def test_step_offset_produces_global_filenames(tmp_path):
    """Two hard-pruning segments writing to same dir produce non-overlapping step files.

    Segment 0: steps 0-9 (offset=0) → step_000000..step_000009
    Segment 1: steps 10-14 (offset=10) → step_000010..step_000014
    No filename collisions.
    """
    store_dir = str(tmp_path / "step_history")
    fake_stack = {"layers": [1, 2, 3]}
    fake_config = {"lr": 0.01}

    # Segment 0: steps 0..9
    lg0 = _ParamsVerifierLogger()
    d0 = LoggerDispatcher(
        [lg0],
        training_program=object(),
        async_logging=True,
        keep_history_on_disk=True,
        save_all_steps=True,
        async_store_location=store_dir,
        n_workers=2,
    )
    d0.on_start(fake_config, fake_stack)
    last_sh: dict[str, Any] = {}
    for step in range(10):
        last_sh = _make_step_history(step)
        d0.on_step(step, fake_config, last_sh, fake_stack)
    d0.on_end(9, fake_config, last_sh, fake_stack)
    d0.shutdown((None, list(range(10)), last_sh))

    # Segment 1: steps 10..14 (same directory, different dispatcher)
    lg1 = _ParamsVerifierLogger()
    d1 = LoggerDispatcher(
        [lg1],
        training_program=object(),
        async_logging=True,
        keep_history_on_disk=True,
        save_all_steps=True,
        async_store_location=store_dir,
        n_workers=2,
    )
    d1.on_start(fake_config, fake_stack)
    for step in range(10, 15):
        last_sh = _make_step_history(step)
        d1.on_step(step, fake_config, last_sh, fake_stack)
    d1.on_end(14, fake_config, last_sh, fake_stack)
    d1.shutdown((None, list(range(5)), last_sh))

    files = sorted((tmp_path / "step_history").glob("step_*.pkl"))
    steps = [int(f.stem.split("_")[1]) for f in files]
    assert steps == list(range(15)), f"Expected 0..14, got {steps}"
