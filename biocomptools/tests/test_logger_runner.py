"""Tests for LoggerRunner — unified live + replay dispatch."""

import threading
import time
from typing import Literal

import numpy as np
import pytest

from biocomptools.history_db import RunHistoryDB
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logger_runner import LoggerRunner
from biocomptools.toollib.loggers.logger import Logger


class RecordingLogger(Logger):
    """Test logger that records on_batch calls."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]
    _steps_seen: list[int] = []
    _end_called: bool = False
    _db_refs: list = []

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._steps_seen = []
        self._end_called = False
        self._db_refs = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            self._steps_seen.append(latest.step_index)
        self._db_refs.append(context.db)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self._end_called = True
        self._db_refs.append(context.db)


@pytest.fixture
def db(tmp_path):
    return RunHistoryDB(tmp_path / "test.db")


def _populate_db(db, n_steps=10):
    db.save_run_info(run_type="test")
    for s in range(n_steps):
        db.save_step(s, float(s), float(s) * 0.1)
        db.save_scalars(s, {"lr": 0.001 * s})
    db.commit()
    db.mark_finished()


def test_replay_mode(db):
    _populate_db(db, 10)

    recorder = RecordingLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    # Should fire at steps 0-9 (step 0 fires for loggers with call_at_interval)
    assert len(recorder._steps_seen) == 10
    assert recorder._end_called


def test_replay_respects_interval(db):
    _populate_db(db, 10)

    recorder = RecordingLogger(call_at_interval=3)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    # Steps 0, 3, 6, 9 fire (interval=3, step 0 always fires with interval)
    assert sorted(recorder._steps_seen) == [0, 3, 6, 9]


def test_replay_on_end_fires(db):
    _populate_db(db, 5)

    recorder = RecordingLogger(call_at_interval=None, call_at=[-1])
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    assert recorder._end_called
    assert len(recorder._steps_seen) == 0  # no on_batch calls


def test_live_mode_polls_and_exits(db):
    """Test live mode: writer in background, runner polls and exits on finish."""
    db.save_run_info(run_type="test")

    recorder = RecordingLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="live", poll_interval=0.05)

    def _writer():
        time.sleep(0.1)
        for s in range(5):
            db.save_step(s, float(s), float(s) * 0.1)
            db.commit()
            time.sleep(0.05)
        db.mark_finished()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    runner.run()
    writer_thread.join(timeout=5)

    assert len(recorder._steps_seen) > 0
    assert recorder._end_called


def test_empty_replay(db):
    db.save_run_info(run_type="test")
    db.mark_finished()

    recorder = RecordingLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    assert len(recorder._steps_seen) == 0


def test_context_passes_db_ref(db):
    """LoggerContext.db should reference the RunHistoryDB in replay mode."""
    _populate_db(db, 3)

    recorder = RecordingLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    assert len(recorder._db_refs) > 0
    assert all(ref is db for ref in recorder._db_refs)


class ArrayBlobLogger(Logger):
    """Logger that requests both array and blob keys."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]
    required_arrays: list[str] = ["yhatdep", "latest_params"]
    _received_keys: list[set[str]] = []

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._received_keys = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            self._received_keys.append(set(latest.arrays.keys()))

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        pass


def test_build_view_routes_array_and_blob_keys(db):
    """required_arrays are split into array_keys and blob_keys by DB table."""
    db.save_run_info(run_type="test")
    for s in range(3):
        db.save_step(s, float(s), float(s) * 0.1)
        # yhatdep -> step_array, latest_params -> step_blob
        db.save_arrays(s, {"yhatdep": np.zeros(4)})
        db.save_blobs(s, {"latest_params": {"w": np.ones(2)}})
        # Also save a blob we do NOT want loaded
        db.save_blobs(s, {"opt_state": {"big_data": np.ones(10000)}})
    db.commit()
    db.mark_finished()

    lg = ArrayBlobLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[lg], mode="replay")
    runner.run()

    assert len(lg._received_keys) > 0
    for keys in lg._received_keys:
        assert "yhatdep" in keys, "array key 'yhatdep' should be loaded"
        assert "latest_params" in keys, "blob key 'latest_params' should be loaded"
        assert "opt_state" not in keys, "unrequested blob 'opt_state' must NOT be loaded"


class InlineLogger(Logger):
    """Logger with inline execution mode that records its call thread."""

    model_config = Logger.model_config.copy()
    model_config["extra"] = "allow"

    call_at_interval: int | None = 1
    call_at: list[int] = [-1]
    execution_mode: Literal["inline", "thread", "process"] = "inline"
    _call_threads: list[str] = []

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._call_threads = []

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        import threading

        self._call_threads.append(threading.current_thread().name)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        pass


def test_inline_execution_runs_on_main_thread(db):
    """Inline loggers should run on the dispatch thread, not in the thread pool."""
    _populate_db(db, 5)

    lg = InlineLogger(call_at_interval=1)
    runner = LoggerRunner(db=db, loggers=[lg], mode="replay")
    runner.run()

    assert len(lg._call_threads) > 0
    for name in lg._call_threads:
        assert "logger_thread" not in name, (
            f"Inline logger ran in thread pool thread '{name}'"
        )
