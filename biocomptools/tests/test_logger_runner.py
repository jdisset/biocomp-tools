"""Tests for LoggerRunner — unified live + replay dispatch."""

import threading
import time

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

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._steps_seen = []
        self._end_called = False

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest:
            self._steps_seen.append(latest.step_index)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self._end_called = True


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

    # Should fire at steps 1-9 (should_fire skips step 0)
    assert len(recorder._steps_seen) == 9
    assert recorder._end_called


def test_replay_respects_interval(db):
    _populate_db(db, 10)

    recorder = RecordingLogger(call_at_interval=3)
    runner = LoggerRunner(db=db, loggers=[recorder], mode="replay")
    runner.run()

    # Steps 3, 6, 9 fire (interval=3, skip 0); sort because thread dispatch order varies
    assert sorted(recorder._steps_seen) == [3, 6, 9]


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
