from typing import Any, cast

import pytest

from biocomp.step_history import StepHistorySnapshot
from biocomptools.logger_dispatch import (
    _extract_final_step_history,
    _extract_final_step_index,
)
from biocomptools.toollib.loggers.logger import Logger


def test_extract_final_step_history_accepts_dict():
    result = (object(), [0.1, 0.05], {"loss": 0.05, "yhatdep": object()})
    step_history = _extract_final_step_history(result)
    assert isinstance(step_history, StepHistorySnapshot)
    assert step_history["loss"] == 0.05


def test_extract_final_step_history_rejects_list():
    result = (object(), [0.1], [{"loss": 0.1}])
    with pytest.raises(TypeError, match="expected result\\[2\\]"):
        _extract_final_step_history(result)


def test_extract_final_step_history_accepts_snapshot():
    result = (object(), [0.1], StepHistorySnapshot.from_raw({"loss": 0.1}))
    step_history = _extract_final_step_history(result)
    assert isinstance(step_history, StepHistorySnapshot)
    assert step_history["loss"] == 0.1


def test_extract_final_step_index_uses_loss_history_length():
    result = (object(), [0.3, 0.2, 0.1], {"loss": 0.1})
    assert _extract_final_step_index(result) == 3


def test_shutdown_raises_sync_callback_error_and_still_shuts_down():
    from biocomptools.logger_dispatch import LoggerDispatcher

    class _StubAsyncHandler:
        def __init__(self) -> None:
            self.shutdown_called = False

        def process_end_loggers(self, **kwargs) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    def failing_callback(step, config, step_history=None, stack=None):
        raise RuntimeError("boom")

    dispatcher = LoggerDispatcher.__new__(LoggerDispatcher)
    stub_handler = _StubAsyncHandler()
    dispatcher._async_handler = cast(Any, stub_handler)
    dispatcher._sync_callbacks = []

    with pytest.raises(RuntimeError, match="boom"):
        dispatcher.shutdown(
            (object(), [0.1], {"loss": 0.1}),
            sync_callbacks=[(-1, failing_callback)],
        )

    assert stub_handler.shutdown_called is True


def test_shutdown_still_shuts_down_on_invalid_step_history_result():
    from biocomptools.logger_dispatch import LoggerDispatcher

    class _StubAsyncHandler:
        def __init__(self) -> None:
            self.shutdown_called = False

        def process_end_loggers(self, **kwargs) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    dispatcher = LoggerDispatcher.__new__(LoggerDispatcher)
    stub_handler = _StubAsyncHandler()
    dispatcher._async_handler = cast(Any, stub_handler)
    dispatcher._sync_callbacks = []

    with pytest.raises(TypeError, match="result\\[2\\]"):
        dispatcher.shutdown((object(), [0.1], [{"loss": 0.1}]))

    assert stub_handler.shutdown_called is True


def test_integration_shutdown_end_callback_receives_snapshot():
    from biocomptools.logger_dispatch import LoggerDispatcher

    class _CaptureEndLogger(Logger):
        async_ok: bool = False
        captured_loss: float | None = None
        captured_type: str | None = None
        captured_step: int | None = None

        def get_callbacks(self, training_program: object):
            def end_callback(step, training_config, step_history=None, stack=None, **kwargs):
                assert step_history is not None
                self.captured_step = step
                self.captured_type = type(step_history).__name__
                self.captured_loss = float(step_history.get("loss"))

            return [(-1, end_callback)]

    logger = _CaptureEndLogger()
    dispatcher = LoggerDispatcher(
        [logger],
        training_program=object(),
        async_logging=False,
        n_workers=1,
    )
    dispatcher.on_end(
        3,
        None,
        StepHistorySnapshot.from_raw({"loss": 0.1}),
        None,
    )
    assert logger.captured_step == 3
    assert logger.captured_type == "StepHistorySnapshot"
    assert logger.captured_loss == pytest.approx(0.1)
