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


def test_shutdown_still_shuts_down_on_invalid_step_history_result():
    from biocomptools.logger_dispatch import LoggerDispatcher

    dispatcher = LoggerDispatcher.__new__(LoggerDispatcher)
    dispatcher._inline_loggers = []
    dispatcher._runner_loggers = []
    dispatcher._all_loggers = []
    dispatcher._last_config = None
    dispatcher._history_db = None
    dispatcher._step_writer = None
    dispatcher._runner = None
    dispatcher._runner_thread = None
    dispatcher._training_program = None
    dispatcher._base_dir = None
    dispatcher._inline_history = None

    with pytest.raises(TypeError, match="result\\[2\\]"):
        dispatcher.shutdown((object(), [0.1], [{"loss": 0.1}]))


def test_integration_shutdown_dispatches_on_end_to_inline_logger():
    from biocomptools.logger_dispatch import LoggerDispatcher

    class _CaptureEndLogger(Logger):
        execution_mode: str = "inline"
        call_at: list[int] = [-1]
        captured_loss: float | None = None
        captured_step: int | None = None

        def on_end(self, view, context):
            self.captured_step = context.current_step
            sh = view.to_step_history()
            self.captured_loss = float(sh.get("loss"))

    lg = _CaptureEndLogger()
    dispatcher = LoggerDispatcher(
        [lg],
        training_program=object(),
        async_logging=False,
        n_workers=1,
    )
    # In the new architecture, on_end is dispatched via shutdown()
    result = (object(), [0.3, 0.2, 0.1], StepHistorySnapshot.from_raw({"loss": 0.1}))
    dispatcher.shutdown(result)
    assert lg.captured_step == 3
    assert lg.captured_loss == pytest.approx(0.1)
