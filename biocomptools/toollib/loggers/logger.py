## {{{                          --     imports     --

from typing import Callable, Literal
from pydantic import BaseModel, ConfigDict

from biocomptools.logger_history import HistoryView, LoggerContext

##────────────────────────────────────────────────────────────────────────────}}}


class Logger(BaseModel):
    """Base class for all loggers.

    Override on_batch() and/or on_end() to implement logger behavior.

    Scheduling:
        call_at_interval: Periodic firing every N steps. None = no periodic calls.
        call_at: Specific step numbers. 0 = before first step, -1 = after last step,
                 positive = after that step. Default [-1] (end only).
        The final call schedule is the union of both.

    Declarative attributes:
        history_window: Number of steps to retain in history (None = all)
        required_metrics: Metric keys to include in HistoryView
        required_arrays: Array keys to include in HistoryView
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    # scheduling
    call_at_interval: int | None = None
    call_at: list[int] = [-1]

    # execution mode
    execution_mode: Literal["inline", "thread", "process"] = "thread"
    metadata: dict[str, object] = {}

    # declarative attributes
    history_window: int | None = None  # number of steps to keep (None = all)
    required_metrics: list[str] = []
    required_arrays: list[str] = []
    required_extra: list[str] = []

    _call_at_set: frozenset[int] = frozenset()

    def model_post_init(self, __context: object) -> None:
        self._call_at_set = frozenset(self.call_at)

    def should_fire(self, step: int) -> bool:
        """Whether this logger should fire at the given step (excluding start/end)."""
        if step <= 0:
            return False
        interval = self.call_at_interval
        if interval is not None and interval > 0 and step % interval == 0:
            return True
        return step in self._call_at_set

    def should_fire_start(self) -> bool:
        return 0 in self._call_at_set

    def should_fire_end(self) -> bool:
        return -1 in self._call_at_set

    def initialize(self, training_program: object) -> None:
        pass

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        pass

    def on_start(self, context: LoggerContext) -> None:
        pass

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        pass

    def get_metrics(self, replicate: int | None = None) -> dict[str, object] | None:
        return None

    def finalize(self) -> None:
        pass

    def find_myself(self, training_program: object = None) -> int:
        if not training_program:
            return 0
        loggers = getattr(training_program, "loggers", [])
        for i, logger_obj in enumerate(loggers):
            if logger_obj is self:
                return i
        return 0


class FunctionLogger(Logger):
    """Legacy logger that wraps raw callback functions.

    Retained for backward compatibility with HyperoptTrainingLogger and
    direct get_callbacks() usage outside LoggerDispatcher.
    """

    call_at_interval: int | dict[str, int] | None = None
    functions: list[Callable[..., object]] = []

    def get_callbacks(self, training_program: object) -> list[tuple[int, Callable[..., object]]]:
        interval = self.call_at_interval
        if isinstance(interval, dict):
            callbacks: list[tuple[int, Callable[..., object]]] = []
            for fn in self.functions:
                fn_interval = interval.get(fn.__name__)
                if fn_interval is not None:
                    callbacks.append((fn_interval, fn))
            return callbacks
        elif isinstance(interval, int):
            return [(interval, fn) for fn in self.functions]
        else:
            return []
