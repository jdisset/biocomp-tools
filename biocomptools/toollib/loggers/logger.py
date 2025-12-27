## {{{                          --     imports     --

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Callable, Union, Literal
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

if TYPE_CHECKING:
    from biocomptools.logger_history import HistoryView, LoggerContext

##────────────────────────────────────────────────────────────────────────────}}}


class Logger(BaseModel):
    """Base class for all loggers.

    Supports two patterns:
    1. Legacy: Override get_callbacks() to return (period, callback) tuples
    2. New: Set declarative attributes and override on_batch/on_end methods

    New pattern attributes:
        frequency: Callback frequency in steps (1 = every step)
        history_window: Number of steps to retain in history (None = all)
        history_mode: "window" (last N), "since_last" (since last callback), "all"
        required_metrics: Metric keys to include in HistoryView
        required_arrays: Array keys to include in HistoryView
        call_at_start: Whether to call on_start at beginning
        call_at_end: Whether to call on_end at end
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    # legacy attributes (still supported)
    periods: Union[int, List[int]] = 1
    async_ok: bool = True
    parallel_ok: bool = False
    metadata: Dict[str, Any] = {}

    # new declarative attributes
    frequency: int = 1  # steps between callbacks (alias for periods when int)
    history_window: Optional[int] = None  # number of steps to keep (None = all)
    history_mode: Literal["window", "since_last", "all"] = "window"
    required_metrics: List[str] = []
    required_arrays: List[str] = []
    call_at_start: bool = False
    call_at_end: bool = False

    # internal tracking
    _last_callback_step: int = -1
    _uses_new_pattern: bool = False

    def model_post_init(self, __context):
        # detect if subclass uses new pattern (overrides on_batch or on_end)
        cls = type(self)
        self._uses_new_pattern = (
            cls.on_batch is not Logger.on_batch or cls.on_end is not Logger.on_end
        )

    def initialize(self, training_program):
        """Optional initialization before training starts."""
        pass

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Return a list of (period, callback_function) tuples for the training loop.

        Legacy pattern - override this for custom callback behavior.
        For new pattern, override on_batch and/or on_end instead.
        """
        if self._uses_new_pattern:
            # return empty - handler will call on_batch/on_end directly
            return []
        raise NotImplementedError("Subclass must implement get_callbacks or on_batch/on_end")

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        """Called every `frequency` steps with requested history.

        New pattern - override this instead of get_callbacks for simpler code.

        Args:
            view: HistoryView containing requested metrics/arrays
            context: LoggerContext with training state
        """
        pass

    def on_start(self, context: LoggerContext) -> None:
        """Called at training start if call_at_start=True."""
        pass

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        """Called at training end if call_at_end=True.

        Receives full history (or windowed based on history_window).
        """
        pass

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return a dictionary of the latest metrics from this logger."""
        return None

    def finalize(self):
        """Optional cleanup after training ends."""
        pass


class FunctionLogger(Logger):
    functions: List[Callable] = []

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        if isinstance(self.periods, int):
            self.periods = [self.periods]
        assert isinstance(self.periods, list)
        if len(self.periods) == 1:
            self.periods = self.periods * len(self.functions)

        assert len(self.periods) == len(self.functions), (
            f"Number of periods in FunctionLogger ({len(self.periods)}) must match number of functions ({len(self.functions)})"
        )

        return list(zip(self.periods, self.functions))
