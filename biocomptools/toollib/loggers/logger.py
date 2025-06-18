## {{{                          --     imports     --

from typing import List, Tuple, Callable, Union
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

##────────────────────────────────────────────────────────────────────────────}}}


class Logger(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )

    periods: Union[int, List[int]] = 1  # Number of steps between logs or list of periods

    metadata: Dict[str, Any] = {}

    def initialize(self, training_program):
        """Optional initialization before training starts."""
        pass

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Return a list of (period, callback_function) tuples for the training loop."""
        raise NotImplementedError

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Return a dictionary of the latest metrics from this logger.
        If `replicate` is specified, metrics for that single replicate are returned.
        If `replicate` is None, metrics for all replicates are returned (structure
        is logger-dependent, often a list of metrics).
        """
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
