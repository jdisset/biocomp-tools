"""Replay saved step history through loggers without re-running optimization.

Usage:
    biocomp-replay +biocomp-jobs/replay/diagnostic.yaml ++history_dir=/path/to/run/step_history_data
    biocomp-replay +replay_config.yaml --last-n 100 --final-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode

from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.async_logger_handler import AsyncLoggerHandler
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
from biocomptools.toollib.common import config

logger = get_logger(__name__)


def _get_all_steps(history_dir: Path) -> list[int]:
    """Get sorted list of step numbers from history directory."""
    steps = set()
    for f in history_dir.glob("step_*.pkl"):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        if len(parts) > 2 and parts[2] in ("end", "start"):
            continue
        try:
            steps.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(steps)


DEFAULT_TYPES = [
    Logger,
    DesignDiagnosticLogger,
]


@dracon_program(
    name='biocomp-replay',
    description='Replay saved step history through loggers.',
    context_types=DEFAULT_TYPES,
    context={'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve()},
)
class ReplayJob(BaseModel):
    """Replay step history data through loggers for visualization iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_dir: Annotated[Path | None, Arg(help='Directory containing step_*.pkl files')] = None
    loggers: Annotated[
        list[DeferredNode[Logger] | Logger], Arg(help='Loggers to replay through')
    ] = Field(default_factory=list)
    output_dir: Annotated[
        Path | None, Arg(help='Output directory (default: history_dir parent)')
    ] = None

    # Step selection options (mutually exclusive)
    last_n: Annotated[int | None, Arg(help='Only replay the last N steps')] = None
    first_n: Annotated[int | None, Arg(help='Only replay the first N steps')] = None
    step_range: Annotated[
        tuple[int, int] | None, Arg(help='Replay steps in range [start, end]')
    ] = None
    period: Annotated[int, Arg(help='Process every Nth step')] = 1

    final_only: Annotated[
        bool, Arg(help='Only call on_end with accumulated history (single output)')
    ] = False
    max_history_len: Annotated[int, Arg(help='Max steps to keep in memory')] = 10000
    n_workers: Annotated[int, Arg(help='Thread pool size')] = 8

    @model_validator(mode='after')
    def _validate_step_selection(self):
        opts = [self.last_n, self.first_n, self.step_range]
        if sum(o is not None for o in opts) > 1:
            raise ValueError("Only one of last_n, first_n, step_range can be specified")
        return self

    def _build_step_filter(self) -> callable | None:
        """Build step filter function based on selection options."""
        if self.last_n is not None:
            steps = _get_all_steps(self.history_dir)
            if self.last_n >= len(steps):
                return None
            min_step = steps[-self.last_n]
            return lambda s: s >= min_step

        if self.first_n is not None:
            steps = _get_all_steps(self.history_dir)
            if self.first_n >= len(steps):
                return None
            max_step = steps[self.first_n - 1]
            return lambda s: s <= max_step

        if self.step_range is not None:
            start, end = self.step_range
            return lambda s: start <= s <= end

        return None

    def _construct_loggers(self, output_dir: Path) -> list[Logger]:
        """Construct deferred loggers with output directory context."""
        constructed = []
        for lg in self.loggers:
            if isinstance(lg, DeferredNode):
                lg = lg.construct(context={'output_dir': str(output_dir), 'save_dir': output_dir})
            # Set output_dir if logger has the attribute
            if hasattr(lg, 'output_dir') and lg.output_dir is None:
                lg.output_dir = str(output_dir)
            constructed.append(lg)
        return constructed

    def run(self) -> dict:
        """Execute replay and return summary."""
        if self.history_dir is None:
            raise ValueError("history_dir is required. Use --history-dir or ++history_dir")

        history_dir = Path(self.history_dir).expanduser().resolve()

        if not history_dir.exists():
            raise FileNotFoundError(f"History directory not found: {history_dir}")

        # Determine output directory
        if self.output_dir:
            output_dir = Path(self.output_dir).expanduser().resolve()
        else:
            output_dir = history_dir.parent / "replay_output"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build step filter
        step_filter = self._build_step_filter()

        # Construct loggers
        loggers = self._construct_loggers(output_dir)
        if not loggers:
            logger.warning("No loggers specified - nothing to replay")
            return {'status': 'no_loggers', 'output_dir': str(output_dir)}

        # Configure loggers for replay
        for lg in loggers:
            if hasattr(lg, 'frequency'):
                lg.frequency = self.period
            if hasattr(lg, 'periods'):
                lg.periods = self.period
            if hasattr(lg, 'history_window'):
                lg.history_window = self.max_history_len
            if hasattr(lg, 'final_figure_only'):
                lg.final_figure_only = self.final_only
            if hasattr(lg, 'call_at_end'):
                lg.call_at_end = True

        # Count available steps
        all_steps = _get_all_steps(history_dir)
        filtered_steps = [s for s in all_steps if step_filter is None or step_filter(s)]

        logger.info(f"Replaying {len(filtered_steps)} steps through {len(loggers)} logger(s)")
        logger.info(f"  History dir: {history_dir}")
        logger.info(f"  Output dir: {output_dir}")
        if self.final_only:
            logger.info("  Mode: final_only (single consolidated output)")

        # Execute replay
        AsyncLoggerHandler.replay(
            history_dir=history_dir,
            loggers=loggers,
            step_filter=step_filter,
            final_only=self.final_only,
            n_workers=self.n_workers,
        )

        logger.info("Replay completed successfully")
        return {
            'status': 'success',
            'output_dir': str(output_dir),
            'steps_processed': len(filtered_steps),
            'loggers': [type(lg).__name__ for lg in loggers],
        }


def main():
    setup_logging()
    result = ReplayJob.cli()  # .cli() calls .run() and returns its result
    logger.info(f"Result: {result}")


if __name__ == '__main__':
    main()
