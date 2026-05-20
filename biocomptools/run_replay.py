# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Replay saved step history through loggers without re-running optimization."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode

from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
from biocomptools.toollib.loggers.designheatmaplogger import DesignHeatmapLogger
from biocomptools.toollib.loggers.designcardlogger import DesignCardLogger
from biocomptools.toollib.loggers.designlosshistorylogger import DesignLossHistoryLogger
from biocomptools.toollib.figuremakers.networkdiagram import LayoutSpec
from jeanplot.core.models import Size
from biocomptools.toollib.common import config

logger = get_logger(__name__)


def _resolve_db_path(history_dir: Path) -> Path:
    db_path = history_dir / "run_history.db"
    if db_path.exists():
        return db_path
    parent_db = history_dir.parent / "run_history.db"
    if parent_db.exists():
        return parent_db
    raise FileNotFoundError(f"run_history.db not found near {history_dir}")


def replay_history(
    history_dir: Path,
    loggers: list[Logger],
    output_dir: Path,
    step_filter: Callable[[int], bool] | None = None,
    final_only: bool = False,
    max_history_len: int = 10000,
) -> None:
    from biocomptools.history_db import RunHistoryDB
    from biocomptools.logger_runner import LoggerRunner

    db_path = _resolve_db_path(history_dir)
    db = RunHistoryDB(db_path, read_only=True)

    info = db.load_run_info()
    assert info is not None, f"Empty RunInfo in {db_path}"

    commit_hashes = json.loads(info.commit_hashes_json) if info.commit_hashes_json else {}
    logger.info(f"Replay from DB: run_type={info.run_type}, host={info.host}")
    if commit_hashes:
        logger.info(f"  Commit hashes: {commit_hashes}")

    runner = LoggerRunner(
        db=db,
        loggers=loggers,
        mode="replay",
        output_dir=output_dir,
    )
    runner.run()


DEFAULT_TYPES = [
    Logger,
    DesignDiagnosticLogger,
    DesignHeatmapLogger,
    DesignCardLogger,
    DesignLossHistoryLogger,
    LayoutSpec,
    Size,
]


@dracon_program(
    name="biocomp-replay",
    description="Replay saved step history through loggers.",
    context_types=DEFAULT_TYPES,
    context={"BIOCOMP_ROOT": Path(config.paths.root).expanduser().resolve()},
)
class ReplayJob(BaseModel):
    """Replay step history data through loggers for visualization iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_dir: Annotated[
        Path | None, Arg(help="Directory containing step history (DB or pkl)")
    ] = None
    loggers: Annotated[
        list[DeferredNode[Logger] | Logger], Arg(help="Loggers to replay through")
    ] = Field(default_factory=list)
    output_dir: Annotated[
        Path | None, Arg(help="Output directory (default: history_dir parent)")
    ] = None

    # Step selection options (mutually exclusive)
    last_n: Annotated[int | None, Arg(help="Only replay the last N steps")] = None
    first_n: Annotated[int | None, Arg(help="Only replay the first N steps")] = None
    step_range: Annotated[
        tuple[int, int] | None, Arg(help="Replay steps in range [start, end]")
    ] = None
    period: Annotated[int, Arg(help="Process every Nth step")] = 1

    final_only: Annotated[
        bool, Arg(help="Only call on_end with accumulated history (single output)")
    ] = False
    max_history_len: Annotated[int, Arg(help="Max steps to keep in memory")] = 10000
    n_workers: Annotated[int, Arg(help="Thread pool size")] = 8

    @model_validator(mode="after")
    def _validate_step_selection(self):
        opts = [self.last_n, self.first_n, self.step_range]
        if sum(o is not None for o in opts) > 1:
            raise ValueError("Only one of last_n, first_n, step_range can be specified")
        return self

    def _build_step_filter(self, history_dir: Path) -> Callable[[int], bool] | None:
        if self.last_n is not None:
            steps = _get_all_steps(history_dir)
            if not steps or self.last_n >= len(steps):
                return None
            min_step = steps[-self.last_n]
            return lambda s: s >= min_step

        if self.first_n is not None:
            steps = _get_all_steps(history_dir)
            if not steps or self.first_n >= len(steps):
                return None
            max_step = steps[self.first_n - 1]
            return lambda s: s <= max_step

        if self.step_range is not None:
            start, end = self.step_range
            return lambda s: start <= s <= end

        return None

    def _construct_loggers(self, output_dir: Path) -> list[Logger]:
        constructed = []
        for lg in self.loggers:
            if isinstance(lg, DeferredNode):
                lg = lg.construct(context={"output_dir": str(output_dir), "save_dir": output_dir})
            if hasattr(lg, "output_dir") and lg.output_dir is None:
                lg.output_dir = str(output_dir)
            constructed.append(lg)
        return constructed

    def run(self) -> dict:
        if self.history_dir is None:
            raise ValueError("history_dir is required. Use --history-dir or ++history_dir")

        history_dir = Path(self.history_dir).expanduser().resolve()

        if not history_dir.exists():
            raise FileNotFoundError(f"History directory not found: {history_dir}")

        if self.output_dir:
            output_dir = Path(self.output_dir).expanduser().resolve()
        else:
            output_dir = history_dir.parent / "replay_output"

        output_dir.mkdir(parents=True, exist_ok=True)

        step_filter = self._build_step_filter(history_dir)

        loggers = self._construct_loggers(output_dir)
        if not loggers:
            logger.warning("No loggers specified - nothing to replay")
            return {"status": "no_loggers", "output_dir": str(output_dir)}

        # Override call_at_interval/history_window only if CLI is more restrictive.
        for lg in loggers:
            if self.period > (lg.call_at_interval or self.period):
                lg.call_at_interval = self.period
            lg_window = lg.history_window or float("inf")
            if self.max_history_len < lg_window:
                lg.history_window = self.max_history_len
            if hasattr(lg, "final_figure_only"):
                lg.final_figure_only = self.final_only

        source = _detect_history_source(history_dir)
        logger.info(f"Detected history source: {source}")
        logger.info(f"  History dir: {history_dir}")
        logger.info(f"  Output dir: {output_dir}")
        if self.final_only:
            logger.info("  Mode: final_only (single consolidated output)")

        replay_history(
            history_dir=history_dir,
            loggers=loggers,
            output_dir=output_dir,
            step_filter=step_filter,
            final_only=self.final_only,
            max_history_len=self.max_history_len,
        )

        logger.info("Replay completed successfully")
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "source": source,
            "loggers": [type(lg).__name__ for lg in loggers],
        }


def main():
    setup_logging()
    result = ReplayJob.cli()
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
