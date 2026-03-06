"""Replay saved step history through loggers without re-running optimization.

Supports two storage backends (auto-detected):
- **DB mode** (``run_history.db``): full-fidelity replay with model, dmanager,
  and ComputeStack reconstruction so every logger works at full capability.
- **Legacy pkl mode** (``step_*.pkl`` files): degraded replay without stack/model
  context (backward compat).

Usage:
    biocomp-replay +biocomp-jobs/replay/diagnostic.yaml ++history_dir=/path/to/run
    biocomp-replay +replay_config.yaml --last-n 100 --final-only
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode

from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.logger_history import (
    BatchData,
    HistoryManager,
    HistoryView,
    LoggerContext,
)
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.designdiagnosticlogger import DesignDiagnosticLogger
from biocomptools.toollib.common import config

logger = get_logger(__name__)


def _get_all_steps(history_dir: Path) -> list[int]:
    """Get sorted list of step numbers from history directory."""
    steps: set[int] = set()
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


def _detect_history_source(history_dir: Path) -> str:
    """Detect whether history_dir contains a DB or legacy pkl files.

    Returns ``"db"`` if ``run_history.db`` is found, ``"pkl"`` if step pkl
    files are found, or raises FileNotFoundError.
    """
    db_path = history_dir / "run_history.db"
    if db_path.exists():
        return "db"
    # Also check parent dir (history_dir might point to step_history_data/)
    parent_db = history_dir.parent / "run_history.db"
    if parent_db.exists():
        return "db"
    if list(history_dir.glob("step_*.pkl")):
        return "pkl"
    raise FileNotFoundError(f"No run_history.db or step_*.pkl files found in {history_dir}")


def _resolve_db_path(history_dir: Path) -> Path:
    """Return the actual path to run_history.db."""
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
    """Replay step history through loggers, auto-detecting DB vs pkl source.

    When a ``run_history.db`` is found, rebuilds full context (model, stack,
    dmanager) so all loggers — including those needing ComputeStack — work.
    """
    source = _detect_history_source(history_dir)

    if source == "db":
        _replay_from_db(history_dir, loggers, output_dir, step_filter, final_only, max_history_len)
    else:
        _replay_from_pkl(history_dir, loggers, output_dir, step_filter, final_only, max_history_len)


def _replay_from_db(
    history_dir: Path,
    loggers: list[Logger],
    output_dir: Path,
    step_filter: Callable[[int], bool] | None,
    final_only: bool,
    max_history_len: int,
) -> None:
    from biocomptools.history_db import RunHistoryDB

    db_path = _resolve_db_path(history_dir)
    db = RunHistoryDB(db_path)

    info = db.load_run_info()
    assert info is not None, f"Empty RunInfo in {db_path}"

    commit_hashes = json.loads(info.commit_hashes_json) if info.commit_hashes_json else {}
    logger.info(f"Replay from DB: run_type={info.run_type}, host={info.host}")
    if commit_hashes:
        logger.info(f"  Commit hashes: {commit_hashes}")

    # Reconstruct context objects
    model = db.load_model()
    dmanager = db.load_dmanager()
    dconfig = db.load_dconfig()
    stack = None

    if model is not None and dmanager is not None:
        stack = _rebuild_stack(dmanager, dconfig, model)
        logger.info(f"  Rebuilt ComputeStack: {stack.get_nb_networks()} networks")
    else:
        logger.warning("  No model/dmanager in DB — replay without stack context")

    # Load steps
    batches = db.load_steps(step_filter=step_filter, show_progress=True)
    logger.info(f"  Loaded {len(batches)} steps from DB")

    _dispatch_replay(
        batches,
        loggers,
        output_dir,
        final_only,
        max_history_len,
        stack=stack,
        model=model,
        dmanager=dmanager,
        extra={"commit_hashes": commit_hashes, "run_type": info.run_type},
    )


def _replay_from_pkl(
    history_dir: Path,
    loggers: list[Logger],
    output_dir: Path,
    step_filter: Callable[[int], bool] | None,
    final_only: bool,
    max_history_len: int,
) -> None:
    logger.info("Replay from legacy pkl files (degraded — no stack/model context)")
    batches = HistoryManager.load_from_step_files(
        history_dir, step_filter=step_filter, show_progress=True
    )
    logger.info(f"  Loaded {len(batches)} steps from pkl files")

    _dispatch_replay(
        batches,
        loggers,
        output_dir,
        final_only,
        max_history_len,
        stack=None,
        model=None,
        dmanager=None,
        extra={},
    )


def _rebuild_stack(
    dmanager: Any,
    dconfig: Any,
    model: Any,
) -> Any:
    """Rebuild ComputeStack from stored model + dmanager, using dconfig params."""
    if dconfig is not None:
        from biocomp.design_prune_controller import build_stack_from_dconf

        return build_stack_from_dconf(dmanager, dconfig, model, lock_ratios=True)
    return dmanager.build_stack(model, unlock_ratios=False)


def _dispatch_replay(
    batches: list[BatchData],
    loggers: list[Logger],
    output_dir: Path,
    final_only: bool,
    max_history_len: int,
    *,
    stack: Any,
    model: Any,
    dmanager: Any,
    extra: dict[str, Any],
) -> None:
    """Iterate batches and dispatch to loggers, honoring scheduling rules."""
    if not batches:
        logger.warning("No steps to replay")
        return

    history = HistoryManager(max_batches=max_history_len)

    # Initialize loggers
    for lg in loggers:
        try:
            lg.initialize(None)
        except Exception as e:
            logger.warning(f"Logger {type(lg).__name__} initialize failed: {e}")

    if not final_only:
        for batch in batches:
            history.append_batch(batch)
            step = batch.step_index

            for lg in loggers:
                if not lg.should_fire(step):
                    continue

                view = history.get_view(window=lg.history_window)
                ctx = LoggerContext(
                    stack=stack,
                    output_dir=output_dir,
                    current_step=step,
                    is_replay=True,
                    dmanager=dmanager,
                    model=model,
                    extra=extra,
                )
                try:
                    lg.on_batch(view, ctx)
                except Exception as e:
                    logger.error(f"on_batch failed for {type(lg).__name__} at step {step}: {e}")
    else:
        for batch in batches:
            history.append_batch(batch)

    # Dispatch on_end for loggers with -1 in call_at
    end_loggers = [lg for lg in loggers if -1 in lg.call_at]
    if end_loggers and batches:
        final_batch = batches[-1]
        view = HistoryView([final_batch])
        ctx = LoggerContext(
            stack=stack,
            output_dir=output_dir,
            current_step=final_batch.step_index,
            is_replay=True,
            is_final=True,
            dmanager=dmanager,
            model=model,
            extra=extra,
        )
        for lg in end_loggers:
            try:
                lg.on_end(view, ctx)
            except Exception as e:
                logger.error(f"on_end failed for {type(lg).__name__}: {e}")

    # Finalize loggers
    for lg in loggers:
        try:
            lg.finalize()
        except Exception as e:
            logger.warning(f"Logger {type(lg).__name__} finalize failed: {e}")


DEFAULT_TYPES = [
    Logger,
    DesignDiagnosticLogger,
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
        """Build step filter function based on selection options."""
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
        """Construct deferred loggers with output directory context."""
        constructed = []
        for lg in self.loggers:
            if isinstance(lg, DeferredNode):
                lg = lg.construct(context={"output_dir": str(output_dir), "save_dir": output_dir})
            if hasattr(lg, "output_dir") and lg.output_dir is None:
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

        # Configure loggers for replay
        for lg in loggers:
            if hasattr(lg, "call_at_interval"):
                lg.call_at_interval = self.period
            if hasattr(lg, "history_window"):
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
