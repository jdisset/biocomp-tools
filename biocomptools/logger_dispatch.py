"""Logger dispatcher: StepWriter + LoggerRunner + inline dispatch.

Partitions loggers by execution_mode:
- "inline": dispatched directly by the optimization loop (GPU-bound loggers)
- "thread"/"process": dispatched by LoggerRunner via DB polling (live mode)
"""

import threading
from pathlib import Path
from typing import Any

from biocomp.logger_dispatch import LoggerDispatch
from biocomp.step_history import StepHistoryLike, ensure_step_history_snapshot
from biocomptools.logger_history import (
    BatchData,
    HistoryManager,
    HistoryView,
    LoggerContext,
)
from biocomptools.logger_runner import LoggerRunner
from biocomptools.logging_config import get_logger
from biocomptools.step_writer import StepWriter
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.write_policy import WritePolicy

logger = get_logger(__name__)


def _extract_final_step_history(result: object):
    if not isinstance(result, tuple) or len(result) <= 2:
        raise TypeError(
            "Optimization result invariant violated: expected a tuple with step_history at index 2."
        )
    try:
        return ensure_step_history_snapshot(result[2], context="result[2] (step_history)")
    except TypeError as exc:
        raise TypeError(
            "Optimization result invariant violated: expected result[2] (step_history) "
            "to be mapping-like."
        ) from exc


def _extract_final_step_index(result: object) -> int:
    if not isinstance(result, tuple) or len(result) <= 1:
        return 0
    losses = result[1]
    if isinstance(losses, list):
        return len(losses)
    return 0


class LoggerDispatcher(LoggerDispatch):
    """Unified logger dispatcher.

    Partitions loggers into:
    - inline (execution_mode="inline"): dispatched directly in optimization loop
    - runner (execution_mode="thread"|"process"): dispatched by LoggerRunner via DB
    """

    def __init__(
        self,
        loggers: list[Logger],
        *,
        training_program: object,
        async_logging: bool = True,
        base_dir: Path | None = None,
        n_workers: int = 8,
        history_db: Any = None,
        write_policy: WritePolicy | None = None,
    ):
        self._training_program = training_program
        self._last_config: object = None
        self._base_dir = base_dir
        self._history_db = history_db

        all_loggers = [lg for lg in loggers if isinstance(lg, Logger)]

        inline_loggers: list[Logger] = []
        runner_loggers: list[Logger] = []
        for lg in all_loggers:
            if lg.execution_mode == "inline":
                inline_loggers.append(lg)
            else:
                runner_loggers.append(lg)

        self._inline_loggers = inline_loggers
        self._runner_loggers = runner_loggers
        self._all_loggers = all_loggers

        # Initialize inline loggers synchronously
        for lg in inline_loggers:
            lg.initialize(training_program)

        # StepWriter: optimization loop → DB
        self._step_writer: StepWriter | None = None
        if history_db is not None:
            self._step_writer = StepWriter(
                db=history_db,
                policy=write_policy or WritePolicy(),
            )

        # LoggerRunner: DB → non-inline loggers (live mode)
        self._runner: LoggerRunner | None = None
        self._runner_thread: threading.Thread | None = None
        if runner_loggers and history_db is not None and async_logging:
            self._runner = LoggerRunner(
                db=history_db,
                loggers=runner_loggers,
                mode="live",
                output_dir=base_dir,
                thread_workers=min(n_workers, 8),
                process_workers=min(2, max(1, n_workers // 4)),
                training_program=training_program,
            )
            self._runner_thread = threading.Thread(
                target=self._runner.run, daemon=True, name="logger_runner"
            )
            self._runner_thread.start()
            logger.info(
                f"LoggerRunner started: {len(runner_loggers)} loggers "
                f"({len(inline_loggers)} inline)"
            )
        elif runner_loggers and (history_db is None or not async_logging):
            # Fallback: no DB or async disabled — use legacy in-process dispatch
            logger.info("No DB or async disabled — runner loggers dispatched inline")
            self._inline_loggers.extend(runner_loggers)
            self._runner_loggers = []
            for lg in runner_loggers:
                lg.initialize(training_program)

        # Inline history manager (for sync dispatch)
        self._inline_history = HistoryManager(max_batches=10000)

    def on_start(self, config: object, stack: object) -> None:
        self._last_config = config

        context = LoggerContext.build(
            step=0,
            training_program=self._training_program,
            training_config=config,
            stack=stack,
            output_dir=self._base_dir,
        )
        for lg in self._inline_loggers:
            if not lg.should_fire_start():
                continue
            name = type(lg).__name__
            try:
                lg.on_start(context)
            except Exception as e:
                logger.error(f"on_start failed for {name}: {e}")
                raise

    def on_step(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        self._last_config = config

        # Write to DB via StepWriter (makes data available to LoggerRunner)
        if self._step_writer is not None:
            self._step_writer.write_step_from_raw(step, step_history)

        # Dispatch inline loggers directly
        self._dispatch_inline(step, config, step_history, stack)

    def on_end(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        # on_end for inline loggers handled via shutdown()
        pass

    def needs_params_sync(self, step: int) -> bool:
        # Only inline loggers need params sync (runner loggers get from DB)
        for lg in self._inline_loggers:
            if lg.should_fire(step) and "latest_params" in lg.required_arrays:
                return True
        return False

    def shutdown(self, result: object) -> None:
        try:
            step_history = _extract_final_step_history(result)
            final_step = _extract_final_step_index(result)
            config = self._last_config

            # Mark run as finished — signals LoggerRunner to exit
            if self._history_db is not None:
                self._history_db.mark_finished()

            # Wait for LoggerRunner thread to complete
            if self._runner_thread is not None and self._runner_thread.is_alive():
                self._runner_thread.join(timeout=120)
                if self._runner_thread.is_alive():
                    logger.warning("LoggerRunner thread did not exit within timeout")

            # Dispatch on_end to inline loggers
            self._dispatch_inline_end(final_step, config, step_history, None)
        except Exception:
            # Still try to clean up
            if self._history_db is not None:
                self._history_db.mark_finished()
            if self._runner_thread is not None and self._runner_thread.is_alive():
                self._runner_thread.join(timeout=10)
            raise

    def finalize(self, loggers: list[Logger]) -> None:
        for lg in loggers:
            if isinstance(lg, Logger):
                lg.finalize()

    @property
    def async_handler(self):
        """Backward compat — returns None (AsyncLoggerHandler removed)."""
        return None

    # ---- Inline dispatch (for execution_mode="inline" loggers) ----

    def _dispatch_inline(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        if not self._inline_loggers:
            return

        firing = [lg for lg in self._inline_loggers if lg.should_fire(step)]
        if not firing:
            return

        normalized = ensure_step_history_snapshot(
            step_history, context=f"inline dispatch at step {step}"
        )
        self._inline_history.append_from_step(step, dict(normalized))

        ctx = LoggerContext.build(
            step=step,
            training_program=self._training_program,
            training_config=config,
            stack=stack,
            output_dir=self._base_dir,
        )
        for lg in firing:
            view = self._inline_history.get_view(window=lg.history_window)
            name = type(lg).__name__
            try:
                lg.on_batch(view, ctx)
            except Exception as e:
                logger.error(f"on_batch failed for {name} at step {step}: {e}")
                raise

    def _dispatch_inline_end(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        end_loggers = [lg for lg in self._inline_loggers if lg.should_fire_end()]
        if not end_loggers:
            return

        normalized = ensure_step_history_snapshot(
            step_history, context=f"inline on_end at step {step}"
        )
        ctx = LoggerContext.build(
            step=step,
            training_program=self._training_program,
            training_config=config,
            stack=stack,
            output_dir=self._base_dir,
            is_final=True,
        )
        batch = BatchData.from_step_history(step, dict(normalized))
        view = HistoryView([batch])

        for lg in end_loggers:
            name = type(lg).__name__
            try:
                logger.info(f"Running on_end for logger: {name}")
                lg.on_end(view, ctx)
            except Exception as e:
                logger.error(f"on_end failed for {name}: {e}")
                raise
