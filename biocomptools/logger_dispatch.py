from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from biocomp.logger_dispatch import LoggerDispatch
from biocomp.step_history import StepHistoryLike, StepHistorySnapshot, ensure_step_history_snapshot
from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from biocomptools.toollib.loggers.logger import Logger
    from biocomptools.async_logger_handler import AsyncLoggerHandler

logger = get_logger(__name__)


def _is_step_period(period: int | None, step: int) -> bool:
    return period is not None and period > 0 and step % period == 0


def _extract_final_step_history(result: object) -> StepHistorySnapshot:
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
    """Single dispatcher replacing the (loggers, logger_objects, async_handler) triple.

    Absorbs the split/merge logic previously in BaseOptimizationProgram.run().
    Wraps Logger objects, separates sync vs async callbacks, manages AsyncLoggerHandler,
    and exposes a unified dispatch interface for optimize()/start().
    """

    def __init__(
        self,
        loggers: list[Logger],
        *,
        training_program: object,
        async_logging: bool = True,
        async_store_location=None,
        base_dir=None,
        keep_history_on_disk: bool = False,
        save_all_steps: bool = False,
        n_workers: int = 8,
    ):
        from biocomptools.toollib.loggers.logger import Logger as LoggerCls

        self._sync_callbacks: list[tuple[int, Callable]] = []
        self._sync_logger_objects: list[Logger] = []
        self._async_handler: "AsyncLoggerHandler | None" = None
        self._last_config: object = None

        sync_loggers = [lg for lg in loggers if isinstance(lg, LoggerCls) and not lg.async_ok]
        async_loggers = [lg for lg in loggers if isinstance(lg, LoggerCls) and lg.async_ok]
        new_pattern_loggers = [
            lg
            for lg in loggers
            if isinstance(lg, LoggerCls) and getattr(lg, "_uses_new_pattern", False)
        ]
        self._new_pattern_loggers = new_pattern_loggers

        for logger_obj in sync_loggers:
            logger_obj.initialize(training_program)

        all_callbacks: list[tuple[int, Callable, Logger]] = []
        for logger_obj in loggers:
            if isinstance(logger_obj, LoggerCls):
                callbacks = logger_obj.get_callbacks(training_program)
                all_callbacks.extend(
                    [(period, callback, logger_obj) for period, callback in callbacks]
                )

        sync_callbacks = [
            (period, callback)
            for period, callback, logger_obj in all_callbacks
            if not logger_obj.async_ok
        ]
        self._sync_logger_objects = [
            logger_obj for period, callback, logger_obj in all_callbacks if not logger_obj.async_ok
        ]
        async_callbacks = [
            (period, callback, logger_obj)
            for period, callback, logger_obj in all_callbacks
            if logger_obj.async_ok
        ]

        self._sync_callbacks = sync_callbacks.copy()

        needs_async_handler = (
            async_callbacks or save_all_steps or keep_history_on_disk or new_pattern_loggers
        )
        if async_logging and needs_async_handler:
            from biocomptools.async_logger_handler import AsyncLoggerHandler

            all_handler_loggers = list(
                {id(lg): lg for lg in async_loggers + new_pattern_loggers}.values()
            )
            self._async_handler = AsyncLoggerHandler(
                logger_callbacks=async_callbacks,
                n_workers=n_workers,
                logger_objects=all_handler_loggers,
                async_store_location=async_store_location,
                base_dir=base_dir,
                keep_history_on_disk=keep_history_on_disk,
                save_all_steps=save_all_steps,
            )
            self._async_handler.initialize_loggers_async(training_program)
            self._async_handler.wait_for_initialization()
            self._sync_callbacks.append((1, self._async_handler.create_callback()))

            if async_callbacks:
                logger.info(
                    f"Async logging: {len(sync_callbacks)} sync, {len(async_callbacks)} async"
                )
            else:
                save_info = ", saving all steps" if save_all_steps else ""
                logger.info(f"Step history recording enabled{save_info}")
        elif async_logging:
            logger.info("Async logging enabled but no async-capable loggers found")
        else:
            self._sync_callbacks = [(period, callback) for period, callback, _ in all_callbacks]
            self._sync_logger_objects = [logger_obj for _, _, logger_obj in all_callbacks]

        # Build effective logger objects for sync detection (sync + async handler's loggers)
        self._effective_logger_objects: list[Logger] = list(self._sync_logger_objects)
        if self._async_handler and hasattr(self._async_handler, "logger_objects"):
            self._effective_logger_objects.extend(self._async_handler.logger_objects)

    def on_start(self, config: object, stack: object) -> None:
        self._last_config = config
        if self._async_handler:
            self._async_handler.process_start_loggers(config, stack)
        else:
            self._run_callbacks(
                0,
                config,
                StepHistorySnapshot(data={}),
                stack,
                lambda p, s: p == 0,
            )
            # New-pattern dispatch: call on_start for loggers with 0 in call_at
            from biocomptools.logger_history import LoggerContext

            for logger_obj in self._new_pattern_loggers:
                if logger_obj.async_ok:
                    continue
                if 0 not in getattr(logger_obj, "call_at", [-1]):
                    continue
                name = type(logger_obj).__name__
                try:
                    context = LoggerContext(
                        training_config=config,
                        stack=stack,
                        output_dir=None,
                        current_step=0,
                    )
                    logger_obj.on_start(context)
                except Exception as e:
                    logger.error(f"on_start failed for {name}: {e}")
                    logger.exception(e)
                    raise

    def on_step(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        self._last_config = config
        self._run_callbacks(step, config, step_history, stack, _is_step_period)
        # Dispatch on_batch for new-pattern sync loggers
        self._dispatch_new_pattern_on_batch(step, config, step_history, stack)

    def on_end(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        if not self._async_handler:
            self._run_callbacks(
                step, config, step_history, stack, lambda p, s: p is None or p == -1
            )
            # Dispatch on_end to new-pattern loggers (non-async path)
            self._dispatch_new_pattern_on_end(step, config, step_history, stack)

    def needs_params_sync(self, step: int) -> bool:
        for logger_obj in self._effective_logger_objects:
            interval = getattr(logger_obj, "call_at_interval", None)
            call_at_set = set(getattr(logger_obj, "call_at", [-1]))
            fires = (
                (interval is not None and interval > 0 and step > 0 and step % interval == 0)
                or (step > 0 and step in call_at_set)
            )
            if fires:
                reqs = getattr(logger_obj, "required_arrays", [])
                if "latest_params" in reqs:
                    return True
        return False

    def shutdown(
        self, result: object, sync_callbacks: list[tuple[int, Callable]] | None = None
    ) -> None:
        """Shutdown async handler and run final sync callbacks."""
        if self._async_handler:
            effective_sync = (
                sync_callbacks
                if sync_callbacks is not None
                else [(p, cb) for p, cb in self._sync_callbacks if p is None or p == -1]
            )
            try:
                step_history = _extract_final_step_history(result)
                final_step = _extract_final_step_index(result)
                config = getattr(self, "_last_config", None)
                self._async_handler.process_end_loggers(
                    step=final_step,
                    training_config=config,
                    step_history=step_history,
                    stack=None,
                )

                for period, callback in effective_sync:
                    if period is None or period == -1:
                        callback(final_step, config, step_history=step_history, stack=None)
            finally:
                self._async_handler.shutdown()

    def finalize(self, loggers: list[Logger]) -> None:
        from biocomptools.toollib.loggers.logger import Logger as LoggerCls

        for logger_obj in loggers:
            if isinstance(logger_obj, LoggerCls):
                logger_obj.finalize()

    @property
    def async_handler(self) -> "AsyncLoggerHandler | None":
        return self._async_handler

    def _dispatch_new_pattern_on_batch(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        """Dispatch on_batch to new-pattern sync loggers at their declared interval."""
        from biocomptools.logger_history import LoggerContext, HistoryManager

        batch_loggers = [
            lg
            for lg in self._new_pattern_loggers
            if not lg.async_ok  # only sync loggers handled here; async via handler
        ]
        if not batch_loggers:
            return

        # Lazily create HistoryManager for sync path
        if not hasattr(self, "_history_manager"):
            self._history_manager = HistoryManager(max_batches=10000)

        normalized = ensure_step_history_snapshot(
            step_history, context=f"new-pattern on_batch step_history at step {step}"
        )
        self._history_manager.append_from_step(step, dict(normalized))

        for logger_obj in batch_loggers:
            interval = logger_obj.call_at_interval
            call_at_set = set(logger_obj.call_at)
            should_fire = (
                (interval is not None and interval > 0 and step > 0 and step % interval == 0)
                or (step > 0 and step in call_at_set)
            )
            if not should_fire:
                continue

            view = self._history_manager.get_view(
                window=logger_obj.history_window,
            )
            context = LoggerContext(
                training_config=config,
                stack=stack,
                output_dir=None,
                current_step=step,
            )
            name = type(logger_obj).__name__
            try:
                logger_obj.on_batch(view, context)
            except Exception as e:
                logger.error(f"on_batch failed for {name} at step {step}: {e}")
                logger.exception(e)
                raise

    def _dispatch_new_pattern_on_end(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        """Dispatch on_end to new-pattern loggers with -1 in call_at."""
        from biocomptools.logger_history import LoggerContext, HistoryView, BatchData

        end_loggers = [lg for lg in self._new_pattern_loggers if -1 in getattr(lg, "call_at", [-1])]
        if not end_loggers:
            return

        normalized = ensure_step_history_snapshot(
            step_history, context=f"new-pattern on_end step_history at step {step}"
        )
        context = LoggerContext(
            training_config=config,
            stack=stack,
            output_dir=None,
            current_step=step,
            is_final=True,
            extra={},
        )
        batch = BatchData.from_step_history(step, dict(normalized))
        view = HistoryView([batch])

        for logger_obj in end_loggers:
            name = type(logger_obj).__name__
            try:
                logger.info(f"Running on_end for new-pattern logger: {name}")
                logger_obj.on_end(view, context)
            except Exception as e:
                logger.error(f"on_end failed for {name}: {e}")
                logger.exception(e)
                raise

    def _run_callbacks(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
        period_filter: Callable[[int | None, int], bool],
    ) -> None:
        normalized_step_history = ensure_step_history_snapshot(
            step_history,
            context=f"step_history at step {step}",
        )
        for period, callback in self._sync_callbacks:
            if period_filter(period, step):
                try:
                    callback(step, config, step_history=normalized_step_history, stack=stack)
                except Exception as e:
                    logger.error(f"Logger callback failed at step {step}: {e}")
                    logger.exception(e)
                    raise
