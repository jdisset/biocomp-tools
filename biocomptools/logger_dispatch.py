from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from biocomp.logger_dispatch import LoggerDispatch
from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from biocomptools.toollib.loggers.logger import Logger
    from biocomptools.async_logger_handler import AsyncLoggerHandler

logger = get_logger(__name__)


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
        self._async_handler: AsyncLoggerHandler | None = None

        sync_loggers = [lg for lg in loggers if isinstance(lg, LoggerCls) and not lg.async_ok]
        async_loggers = [lg for lg in loggers if isinstance(lg, LoggerCls) and lg.async_ok]
        new_pattern_loggers = [
            lg
            for lg in loggers
            if isinstance(lg, LoggerCls) and getattr(lg, '_uses_new_pattern', False)
        ]

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
            async_callbacks
            or save_all_steps
            or keep_history_on_disk
            or new_pattern_loggers
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
        if self._async_handler and hasattr(self._async_handler, 'logger_objects'):
            self._effective_logger_objects.extend(self._async_handler.logger_objects)

    def on_start(self, config: object, stack: object) -> None:
        if self._async_handler:
            self._async_handler.process_start_loggers(config, stack)
        else:
            self._run_callbacks(0, config, {}, stack, lambda p, s: p == 0)

    def on_step(self, step: int, config: object, step_history: dict, stack: object) -> None:
        self._run_callbacks(step, config, step_history, stack, lambda p, s: p > 0 and s % p == 0)

    def on_end(self, step: int, config: object, step_history: dict, stack: object) -> None:
        if not self._async_handler:
            self._run_callbacks(
                step, config, step_history, stack, lambda p, s: p is None or p == -1
            )

    def needs_params_sync(self, step: int) -> bool:
        for logger_obj in self._effective_logger_objects:
            period = getattr(logger_obj, 'periods', None)
            if period is None:
                period = getattr(logger_obj, 'frequency', 1)
            if isinstance(period, list):
                period = period[0] if period else 1
            if period is not None and period > 0 and step > 0 and step % period == 0:
                reqs = getattr(logger_obj, 'required_arrays', [])
                if 'latest_params' in reqs:
                    return True
        return False

    def shutdown(self, result: object, sync_callbacks: list[tuple[int, Callable]] | None = None) -> None:
        """Shutdown async handler and run final sync callbacks."""
        if self._async_handler:
            step_history = result[-1] if isinstance(result, tuple) and len(result) > 2 else None  # type: ignore[index]
            losses = result[1] if isinstance(result, tuple) and len(result) > 1 else []  # type: ignore[index]
            final_step = len(losses) if losses else 0

            self._async_handler.process_end_loggers(
                step=final_step,
                training_config=None,
                step_history=step_history,
                stack=None,
            )

            effective_sync = sync_callbacks if sync_callbacks is not None else [
                (p, cb) for p, cb in self._sync_callbacks
                if p is None or p == -1
            ]
            for period, callback in effective_sync:
                if period is None or period == -1:
                    try:
                        callback(final_step, None, step_history=step_history, stack=None)
                    except Exception as e:
                        logger.error(f"Sync logger final callback failed: {e}")
                        logger.exception(e)

            self._async_handler.shutdown()

    def finalize(self, loggers: list[Logger]) -> None:
        from biocomptools.toollib.loggers.logger import Logger as LoggerCls

        for logger_obj in loggers:
            if isinstance(logger_obj, LoggerCls):
                logger_obj.finalize()

    @property
    def async_handler(self) -> AsyncLoggerHandler | None:
        return self._async_handler

    def _run_callbacks(
        self,
        step: int,
        config: object,
        step_history: dict,
        stack: object,
        period_filter: Callable[[int | None, int], bool],
    ) -> None:
        for period, callback in self._sync_callbacks:
            if period_filter(period, step):
                try:
                    callback(step, config, step_history=step_history, stack=stack)
                except Exception as e:
                    logger.error(f"Logger callback failed at step {step}: {e}")
                    logger.exception(e)
                    raise
