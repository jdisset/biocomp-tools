from biocomp.logger_dispatch import LoggerDispatch
from biocomp.step_history import StepHistoryLike, StepHistorySnapshot, ensure_step_history_snapshot
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.async_logger_handler import AsyncLoggerHandler
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


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
    """Unified logger dispatcher.

    Splits loggers into sync (async_ok=False) and async (async_ok=True),
    manages AsyncLoggerHandler for async loggers, and dispatches on_batch/on_end
    directly for sync loggers.
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
        history_db: object | None = None,
    ):
        self._async_handler: AsyncLoggerHandler | None = None
        self._async_callback = None
        self._last_config: object = None
        self._training_program = training_program

        all_loggers = [lg for lg in loggers if isinstance(lg, Logger)]
        sync_loggers = [lg for lg in all_loggers if not lg.async_ok]
        async_loggers = [lg for lg in all_loggers if lg.async_ok]
        self._sync_loggers = sync_loggers

        for lg in sync_loggers:
            lg.initialize(training_program)

        needs_async_handler = async_loggers or save_all_steps or keep_history_on_disk
        if async_logging and needs_async_handler:
            self._async_handler = AsyncLoggerHandler(
                n_workers=n_workers,
                logger_objects=async_loggers,
                async_store_location=async_store_location,
                base_dir=base_dir,
                keep_history_on_disk=keep_history_on_disk,
                save_all_steps=save_all_steps,
                history_db=history_db,
            )
            self._async_handler.initialize_loggers_async(training_program)
            self._async_handler.wait_for_initialization()
            self._async_callback = self._async_handler.create_callback()
            logger.info(f"Async logging: {len(sync_loggers)} sync, {len(async_loggers)} async")
        elif async_logging and not needs_async_handler:
            logger.info("Async logging enabled but no async-capable loggers found")

        self._all_loggers = all_loggers

    def on_start(self, config: object, stack: object) -> None:
        self._last_config = config
        if self._async_handler:
            self._async_handler.process_start_loggers(config, stack)
        else:
            from biocomptools.logger_history import LoggerContext

            tp = self._training_program
            context = LoggerContext(
                training_config=config,
                stack=stack,
                output_dir=None,
                current_step=0,
                dmanager=getattr(tp, "_dmanager", None),
                model=getattr(tp, "_model", None),
                training_program=tp,
            )
            for lg in self._sync_loggers:
                if 0 not in lg.call_at:
                    continue
                name = type(lg).__name__
                try:
                    lg.on_start(context)
                except Exception as e:
                    logger.error(f"on_start failed for {name}: {e}")
                    logger.exception(e)
                    raise

    def on_step(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        self._last_config = config
        # Queue data to async handler
        if self._async_callback is not None:
            self._async_callback(step, config, step_history=step_history, stack=stack)
        # Dispatch on_batch for sync loggers
        self._dispatch_on_batch(step, config, step_history, stack)

    def on_end(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        if not self._async_handler:
            self._dispatch_on_end(step, config, step_history, stack)

    def needs_params_sync(self, step: int) -> bool:
        for lg in self._all_loggers:
            if lg.should_fire(step) and "latest_params" in lg.required_arrays:
                return True
        return False

    def shutdown(self, result: object) -> None:
        if self._async_handler:
            try:
                step_history = _extract_final_step_history(result)
                final_step = _extract_final_step_index(result)
                config = self._last_config
                self._async_handler.process_end_loggers(
                    step=final_step,
                    training_config=config,
                    step_history=step_history,
                    stack=None,
                )
                # Dispatch on_end to sync loggers (not handled by async handler)
                self._dispatch_on_end(final_step, config, step_history, None)
            finally:
                self._async_handler.shutdown()

    def finalize(self, loggers: list[Logger]) -> None:
        for lg in loggers:
            if isinstance(lg, Logger):
                lg.finalize()

    @property
    def async_handler(self) -> AsyncLoggerHandler | None:
        return self._async_handler

    def _dispatch_on_batch(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        """Dispatch on_batch to sync loggers at their declared interval."""
        from biocomptools.logger_history import LoggerContext, HistoryManager

        if not self._sync_loggers:
            return

        if not hasattr(self, "_history_manager"):
            self._history_manager = HistoryManager(max_batches=10000)

        normalized = ensure_step_history_snapshot(
            step_history, context=f"on_batch step_history at step {step}"
        )
        self._history_manager.append_from_step(step, dict(normalized))

        firing = [lg for lg in self._sync_loggers if lg.should_fire(step)]
        if not firing:
            return

        tp = self._training_program
        ctx = LoggerContext(
            training_config=config,
            stack=stack,
            output_dir=None,
            current_step=step,
            dmanager=getattr(tp, "_dmanager", None),
            model=getattr(tp, "_model", None),
            training_program=tp,
        )
        for lg in firing:
            view = self._history_manager.get_view(window=lg.history_window)
            name = type(lg).__name__
            try:
                lg.on_batch(view, ctx)
            except Exception as e:
                logger.error(f"on_batch failed for {name} at step {step}: {e}")
                logger.exception(e)
                raise

    def _dispatch_on_end(
        self,
        step: int,
        config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        """Dispatch on_end to sync loggers with -1 in call_at."""
        from biocomptools.logger_history import LoggerContext, HistoryView, BatchData

        end_loggers = [lg for lg in self._sync_loggers if -1 in lg.call_at]
        if not end_loggers:
            return

        normalized = ensure_step_history_snapshot(
            step_history, context=f"on_end step_history at step {step}"
        )
        tp = self._training_program
        context = LoggerContext(
            training_config=config,
            stack=stack,
            output_dir=None,
            current_step=step,
            is_final=True,
            dmanager=getattr(tp, "_dmanager", None),
            model=getattr(tp, "_model", None),
            training_program=tp,
            extra={},
        )
        batch = BatchData.from_step_history(step, dict(normalized))
        view = HistoryView([batch])

        for lg in end_loggers:
            name = type(lg).__name__
            try:
                logger.info(f"Running on_end for logger: {name}")
                lg.on_end(view, context)
            except Exception as e:
                logger.error(f"on_end failed for {name}: {e}")
                logger.exception(e)
                raise
