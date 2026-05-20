# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""LoggerRunner: dispatches loggers in live and replay modes."""

import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

from biocomptools.history_db import RunHistoryDB
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger
from biocomptools.toollib.loggers.logger import Logger

logger = get_logger(__name__)


class LoggerRunner:
    def __init__(
        self,
        db: RunHistoryDB,
        loggers: list[Logger],
        *,
        mode: Literal["live", "replay"] = "replay",
        poll_interval: float = 0.1,
        output_dir: Path | None = None,
        thread_workers: int = 4,
        process_workers: int = 2,
        training_program: object | None = None,
    ) -> None:
        self._db = db
        self._loggers = loggers
        self._mode = mode
        self._poll_interval = poll_interval
        self._output_dir = output_dir
        self._training_program = training_program

        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_workers, thread_name_prefix="logger_thread"
        )
        self._process_pool: ProcessPoolExecutor | None = None
        process_loggers = [lg for lg in loggers if lg.execution_mode == "process"]
        if process_loggers and process_workers > 0:
            ctx = get_context("spawn")
            self._process_pool = ProcessPoolExecutor(max_workers=process_workers, mp_context=ctx)

        self._pending_futures: list[Future[Any]] = []

    def run(self) -> None:
        self._initialize_loggers()
        last_step = -1

        try:
            while True:
                new_steps = self._db.get_steps_since(last_step)

                if not new_steps:
                    if self._mode == "replay" or self._db.is_run_finished():
                        break
                    time.sleep(self._poll_interval)
                    continue

                for step in new_steps:
                    self._dispatch_step(step)
                    last_step = step

            self._wait_pending()
            self._dispatch_on_end(last_step)
        finally:
            self._finalize_loggers()
            self._shutdown_pools()

    def _dispatch_step(self, step: int) -> None:
        self._clean_futures()

        for lg in self._loggers:
            if not lg.should_fire(step):
                continue

            if lg.execution_mode == "process" and self._process_pool is not None:
                future = self._process_pool.submit(
                    _run_logger_in_process,
                    db_path=self._db.path,
                    step=step,
                    logger_config=lg.model_dump(),
                    logger_class_module=type(lg).__module__,
                    logger_class_name=type(lg).__name__,
                    output_dir=self._output_dir,
                )
                future.add_done_callback(_make_done_callback(type(lg).__name__, step))
                self._pending_futures.append(future)
            elif lg.execution_mode == "inline":
                try:
                    self._run_logger_in_thread(lg, step)
                except Exception as e:
                    logger.error(f"on_batch failed for {type(lg).__name__} at step {step}: {e}")
            else:
                future = self._thread_pool.submit(self._run_logger_in_thread, lg, step)
                future.add_done_callback(_make_done_callback(type(lg).__name__, step))
                self._pending_futures.append(future)

    def _run_logger_in_thread(self, lg: Logger, step: int) -> None:
        t0 = time.time()
        view = self._build_view(lg, step)
        t_load = time.time() - t0
        context = LoggerContext.build(
            step=step,
            training_program=self._training_program,
            output_dir=self._output_dir,
            is_replay=(self._mode == "replay"),
            db=self._db,
        )
        t1 = time.time()
        lg.on_batch(view, context)
        t_run = time.time() - t1
        t_total = t_load + t_run
        name = type(lg).__name__
        logger.debug(f"{t_total:.2f}s {name}@{step} (load={t_load:.2f}s)")

    def _build_view(self, lg: Logger, step: int) -> HistoryView:
        window = lg.history_window
        start = max(0, step - window) if window is not None else 0
        scalar_keys = lg.required_metrics or None

        # None = load all, [] = load nothing
        array_keys: list[str] | None = None
        blob_keys: list[str] | None = None
        if lg.required_arrays:
            from biocomptools.step_history_triage import partition_required_keys

            array_keys, blob_keys = partition_required_keys(lg.required_arrays)

        batches = self._db.load_step_range_data(
            start,
            step,
            scalar_keys=scalar_keys,
            array_keys=array_keys,
            blob_keys=blob_keys,
        )
        return HistoryView(batches)

    def _dispatch_on_end(self, last_step: int) -> None:
        if last_step < 0:
            return

        for lg in self._loggers:
            if not lg.should_fire_end():
                continue

            view = self._build_view(lg, last_step)
            context = LoggerContext.build(
                step=last_step,
                training_program=self._training_program,
                output_dir=self._output_dir,
                is_replay=(self._mode == "replay"),
                is_final=True,
                db=self._db,
            )
            name = type(lg).__name__
            try:
                logger.info(f"Running on_end for logger: {name}")
                t0 = time.time()
                lg.on_end(view, context)
                logger.info(f"on_end completed: {name} ({time.time() - t0:.2f}s)")
            except Exception as e:
                logger.error(f"on_end failed for {name}: {e}")

    def _initialize_loggers(self) -> None:
        for lg in self._loggers:
            name = type(lg).__name__
            try:
                lg.initialize(self._training_program)
            except Exception as e:
                logger.warning(f"Logger {name} initialize failed: {e}")

        for lg in self._loggers:
            if not lg.should_fire_start():
                continue
            context = LoggerContext.build(
                step=0,
                training_program=self._training_program,
                output_dir=self._output_dir,
                is_replay=(self._mode == "replay"),
                db=self._db,
            )
            try:
                lg.on_start(context)
            except Exception as e:
                logger.warning(f"Logger {type(lg).__name__} on_start failed: {e}")

    def _finalize_loggers(self) -> None:
        for lg in self._loggers:
            try:
                lg.finalize()
            except Exception as e:
                logger.warning(f"Logger {type(lg).__name__} finalize failed: {e}")

    def _clean_futures(self) -> None:
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def _wait_pending(self, timeout: float = 60.0) -> None:
        for f in self._pending_futures:
            try:
                f.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Pending logger future failed: {e}")
        self._pending_futures.clear()

    def _shutdown_pools(self) -> None:
        self._thread_pool.shutdown(wait=True, cancel_futures=False)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=False)


# Top-level for ProcessPoolExecutor picklability.


def _run_logger_in_process(
    db_path: Path,
    step: int,
    logger_config: dict[str, Any],
    logger_class_module: str,
    logger_class_name: str,
    output_dir: Path | None,
) -> None:
    import importlib

    from biocomptools.history_db import RunHistoryDB

    db = RunHistoryDB(db_path, read_only=True)

    mod = importlib.import_module(logger_class_module)
    logger_class = getattr(mod, logger_class_name)
    lg = logger_class(**logger_config)

    stack, model, dmanager = None, None, None
    if getattr(lg, "_needs_stack", False):
        model = db.load_artifact("model")
        dmanager = db.load_artifact("dmanager")
        dconfig = db.load_artifact("dconfig")
        if model and dmanager:
            stack = _rebuild_stack_from_artifacts(dmanager, dconfig, model)

    window = lg.history_window
    start = max(0, step - (window or step))
    scalar_keys = lg.required_metrics or None
    array_keys: list[str] | None = None
    blob_keys: list[str] | None = None
    if lg.required_arrays:
        from biocomptools.step_history_triage import partition_required_keys

        array_keys, blob_keys = partition_required_keys(lg.required_arrays)
    batches = db.load_step_range_data(
        start,
        step,
        scalar_keys=scalar_keys,
        array_keys=array_keys,
        blob_keys=blob_keys,
    )
    view = HistoryView(batches)

    context = LoggerContext.build(
        step=step,
        output_dir=output_dir,
        stack=stack,
        model=model,
        dmanager=dmanager,
    )
    lg.on_batch(view, context)
    db.close()


def _rebuild_stack_from_artifacts(
    dmanager: Any,
    dconfig: Any,
    model: Any,
) -> Any:
    if dconfig is not None:
        try:
            from biocomp.design_prune_controller import build_stack_from_dconf

            return build_stack_from_dconf(dmanager, dconfig, model, lock_ratios=True)
        except ImportError:
            pass
    return dmanager.build_stack(model, unlock_ratios=False)


def _make_done_callback(name: str, step: int):
    def _on_done(fut: Future[Any]) -> None:
        try:
            fut.result()
        except Exception as e:
            logger.error(f"on_batch failed for {name} at step {step}: {e}")

    return _on_done
