"""Async logger handler with hybrid thread/process pools.

Each logger declares its own execution mode via `callback_mode` ("thread" or
"process"). The handler reads that field and dispatches accordingly — no
magic name-based classification.

Key features:
- Hybrid thread/process model: threads for I/O, processes for CPU-bound work
- Batched step processing for throughput
- In-order result aggregation despite parallel processing
"""

import time
import tempfile
import shutil
import queue
import atexit
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Thread
from dataclasses import replace

import dill
from pydantic import BaseModel, Field, field_validator, ConfigDict

from collections import deque

from biocomp.step_history import StepHistoryLike, ensure_step_history_snapshot
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomptools.logger_history import HistoryManager, LoggerContext, HistoryView, BatchData

logger = get_logger(__name__)


class AsyncLoggerHandler(BaseModel):
    """Multi-threaded async logger handler."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_workers: int = Field(default=4, gt=0, le=16)
    logger_objects: list["Logger"] = Field(default_factory=list)
    async_store_location: Path | None = None
    base_dir: Path | None = None
    keep_history_on_disk: bool = False
    save_all_steps: bool = False
    history_db: Any = Field(exclude=True, default=None)  # RunHistoryDB | None
    batch_size: int = Field(default=4, gt=0, le=32, description="Steps to batch process")

    # Internal state (excluded from serialization)
    tmpdir: Path | None = Field(exclude=True, default=None)
    cleanup_tmpdir: bool = Field(exclude=True, default=True)
    step_queue: Any = Field(exclude=True, default=None)  # queue.Queue[dict | None]
    should_stop: bool = Field(exclude=True, default=False)
    thread_executor: ThreadPoolExecutor | None = Field(exclude=True, default=None)
    consumer_thread: Thread | None = Field(exclude=True, default=None)
    initialization_futures: list[Future[Any]] = Field(exclude=True, default_factory=list)
    finalization_futures: list[Future[Any]] = Field(exclude=True, default_factory=list)
    embedding_snapshots: deque[tuple[int, dict[str, Any]]] = Field(exclude=True, default=None)

    @field_validator("async_store_location", "base_dir", mode="before")
    @classmethod
    def convert_paths(cls, v: str | Path | None) -> Path | None:
        return Path(v) if v is not None else None

    def model_post_init(self, __context):
        self._setup_infrastructure()
        self._initialize()
        atexit.register(self.cleanup)

        logger.info(
            f"AsyncLoggerHandler: n_workers={self.n_workers}, "
            f"batch_size={self.batch_size}, tmpdir: {self.tmpdir}"
        )

    def _setup_infrastructure(self):
        if self.async_store_location:
            self.tmpdir = (
                self.async_store_location
                if self.async_store_location.is_absolute()
                else (self.base_dir or Path()) / self.async_store_location
            )
            self.tmpdir.mkdir(parents=True, exist_ok=True)
            self.cleanup_tmpdir = False
        else:
            self.tmpdir = Path(tempfile.mkdtemp(prefix="biocomp_async_log_v2_"))
            self.cleanup_tmpdir = not self.keep_history_on_disk

        self.step_queue = queue.Queue()
        self.should_stop = False

        self.embedding_snapshots = deque(maxlen=10000)
        self._history_manager = HistoryManager(max_batches=10000)

        # Cache for immutable objects — written once, read by consumer
        self._cached_stack: Any = None
        self._cached_training_config: Any = None
        # Single-slot latest params cache for dispatch injection
        self._latest_params_for_dispatch: Any = None
        # Training program reference for LoggerContext
        self._training_program: Any = None

    def _initialize(self):
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.n_workers, thread_name_prefix="async_log_thread"
        )

        self.consumer_thread = Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

    def _save_step_data(
        self, step: int, data: dict[str, object], event_type: str = "regular"
    ) -> Path:
        assert self.tmpdir is not None, (
            "AsyncLoggerHandler.tmpdir must be initialized before saving step data"
        )
        step_file = self.tmpdir / (
            f"step_{step:06d}.pkl"
            if event_type == "regular"
            else f"step_{step:06d}_{event_type}_{time.time()}.pkl"
        )
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        with open(step_file, "wb") as f:
            dill.dump(data, f)
        return step_file

    def _consumer_loop(self):
        """Consumer thread: batch data items and dispatch to workers."""
        while not self.should_stop:
            batch: list[dict[str, Any]] = []
            deadline = time.time() + 0.1  # 100ms batch window

            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    data_item = self.step_queue.get(timeout=0.05)
                    if data_item is None:  # Shutdown signal
                        self.should_stop = True
                        break
                    batch.append(data_item)
                except queue.Empty:
                    continue

            if not batch:
                continue

            logger.debug(f"Processing batch of {len(batch)} steps: {[d['step'] for d in batch]}")
            self._process_batch(batch)

    def _process_batch(self, items: list[dict[str, Any]]) -> None:
        """Process a batch of data items from the in-memory queue."""
        for data_item in items:
            step = data_item["step"]
            raw_step_history = data_item.get("step_history")
            timestamp = float(data_item.get("timestamp", 0.0))

            sh_snapshot = None
            if raw_step_history is not None:
                sh_snapshot = ensure_step_history_snapshot(
                    raw_step_history,
                    context=f"consumer step_history at step {step}",
                )
                sh_dict = dict(sh_snapshot)

                snapshot = self._extract_quantization_snapshot(sh_dict)
                if snapshot:
                    self.embedding_snapshots.append((step, snapshot))

                if "latest_params" in sh_dict:
                    self._latest_params_for_dispatch = sh_dict["latest_params"]

                # Strip heavy arrays before HistoryManager accumulation
                sh_dict_for_history = {
                    k: v for k, v in sh_dict.items() if k not in ("latest_params", "opt_state")
                }
                self._history_manager.append_from_step(
                    step=step,
                    step_history=sh_dict_for_history,
                    timestamp=timestamp,
                )

            # Save to DB if configured
            if self.history_db is not None and sh_snapshot is not None:
                self.history_db.save_step(step, timestamp, sh_dict)

            # Save to disk if configured (legacy pkl fallback)
            if self.keep_history_on_disk or self.save_all_steps:
                disk_data: dict[str, object] = {
                    "step": step,
                    "step_history": sh_snapshot,
                    "timestamp": timestamp,
                }
                self._save_step_data(step, disk_data)

            # Dispatch on_batch for async loggers
            self._dispatch_on_batch(step)

    def _build_extra(self, required_keys: set[str]) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if "embedding_snapshots" in required_keys:
            extra["embedding_snapshots"] = list(self.embedding_snapshots)
        return extra

    def _dispatch_on_batch(self, step: int) -> None:
        """Dispatch on_batch to async loggers via thread pool (fire-and-forget)."""
        assert self.thread_executor is not None, "Thread executor must be initialized"

        firing = [lg for lg in self.logger_objects if lg.async_ok and lg.should_fire(step)]

        if not firing:
            return

        required = set()
        for lg in firing:
            required.update(lg.required_extra)
        extra = self._build_extra(required)

        tp = self._training_program
        context = LoggerContext(
            training_config=self._cached_training_config,
            stack=self._cached_stack,
            output_dir=self.base_dir,
            current_step=step,
            dmanager=getattr(tp, "_dmanager", None),
            model=getattr(tp, "_model", None),
            training_program=tp,
            extra=extra,
        )

        for lg in firing:
            view = self._history_manager.get_view(window=lg.history_window)

            # Inject cached latest_params into view's latest batch
            needs_params = "latest_params" in getattr(lg, "required_arrays", [])
            if needs_params and self._latest_params_for_dispatch is not None and view._batches:
                latest_batch = view._batches[-1]
                view._batches[-1] = replace(
                    latest_batch,
                    arrays={
                        **latest_batch.arrays,
                        "latest_params": self._latest_params_for_dispatch,
                    },
                )

            name = type(lg).__name__

            def _on_done(fut: Future[Any], _name: str = name, _step: int = step) -> None:
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"on_batch failed for {_name} at step {_step}: {e}")

            future = self.thread_executor.submit(lg.on_batch, view, context)
            future.add_done_callback(_on_done)

    @staticmethod
    def _extract_quantization_snapshot(step_history: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        params = step_history.get("latest_params")
        if params is None:
            return {}
        result: dict[str, Any] = {}
        for emb_type in ("tc_rate", "tl_rate", "affinity"):
            try:
                arr = np.asarray(params[f"shared/quantization/values/{emb_type}"])
                while arr.ndim > 2:
                    arr = arr[0]
                result[emb_type] = [arr[i].ravel().tolist() for i in range(arr.shape[0])]
            except (KeyError, TypeError, AttributeError):
                continue
        return result

    def create_callback(self):
        """Create callback for training loop to call on each step.

        Lightweight: caches immutable objects and queues raw data references
        to the consumer thread. All heavy work happens on the consumer thread.
        """

        def async_callback(
            step: int,
            training_config: object,
            step_history: StepHistoryLike | None = None,
            stack: object = None,
        ) -> None:
            if self._cached_stack is None and stack is not None:
                self._cached_stack = stack
            if self._cached_training_config is None and training_config is not None:
                self._cached_training_config = training_config

            triggering = [
                type(lg).__name__ for lg in self.logger_objects
                if lg.async_ok and lg.should_fire(step)
            ]

            should_save = self.save_all_steps or bool(triggering)
            if not should_save:
                return

            logger.debug(f"Step {step}: queueing for {triggering or 'all'}")

            self.step_queue.put(
                {
                    "step": step,
                    "step_history": step_history,
                    "timestamp": time.time(),
                }
            )

        return async_callback

    def process_start_loggers(self, training_config: object, stack: object) -> None:
        if self._cached_stack is None:
            self._cached_stack = stack
        if self._cached_training_config is None:
            self._cached_training_config = training_config

        logger.info("Processing start loggers...")
        tp = self._training_program
        context = LoggerContext(
            training_config=training_config,
            stack=stack,
            output_dir=self.base_dir,
            current_step=0,
            dmanager=getattr(tp, "_dmanager", None),
            model=getattr(tp, "_model", None),
            training_program=tp,
        )
        for lg in self.logger_objects:
            if 0 not in lg.call_at:
                continue
            name = type(lg).__name__
            try:
                lg.on_start(context)
                logger.debug(f"Start logger {name} completed")
            except Exception as e:
                logger.error(f"Start logger {name} failed: {e}")
                raise

    def process_end_loggers(
        self,
        step: int,
        training_config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        logger.info("Processing end loggers...")
        normalized_step_history = ensure_step_history_snapshot(
            step_history,
            context=f"end step_history at step {step}",
        )

        end_loggers = [lg for lg in self.logger_objects if -1 in lg.call_at]
        if not end_loggers:
            return

        required = set()
        for lg in end_loggers:
            required.update(lg.required_extra)
        tp = self._training_program
        context = LoggerContext(
            training_config=training_config,
            stack=stack,
            output_dir=self.base_dir,
            current_step=step,
            is_final=True,
            dmanager=getattr(tp, "_dmanager", None),
            model=getattr(tp, "_model", None),
            training_program=tp,
            extra=self._build_extra(required),
        )
        batch = BatchData.from_step_history(step, dict(normalized_step_history))
        view = HistoryView([batch])

        for lg in end_loggers:
            name = type(lg).__name__
            try:
                logger.info(f"Running on_end for logger: {name}")
                t0 = time.time()
                lg.on_end(view, context)
                logger.info(f"on_end completed: {name} ({time.time() - t0:.2f}s)")
            except Exception as e:
                logger.error(f"on_end failed for {name}: {e}")
                raise

    def initialize_loggers_async(self, training_program: object) -> None:
        self._training_program = training_program
        if not self.logger_objects:
            return
        assert self.thread_executor is not None, "Thread executor must be initialized"

        def init_logger(logger_obj: Logger) -> None:
            name = type(logger_obj).__name__
            try:
                logger_obj.initialize(training_program)
                logger.info(f"Initialized {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

        self.initialization_futures = [
            self.thread_executor.submit(init_logger, obj) for obj in self.logger_objects
        ]
        logger.info(f"Started async initialization of {len(self.initialization_futures)} loggers")

    def wait_for_initialization(self) -> None:
        if not self.initialization_futures:
            return

        logger.info("Waiting for logger initialization...")
        for future in as_completed(self.initialization_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Logger init failed: {e}")

        self.initialization_futures.clear()
        logger.info("All loggers initialized")

    def finalize_loggers_async(self) -> None:
        if not self.logger_objects:
            return
        assert self.thread_executor is not None, "Thread executor must be initialized"

        def finalize_logger(logger_obj: Logger) -> None:
            name = type(logger_obj).__name__
            try:
                logger_obj.finalize()
                logger.info(f"Finalized {name}")
            except Exception as e:
                logger.error(f"Failed to finalize {name}: {e}")

        self.finalization_futures = [
            self.thread_executor.submit(finalize_logger, obj) for obj in self.logger_objects
        ]

    def wait_for_finalization(self) -> None:
        if not self.finalization_futures:
            return

        logger.info("Waiting for logger finalization...")
        for future in as_completed(self.finalization_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Logger finalize failed: {e}")

        self.finalization_futures.clear()
        logger.info("All loggers finalized")

    def shutdown(self, timeout: float = 30.0) -> None:
        logger.info("Shutting down AsyncLoggerHandler...")

        self.step_queue.put(None)

        start = time.time()
        while not self.step_queue.empty() and (time.time() - start) < timeout:
            time.sleep(0.1)

        self.should_stop = True

        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5)

        self.finalize_loggers_async()
        self.wait_for_finalization()

        if self.thread_executor:
            self.thread_executor.shutdown(wait=True, cancel_futures=False)

        if self.history_db is not None:
            self.history_db.update_end_time()

        self.cleanup()
        logger.info("AsyncLoggerHandler shutdown complete")

    def cleanup(self) -> None:
        if self.tmpdir and self.tmpdir.exists() and self.cleanup_tmpdir:
            try:
                shutil.rmtree(self.tmpdir)
                logger.info(f"Cleaned up temp directory: {self.tmpdir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        elif self.keep_history_on_disk and self.tmpdir and self.tmpdir.exists():
            logger.info(f"Keeping step history at: {self.tmpdir}")

    def get_stats(self) -> dict[str, object]:
        return {
            "queue_size": self.step_queue.qsize() if self.step_queue else 0,
            "n_workers": self.n_workers,
        }


def _rebuild_models() -> None:
    from biocomptools.toollib.loggers.logger import Logger

    AsyncLoggerHandler.model_rebuild(_types_namespace={"Logger": Logger})


_rebuild_models()
