"""Async logger handler with hybrid thread/process pools.

Each logger declares its own execution mode via `callback_mode` ("thread" or
"process"). The handler reads that field and dispatches accordingly — no
magic name-based classification.

Key features:
- Hybrid thread/process model: threads for I/O, processes for CPU-bound work
- Batched step processing for throughput
- In-order result aggregation despite parallel processing
"""

from __future__ import annotations

import time
import tempfile
import shutil
import queue
import atexit
from pathlib import Path
from typing import Callable, TYPE_CHECKING, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from threading import Thread, Lock
from dataclasses import dataclass, field, replace
import multiprocessing as mp

import dill
from pydantic import BaseModel, Field, field_validator, ConfigDict

from collections import deque

from biocomp.step_history import StepHistoryLike, StepHistorySnapshot, ensure_step_history_snapshot
from biocomptools.logging_config import get_logger
from biocomptools.logger_history import HistoryManager, LoggerContext, HistoryView, BatchData

if TYPE_CHECKING:
    from biocomptools.toollib.loggers.logger import Logger

logger = get_logger(__name__)


@dataclass
class CallbackResult:
    """Result from a stateless callback execution."""

    logger_name: str
    step: int
    metrics: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)  # Serialized outputs
    error: str | None = None
    duration_ms: float = 0.0


@dataclass(order=True)
class PendingResult:
    """For ordered aggregation via heapq."""

    step: int
    results: list[CallbackResult] = field(compare=False)


def _execute_callback_in_process(
    callback_data: bytes,
    step_data: bytes,
) -> bytes:
    """Process pool worker function. Receives/returns serialized data."""
    try:
        callback_info = dill.loads(callback_data)
        data = dill.loads(step_data)

        callback_fn = callback_info["callback"]
        logger_name = callback_info["logger_name"]

        start = time.perf_counter()

        # Execute callback - should return metrics dict, not mutate state
        raw_step_history = data.get("step_history")
        step_history = (
            None
            if raw_step_history is None
            else ensure_step_history_snapshot(
                raw_step_history,
                context=f"async process callback step_history at step {data['step']}",
            )
        )

        result = callback_fn(
            data["step"],
            data["training_config"],
            step_history=step_history,
            stack=data.get("stack"),
        )

        duration_ms = (time.perf_counter() - start) * 1000

        # Package result
        callback_result = CallbackResult(
            logger_name=logger_name,
            step=data["step"],
            metrics=result if isinstance(result, dict) else {},
            duration_ms=duration_ms,
        )
        return dill.dumps(callback_result)

    except Exception as e:
        callback_result = CallbackResult(
            logger_name=callback_info.get("logger_name", "unknown"),
            step=data.get("step", -1) if "data" in dir() else -1,
            error=str(e),
        )
        return dill.dumps(callback_result)


class AsyncLoggerHandler(BaseModel):
    """Multi-process async logger handler with ordering guarantees."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    logger_callbacks: list[tuple[int, Callable[..., object], "Logger"]]
    n_workers: int = Field(default=4, gt=0, le=16)
    logger_objects: list["Logger"] = Field(default_factory=list)
    async_store_location: Path | None = None
    base_dir: Path | None = None
    keep_history_on_disk: bool = False
    save_all_steps: bool = False
    batch_size: int = Field(default=4, gt=0, le=32, description="Steps to batch process")

    # Internal state (excluded from serialization)
    # Note: Some types use Any because they're runtime-initialized threading primitives
    # that can't be properly type-annotated (Lock is a function, not a class)
    tmpdir: Path | None = Field(exclude=True, default=None)
    cleanup_tmpdir: bool = Field(exclude=True, default=True)
    step_queue: Any = Field(exclude=True, default=None)  # queue.Queue[dict | None]
    results_queue: Any = Field(exclude=True, default=None)  # queue.Queue[CallbackResult]
    should_stop: bool = Field(exclude=True, default=False)
    thread_executor: ThreadPoolExecutor | None = Field(exclude=True, default=None)
    process_executor: ProcessPoolExecutor | None = Field(exclude=True, default=None)
    consumer_thread: Thread | None = Field(exclude=True, default=None)
    aggregator_thread: Thread | None = Field(exclude=True, default=None)
    initialization_futures: list[Future[Any]] = Field(exclude=True, default_factory=list)
    finalization_futures: list[Future[Any]] = Field(exclude=True, default_factory=list)
    aggregation_lock: Any = Field(exclude=True, default=None)  # Lock instance
    next_step_to_aggregate: int = Field(exclude=True, default=1)
    pending_results: list[PendingResult] = Field(exclude=True, default_factory=list)
    loggers_by_name: dict[str, "Logger"] = Field(exclude=True, default_factory=dict)
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
        """Set up directories and data structures."""
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
        self.results_queue = queue.Queue()
        self.should_stop = False
        self.aggregation_lock = Lock()
        self.next_step_to_aggregate = 1
        self.pending_results = []

        self.embedding_snapshots = deque(maxlen=10000)
        self._history_manager = HistoryManager(max_batches=10000)

        # Cache for immutable objects — written once, read by consumer (Fix 2)
        self._cached_stack: Any = None
        self._cached_training_config: Any = None
        # Single-slot latest params cache for dispatch injection (Fix 3)
        self._latest_params_for_dispatch: Any = None

        # Build logger lookup
        self.loggers_by_name = {}
        for _, _, logger_obj in self.logger_callbacks:
            name = type(logger_obj).__name__
            self.loggers_by_name[name] = logger_obj

    def _initialize(self):
        """Start thread/process pools and consumer threads."""
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.n_workers, thread_name_prefix="async_log_thread"
        )

        # Use spawn context for cleaner process isolation
        ctx = mp.get_context("spawn")
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
        )

        self.consumer_thread = Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        self.aggregator_thread = Thread(target=self._aggregator_loop, daemon=True)
        self.aggregator_thread.start()

    def _save_step_data(
        self, step: int, data: dict[str, object], event_type: str = "regular"
    ) -> Path:
        """Serialize step data to disk."""
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

    def _load_step_data(self, step_file: Path) -> dict[str, object]:
        """Load step data from disk."""
        with open(step_file, "rb") as f:
            return dill.load(f)

    def _cleanup_step_file(self, step_file: Path):
        """Remove step file if not keeping history."""
        if not self.keep_history_on_disk:
            step_file.unlink(missing_ok=True)

    def _should_run_callback(self, period: int | None, step: int, event_type: str) -> bool:
        """Determine if callback should run for this step/event."""
        if event_type == "start":
            return period == 0 and step == 0
        elif event_type == "end":
            return period is None or period == -1
        else:  # regular
            return period is not None and period > 0 and step > 0 and step % period == 0

    def _consumer_loop(self):
        """Consumer thread: batch data items and dispatch to workers.

        Receives raw data dicts from the in-memory queue (put by the main-thread
        callback) and performs all heavy work: normalization, snapshot extraction,
        disk I/O, history accumulation, and logger dispatch.
        """
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
        """Process a batch of data items from the in-memory queue.

        All heavy work happens here on the consumer thread:
        - Step history normalization (Fix 1: moved from main thread)
        - Quantization snapshot extraction (Fix 1: moved from main thread)
        - Disk save without stack/config (Fix 2: immutable caching)
        - History accumulation without latest_params (Fix 3: memory management)
        - Legacy and new-pattern logger dispatch
        """
        assert self.thread_executor is not None, "Thread executor must be initialized"
        assert self.process_executor is not None, "Process executor must be initialized"
        futures: list[tuple[int, str, Future[Any], str]] = []

        for data_item in items:
            step = data_item["step"]
            raw_step_history = data_item.get("step_history")
            timestamp = float(data_item.get("timestamp", 0.0))

            # Normalize step history on consumer thread (Fix 1)
            sh_snapshot = None
            if raw_step_history is not None:
                sh_snapshot = ensure_step_history_snapshot(
                    raw_step_history,
                    context=f"consumer step_history at step {step}",
                )
                sh_dict = dict(sh_snapshot)

                # Extract embedding snapshots on consumer thread (Fix 1)
                snapshot = self._extract_quantization_snapshot(sh_dict)
                if snapshot:
                    self.embedding_snapshots.append((step, snapshot))

                # Cache latest_params before stripping (Fix 3)
                if "latest_params" in sh_dict:
                    self._latest_params_for_dispatch = sh_dict["latest_params"]

                # Strip heavy arrays before HistoryManager accumulation (Fix 3):
                # latest_params and opt_state would otherwise be kept in the deque
                # for up to 10,000 entries, preventing GC of old parameter states.
                sh_dict_for_history = {
                    k: v for k, v in sh_dict.items() if k not in ("latest_params", "opt_state")
                }
                self._history_manager.append_from_step(
                    step=step,
                    step_history=sh_dict_for_history,
                    timestamp=timestamp,
                )

            # Save to disk if configured — without stack/config (Fix 2)
            if self.keep_history_on_disk or self.save_all_steps:
                disk_data: dict[str, object] = {
                    "step": step,
                    "step_history": sh_snapshot,
                    "timestamp": timestamp,
                }
                self._save_step_data(step, disk_data)

            # Build full data dict with cached immutable objects for dispatch (Fix 2)
            data: dict[str, object] = {
                "step": step,
                "training_config": self._cached_training_config,
                "step_history": sh_snapshot,
                "stack": self._cached_stack,
                "timestamp": timestamp,
            }

            # Dispatch on_batch for new-pattern async loggers
            self._dispatch_new_pattern_on_batch(step, data)

            # Legacy callback dispatch
            for period, callback, logger_obj in self.logger_callbacks:
                if not self._should_run_callback(period, step, "regular"):
                    continue

                logger_name = type(logger_obj).__name__
                mode = getattr(logger_obj, "callback_mode", "thread")

                if mode == "process":
                    callback_data = dill.dumps(
                        {
                            "callback": callback,
                            "logger_name": logger_name,
                        }
                    )
                    step_data = dill.dumps(data)
                    future = self.process_executor.submit(
                        _execute_callback_in_process,
                        callback_data,
                        step_data,
                    )
                    futures.append((step, logger_name, future, mode))
                else:
                    future = self.thread_executor.submit(
                        self._execute_callback_thread,
                        callback,
                        logger_name,
                        data,
                    )
                    futures.append((step, logger_name, future, mode))

        # Collect futures
        for step, logger_name, future, mode in futures:
            try:
                if mode == "process":
                    result_bytes = future.result(timeout=60)
                    result = dill.loads(result_bytes)
                else:
                    result = future.result(timeout=60)
                    if not isinstance(result, CallbackResult):
                        result = CallbackResult(
                            logger_name=logger_name,
                            step=step,
                            metrics=result if isinstance(result, dict) else {},
                        )
                self.results_queue.put(result)
            except Exception as e:
                logger.error(f"Callback {logger_name} failed for step {step}: {e}")
                self.results_queue.put(
                    CallbackResult(
                        logger_name=logger_name,
                        step=step,
                        error=str(e),
                    )
                )

    def _build_extra(self, required_keys: set[str]) -> dict[str, Any]:
        """Build extra context dict for loggers, producing only requested keys.

        Register new accumulated state here; loggers declare what they need
        via ``required_extra``.
        """
        extra: dict[str, Any] = {}
        if "embedding_snapshots" in required_keys:
            extra["embedding_snapshots"] = list(self.embedding_snapshots)
        return extra

    def _dispatch_new_pattern_on_batch(self, step: int, data: dict[str, object]) -> None:
        """Dispatch on_batch to new-pattern async loggers via thread pool (fire-and-forget).

        For loggers that declare ``required_arrays: ["latest_params"]``, the cached
        latest_params are injected into the view's latest batch via dataclasses.replace.
        This avoids accumulating params in the HistoryManager deque (Fix 3).
        """
        assert self.thread_executor is not None, "Thread executor must be initialized"

        # Collect loggers that will fire this step
        firing: list[Logger] = []
        for logger_obj in self.logger_objects:
            if not getattr(logger_obj, "_uses_new_pattern", False):
                continue
            if not logger_obj.async_ok:
                continue
            interval = getattr(logger_obj, "call_at_interval", None)
            call_at_set = set(getattr(logger_obj, "call_at", [-1]))
            should_fire = (
                interval is not None and interval > 0 and step > 0 and step % interval == 0
            ) or (step > 0 and step in call_at_set)
            if not should_fire:
                continue
            firing.append(logger_obj)

        if not firing:
            return

        # Build shared extra from union of required_extra across firing loggers
        required = set()
        for lg in firing:
            required.update(getattr(lg, "required_extra", []))
        extra = self._build_extra(required)

        context = LoggerContext(
            training_config=data.get("training_config"),
            stack=data.get("stack"),
            output_dir=self.base_dir,
            current_step=step,
            extra=extra,
        )

        for logger_obj in firing:
            view = self._history_manager.get_view(
                window=getattr(logger_obj, "history_window", None),
            )

            # Inject cached latest_params into view's latest batch (Fix 3).
            # The view._batches is a new list (from get_view), so replacing the
            # last element doesn't mutate the HistoryManager's deque.
            needs_params = "latest_params" in getattr(logger_obj, "required_arrays", [])
            if needs_params and self._latest_params_for_dispatch is not None and view._batches:
                latest_batch = view._batches[-1]
                view._batches[-1] = replace(
                    latest_batch,
                    arrays={
                        **latest_batch.arrays,
                        "latest_params": self._latest_params_for_dispatch,
                    },
                )

            name = type(logger_obj).__name__

            def _on_done(fut: Future[Any], _name: str = name, _step: int = step) -> None:
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"on_batch failed for {_name} at step {_step}: {e}")

            future = self.thread_executor.submit(logger_obj.on_batch, view, context)
            future.add_done_callback(_on_done)

    def _execute_callback_thread(
        self, callback: Callable[..., object], logger_name: str, data: dict[str, object]
    ) -> dict[str, object]:
        """Execute callback in thread pool."""
        start = time.perf_counter()
        try:
            raw_step_history = data.get("step_history")
            step_history = (
                None
                if raw_step_history is None
                else ensure_step_history_snapshot(
                    raw_step_history,
                    context=f"async thread callback step_history at step {data['step']}",
                )
            )
            result = callback(
                data["step"],
                data["training_config"],
                step_history=step_history,
                stack=data.get("stack"),
            )
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{logger_name} completed in {duration_ms:.1f}ms (step={data['step']})")
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"{logger_name} failed: {e}")
            raise

    def _aggregator_loop(self) -> None:
        """Aggregator thread: collect results and aggregate in order."""
        # Group results by step
        step_results: dict[int, list[CallbackResult]] = {}

        while not self.should_stop or not self.results_queue.empty():
            try:
                result = self.results_queue.get(timeout=0.1)
                step = result.step

                if step not in step_results:
                    step_results[step] = []
                step_results[step].append(result)

                # Try to aggregate completed steps in order
                self._try_aggregate_ordered(step_results)

            except queue.Empty:
                continue

        # Final aggregation of any remaining results
        self._try_aggregate_ordered(step_results, force=True)

    def _try_aggregate_ordered(
        self, step_results: dict[int, list[CallbackResult]], force: bool = False
    ) -> None:
        """Aggregate results in step order."""
        with self.aggregation_lock:
            while True:
                next_step = self.next_step_to_aggregate

                if next_step not in step_results:
                    if force and step_results:
                        # Process any step if forcing (shutdown)
                        next_step = min(step_results.keys())
                    else:
                        break

                results = step_results.pop(next_step)

                for result in results:
                    if result.error:
                        logger.warning(f"Step {result.step} {result.logger_name}: {result.error}")
                        continue

                    # Aggregate into logger if it has _aggregate method
                    logger_obj = self.loggers_by_name.get(result.logger_name)
                    if logger_obj and hasattr(logger_obj, "_aggregate"):
                        try:
                            logger_obj._aggregate(result.step, result.metrics)
                        except Exception as e:
                            logger.error(f"Aggregation failed for {result.logger_name}: {e}")

                    if result.duration_ms > 0:
                        logger.debug(
                            f"Aggregated {result.logger_name} step {result.step} "
                            f"({result.duration_ms:.1f}ms)"
                        )

                self.next_step_to_aggregate = next_step + 1

    @staticmethod
    def _extract_quantization_snapshot(step_history: dict[str, Any]) -> dict[str, Any]:
        """Extract lightweight quantization embedding values from step_history params."""
        import numpy as np

        params = step_history.get("latest_params")
        if params is None:
            return {}
        result: dict[str, Any] = {}
        for emb_type in ("tc_rate", "tl_rate", "affinity"):
            try:
                arr = np.asarray(params[f"shared/quantization/values/{emb_type}"])
                # Strip to (n_embeddings, rate_dim) — take first replicate if vmapped
                while arr.ndim > 2:
                    arr = arr[0]
                result[emb_type] = [arr[i].ravel().tolist() for i in range(arr.shape[0])]
            except (KeyError, TypeError, AttributeError):
                continue
        return result

    def create_callback(self) -> Callable[..., None]:
        """Create callback for training loop to call on each step.

        The callback is lightweight: it caches immutable objects (stack, config)
        and queues raw data references to the consumer thread via an in-memory
        queue. All heavy work (normalization, snapshot extraction, disk I/O)
        happens on the consumer thread.
        """

        def async_callback(
            step: int,
            training_config: object,
            step_history: StepHistoryLike | None = None,
            stack: object = None,
        ) -> None:
            # Cache immutable objects on first call (Fix 2)
            if self._cached_stack is None and stack is not None:
                self._cached_stack = stack
            if self._cached_training_config is None and training_config is not None:
                self._cached_training_config = training_config

            # Check if any logger needs this step (legacy callbacks)
            triggering = []
            for period, _, logger_obj in self.logger_callbacks:
                if self._should_run_callback(period, step, "regular"):
                    triggering.append(type(logger_obj).__name__)

            # Also check new-pattern async loggers
            for logger_obj in self.logger_objects:
                if not getattr(logger_obj, "_uses_new_pattern", False):
                    continue
                if not logger_obj.async_ok:
                    continue  # sync loggers dispatched by LoggerDispatcher
                interval = getattr(logger_obj, "call_at_interval", None)
                call_at_set = set(getattr(logger_obj, "call_at", [-1]))
                if (
                    interval is not None and interval > 0 and step > 0 and step % interval == 0
                ) or (step > 0 and step in call_at_set):
                    triggering.append(type(logger_obj).__name__)

            should_save = self.save_all_steps or bool(triggering)
            if not should_save:
                return

            logger.debug(f"Step {step}: queueing for {triggering or 'all'}")

            # Queue raw data for consumer thread (Fix 1: no serialization on main thread).
            # step_history contains concrete JAX arrays (immutable after block_until_ready)
            # so passing by reference to the consumer thread is safe.
            self.step_queue.put(
                {
                    "step": step,
                    "step_history": step_history,
                    "timestamp": time.time(),
                }
            )

        return async_callback

    def process_start_loggers(self, training_config: object, stack: object) -> None:
        """Process start-only loggers (period=0)."""
        # Cache immutable objects early (before any steps are queued)
        if self._cached_stack is None:
            self._cached_stack = stack
        if self._cached_training_config is None:
            self._cached_training_config = training_config

        logger.info("Processing start loggers...")
        for period, callback, logger_obj in self.logger_callbacks:
            if self._should_run_callback(period, 0, "start"):
                name = type(logger_obj).__name__
                try:
                    callback(
                        0, training_config, step_history=StepHistorySnapshot(data={}), stack=stack
                    )
                    logger.debug(f"Start logger {name} completed")
                except Exception as e:
                    logger.error(f"Start logger {name} failed: {e}")
                    raise

        # New-pattern dispatch: call on_start for loggers with 0 in call_at
        from biocomptools.logger_history import LoggerContext

        for logger_obj in self.logger_objects:
            if not getattr(logger_obj, "_uses_new_pattern", False):
                continue
            if 0 not in getattr(logger_obj, "call_at", [-1]):
                continue
            name = type(logger_obj).__name__
            try:
                context = LoggerContext(
                    training_config=training_config,
                    stack=stack,
                    output_dir=self.base_dir,
                    current_step=0,
                )
                logger_obj.on_start(context)
                logger.debug(f"Start new-pattern logger {name} completed")
            except Exception as e:
                logger.error(f"Start new-pattern logger {name} failed: {e}")
                raise

    def process_end_loggers(
        self,
        step: int,
        training_config: object,
        step_history: StepHistoryLike,
        stack: object,
    ) -> None:
        """Process end-only loggers (period=-1 or None)."""
        logger.info("Processing end loggers...")
        normalized_step_history = ensure_step_history_snapshot(
            step_history,
            context=f"end step_history at step {step}",
        )

        # Legacy callback dispatch
        for period, callback, logger_obj in self.logger_callbacks:
            if self._should_run_callback(period, step, "end"):
                name = type(logger_obj).__name__
                try:
                    logger.info(f"Running end logger: {name}")
                    t0 = time.time()
                    callback(
                        step, training_config, step_history=normalized_step_history, stack=stack
                    )
                    logger.info(f"End logger completed: {name} ({time.time() - t0:.2f}s)")
                except Exception as e:
                    logger.error(f"End logger {name} failed: {e}")
                    raise

        # New-pattern dispatch: call on_end for loggers with -1 in call_at
        new_pattern_end_loggers = [
            lg
            for lg in self.logger_objects
            if getattr(lg, "_uses_new_pattern", False) and -1 in getattr(lg, "call_at", [-1])
        ]
        if new_pattern_end_loggers:
            required = set()
            for lg in new_pattern_end_loggers:
                required.update(getattr(lg, "required_extra", []))
            context = LoggerContext(
                training_config=training_config,
                stack=stack,
                output_dir=self.base_dir,
                current_step=step,
                is_final=True,
                extra=self._build_extra(required),
            )
            batch = BatchData.from_step_history(step, dict(normalized_step_history))
            view = HistoryView([batch])

            for logger_obj in new_pattern_end_loggers:
                name = type(logger_obj).__name__
                try:
                    logger.info(f"Running on_end for new-pattern logger: {name}")
                    t0 = time.time()
                    logger_obj.on_end(view, context)
                    logger.info(f"on_end completed: {name} ({time.time() - t0:.2f}s)")
                except Exception as e:
                    logger.error(f"on_end failed for {name}: {e}")
                    raise

    def initialize_loggers_async(self, training_program: object) -> None:
        """Initialize all loggers in parallel."""
        if not self.logger_objects:
            return
        assert self.thread_executor is not None, "Thread executor must be initialized"

        def init_logger(logger_obj: "Logger") -> None:
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
        """Block until all loggers initialized."""
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
        """Finalize all loggers in parallel."""
        if not self.logger_objects:
            return
        assert self.thread_executor is not None, "Thread executor must be initialized"

        def finalize_logger(logger_obj: "Logger") -> None:
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
        """Block until all loggers finalized."""
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
        """Graceful shutdown with timeout."""
        logger.info("Shutting down AsyncLoggerHandler...")

        # Signal consumer to stop
        self.step_queue.put(None)

        # Wait for queue to drain
        start = time.time()
        while not self.step_queue.empty() and (time.time() - start) < timeout:
            time.sleep(0.1)

        self.should_stop = True

        # Wait for consumer thread
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5)

        # Wait for aggregator thread
        if self.aggregator_thread and self.aggregator_thread.is_alive():
            self.aggregator_thread.join(timeout=5)

        # Finalize loggers
        self.finalize_loggers_async()
        self.wait_for_finalization()

        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True, cancel_futures=False)
        if self.process_executor:
            self.process_executor.shutdown(wait=True, cancel_futures=False)

        self.cleanup()
        logger.info("AsyncLoggerHandler shutdown complete")

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.tmpdir and self.tmpdir.exists() and self.cleanup_tmpdir:
            try:
                shutil.rmtree(self.tmpdir)
                logger.info(f"Cleaned up temp directory: {self.tmpdir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        elif self.keep_history_on_disk and self.tmpdir and self.tmpdir.exists():
            logger.info(f"Keeping step history at: {self.tmpdir}")

    def get_stats(self) -> dict[str, object]:
        """Get handler statistics for debugging."""
        return {
            "queue_size": self.step_queue.qsize() if self.step_queue else 0,
            "results_pending": self.results_queue.qsize() if self.results_queue else 0,
            "next_step_to_aggregate": self.next_step_to_aggregate,
            "n_workers": self.n_workers,
        }


def _rebuild_models() -> None:
    from biocomptools.toollib.loggers.logger import Logger

    AsyncLoggerHandler.model_rebuild(_types_namespace={"Logger": Logger})


_rebuild_models()
