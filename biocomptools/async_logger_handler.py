"""Multi-process async logger handler with ordering guarantees.

Key features:
- True CPU parallelism via ProcessPoolExecutor (bypasses GIL)
- Batched step processing for throughput
- Stateless callbacks with centralized aggregation
- In-order result aggregation despite parallel processing
- Hybrid thread/process model: threads for I/O, processes for CPU-bound work

Architecture:
                                    ┌─────────────────┐
    Training Loop ──► step_queue ──►│ Consumer Thread │──► Process Pool ──► results_queue
                                    └─────────────────┘           │
                                                                  ▼
                                    ┌─────────────────┐    ┌─────────────┐
                                    │ Aggregator      │◄───│ Worker 1..N │
                                    │ Thread (ordered)│    └─────────────┘
                                    └────────┬────────┘
                                             │
                                             ▼
                                    Logger._aggregate(results)
"""

import time
import tempfile
import shutil
import queue
import atexit
from pathlib import Path
from typing import List, Tuple, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from threading import Thread, Lock
from dataclasses import dataclass, field
from enum import Enum, auto
import multiprocessing as mp

import dill
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class CallbackMode(Enum):
    """How a callback should be executed."""
    THREAD = auto()      # Fast, I/O-bound (file writes, network)
    PROCESS = auto()     # Slow, CPU-bound (matplotlib, heavy computation)
    INLINE = auto()      # Must run in main thread (GPU operations)


@dataclass
class CallbackResult:
    """Result from a stateless callback execution."""
    logger_name: str
    step: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, bytes] = field(default_factory=dict)  # Serialized outputs
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass(order=True)
class PendingResult:
    """For ordered aggregation via heapq."""
    step: int
    results: List[CallbackResult] = field(compare=False)


def _execute_callback_in_process(
    callback_data: bytes,
    step_data: bytes,
) -> bytes:
    """Process pool worker function. Receives/returns serialized data."""
    try:
        callback_info = dill.loads(callback_data)
        data = dill.loads(step_data)

        callback_fn = callback_info['callback']
        logger_name = callback_info['logger_name']

        start = time.perf_counter()

        # Execute callback - should return metrics dict, not mutate state
        result = callback_fn(
            data['step'],
            data['training_config'],
            step_history=data.get('step_history'),
            stack=data.get('stack'),
        )

        duration_ms = (time.perf_counter() - start) * 1000

        # Package result
        callback_result = CallbackResult(
            logger_name=logger_name,
            step=data['step'],
            metrics=result if isinstance(result, dict) else {},
            duration_ms=duration_ms,
        )
        return dill.dumps(callback_result)

    except Exception as e:
        callback_result = CallbackResult(
            logger_name=callback_info.get('logger_name', 'unknown'),
            step=data.get('step', -1) if 'data' in dir() else -1,
            error=str(e),
        )
        return dill.dumps(callback_result)


class StatelessCallbackWrapper:
    """Wraps a logger callback to make it stateless.

    Instead of mutating logger._history, returns data to be aggregated.
    """

    def __init__(self, logger_obj: Any, original_callback: Callable):
        self.logger_name = type(logger_obj).__name__
        self.original_callback = original_callback
        self._logger_ref = logger_obj  # Kept for inline mode

    def __call__(self, step, training_config, step_history=None, stack=None, **kwargs):
        """Execute callback and capture any returned metrics."""
        result = self.original_callback(
            step, training_config,
            step_history=step_history,
            stack=stack,
            **kwargs
        )
        return result if isinstance(result, dict) else {}


class AsyncLoggerHandler(BaseModel):
    """Multi-process async logger handler with ordering guarantees."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    logger_callbacks: List[Tuple[int, Callable, Any]]
    n_thread_workers: int = Field(default=4, gt=0, le=16)
    n_process_workers: int = Field(default=2, gt=0, le=8)
    logger_objects: List[Any] = Field(default_factory=list)
    async_store_location: Optional[Path] = None
    base_dir: Optional[Path] = None
    keep_history_on_disk: bool = False
    save_all_steps: bool = False
    batch_size: int = Field(default=4, gt=0, le=32, description="Steps to batch process")

    # Callback mode overrides (logger class name -> mode)
    callback_modes: Dict[str, CallbackMode] = Field(default_factory=dict)

    # Internal state (excluded from serialization)
    tmpdir: Path = Field(exclude=True, default=None)
    cleanup_tmpdir: bool = Field(exclude=True, default=True)
    step_queue: Any = Field(exclude=True, default=None)
    results_queue: Any = Field(exclude=True, default=None)
    should_stop: bool = Field(exclude=True, default=False)
    thread_executor: Any = Field(exclude=True, default=None)
    process_executor: Any = Field(exclude=True, default=None)
    consumer_thread: Any = Field(exclude=True, default=None)
    aggregator_thread: Any = Field(exclude=True, default=None)
    initialization_futures: List[Any] = Field(exclude=True, default_factory=list)
    finalization_futures: List[Any] = Field(exclude=True, default_factory=list)
    aggregation_lock: Any = Field(exclude=True, default=None)
    next_step_to_aggregate: int = Field(exclude=True, default=1)
    pending_results: List = Field(exclude=True, default_factory=list)  # heapq
    loggers_by_name: Dict[str, Any] = Field(exclude=True, default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def handle_n_workers_compat(cls, data):
        """Backwards compatibility: distribute n_workers to thread/process workers."""
        if isinstance(data, dict) and 'n_workers' in data:
            n = data.pop('n_workers')
            if n is not None:
                data.setdefault('n_thread_workers', max(1, n // 2))
                data.setdefault('n_process_workers', max(1, n // 4))
        return data

    @field_validator('async_store_location', 'base_dir', mode='before')
    @classmethod
    def convert_paths(cls, v):
        return Path(v) if v is not None else None

    def model_post_init(self, __context):
        self._setup_infrastructure()
        self._classify_callbacks()
        self._initialize()
        atexit.register(self.cleanup)

        mode_info = f"threads={self.n_thread_workers}, processes={self.n_process_workers}"
        batch_info = f"batch_size={self.batch_size}"
        logger.info(f"AsyncLoggerHandler: {mode_info}, {batch_info}, tmpdir: {self.tmpdir}")

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

        # Build logger lookup
        self.loggers_by_name = {}
        for _, _, logger_obj in self.logger_callbacks:
            name = type(logger_obj).__name__
            self.loggers_by_name[name] = logger_obj

    def _classify_callbacks(self):
        """Determine execution mode for each callback based on logger type."""
        # Default modes based on logger class patterns
        process_bound_patterns = ['Plot', 'Subloss', 'Diagnostic', 'Figure', 'Render']
        inline_patterns = ['Validation']  # May need GPU access

        for _, _callback, logger_obj in self.logger_callbacks:
            name = type(logger_obj).__name__
            if name in self.callback_modes:
                continue  # Already set explicitly

            # Auto-classify
            if any(p in name for p in process_bound_patterns):
                self.callback_modes[name] = CallbackMode.PROCESS
            elif any(p in name for p in inline_patterns):
                self.callback_modes[name] = CallbackMode.INLINE
            else:
                self.callback_modes[name] = CallbackMode.THREAD

        mode_summary = {m.name: [] for m in CallbackMode}
        for name, mode in self.callback_modes.items():
            mode_summary[mode.name].append(name)
        logger.debug(f"Callback modes: {mode_summary}")

    def _initialize(self):
        """Start thread/process pools and consumer threads."""
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.n_thread_workers,
            thread_name_prefix="async_log_thread"
        )

        # Use spawn context for cleaner process isolation
        ctx = mp.get_context('spawn')
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.n_process_workers,
            mp_context=ctx,
        )

        self.consumer_thread = Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()

        self.aggregator_thread = Thread(target=self._aggregator_loop, daemon=True)
        self.aggregator_thread.start()

    def _save_step_data(self, step: int, data: dict, event_type: str = "regular") -> Path:
        """Serialize step data to disk."""
        step_file = self.tmpdir / (
            f"step_{step:06d}.pkl"
            if event_type == "regular"
            else f"step_{step:06d}_{event_type}_{time.time()}.pkl"
        )
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        with open(step_file, 'wb') as f:
            dill.dump(data, f)
        return step_file

    def _load_step_data(self, step_file: Path) -> dict:
        """Load step data from disk."""
        with open(step_file, 'rb') as f:
            return dill.load(f)

    def _cleanup_step_file(self, step_file: Path):
        """Remove step file if not keeping history."""
        if not self.keep_history_on_disk:
            step_file.unlink(missing_ok=True)

    def _should_run_callback(self, period: Optional[int], step: int, event_type: str) -> bool:
        """Determine if callback should run for this step/event."""
        if event_type == 'start':
            return period == 0 and step == 0
        elif event_type == 'end':
            return period is None or period == -1
        else:  # regular
            return period is not None and period > 0 and step > 0 and step % period == 0

    def _consumer_loop(self):
        """Consumer thread: batch steps and dispatch to workers."""
        while not self.should_stop:
            # Batch collection with timeout
            batch = []
            deadline = time.time() + 0.1  # 100ms batch window

            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    step = self.step_queue.get(timeout=0.05)
                    if step is None:  # Shutdown signal
                        self.should_stop = True
                        break
                    batch.append(step)
                except queue.Empty:
                    continue

            if not batch:
                continue

            # Process batch
            logger.debug(f"Processing batch of {len(batch)} steps: {batch}")
            self._process_batch(batch)

    def _process_batch(self, steps: List[int]):
        """Process a batch of steps, dispatching to appropriate executors."""
        futures: List[Tuple[int, str, Future, CallbackMode]] = []

        for step in steps:
            step_file = self.tmpdir / f"step_{step:06d}.pkl"
            if not step_file.exists():
                logger.warning(f"Step file not found: {step_file}")
                continue

            data = self._load_step_data(step_file)

            for period, callback, logger_obj in self.logger_callbacks:
                if not self._should_run_callback(period, step, 'regular'):
                    continue

                logger_name = type(logger_obj).__name__
                mode = self.callback_modes.get(logger_name, CallbackMode.THREAD)

                if mode == CallbackMode.PROCESS:
                    # Serialize for process pool
                    callback_data = dill.dumps({
                        'callback': callback,
                        'logger_name': logger_name,
                    })
                    step_data = dill.dumps(data)
                    future = self.process_executor.submit(
                        _execute_callback_in_process,
                        callback_data,
                        step_data,
                    )
                    futures.append((step, logger_name, future, mode))

                elif mode == CallbackMode.THREAD:
                    future = self.thread_executor.submit(
                        self._execute_callback_thread,
                        callback, logger_name, data,
                    )
                    futures.append((step, logger_name, future, mode))

                else:  # INLINE - execute immediately in consumer thread
                    result = self._execute_callback_thread(callback, logger_name, data)
                    self.results_queue.put(CallbackResult(
                        logger_name=logger_name,
                        step=step,
                        metrics=result,
                    ))

            self._cleanup_step_file(step_file)

        # Collect futures
        for step, logger_name, future, mode in futures:
            try:
                if mode == CallbackMode.PROCESS:
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
                self.results_queue.put(CallbackResult(
                    logger_name=logger_name,
                    step=step,
                    error=str(e),
                ))

    def _execute_callback_thread(
        self, callback: Callable, logger_name: str, data: dict
    ) -> Dict[str, Any]:
        """Execute callback in thread pool."""
        start = time.perf_counter()
        try:
            result = callback(
                data['step'],
                data['training_config'],
                step_history=data.get('step_history'),
                stack=data.get('stack'),
            )
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{logger_name} completed in {duration_ms:.1f}ms (step={data['step']})")
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"{logger_name} failed: {e}")
            raise

    def _aggregator_loop(self):
        """Aggregator thread: collect results and aggregate in order."""
        # Group results by step
        step_results: Dict[int, List[CallbackResult]] = {}

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
        self, step_results: Dict[int, List[CallbackResult]], force: bool = False
    ):
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
                        logger.warning(
                            f"Step {result.step} {result.logger_name}: {result.error}"
                        )
                        continue

                    # Aggregate into logger if it has _aggregate method
                    logger_obj = self.loggers_by_name.get(result.logger_name)
                    if logger_obj and hasattr(logger_obj, '_aggregate'):
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

    def create_callback(self):
        """Create callback for training loop to call on each step."""
        def async_callback(step, training_config, step_history=None, stack=None):
            # Check if any logger needs this step
            triggering = []
            for period, _, logger_obj in self.logger_callbacks:
                if self._should_run_callback(period, step, 'regular'):
                    triggering.append(type(logger_obj).__name__)

            should_save = self.save_all_steps or bool(triggering)
            if not should_save:
                return

            logger.debug(f"Step {step}: queueing for {triggering or 'all'}")

            try:
                data = {
                    'step': step,
                    'training_config': training_config,
                    'step_history': step_history,
                    'stack': stack,
                    'timestamp': time.time(),
                }
                self._save_step_data(step, data)
                self.step_queue.put(step)
            except Exception as e:
                logger.error(f"Failed to queue step {step}: {e}")

        return async_callback

    def process_start_loggers(self, training_config, stack):
        """Process start-only loggers (period=0)."""
        logger.info("Processing start loggers...")
        for period, callback, logger_obj in self.logger_callbacks:
            if self._should_run_callback(period, 0, 'start'):
                name = type(logger_obj).__name__
                try:
                    callback(0, training_config, step_history={}, stack=stack)
                    logger.debug(f"Start logger {name} completed")
                except Exception as e:
                    logger.error(f"Start logger {name} failed: {e}")

    def process_end_loggers(self, step, training_config, step_history, stack):
        """Process end-only loggers (period=-1 or None)."""
        logger.info("Processing end loggers...")

        for period, callback, logger_obj in self.logger_callbacks:
            if self._should_run_callback(period, step, 'end'):
                name = type(logger_obj).__name__
                try:
                    callback(step, training_config, step_history=step_history, stack=stack)
                    logger.debug(f"End logger {name} completed")
                except Exception as e:
                    logger.error(f"End logger {name} failed: {e}")

    def initialize_loggers_async(self, training_program):
        """Initialize all loggers in parallel."""
        if not self.logger_objects:
            return

        def init_logger(logger_obj):
            name = type(logger_obj).__name__
            try:
                logger_obj.initialize(training_program)
                logger.info(f"Initialized {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

        self.initialization_futures = [
            self.thread_executor.submit(init_logger, obj)
            for obj in self.logger_objects
        ]
        logger.info(f"Started async initialization of {len(self.initialization_futures)} loggers")

    def wait_for_initialization(self):
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

    def finalize_loggers_async(self):
        """Finalize all loggers in parallel."""
        if not self.logger_objects:
            return

        def finalize_logger(logger_obj):
            name = type(logger_obj).__name__
            try:
                logger_obj.finalize()
                logger.info(f"Finalized {name}")
            except Exception as e:
                logger.error(f"Failed to finalize {name}: {e}")

        self.finalization_futures = [
            self.thread_executor.submit(finalize_logger, obj)
            for obj in self.logger_objects
        ]

    def wait_for_finalization(self):
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

    def shutdown(self, timeout: float = 30.0):
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

    def cleanup(self):
        """Clean up temporary files."""
        if self.tmpdir and self.tmpdir.exists() and self.cleanup_tmpdir:
            try:
                shutil.rmtree(self.tmpdir)
                logger.info(f"Cleaned up temp directory: {self.tmpdir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        elif self.keep_history_on_disk and self.tmpdir and self.tmpdir.exists():
            logger.info(f"Keeping step history at: {self.tmpdir}")

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics for debugging."""
        return {
            'queue_size': self.step_queue.qsize() if self.step_queue else 0,
            'results_pending': self.results_queue.qsize() if self.results_queue else 0,
            'next_step_to_aggregate': self.next_step_to_aggregate,
            'callback_modes': {k: v.name for k, v in self.callback_modes.items()},
            'n_thread_workers': self.n_thread_workers,
            'n_process_workers': self.n_process_workers,
        }


# Mixin for loggers to support aggregation
class AggregatingLoggerMixin:
    """Mixin for loggers that need to aggregate results from parallel execution.

    Usage:
        class MyLogger(Logger, AggregatingLoggerMixin):
            def _aggregate(self, step: int, metrics: Dict[str, Any]):
                self._history.append({'step': step, **metrics})
    """

    aggregation_lock: Lock = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.aggregation_lock = Lock()

    def _aggregate(self, step: int, metrics: Dict[str, Any]):
        """Override this to aggregate results from parallel callbacks.

        Thread-safe via aggregation_lock.
        """
        raise NotImplementedError("Subclasses must implement _aggregate")

    def _safe_aggregate(self, step: int, metrics: Dict[str, Any]):
        """Thread-safe wrapper around _aggregate."""
        if self.aggregation_lock is None:
            self.aggregation_lock = Lock()
        with self.aggregation_lock:
            self._aggregate(step, metrics)
