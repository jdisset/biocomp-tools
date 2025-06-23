import dill
import time
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Callable, Any, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from threading import Thread
import queue
import atexit
import shutil

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class AsyncLoggerHandler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    logger_callbacks: List[Tuple[int, Callable]]
    min_period: int
    n_workers: int = Field(default=8, gt=0, le=32)
    logger_objects: List[Any] = Field(default_factory=list)
    async_store_location: Optional[Path] = None
    base_dir: Optional[Path] = None
    keep_history_on_disk: bool = False
    save_all_steps: bool = False
    replay_mode: bool = False
    replay_base_dir: Optional[Path] = None

    # Private execution state
    tmpdir: Path = Field(exclude=True, default=None)
    cleanup_tmpdir: bool = Field(exclude=True, default=True)
    step_queue: Any = Field(exclude=True, default=None)
    should_stop: bool = Field(exclude=True, default=False)
    executor: Any = Field(exclude=True, default=None)
    processing_thread: Any = Field(exclude=True, default=None)
    initialization_futures: List[Any] = Field(exclude=True, default_factory=list)
    finalization_futures: List[Any] = Field(exclude=True, default_factory=list)

    @field_validator('async_store_location', 'base_dir', 'replay_base_dir', mode='before')
    @classmethod
    def convert_paths(cls, v):
        return Path(v) if v is not None else None

    @field_validator('replay_base_dir')
    @classmethod
    def validate_replay_base_dir(cls, v, info):
        replay_mode = info.data.get('replay_mode', False)
        v = Path(v).expanduser().resolve() if v else None
        if replay_mode and v and not v.exists():
            raise ValueError(f"Replay directory {v} does not exist")
        return v

    @model_validator(mode='after')
    def validate_replay_mode_consistency(self):
        if self.replay_mode and not self.replay_base_dir:
            raise ValueError("replay_base_dir is required when replay_mode=True")
        return self

    def model_post_init(self, __context):
        self._setup_directories_and_infrastructure()
        if not self.replay_mode:
            self._initialize()
            atexit.register(self.cleanup)
        save_info = " (saving all steps)" if self.save_all_steps else ""
        logger.info(f"AsyncLoggerHandler: Thread mode{save_info}, tmpdir: {self.tmpdir}")

    def _setup_directories_and_infrastructure(self):
        if self.async_store_location:
            self.tmpdir = (
                self.async_store_location
                if self.async_store_location.is_absolute()
                else (self.base_dir or Path()) / self.async_store_location
            )
            self.tmpdir.mkdir(parents=True, exist_ok=True)
            self.cleanup_tmpdir = False
        elif self.replay_mode:
            self.tmpdir = self.replay_base_dir
            self.cleanup_tmpdir = False
        else:
            self.tmpdir = Path(tempfile.mkdtemp(prefix="biocomp_async_log_"))
            self.cleanup_tmpdir = not self.keep_history_on_disk

        import queue

        self.step_queue = queue.Queue()
        self.should_stop = False
        self.executor = None
        self.processing_thread = None
        self.initialization_futures = []
        self.finalization_futures = []

    def _save_step_data(self, step: int, data: dict, event_type: str = "regular") -> Path:
        step_file = self.tmpdir / (
            f"step_{step:06d}.pkl"
            if event_type == "regular"
            else f"step_{step:06d}_{event_type}_{time.time()}.pkl"
        )
        with open(step_file, 'wb') as f:
            dill.dump(data, f)
        return step_file

    def _load_step_data(self, step_file: Path) -> dict:
        with open(step_file, 'rb') as f:
            return dill.load(f)

    def _cleanup_step_file(self, step_file: Path):
        if not self.keep_history_on_disk:
            step_file.unlink(missing_ok=True)

    def _execute_callbacks(self, step_file: Path, step: int, filter_func: Callable):
        data = self._load_step_data(step_file)
        callbacks_to_run = [(p, c) for p, c in self.logger_callbacks if filter_func(p, c, step)]

        if not callbacks_to_run:
            return

        futures = [self.executor.submit(self._safe_call, c, data) for _, c in callbacks_to_run]
        [f.result(timeout=300) for f in futures]

    def _safe_call(self, callback, data):
        start_time = time.time()
        try:
            callback(
                data['step'],
                data['training_config'],
                step_history=data.get('step_history'),
                stack=data.get('stack'),
            )
            logger.info(
                f"Logger completed in {time.time() - start_time:.2f}s (step={data['step']})"
            )
        except Exception as e:
            logger.error(f"Logger failed after {time.time() - start_time:.2f}s: {e}")

    def _should_run_start_loggers(
        self, period: Optional[int], callback: Callable, step: int
    ) -> bool:
        return period == 0 and step == 0

    def _should_run_regular_loggers(
        self, period: Optional[int], callback: Callable, step: int
    ) -> bool:
        return period is not None and period > 0 and step > 0 and step % period == 0

    def _should_run_end_loggers(self, period: Optional[int], callback: Callable, step: int) -> bool:
        return period is None or period == -1

    def _process_step_unified(
        self, step: int, training_config, step_history, stack, event_type: str = "regular"
    ):
        data = {
            'step': step,
            'training_config': training_config,
            'step_history': step_history,
            'stack': stack,
            'timestamp': time.time(),
        }
        step_file = self._save_step_data(step, data, event_type)

        filter_func = {
            'start': self._should_run_start_loggers,
            'end': self._should_run_end_loggers,
        }.get(event_type, self._should_run_regular_loggers)

        try:
            self._execute_callbacks(step_file, step, filter_func)
        finally:
            self._cleanup_step_file(step_file)

    def process_start_loggers(self, training_config, stack):
        logger.info("Processing start loggers...")
        self._process_step_unified(0, training_config, {}, stack, "start")

    def process_end_loggers(self, step, training_config, step_history, stack):
        logger.info("Processing end loggers...")
        self._process_step_unified(step, training_config, step_history, stack, "end")

    def create_callback(self):
        def async_callback(step, training_config, step_history=None, stack=None):
            try:
                data = {
                    'step': step,
                    'training_config': training_config,
                    'step_history': step_history,
                    'stack': stack,
                    'timestamp': time.time(),
                }
                self._save_step_data(step, data, "regular")
                self.step_queue.put(step)
            except Exception as e:
                logger.error(f"Failed to save step data: {e}")

        return async_callback

    def _process_queue(self):
        while not self.should_stop:
            try:
                step = self.step_queue.get(timeout=1.0)
                if step is None:
                    break
                step_file = self.tmpdir / f"step_{step:06d}.pkl"
                if step_file.exists():
                    self._execute_callbacks(step_file, step, self._should_run_regular_loggers)
                    self._cleanup_step_file(step_file)
                self.step_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def _initialize(self):
        self.executor = ThreadPoolExecutor(
            max_workers=self.n_workers, thread_name_prefix="async_logger"
        )
        self.processing_thread = Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

    def initialize_loggers_async(self, training_program):
        if not self.logger_objects:
            return

        def init_logger(logger_obj):
            try:
                logger_obj.initialize(training_program)
                logger.info(f"Initialized logger {type(logger_obj).__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize logger {type(logger_obj).__name__}: {e}")

        # Submit initialization jobs asynchronously
        self.initialization_futures = [
            self.executor.submit(init_logger, obj) for obj in self.logger_objects
        ]
        logger.info(f"Started async initialization of {len(self.initialization_futures)} loggers")

    def wait_for_initialization(self):
        if not self.initialization_futures:
            return

        logger.info("Waiting for logger initialization to complete...")
        # Wait for all initialization futures to complete
        for future in concurrent.futures.as_completed(self.initialization_futures):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                logger.error(f"Logger initialization failed: {e}")

        logger.info("All logger initialization completed")
        self.initialization_futures.clear()

    def finalize_loggers_async(self):
        if not self.logger_objects:
            return

        def finalize_logger(logger_obj):
            try:
                logger_obj.finalize()
                logger.info(f"Finalized logger {type(logger_obj).__name__}")
            except Exception as e:
                logger.error(f"Failed to finalize logger {type(logger_obj).__name__}: {e}")

        # Submit finalization jobs asynchronously
        self.finalization_futures = [
            self.executor.submit(finalize_logger, obj) for obj in self.logger_objects
        ]
        logger.info(f"Started async finalization of {len(self.finalization_futures)} loggers")

    def wait_for_finalization(self):
        if not self.finalization_futures:
            return

        logger.info("Waiting for logger finalization to complete...")
        # Wait for all finalization futures to complete
        for future in concurrent.futures.as_completed(self.finalization_futures):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                logger.error(f"Logger finalization failed: {e}")

        logger.info("All logger finalization completed")
        self.finalization_futures.clear()

    def shutdown(self):
        # Start async finalization first
        self.finalize_loggers_async()

        # Shutdown queue processing
        self.step_queue.join()
        self.should_stop = True
        self.step_queue.put(None)
        if self.processing_thread:
            self.processing_thread.join(timeout=10)

        # Wait for finalization to complete before shutting down executor
        self.wait_for_finalization()

        if self.executor:
            self.executor.shutdown(wait=True)
        self.cleanup()

    def cleanup(self):
        if self.tmpdir.exists() and self.cleanup_tmpdir:
            try:
                shutil.rmtree(self.tmpdir)
                logger.info(f"Cleaned up async logger temp directory: {self.tmpdir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        elif self.keep_history_on_disk and self.tmpdir.exists():
            logger.info(f"Keeping step history on disk at: {self.tmpdir}")

    @classmethod
    def replay_from_disk(
        cls,
        logger_callbacks: List[Tuple[int, Callable]],
        replay_base_dir: Union[str, Path],
        n_workers: int = 8,
        logger_objects: Optional[List] = None,
        step_filter: Optional[Callable[[int], bool]] = None,
    ) -> 'AsyncLoggerHandler':
        """
        Create an AsyncLoggerHandler in replay mode to process previously saved step history.

        Args:
            logger_callbacks: List of (period, callback) tuples for loggers to replay
            replay_base_dir: Directory containing saved step history files
            n_workers: Number of workers for parallel processing
            logger_objects: Logger objects for initialization (if needed)
            step_filter: Optional function to filter which steps to replay (lambda step: bool)

        Returns:
            AsyncLoggerHandler instance configured for replay mode
        """
        return cls(
            logger_callbacks=logger_callbacks,
            min_period=1,
            n_workers=n_workers,
            logger_objects=logger_objects or [],
            replay_mode=True,
            replay_base_dir=replay_base_dir,
        )

    def replay_steps(self, step_filter: Optional[Callable[[int], bool]] = None):
        if not self.replay_mode:
            raise ValueError("Handler not in replay mode")

        all_files = sorted(self.tmpdir.glob("step_*.pkl"))
        if not all_files:
            logger.warning(f"No step history files found in {self.tmpdir}")
            return

        logger.info(f"Found {len(all_files)} step history files for replay")

        # initialize loggers with simple mock
        if self.logger_objects and all_files:
            sample_data = self._load_step_data(all_files[0])
            mock_program = type(
                'MockTrainingProgram',
                (),
                {'training_conf': sample_data.get('training_config'), '_save_dir': self.tmpdir},
            )()
            for logger_obj in self.logger_objects:
                try:
                    logger_obj.initialize(mock_program)
                except Exception as e:
                    logger.error(f"Failed to initialize logger {logger_obj}: {e}")

        if not self.executor:
            self._initialize()

        processed_count = 0
        for step_file in all_files:
            try:
                parts = step_file.stem.split('_')
                step = int(parts[1])
                if step_filter and not step_filter(step):
                    continue

                # determine filter function
                event_type = parts[2] if len(parts) > 2 else "regular"
                filter_func = {
                    'start': self._should_run_start_loggers,
                    'end': self._should_run_end_loggers,
                }.get(event_type, self._should_run_regular_loggers)

                self._execute_callbacks(step_file, step, filter_func)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to replay step from {step_file}: {e}")

        logger.info(f"Replay completed: processed {processed_count} steps")

        # Use async finalization for replay as well
        self.finalize_loggers_async()
        self.wait_for_finalization()

        # Clean shutdown (without additional finalization since we just did it)
        self.step_queue.join()
        self.should_stop = True
        self.step_queue.put(None)
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        if self.executor:
            self.executor.shutdown(wait=True)
        self.cleanup()
