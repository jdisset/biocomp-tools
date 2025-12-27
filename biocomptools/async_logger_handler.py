import dill
import time
import tempfile
import shutil
import queue
import atexit
from pathlib import Path
from typing import List, Tuple, Callable, Any, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator, ConfigDict

from biocomptools.logging_config import get_logger
from biocomptools.logger_history import (
    HistoryView,
    HistoryManager,
    LoggerContext,
)

logger = get_logger(__name__)


class AsyncLoggerHandler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    logger_callbacks: List[Tuple[int, Callable, Any]]
    n_workers: int = Field(default=8, gt=0, le=32)
    logger_objects: List[Any] = Field(default_factory=list)
    async_store_location: Optional[Path] = None
    base_dir: Optional[Path] = None
    keep_history_on_disk: bool = False
    save_all_steps: bool = False
    replay_mode: bool = False
    replay_base_dir: Optional[Path] = None
    batches_per_step: int = 1  # for step->batch conversion
    max_history_batches: int = 10000  # max batches to keep in memory

    tmpdir: Path = Field(exclude=True, default=None)
    cleanup_tmpdir: bool = Field(exclude=True, default=True)
    step_queue: Any = Field(exclude=True, default=None)
    should_stop: bool = Field(exclude=True, default=False)
    executor: Any = Field(exclude=True, default=None)
    processing_thread: Any = Field(exclude=True, default=None)
    initialization_futures: List[Any] = Field(exclude=True, default_factory=list)
    finalization_futures: List[Any] = Field(exclude=True, default_factory=list)
    _history_manager: HistoryManager = PrivateAttr(default=None)
    _training_config: Any = PrivateAttr(default=None)
    _stack: Any = PrivateAttr(default=None)
    _last_callback_step: Dict[int, int] = PrivateAttr(default_factory=dict)

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
        self._history_manager = HistoryManager(max_batches=self.max_history_batches)
        self._last_callback_step = {}
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
        self.tmpdir.mkdir(parents=True, exist_ok=True)
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
        callbacks_to_run = [(p, c) for p, c, _ in self.logger_callbacks if filter_func(p, c, step)]

        if not callbacks_to_run:
            return

        futures = [self.executor.submit(self._safe_call, c, data) for _, c in callbacks_to_run]
        [f.result() for f in futures]

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
            logger.exception(e)

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
            triggering_loggers = []
            if not self.save_all_steps:
                for p, _, logger_obj in self.logger_callbacks:
                    if p is not None and p > 0 and step > 0 and step % p == 0:
                        name = getattr(logger_obj, 'name', type(logger_obj).__name__)
                        triggering_loggers.append(name)

            should_save_this_step = self.save_all_steps or bool(triggering_loggers)

            if not should_save_this_step:
                return

            if self.save_all_steps:
                reason = "save_all_steps is True"
            else:
                unique_triggers = sorted(list(set(triggering_loggers)))
                reason = f"triggered by logger(s): {', '.join(unique_triggers)}"

            logger.info(f"Step {step}: Saving step data. Reason: {reason}.")

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
                logger.error(f"Failed to save step data for step {step}: {e}")
                logger.exception(e)

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
                logger.exception(e)

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
                logger.exception(e)

        self.initialization_futures = [
            self.executor.submit(init_logger, obj) for obj in self.logger_objects
        ]
        logger.info(f"Started async initialization of {len(self.initialization_futures)} loggers")

    def wait_for_initialization(self):
        if not self.initialization_futures:
            return

        logger.info("Waiting for logger initialization to complete...")
        for future in as_completed(self.initialization_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Logger initialization failed: {e}")
                logger.exception(e)

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
                logger.exception(e)

        self.finalization_futures = [
            self.executor.submit(finalize_logger, obj) for obj in self.logger_objects
        ]
        logger.info(f"Started async finalization of {len(self.finalization_futures)} loggers")

    def wait_for_finalization(self):
        if not self.finalization_futures:
            return

        logger.info("Waiting for logger finalization to complete...")
        for future in as_completed(self.finalization_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Logger finalization failed: {e}")
                logger.exception(e)

        logger.info("All logger finalization completed")
        self.finalization_futures.clear()

    def shutdown(self):
        self.finalize_loggers_async()
        self.step_queue.join()
        self.should_stop = True
        self.step_queue.put(None)
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
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
        logger_callbacks: List[Tuple[int, Callable, Any]],
        replay_base_dir: Union[str, Path],
        n_workers: int = 8,
        logger_objects: Optional[List] = None,
        step_filter: Optional[Callable[[int], bool]] = None,
    ) -> 'AsyncLoggerHandler':
        return cls(
            logger_callbacks=logger_callbacks,
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
                    logger.exception(e)

        if not self.executor:
            self._initialize()

        processed_count = 0
        for step_file in all_files:
            try:
                parts = step_file.stem.split('_')
                step = int(parts[1])
                if step_filter and not step_filter(step):
                    continue

                event_type = parts[2] if len(parts) > 2 else "regular"
                filter_func = {
                    'start': self._should_run_start_loggers,
                    'end': self._should_run_end_loggers,
                }.get(event_type, self._should_run_regular_loggers)

                self._execute_callbacks(step_file, step, filter_func)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to replay step from {step_file}: {e}")
                logger.exception(e)

        logger.info(f"Replay completed: processed {processed_count} steps")
        self.finalize_loggers_async()
        self.wait_for_finalization()
        self.step_queue.join()
        self.should_stop = True
        self.step_queue.put(None)
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        if self.executor:
            self.executor.shutdown(wait=True)
        self.cleanup()

    # ──────────────────────────────────────────────────────────────────────────
    # New pattern support methods
    # ──────────────────────────────────────────────────────────────────────────

    def _make_context(
        self,
        step: int,
        training_config: Any = None,
        stack: Any = None,
        is_final: bool = False,
        is_replay: bool = False,
    ) -> LoggerContext:
        """Create LoggerContext for new-pattern callbacks."""
        return LoggerContext(
            training_config=training_config or self._training_config,
            stack=stack or self._stack,
            output_dir=self.base_dir,
            current_batch=step * self.batches_per_step,
            current_step=step,
            total_batches=None,
            total_steps=None,
            batches_per_step=self.batches_per_step,
            is_replay=is_replay or self.replay_mode,
            is_final=is_final,
        )

    def _get_view_for_logger(self, logger_obj: Any) -> HistoryView:
        """Get appropriate HistoryView based on logger's requirements."""
        window = getattr(logger_obj, 'history_window', None)
        mode = getattr(logger_obj, 'history_mode', 'window')
        metrics = getattr(logger_obj, 'required_metrics', None)
        arrays = getattr(logger_obj, 'required_arrays', None)

        logger_id = id(logger_obj)
        since_batch = None
        if mode == 'since_last':
            since_batch = self._last_callback_step.get(logger_id, 0)

        return self._history_manager.get_view(
            window=window,
            since_batch=since_batch,
            metrics=metrics or [],
            arrays=arrays or [],
        )

    def _dispatch_new_pattern_loggers(
        self,
        step: int,
        training_config: Any,
        stack: Any,
        event_type: str = 'regular',
    ):
        """Dispatch to loggers using new pattern (on_batch/on_end)."""
        context = self._make_context(step, training_config, stack, is_final=(event_type == 'end'))

        for logger_obj in self.logger_objects:
            uses_new = getattr(logger_obj, '_uses_new_pattern', False)
            if not uses_new:
                continue

            try:
                if event_type == 'start':
                    if getattr(logger_obj, 'call_at_start', False):
                        logger_obj.on_start(context)
                elif event_type == 'end':
                    if getattr(logger_obj, 'call_at_end', False):
                        view = self._get_view_for_logger(logger_obj)
                        logger_obj.on_end(view, context)
                else:
                    freq = getattr(logger_obj, 'frequency', 1)
                    periods = getattr(logger_obj, 'periods', freq)
                    if isinstance(periods, int):
                        freq = periods
                    if step > 0 and step % freq == 0:
                        view = self._get_view_for_logger(logger_obj)
                        logger_obj.on_batch(view, context)
                        self._last_callback_step[id(logger_obj)] = (
                            self._history_manager.current_batch_index
                        )
            except Exception as e:
                name = type(logger_obj).__name__
                logger.error(f"New-pattern logger {name} failed: {e}")
                logger.exception(e)

    def log_step_unified(
        self,
        step: int,
        training_config: Any,
        step_history: dict,
        stack: Any,
        event_type: str = 'regular',
    ):
        """Unified step logging that supports both patterns.

        Call this from training loop instead of create_callback for full support.
        """
        self._training_config = training_config
        self._stack = stack

        # add to centralized history
        self._history_manager.append_from_step(
            step=step,
            step_history=step_history,
            batches_per_step=self.batches_per_step,
            timestamp=time.time(),
        )

        # dispatch to new-pattern loggers
        self._dispatch_new_pattern_loggers(step, training_config, stack, event_type)

        # existing legacy callback flow
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

    # ──────────────────────────────────────────────────────────────────────────
    # First-class replay mode
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def replay(
        cls,
        history_dir: Union[str, Path],
        loggers: List[Any],
        step_filter: Optional[Callable[[int], bool]] = None,
        final_only: bool = False,
        n_workers: int = 8,
    ) -> None:
        """First-class replay from saved history.

        Args:
            history_dir: Directory containing step_*.pkl or batch_*.pkl files
            loggers: Logger instances to replay through
            step_filter: Optional filter for which steps to include
            final_only: If True, only call on_end with accumulated history
            n_workers: Thread pool size
        """
        history_dir = Path(history_dir)

        # load batches from disk (try batch files first, fall back to step files)
        # Pass step_filter to loader to avoid loading unnecessary files
        batches = HistoryManager.load_batches(history_dir)
        if not batches:
            batches = HistoryManager.load_from_step_files(
                history_dir, step_filter=step_filter, show_progress=True
            )
        elif step_filter:
            # If we loaded from batch files, apply filter after
            batches = [b for b in batches if step_filter(b.step_index)]

        if not batches:
            logger.warning(f"No history files found in {history_dir}")
            return

        logger.info(f"Replaying {len(batches)} batches through {len(loggers)} loggers")

        # build history manager with loaded batches
        history_manager = HistoryManager()
        for batch in batches:
            history_manager.append_batch(batch)

        last_batch = batches[-1]

        # initialize loggers
        for logger_obj in loggers:
            try:
                # create mock training program
                mock_program = type(
                    'MockTrainingProgram',
                    (),
                    {'training_conf': None, '_save_dir': history_dir.parent},
                )()
                logger_obj.initialize(mock_program)
            except Exception as e:
                logger.error(f"Failed to initialize logger {logger_obj}: {e}")

        # process batches
        last_callback_step: Dict[int, int] = {}

        from tqdm import tqdm
        for batch in tqdm(batches, desc="Processing steps", unit="step"):
            step = batch.step_index
            step_context = LoggerContext(
                output_dir=history_dir.parent,
                current_batch=batch.batch_index,
                current_step=step,
                is_replay=True,
                is_final=False,
            )

            if not final_only:
                for logger_obj in loggers:
                    uses_new = getattr(logger_obj, '_uses_new_pattern', False)
                    freq = getattr(logger_obj, 'frequency', 1)
                    periods = getattr(logger_obj, 'periods', freq)
                    if isinstance(periods, int):
                        freq = periods

                    if uses_new and step > 0 and step % freq == 0:
                        try:
                            window = getattr(logger_obj, 'history_window', None)
                            mode = getattr(logger_obj, 'history_mode', 'window')
                            since = (
                                last_callback_step.get(id(logger_obj), 0)
                                if mode == 'since_last'
                                else None
                            )
                            view = history_manager.get_view(window=window, since_batch=since)
                            logger_obj.on_batch(view, step_context)
                            last_callback_step[id(logger_obj)] = batch.batch_index
                        except Exception as e:
                            logger.error(f"Logger {type(logger_obj).__name__} failed: {e}")

        # call on_end for all loggers that want it
        final_context = LoggerContext(
            output_dir=history_dir.parent,
            current_batch=last_batch.batch_index,
            current_step=last_batch.step_index,
            is_replay=True,
            is_final=True,
        )
        for logger_obj in loggers:
            uses_new = getattr(logger_obj, '_uses_new_pattern', False)
            call_at_end = getattr(logger_obj, 'call_at_end', False)
            if uses_new and call_at_end:
                try:
                    logger.info(f"Generating final figures with {type(logger_obj).__name__}...")
                    window = getattr(logger_obj, 'history_window', None)
                    view = history_manager.get_view(window=window)
                    logger_obj.on_end(view, final_context)
                except Exception as e:
                    logger.error(f"Logger {type(logger_obj).__name__} on_end failed: {e}")

        # finalize
        for logger_obj in loggers:
            try:
                logger_obj.finalize()
            except Exception as e:
                logger.error(f"Failed to finalize logger {logger_obj}: {e}")

        logger.info("Replay completed")
