import dill
import time
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from threading import Thread
import queue
import atexit
import shutil

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

# Global registry for callback functions (avoids pickling issues)
_CALLBACK_REGISTRY = {}


# Ray remote function for executing logger callbacks
def _setup_jax_cpu():
    os.environ.update(
        {"JAX_PLATFORM_NAME": "cpu", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
    )
    try:
        import jax

        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_enable_x64', False)
    except:
        pass


def _ray_logger_worker(callback_id, step_file_path, period, step, logger_name=None, logger_type=None):
    _setup_jax_cpu()
    start_time = time.time()
    try:
        callback = _CALLBACK_REGISTRY.get(callback_id)
        if not callback:
            return {"success": False, "step": step, "error": f"Callback {callback_id} not found"}

        step_file = Path(step_file_path)
        if not step_file.exists():
            return {"success": False, "step": step, "error": f"Step file not found"}

        with open(step_file, 'rb') as f:
            data = dill.load(f)

        callback(
            data['step'],
            data['training_config'],
            step_history=data.get('step_history'),
            stack=data.get('stack'),
        )
        elapsed_time = time.time() - start_time
        return {
            "success": True, 
            "step": step, 
            "period": period, 
            "callback_id": callback_id, 
            "elapsed_time": elapsed_time,
            "logger_name": logger_name,
            "logger_type": logger_type
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False, 
            "step": step, 
            "error": str(e), 
            "callback_id": callback_id, 
            "elapsed_time": elapsed_time,
            "logger_name": logger_name,
            "logger_type": logger_type
        }


class AsyncLoggerHandler:
    def __init__(
        self,
        logger_callbacks: List[Tuple[int, Callable]],
        min_period: int,
        use_ray: bool = False,
        n_workers: int = 8,
        logger_objects: Optional[List] = None,  # added for async init
    ):
        self.logger_callbacks = logger_callbacks
        self.logger_objects = logger_objects or []
        self.use_ray = use_ray
        self.tmpdir = Path(tempfile.mkdtemp(prefix="biocomp_async_log_"))
        self.step_queue = queue.Queue()
        self.should_stop = False
        self.callback_ids = {}
        self.n_workers = n_workers
        self.ray_remote_worker = None
        self.ray_initialized = False
        self.executor = None
        self.processing_thread = None
        self.init_futures = []  # track async initialization
    
        if use_ray:
            self._register_callbacks()
        self._initialize()
        atexit.register(self.cleanup)
    
        mode = "Ray" if use_ray else "Thread"
        logger.info(f"AsyncLoggerHandler: {mode} mode, tmpdir: {self.tmpdir}")


    def _extract_logger_info(self, callback):
        """Extract logger name and type from callback function"""
        logger_info = {'name': 'unknown', 'type': 'unknown'}
        
        # Try to get information from closure variables
        if hasattr(callback, '__closure__') and callback.__closure__:
            for cell in callback.__closure__:
                if hasattr(cell.cell_contents, '__class__'):
                    obj = cell.cell_contents
                    class_name = obj.__class__.__name__
                    
                    # Check if it's a logger instance
                    if hasattr(obj, 'name') and hasattr(obj, '__class__'):
                        logger_info['type'] = class_name
                        logger_info['name'] = getattr(obj, 'name', class_name)
                        break
                    
        # Fallback to function name analysis
        if logger_info['name'] == 'unknown' and hasattr(callback, '__name__'):
            func_name = callback.__name__
            if 'validation' in func_name.lower():
                logger_info['type'] = 'ValidationLogger'
            elif 'checkpoint' in func_name.lower():
                logger_info['type'] = 'CheckpointLogger'
            elif 'plot' in func_name.lower():
                logger_info['type'] = 'PlotLogger'
            logger_info['name'] = func_name
            
        return logger_info

    def _register_callbacks(self):
        for i, (period, callback) in enumerate(self.logger_callbacks):
            callback_id = f"async_{id(self)}_{i}_{id(callback)}"
            _CALLBACK_REGISTRY[callback_id] = callback
            
            # Try to extract logger information
            logger_info = self._extract_logger_info(callback)
            callback_name = logger_info.get('name', f'callback_{i}')
            logger_type = logger_info.get('type', 'unknown')
            
            self.callback_ids[(period, callback)] = {
                'id': callback_id,
                'name': callback_name,
                'type': logger_type,
                'period': period
            }
    
    def _get_callback_info(self, period, callback):
        """Get callback info, creating it if needed for thread mode"""
        if (period, callback) not in self.callback_ids:
            # For thread mode, create callback info on demand
            logger_info = self._extract_logger_info(callback)
            callback_name = logger_info.get('name', f'callback_{hash(callback)}')
            logger_type = logger_info.get('type', 'unknown')
            
            self.callback_ids[(period, callback)] = {
                'id': f"thread_{id(self)}_{id(callback)}",
                'name': callback_name,
                'type': logger_type,
                'period': period
            }
        
        return self.callback_ids[(period, callback)]

    def _initialize(self):
        if self.use_ray:
            self._init_ray()
        else:
            self._init_threads()

    def _init_ray(self):
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
                self.ray_initialized = True
            self.ray_remote_worker = ray.remote(_ray_logger_worker)
            self.processing_thread = Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()
        except ImportError:
            logger.error("Ray not available, falling back to threads")
            self.use_ray = False
            self._init_threads()

    def _init_threads(self):
        self.executor = ThreadPoolExecutor(
            max_workers=self.n_workers, thread_name_prefix="async_logger"
        )
        self.processing_thread = Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
    
    def initialize_loggers_async(self, training_program):
        """Initialize async_ok loggers asynchronously"""
        if not self.logger_objects:
            return
        
        logger.info(f"Starting async initialization of {len(self.logger_objects)} loggers")
        
        if self.use_ray:
            # for ray mode, we need special handling since loggers aren't picklable
            logger.warning("Ray mode for logger initialization not yet implemented, falling back to sync")
            for logger_obj in self.logger_objects:
                logger_obj.initialize(training_program)
        else:
            # thread mode
            def init_logger(logger_obj, prog):
                try:
                    start_time = time.time()
                    logger_name = getattr(logger_obj, 'name', logger_obj.__class__.__name__)
                    logger.info(f"AsyncLoggerHandler: Initializing {logger_name} asynchronously")
                    logger_obj.initialize(prog)
                    elapsed = time.time() - start_time
                    logger.info(f"AsyncLoggerHandler: {logger_name} initialized in {elapsed:.2f} seconds")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize logger {logger_obj}: {e}")
                    logger.exception(e)
                    return False
            
            for logger_obj in self.logger_objects:
                future = self.executor.submit(init_logger, logger_obj, training_program)
                self.init_futures.append(future)
    
    def wait_for_initialization(self):
        """Wait for all async logger initializations to complete"""
        if not self.init_futures:
            return
        
        logger.info("Waiting for async logger initialization to complete...")
        success_count = 0
        fail_count = 0
        
        for future in concurrent.futures.as_completed(self.init_futures):
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Logger initialization failed with exception: {e}")
                fail_count += 1
        
        self.init_futures.clear()
        logger.info(f"Async logger initialization complete: {success_count} succeeded, {fail_count} failed")
    def process_start_loggers(self, training_config, stack):
        logger.info("Processing start loggers...")
        self._process_loggers(0, training_config, {}, stack, lambda p, c: p == 0)

    def _process_loggers(self, step, training_config, step_history, stack, filter_fn):
        data = {
            'step': step,
            'training_config': training_config,
            'step_history': step_history,
            'stack': stack,
            'timestamp': time.time(),
        }

        step_file = self.tmpdir / f"temp_{step}_{time.time()}.pkl"
        with open(step_file, 'wb') as f:
            dill.dump(data, f)

        futures = []
        for period, callback in self.logger_callbacks:
            if filter_fn(period, callback):
                if self.use_ray:
                    callback_info = self._get_callback_info(period, callback)
                    future = self.ray_remote_worker.remote(
                        callback_info['id'], str(step_file), period or -1, step,
                        callback_info['name'], callback_info['type']
                    )
                else:
                    future = self.executor.submit(self._safe_call, callback, data, self._get_callback_info(period, callback))
                futures.append(future)

        if futures:
            if self.use_ray:
                import ray

                remaining = set(futures)
                while remaining:
                    done, remaining = ray.wait(list(remaining), num_returns=1, timeout=None)
                    for obj_ref in done:
                        result = ray.get(obj_ref)
                        if not result.get("success", True):
                            logger.error(f"Logger failed: {result.get('error')}")
                        else:
                            elapsed = result.get('elapsed_time', 0)
                            logger_name = result.get('logger_name', 'unknown')
                            logger_type = result.get('logger_type', 'unknown')
                            logger.info(f"AsyncLoggerHandler: {logger_type} '{logger_name}' completed in {elapsed:.2f} seconds (step={result.get('step')})")
            else:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"Logger failed: {e}")
                        logger.exception(e)

        step_file.unlink(missing_ok=True)

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
                step_file = self.tmpdir / f"step_{step:06d}.pkl"
                with open(step_file, 'wb') as f:
                    dill.dump(data, f)
                self.step_queue.put(step)
            except Exception as e:
                logger.error(f"Failed to save step data: {e}")
                logger.exception(e)
                return

        return async_callback

    def _process_queue(self):
        while not self.should_stop:
            try:
                step = self.step_queue.get(timeout=1.0)
                if step is None:
                    break
                self._process_step(step)
                self.step_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                logger.exception(e)
                return

    def _process_step(self, step: int):
        step_file = self.tmpdir / f"step_{step:06d}.pkl"
        if not step_file.exists():
            logger.warning(f"Step file {step_file} not found")
            return

        try:
            with open(step_file, 'rb') as f:
                data = dill.load(f)

            futures = []
            for period, callback in self.logger_callbacks:
                should_call = (
                    (period == 0 and step == 0)
                    or (period is None or period == -1)  # skip end loggers here
                    or (period and period > 0 and step % period == 0)
                )
                if should_call and not (period is None or period == -1):
                    if self.use_ray:
                        callback_info = self._get_callback_info(period, callback)
                        future = self.ray_remote_worker.remote(
                            callback_info['id'], str(step_file), period, step,
                            callback_info['name'], callback_info['type']
                        )
                    else:
                        future = self.executor.submit(self._safe_call, callback, data, self._get_callback_info(period, callback))
                    futures.append(future)

            if futures:
                if self.use_ray:
                    import ray

                    results = ray.get(futures)
                    for result in results:
                        if not result.get("success", True):
                            logger.error(f"Ray logger failed: {result.get('error')}")
                        else:
                            elapsed = result.get('elapsed_time', 0)
                            logger_name = result.get('logger_name', 'unknown')
                            logger_type = result.get('logger_type', 'unknown')
                            logger.info(f"AsyncLoggerHandler: {logger_type} '{logger_name}' completed in {elapsed:.2f} seconds (step={step})")
                else:
                    for future in futures:
                        try:
                            future.result(timeout=300)
                        except Exception as e:
                            logger.error(f"Logger failed: {e}")

            step_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Step {step} processing failed: {e}")
            logger.exception(e)
            return

    def _safe_call(self, callback, data, callback_info):
        start_time = time.time()
        logger_name = callback_info.get('name', 'unknown')
        logger_type = callback_info.get('type', 'unknown')
        try:
            callback(
                data['step'],
                data['training_config'],
                step_history=data.get('step_history'),
                stack=data.get('stack'),
            )
            elapsed_time = time.time() - start_time
            logger.info(f"AsyncLoggerHandler: {logger_type} '{logger_name}' completed in {elapsed_time:.2f} seconds (step={data['step']}, thread mode)")
            return {"success": True, "elapsed_time": elapsed_time, "logger_name": logger_name, "logger_type": logger_type}
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"AsyncLoggerHandler: {logger_type} '{logger_name}' failed after {elapsed_time:.2f} seconds: {e}")
            logger.exception(e)
            return {"success": False, "elapsed_time": elapsed_time, "logger_name": logger_name, "logger_type": logger_type}

    def process_end_loggers(self, step, training_config, step_history, stack):
        logger.info("Processing end loggers...")
        self._process_loggers(
            step, training_config, step_history, stack, lambda p, c: p is None or p == -1
        )

    def shutdown(self):
        logger.info("Shutting down AsyncLoggerHandler...")
        self.step_queue.join()
        self.should_stop = True
        self.step_queue.put(None)
        if self.processing_thread:
            self.processing_thread.join(timeout=10)

        if self.use_ray:
            try:
                for callback_info in self.callback_ids.values():
                    if isinstance(callback_info, dict):
                        _CALLBACK_REGISTRY.pop(callback_info['id'], None)
                    else:
                        # backward compatibility
                        _CALLBACK_REGISTRY.pop(callback_info, None)
                if self.ray_initialized:
                    import ray

                    ray.shutdown()
            except Exception as e:
                logger.warning(f"Ray shutdown error: {e}")
        elif self.executor:
            self.executor.shutdown(wait=True)

        self.cleanup()

    def cleanup(self):
        if self.tmpdir.exists():
            try:
                shutil.rmtree(self.tmpdir)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
