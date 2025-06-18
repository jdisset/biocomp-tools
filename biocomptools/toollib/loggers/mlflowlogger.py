import mlflow
import sys
from pathlib import Path
import json
from typing import Any, List, Optional, Tuple, Callable, Dict
from pydantic import Field, ConfigDict, BaseModel
import numpy as np
from dracon.deferred import DeferredNode
from biocomptools.toollib.common import config
import time
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.trainutils import make_json_ready


from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class MLflowLogger(Logger):
    """
    Logs experiments, runs, metrics, parameters, models and artifacts to MLflow.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tracking_url: str = Field(
        default=config.mlflow.server, description="URL of the MLflow tracking server"
    )
    log_artifacts: bool = Field(default=True, description="Whether to log model artifacts")
    log_plots: bool = Field(default=True, description="Whether to log generated plots")

    # Valid file extensions to log as artifacts
    valid_extensions: list[str] = Field(
        default=[
            '.pdf',
            '.txt',
            '.md',
            '.png',
            '.jpg',
            '.jpeg',
            '.gif',
            '.bmp',
            '.tiff',
            '.webp',
            '.svg',
            '.mp4',
            '.json',
            '.csv',
            '.yaml',
            '.pickle',
            '.pkl',
            '.npy',
        ],
        description="Valid file extensions to log",
    )

    # Private state
    _training_program: Optional[Any] = None
    _logged_plots: set[str] = set()  # Track logged plots by content hash
    _active_run = None

    def _flatten_dict(self, d: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten a nested dictionary or a list of dictionaries."""
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = parent_key + sep + str(i)
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((parent_key, d))
        return dict(items)

    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize metric names for MLFlow compatibility.
        
        MLFlow only allows: alphanumerics, underscores (_), dashes (-), 
        periods (.), spaces ( ), colon(:) and slashes (/).
        """
        import re
        # Replace @ and other invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_\-.\s:/]', '_', name)
        return sanitized

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all metric names in a dictionary."""
        return {self._sanitize_metric_name(k): v for k, v in metrics.items()}

    def _log_new_artifacts(self):
        """Log any new plots/images/text files that haven't been logged yet"""
        assert self._training_program is not None
        self._log_artifacts_in_dir(self._training_program._save_dir / "plots")
        self._log_artifacts_in_dir(self._training_program._save_dir / "predictions")
        self._log_artifacts_in_dir(self._training_program._save_dir / "training")

    def _log_artifacts_in_dir(self, dir_path: Path):
        """Log artifacts from a directory, maintaining proper relative paths"""
        import xxhash

        assert self._training_program is not None
        try:
            if not dir_path.exists():
                return

            for ext in self.valid_extensions:
                for file_path in dir_path.glob(f"**/*{ext}"):
                    # Calculate hash of file content + path to ensure uniqueness
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        file_hash = xxhash.xxh64(content + str(file_path).encode()).hexdigest()

                    if file_hash not in self._logged_plots:
                        # Get path relative to training program save dir for artifact logging
                        rel_path = file_path.relative_to(self._training_program._save_dir)
                        # Extract just the directory part for artifact_path
                        artifact_dir = str(rel_path.parent)
                        mlflow.log_artifact(str(file_path), artifact_dir)
                        self._logged_plots.add(file_hash)
                        logger.debug(f"Logged new file: {rel_path} (hash: {file_hash})")

        except Exception as e:
            logger.error(f"Failed to log files in directory {dir_path}: {e}")

    def validate_connection(self) -> bool:
        """Test connection to MLflow tracking server"""
        try:
            logger.debug(f"Testing connection to MLflow server at {self.tracking_url}")
            mlflow.search_experiments()
            logger.debug("Successfully connected to MLflow tracking server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MLflow tracking server: {e}")
            return False

    def _log_system_info(self):
        """Log system information as tags"""
        import platform

        mlflow.set_tags(
            {
                "python_version": platform.python_version(),
                "system": platform.system(),
                "processor": platform.processor(),
                "hostname": platform.node(),
            }
        )

    def initialize(self, training_program):
        """Set up MLflow experiment and start run"""
        logger.debug("Initializing MLflow logger")
        self._training_program = training_program
        self._logged_plots = set()

        # Set up MLflow connection
        mlflow.set_tracking_uri(self.tracking_url)

        if not self.validate_connection():
            error_msg = f"Failed to validate MLflow connection to {self.tracking_url}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get or create the experiment
        try:
            # Use set_experiment which handles creation automatically
            mlflow.set_experiment(training_program.experiment_name)
            
            # Start the run with the same name as the training program's run
            self._active_run = mlflow.start_run(
                run_name=training_program._run_name
            )

            # Enable system metrics logging
            mlflow.enable_system_metrics_logging()

            # Log initial metadata and config
            self._log_system_info()
            self._log_training_config()
            self._log_datamanager_info()

        except Exception as e:
            error_msg = f"Failed to set up MLflow experiment '{training_program.experiment_name}': {e}"
            logger.error(error_msg)
            logger.exception(e)
            raise RuntimeError(error_msg)

    def _log_training_config(self):
        """Log all configuration objects and metadata"""
        t0 = time.time()
        logger.debug("Logging training configuration")
        assert self._training_program is not None

        try:
            # Log command line info
            cmd_info = {
                "command": " ".join(sys.argv),
                "cwd": str(Path.cwd()),
                "python_executable": sys.executable,
            }
            mlflow.log_params({"command_line": json.dumps(cmd_info)})

            # Log configuration parameters
            mlflow.log_params(
                {
                    "training_config": make_json_ready(self._training_program.training_conf),
                    "compute_config": make_json_ready(self._training_program.compute_conf),
                    "data_config": make_json_ready(self._training_program.data_conf),
                    "training_set": make_json_ready(self._training_program.training_set.content),
                }
            )

            mlflow.log_text(self._training_program._yamldump, "fullconfig.yaml")

            logger.debug(f"Done logging training configuration. Took {time.time() - t0:.2f}s")

        except Exception as e:
            logger.error("Failed to log training config")
            logger.exception(e)

    def _log_datamanager_info(self):
        """Log information about the training and validation datasets"""
        assert self._training_program is not None

        try:
            if hasattr(self._training_program, '_training_dman'):
                dman = self._training_program._training_dman

                # Create a summary of the dataset
                dataset_info = {
                    "training_set_size": sum(x.shape[0] for x in dman.get_X()),
                    "n_networks": len(dman.get_networks()),
                    "networks": [n.name for n in dman.get_networks()],
                    "input_dimensions": [x.shape[1] for x in dman.get_X()],
                    "output_dimensions": [y.shape[1] for y in dman.get_Y()],
                    "rescaler": dman.data_cfg.rescaler.__class__.__name__,
                    "data_config": dman.data_cfg.model_dump(),
                }

                # Log dataset summary
                mlflow.log_dict(dataset_info, "dataset_info.json")

        except Exception as e:
            logger.error("Failed to log dataset info")
            logger.exception(e)

    def _log_checkpoint(self, step: int, params: Any, losses: Optional[List[np.ndarray]] = None):
        """Log model checkpoint and register if it's the best model"""
        assert self._training_program is not None
        try:
            save_dir = self._training_program._save_dir / "training/checkpoints"
            save_dir.mkdir(exist_ok=True, parents=True)

            # Save model locally first
            file_name = f"best_model_at_step_{step}.pickle"
            model_path = save_dir / file_name
            self._training_program.save_best(
                all_models=params,
                all_losses=losses,
                save_dir=save_dir,
                name=file_name,
            )

            # Log the model file as an artifact
            if model_path.exists():
                logger.debug(f"Logging model checkpoint for step {step}")
                mlflow.log_artifact(str(model_path), f"models/step_{step}")
                logger.debug("Done logging model checkpoint")
            else:
                logger.error(f"Failed to save model checkpoint for step {step}")

        except Exception as e:
            logger.error("Failed to log model checkpoint")
            logger.exception(e)

    def _log_artifacts_in_dir(self, dir_path: Path):
        """Log artifacts from a directory, maintaining proper relative paths"""
        import xxhash

        assert self._training_program is not None
        try:
            if not dir_path.exists():
                return

            for ext in self.valid_extensions:
                for file_path in dir_path.glob(f"**/*{ext}"):
                    # Calculate hash of file content + path to ensure uniqueness
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        file_hash = xxhash.xxh64(content + str(file_path).encode()).hexdigest()

                    if file_hash not in self._logged_plots:
                        # Get path relative to training program save dir
                        rel_path = file_path.relative_to(self._training_program._save_dir)
                        # Use just the parent directory as artifact_path
                        artifact_path = str(rel_path.parent)
                        # Log with the original filename but in the correct artifact path
                        mlflow.log_artifact(str(file_path), artifact_path)
                        self._logged_plots.add(file_hash)
                        logger.debug(f"Logged new file: {rel_path} (hash: {file_hash})")

        except Exception as e:
            logger.error(f"Failed to log files in directory {dir_path}: {e}")

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Return callbacks for logging during training"""
        assert self._training_program is not None

        def log_metrics(step, training_config, step_history=None, **kwargs):
            logger.debug(f"Processing metrics for step {step}")
            if step_history is None:
                return

            try:
                # Log training loss from step_history
                losses = step_history.get('loss')
                if losses is not None:
                    metrics = {
                        f"loss_replicate_{i}": float(np.mean(loss))
                        for i, loss in enumerate(losses)
                        if not np.isnan(loss).any()
                    }
                    valid_losses = [l for l in losses if not np.isnan(l).any()]
                    if valid_losses:
                        metrics["loss_min"] = float(np.min(np.concatenate(valid_losses)))
                    sanitized_metrics = self._sanitize_metrics(metrics)
                    mlflow.log_metrics(sanitized_metrics, step=step)

                # Log metrics from other loggers
                all_logger_metrics = {}
                for logger_obj in self._training_program.loggers:
                    if logger_obj is self:
                        continue

                    logger_metrics = logger_obj.get_metrics(replicate=None)
                    if logger_metrics:
                        flat_metrics = self._flatten_dict(logger_metrics)
                        all_logger_metrics.update(flat_metrics)

                if all_logger_metrics:
                    numeric_metrics = {
                        k: v
                        for k, v in all_logger_metrics.items()
                        if isinstance(v, (int, float, np.number))
                    }
                    if len(numeric_metrics) != len(all_logger_metrics):
                        logger.warning(
                            "Some logger metrics were non-numeric and skipped by MLflow."
                        )

                    if numeric_metrics:
                        sanitized_metrics = self._sanitize_metrics(numeric_metrics)
                        mlflow.log_metrics(sanitized_metrics, step=step)
                        logger.debug(f"Logged {len(sanitized_metrics)} metrics from other loggers.")

                # Log any other metrics from step_history and new plots
                other_metrics = step_history.get('metrics', {})
                if other_metrics:
                    sanitized_other_metrics = self._sanitize_metrics(other_metrics)
                    mlflow.log_metrics(sanitized_other_metrics, step=step)
                if self.log_plots:
                    self._log_new_artifacts()

            except Exception as e:
                logger.error(f"Error logging metrics for step {step}: {e}")
                logger.exception(e)

        return [(self.periods, log_metrics)]

    def finalize(self):
        """Final cleanup and logging"""
        logger.debug("Finalizing MLflow logger")
        try:
            # Do one final check for new plots
            if self.log_plots:
                self._log_new_artifacts()

            # End the MLflow run
            if self._active_run:
                mlflow.end_run()

            logger.debug("MLflow logging completed successfully")

        except Exception as e:
            logger.error(f"Error during MLflow logger cleanup: {e}")
