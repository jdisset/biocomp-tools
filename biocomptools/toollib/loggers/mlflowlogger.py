import mlflow
import sys
from pathlib import Path
import json
from typing import Any, List, Optional, Tuple, Callable
from pydantic import Field, ConfigDict, BaseModel
import numpy as np
from dracon.deferred import DeferredNode
from biocomptools.toollib.common import config
from biocomptools.toollib.loggers.logger import Logger


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

    def _make_exportable_program_dump(self, trainprog):
        """Roundtrip to json to iron out any weakref/unpickleable issues with DeferredNodes"""

        def convert(o):
            if isinstance(o, DeferredNode):
                return {f'{o.value.tag}': 'deferred'}
            elif isinstance(o, BaseModel):
                return o.model_dump()
            else:
                logger.error(f"Unhandled type during json serialization: {type(o)}")
                return str(o)

        dmp = json.dumps(trainprog, default=convert)

        return json.loads(dmp)

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
            logger.error("Failed to validate MLflow connection - logging will be disabled")
            return

        # Get or create the experiment
        try:
            experiment = mlflow.get_experiment_by_name(training_program.experiment_name)
            if experiment is None:
                logger.debug(f"Creating new experiment '{training_program.experiment_name}'")
                experiment_id = mlflow.create_experiment(training_program.experiment_name)
            else:
                logger.debug(f"Using existing experiment '{training_program.experiment_name}'")
                experiment_id = experiment.experiment_id

            # Start the run with the same name as the training program's run
            self._active_run = mlflow.start_run(
                experiment_id=experiment_id, run_name=training_program._run_name
            )

            # Enable system metrics logging
            mlflow.enable_system_metrics_logging()

            # Log initial metadata and config
            self._log_system_info()
            self._log_training_config()
            self._log_datamanager_info()

        except Exception as e:
            logger.error("Failed to set up MLflow experiment")
            logger.exception(e)
            return

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
                    "training_config": self._make_exportable_program_dump(
                        self._training_program.training_conf
                    ),
                    "compute_config": self._make_exportable_program_dump(
                        self._training_program.compute_conf
                    ),
                    "data_config": self._make_exportable_program_dump(
                        self._training_program.data_conf
                    ),
                    "training_set": self._make_exportable_program_dump(
                        self._training_program.training_set.content
                    ),
                    "validation_set": self._make_exportable_program_dump(
                        self._training_program.validation_set.content
                    ),
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
                logger.info("No step history provided")
                return

            try:
                # Log loss metrics
                losses = step_history.get('loss')
                if losses is not None:
                    metrics = {}

                    # Log individual replicate losses
                    for i, loss in enumerate(losses):
                        if not np.isnan(loss).any():
                            replicate_mean = float(np.mean(loss))
                            metrics[f"loss_replicate_{i}"] = replicate_mean

                    # Log aggregate metrics
                    valid_losses = [l for l in losses if not np.isnan(l).any()]
                    if valid_losses:
                        all_losses = np.concatenate(valid_losses)
                        metrics.update(
                            {
                                "loss_min": float(np.min(all_losses)),
                            }
                        )

                    mlflow.log_metrics(metrics, step=step)
                    logger.debug(f"Logged loss metrics for {len(losses)} replicates")

                # Log any other metrics from step_history
                other_metrics = step_history.get('metrics', {})
                if other_metrics:
                    mlflow.log_metrics(other_metrics, step=step)
                    logger.debug(f"Logged {len(other_metrics)} additional metrics")

                # Log any new plots
                if self.log_plots:
                    self._log_new_artifacts()

            except Exception as e:
                logger.error(f"Error logging metrics for step {step}: {e}")

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
