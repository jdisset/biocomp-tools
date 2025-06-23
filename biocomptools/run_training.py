## {{{                          --     imports     --
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.modelmodel import BiocompModel, get_shared_params
from biocomptools.toollib.datasources import DataSource, DBSource
from biocomptools.toollib.common import config
from biocomptools.toollib.hashutils import get_package_git_hashes
from biocomptools.toollib.loggers.logger import Logger, FunctionLogger
from biocomptools.toollib.loggers.plotlogger import PlotLogger
from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger, ConsoleLogger
from biocomptools.toollib.loggers.checkpointlogger import CheckpointLogger
from biocomptools.trainutils import (
    plot_loss,
    make_unique_dir,
    print_matadata,
    get_best_smoothed_loss_replicate_id,
    get_latest_avg_loss,
    make_json_ready,
)

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG
from biocomp.library import PartsLibrary
from biocomp.utils import PartialFunction, load_lib, save
from biocomp.train import TrainingConfig
from functools import partial

import dracon as dr
from dracon.deferred import DeferredNode
from dracon.commandline import make_program, Arg
import asyncio

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Annotated, TypeVar, Any, Callable, Dict
from pydantic import Field, BaseModel, ConfigDict
from sqlmodel import Session
from datetime import datetime


from biocomptools.toollib.networkselector import (
    build_data_manager,
    NetworkSelector,
    Regex,
    NetworkDataPair,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    NetworkFilter,
    NetworkSet,
    UorfFilter,
)

from biocomptools.plot import DEFAULT_TYPES as PLOT_TYPES
from biocomptools.plot import NetworkPrediction


setup_logging(force=False)
logger = get_logger(__name__)


DEFAULT_TYPES = [
    Regex,
    NetworkSelector,
    NetworkSet,
    NetworkDataPair,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    PartialFunction,
    NetworkFilter,
    UorfFilter,
    DataSource,
    DBSource,
    NetworkPrediction,
    BiocompModel,
    PlotLogger,
    EnhancedConsoleLogger,
    ConsoleLogger,
    FunctionLogger,
    CheckpointLogger,
] + PLOT_TYPES

DEFAULT_TYPES = list(set(DEFAULT_TYPES))


def make_context_from_types(types):
    return {t.__name__: t for t in types}


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                      --     TrainingProgram     --

T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class TrainingProgram(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    training_conf: Annotated[TrainingConfig, Arg(help='Training config')] = Field(
        default_factory=TrainingConfig
    )
    compute_conf: Annotated[ComputeConfig, Arg(help='Compute config')] = DEFAULT_COMPUTE_CONFIG
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = DEFAULT_DATA_CONFIG

    training_set: Annotated[NetworkSet, Arg(help='Networks in training set')] = Field(
        default_factory=NetworkSet
    )

    base_dir: Annotated[
        str,
        Arg(
            help='Base directory to save model (will be saved under <base_dir>/<experiment_name>/<run_name>)'
        ),
    ] = config.paths.training_output

    loggers: Annotated[List[MaybeDeferred[Logger]], Arg(help='Loggers to use')] = Field(
        default_factory=lambda: []
    )

    experiment_name: Annotated[str, Arg(help='Name of the experiment')] = 'default_xp'
    metadata: Dict[str, Any] = {}

    run_name_suffix: str = ''

    use_jax_sampling: bool = True

    # logging configuration
    async_logging: bool = True
    async_store_location: Optional[Path] = Field(
        default_factory=lambda: Path('step_history_data'),
        description='Location to store async logger data. If None, uses a temporary directory.',
    )
    keep_history_on_disk: bool = Field(
        default=True,
        description='Whether to keep step history files on disk indefinitely for replay mode.',
    )
    save_all_steps: bool = Field(
        default=True,
        description='Whether to save step history for every training step, regardless of logger periods.',
    )
    n_workers: int = 8

    # Private
    _lib: Optional[PartsLibrary] = None
    _yamldump: str = ''
    _modeldump: dict = {}
    _save_dir: Path = Path('.')
    _run_name: Optional[str] = None
    _training_dman: Optional[Any] = None
    _training_id: Optional[str] = None

    @property
    def training_id(self) -> str:
        """Unique identifier for this training run."""
        if self._training_id is None:
            import uuid
            self._training_id = str(uuid.uuid4())
        return self._training_id

    @property
    def _engine(self):
        """Lazy-load the database engine when needed (otherwise unpicklable)."""
        from biocomptools.toollib.models import get_biocompdb_sqlite_engine
        from biocomptools.toollib.common import config

        _db_engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)
        return _db_engine

    @property
    def db_session(self):
        return Session(self._engine)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    @property
    def parts_library(self):
        assert self._lib, "Library not loaded"
        return self._lib

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        logger.debug("Training program post init")
        self._lib = load_lib()
        self.base_dir = str(Path(self.base_dir).expanduser().resolve())

        with self.db_session as session:
            self.training_set.run_selectors(session)
            session.expunge_all()
            session.close()

        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()
        self._save_dir = make_unique_dir(
            Path(self.base_dir) / self.experiment_name, suffix=self.run_name_suffix
        )
        self._run_name = self._save_dir.name

        # construct loggers
        new_loggers = []

        for logg in self.loggers:
            if isinstance(logg, DeferredNode):
                logg = logg.construct(
                    context={
                        'save_dir': self._save_dir,
                        'compute_conf': self.compute_conf,
                        'data_conf': self.data_conf,
                        'training_set': self.training_set,
                    }
                )
            new_loggers.append(logg)
        self.loggers = new_loggers
        self.gen_metadata()

    def gen_metadata(self):
        import os
        import socket

        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        hashes = get_package_git_hashes(['dracon', 'biocomp', 'biocomptools'])

        self._metadata = {
            'training_id': self.training_id,
            'start_time': starttime,
            'host': f"{os.environ.get('USER')}@{socket.gethostname()}",
            'biocomp_hash': hashes.get('biocomp', 'unknown'),
            'biocomptools_hash': hashes.get('biocomptools', 'unknown'),
            'dracon_hash': hashes.get('dracon', 'unknown'),
        }

        self._metadata.update(self.metadata)

    def _build_dman(self):
        self._training_dman = build_data_manager(
            lib=self.parts_library,
            db_session=self.db_session,
            path_prefix=self.path_prefix,
            data_conf=self.data_conf,
            dataset=self.training_set,
        )
        self._training_dman.jax_sampling = self.use_jax_sampling

    def _enrich_metadata(self):
        """Adds detailed run and data information to the metadata dictionary."""
        assert self._training_dman is not None
        dman = self._training_dman

        dataman_info = {
            "network_names": [n.name for n in dman.get_networks()],
            "input_dimensions": [x.shape[1] for x in dman.get_X()],
            "output_dimensions": [y.shape[1] for y in dman.get_Y()],
            "data_config": dman.data_cfg.model_dump(),
        }

        self._metadata.update(
            {
                'run_name': self._run_name,
                'experiment_name': self.experiment_name,
                'training_set': {
                    'content': self.training_set.content,
                    'name': self.training_set.name,
                },
                'training_conf': self.training_conf,
                'compute_conf': self.compute_conf,
                'data_conf': self.data_conf,
                'data_manager_info': dataman_info,
                'final_model_dump': self._modeldump,
            }
        )

    async def run(self):
        from biocomp.train import start

        # Prepare output directory
        self._training_dir = self._save_dir / 'training'
        self._training_dir.mkdir(exist_ok=True, parents=True)
        with open(self._training_dir / 'training_program_dump.yaml', 'w') as f:
            f.write(self._yamldump)

        self._build_dman()
        self._enrich_metadata()

        # Initialize loggers
        logger.debug(
            f"Initializing {len(self.loggers)} loggers of types {[type(l) for l in self.loggers]}"
        )

        # separate sync and async loggers for initialization
        sync_loggers = [l for l in self.loggers if not l.async_ok]
        async_loggers = [l for l in self.loggers if l.async_ok]

        # initialize sync loggers immediately
        for logger_obj in sync_loggers:
            logger_obj.initialize(self)

        # async logger initialization will be handled by AsyncLoggerHandler if enabled

        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()

        # add logger metadata to the metadata
        logger_metadata = [m for m in (logger.metadata for logger in self.loggers) if m]
        if logger_metadata:
            self._metadata['loggers'] = make_json_ready(logger_metadata)

        # Set up logging to file
        log_file = self._training_dir / 'output.log.txt'
        log_file.parent.mkdir(exist_ok=True, parents=True)

        # Collect and separate sync/async logger callbacks
        all_callbacks = []
        for logger_obj in self.loggers:
            callbacks = logger_obj.get_callbacks(self)
            # add logger reference to each callback for async_ok check
            all_callbacks.extend([(period, callback, logger_obj) for period, callback in callbacks])

        # Separate sync and async callbacks
        sync_callbacks = [
            (period, callback)
            for period, callback, logger_obj in all_callbacks
            if not logger_obj.async_ok
        ]
        async_callbacks = [
            (period, callback)
            for period, callback, logger_obj in all_callbacks
            if logger_obj.async_ok
        ]

        logger_callbacks = sync_callbacks.copy()  # start with sync callbacks

        # Handle async logging if enabled and we have async-capable loggers
        async_handler = None
        if self.async_logging and async_callbacks:
            from biocomptools.async_logger_handler import AsyncLoggerHandler

            # find minimum period for async callbacks only
            async_periods = [period for period, _ in async_callbacks if period and period > 0]
            min_period = min(async_periods) if async_periods else 1

            # If save_all_steps is enabled, force min_period to 1 to save every step
            if self.save_all_steps:
                min_period = 1

            # create async handler for async callbacks only
            async_handler = AsyncLoggerHandler(
                logger_callbacks=async_callbacks,
                min_period=min_period,
                n_workers=self.n_workers,
                logger_objects=async_loggers,
                async_store_location=self.async_store_location,
                base_dir=self._save_dir,
                keep_history_on_disk=self.keep_history_on_disk,
                save_all_steps=self.save_all_steps,
            )

            # initialize async loggers asynchronously
            async_handler.initialize_loggers_async(self)

            # wait for initialization to complete before starting training
            async_handler.wait_for_initialization()

            # add single async callback to logger_callbacks
            logger_callbacks.append((min_period, async_handler.create_callback()))

            save_info = f", saving all steps" if self.save_all_steps else ""
            logger.info(
                f"Async logging: {len(sync_callbacks)} sync, {len(async_callbacks)} async (ThreadPool, min_period={min_period}{save_info})"
            )
        elif self.async_logging:
            logger.info("Async logging enabled but no async-capable loggers found")
        else:
            logger_callbacks = [(period, callback) for period, callback, _ in all_callbacks]

        # Start training
        all_params, all_losses, step_history = start(
            self._training_dman,
            self.training_conf,
            self.compute_conf,
            loggers=logger_callbacks,
            async_handler=async_handler,
        )

        if self.async_logging and async_handler:
            async_handler.process_end_loggers(
                step=len(all_losses) if all_losses else 0,
                training_config=self.training_conf,
                step_history=step_history,
                stack=None,
            )
            async_handler.shutdown()

        self._metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_outputs(all_params, all_losses, self._training_dir)

        for logger_obj in self.loggers:
            logger_obj.finalize()

    def get_replicate_model_func(self):
        """Returns a factory function for creating a BiocompModel for a specific replicate."""
        from copy import deepcopy

        compute_conf = deepcopy(self.compute_conf)
        data_conf = deepcopy(self.data_conf)
        # self._metadata is now fully enriched, so we just need to pass it
        base_metadata = make_json_ready(self._metadata)

        return partial(
            create_replicate_model,
            compute_conf=compute_conf,
            rescaler=data_conf.rescaler,
            base_metadata=base_metadata,
            loggers=self.loggers,
        )

    def get_best_model_func(self):
        """Returns a factory function for creating the best BiocompModel."""
        replicate_model_factory = self.get_replicate_model_func()
        return partial(get_best_model, model_factory=replicate_model_factory)

    def save_best(self, all_params, all_losses: list[np.ndarray], save_dir, name=None):
        model_factory = self.get_best_model_func()
        model = model_factory(all_params=all_params, all_losses=all_losses)
        if model is None:
            logger.error("!!!!!! No best model found !!!!!")
            return
        if name is None:
            name = f"{model.signature}.bestmodel"
        fname = save_dir / f'{name}.pickle'
        model.save(fname)
        logger.debug(f"Saved best model to {fname}")

    def save_outputs(self, all_params, all_losses: list, save_dir: Path):
        self.save_best(all_params, all_losses, save_dir)

        # Save loss history
        np.save(save_dir / 'loss_history.npy', all_losses)

        # Collect per-replicate metrics from loggers for the final run metadata
        logger_metrics = [
            m for m in (logger.get_metrics(replicate=None) for logger in self.loggers) if m
        ]
        if logger_metrics:
            self._metadata['logger_metrics_all_replicates'] = make_json_ready(logger_metrics)

        fig, ax = plot_loss(all_losses)
        assert self._metadata, "Metadata not set"
        assert self._run_name, "Run name not set"

        fig = print_matadata(fig, ax, self._metadata, run_name=self._run_name)
        fig.savefig(save_dir / 'summary_loss_plot.pdf')

        with open(save_dir / 'metadata.json', 'w') as f:
            import json

            json.dump(make_json_ready(self._metadata), f, indent=2)

        logger.debug(f"Saved summary plot to {save_dir / 'summary_loss_plot.pdf'}")



##────────────────────────────────────────────────────────────────────────────}}}


def create_replicate_model(
    all_params, all_losses, replicate_id, compute_conf, rescaler, base_metadata, loggers
):
    """Creates a BiocompModel for a single, specific replicate with its latest metrics."""
    from biocomp.jaxutils import tree_get, tree_to_np
    import pickle

    params = tree_get(all_params, replicate_id)
    if params is None:
        logger.warning(f"No parameters found for replicate {replicate_id}.")
        return None

    shared_params = get_shared_params(pickle.loads(pickle.dumps(params)))
    if shared_params is None:
        return None

    local_metadata = base_metadata.copy()
    local_metadata['replicate_number'] = replicate_id

    latest_loss = get_latest_avg_loss(all_losses, replicate_id)
    if not np.isnan(latest_loss):
        local_metadata['training_loss'] = latest_loss

    rep_metrics = [
        m for m in (logger.get_metrics(replicate=replicate_id) for logger in loggers) if m
    ]
    if rep_metrics:
        local_metadata['logger_metrics'] = make_json_ready(rep_metrics)

    model = BiocompModel(
        compute_config=compute_conf,
        rescaler=rescaler,
        shared_params=tree_to_np(shared_params),
        metadata=make_json_ready(local_metadata),
    )
    return model


def get_best_model(all_params, all_losses, model_factory: Callable):
    """Finds the best replicate and uses the model_factory to create its model."""
    import numpy as np
    
    # Handle case where all_losses is not a list (single loss array)
    if not isinstance(all_losses, list):
        all_losses = [all_losses]
    
    # Check dimensions match parameter tree structure
    if all_params is not None and len(all_losses) > 0:
        try:
            # Get expected number of replicates from parameter structure
            first_loss = all_losses[0]
            if hasattr(first_loss, 'shape') and len(first_loss.shape) > 0:
                n_replicates_from_loss = first_loss.shape[0]
                
                # Check if params structure matches
                if hasattr(all_params, 'iter_leaves'):
                    # Get a sample parameter to check replicate dimension
                    for path, param in all_params.iter_leaves():
                        if hasattr(param, 'shape') and len(param.shape) > 0:
                            n_replicates_from_params = param.shape[0]
                            if n_replicates_from_params != n_replicates_from_loss:
                                logger.warning(f"Dimension mismatch: params have {n_replicates_from_params} replicates, losses have {n_replicates_from_loss}")
                            break
        except Exception as e:
            logger.debug(f"Could not check parameter dimensions: {e}")
    
    best_model_id, _, _ = get_best_smoothed_loss_replicate_id(all_losses)
    if best_model_id == -1:
        logger.warning("Could not determine best model.")
        return None

    logger.debug(f"Best model is replicate number {best_model_id}")
    return model_factory(all_params=all_params, all_losses=all_losses, replicate_id=best_model_id)


async def main_async():
    cliprog = make_program(
        TrainingProgram,
        name='biocomp-train',
        description='Start training biocomp models.',
    )
    trainprog, _ = cliprog.parse_args(
        sys.argv[1:],
        context={
            **make_context_from_types(DEFAULT_TYPES),
            'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
        },
        capture_globals=False,
    )
    assert isinstance(trainprog, TrainingProgram), f"{trainprog=}"
    await trainprog.run()


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
