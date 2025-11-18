from biocomptools.logging_config import get_logger
from biocomptools.toollib.common import config
from biocomptools.toollib.hashutils import get_package_git_hashes
from biocomptools.trainutils import make_unique_dir, make_json_ready, plot_loss, print_matadata
from biocomp.utils import PartialFunction
from biocomp.library import load_lib
from biocomptools.toollib.loggers.logger import Logger, FunctionLogger
from biocomptools.toollib.loggers.plotlogger import PlotLogger
from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger, ConsoleLogger
from biocomptools.toollib.loggers.checkpointlogger import CheckpointLogger
from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.datasources import DataSource, DBSource
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.toollib.networkselector import (
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

import dracon as dr
from dracon.deferred import DeferredNode
from dracon.commandline import make_program, Arg

# Import types for both training and design contexts
from biocomp.train import TrainingConfig
from biocomp.compute import ComputeConfig
from biocomp.datautils import DataConfig
from biocomp.design import DesignManager, DesignConfig, Target
from biocomp.old_network.network import Network, CoTransfection, TranscriptionUnit, Slot

# Import plot types
from biocomptools.plot import DEFAULT_TYPES as PLOT_TYPES
from biocomptools.plot import NetworkPrediction

from pathlib import Path
from typing import Optional, Annotated, TypeVar, Any, Union
from pydantic import Field, BaseModel, ConfigDict
from sqlmodel import Session
from datetime import datetime
from abc import ABC, abstractmethod

logger = get_logger(__name__)

T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class BaseOptimizationProgram(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_dir: Annotated[str, Arg(help='Base directory to save outputs')] = (
        config.paths.training_output
    )
    experiment_name: Annotated[str, Arg(help='Name of the experiment')] = 'default_xp'
    run_name_suffix: str = ''
    metadata: dict[str, Any] = {}

    loggers: Annotated[list[Union[MaybeDeferred[Logger], Logger]], Arg(help='Loggers to use')] = (
        Field(default_factory=lambda: [])
    )

    async_logging: bool = True
    async_store_location: Optional[Path] = Field(
        default_factory=lambda: Path('step_history_data'),
        description='Location to store async logger data.',
    )
    keep_history_on_disk: bool = Field(
        default=False, description='Whether to keep step history files on disk for replay mode.'
    )
    save_all_steps: bool = Field(
        default=False, description='Whether to save step history for every step.'
    )
    n_workers: int = 8

    _lib: Optional[Any] = None
    _yamldump: str = ''
    _modeldump: dict = {}
    _save_dir: Path = Path('.')
    _run_name: Optional[str] = None
    _unique_id: Optional[str] = None

    @property
    def unique_id(self) -> str:
        if self._unique_id is None:
            import uuid

            self._unique_id = str(uuid.uuid4())
        return self._unique_id

    @property
    def _engine(self):
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

    @abstractmethod
    def get_output_subdir(self) -> str:
        pass

    @abstractmethod
    def initialize_context(self):
        pass

    @abstractmethod
    def enrich_metadata(self):
        pass

    @abstractmethod
    async def execute_optimization(self, logger_callbacks, async_handler) -> Any:
        pass

    @abstractmethod
    def save_outputs(self, *args, **kwargs):
        pass

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        logger.debug(f"{self.__class__.__name__} post init")
        self._lib = load_lib()
        self.base_dir = str(Path(self.base_dir).expanduser().resolve())

        self.initialize_context()

        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()
        self._save_dir = make_unique_dir(
            Path(self.base_dir) / self.experiment_name, suffix=self.run_name_suffix
        )
        self._run_name = self._save_dir.name

        self._construct_loggers()
        self.gen_metadata()

    def _construct_loggers(self):
        new_loggers = []
        context = self._get_logger_context()

        for logg in self.loggers:
            if isinstance(logg, DeferredNode):
                logg = logg.construct(context=context)
            new_loggers.append(logg)
        self.loggers = new_loggers

    def _get_logger_context(self) -> dict:
        return {'save_dir': self._save_dir}

    def gen_metadata(self):
        import os
        import socket

        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hashes = get_package_git_hashes(['dracon', 'biocomp', 'biocomptools'])

        self._metadata = {
            f'{self.__class__.__name__.lower()}_id': self.unique_id,
            'start_time': starttime,
            'host': f"{os.environ.get('USER')}@{socket.gethostname()}",
            'biocomp_hash': hashes.get('biocomp', 'unknown'),
            'biocomptools_hash': hashes.get('biocomptools', 'unknown'),
            'dracon_hash': hashes.get('dracon', 'unknown'),
        }

        self._metadata.update(self.metadata)

    async def run(self):
        output_dir = self._save_dir / self.get_output_subdir()
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / f'{self.__class__.__name__.lower()}_dump.yaml', 'w') as f:
            f.write(self._yamldump)

        self.enrich_metadata()

        logger.debug(
            f"Initializing {len(self.loggers)} loggers of types {[type(lg) for lg in self.loggers]}"
        )

        # At this point all loggers should be constructed (not deferred)
        sync_loggers = [lg for lg in self.loggers if isinstance(lg, Logger) and not lg.async_ok]
        async_loggers = [lg for lg in self.loggers if isinstance(lg, Logger) and lg.async_ok]

        for logger_obj in sync_loggers:
            logger_obj.initialize(self)

        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()

        logger_metadata = [
            m for m in (lg.metadata for lg in self.loggers if isinstance(lg, Logger)) if m
        ]
        if logger_metadata:
            self._metadata['loggers'] = make_json_ready(logger_metadata)

        log_file = output_dir / 'output.log.txt'
        log_file.parent.mkdir(exist_ok=True, parents=True)

        all_callbacks = []
        for logger_obj in self.loggers:
            if isinstance(logger_obj, Logger):
                callbacks = logger_obj.get_callbacks(self)
                all_callbacks.extend(
                    [(period, callback, logger_obj) for period, callback in callbacks]
                )

        sync_callbacks = [
            (period, callback)
            for period, callback, logger_obj in all_callbacks
            if not logger_obj.async_ok
        ]
        async_callbacks = [
            (period, callback, logger_obj)
            for period, callback, logger_obj in all_callbacks
            if logger_obj.async_ok
        ]

        logger_callbacks = sync_callbacks.copy()

        async_handler = None
        if self.async_logging and async_callbacks:
            async_handler = self._setup_async_handler(async_callbacks, async_loggers)
            logger_callbacks.append((1, async_handler.create_callback()))

            save_info = ", saving all steps" if self.save_all_steps else ""
            logger.info(
                f"Async logging: {len(sync_callbacks)} sync, {len(async_callbacks)} async (ThreadPool{save_info})"
            )
        elif self.async_logging:
            logger.info("Async logging enabled but no async-capable loggers found")
        else:
            logger_callbacks = [(period, callback) for period, callback, _ in all_callbacks]

        result = await self.execute_optimization(logger_callbacks, async_handler)

        if self.async_logging and async_handler:
            self._shutdown_async_handler(async_handler, result)

        self._metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.save_outputs(*result if isinstance(result, tuple) else [result])

        for logger_obj in self.loggers:
            if isinstance(logger_obj, Logger):
                logger_obj.finalize()

    def _setup_async_handler(self, async_callbacks, async_loggers):
        from biocomptools.async_logger_handler import AsyncLoggerHandler

        async_handler = AsyncLoggerHandler(
            logger_callbacks=async_callbacks,
            n_workers=self.n_workers,
            logger_objects=async_loggers,
            async_store_location=self.async_store_location,
            base_dir=self._save_dir,
            keep_history_on_disk=self.keep_history_on_disk,
            save_all_steps=self.save_all_steps,
        )

        async_handler.initialize_loggers_async(self)
        async_handler.wait_for_initialization()

        return async_handler

    def _shutdown_async_handler(self, async_handler, result):
        step_history = result[-1] if isinstance(result, tuple) and len(result) > 2 else None
        losses = result[1] if isinstance(result, tuple) and len(result) > 1 else []

        async_handler.process_end_loggers(
            step=len(losses) if losses else 0,
            training_config=getattr(self, 'training_conf', getattr(self, 'design_conf', None)),
            step_history=step_history,
            stack=None,
        )
        async_handler.shutdown()

    def save_metadata(self, save_dir: Path):
        with open(save_dir / 'metadata.json', 'w') as f:
            import json

            json.dump(make_json_ready(self._metadata), f, indent=2)

    def save_loss_plot(self, all_losses, save_dir: Path):
        fig, ax = plot_loss(all_losses)
        assert self._metadata, "Metadata not set"
        assert self._run_name, "Run name not set"

        fig = print_matadata(fig, ax, self._metadata, run_name=self._run_name)
        fig.savefig(save_dir / 'summary_loss_plot.pdf')
        logger.debug(f"Saved summary plot to {save_dir / 'summary_loss_plot.pdf'}")


# Import the actual Logger class from the loggers module
# The Logger protocol is already defined in biocomptools.toollib.loggers.logger


# Consolidated default types for both training and design contexts
DEFAULT_TYPES = list(
    set(
        [
            # Network selection and filtering
            Regex,
            NetworkSelector,
            NetworkSet,
            NetworkDataPair,
            NetworkSetUnion,
            NetworkSetIntersection,
            NetworkSetDifference,
            NetworkFilter,
            UorfFilter,
            # Network specifics
            Network,
            TranscriptionUnit,
            CoTransfection,
            Slot,
            # Data and models
            DataSource,
            DBSource,
            BiocompModel,
            ModelSelector,
            NetworkPrediction,
            # Loggers
            Logger,
            FunctionLogger,
            PlotLogger,
            EnhancedConsoleLogger,
            ConsoleLogger,
            CheckpointLogger,
            # Configuration types
            TrainingConfig,
            ComputeConfig,
            DataConfig,
            DesignConfig,
            DesignManager,
            Target,
            Network,
            # Utility types
            PartialFunction,
        ]
        + PLOT_TYPES
    )
)


def make_context_from_types(types):
    return {t.__name__: t for t in types}


async def run_optimization_program(
    program_class: type[BaseOptimizationProgram],
    program_name: str,
    description: str,
    argv: list[str],
    default_types: list[type] = DEFAULT_TYPES,
    context_additions: Optional[dict] = None,
):
    cliprog = make_program(
        program_class,
        name=program_name,
        description=description,
    )

    context = {
        **make_context_from_types(default_types),
        'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
    }
    if context_additions:
        context.update(context_additions)

    program, _ = cliprog.parse_args(
        argv,
        context=context,
        capture_globals=False,
    )

    assert isinstance(program, program_class), f"{program=}"
    await program.run()
