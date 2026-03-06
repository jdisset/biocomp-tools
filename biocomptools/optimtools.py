from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.toollib.common import config, make_context_from_types
from biocomptools.toollib.hashutils import get_package_git_hashes
from biocomptools.trainutils import make_unique_dir, make_json_ready, plot_loss, print_matadata
from biocomp.utils import PartialFunction
from biocomp.library import load_lib
from biocomptools.toollib.loggers.logger import Logger, FunctionLogger
from biocomptools.toollib.loggers.plotlogger import PlotLogger
from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger, ConsoleLogger
from biocomptools.toollib.loggers.checkpointlogger import CheckpointLogger
from biocomptools.toollib.loggers.design_summary_logger import DesignSummaryLogger
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
from biocomp.design import DesignManager, DesignConfig
from biocomp.design_targets import SVGTarget, DataTarget
from biocomp.network import Network
from biocomp.recipe import CoTransfection, TranscriptionUnit, Slot

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

T = TypeVar("T")
MaybeDeferred = DeferredNode[T] | T


class BaseOptimizationProgram(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_dir: Annotated[str, Arg(help="Base directory to save outputs")] = (
        config.paths.training_output
    )
    experiment_name: Annotated[str, Arg(help="Name of the experiment")] = "default_xp"
    run_name_suffix: str = ""
    metadata: dict[str, Any] = {}

    loggers: Annotated[list[Union[MaybeDeferred[Logger], Logger]], Arg(help="Loggers to use")] = (
        Field(default_factory=lambda: [])
    )

    async_logging: bool = True
    async_store_location: Optional[Path] = Field(
        default_factory=lambda: Path("step_history_data"),
        description="Location to store async logger data.",
    )
    keep_history_on_disk: bool = Field(
        default=False, description="Whether to keep step history files on disk for replay mode."
    )
    save_all_steps: bool = Field(
        default=False, description="Whether to save step history for every step."
    )
    use_history_db: bool = Field(
        default=True,
        description="Use per-run SQLite DB for step history (enables full-fidelity replay).",
    )
    write_policy: Optional[Any] = Field(
        default=None,
        description="WritePolicy for step data persistence (None = default policy).",
    )
    n_workers: int = 8

    _lib: Optional[Any] = None
    _history_db: Optional[Any] = None
    _yamldump: str = ""
    _modeldump: dict = {}
    _save_dir: Path = Path(".")
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
    async def execute_optimization(self, dispatch) -> Any:
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
        logger.info(f"Experiment: {self.experiment_name} | Run: {self._run_name}")
        logger.info(f"Output directory: {self._save_dir}")

        log_file = self._save_dir / "run.log"
        setup_logging(log_file=log_file, force=True)
        logger.info(f"Logging to file: {log_file}")

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
        return {"save_dir": self._save_dir}

    def _create_history_db(self) -> Any:
        """Create RunHistoryDB and save initial run info + artifacts."""
        if not self.use_history_db:
            return None

        from biocomptools.history_db import RunHistoryDB

        db = RunHistoryDB(self._save_dir / "run_history.db")
        metadata = self._metadata if hasattr(self, "_metadata") else {}

        model = getattr(self, "_model", None)
        dmanager = getattr(self, "_dmanager", None)
        dconfig = getattr(self, "design_conf", None)
        run_type = "design" if dmanager is not None else "training"

        db.save_run_info(
            run_type=run_type,
            config=metadata,
            commit_hashes={
                pkg: metadata.get(f"{pkg}_hash", "unknown")
                for pkg in ("dracon", "biocomp", "biocomptools")
            },
            host=metadata.get("host", "unknown"),
            model_signature=getattr(model, "signature", None),
        )

        # Save large objects as separate artifacts (not in RunInfo row)
        if model is not None:
            db.save_artifact("model", model)
        if dmanager is not None:
            db.save_artifact("dmanager", dmanager)
        if dconfig is not None:
            db.save_artifact("dconfig", dconfig)

        logger.info(f"Created history DB: {db.path}")
        self._history_db = db
        return db

    def gen_metadata(self):
        import os
        import socket

        starttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hashes = get_package_git_hashes(["dracon", "biocomp", "biocomptools"])

        self._metadata = {
            f"{self.__class__.__name__.lower()}_id": self.unique_id,
            "start_time": starttime,
            "host": f"{os.environ.get('USER')}@{socket.gethostname()}",
            "biocomp_hash": hashes.get("biocomp", "unknown"),
            "biocomptools_hash": hashes.get("biocomptools", "unknown"),
            "dracon_hash": hashes.get("dracon", "unknown"),
        }

        self._metadata.update(self.metadata)

    async def run(self):
        from biocomptools.logger_dispatch import LoggerDispatcher

        output_dir = self._save_dir / self.get_output_subdir()
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / f"{self.__class__.__name__.lower()}_dump.yaml", "w") as f:
            f.write(self._yamldump)

        self.enrich_metadata()

        logger.debug(
            f"Initializing {len(self.loggers)} loggers of types {[type(lg) for lg in self.loggers]}"
        )

        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()

        logger_metadata = [
            m for m in (lg.metadata for lg in self.loggers if isinstance(lg, Logger)) if m
        ]
        if logger_metadata:
            self._metadata["loggers"] = make_json_ready(logger_metadata)

        log_file = output_dir / "output.log.txt"
        log_file.parent.mkdir(exist_ok=True, parents=True)

        history_db = self._create_history_db()

        dispatch = LoggerDispatcher(
            self.loggers,
            training_program=self,
            async_logging=self.async_logging,
            base_dir=self._save_dir,
            n_workers=self.n_workers,
            history_db=history_db,
            write_policy=self.write_policy,
        )

        result = await self.execute_optimization(dispatch)

        dispatch.shutdown(result)

        self._metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.save_outputs(*result if isinstance(result, tuple) else [result])

        dispatch.finalize(self.loggers)

    def save_metadata(self, save_dir: Path):
        with open(save_dir / "metadata.json", "w") as f:
            import json

            json.dump(make_json_ready(self._metadata), f, indent=2)

    def save_loss_plot(self, all_losses, save_dir: Path):
        fig, ax = plot_loss(all_losses)
        assert self._metadata, "Metadata not set"
        assert self._run_name, "Run name not set"

        fig = print_matadata(fig, ax, self._metadata, run_name=self._run_name)
        fig.savefig(save_dir / "summary_loss_plot.pdf")
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
            DesignSummaryLogger,
            # Configuration types
            TrainingConfig,
            ComputeConfig,
            DataConfig,
            DesignConfig,
            DesignManager,
            SVGTarget,
            DataTarget,
            Network,
            # Utility types
            PartialFunction,
        ]
        + PLOT_TYPES
    )
)


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
        "BIOCOMP_ROOT": Path(config.paths.root).expanduser().resolve(),
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
