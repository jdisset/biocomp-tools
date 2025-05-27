## {{{                          --     imports     --
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.modelmodel import BiocompModel, get_shared_params
from biocomptools.toollib.datasources import DataSource, DBSource
from biocomptools.toollib.common import config
from biocomptools.toollib.hashutils import get_package_git_hashes
from biocomptools.toollib.loggers.logger import Logger, FunctionLogger
from biocomptools.toollib.loggers.plotlogger import PlotLogger
from biocomptools.toollib.loggers.consolelogger import EnhancedConsoleLogger, ConsoleLogger
from biocomptools.trainutils import (
    plot_loss,
    make_unique_dir,
    print_matadata,
    get_best_smoothed_loss_replicate_id,
    make_json_ready,
)

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG
from biocomp.library import PartsLibrary
from biocomp.utils import PartialFunction, load_lib, save
from biocomp.train import TrainingConfig

import dracon as dr
from dracon.deferred import DeferredNode
from dracon.commandline import make_program, Arg

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Annotated, TypeVar, Any
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

    validation_set: Annotated[NetworkSet, Arg(help='Networks in validation set')] = Field(
        default_factory=NetworkSet
    )

    base_dir: Annotated[
        str,
        Arg(
            help='Base directory to save model (will be saved in base_dir/experiment_name/run_name)'
        ),
    ] = './training_output'

    loggers: Annotated[List[MaybeDeferred[Logger]], Arg(help='Loggers to use')] = Field(
        default_factory=lambda: []
    )

    experiment_name: Annotated[str, Arg(help='Name of the experiment')] = 'default_xp'
    metadata: dict[str, Any] = {}

    run_name_suffix: str = ''

    # Private
    _lib: Optional[PartsLibrary] = None
    _yamldump: str = ''
    _modeldump: dict = {}
    _save_dir: Path = Path('.')
    _run_name: Optional[str] = None

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
            self.validation_set.run_selectors(session)
            session.expunge_all()
            session.close()

        self.gen_metadata()

        self._save_dir = make_unique_dir(
            Path(self.base_dir) / self.experiment_name, suffix=self.run_name_suffix
        )

        self._run_name = self._save_dir.name

        # construct loggers
        new_loggers = []
        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()

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

    def gen_metadata(self):
        import os

        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        def get_hostmachine():
            import socket

            return socket.gethostname()

        hashes = get_package_git_hashes(['dracon', 'biocomp', 'biocomptools'])

        self._metadata = {
            'start_time': starttime,
            'host': f"{os.environ.get('USER')}@{get_hostmachine()}",
            'biocomp_hash': hashes.get('biocomp', 'unknown'),
            'biocomptools_hash': hashes.get('biocomptools', 'unknown'),
            'dracon_hash': hashes.get('dracon', 'unknown'),
            'yaml_dump': self._yamldump,
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

    def run(self):
        from biocomp.train import start

        # Prepare output directory
        self._training_dir = self._save_dir / 'training'
        self._training_dir.mkdir(exist_ok=True, parents=True)
        with open(self._training_dir / 'training_program_dump.yaml', 'w') as f:
            f.write(self._yamldump)

        self._build_dman()

        # Initialize loggers
        logger.debug(
            f"Initializing {len(self.loggers)} loggers of types {[type(l) for l in self.loggers]}"
        )
        for logger_obj in self.loggers:
            if isinstance(logger_obj, DeferredNode):
                logger_obj = logger_obj.construct()

            logger.debug(f"Initializing a {type(logger_obj)}")
            logger_obj.initialize(self)

        # Set up logging to file
        log_file = self._training_dir / 'output.log.txt'
        log_file.parent.mkdir(exist_ok=True, parents=True)

        # Collect logger callbacks
        logger_callbacks = []
        for logger_obj in self.loggers:
            callbacks = logger_obj.get_callbacks(self)
            logger_callbacks.extend(callbacks)

        # Start training
        all_params, all_losses, step_history = start(
            self._training_dman,
            self.training_conf,
            self.compute_conf,
            loggers=logger_callbacks,
        )

        self._metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_outputs(all_params, all_losses, self._training_dir)

        for logger_obj in self.loggers:
            logger_obj.finalize()

    def get_best_model_func(self):
        from copy import deepcopy

        compute_conf = deepcopy(self.compute_conf)
        data_conf = deepcopy(self.data_conf)

        metadata = {}
        if self._metadata:
            metadata = self._metadata.copy()

        dman = self._training_dman
        metadata['run_name'] = self._run_name
        metadata['experiment_name'] = self.experiment_name
        metadata['training_set'] = make_json_ready(self.training_set.content)
        metadata['validation_set'] = make_json_ready(self.validation_set.content)
        metadata['training_conf'] = make_json_ready(self.training_conf)
        dataman_info = {}
        dataman_info["network_names"] = [n.name for n in dman.get_networks()]
        dataman_info["input_dimensions"] = [x.shape[1] for x in dman.get_X()]
        dataman_info["output_dimensions"] = [y.shape[1] for y in dman.get_Y()]
        dataman_info["data_config"] = (dman.data_cfg.model_dump(),)
        metadata['data_manager_info'] = make_json_ready(dataman_info)

        self._metadata.update(metadata)

        metadata = make_json_ready(metadata)

        def get_best_model(all_params, all_losses):
            from biocomp.jaxutils import tree_get, tree_to_np
            import pickle

            best_model_id, smoothed_loss, end_loss = get_best_smoothed_loss_replicate_id(all_losses)
            logger.debug(f"Best model is replicate number {best_model_id}")

            params = tree_get(all_params, best_model_id)
            if params is None:
                return None

            copied_params = pickle.loads(pickle.dumps(params))

            best_params = get_shared_params(copied_params)

            if best_params is None:
                return None

            local_metadata = metadata.copy()
            local_metadata['replicate_number'] = best_model_id
            local_metadata['end_loss'] = end_loss

            model = BiocompModel(
                compute_config=compute_conf,
                rescaler=data_conf.rescaler,
                shared_params=tree_to_np(best_params),
                metadata=make_json_ready(local_metadata),
            )

            return model

        return get_best_model

    def save_best(self, all_params, all_losses: list[np.ndarray], save_dir, name=None):
        model = self.get_best_model_func()(all_params, all_losses)
        if model is None:
            logger.warning("!!!!!! No best model found !!!!!")
            return
        if name is None:
            name = f"{model.signature}.bestmodel"
        fname = save_dir / f'{name}.pickle'
        model.save(fname)
        logger.debug(f"Saved best model to {fname}")

    def save_outputs(self, all_params, all_losses: list, save_dir: Path):
        save(all_params, save_dir / 'final_all_models.pickle')
        self.save_best(all_params, all_losses, save_dir)

        # Save loss history
        np.save(save_dir / 'loss_history.npy', all_losses)

        fig, ax = plot_loss(all_losses)
        assert self._metadata, "Metadata not set"
        assert self._run_name, "Run name not set"

        fig = print_matadata(fig, ax, self._metadata, run_name=self._run_name)

        fig.savefig(save_dir / 'summary_loss_plot.pdf')
        # save metadata as json
        with open(save_dir / 'metadata.json', 'w') as f:
            import json

            json.dump(self._metadata, f)

        logger.debug(f"Saved summary plot to {save_dir / 'summary_loss_plot.pdf'}")


##────────────────────────────────────────────────────────────────────────────}}}


def main():
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
    trainprog.run()


if __name__ == '__main__':
    main()
