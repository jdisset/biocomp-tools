## {{{                          --     imports     --
from biocomptools.logging_config import get_logger, setup_logging, print_logger_hierarchy
import biocomp.utils as ut
from biocomptools.modelmodel import BiocompModel, get_shared_params
from typing import TypeVar
import dracon as dr
from biocomp import train
from pathlib import Path
from dracon.deferred import DeferredNode
import numpy as np
from typing import List, Optional, Tuple, Annotated
from pydantic import Field, BaseModel, ConfigDict
from biocomptools.toollib.common import config, get_package_git_hashes
from dracon.commandline import make_program, Arg
import sys

from biocomptools.toollib.datasources import DataSource, DBSource
from biocomp.train import TrainingConfig
from biocomp.library import PartsLibrary


from biocomptools.trainutils import (
    Logger,
    PlotLogger,
    EnhancedConsoleLogger,
    plot_loss,
    make_unique_dir,
    add_metadata,
    get_best_smoothed_loss_id,
)

from sqlmodel import Session
from datetime import datetime
from biocomp.utils import PartialFunction

from biocomp.utils import (
    load_lib,
    save,
)

from biocomptools.toollib.networkselector import (
    build_data_manager,
    NetworkSelector,
    Regex,
    NetworkDataId,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    NetworkFilter,
    NetworkSet,
    UorfFilter,
)

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG

from biocomptools.plot import DEFAULT_TYPES as PLOT_TYPES
from biocomptools.plot import NetworkPrediction

import biocomptools.toollib.models as md

setup_logging(force=False)
logger = get_logger(__name__)


DEFAULT_TYPES = [
    Regex,
    NetworkSelector,
    NetworkSet,
    NetworkDataId,
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

    # Then update db_session to use this property
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
        self._lib = load_lib()
        # self._engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)

        self.base_dir = str(Path(self.base_dir).expanduser().resolve())

        with self.db_session as session:
            self.training_set.run_selectors(session)
            self.validation_set.run_selectors(session)
        self.gen_metadata()

        self._save_dir = make_unique_dir(
            Path(self.base_dir) / self.experiment_name,
            # prefix='__running__',
        )

        self._run_name = self._save_dir.name

        # construct loggers
        new_loggers = []
        self._yamldump = dr.dump(self)
        self._modeldump = self.model_dump()

        for logger in self.loggers:
            if isinstance(logger, DeferredNode):
                logger = logger.construct(context={'training_program': self})
            new_loggers.append(logger)
        self.loggers = new_loggers

    def gen_metadata(self):
        import os

        starttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        def get_hostmachine():
            import socket

            return socket.gethostname()

        hashes = get_package_git_hashes(['dracon', 'biocomp', 'biocomptools'])

        self._metadata = {
            'start time': starttime,
            'host': f"{os.environ.get('USER')}@{get_hostmachine()}",
            'biocomp hash': hashes.get('biocomp', 'unknown'),
            'biocomptools hash': hashes.get('biocomptools', 'unknown'),
            'dracon hash': hashes.get('dracon', 'unknown'),
        }

    def _build_dman(self):
        # Build data manager from data_conf
        self._training_dman = build_data_manager(
            lib=self.parts_library,
            db_session=self.db_session,
            path_prefix=self.path_prefix,
            data_conf=self.data_conf,
            dataset=self.training_set,
        )

    def run(self):
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
                logger_obj = logger_obj.construct(context={'training_program': self})

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
        params, loss_history, step_history = train.start(
            self._training_dman,
            self.training_conf,
            self.compute_conf,
            loggers=logger_callbacks,
        )

        self._metadata['end time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_outputs(params, loss_history, self._training_dir)

        for logger_obj in self.loggers:
            logger_obj.finalize()

    def get_best_model(self, all_models, all_losses):
        import pickle

        best_model_id, _ = get_best_smoothed_loss_id(all_losses)
        logger.debug(f"Best model is replicate number {best_model_id}")
        params = ut.tree_get(all_models, best_model_id)
        if params is None:
            return None

        copied_params = pickle.loads(pickle.dumps(params))

        best_params = get_shared_params(copied_params)

        if best_params is None:
            return None

        model = BiocompModel(
            compute_config=self.compute_conf,
            rescaler=self.data_conf.rescaler,
            shared_params=ut.tree_to_np(best_params),
        )

        return model

    def save_best(self, all_models, all_losses, save_dir, name='best_model'):
        model = self.get_best_model(all_models, all_losses)
        if model is None:
            logger.warning("!!!!!! No best model found !!!!!")
            return
        fname = save_dir / f'{name}.pickle'
        model.save(fname)

        model2 = BiocompModel.load(fname)
        logger.debug(f"Saved best model to {fname}")

        # assert model.shared_params == model2.shared_params

    def save_outputs(self, params, loss_history, save_dir: Path):
        save(params, save_dir / 'final_all_models.pickle')
        self.save_best(params, loss_history, save_dir)

        # Save loss history
        np.save(save_dir / 'loss_history.npy', loss_history)

        fig, ax = plot_loss(loss_history)
        assert self._metadata, "Metadata not set"
        assert self._run_name, "Run name not set"

        fig = add_metadata(fig, ax, self._metadata, run_name=self._run_name)

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
    )
    assert isinstance(trainprog, TrainingProgram), f"{trainprog=}"

    trainprog.run()


if __name__ == '__main__':
    main()
