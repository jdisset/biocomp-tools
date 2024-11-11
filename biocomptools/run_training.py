## {{{                          --     imports     --

import dracon as dr
import logging
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import numpy as np
from numpy import ndarray as ndArray
from typing import List, Optional, Tuple, Annotated
from pydantic import Field, BaseModel, ConfigDict
from biocomptools.toollib.common import config
from biocomptools.toollib.networkselector import NetworkSet, build_data_manager
from dracon.commandline import Program, make_program, Arg
import biocomp as bc
from biocomp.utils import ArbitraryModel
import sys
from biocomp.train import TrainingConfig
from biocomp.library import PartsLibrary
from biocomptools.trainutils import Logger
from sqlmodel import select, Session, col

from biocomp.utils import (
    load_lib,
    save,
)

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG

import biocomptools.toollib.models as md


logging.getLogger('dracon.commandline').setLevel(logging.DEBUG)
##────────────────────────────────────────────────────────────────────────────}}}


## {{{                      --     TrainingProgram     --

DEFAULT_LOGGERS = []


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
    outputdir: Annotated[str, Arg(help='Output directory to save model')] = './training_output'
    loggers: Annotated[List[Logger], Arg(help='Loggers to use')] = DEFAULT_LOGGERS

    _lib: Optional[PartsLibrary] = None

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
        self._engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)
        with self.db_session as session:
            self.training_set.run_selectors(session)
            self.validation_set.run_selectors(session)

    @staticmethod
    def setup_logging(log_file):
        import logging

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def run(self):
        self._fulldump = dr.dump(self)
        # Prepare output directory
        save_dir = Path(self.outputdir)
        training_dir = save_dir / 'training'
        training_dir.mkdir(exist_ok=True, parents=True)
        with open(training_dir / 'training_program_dump.yaml', 'w') as f:
            f.write(self._fulldump)

        # Build data manager from data_conf
        self._training_dman = build_data_manager(
            lib=self.parts_library,
            db_session=self.db_session,
            path_prefix=self.path_prefix,
            data_conf=self.data_conf,
            dataset=self.training_set,
        )

        # Initialize loggers
        for logger in self.loggers:
            logger.initialize(self)

        # Set up logging to file
        log_file = training_dir / 'output.log.txt'
        log_file.parent.mkdir(exist_ok=True, parents=True)
        self.setup_logging(log_file)

        # Collect logger callbacks
        logger_callbacks = []
        for logger in self.loggers:
            callbacks = logger.get_callbacks(self)
            logger_callbacks.extend(callbacks)

        # Start training
        params, loss_history, step_history = bc.train.start(
            self._training_dman,
            self.training_conf,
            self.compute_conf,
            loggers=logger_callbacks,
        )

        # Finalize loggers
        for logger in self.loggers:
            logger.finalize()

        # Save the model and other outputs
        self.save_outputs(params, loss_history, save_dir)

    def save_outputs(self, params, loss_history, save_dir):
        save(params, save_dir / 'training' / 'all_models.pickle')

        # Save loss history
        np.save(save_dir / 'training' / 'loss_history.npy', loss_history)

        # Generate and save loss plots
        # plot_losses(loss_history, save_dir / 'training' / 'losses.png')

        # Save the training program configuration


##────────────────────────────────────────────────────────────────────────────}}}


def main():
    cliprog = make_program(
        TrainingProgram,
        name='biocomp-train',
        description='Start training biocomp models.',
    )
    trainprog, _ = cliprog.parse_args(
        sys.argv[1:],
        context={'NetworkSet': NetworkSet},
    )
    assert isinstance(trainprog, TrainingProgram), f"{trainprog=}"

    trainprog.run()


if __name__ == '__main__':
    main()
