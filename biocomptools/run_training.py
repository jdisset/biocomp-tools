## {{{                          --     imports     --

import dracon as dr
import logging
from pathlib import Path
import numpy as np
from typing import List, Optional, Tuple, Annotated
from pydantic import Field, BaseModel, ConfigDict
from biocomptools.toollib.common import config, get_git_hash, get_package_git_hashes
from biocomptools.toollib.networkselector import NetworkSet, build_data_manager
from dracon.commandline import Program, make_program, Arg
import biocomp as bc
import sys
from biocomp.train import TrainingConfig
from biocomp.library import PartsLibrary
from biocomptools.trainutils import Logger, plot_loss, generate_unique_name, add_metadata
from sqlmodel import Session
from datetime import datetime

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
    outputdir: Annotated[
        str, Arg(help='Base directory to save model (will be saved in outputdir/run_name')
    ] = './training_output'
    loggers: Annotated[List[Logger], Arg(help='Loggers to use')] = DEFAULT_LOGGERS

    run_name: Annotated[
        Optional[str], Arg(help='Name of the run (automatically generated if None)')
    ] = None

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
        self.gen_metadata()
        if not self.run_name:
            self.run_name = generate_unique_name(
                self.outputdir,
            )

        assert self.run_name, "Run name not set"
        self._save_dir = Path(self.outputdir).expanduser().resolve() / f"__running__{self.run_name}"

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
        self._training_dir = self._save_dir / 'training'
        self._training_dir.mkdir(exist_ok=True, parents=True)
        with open(self._training_dir / 'training_program_dump.yaml', 'w') as f:
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
        log_file = self._training_dir / 'output.log.txt'
        log_file.parent.mkdir(exist_ok=True, parents=True)
        self.setup_logging(log_file)

        # Collect logger callbacks
        logger_callbacks = []
        for logger in self.loggers:
            callbacks = logger.get_callbacks(self)
            logger_callbacks.extend(callbacks)

        print(f"Starting training run {self.run_name}")
        print(f"{self._metadata}")

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

        self._metadata['end time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Save the model and other outputs
        self.save_outputs(params, loss_history, self._training_dir)

        # rename the directory to indicate that the run is complete
        self._save_dir.rename(self._save_dir.with_name(self.run_name))

    def save_outputs(self, params, loss_history, save_dir: Path):
        save(params, save_dir / 'all_models.pickle')

        # Save loss history
        np.save(save_dir / 'loss_history.npy', loss_history)

        fig, ax = plot_loss(loss_history)
        assert self._metadata, "Metadata not set"
        assert self.run_name, "Run name not set"

        fig = add_metadata(fig, ax, self._metadata, run_name=self.run_name)

        fig.savefig(save_dir / 'summary_plot.pdf')
        # save metadata as json
        with open(save_dir / 'metadata.json', 'w') as f:
            import json

            json.dump(self._metadata, f)

        print(f"Saved summary plot to {save_dir / 'summary_plot.pdf'}")


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
