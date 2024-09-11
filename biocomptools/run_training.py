## {{{                          --     imports     --

import dracon as dr
import logging
import json
from dracon.utils import with_indent
from dracon.interpolation import LazyDraconModel
import pandas as pd
import time
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import dracon
import calibry as cal
from pathlib import Path
from typing import List, Tuple, Union, Annotated, Dict, Any, Optional, Literal
from pydantic import BaseModel
import sys
from pathlib import Path
import json5
from calibry.pipeline import DiagnosticFigure
from calibry import (
    GatingTask,
    PolygonGate,
    LoadControls,
    LinearCompensation,
    ProteinMapping,
    MEFBeadsCalibration,
    MEFBeadsTransform,
    PandasExport,
    AbundanceCutoff,
    Pipeline,
)
import dracon as dr
import matplotlib
from biocomp.train import TrainingConfig
from biocomp.utils import (
    ArbitraryModel,
    EncodedPartialFunction,
    PartialFunction,
    PartialFunctionResult,
)
from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG, ValueRange
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from sqlmodel import select
from tqdm import tqdm

import biocomptools.toollib.models as md

logging.getLogger('dracon.commandline').setLevel(logging.DEBUG)
##────────────────────────────────────────────────────────────────────────────}}}


class NetworkSelector(BaseModel):
    experiment_name: Optional[str]
    recipe_name: Optional[str] = None
    calibration_name: Optional[str] = None
    build: Literal['all', 'shortest'] = 'all'


class TrainingSet(BaseModel):
    name: Optional[str] = None
    networks: List[NetworkSelector] = Field(default_factory=list)

    def get_networks_and_datafiles(self, session, lib, use_cache=None):
        self._networks: List[md.Network] = []
        self._datafiles: List[md.DataFile] = []

        for selector in tqdm(self.networks, desc="Processing network selectors"):
            query = select(md.Recipe).join(md.Experiment)

            if selector.experiment_name:
                query = query.where(md.Experiment.name.regexp_match(selector.experiment_name))  # type: ignore
            if selector.recipe_name:
                query = query.where(md.Recipe.name.regexp_match(selector.recipe_name))  # type: ignore

            recipes = session.exec(query).all()

            for recipe in recipes:
                built_networks = recipe.build_networks(
                    lib=lib,
                    inverse=selector.build,
                    use_cache=use_cache,
                    add_to_self=True,
                )

                for network in built_networks:
                    if selector.calibration_name:
                        datafile_query = (
                            select(md.DataFile)
                            .where(
                                md.DataFile.recipe_name == recipe.name,
                                md.DataFile.calibration_name.regexp_match(  # type: ignore
                                    selector.calibration_name
                                ),
                            )
                            .order_by(md.DataFile.priority.desc())  # type: ignore
                        )
                        datafile = session.exec(datafile_query).first()
                    else:
                        datafile = recipe.get_best_datafile()

                    assert datafile, f"No datafile found for {recipe.name}"
                    self._networks.append(network)
                    self._datafiles.append(datafile)

        assert len(self._networks) == len(
            self._datafiles
        ), f"{len(self._networks)=} != {len(self._datafiles)=}"

        return self._networks, self._datafiles


DEFAULT_LOGGERS = []


class TrainingProgram(BaseModel):
    training_conf: Annotated[TrainingConfig, Arg(help='Training config')] = Field(
        default_factory=TrainingConfig
    )
    compute_conf: Annotated[ComputeConfig, Arg(help='Compute config')] = DEFAULT_COMPUTE_CONFIG
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = DEFAULT_DATA_CONFIG

    training_set: Annotated[TrainingSet, Arg(help='Training set to use')] = Field(
        default_factory=TrainingSet
    )
    outputdir: Annotated[str, Arg(help='Output directory to save model')] = './model'
    # tuples (period:int, logger: Callable)
    loggers: Optional[List[Tuple[int, EncodedPartialFunction]]] = DEFAULT_LOGGERS

    def run(self):
        self_dump = self.model_dump_json(indent=2)
        print(self_dump)


def main():
    cliprog = make_program(
        TrainingProgram,
        name='biocomp-train',
        description='Start training biocomp models.',
    )
    trainprog, _ = cliprog.parse_args(sys.argv[1:])
    assert isinstance(trainprog, TrainingProgram), f"{trainprog=}"

    trainprog.run()


if __name__ == '__main__':
    main()
