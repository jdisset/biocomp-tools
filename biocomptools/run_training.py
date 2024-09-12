## {{{                          --     imports     --

import dracon as dr
import logging
import json
from biocomptools.toollib.common import config
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
from pydantic import BaseModel, Field, BeforeValidator, model_validator
from typing import List, Optional, Union
from sqlmodel import select, Session, col
from tqdm import tqdm

import biocomptools.toollib.models as md

logging.getLogger('dracon.commandline').setLevel(logging.DEBUG)
##────────────────────────────────────────────────────────────────────────────}}}


class NetworkAndData(BaseModel):
    """
    A network name and datafile path pair.
    The point is to keep it simple and explicit for repeatable training.
    """

    network_name: str
    file_path: str

    def __hash__(self):
        return hash((self.network_name, self.file_path))

    def get_network_and_data(self, session) -> Tuple[md.Network, md.DataFile]:
        network = session.exec(
            select(md.Network).where(md.Network.name == self.network_name)
        ).first()
        datafile = session.exec(
            select(md.DataFile).where(md.DataFile.file == self.file_path)
        ).first()

        assert network, f"No network found for {self.network_name}"
        assert datafile, f"No datafile found for {self.file_path}"

        return network, datafile


class NetworkSelector(BaseModel):
    """
    Manually writing a NetworkAndData can be very annoying and verbose and error-prone.
    This class allows to batch select networks based on their names, recipes, experiments, etc.
    """

    experiment_name: Optional[str]
    recipe_name: Optional[str] = None
    calibration_name: Optional[str] = None
    output_name: Optional[str] = None

    def get_networks_and_data(self, session) -> List[NetworkAndData]:
        query = select(md.Network).join(md.Recipe).join(md.Experiment)

        if self.experiment_name:
            query = query.where(col(md.Experiment.name).regexp_match(self.experiment_name))
        if self.recipe_name:
            query = query.where(col(md.Recipe.name).regexp_match(self.recipe_name))

        networks = session.exec(query).all()

        if self.output_name is not None:
            networks = [
                network
                for network in networks
                if network.network_info['dependent_outputs'][0].upper() == self.output_name.upper()
            ]

        network_and_data = []
        for network in networks:
            if self.calibration_name:
                datafile_query = (
                    select(md.DataFile)
                    .where(
                        md.DataFile.recipe_name == network.recipe_name,
                        col(md.DataFile.calibration_name).regexp_match(self.calibration_name),
                    )
                    .order_by(col(md.DataFile.priority).desc())
                )
                datafile = session.exec(datafile_query).first()
            else:
                datafile = network.recipe.get_best_datafile()

            assert datafile, f"No datafile found for {network.recipe_name}"
            network_and_data.append(
                NetworkAndData(network_name=network.name, file_path=datafile.file)
            )

        return network_and_data


class NetworkSet(BaseModel):
    content: List[NetworkAndData | NetworkSelector] = []

    def run_selectors(self, session):
        # we want to run this as a post-init so that we store the content in a more
        # repeatable / serializable way than selectors when dumping the whole config
        new_content = []
        for n in self.content:
            if isinstance(n, NetworkSelector):
                new_content.extend(n.get_networks_and_data(session))
            else:
                assert isinstance(n, NetworkAndData)
                new_content.append(n)
        self.content = new_content

    def get_networks_and_data(self, session) -> List[Tuple[md.Network, md.DataFile]]:
        res = []
        for n in self.content:
            assert isinstance(n, NetworkAndData)
            res.append(n.get_network_and_data(session))
        return res


class NetworkSetUnion(NetworkSet):
    """
    A union of multiple NetworkSets. (and itself)
    """

    sets: List[NetworkSet] = []

    allow_duplicates: bool = False

    def run_selectors(self, session):
        for s in self.sets:
            s.run_selectors(session)

        new_content = []
        for s in self.sets:
            new_content.extend(s.content)
        new_content.extend(self.content)
        if not self.allow_duplicates:
            new_content = list(set(new_content))
        self.content = new_content


class NetworkSetIntersection(NetworkSet):
    """
    An intersection of multiple NetworkSets. (and itself)
    """

    sets: List[NetworkSet] = []

    def run_selectors(self, session):
        for s in self.sets:
            s.run_selectors(session)

        new_content = self.sets[0].content
        for s in self.sets[1:]:
            new_content = list(set(new_content) & set(s.content))
        new_content = list(set(new_content) & set(self.content))
        self.content = new_content


class NetworkSetDifference(NetworkSet):
    """
    A difference of two NetworkSets.
    """

    set1: NetworkSet
    set2: NetworkSet

    # make sure content is empty
    @model_validator(mode='before')
    def check_content(self):
        assert not self.content, "content of a SetDifference should be empty"

    def run_selectors(self, session):
        self.set1.run_selectors(session)
        self.set2.run_selectors(session)

        new_content = list(set(self.set1.content) - set(self.set2.content))
        self.content = new_content


##


DEFAULT_LOGGERS = []


class TrainingProgram(BaseModel):
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

    outputdir: Annotated[str, Arg(help='Output directory to save model')] = './model'

    loggers: Annotated[
        Optional[List[Tuple[int, EncodedPartialFunction]]], Arg(help='Loggers (period, function)')
    ] = DEFAULT_LOGGERS

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)
        with Session(engine) as session:
            self.training_set.run_selectors(session)
            self.validation_set.run_selectors(session)

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
