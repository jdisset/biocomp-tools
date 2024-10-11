## {{{                          --     imports     --
import glob
from pydantic.functional_validators import BeforeValidator
from typing import Any, Dict, Union, Annotated, Optional, List, Callable
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector

from functools import partial

import pandas as pd
import numpy as np

from biocomp.recipe import get_network_XY
from biocomp.utils import (
    ArbitraryModel,
    load_lib,
    save,
    EncodedPartialFunction,
    PartialFunction,
    PartialFunctionResult,
)
from pathlib import Path
import biocomp as bc
import biocomp.utils as ut
from biocomp.plotutils import PlotData
import biocomp.plotutils as pu

from biocomp.utils import PartialFunction, ArbitraryModel
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.resolvable import resolved
from biocomptools.modelmodel import SingleNetworkModel

import logging
from sqlmodel import tuple_
from sqlmodel import Field, Session, SQLModel, create_engine, select, text
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.elements import TextClause


def truncated_path(path: str | Path, max_len=50) -> str:
    if isinstance(path, Path):
        path = path.as_posix()
    if len(path) > max_len:
        return '...' + path[-max_len:]
    return path


def to_str(data: Any) -> Any:
    if not isinstance(data, str) and data is not None:
        return str(data)
    return data

config = cm.config

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     datasource     --


class DataSource(ArbitraryModel):
    metadata: dict = {}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     DB   --

config = cm.config


def to_text_clause(data: Any) -> TextClause:
    if isinstance(data, str):
        return text(data)
    return data


class DBSource(DataSource, NetworkSet):
    input_order: Optional[List[int]] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._lib = load_lib()
        self._engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)
        with Session(self._engine) as session:
            self.run_selectors(session)

    @property
    def db_session(self):
        return Session(self._engine)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    def data_from_network(self, network: md.Network, data_file: str | Path) -> PlotData:
        network.build(self._lib)
        actual_network = network._network
        assert isinstance(actual_network, bc.network.Network)

        data_file = Path(data_file).expanduser().resolve()
        metadata = resolved(self.metadata)
        metadata['built_network'] = actual_network
        metadata['network'] = network
        metadata['network_info'] = network.network_info
        metadata['source_type'] = 'DB'
        metadata = {**metadata, **network.model_dump()}
        metadata['file_stem'] = data_file.stem

        if not data_file.exists():
            raise ValueError(f'Data file {data_file} does not exist for network {network.name}')

        X, Y = bc.recipe.get_network_XY(actual_network, data_file)

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        pdata = pu.extract_plot_data_from_network(
            actual_network,
            X,
            Y,
            input_order=self.input_order,
            metadata=metadata,
        )

        return pdata

    def get_data(self) -> List[PlotData]:
        data = self.get_networks_and_data(self.db_session)
        assert len(data), 'No data returned from query'
        networks, datafiles = zip(*data)
        return [
            self.data_from_network(n, self.path_prefix / f.file)
            for n, f in zip(networks, datafiles)
        ]


##────────────────────────────────────────────────────────────────────────────}}}


class NetworkPrediction(DBSource):

    predict_at: np.ndarray
    network_model: SingleNetworkModel

    seed: int = 0

    def get_data(self) -> List[PlotData]:
        ...

        

    






## {{{                            --     XP     --


class XPDataSource(DataSource):
    # TODO

    xp_path: str
    recipe_names: Optional[List[str]] = None
    source_type: str = 'xp'

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}XPDataSource({truncated_path(self.xp_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
