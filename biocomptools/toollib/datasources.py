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

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import maybetqdm
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

    def data_from_network(self, network: md.Network, datafile: md.DataFile) -> PlotData:
        network.build(self._lib)
        actual_network = network._network
        assert isinstance(actual_network, bc.network.Network)

        datafile_path = Path(self.path_prefix / datafile.file).expanduser().resolve()
        metadata = resolved(self.metadata)
        metadata['network'] = network
        metadata['built_network'] = actual_network
        metadata['network_info'] = network.network_info
        metadata['source_type'] = 'DB'
        metadata = {**metadata, **network.model_dump()}
        metadata['datafile'] = datafile
        metadata['file_stem'] = datafile_path.stem

        if not datafile_path.exists():
            raise ValueError(f'Data file {datafile_path} does not exist for network {network.name}')

        def get_XY(_):
            X, Y = bc.recipe.get_network_XY(actual_network, datafile_path)
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)
            return X, Y

        pdata = pu.extract_lazy_plot_data_from_network(
            actual_network,
            get_XY,
            input_order=self.input_order,
            metadata=metadata,
        )

        return pdata

    def get_data(self) -> List[PlotData]:
        with self.db_session as session:
            data = self.get_networks_and_data(session)
            if not data:
                msg = f'No data found for {self.content}'
                raise ValueError(msg)
            networks, datafiles = zip(*data)

            res = []

            for n, f in maybetqdm(zip(networks, datafiles), desc='Loading recipes'):
                res.append(self.data_from_network(n, f))

            return res


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                     --     NetworkPrediction     --


class NetworkPrediction(DBSource):
    predict_at: np.ndarray
    network_model: SingleNetworkModel

    ground_truth: Optional[np.ndarray] = None

    seed: int = 0

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

    def get_data(self) -> List[PlotData]:
        import time

        def get_XY(pdata):
            t0 = time.time()
            yhat = self.network_model.predict(self.predict_at, self.seed)
            t1 = time.time()
            pdata.metadata['prediction_time'] = t1 - t0
            if self.ground_truth is not None:
                assert len(yhat) == len(self.ground_truth)
                mse = np.mean((yhat - self.ground_truth) ** 2)
                pdata.metadata['mse'] = mse
            return self.predict_at, yhat

        metadata = self.metadata
        metadata['source_type'] = 'prediction'
        metadata['seed'] = self.seed
        metadata['model'] = self.network_model.model
        metadata['network'] = self.network_model.network
        metadata['network_info'] = self.network_model.network.network_info
        metadata['built_network'] = self.network_model.network.network
        metadata['n_predictions'] = len(self.predict_at)

        plot_data = pu.extract_lazy_plot_data_from_network(
            self.network_model.network.network,
            get_XY,
            input_order=self.input_order,
            metadata=metadata,
        )

        return [plot_data]


##────────────────────────────────────────────────────────────────────────────}}}
