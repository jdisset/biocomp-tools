## {{{                          --     imports     --
from pydantic import model_validator
from typing import Any, Optional, List

import numpy as np

from biocomp.utils import (
    ArbitraryModel,
    load_lib,
)
from pathlib import Path
import biocomp as bc
from biocomp.plotutils import PlotData
import biocomp.plotutils as pu

from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector, NetworkDataId

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import maybetqdm, make_pretty_input_names
import biocomptools.toollib.models as md
from biocomptools.logging_config import get_logger

from sqlmodel import Session

config = cm.config


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


logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     datasource     --


class DataSource(ArbitraryModel):
    metadata: dict = {}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     DB   --

config = cm.config


class DBSource(DataSource, NetworkSet):
    input_order: Optional[List[int]] = None

    @model_validator(mode='before')
    def validate_content(cls, values):
        content = values.get('content')
        if isinstance(content, (NetworkSelector, NetworkDataId)):
            values['content'] = [content]
        elif isinstance(content, NetworkSet):
            values['content'] = content.content
        return values

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

    def data_from_network(self, network: md.Network, datafile: md.DataFile) -> Optional[PlotData]:
        try:
            network.build(self._lib)
        except Exception as e:
            logger.error(f"Error building network {network.name}: {e}")
            return None

        actual_network = network._network
        assert isinstance(actual_network, bc.network.Network)

        datafile_path = Path(self.path_prefix / datafile.file).expanduser().resolve()
        metadata = self.metadata
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
            logger.debug(f"DBSource: getting XY data for network {network.name}")
            try:
                X, Y = bc.recipe.get_network_XY(actual_network, datafile_path)
            except Exception as e:
                logger.error(f"Error getting XY data for network {network.name}: {e}")
                return None, None
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)
            return X, Y

        try:
            pdata = pu.extract_lazy_plot_data_from_network(
                actual_network,
                get_XY,
                input_order=self.input_order,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error extracting data from network {network.name}: {e}")
            return None

        pdata.metadata['pretty_inputs'] = make_pretty_input_names(
            metadata['network_info']['cotx'],
            pdata.input_names,
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
                try:
                    d = self.data_from_network(n, f)
                except Exception as e:
                    logger.error(f"Error loading data for {n.name}: {e}")
                    d = None

                if d is not None:
                    res.append(d)

            return res


##────────────────────────────────────────────────────────────────────────────}}}
