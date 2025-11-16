## {{{                          --     imports     --
from pydantic import model_validator
from typing import Any, Optional, List, Union
from sqlalchemy.orm.session import make_transient
import numpy as np
from biocomp.utils import load_lib

from pathlib import Path
import biocomp as bc
from biocomp.plotutils import PlotData
import biocomp.plotutils as pu

from biocomptools.toollib.networkselector import NetworkSet

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import maybetqdm, make_pretty_input_names
import biocomptools.toollib.models as md
from biocomptools.logging_config import get_logger
from pydantic import BaseModel


from sqlalchemy.inspection import inspect


def detach_object_tree(obj):
    """Recursively detach an object and all its loaded relationships"""
    make_transient(obj)

    # Get the mapper for this object
    mapper = inspect(obj.__class__)

    # Process all relationship attributes
    for relationship_prop in mapper.relationships:
        # Skip unloaded relationships to avoid triggering lazy loads
        if relationship_prop.key not in inspect(obj).unloaded:
            related_obj = getattr(obj, relationship_prop.key)

            # Handle collections (lists, etc.)
            if relationship_prop.uselist:
                if related_obj is not None:
                    for item in related_obj:
                        detach_object_tree(item)
            # Handle scalar relationships
            elif related_obj is not None:
                detach_object_tree(related_obj)

    return obj


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


class DataSource(BaseModel):
    metadata: dict = {}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     DB   --

config = cm.config


class DBSource(DataSource, NetworkSet):
    input_order: Optional[Union[List[int], str]] = None

    @model_validator(mode='before')
    def show_data(cls, values):
        logger.debug(f"DBSource being constructed with {values=}")
        return values

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._lib = load_lib()
        with self.db_session as session:
            self.run_selectors(session)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    def data_from_network(self, network: md.Network, datafile: md.DataFile) -> Optional[PlotData]:
        try:
            network.build(self._lib)
        except Exception as e:
            logger.error(f"Error building network {network.name}: {e}")
            logger.exception(e)
            raise

        actual_network = network.network

        assert isinstance(actual_network, bc.network.Network)

        datafile_path = Path(self.path_prefix / datafile.file).expanduser().resolve()
        metadata = self.metadata.copy()

        metadata['network'] = network.model_dump()
        metadata['network_name'] = network.name
        metadata['network_info'] = network.network_info
        metadata['built_network'] = actual_network
        metadata['datasource_type'] = 'database'

        metadata['datafile'] = datafile.model_dump()
        metadata['file_stem'] = datafile_path.stem

        if not datafile_path.exists():
            raise ValueError(f'Data file {datafile_path} does not exist for network {network.name}')

        def get_XY(_):
            logger.debug(f"DBSource: getting XY data for network {network.name}")
            try:
                from biocomp.datautils import get_network_XY

                X, Y = get_network_XY(actual_network, datafile_path)
                if X.shape[1] != actual_network.nb_inputs:
                    raise ValueError(
                        f"Input size mismatch for network {network.name}: expected {actual_network.nb_inputs} inputs, got {X.shape[1]} inputs"
                    )
            except Exception as e:
                logger.error(f"Error getting XY data for network {network.name}: {e}")
                logger.exception(e)
                return None, None
            assert isinstance(X, np.ndarray)
            assert isinstance(Y, np.ndarray)
            print(
                f"DBSource: got XY data for network {network.name} with shapes {X.shape}, {Y.shape}"
            )
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
            logger.exception(e)
            return None

        pdata.metadata['pretty_inputs'] = make_pretty_input_names(
            metadata['network_info']['cotx'],
            pdata.input_names,
        )
        pdata.metadata['ordered_input_names'] = '-'.join(pdata.input_names)

        return pdata

    @property
    def networks_and_datafiles(self):
        data = self.get_networks_and_data()
        if not data:
            msg = f'No data found for {self.content}'
            raise ValueError(msg)
        return data

    def get_data(self) -> List[PlotData]:
        res = []
        for n, f in maybetqdm(self.networks_and_datafiles, desc='Loading data'):
            try:
                d = self.data_from_network(n, f)
            except Exception as e:
                logger.error(f"Error loading data for {n.name}: {e}")
                logger.exception(e)
                d = None
                raise

            if d is not None:
                res.append(d)

        return res


##────────────────────────────────────────────────────────────────────────────}}}
