## {{{                          --     imports     --
import glob
from pydantic.functional_validators import BeforeValidator
from typing import Any, Dict, Union, Annotated, Optional, List, Callable

from functools import partial

import pandas as pd
import numpy as np

from pathlib import Path
import biocomp as bc
import biocomp.utils as ut
from biocomp.plotutils import PlotData
import biocomp.plotutils as pu

from biocomp.utils import PartialFunction
import biocomptools.toollib.common as cm
import biocomptools.toollib.models as md
from biocomptools.toollib.resolvable import resolved


from biocomptools.toollib.plot import (
    DataSource,
    SpecializedDataSource,
    datalog,
)


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


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     Recipe     --

ForcedStr = Annotated[str, BeforeValidator(to_str)]
ForcedOptionalStr = Annotated[Optional[str], BeforeValidator(to_str)]


class RecipeDataSource(SpecializedDataSource):

    recipe_path: ForcedStr
    data_path: ForcedStr
    cache_dir: ForcedOptionalStr = None
    color_aliases: Optional[Dict[str, str]] = cm.config.protein_aliases
    input_order: Optional[List[int]] = None

    def model_post_init(self, *a):
        super().model_post_init(*a)
        datalog.debug('Initializing RecipeDataSource with figure_maker: %s', self.figure_maker)

    def get_data(self) -> List[PlotData]:

        lib = ut.load_lib()

        recipe_file = Path(self.recipe_path).expanduser().resolve()
        data_file = Path(self.data_path).expanduser().resolve()

        candidate_networks = bc.recipe.network_from_recipe(
            recipe_file, lib, inverse='shortest', use_cache=self.cache_dir
        )
        assert isinstance(candidate_networks, list)
        if len(candidate_networks) == 0:
            raise ValueError(f'No networks built for recipe {self.recipe_path}')
        assert len(candidate_networks) == 1
        X, Y = bc.recipe.get_network_XY(
            candidate_networks[0], data_file, color_aliases=self.color_aliases
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        metadata = resolved(self.metadata)
        metadata['filename'] = data_file.name
        metadata['file_path'] = data_file.as_posix()
        metadata['file_stem'] = data_file.stem
        metadata['recipe_path'] = recipe_file.as_posix()
        metadata['recipe_stem'] = recipe_file.stem
        metadata['network_info'] = bc.network.generate_network_info(candidate_networks[0])

        pdata = pu.extract_plot_data_from_network(
            candidate_networks[0],
            X,
            Y,
            input_order=self.input_order,
            protein_aliases=self.color_aliases,
            metadata=metadata,
        )
        return [pdata]

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}RecipeDataSource({truncated_path(self.recipe_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                            --     Raw     --


class RawDataSource(SpecializedDataSource):

    data_path: Union[Path, str]
    output_column: Optional[str] = None
    input_columns: Optional[List[str]] = None
    input_names: Optional[List[str]] = None  # alias to use for input_columns
    output_name: Optional[str] = None  # alias to use for output_column

    file_order: Optional[PartialFunction | Callable] = None

    def model_post_init(self, *_):
        if self.file_order is None:
            self.file_order = partial(sorted, key=lambda x: x.stem)

    def check_file(self, data_file):
        SUPPORTED_EXTENSIONS = ['.csv']
        if not data_file.exists():
            raise ValueError(f'Data path {data_file} does not exist')
        extension = data_file.suffix
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'''Unsupported extension {extension} for {data_file}.
                    Supported extensions: {SUPPORTED_EXTENSIONS}'''
            )

    def load_file(self, data_file) -> List[PlotData]:

        extension = data_file.suffix

        if extension == '.csv':
            df = pd.read_csv(data_file, engine="pyarrow")
            assert isinstance(df, pd.DataFrame)
            assert self.input_columns is not None
            for col in self.input_columns:
                if col not in df.columns:
                    raise ValueError(
                        f'Column {col} not found in {data_file}. Available: {df.columns}'
                    )
            assert self.output_column is not None
            if self.output_column not in df.columns:
                raise ValueError(
                    f'''Column {self.output_column} not found in {data_file}.
                Available: {df.columns}'''
                )

            input_names = self.input_columns
            output_name = self.output_column

            if self.input_names is not None:
                assert len(self.input_names) == len(self.input_columns)
                input_names = self.input_names
            if self.output_name is not None:
                assert isinstance(self.output_column, str)
                output_name = self.output_name

            x = df[self.input_columns].to_numpy()
            y = df[self.output_column].to_numpy()

            metadata = resolved(self.metadata)
            metadata['filename'] = data_file.name
            metadata['file_path'] = data_file.as_posix()
            metadata['file_stem'] = data_file.stem

            return [
                pu.PlotData(
                    x=x,
                    y=y,
                    input_names=input_names,
                    output_name=output_name,
                    metadata=metadata,
                )
            ]

        else:
            raise NotImplementedError(f'Extension {extension} not implemented')

    def get_data(self) -> List[PlotData]:
        if isinstance(self.data_path, str):
            # data_path can contain wildcards in the filename so we glob all into a list
            datapath = Path(self.data_path).expanduser().resolve().absolute().as_posix()
            all_data_files = [Path(f) for f in glob.glob(datapath)]
        else:
            all_data_files = [self.data_path]

        all_data_files = sorted(all_data_files, key=lambda x: x.stem)
        all_data = []
        for data_file in all_data_files:
            self.check_file(data_file)
            all_data += self.load_file(data_file)
        return all_data

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}RawDataSource({truncated_path(self.data_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                          --     DB   --

from sqlmodel import tuple_
from sqlmodel import Field, Session, SQLModel, create_engine, select, text
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.elements import TextClause

config = cm.config

def to_text_clause(data: Any) -> TextClause:
    if isinstance(data, str):
        return text(data)
    return data

class DBSource(SpecializedDataSource):
    network_query: Annotated[TextClause, BeforeValidator(to_text_clause)] = 'select * from network'
    color_aliases: Optional[Dict[str, str]] = cm.config.protein_aliases
    input_order: Optional[List[int]] = None

    root_path: str = config.paths.root
    db_path: str = config.db.sqlite.path

    def model_post_init(self, *a):
        super().model_post_init(*a)


    def data_from_network(self, network:md.Network) -> PlotData:

        actual_network = md.build(network, path_prefix=self.root_path, overwrite_info=True)
        assert isinstance(actual_network, bc.network.Network)

        metadata = resolved(self.metadata)
        metadata['built_network'] = actual_network
        metadata['query'] = self.network_query
        metadata['network'] = network
        metadata['network_info'] = network.network_info
        metadata['source_type'] = 'DB'
        metadata = {**metadata, **network.model_dump()}

        if network.data_file is None:
            raise ValueError(f'No data file specified for network {network.name}')

        metadata['file_stem'] = Path(network.data_file).expanduser().resolve().stem

        data_file = Path(self.root_path).expanduser().resolve() / network.data_file

        if not data_file.exists():
            raise ValueError(f'Data file {data_file} does not exist for network {network.name}')

        X, Y = bc.recipe.get_network_XY(
            actual_network,
            data_file,
            color_aliases=self.color_aliases
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        pdata = pu.extract_plot_data_from_network(
            actual_network,
            X,
            Y,
            input_order=self.input_order,
            protein_aliases=self.color_aliases,
            metadata=metadata,
        )

        return pdata


    def get_data(self) -> List[PlotData]:
        engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, False)
        with Session(engine) as session:
            result = session.exec(self.network_query)
            data = result.fetchall()
            if len(data) == 0:
                raise ValueError(f'No data returned for query {self.network_query}')
            columns = result.keys()
            datalog.info(
                'Loaded %d rows with %d columns from query %s',
                len(data),
                len(columns),
                str(self.network_query),
            )

            networks = [md.Network(**dict(zip(columns, row))) for row in data]
            all_data = [self.data_from_network(network) for network in networks]
            return all_data



    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}RecipeDataSource({truncated_path(self.recipe_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
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
