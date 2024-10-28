### {{{                          --     import     --
from biocomp import utils as ut
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
import xxhash
from omegaconf import DictConfig, ListConfig, open_dict

# using base58 instead of base64 because it's url-safe
import base58

import biocomp.utils as ut
from biocomp.utils import ArbitraryModel
import biocomp as bc


import dracon as dr

from typing import (
    Union,
    List,
    Tuple,
    Type,
    Dict,
    Any,
    TypeVar,
    Optional,
)

from pydantic import BaseModel, Field

from biocomptools.logging_config import get_logger

tlog = get_logger(__name__)


PathLike = Union[str, Path]


config = dr.load('pkg:biocomptools:configs/default.yaml', enable_interpolation=True)
dr.draconstructor.resolve_all_lazy(config)


def maybetqdm(x, min_len=5, **kw):
    if isinstance(x, zip):
        x = list(x)
    if len(x) > min_len:
        return ut.tqdm(x, **kw)
    return x


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     types     --
T = TypeVar('T')
U = TypeVar('U')

ListOrSingle = Union[List[T], T]
Pair = Tuple[T, T]
DictOrList = Union[Dict[U, T], List[T]]
DictLike = Union[Dict, DictConfig]
AnyConfig = Union[DictConfig, ListConfig]


def dict_like(obj) -> bool:
    return (
        hasattr(obj, 'keys')
        and hasattr(obj, 'get')
        and hasattr(obj, '__getitem__')
        and hasattr(obj, '__contains__')
        and hasattr(obj, '__iter__')
    )


class ArbitraryTargetModel(ArbitraryModel):
    """A pydantic model that has a _target_ set to the name of the class
    for easy (de)serialization"""

    target_: Optional[str | Type] = Field(None, alias='_target_')

    def model_post_init(self, *a) -> None:
        # if target_ is None, we set the target attribute to the name of the class
        self.target_ = (
            self.__class__.__module__ + '.' + self.__class__.__name__
            if self.target_ is None
            else self.target_
        )
        super().model_post_init(*a)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     general utils     --


def notnull(x):
    return x != '' and x is not None and x.lower() != 'none'


def isnull(x):
    return not notnull(x)


def parse_list(input_string):
    # Split the string by comma and then strip whitespaces from each element
    if input_string is None:
        return []
    if isinstance(input_string, list):
        return input_string
    return [element.strip() for element in input_string.split(',')]


def get_name_hash(name):
    # base58 encode the xxhash of the name
    hh = xxhash.xxh128(name).digest()
    return base58.b58encode(hh).decode('utf-8')


def filter_df(df, **filters):
    for key, value in filters.items():
        if len(df) == 0:
            return df
        if callable(value):
            df = df[df[key].apply(value)]
        else:
            df = df[df[key] == value]
    return df


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                      --     omegaconf utils     --
@contextmanager
def open_dictlike(obj: Any):
    if isinstance(obj, DictConfig):
        with open_dict(obj):
            yield
    else:
        yield


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     collection utils     --


def get_networks_in_collections(collections):
    if not isinstance(collections, list):
        collections = [collections]
    # first we launch a query to check that all collections exist
    query = "SELECT * FROM collections WHERE name IN %s"
    existing_collections = query_to_df(query, (tuple(collections),))
    diff = set(collections) - set(existing_collections['name'])
    if len(diff) > 0:
        raise ValueError(f'Collections {diff} not found in database')

    param_placeholders = ', '.join(['%s'] * len(collections))

    query = f"""
    SELECT DISTINCT n.*, collection_name FROM collection_network cn
    JOIN network n ON cn.network_name = n.name
    WHERE cn.collection_name IN ({param_placeholders})
    """
    return query_to_df(query, collections)


def create_collection(name, description):
    execute_query(
        "INSERT INTO collections (name, description) VALUES (%s, %s)", (name, description)
    )


def remove_all_from_collection(collection_name):
    execute_query("DELETE FROM collection_network WHERE collection_name = %s", (collection_name,))


def delete_collection(collection_name):
    execute_query("DELETE FROM collections WHERE name = %s", (collection_name,))


def add_networks_to_collection(collection_name, network_names):
    if not isinstance(network_names, list):
        network_names = [network_names]
    values = [(collection_name, net) for net in network_names]
    execute_many("INSERT INTO collection_network (collection_name, network_name) VALUES %s", values)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     network objects loading utils     --

# TODO MAYBE:
# parallel load_network_and_data with ray


def resolve_path(filepath, path_prefix):
    if isnull(filepath):
        raise ValueError(f'File path information missing')
    filepath = Path(path_prefix) / filepath
    filepath = Path(filepath).resolve()
    if not Path(filepath).exists():
        raise ValueError(f'File {filepath} not found')
    return filepath


def load_network_and_data_from_row(
    network_row,
    lib,
    path_prefix=config.paths.root,
    protein_aliases=config.protein_aliases,
    error_handler=None,
    cache_dir: Optional[PathLike] = config.paths.cache.networks,
):
    if error_handler is None:

        def __raise_on_error(name, e):
            raise e

        error_handler = __raise_on_error

    try:
        data_file = resolve_path(network_row['data_file'], path_prefix)
        recipe_file = resolve_path(network_row['recipe_file'], path_prefix)
        candidate_networks = bc.recipe.network_from_recipe(
            recipe_file, lib, inverse='all', use_cache=cache_dir
        )
        # when trying to load a network from the database, we have to select the right one
        # After reading a recipe, we have several candidate networks, one for each possible inversion
        # we can use the markers to select the right one
        candidate_markers = [
            set(bc.recipe.escape(n.get_inverted_input_proteins())) for n in candidate_networks
        ]
        target_markers = set(parse_list(network_row['markers']))
        escaped_target_markers = bc.recipe.escape(target_markers)
        network = candidate_networks[candidate_markers.index(escaped_target_markers)]
        X, Y = bc.recipe.get_network_XY(network, data_file, color_aliases=protein_aliases)
    except Exception as e:
        return error_handler(network_row['name'], e)

    return network, X, Y


def get_network_row(netdf, net_name):
    if net_name not in netdf['name'].values:
        raise ValueError(f'Network id {net_name} not found in database')
    net_row = netdf[netdf['name'] == net_name]
    if len(net_row) > 1:
        raise ValueError(f'Network name {net_name} is not unique in database')
    return net_row.iloc[0]


def load_network_and_data(
    netdf: pd.DataFrame,
    net_name: str,
    lib: bc.recipe.PartsLibrary,
    path_prefix: PathLike = config.paths.root,
    protein_aliases: Dict = config.protein_aliases,
    cache_dir: Optional[PathLike] = config.paths.cache.networks,
):
    return load_network_and_data_from_row(
        get_network_row(netdf, net_name),
        lib,
        path_prefix=path_prefix,
        protein_aliases=protein_aliases,
        cache_dir=cache_dir,
    )


def load_networks_and_data(netdf, lib, **kwargs):
    # loads from a dataframe with network information
    # needs columns: data_file, recipe_file, markers
    # (markers is needed to disambiguate from all the potential recipe inversions)
    networks, Xs, Ys = [], [], []
    for _, row in netdf.iterrows():
        try:
            tlog.info(f'Loading network {row["name"]}')
            network, X, Y = load_network_and_data_from_row(row, lib, **kwargs)
            networks.append(network)
            Xs.append(X)
            Ys.append(Y)
        except Exception as e:
            tlog.error(f'Error loading network {row["name"]}: {e}')

    return networks, Xs, Ys


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                        --     CLIProgram     --
class CLIProgram:
    def __init__(self):
        self.is_notebook = 'ipykernel' in sys.modules
        self.parser = argparse.ArgumentParser()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, default_args=None):
        extra_args = default_args if default_args is not None else []

        # combine parsed args and extra_args. parsed args have priority over extra_args.
        # if we're in a notebook, only use extra_args. Otherwise we can combine them.
        if self.is_notebook:
            self.args = self.parser.parse_args(extra_args)
        else:
            self.args = self.parser.parse_args(extra_args + sys.argv[1:])
            ut.logger.info(f'args: {self.args}')

        self._postprocess_args()

    def _postprocess_args(self):
        pass

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self, 'args') and hasattr(self.args, attr):
            return getattr(self.args, attr)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{attr}'")


##────────────────────────────────────────────────────────────────────────────}}}


### {{{                         --     df tools     --
def merge_update(
    left_df, right_df, key_column, priority, use_left=None, use_right=None, how='outer'
):
    """
    Merge two pandas dataframes with priority-based column selection.

    Parameters:
    priority (str): 'left' or 'right' to set priority dataframe for overlapping columns.
    use_left (list[str], optional): Columns to forcibly use from left_df.
    use_right (list[str], optional): Columns to forcibly use from right_df.

    Returns:
    pd.DataFrame: Merged dataframe based on the specified rules.
    """

    tlog.debug(
        f"""Merging dataframes with priority {priority}.
    Left columns: {left_df.columns} with shape {left_df.shape}.
    Right columns: {right_df.columns} with shape {right_df.shape}."""
    )

    use_right = use_right or []
    use_left = use_left or []

    if not set(use_left).isdisjoint(use_right):
        raise ValueError("Columns in use_left and use_right must be disjoint")

    common_columns = (
        set(left_df.columns).intersection(set(right_df.columns)).difference([key_column])
    )

    # Rename common columns in right_df to avoid suffixes in the merged dataframe
    rename_columns = {col: col + '_right' for col in common_columns if col not in use_left}
    right_df_renamed = right_df.rename(columns=rename_columns)
    merged_df = pd.merge(left_df, right_df_renamed, on=key_column, how=how)

    tlog.debug(f'Common columns: {common_columns}, renamed to {rename_columns}')

    # Apply use_left, use_right, and priority rules
    for col in common_columns:
        if col in use_right or (col + '_right' in merged_df and priority == 'right'):
            merged_df[col] = merged_df[col + '_right']

        merged_df.drop(columns=[col + '_right'], inplace=True, errors='ignore')

    return merged_df


def reorder_columns_front(df, columns):
    """
    Puts the specified columns in front of the dataframe.
    """
    columns = list(columns)
    return df[columns + [col for col in df.columns if col not in columns]]


def reorder_columns_back(df, columns):
    """
    Puts the specified columns in back of the dataframe.
    """
    columns = list(columns)
    return df[[col for col in df.columns if col not in columns] + columns]


##────────────────────────────────────────────────────────────────────────────}}}
