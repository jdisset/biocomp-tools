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
