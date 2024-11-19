### {{{                          --     import     --
from biocomp import utils as ut
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
import xxhash
from omegaconf import DictConfig, ListConfig, open_dict
import pkg_resources
import subprocess
from typing import Dict, List, Optional
import os
import importlib.metadata
import importlib.util

# using base58 instead of base64 because it's url-safe
import base58

import biocomp.utils as ut
from biocomp.utils import ArbitraryModel
import biocomp as bc


import dracon as dr
from dracon.lazy import resolve_all_lazy

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
resolve_all_lazy(config)


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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                       --     general utils     --


def get_package_git_hashes(package_names: List[str]) -> Dict[str, Optional[str]]:
    """
    Get the git hash for specified locally installed Python packages.
    """

    from importlib.resources import files, as_file

    results = {}
    for package_name in package_names:
        try:
            pkg = files(package_name)
            with as_file(pkg) as pkg_path:
                base_dir = Path(pkg_path).parent.resolve()
                git_dir = base_dir / ".git"
                if git_dir.exists():
                    try:
                        result = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=base_dir,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        git_hash = result.stdout.strip()
                        results[package_name] = git_hash
                    except (subprocess.SubprocessError, FileNotFoundError):
                        results[package_name] = None
                else:
                    results[package_name] = None

        except Exception as e:
            results[package_name] = None

    return results


def get_git_hash():
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


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
