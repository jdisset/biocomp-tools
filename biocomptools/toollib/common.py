### {{{                          --     import     --
from tqdm import tqdm
from pathlib import Path

# using base58 instead of base64 because it's url-safe
import base58

from dracon.lazy import resolve_all_lazy

from typing import (
    Union,
    List,
    Tuple,
    Dict,
    TypeVar,
    Optional,
)

import logging
from biocomptools.toollib.config import config as config

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


resolve_all_lazy(config)


def maybetqdm(x, min_len=5, **kw):
    if isinstance(x, zip):
        x = list(x)
    if len(x) > min_len:
        return tqdm(x, **kw)
    return x


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     types     --
T = TypeVar('T')
U = TypeVar('U')

ListOrSingle = Union[List[T], T]
Pair = Tuple[T, T]
DictOrList = Union[Dict[U, T], List[T]]


def make_context_from_types(types):
    """Convert a list of types to a context dict mapping name -> type."""
    return {t.__name__: t for t in types}


##────────────────────────────────────────────────────────────────────────────}}}


DEFAULT_NAME_LOOKUP = {
    'mNeonGreen': 'mNG',
    'PgU': 'Pgu',
}


def make_pretty_input_names(ratios, ordered_input_names, name_lookup=None):
    """create formatted input names for display"""
    if name_lookup is None:
        name_lookup = DEFAULT_NAME_LOOKUP

    fluo_markers = [p[0][-1].upper() for p in ratios]
    names = []

    for p in ordered_input_names:
        x = ''
        if p.upper() in fluo_markers:
            idx = fluo_markers.index(p.upper())
            content = ' + '.join(ratios[idx][0][:-1])
            if content:
                x += rf"${content}$"

        if name_lookup is not None:
            for k, v in name_lookup.items():
                x = x.replace(k, v)

        names.append(x)

    logger.debug(f"Created pretty input names: {names} from ordered inputs: {ordered_input_names}")
    return names
