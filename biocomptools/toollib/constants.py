### {{{                          --     import     --
import biocomp as bc
from biocomp import utils as ut
import json
from pathlib import Path

import logging
import os

from typing import Union, List, Tuple, Dict, Any, Callable, Collection

tlog = logging.getLogger('biocomp_tools_common')
# tlog.setLevel(logging.DEBUG)

PathLike = Union[str, Path]
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     default variables     --
DEFAULT_LOCAL_VAR_FILE = '__local_vars.py'

DEFAULT_CALIB_PATHS = [
    './data/calibrated_data_v3',
    './data/calibrated_data_v2',
    './data/calibrated_data',
]
DEFAULT_CALIB_NAMES = ['v3', 'v2', 'old']

DEFAULT_XP_PATH = ut.DEFAULT_XP_PATH
DEFAULT_RECIPE_PATH = ut.DEFAULT_RECIPE_PATH

DEFAULT_XP_CACHE_DIR = './devtmp/cache/xp_objs'

DEFAULT_PROTEIN_ALIASES = {'EBFP': 'EBFP2', 'L0.G_MNEONGREEN': 'MNEONGREEN'}

DEFAULT_BIOCOMP_ROOT = '/Users/jeandisset/Dropbox (MIT)/Biocomp'


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     get env or local     --
BIOCOMP_LOCAL_VAR_FILE = os.getenv('BIOCOMP_LOCAL_VAR_FILE', DEFAULT_LOCAL_VAR_FILE)
BIOCOMP_LOCAL_VARS = {}

def get_env_or_local(varname, default=None, filename=BIOCOMP_LOCAL_VAR_FILE, raise_error=True):
    """
    Get a variable from the environment (in priority), or from local var file if available.
    """
    global BIOCOMP_LOCAL_VARS

    if varname in os.environ:
        tlog.debug(f'Variable {varname} found in environment')
        return os.environ[varname]

    if filename not in BIOCOMP_LOCAL_VARS:
        if os.path.exists(filename):
            tlog.debug(f'Loading local vars from {filename}')
            with open(filename, 'r') as f:
                local_vars = {}
                exec(f.read(), {}, local_vars)
                tlog.debug(f'Local vars found: {local_vars.keys()}')
                BIOCOMP_LOCAL_VARS[filename] = local_vars

    if filename in BIOCOMP_LOCAL_VARS and varname in BIOCOMP_LOCAL_VARS[filename]:
        tlog.debug(f'Variable {varname} found in local vars')
        return BIOCOMP_LOCAL_VARS[filename][varname]

    if raise_error and default is None:
        raise ValueError(f'Variable {varname} not found in either environment or {filename}')
    else:
        tlog.debug(f'Variable {varname} not found in either env or {filename}, using default value')
        return default

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                     --     biocomp constants     --

BIOCOMP_DB_NAME = get_env_or_local('BIOCOMP_DB_NAME')
BIOCOMP_DB_USER = get_env_or_local('BIOCOMP_DB_USER')
BIOCOMP_DB_PASS = get_env_or_local('BIOCOMP_DB_PASS')
BIOCOMP_DB_HOST = get_env_or_local('BIOCOMP_DB_HOST')
BIOCOMP_DB_PORT = get_env_or_local('BIOCOMP_DB_PORT')

BIOCOMP_ROOT = get_env_or_local('BIOCOMP_ROOT', default=DEFAULT_BIOCOMP_ROOT, raise_error=False)
assert BIOCOMP_ROOT, 'BIOCOMP_ROOT is not set'

BIOCOMP_PROTEIN_ALIASES = get_env_or_local(
    'BIOCOMP_PROTEIN_ALIASES', default=DEFAULT_PROTEIN_ALIASES, raise_error=False
)
if BIOCOMP_PROTEIN_ALIASES is None:
    BIOCOMP_PROTEIN_ALIASES = {}
if not isinstance(BIOCOMP_PROTEIN_ALIASES, dict):
    BIOCOMP_PROTEIN_ALIASES = json.loads(BIOCOMP_PROTEIN_ALIASES)


DEFAULT_CACHE_DIR = Path(BIOCOMP_ROOT) / '__cache'

BIOCOMP_CACHE_DIR = get_env_or_local('BIOCOMP_CACHE_DIR', default=DEFAULT_CACHE_DIR)
if isinstance(BIOCOMP_CACHE_DIR, str):
    BIOCOMP_CACHE_DIR = Path(BIOCOMP_CACHE_DIR)


BIOCOMP_NETWORK_CACHE_DIR = None
if BIOCOMP_CACHE_DIR is not None:
    BIOCOMP_NETWORK_CACHE_DIR = BIOCOMP_CACHE_DIR / 'networks'


##────────────────────────────────────────────────────────────────────────────}}}

