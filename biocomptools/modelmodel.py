## {{{                          --     imports     --
from biocomp import utils as ut
import biocomp as bc
import optax
from labellines import labelLines, labelLine
from biocomp.recipe import get_network_XY
import jax
import biocomp.compute as cmp
from pathlib import Path
from functools import partial
import numpy as np
import biocomp.datautils as du
from biocomp import nodes
from pydantic import Field, BaseModel, BeforeValidator

import dracon as dr
from sqlmodel import Session, select
import time
import pandas as pd
from typing import Optional, List, Tuple, TypeVar, Dict, Any, Callable, Annotated
from tqdm import tqdm
from biocomp.utils import EncodedPartialFunction
from jax.random import PRNGKey

import logging
from biocomptools.toollib.common import config
import biocomptools.toollib.models as md
from rich.console import Console
import biocomp.parameters as pr

logger = logging.getLogger('build_xp_table')
logger.setLevel(logging.DEBUG)
logging.getLogger('biocomp').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)

ROOT = Path(config.paths.root).expanduser().resolve()
lib = ut.load_lib()

##────────────────────────────────────────────────────────────────────────────}}}


# load train_results.pkl
fpath = Path('tmp/train_results.pkl')
train_results = bc.utils.load(fpath)
params = train_results['params']


def get_shared_params(params):
    _, shared = params.filter_by_tag(['local'])
    return shared


def get_local_params(params):
    local, _ = params.filter_by_tag(['local'])
    return local


class BiocompModel(BaseModel):
    compute_config: cmp.ComputeConfig
    network: md.Network

    shared_params: Annotated[pr.ParameterTree, BeforeValidator(get_shared_params)]

    def model_post_init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network.build(lib=lib, use_cache=config.paths.cache.networks)
        assert self.network.network is not None

        self._stack: cmp.ComputeStack = cmp.ComputeStack([self.network.network])
        self._stack.build(self.compute_config)
        self._local_params = get_local_params(self._stack.init(PRNGKey(0)))

        self._params = pr.ParameterTree.merge(self.shared_params, self._local_params)

    def run_prediction(self, X: np.ndarray, Q: np.ndarray, key) -> np.ndarray:
        assert isinstance(self._stack, cmp.ComputeStack)
        res, _ = jax.jit(self._stack.apply)(self._params, X, Q, key)
        return res

# TODO: 
# merge with the plot stuff so that it uses the same visualization

##

import asteval
a=2


aeval = asteval.Interpreter()
aeval.symtable['a'] = a
aeval.symtable['symtable'] = aeval.symtable

def getval(varname, default=None):
    try:
        varname = varname.lstrip().rstrip()
        if not varname.isidentifier():
            raise ValueError(f'Invalid identifier: {varname}')
        return eval(varname)
    except NameError:
        return default

aeval.symtable['getval'] = getval
aeval.eval('getval("b ", -1)')


# del b
##

aeval.eval('[i for i in range(10)]')

