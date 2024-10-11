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
from pydantic import Field, BaseModel, BeforeValidator, ConfigDict
from biocomp.utils import ArbitraryModel

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


def random_split_like_tree(rng_key, target=None, treedef=None):
    import jax

    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    import jax

    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree.map(
        lambda x, k: jax.random.normal(k, shape=x.shape),
        target,
        keys_tree,
    )


def get_shared_params(params):
    shared, _ = params.filter_by_tag(['shared'])
    return shared


def get_nonshared_params(params):
    _, nonshared = params.filter_by_tag(['shared'])
    return nonshared


def load_params(maybe_path):
    import pickle

    if isinstance(maybe_path, pr.ParameterTree):
        # already loaded
        return maybe_path

    if isinstance(maybe_path, str):
        maybe_path = Path(maybe_path)

    with open(maybe_path, 'rb') as f:
        return pickle.load(f)


class BiocompModel(ArbitraryModel):
    compute_config: cmp.ComputeConfig
    shared_params: Annotated[
        pr.ParameterTree,
        BeforeValidator(get_shared_params),
        BeforeValidator(load_params),
    ]

    def save(self, path):
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import pickle

        with open(path, 'rb') as f:
            m = pickle.load(f)
            assert isinstance(m, cls)
            return m


def load_model(maybe_path):
    if isinstance(maybe_path, BiocompModel):
        # already loaded
        return maybe_path

    if isinstance(maybe_path, str):
        maybe_path = Path(maybe_path)

    return BiocompModel.load(maybe_path)


class SingleNetworkModel(ArbitraryModel):
    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    network: md.Network

    def model_post_init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network.build(lib=lib, use_cache=config.paths.cache.networks)
        assert self.network.network is not None

        self._stack: cmp.ComputeStack = cmp.ComputeStack([self.network.network])
        self._stack.build(self.model.compute_config)
        self._local_params = get_nonshared_params(self._stack.init(PRNGKey(0)))

        self._params = pr.ParameterTree.merge(self.model.shared_params, self._local_params)

    def predict(self, X: np.ndarray, key) -> np.ndarray:
        assert isinstance(self._stack, cmp.ComputeStack)
        Z = jax.random.uniform(key, ybatches.shape)

        batch_apply = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
