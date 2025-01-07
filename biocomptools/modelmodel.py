## {{{                          --     imports     --
from biocomp import utils as ut
import jax
import biocomp.compute as cmp
from biocomp.datautils import DataRescaler
from pathlib import Path
import numpy as np
from pydantic import BeforeValidator, Field, model_validator, BaseModel, ConfigDict
from biocomp.utils import ArbitraryModel

from typing import Callable, Annotated, Optional
from jax.random import PRNGKey

import logging
from biocomptools.toollib.common import config, dict_like
import biocomptools.toollib.models as md
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
    if params is None:
        return None

    shared, _ = params.filter_by_tag(['shared'])
    return shared


def get_nonshared_params(params):
    if params is None:
        return None

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
        return ut.tree_to_np(pickle.load(f))


def empty_params():
    return pr.ParameterTree()


class BiocompModel(ArbitraryModel):
    compute_config: cmp.ComputeConfig
    rescaler: DataRescaler
    shared_params: Annotated[
        pr.ParameterTree,
        BeforeValidator(get_shared_params),
        BeforeValidator(load_params),
    ] = Field(default_factory=empty_params)

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
    print("In load_model validator")
    if isinstance(maybe_path, BiocompModel):
        # already loaded
        return maybe_path

    if dict_like(maybe_path):
        # direct model construction from dict
        return BiocompModel(**maybe_path)

    if isinstance(maybe_path, str):
        maybe_path = Path(maybe_path)

    return BiocompModel.load(maybe_path)


def validate_network(network):
    print("In validate_network BeforeValidator")
    print(f"network: {network}")
    if isinstance(network, list):
        for n in network:
            print(f"network entry type: {type(n)}")
            if hasattr(n, '_network'):
                print(f"internal network: {n._network}")
    else:
        print(f"network type: {type(network)}")
        if hasattr(network, '_network'):
            print(f"internal network: {network._network}")
    return network


class NetworkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', validate_default=False)

    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    network: Annotated[md.Network | list[md.Network], BeforeValidator(validate_network)]

    @model_validator(mode='after')
    def validate_network(self):
        print("Validating NetworkModel...")
        print(f"Network: {self.network}")
        print(f"Network type: {type(self.network)}")
        if hasattr(self.network, '_network'):
            print(f"Internal network: {self.network._network}")
        return self

    @model_validator(mode='before')
    @classmethod
    def prepare_network(cls, data):
        print("Preparing network...")
        return data

    def model_post_init(self, *args, **kwargs):
        print("Starting NetworkModel post_init")
        print("In NetworkModel::postinit: Building networks...")

        if not isinstance(self.network, list):
            self.network = [self.network]

        for n in self.network:
            if hasattr(n, 'recipe') and n.recipe is not None:
                n.build(lib=lib, use_cache=config.paths.cache.networks)

        print("Networks built")

        # Create and build stack
        self._stack: cmp.ComputeStack = cmp.ComputeStack([n._network for n in self.network])
        self._stack.build(self.model.compute_config)

        # Initialize parameters
        self._local_params = get_nonshared_params(self._stack.init(PRNGKey(0)))
        self._params = pr.ParameterTree.merge(self.model.shared_params, self._local_params)
        self._batch_apply = jax.jit(jax.vmap(self._stack.apply, in_axes=(None, 0, 0, 0, 0, None)))

        print("Stack created")

    def with_shared_params(self, shared_params) -> 'NetworkModel':
        print("shared_params", shared_params)
        newmodel = self.model_copy()
        newmodel._params = pr.ParameterTree.merge(shared_params, self._local_params)
        return newmodel

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkModel':
        print("with_shared_params")
        return self.with_shared_params(model.shared_params)

    def predict_unscaled(
        self, X: np.ndarray, key=None, ground_truth=None, ground_truth_output_id=None
    ) -> np.ndarray:
        if key is None:
            key = PRNGKey(0)
        if isinstance(key, int):
            key = PRNGKey(key)

        num_z = self._params["global/number_of_quantile_variables"]
        Z = jax.random.uniform(key, (X.shape[0], num_z))
        keys = jax.random.split(key, X.shape[0])

        print(f'Going to predict with {X.shape=}, {Z.shape=}')

        mse = None

        x = self.model.rescaler.fwd(X)

        yhat, (grads_wrt_inputs, full_output) = self._batch_apply(
            self._params, x, Z, keys, None, None
        )
        print(f'with {x=}, {yhat=}')

        if ground_truth is not None:
            ground_truth = self.model.rescaler.fwd(ground_truth)
            logger.info("yhat shape: %s", yhat.shape)
            logger.info("ground_truth shape: %s", ground_truth.shape)
            mse = np.mean((yhat - jax.numpy.concatenate([x, ground_truth], axis=1)) ** 2)
            rmse = np.sqrt(mse)
            logger.info(f"MSE: {mse}")
            logger.info(f"RMSE: {rmse}")
            # pick 20 random samples to log
            idx = np.random.choice(X.shape[0], 20, replace=False)
            logger.info("Random samples:")
            for i in idx:
                logger.info(f"-- Sample {i} --")
                logger.info(f"x: {x[i]}")
                logger.info(f"yhat: {yhat[i]}")
                logger.info(f"ground_truth: {ground_truth[i]}")

        yhat = self.model.rescaler.inv(yhat)

        return yhat

    def predict(self, X: np.ndarray, key=None) -> np.ndarray:
        if key is None:
            key = PRNGKey(0)
        if isinstance(key, int):
            key = PRNGKey(key)

        num_z = self._params["global/number_of_quantile_variables"]
        Z = jax.random.uniform(key, (X.shape[0], num_z))
        keys = jax.random.split(key, X.shape[0])

        yhat, grads = self._batch_apply(self._params, X, Z, keys)

        return yhat
