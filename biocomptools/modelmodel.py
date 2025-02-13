## {{{                          --     imports     --
from biocomp import utils as ut
import jax
import jax.numpy as jnp
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

    if isinstance(maybe_path, pr.ParameterTree):  # already loaded
        return ut.tree_to_np(maybe_path)

    if isinstance(maybe_path, str):
        if maybe_path.startswith('mlflow-'):
            # TODO: implement this
            raise NotImplementedError("mlflow loading not implemented")
        else:
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


class NetworkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', validate_default=False)

    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    network: md.Network | list[md.Network]

    _stack: Optional[cmp.ComputeStack] = None

    @model_validator(mode='before')
    @classmethod
    def prepare_network(cls, data):
        print("Preparing network...")
        return data

    def model_post_init(self, *args, **kwargs):
        if not isinstance(self.network, list):
            self.network = [self.network]

        for n in self.network:
            if hasattr(n, 'recipe') and n.recipe is not None:
                n.build(lib=lib, use_cache=config.paths.cache.networks)

        # Create and build stack
        self._stack = cmp.ComputeStack([n._network for n in self.network])
        self._stack.build(self.model.compute_config)

        # Initialize parameters
        self._local_params = get_nonshared_params(self._stack.init(PRNGKey(0)))
        self._params = pr.ParameterTree.merge(self.model.shared_params, self._local_params)
        self._batch_apply = jax.jit(jax.vmap(self._stack.apply, in_axes=(None, 0, 0, 0, 0, None)))

    def with_shared_params(self, shared_params) -> 'NetworkModel':
        print("shared_params", shared_params)
        newmodel = self.model_copy()
        newmodel._params = pr.ParameterTree.merge(shared_params, self._local_params)
        return newmodel

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkModel':
        print("with_shared_params")
        return self.with_shared_params(model.shared_params)

    def predict_unscaled(self, X: np.ndarray, key=None) -> np.ndarray:
        return self.model.rescaler.inv(self.predict(self.model.rescaler.fwd(X), key))

    def predict(
        self,
        X: np.ndarray,
        key=None,
        max_batch_size=10000,
        disable_variational: bool = True,
    ):
        if key is None:
            key = PRNGKey(0)
        if isinstance(key, int):
            key = PRNGKey(key)

        num_z = self._params["global/number_of_quantile_variables"]
        n_samples = X.shape[0]
        n_batches = (n_samples + max_batch_size - 1) // max_batch_size
        print(
            f"Predicting {n_samples} samples in {n_batches} batches, with {len(self.network)} networks"
        )

        all_yhats = []
        from tqdm import tqdm

        params = self._params

        if disable_variational:
            logstd = params['shared']['quantization']['logstdevs']
            smallog = jax.tree_map(lambda x: jnp.ones_like(x) * -100, logstd)
            params.set_subtree_at('shared/quantization/logstdevs', smallog)

        print(
            f"Predicting {n_samples} samples in {n_batches} batches, with {len(self.network)} networks"
        )

        batch_keys = jax.random.split(key, n_batches)
        for i, batch_key in tqdm(list(enumerate(batch_keys)), desc="Predicting"):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, n_samples)
            batch_size = end_idx - start_idx

            Z_batch = jax.random.uniform(batch_key, (batch_size, num_z))
            keys_batch = jax.random.split(batch_key, batch_size)

            yhat_batch, _ = self._batch_apply(
                params, X[start_idx:end_idx], Z_batch, keys_batch, None, None
            )

            all_yhats.append(yhat_batch)

        yhat = np.concatenate(all_yhats, axis=0)
        print("yhat shape", yhat.shape)

        return yhat
