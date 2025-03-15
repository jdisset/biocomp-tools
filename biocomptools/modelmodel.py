from pydantic import BaseModel, ConfigDict, model_validator, BeforeValidator, Field
from dracon.utils import ser_debug
from typing import Optional, List, Union, Tuple, Callable
import numpy as np
from pathlib import Path
import pickle
import biocomp as bc
import xxhash
from typing import Annotated
import biocomp.compute as cmp
from biocomp.datautils import DataRescaler
import biocomptools.toollib.models as md
import biocomp.parameters as pr
from biocomptools.logging_config import get_logger
from biocomptools.toollib.common import config, dict_like
from biocomp.utils import load_lib, ArbitraryModel
from tqdm import tqdm

logger = get_logger(__name__)
lib = load_lib()


def load_params(maybe_path):
    """load parameters from file or use directly if already a parameter tree"""
    from biocomp.jaxutils import tree_to_np

    if isinstance(maybe_path, pr.ParameterTree):  # already loaded
        return tree_to_np(maybe_path)

    if isinstance(maybe_path, str):
        if maybe_path.startswith('mlflow-'):
            raise NotImplementedError("mlflow loading not implemented")
        else:
            maybe_path = Path(maybe_path)

    with open(maybe_path, 'rb') as f:
        return tree_to_np(pickle.load(f))


def get_shared_params(params):
    """extract shared parameters from parameter tree"""
    if params is None:
        return None

    shared, _ = params.filter_by_tag(['shared'])
    return shared


def get_nonshared_params(params):
    """extract non-shared parameters from parameter tree"""
    if params is None:
        return None

    _, nonshared = params.filter_by_tag(['shared'])
    return nonshared


def empty_params():
    """create empty parameter tree"""
    return pr.ParameterTree()


class NodeSpec(BaseModel):
    """specification for a node in the network"""

    network_id: int
    node_id: int


class BiocompModel(ArbitraryModel):
    """model containing compute configuration, rescaler, and shared parameters"""

    compute_config: cmp.ComputeConfig
    rescaler: DataRescaler
    shared_params: Annotated[
        pr.ParameterTree,
        BeforeValidator(get_shared_params),
        BeforeValidator(load_params),
    ] = Field(default_factory=empty_params)

    def save(self, path):
        """save model to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def signature(self):
        """compute unique signature for this model"""
        import base58

        paramspickle = pickle.dumps(self.shared_params)
        this_str = str(self.compute_config) + str(self.rescaler) + str(paramspickle)
        h = xxhash.xxh64(this_str)
        sig = base58.b58encode(h.digest()).decode()
        return sig

    @classmethod
    def load(cls, path):
        """load model from file"""
        with open(path, 'rb') as f:
            m = pickle.load(f)
            assert isinstance(m, cls)
            return m


def load_model(maybe_path):
    """load model from file or use directly if already a model"""
    if isinstance(maybe_path, BiocompModel):
        # already loaded
        return maybe_path

    if dict_like(maybe_path):
        # direct model construction from dict
        return BiocompModel(**maybe_path)

    if isinstance(maybe_path, str):
        maybe_path = Path(maybe_path)

    return BiocompModel.load(maybe_path)


def make_list(x):
    """convert single item to list if not already a list"""
    if not isinstance(x, list):
        return [x]
    return x


class NetworkModel(BaseModel):
    """
    model that combines a biocomp model with networks for prediction

    this class manages the compute stack, parameters, and provides methods
    for prediction including node collection points
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', validate_default=False)

    model: Annotated[BiocompModel, BeforeValidator(load_model)]
    network: bc.Network | list[bc.Network]

    _stack: Optional[cmp.ComputeStack] = None
    _params: Optional[pr.ParameterTree] = None
    _batch_apply: Optional[Callable] = None

    @model_validator(mode='before')
    @classmethod
    def prepare_network(cls, data):
        """prepare network data before model initialization"""
        return data

    def model_post_init(self, *args, **kwargs):
        """initialize model after validation"""
        super().model_post_init(*args, **kwargs)

        # ensure nework is a list
        if not isinstance(self.network, list):
            self.network = [self.network]

        # initialize stack and parameters
        self.build_stack()
        self.update_params()

    def build_stack(self):
        """build compute stack from networks"""
        try:
            import jax

            self._stack = cmp.ComputeStack(networks=self.network)
            self._stack.build(self.model.compute_config)
            self._batch_apply = jax.jit(
                jax.vmap(self._stack.apply, in_axes=(None, 0, 0, 0, 0, None))
            )
        except Exception as e:
            logger.error(f"error building stack: {e}")
            raise e

    def update_params(self):
        """update parameters from model and initialize local parameters"""
        from jax.random import PRNGKey

        assert self._stack is not None
        try:
            init_params = self._stack.init(PRNGKey(0))
        except Exception as e:
            logger.error(f"error initializing stack: {e}")
            logger.error(f"networks: {self.network}")
            logger.error("compute graphs:")
            networks = self.network
            if not isinstance(self.network, list):
                networks = [self.network]
            for n in networks:
                logger.error(n.compute_graph)
            raise e
        try:
            self._local_params = get_nonshared_params(init_params)
            self._params = load_params(
                pr.ParameterTree.merge(self.model.shared_params, self._local_params)
            )
        except Exception as e:
            logger.error(f"error updating params: {e}")
            logger.error(f"networks: {self.network}")
            raise e

    def signature(self):
        """get unique signature for this model"""
        return self.model.signature()

    def with_model(self, model: BiocompModel) -> 'NetworkModel':
        """create a new network model with a different biocomp model"""
        logger.debug(f"creating new network model with model {model.signature()=}")
        new_model = self.model_copy(update={'model': model})
        new_model.update_params()

        ser_err = ser_debug(new_model)
        if not ser_err:
            logger.debug("No serialization errors in NetworkModel after with_model")

        return new_model

    def get_node_indices(self, network_id: int, node_id: int):
        """
        get input and output indices for a virtual node in the compute stack

        parameters:
            network_id: index of the network
            node_id: id of the node within the network

        returns:
            tuple of (input_indices, output_indices) where each is a list of arrays
        """
        if self._stack is None:
            raise ValueError("stack not built")

        layer_id, n_id = self._stack.node_map[(network_id, node_id)]
        layer = self._stack.layers[layer_id]
        node = layer.nodes[n_id]

        input_indices = []
        if layer.f_input_shapes:  # skip for input layer
            for input_slot in range(len(layer.f_input_shapes)):
                start_idx = self._stack.get_node_input_start_index(node, input_slot)
                input_shape = layer.f_input_shapes[input_slot]
                length = int(np.prod(input_shape))
                input_indices.append(np.arange(start_idx, start_idx + length))

        output_indices = []
        for output_slot in range(len(layer.f_out_shapes)):
            start_idx = self._stack.get_node_output_start_index(node, output_slot)
            output_shape = layer.f_out_shapes[output_slot]
            length = int(np.prod(output_shape))
            output_indices.append(np.arange(start_idx, start_idx + length))

        return input_indices, output_indices

    def get_network_output_indices(self, network_idx: int):
        """
        get output indices and shapes for a network

        parameters:
            network_idx: index of the network

        returns:
            tuple of (output_indices, output_shapes)
        """
        if self._stack is None:
            raise ValueError("stack not built")

        return self._stack.get_network_output_indices(network_idx)

    def predict_unscaled(
        self,
        X: np.ndarray,
        key=None,
        max_batch_size=500,
        disable_variational: bool = True,
        collection_points: Optional[List[NodeSpec]] = None,
        z_value: Union[str, float] = 'uniform',
    ) -> np.ndarray:
        """
        predict but rescale input data to latent space before prediction
        (and rescale to original range back after)

        parameters: c.f. predict()

        returns:
            predictions in original range
        """
        logger.debug("Rescaling input data into latent space before prediction")
        scaled_X = self.model.rescaler.fwd(X)
        scaled_result, collections = self.predict(
            scaled_X,
            key=key,
            max_batch_size=max_batch_size,
            disable_variational=disable_variational,
            collection_points=collection_points,
            z_value=z_value,
        )
        scaled_yhat = self.model.rescaler.inv(scaled_result)
        return scaled_yhat, collections

    def predict(
        self,
        X: np.ndarray,
        key=None,
        max_batch_size=500,
        disable_variational: bool = True,
        collection_points: Optional[List[NodeSpec]] = None,
        z_value: Union[str, float] = 'uniform',
    ) -> np.ndarray:
        """
        make predictions using the model

        parameters:
            X: input data
            key: random key for predictions
            max_batch_size: maximum batch size for predictions
            disable_variational: whether to disable variational parameters
            collection_points: list of points to collect data from
            z_value: value for z latents, either 'uniform' or a float

        returns:
            predictions
        """
        import jax
        import jax.numpy as jnp
        from jax.random import PRNGKey

        if key is None:
            key = PRNGKey(0)
        if isinstance(key, int):
            key = PRNGKey(key)

        # prepare parameters
        params = self._params
        if disable_variational:
            logger.debug("disabling variational embeddings")
            logstd = params['shared']['quantization']['logstdevs']
            for path, value in logstd.iter_leaves():
                logstd[path] = jnp.ones_like(value) * -100

        # determine collection indices
        collect_in_indices, collect_out_indices = self.prepare_collection_indices(collection_points)
        logger.debug(f"collect_in_indices: {collect_in_indices}")
        logger.debug(f"collect_out_indices: {collect_out_indices}")

        # prepare for batched prediction
        num_z = self._params["global/number_of_quantile_variables"]
        n_samples = X.shape[0]
        n_batches = (n_samples + max_batch_size - 1) // max_batch_size
        logger.debug(
            f"predicting {n_samples} samples in {n_batches} batches, with {len(self.network)} networks. signature: {self.signature()}"
        )

        # make predictions in batches
        all_yhats = []
        collections_x = []
        collections_y = []
        batch_keys = jax.random.split(key, n_batches)
        for i, batch_key in tqdm(list(enumerate(batch_keys)), desc="predicting"):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, n_samples)
            batch_size = end_idx - start_idx

            if z_value == 'uniform':
                Z_batch = jax.random.uniform(batch_key, (batch_size, num_z))
            else:
                Z_batch = jnp.ones((batch_size, num_z)) * z_value

            keys_batch = jax.random.split(batch_key, batch_size)

            # TODO: prevent cache misses for last batch (when it doesn't fill the batch)

            out, (_, fullout) = self._batch_apply(
                params,
                X[start_idx:end_idx],
                Z_batch,
                keys_batch,
                None,
                None,
            )

            if len(collect_out_indices) > 0:
                collections_y_batch = np.asarray(fullout[:, collect_out_indices])
                collections_x_batch = np.asarray(fullout[:, collect_in_indices])
                collections_x.append(collections_x_batch)
                collections_y.append(collections_y_batch)

            yhat_batch = np.asarray(out)
            all_yhats.append(yhat_batch)

        # combine batches
        yhat = np.concatenate(all_yhats, axis=0)

        if len(collections_x) > 0:
            collections = (
                np.concatenate(collections_x, axis=0),
                np.concatenate(collections_y, axis=0),
            )
        else:
            collections = (None, None)

        return yhat, collections

    def prepare_collection_indices(
        self,
        collection_points: Optional[List[NodeSpec]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        prepare indices for collection points

        parameters:
            collection_points: list of points to collect data from

        returns:
            tuple of (collect_in_indices, collect_out_indices)
        """
        if collection_points is None:
            return [], []

        collect_out_indices = []
        collect_in_indices = []
        if collection_points:
            for point in collection_points:
                collect_in_idx, collect_out_idx = self.get_node_indices(
                    point.network_id, point.node_id
                )
                collect_in_indices.extend(collect_in_idx)
                collect_out_indices.extend(collect_out_idx)

        return (
            np.asarray(collect_in_indices).flatten(),
            np.asarray(collect_out_indices).flatten(),
        )

    def split_outputs_per_network(
        self, yhat: np.ndarray, max_samples: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        split a stacked output into per-network outputs

        parameters:
            yhat: stacked output array
            max_samples: maximum number of samples to include (for truncation)

        returns:
            list of per-network output arrays
        """
        if self._stack is None:
            raise ValueError("stack not built")

        network_outputs = []
        output_start_id = 0

        # TODO: when we use different collection points than the regular output,
        # we can't simply split by network_output_indices. we need to figure out the
        # shape of each output collection
        for i, _ in enumerate(self.network):
            # get output shapes for this network
            _, output_shapes = self._stack.get_network_output_indices(i)
            assert isinstance(output_shapes, list)

            # process each output
            outputs = []
            for output_shape in output_shapes:
                nout = np.prod(output_shape)
                output = yhat[:, output_start_id : output_start_id + nout].reshape(
                    -1, *output_shape
                )
                outputs.append(output)
                output_start_id += nout

            if not all(output.shape == outputs[0].shape for output in outputs):
                raise ValueError(
                    f"Outputs have different shapes: {[output.shape for output in outputs]}"
                )
            network_output = np.concatenate(outputs, axis=1)

            # truncate if needed
            if max_samples is not None and max_samples < network_output.shape[0]:
                network_output = network_output[:max_samples]

            network_outputs.append(network_output)

        return network_outputs

    def visualize_stack(self, output_path='/tmp/stackviz.html'):
        """visualize the compute stack as html"""
        import biocomptools.toollib.stackviz as sv

        sv.save_stackviz(self._stack, output_path)
        logger.debug(f"saved stack visualization to {output_path}")
