from pydantic import BaseModel, ConfigDict, model_validator, BeforeValidator, Field
from dracon.utils import ser_debug
from typing import Optional, List, Union, Tuple, Callable, Type, TypeVar, Literal
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


M = TypeVar('M', bound='BiocompModel')


class BiocompModel(ArbitraryModel):
    """model containing compute configuration, rescaler, and shared parameters"""

    compute_config: cmp.ComputeConfig
    rescaler: DataRescaler
    shared_params: Annotated[
        pr.ParameterTree,
        BeforeValidator(get_shared_params),
        BeforeValidator(load_params),
    ] = Field(default_factory=empty_params)
    metadata: dict = {}

    def model_post_init(self, *argc, **kwargs):
        super().model_post_init(*argc, **kwargs)
        logger.debug(f"Initialized model '{self.signature}'")

    def save(self, path):
        """save model to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def signature(self):
        """compute unique signature for this model"""
        import biocomptools.toollib.hashutils as bch

        paramspickle = pickle.dumps(self.shared_params)
        this_str = str(self.compute_config) + str(self.rescaler) + str(paramspickle)
        return bch.pronounceable_hash64(this_str)

    @classmethod
    def load(cls, path):
        """load model from file"""
        logger.debug(f"loading model from {path}")
        with open(path, 'rb') as f:
            m = pickle.load(f)
            assert isinstance(m, cls)
            logger.debug(f"loaded model with signature {m.signature}")
            return m

    def save_h5(self, filename: str):
        """Saves the entire BiocompModel to a single HDF5 file."""
        import h5py
        import json

        with h5py.File(filename, 'w') as f:
            f.attrs['__model_class__'] = f"{self.__class__.__module__}.{self.__class__.__name__}"

            f.attrs['compute_config'] = json.dumps(self.compute_config.model_dump())
            f.attrs['rescaler'] = self.rescaler.model_dump_json()
            f.attrs['metadata'] = json.dumps(self.metadata)

            params_group = f.create_group('shared_params')
            params_group.attrs['tagnames'] = self.shared_params.tagnames

            data_group = params_group.create_group('data')
            pr.save_ptree_to_hdf5_group(self.shared_params.data, data_group)

            tags_group = params_group.create_group('tags')
            pr.save_ptree_to_hdf5_group(self.shared_params.tags, tags_group)

            print(f"Saved {self.__class__.__name__} to {filename}")

    @classmethod
    def load_h5(cls: Type[M], filename: str) -> M:
        """Loads a BiocompModel from an HDF5 file."""
        import h5py
        import json
        from pydantic import TypeAdapter

        with h5py.File(filename, 'r') as f:
            compute_config_data = json.loads(f.attrs['compute_config'])
            compute_config = cmp.ComputeConfig.model_validate(compute_config_data)

            # Pydantic v2's TypeAdapter is great for handling Unions like DataRescaler
            rescaler_adapter = TypeAdapter(DataRescaler)
            rescaler = rescaler_adapter.validate_json(f.attrs['rescaler'])

            metadata = json.loads(f.attrs['metadata'])

            # 2. Load the ParameterTree
            params_group = f['shared_params']
            tagnames = list(params_group.attrs.get('tagnames', []))

            data_tree = pr.load_ptree_from_hdf5_group(params_group['data'])
            tags_tree = pr.load_ptree_from_hdf5_group(params_group['tags'])

            shared_params = pr.ParameterTree(data=data_tree, tags=tags_tree, tagnames=tagnames)

            # 3. Instantiate the model class with the loaded components
            return cls(
                compute_config=compute_config,
                rescaler=rescaler,
                shared_params=shared_params,
                metadata=metadata,
            )


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

    max_points_per_batch: int = 30000

    _stack: Optional[cmp.ComputeStack] = None
    _params: Optional[pr.ParameterTree] = None
    _batch_apply: Optional[Callable] = None
    _batch_apply_cpu: Optional[Callable] = None
    _batch_apply_gpu: Optional[Callable] = None

    def model_post_init(self, *args, **kwargs):
        """initialize model after validation"""
        super().model_post_init(*args, **kwargs)
        logger.debug(f"Initialized network model '{self.signature}'")

        # ensure nework is a list
        if not isinstance(self.network, list):
            self.network = [self.network]

        # initialize stack and parameters
        self.build_stack()
        self.update_params()
        self._precompile_batch_apply()

    def build_stack(self):
        """build compute stack from networks"""
        try:
            import jax
            import jax.numpy as jnp
            from time import time
    
            self._stack = cmp.ComputeStack(networks=self.network)
            self._stack.build(self.model.compute_config)
    
            # create a general batch apply function
            # using same signature as training code: (params, inputs, quantiles, keys)
            batch_apply_fn = jax.vmap(self._stack.apply, in_axes=(None, 0, 0, 0))
    
            # JIT compile for both CPU and GPU devices
            cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
            try:
                gpu_devices = jax.devices('gpu') if jax.devices('gpu') else []
            except RuntimeError:
                # No GPU backend available
                gpu_devices = []
    
            if cpu_device:
                self._batch_apply_cpu = jax.jit(batch_apply_fn, device=cpu_device)
            else:
                self._batch_apply_cpu = jax.jit(batch_apply_fn)
    
            if gpu_devices:
                self._batch_apply_gpu = jax.jit(batch_apply_fn, device=gpu_devices[0])
            else:
                self._batch_apply_gpu = self._batch_apply_cpu
    
            # default batch apply (for backward compatibility)
            self._batch_apply = self._batch_apply_cpu
    
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
    
    def _precompile_batch_apply(self):
        """Precompile batch_apply functions with the correct batch size"""
        if self._stack is None or self._params is None:
            logger.warning("Cannot precompile: stack or params not initialized")
            return
            
        try:
            import jax
            import jax.numpy as jnp
            from time import time
            
            logger.info("Precompiling batch_apply functions...")
            
            # calculate the actual batch size that will be used during prediction
            outputs_per_sample = int(self._stack.total_nb_of_outputs)
            effective_batch_size = max(1, self.max_points_per_batch // outputs_per_sample)
            
            logger.debug(
                f"Precompiling with batch size {effective_batch_size} "
                f"(max_points={self.max_points_per_batch}, outputs_per_sample={outputs_per_sample})"
            )
            
            # create dummy inputs for precompilation with correct batch size
            dummy_x = jnp.zeros((effective_batch_size, self._stack.total_nb_of_inputs))
            dummy_z = jnp.zeros((effective_batch_size, self._params["global/number_of_quantile_variables"]))
            dummy_key = jax.random.PRNGKey(0)
            dummy_keys = jax.random.split(dummy_key, effective_batch_size)
            
            logger.debug(f"Precompilation: dummy_keys shape: {dummy_keys.shape}, effective_batch_size: {effective_batch_size}")
            
            # use the actual params that will be used during prediction
            dummy_params = self._params
            
            # precompile CPU version
            if self._batch_apply_cpu is not None:
                try:
                    start_time = time()
                    _ = self._batch_apply_cpu(
                        dummy_params, dummy_x, dummy_z, dummy_keys
                    ).block_until_ready()
                    cpu_compile_time = time() - start_time
                    logger.info(f"CPU batch_apply precompiled in {cpu_compile_time:.2f} seconds")
                except Exception as e:
                    logger.warning(f"Failed to precompile CPU batch_apply: {e}")
            
            # precompile GPU version if different from CPU
            if self._batch_apply_gpu is not self._batch_apply_cpu:
                try:
                    start_time = time()
                    _ = self._batch_apply_gpu(
                        dummy_params, dummy_x, dummy_z, dummy_keys
                    ).block_until_ready()
                    gpu_compile_time = time() - start_time
                    logger.info(f"GPU batch_apply precompiled in {gpu_compile_time:.2f} seconds")
                except Exception as e:
                    logger.warning(f"Failed to precompile GPU batch_apply: {e}")
                    
        except Exception as e:
            logger.warning(f"Precompilation failed: {e}")
    
    def with_model(self, model: BiocompModel) -> 'NetworkModel':
        """create a new network model with a different biocomp model"""
        logger.debug(f"creating new network model with model {model.signature=}")
        new_model = self.model_copy(update={'model': model})
        new_model.update_params()

        ser_err = ser_debug(new_model)
        if not ser_err:
            logger.debug("No serialization errors in NetworkModel after with_model")

        return new_model

    @property
    def signature(self):
        return self.model.signature

    def get_node_indices(self, network_id: int, node_id: int):
        """
        get input and output indices for a virtual node in the compute stack
        """
        if self._stack is None:
            raise ValueError("stack not built")

        layer_id, n_id = self._stack.node_map[(network_id, node_id)]
        layer = self._stack.layers[layer_id]
        node = layer.nodes[n_id]

        input_indices = []
        input_shapes = []
        if layer.f_input_shapes:  # skip for input layer
            for input_slot in range(len(layer.f_input_shapes)):
                start_idx = self._stack.get_node_input_start_index(node, input_slot)
                input_shape = layer.f_input_shapes[input_slot]
                length = int(np.prod(input_shape))
                input_indices.append(np.arange(start_idx, start_idx + length))
                input_shapes.append(input_shape)

        output_indices = []
        output_shapes = []
        for output_slot in range(len(layer.f_out_shapes)):
            start_idx = self._stack.get_node_output_start_index(node, output_slot)
            output_shape = layer.f_out_shapes[output_slot]
            length = int(np.prod(output_shape))
            output_indices.append(np.arange(start_idx, start_idx + length))
            output_shapes.append(output_shape)

        return (input_indices, input_shapes), (output_indices, output_shapes)

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
        max_points_per_batch=None,
        disable_variational: bool = True,
        z_value: Union[str, float] = 'uniform',
        with_shared_params: Optional[pr.ParameterTree] = None,
        collect_in_indices: Optional[np.ndarray] = None,
        collect_out_indices: Optional[np.ndarray] = None,
        device: Literal['cpu', 'gpu'] = 'cpu',
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
            max_points_per_batch=max_points_per_batch,
            disable_variational=disable_variational,
            z_value=z_value,
            with_shared_params=with_shared_params,
            collect_in_indices=collect_in_indices,
            collect_out_indices=collect_out_indices,
            device=device,
        )
        scaled_yhat = self.model.rescaler.inv(scaled_result)
        return scaled_yhat, collections

    def predict(
        self,
        X: np.ndarray,
        key=None,
        max_points_per_batch=None,
        disable_variational: bool = True,
        z_value: Union[str, float] = 'uniform',
        with_shared_params: Optional[pr.ParameterTree] = None,
        collect_in_indices: Optional[np.ndarray] = None,
        collect_out_indices: Optional[np.ndarray] = None,
        device: Literal['cpu', 'gpu'] = 'cpu',
    ):
        """
        make predictions using the model

        parameters:
            X: input data
            key: random key for predictions
            max_points_per_batch: maximum number of output points per batch
            disable_variational: whether to disable variational parameters

            collection_points: list of points to collect data from. Each point is a NodeSpec,
            meaning a network id and node id that will be used to index the full, flattened, running output of the network
            in order to collect input/output data from these specific nodes. Super useful for debugging and visualization
            (required for inner node plots)

            z_value: value for z latents, either 'uniform' or a float

        returns:
            predictions (the "regular" output of the network stack) and collections (input/output data from specific nodes)
        """
        import jax
        import jax.numpy as jnp

        max_points_per_batch = max_points_per_batch or self.max_points_per_batch

        if key is None:
            key = jax.random.PRNGKey(0)
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        # prepare parameters
        if with_shared_params is None:
            params = self._params
        else:
            shared_params = get_shared_params(with_shared_params)
            params = pr.ParameterTree.merge(shared_params, self._local_params)

        if disable_variational:
            logger.debug("disabling variational embeddings")
            logstd = params['shared']['quantization']['logstdevs']
            for path, value in logstd.iter_leaves():
                logstd[path] = jnp.ones_like(value) * -100

        assert self._stack is not None and self._stack.total_nb_of_outputs is not None
        outputs_per_sample = int(self._stack.total_nb_of_outputs)
        effective_batch_size = max(1, max_points_per_batch // outputs_per_sample)

        num_z = self._params["global/number_of_quantile_variables"]
        n_samples = X.shape[0]
        n_batches = (n_samples + effective_batch_size - 1) // effective_batch_size

        # pad X for even batch sizes
        padded_samples = n_batches * effective_batch_size
        pad_size = padded_samples - n_samples

        if pad_size > 0:
            padding_shape = list(X.shape)
            padding_shape[0] = pad_size
            padding = np.zeros(padding_shape, dtype=X.dtype)
            X_padded = np.concatenate([X, padding], axis=0)
        else:
            X_padded = X

        # select the appropriate batch apply function based on device
        if device == 'gpu' and self._batch_apply_gpu is not None:
            batch_apply = self._batch_apply_gpu
            logger.debug("Using GPU for predictions")
        else:
            batch_apply = self._batch_apply_cpu or self._batch_apply
            logger.debug("Using CPU for predictions")

        # make predictions in batches
        all_yhats = []
        all_collected_in = []
        all_collected_out = []
        batch_keys = jax.random.split(key, n_batches)
        for i, batch_key in tqdm(list(enumerate(batch_keys)), desc="predicting"):
            start_idx = i * effective_batch_size
            end_idx = (i + 1) * effective_batch_size
            batch_size = effective_batch_size

            if z_value == 'uniform':
                Z_batch = jax.random.uniform(batch_key, (batch_size, num_z))
            else:
                Z_batch = jnp.ones((batch_size, num_z)) * z_value

            keys_batch = jax.random.split(batch_key, batch_size)

            assert batch_apply is not None
            
            # track potential recompilation
            import time
            batch_start = time.time()
            out, (_, fullout) = batch_apply(
                params,
                X_padded[start_idx:end_idx],
                Z_batch,
                keys_batch,
            )
            # ensure computation is complete
            out.block_until_ready()
            batch_time = time.time() - batch_start
            
            # warn if batch took suspiciously long (likely recompilation)
            if i == 0 and batch_time > 1.0:
                logger.warning(
                    f"First batch took {batch_time:.2f}s - possible JAX recompilation. "
                    f"Consider precompiling with expected batch sizes."
                )
            elif i > 0 and batch_time > 0.5:
                logger.warning(
                    f"Batch {i} took {batch_time:.2f}s - possible JAX recompilation "
                    f"due to shape change or device switch."
                )

            all_yhats.append(np.asarray(out, dtype=np.float32))
            if collect_in_indices is not None:
                all_collected_in.append(
                    np.asarray(fullout[:, collect_in_indices], dtype=np.float32)
                )
            if collect_out_indices is not None:
                all_collected_out.append(
                    np.asarray(fullout[:, collect_out_indices], dtype=np.float32)
                )

        yhat = np.concatenate(all_yhats, axis=0, dtype=np.float32)[:n_samples]
        collected_in = (
            np.concatenate(all_collected_in, axis=0, dtype=np.float32)[:n_samples]
            if collect_in_indices is not None
            else None
        )
        collected_out = (
            np.concatenate(all_collected_out, axis=0, dtype=np.float32)[:n_samples]
            if collect_out_indices is not None
            else None
        )

        return yhat, (collected_in, collected_out)

    def prepare_collection_indices(
        self,
        collection_points: Optional[List[NodeSpec]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        prepare indices for collection points

        parameters:
            collection_points: list of points to collect data from

        returns:
            tuple of (collect_in_indices, collect_out_indices`)`
        """
        if collection_points is None:
            return [], []

        collect_out_indices = []
        collect_in_indices = []
        if collection_points:
            for point in collection_points:
                (collect_in_idx, _), (collect_out_idx, _) = self.get_node_indices(
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
    ) -> list[np.ndarray]:
        # TODO: when we use different collection points than the regular output,
        # we can't simply split by network_output_indices. we need to figure out the
        # shape of each output collection

        if self._stack is None:
            raise ValueError("stack not built")

        return self._stack.split_stack_outputs_per_network(np.asarray(yhat), max_samples)

    def visualize_stack(self, output_path='/tmp/stackviz.html'):
        """visualize the compute stack as html"""
        import biocomptools.toollib.stackviz as sv

        sv.save_stackviz(self._stack, output_path)
        logger.debug(f"saved stack visualization to {output_path}")
