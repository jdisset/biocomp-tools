from typing import List, Optional, Union, Dict, Any, Tuple, Literal, TypeVar, Annotated, Callable
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
import numpy as np
import jax
import jax.numpy as jnp
from biocomp.utils import ArbitraryModel, tree_to_jax
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
from biocomptools.toollib.datasources import DataSource, make_pretty_input_names
from biocomptools.modelmodel import NetworkModel
from biocomptools.logging_config import get_logger
import biocomp.plotutils as pu

logger = get_logger(__name__)


def validate_predict_at(v):
    if isinstance(v, np.ndarray):
        return [v]
    return v


def validate_ground_truth(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [v]
    return v


def validate_input_order(v):
    if v is None:
        return None

    if isinstance(v, (list, tuple, np.ndarray)):
        v = np.asarray(v)
        if v.ndim == 1:
            return [v.tolist()]
        elif v.ndim == 2:
            return v.tolist()
    return v


class NodeSpec(BaseModel):
    network_id: int
    node_id: int
    shape: Optional[tuple] = None  # If None, will be inferred from the stack


def get_node_indices(stack, network_id, node_id) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Gets the input and output indices for a virtual node in the compute stack."""
    layer_id, n_id = stack.node_map[(network_id, node_id)]
    layer = stack.layers[layer_id]
    node = layer.nodes[n_id]

    input_indices = []
    if layer.f_input_shapes:  # Skip for input layer
        for input_slot in range(len(layer.f_input_shapes)):
            start_idx = stack.get_node_input_start_index(node, input_slot)
            input_shape = layer.f_input_shapes[input_slot]
            length = int(np.prod(input_shape))
            input_indices.append(np.arange(start_idx, start_idx + length))

    output_indices = []
    for output_slot in range(len(layer.f_out_shapes)):
        start_idx = stack.get_node_output_start_index(node, output_slot)
        output_shape = layer.f_out_shapes[output_slot]
        length = int(np.prod(output_shape))
        output_indices.append(np.arange(start_idx, start_idx + length))

    return input_indices, output_indices


def make_pretty_input_names(ratios, ordered_input_names, name_lookup=None):
    if name_lookup is None:
        name_lookup = {
            'mNeonGreen': 'mNG',
            'PgU': 'Pgu',
        }

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

    return names


class NetworkPrediction(DataSource):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    network_model: NetworkModel
    predict_at: Annotated[Union[np.ndarray, List[np.ndarray]], BeforeValidator(validate_predict_at)]
    ground_truth: Annotated[
        Optional[Union[np.ndarray, List[Optional[np.ndarray]]]],
        BeforeValidator(validate_ground_truth),
    ] = None
    input_order: Annotated[Optional[List[List[int]]], BeforeValidator(validate_input_order)] = None

    # Node injection/collection configuration
    injection_points: Optional[List[NodeSpec]] = None
    collection_points: Optional[List[NodeSpec]] = None
    injection_values: Optional[List[np.ndarray]] = None

    # Computation settings
    max_evals: int = 300000
    z: Union[Literal['uniform'], float] = 'uniform'
    seed: int = 0
    use_output_as_input: bool = False

    _yhats: Optional[np.ndarray] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        if len(self.predict_at) != len(self.network_model.network):
            raise ValueError(
                f"Number of predict_at arrays ({len(self.predict_at)}) "
                f"does not match number of networks ({len(self.network_model.network)})"
            )

        if self.ground_truth is not None:
            if len(self.ground_truth) != len(self.predict_at):
                raise ValueError(
                    f"Number of ground truth arrays ({len(self.ground_truth)}) "
                    f"does not match number of predict_at arrays ({len(self.predict_at)})"
                )
        else:
            self.ground_truth = [None] * len(self.predict_at)

        # shuffle predict_at (and ground_truth)
        new_predict_at = []
        new_gt = []
        for x, gt in zip(self.predict_at, self.ground_truth):
            order = np.random.permutation(len(x))
            new_predict_at.append(x[order])
            new_gt.append(gt[order] if gt is not None else None)
        self.predict_at = new_predict_at
        self.ground_truth = new_gt

        if self.injection_points is not None:
            if self.injection_values is None:
                raise ValueError("injection_values must be provided when injection_points is set")
            if len(self.injection_points) != len(self.injection_values):
                raise ValueError("Number of injection points must match number of injection values")

        self._aligned_x, self._aligned_ground_truth = self._prepare_inputs()

    def _prepare_inputs(self) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
        """Prepare inputs by padding or truncating to the same length."""
        max_prediction_length = max(len(x) for x in self.predict_at)

        if self.max_evals < 0:
            self.max_evals = max_prediction_length

        self.max_evals = min(self.max_evals, max_prediction_length)

        aligned_predict_at = []
        aligned_ground_truth = []

        for x, gt in zip(self.predict_at, self.ground_truth):
            if len(x) < self.max_evals:
                zeros = np.zeros((self.max_evals - len(x), x.shape[1]))
                aligned_predict_at.append(np.vstack([x, zeros]))
                if gt is not None:
                    gtzeros = np.zeros((self.max_evals - len(x), gt.shape[1]))
                    aligned_ground_truth.append(np.vstack([gt, gtzeros]))
                else:
                    aligned_ground_truth.append(None)
            else:
                aligned_predict_at.append(x[: self.max_evals])
                aligned_ground_truth.append(gt[: self.max_evals] if gt is not None else None)

        return aligned_predict_at, aligned_ground_truth

    def _get_injection_indices(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get input and output indices for injection points."""
        if self.injection_points is None:
            return [], []

        all_input_indices = []
        all_output_indices = []

        for point in self.injection_points:
            input_indices, output_indices = get_node_indices(
                self.network_model._stack, point.network_id, point.node_id
            )
            all_input_indices.extend(input_indices)
            all_output_indices.extend(output_indices)

        return all_input_indices, all_output_indices

    def _get_collection_indices(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get input and output indices for collection points."""
        if self.collection_points is None:
            # Default to collecting final outputs
            return [], [np.arange(self.network_model._stack.total_nb_of_outputs)]

        all_input_indices = []
        all_output_indices = []

        for point in self.collection_points:
            input_indices, output_indices = get_node_indices(
                self.network_model._stack, point.network_id, point.node_id
            )
            all_input_indices.extend(input_indices)
            all_output_indices.extend(output_indices)

        return all_input_indices, all_output_indices

    def _split_yhat_per_network(self, yhat: np.ndarray):
        """Takes the whole stack output and returns a list of per-network outputs"""
        self._x = []
        self._yhats = []
        self._gtruths = []

        logger.debug(f"Going to split a full yhat of shape {yhat.shape}")

        output_start_id = 0
        for i, x in enumerate(self.predict_at):
            _, output_shapes = self.network_model._stack.get_network_output_indices(i)
            assert isinstance(output_shapes, list)
            outputs = []
            for output_shape in output_shapes:
                nout = np.prod(output_shape)
                output = yhat[:, output_start_id : output_start_id + nout].reshape(
                    -1, *output_shape
                )
                outputs.append(output)
                output_start_id += nout

            # assumes all outputs are of same shape
            assert all(output.shape == outputs[0].shape for output in outputs)
            network_i_outputs = np.concatenate(outputs, axis=1)

            assert (
                network_i_outputs.shape[0] == self.max_evals
            ), f"Expected {self.max_evals} but got {len(network_i_outputs)}"

            # truncate to remove padding
            network_i_outputs = network_i_outputs[: min(len(x), self.max_evals)]

            self._yhats.append(network_i_outputs)
            if self.ground_truth is not None and self.ground_truth[i] is not None:
                self._gtruths.append(self._aligned_ground_truth[i][: len(network_i_outputs)])
            else:
                self._gtruths.append(None)

            self._x.append(x[: min(len(x), self.max_evals)])

    def compute_predictions(self, key=None):
        """Compute predictions for all inputs."""
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        assert self.network_model._params is not None

        num_z = self.network_model._params["global/number_of_quantile_variables"]
        params = tree_to_jax(self.network_model._params)
        npoints = self.max_evals

        # prepare Z values
        if self.z == 'uniform':
            z_values = jax.random.uniform(key, (npoints, int(num_z)))
        else:
            z_values = jnp.ones((npoints, int(num_z))) * self.z

        stack_inputs = np.column_stack(self._aligned_x)

        # prepare injections if any
        injection_in_indices, injection_out_indices = self._get_injection_indices()
        collection_in_indices, collection_out_indices = self._get_collection_indices()
        injection_values = None
        if self.injection_values is not None:
            injection_values = np.concatenate(
                [v.reshape(npoints, -1) for v in self.injection_values], axis=1
            )
        overwrite_at = None
        if injection_in_indices:
            overwrite_at = jnp.concatenate([arr for arr in injection_in_indices])

        keys = jax.random.split(key, npoints)
        out, (grads, fullout) = self.network_model._batch_apply(
            params, stack_inputs, z_values, keys, injection_values, overwrite_at
        )

        self._split_yhat_per_network(out)

        # store collected values
        self._collected_inputs = []
        self._collected_outputs = []

        for in_indices, out_indices in zip(collection_in_indices, collection_out_indices):
            if len(in_indices) > 0:
                self._collected_inputs.append(fullout[:, in_indices])
            if len(out_indices) > 0:
                self._collected_outputs.append(fullout[:, out_indices])

    def _calculate_prediction_stats(self, yhat, gt, output_pos) -> Dict[str, float]:
        """Calculate statistics for the predictions in latent space."""
        latent_yhats = np.asarray(self.network_model.model.rescaler.fwd(yhat))
        latent_yhat = latent_yhats[:, output_pos]

        stats = {
            'samples': yhat.shape[0],
            'mean': float(latent_yhat.mean()),
            'std': float(latent_yhat.std()),
            'min': float(latent_yhat.min()),
            'max': float(latent_yhat.max()),
        }

        if gt is not None:
            latent_gt = np.asarray(self.network_model.model.rescaler.fwd(gt))
            stats['mse'] = float(np.mean((latent_yhat - latent_gt.flatten()) ** 2))
            stats['rmse'] = float(np.sqrt(stats['mse']))

            # Debug high RMSE cases
            if stats['rmse'] > 0.18:
                self._debug_high_rmse(latent_gt, latent_yhats, latent_yhat)

        return stats

    def _debug_high_rmse(
        self, latent_gt: np.ndarray, latent_yhats: np.ndarray, latent_yhat: np.ndarray
    ) -> None:
        """Save debug information for high RMSE cases."""
        debug_data = {
            'latent_gt.npy': latent_gt,
            'latent_yhats.npy': latent_yhats,
            'latent_yhat.npy': latent_yhat,
        }
        for filename, data in debug_data.items():
            with open(f'/tmp/prediction_{filename}', 'wb') as f:
                np.save(f, data)

    def _prepare_network_metadata(
        self, network_idx: int, prediction_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Prepare metadata for a network's plot data."""
        network = self.network_model.network[network_idx]
        network_info = network.get_info()

        metadata = {
            'source_type': 'prediction',
            'seed': self.seed,
            'model_signature': self.network_model.model.signature(),
            'network': network,
            'network_info': network_info,
            'built_network': network._network,
            'n_predictions': len(self.predict_at[network_idx]),
            'network_index': network_idx,
            'prediction_stats': prediction_stats,
        }

        return metadata

    def _get_network_data(
        self, network_idx: int
    ) -> Callable[[PlotData], Tuple[np.ndarray, np.ndarray]]:
        """Create a data getter function for a specific network."""

        def get_XY(pdata: PlotData) -> Tuple[np.ndarray, np.ndarray]:
            logger.debug(
                f"Getting XY for network {network_idx} with model {self.network_model.signature()}"
            )

            # Ensure predictions are computed
            if not hasattr(self, '_yhats') or self._yhats is None:
                logger.debug(f"Computing predictions with model {self.network_model.signature()}")
                self.compute_predictions()

            assert self._x is not None
            assert self._yhats is not None
            assert self._gtruths is not None

            x = self._x[network_idx]
            yhat = self._yhats[network_idx]
            gt = self._gtruths[network_idx]
            network = self.network_model.network[network_idx].network

            # Calculate prediction statistics
            _, output_pos, _, _ = get_reordered_protein_names(network)
            prediction_stats = self._calculate_prediction_stats(yhat, gt, output_pos)

            # Add stats to plot data metadata
            pdata.metadata['prediction_stats'] = prediction_stats

            if self.use_output_as_input:
                return yhat, yhat
            return x, yhat

        return get_XY

    def get_data_lazy(self) -> List[LazyPlotData]:
        """Get lazy plot data for all networks."""
        logger.debug(f"Getting data lazily for model {self.network_model.signature()}")

        plot_data_list = []

        if self.input_order is None:
            input_orders = [None] * len(self.network_model.network)
        else:
            input_orders = self.input_order
            if len(input_orders) == 1:
                input_orders = input_orders * len(self.network_model.network)

        for i, network in enumerate(self.network_model.network):
            metadata = self._prepare_network_metadata(i, {})

            plot_data = pu.extract_lazy_plot_data_from_network(
                network.network,
                self._get_network_data(i),
                input_order=input_orders[i],
                metadata=metadata,
            )

            # Add pretty input names
            plot_data.metadata['pretty_inputs'] = make_pretty_input_names(
                metadata['network_info']['cotx'], plot_data.input_names
            )

            plot_data_list.append(plot_data)

        return plot_data_list

    def get_data(self) -> List[PlotData]:
        """Get concrete plot data by evaluating lazy plot data."""
        plot_data_list = self.get_data_lazy()
        for plot_data in plot_data_list:
            plot_data.set_xy()
        return plot_data_list
