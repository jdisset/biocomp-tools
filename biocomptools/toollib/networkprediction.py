from pydantic.functional_validators import BeforeValidator
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Any, Optional, List, Union, Dict, Annotated, Literal, Tuple, TypeAlias, Callable
import numpy as np
import jax.numpy as jnp
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
from biocomp.datautils import DataRescaler, IdentityRescaler
from biocomptools.toollib.common import make_pretty_input_names
from biocomptools.modelmodel import NetworkModel, BiocompModel, NodeSpec
from biocomptools.logging_config import get_logger
import biocomp.parameters as pr
from biocomptools.toollib.datasources import DataSource
from biocomp.plotting.plotting_core import knn_stats, build_tree
from pathlib import Path
import time

# from concurrent.futures import ProcessPoolExecutor as PoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed
from functools import partial

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


def reconstruct_from_flat(flat_values, shapes):
    """reconstruct a list of arrays from flat values and shapes"""
    result = []
    offset = 0
    for shape in shapes:
        n = np.prod(shape)
        result.append(flat_values[:, offset : offset + n].reshape(-1, *shape))
        offset += n
    return result


def make_hypercube(ndim: int, res: int = 100, xmin: float = 0, xmax: float = 1) -> NdArray:
    """
    Create a hypercube grid of points in n dimensions.
    """
    assert ndim > 0, "ndim must be greater than 0"
    assert res > 0, "res must be greater than 0"
    grid = np.meshgrid(*[np.linspace(xmin, xmax, res) for _ in range(ndim)])
    return np.vstack([g.ravel() for g in grid]).T


def validate_predict_at(v: Any) -> List[NdArray]:
    """convert single numpy array to list of arrays"""
    if isinstance(v, NdArray):
        return [np.asarray(v, dtype=np.float32)]
    if isinstance(v, list) and all(isinstance(x, NdArray) for x in v):
        return [np.asarray(x, dtype=np.float32) for x in v]
    return v


def validate_ground_truth(v: Any) -> Optional[List[Optional[NdArray]]]:
    """convert single numpy array to list of arrays or none to list of none"""
    if v is None:
        logger.warning("ground_truth is None, will not compare predictions to ground truth")
        return None
    if isinstance(v, NdArray):
        logger.warning(f"ground_truth is a single array of shape {v.shape}, converting to list")
        aslist = [np.asarray(v, dtype=np.float32)]
        logger.info(f"ground_truth converted to list of shapes {[x.shape for x in aslist]}")
    return v


def validate_input_order(v: Any) -> Optional[List[List[int]]]:
    """convert list of ints to list of lists of ints"""
    if v is None:
        return None
    if isinstance(v, list) and all(isinstance(x, int) for x in v):
        return [v]
    return v


# static function for parallel processing
def _calculate_single_network_stats(
    network_idx: int,
    yhat: NdArray,
    gt: Optional[NdArray],
    x: NdArray,
    dependent_output_pos: Union[int, List[int]],
    nb_points_in_eval: int,
    rescaler,
    gridstats_params: Dict[str, Any],
    network_info: Dict[str, Any],
    enable_gridstats: bool = True,
) -> Dict[str, Any]:
    """calculate statistics for a single network (used in parallel processing)"""
    latent_yhats = np.asarray(rescaler.fwd(yhat), dtype=np.float32)
    # ensure dependent_output_pos is always a list for consistent indexing
    if isinstance(dependent_output_pos, int):
        dependent_output_pos = [dependent_output_pos]
    latent_yhat = latent_yhats[:, dependent_output_pos]

    # check for empty output - this can happen if output_pos selects invalid columns
    if latent_yhat.size == 0:
        result = {
            'xp_name': network_info.get('xp_name'),
            'recipe_name': network_info.get('recipe_name'),
            'network_name': network_info.get('network_name', f"Network_{network_idx}"),
            'eval_npoints': nb_points_in_eval,
            'samples': yhat.shape[0],
            'mse': None,
            'rmse': None,
            'latent_mean': np.nan,
            'latent_std': np.nan,
            'latent_min': np.nan,
            'latent_max': np.nan,
            'error': 'Empty output array - invalid dependent_output_pos selection',
        }
        # include extra_prediction_info if available
        if 'extra_prediction_info' in network_info:
            result['extra_prediction_info'] = network_info['extra_prediction_info']
        return result

    network_stats = {
        'xp_name': network_info.get('xp_name'),
        'recipe_name': network_info.get('recipe_name'),
        'network_name': network_info.get('network_name', f"Network_{network_idx}"),
        'eval_npoints': nb_points_in_eval,
        'samples': yhat.shape[0],  # number of actual prediction points used
        'mse': None,
        'rmse': None,
        'latent_mean': float(latent_yhat.mean()),
        'latent_std': float(latent_yhat.std()),
        'latent_min': float(latent_yhat.min()),
        'latent_max': float(latent_yhat.max()),
    }

    # include extra_prediction_info if available
    if 'extra_prediction_info' in network_info:
        network_stats['extra_prediction_info'] = network_info['extra_prediction_info']

    # add comparison stats if ground truth available
    if gt is not None:
        latent_x = rescaler.fwd(x)
        latent_gt = np.asarray(rescaler.fwd(gt), dtype=np.float32)
        # Handle dimension mismatch between gt and yhat. This can happen when:
        # - force_single_output=True was applied to gt, reducing it to 1 column
        # - yhat has multiple outputs that were sliced with dependent_output_pos
        if latent_gt.shape[1] > 1:
            # gt has multiple columns, apply the same slicing as yhat
            latent_gt = latent_gt[:, dependent_output_pos]
        elif latent_yhat.shape[1] > 1 and latent_gt.shape[1] == 1:
            # gt was pre-transformed to single column, slice yhat to match
            # use only the first column of yhat for comparison
            latent_yhat = latent_yhat[:, :1]
        # Now dimensions should match
        assert latent_gt.shape[1] == latent_yhat.shape[1], (
            f"After dimension adjustment, latent_gt.shape[1]={latent_gt.shape[1]} "
            f"!= latent_yhat.shape[1]={latent_yhat.shape[1]}"
        )

        # Debug traces for validation investigation
        from biocomptools.logging_config import get_logger

        debug_logger = get_logger(__name__)

        debug_logger.debug(f"Network {network_info.get('network_name', f'Network_{network_idx}')}:")
        debug_logger.debug(f"  yhat shape: {yhat.shape}, gt shape: {gt.shape}")
        debug_logger.debug(
            f"  latent_yhat shape: {latent_yhat.shape}, latent_gt shape: {latent_gt.shape}"
        )
        debug_logger.debug(
            f"  yhat stats: mean={yhat.mean():.6f}, std={yhat.std():.6f}, min={yhat.min():.6f}, max={yhat.max():.6f}"
        )
        debug_logger.debug(
            f"  gt stats: mean={gt.mean():.6f}, std={gt.std():.6f}, min={gt.min():.6f}, max={gt.max():.6f}"
        )
        debug_logger.debug(
            f"  latent_yhat stats: mean={latent_yhat.mean():.6f}, std={latent_yhat.std():.6f}"
        )
        debug_logger.debug(
            f"  latent_gt stats: mean={latent_gt.mean():.6f}, std={latent_gt.std():.6f}"
        )
        debug_logger.debug(f"  dependent_output_pos: {dependent_output_pos}")

        # BUGFIX: Ensure gt and yhat have matching shapes by extracting same columns
        if latent_gt.shape[1] > latent_yhat.shape[1]:
            debug_logger.debug(
                f"  gt has more columns ({latent_gt.shape[1]}) than yhat ({latent_yhat.shape[1]}), extracting dependent outputs from gt"
            )
            latent_gt = latent_gt[:, dependent_output_pos]
            debug_logger.debug(f"  after extraction: latent_gt shape: {latent_gt.shape}")

        mse = float(np.mean((latent_yhat - latent_gt) ** 2))
        rmse = float(np.sqrt(mse))
        network_stats['mse'] = float(mse)
        network_stats['rmse'] = rmse

        debug_logger.debug(f"  computed MSE: {mse:.6f}, RMSE: {rmse:.6f}")

        if enable_gridstats:
            grid_stats = _calculate_grid_stats(latent_yhat, latent_gt, latent_x, gridstats_params)
            network_stats.update(grid_stats)

    return network_stats


def _calculate_grid_stats(
    latent_yhat: NdArray,
    latent_gt: NdArray,
    latent_x: NdArray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    calculate grid statistics (extracted for parallel processing)
    """
    grid = make_hypercube(
        latent_x.shape[1],
        res=params['hypercube_res'],
        xmin=params['hypercube_min'],
        xmax=params['hypercube_max'],
    )

    latent_yhat = latent_yhat.reshape(-1, 1) if latent_yhat.ndim == 1 else latent_yhat
    latent_gt = latent_gt.reshape(-1, 1) if latent_gt.ndim == 1 else latent_gt

    # mask out NaN or inf values
    valid_x_mask = np.all(np.isfinite(latent_x), axis=1)
    valid_latent_x = np.asarray(latent_x[valid_x_mask])
    valid_latent_yhat = np.asarray(latent_yhat[valid_x_mask])
    valid_latent_gt = np.asarray(latent_gt[valid_x_mask])

    tree = build_tree(valid_latent_x)
    iw, gt_stdev, gt_mean = knn_stats(
        grid,
        valid_latent_gt,
        tree=tree,
        stats=['iw', 'std', 'mean'],
        k=params['k'],
        radius=params['radius'],
        min_points=params['min_points'],
    )  # type: ignore
    yhat_stdev, yhat_mean = knn_stats(
        grid,
        valid_latent_yhat,
        iw=iw,
        stats=['std', 'mean'],
        k=params['k'],
        radius=params['radius'],
        min_points=params['min_points'],
    )  # type: ignore

    grid_mse = np.nanmean((yhat_mean - gt_mean) ** 2)
    grid_rmse = np.sqrt(grid_mse)

    grid_gt_var = np.nanvar(gt_mean)
    EPSILON = 1e-9
    grid_r_squared = 1 - (grid_mse / (grid_gt_var + EPSILON))

    # KL divergence
    # All operations here are element-wise. kl_divergences will be a
    # (n_grid_points, D) array. np.nanmean averages them all into a single scalar,
    # representing the average KL divergence across all dimensions and grid points.
    safe_gt_stdev = np.maximum(gt_stdev, EPSILON)
    safe_yhat_stdev = np.maximum(yhat_stdev, EPSILON)
    log_term = np.log(safe_yhat_stdev / safe_gt_stdev)
    numerator_term = safe_gt_stdev**2 + (gt_mean - yhat_mean) ** 2
    denominator_term = 2 * safe_yhat_stdev**2
    kl_divergences = log_term + numerator_term / denominator_term - 0.5
    kl_divergences = np.maximum(kl_divergences, 0.0)
    kl_mean = np.nanmean(kl_divergences)
    kl_similarities = np.exp(-kl_divergences)
    grid_kl_similarity = np.nanmean(kl_similarities) * 100

    stats = {
        'grid_gt_var': grid_gt_var,
        'grid_mse': grid_mse,
        'grid_rmse': grid_rmse,
        'grid_kl': kl_mean,
        'grid_kl_similarity': grid_kl_similarity,
        'grid_r_squared': grid_r_squared,
    }

    return stats


class NetworkPrediction(DataSource):
    """
    Performs predictions using a networkmodel and prepares data for plotting

    responsible for:
    - preparing input data for prediction
    - making predictions using the network model
    - comparing predictions to ground truth
    - generating plot data for visualization
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    predict_at: Annotated[Union[NdArray, List[NdArray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Annotated[Optional[List[List[int]]], BeforeValidator(validate_input_order)] = None
    ground_truth: Annotated[
        Optional[Union[NdArray, List[Optional[NdArray]]]],
        BeforeValidator(validate_ground_truth),
    ] = None

    per_prediction_info: Optional[list[Dict[str, Any]]] = None  # metadata for each prediction

    collection_points: Optional[List[NodeSpec]] = None  # collection points to examine inner nodes

    seed: int = 0
    max_evals: int = 300000
    use_output_as_input: bool = False
    z_value: Union[Literal['uniform'], float] = 'uniform'
    disable_variational: bool = True
    device: Literal['cpu', 'gpu'] = 'cpu'  # device preference for predictions

    save_csv_to: Optional[str] = None  # save prediction statistics to a CSV file

    already_latent: bool = False  # no need to rescale if the input data is already in latent space

    enable_gridstats: bool = True  # enable grid statistics calculation
    gridstats_hypercube_res: int = 40  # resolution for hypercube grid
    gridstats_hypercube_min: float = 0.0  # minimum value for hypercube grid
    gridstats_hypercube_max: float = 1.0  # maximum value for hypercube grid
    gridstats_k: int = 400
    gridstats_radius: float = 0.1
    gridstats_min_points: int = 40

    shuffle_inputs: bool = True  # shuffle inputs before prediction

    n_stats_workers: int = 8  # number of workers for parallel processing of statistics

    verbose: bool = False  # print prediction statistics to the console

    _yhats: Optional[List[NdArray]] = None
    _x: Optional[List[NdArray]] = None
    _gtruths: Optional[List[Optional[NdArray]]] = None
    _aligned_x: Optional[List[NdArray]] = None
    _aligned_ground_truth: Optional[List[Optional[NdArray]]] = None
    _network_stats: Optional[List[Dict[str, Any]]] = None

    def model_post_init(self, *args, **kwargs):
        """initialize the model after validation"""
        super().model_post_init(*args, **kwargs)

        logger.debug(
            f"predict_at: {len(self.predict_at)} arrays, shapes: {[x.shape for x in self.predict_at]}"
        )

        assert isinstance(self.network_model.network, list)

        # validate number of networks matches input data
        if len(self.predict_at) != len(self.network_model.network):
            raise ValueError(
                f"number of predict_at arrays ({len(self.predict_at)}) "
                f"does not match number of networks ({len(self.network_model.network)})"
            )

        # validate and fix input dimensions for each network
        fixed_predict_at = []
        for i, (x, net) in enumerate(zip(self.predict_at, self.network_model.network)):
            expected_inputs = net.nb_inputs
            actual_inputs = x.shape[1] if x.ndim > 1 else 1
            if actual_inputs > expected_inputs:
                logger.warning(
                    f"Network {i} ({net.name}): predict_at has {actual_inputs} columns but "
                    f"network expects {expected_inputs} inputs. Truncating to first {expected_inputs} columns. "
                    f"(This typically happens when force_single_output=True reshapes the data for plotting.)"
                )
                fixed_predict_at.append(x[:, :expected_inputs])
            elif actual_inputs < expected_inputs:
                raise ValueError(
                    f"Network {i} ({net.name}): predict_at has {actual_inputs} columns but "
                    f"network expects {expected_inputs} inputs."
                )
            else:
                fixed_predict_at.append(x)
        self.predict_at = fixed_predict_at

        if self.ground_truth is None:
            self.ground_truth = [None] * len(self.predict_at)
        else:
            if len(self.ground_truth) != len(self.predict_at):
                raise ValueError(
                    f"number of ground truth arrays ({len(self.ground_truth)}) "
                    f"does not match number of predict_at arrays ({len(self.predict_at)})"
                )

        # shuffle inputs with fixed random seed
        if self.shuffle_inputs:
            self._shuffle_inputs()

        # prepare aligned inputs
        self._aligned_x, self._aligned_ground_truth = self._prepare_inputs()
        logger.debug(f"aligned_x shapes: {[x.shape for x in self._aligned_x]}")
        logger.debug(
            f"aligned_ground_truth shapes: {[None if gt is None else gt.shape for gt in self._aligned_ground_truth]}"
        )

    def with_shared_from_model(self, model: BiocompModel) -> 'NetworkPrediction':
        """create a new networkprediction with a different model"""
        logger.debug(f"creating new networkprediction with model {model.signature=}")

        # create new instance with updated network model
        new = self.model_copy(update={'network_model': self.network_model.with_model(model)})
        new._yhats = None  # clear the cache
        new._network_stats = None  # clear the stats cache

        return new

    def with_csv_output_path(self, path: str) -> 'NetworkPrediction':
        """set path to save prediction statistics to a CSV file"""
        return self.model_copy(update={'save_csv_to': path})

    def with_device(self, device: Literal['cpu', 'gpu']) -> 'NetworkPrediction':
        """set device preference for predictions"""
        return self.model_copy(update={'device': device})

    def _shuffle_inputs(self):
        """shuffle inputs and ground truth with the same random order"""
        # set random seed for reproducibility
        rng = np.random.RandomState(self.seed)

        new_predict_at = []
        new_gt = []
        for i, (x, gt) in enumerate(zip(self.predict_at, self.ground_truth)):
            order = rng.permutation(len(x))
            new_predict_at.append(x[order])
            new_gt.append(gt[order] if gt is not None else None)
        self.predict_at = new_predict_at
        self.ground_truth = new_gt

    def _prepare_inputs(self) -> Tuple[List[NdArray], List[Optional[NdArray]]]:
        """prepare inputs by padding or truncating to the same length"""
        max_prediction_length = max(len(x) for x in self.predict_at)
        logger.debug(f"max_prediction_length across networks: {max_prediction_length}")

        effective_max_evals = min(
            self.max_evals if self.max_evals > 0 else max_prediction_length, max_prediction_length
        )
        logger.debug(f"effective_max_evals: {effective_max_evals}")

        aligned_predict_at = []
        aligned_ground_truth = []

        for i, (x, gt) in enumerate(zip(self.predict_at, self.ground_truth)):
            logger.debug(
                f"aligning network {i}: x.shape={x.shape}, gt={None if gt is None else gt.shape}"
            )

            if len(x) < effective_max_evals:
                # pad with zeros if shorter than desired length
                f"padding prediction queries for network {i} from {len(x)} to {effective_max_evals} points"
                zeros = np.zeros((effective_max_evals - len(x), x.shape[1]))
                padded_x = np.vstack([x, zeros])
                aligned_predict_at.append(padded_x)

                if gt is not None:
                    gtzeros = np.zeros((effective_max_evals - len(x), gt.shape[1]))
                    padded_gt = np.vstack([gt, gtzeros])
                    aligned_ground_truth.append(padded_gt)
                else:
                    aligned_ground_truth.append(None)
            else:
                # truncate if longer than desired length
                logger.debug(
                    f"truncating prediction queries for network {i} from {len(x)} to {effective_max_evals} points"
                )
                truncated_x = x[:effective_max_evals]
                aligned_predict_at.append(truncated_x)

                if gt is not None:
                    truncated_gt = gt[:effective_max_evals]
                    aligned_ground_truth.append(truncated_gt)
                else:
                    aligned_ground_truth.append(None)

        return aligned_predict_at, aligned_ground_truth

    def compute_all_network_predictions(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ):
        """compute predictions for all networks"""

        start_time = time.time()

        # stack inputs from all networks
        assert isinstance(self._aligned_x, list)
        stacked_x = np.column_stack(self._aligned_x)

        effective_max_evals = len(self._aligned_x[0])

        predict_f = (
            self.network_model.predict
            if self.already_latent
            else self.network_model.predict_unscaled
        )

        if self.verbose:
            logger.debug(f"computing predictions with model {self.network_model.signature}")
            logger.debug(
                f"prediction params: seed={self.seed}, disable_variational={self.disable_variational}, z_value={self.z_value}, device={self.device}"
            )
            logger.debug(f"effective_max_evals: {effective_max_evals}")

        collect_in_idx = []
        collect_out_idx = []
        self._input_shapes = []
        self._output_shapes = []
        if self.collection_points is not None and len(self.collection_points) > 0:
            for collection_point in self.collection_points:
                (in_idx, input_shapes), (out_idx, output_shapes) = (
                    self.network_model.get_node_indices(
                        collection_point.network_id,
                        collection_point.node_id,
                    )
                )
                collect_in_idx.append(in_idx)
                collect_out_idx.append(out_idx)
                self._input_shapes.append(input_shapes)
                self._output_shapes.append(output_shapes)

        stacked_yhats, (self._collected_in, self._collected_out) = predict_f(
            stacked_x,
            key=self.seed,
            z_value=self.z_value,
            disable_variational=self.disable_variational,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
            collect_in_indices=np.concatenate(collect_in_idx).flatten() if collect_in_idx else None,
            collect_out_indices=np.concatenate(collect_out_idx).flatten()
            if collect_out_idx
            else None,
            device=self.device,
        )

        prediction_time = time.time() - start_time
        logger.info(f"Network predictions completed in {prediction_time:.2f} seconds")

        # split the outputs by network
        network_outputs = self.network_model.split_outputs_per_network(
            stacked_yhats, effective_max_evals
        )

        self._process_prediction_results(network_outputs, effective_max_evals)

        stats_start_time = time.time()
        self._network_stats = self._calculate_all_network_stats()
        stats_time = time.time() - stats_start_time
        logger.info(f"Network statistics calculated in {stats_time:.2f} seconds")

    def _process_prediction_results(self, network_outputs: List[NdArray], max_evals: int):
        """process and store prediction results"""
        self._x = []
        self._yhats = []
        self._gtruths = []
        assert isinstance(self.ground_truth, list)
        assert isinstance(self._aligned_ground_truth, list)

        for i, (network_output, x) in enumerate(zip(network_outputs, self.predict_at)):
            effective_max_evals = min(max_evals, len(x))

            # store prediction results
            truncated_output = network_output[:effective_max_evals]
            self._yhats.append(np.asarray(truncated_output, dtype=np.float32))

            # store ground truth if available
            if self.ground_truth[i] is not None:
                truncated_gt = self._aligned_ground_truth[i][:effective_max_evals]
                self._gtruths.append(np.asarray(truncated_gt, dtype=np.float32))
            else:
                self._gtruths.append(None)

            # store inputs
            truncated_x = x[:effective_max_evals]
            self._x.append(np.asarray(truncated_x, dtype=np.float32))

        # validate results
        assert len(self._x) == len(self.predict_at)
        assert len(self._x) == len(self.network_model.network)

    def _create_xy_function(
        self,
        network_idx: int,
        rescale_latent: bool = False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> Callable[[PlotData], Tuple[NdArray, NdArray]]:
        """create a function to get x and y data for a specific network"""

        def get_xy(pdata: PlotData) -> Tuple[NdArray, NdArray]:
            if self.verbose:
                print(
                    f"getting xy for network {network_idx} with model {self.network_model.signature}"
                )

            # compute predictions if not already computed
            if (
                not hasattr(self, '_yhats')
                or self._yhats is None
                or with_shared_params is not None
                or with_local_params is not None
            ):
                self.compute_all_network_predictions(
                    with_shared_params=with_shared_params,
                    with_local_params=with_local_params,
                )
                if self.save_csv_to:
                    self.save_csv()

            assert isinstance(self._x, list)
            assert isinstance(self._yhats, list)
            assert isinstance(self._gtruths, list)

            x = self._x[network_idx]
            yhat = self._yhats[network_idx]
            gt = self._gtruths[network_idx]

            if self.verbose:
                print(
                    f"x shape: {x.shape}, yhat shape: {yhat.shape}, gt: {None if gt is None else gt.shape}"
                )

            assert isinstance(self.network_model.network, list)

            network = self.network_model.network[network_idx]

            # get output position for this network (can be int for single output or list for multiple outputs)
            _, dependent_output_pos, _, _ = get_reordered_protein_names(network)
            if self.verbose:
                logger.info(f"dep_output_pos: {dependent_output_pos}")

            stats = self.get_network_stats()
            assert isinstance(stats, list) and (len(stats) == len(self.network_model.network))
            # add stats to metadata
            pdata.metadata['prediction_stats'] = stats[network_idx].copy()

            # apply inverse rescaling if requested and data is already latent
            if self.already_latent and rescale_latent:
                x = self.network_model.model.rescaler.inv(x)
                yhat = self.network_model.model.rescaler.inv(yhat)
                if self.verbose:
                    print("applied inverse rescaling to x and yhat")

            if self.use_output_as_input:
                if self.verbose:
                    print("using output as input, returning yhat as both x and y")
                return yhat, yhat
            else:
                if self.verbose:
                    print("using original input, returning x and yhat")
                return x, yhat

        return get_xy

    def _log_stats(
        self,
        network_idx: int,
        prediction_stats: Dict[str, Any],
        latent_gt: NdArray,
        latent_yhat: NdArray,
        latent_x: NdArray,
    ):
        if self.verbose:
            n_samples = 5

            assert len(latent_gt) == len(latent_yhat) == len(latent_x)
            sample_ids = np.random.choice(len(latent_gt), n_samples, replace=False)
            gt_sample = latent_gt[sample_ids]
            yhat_sample = latent_yhat[sample_ids]
            latentx_sample = latent_x[sample_ids]
            se_sample = (yhat_sample - gt_sample) ** 2

            original_yhat = self.network_model.model.rescaler.inv(yhat_sample)
            original_gt = self.network_model.model.rescaler.inv(gt_sample)
            unscaledx_sample = self.network_model.model.rescaler.inv(latentx_sample)
            unscaledx_sample = [tuple(np.round(x, 3) for x in xs) for xs in unscaledx_sample]
            latentx_sample = [tuple(np.round(x, 3) for x in xs) for xs in latentx_sample]
            import pandas as pd

            df = pd.DataFrame(
                {
                    'unscaled X': unscaledx_sample,
                    'unscaled gt': original_gt.tolist(),
                    'unscaled yhat': original_yhat.tolist(),
                    'latent X': latentx_sample,
                    'latent gt': gt_sample.tolist(),
                    'latent yhat': yhat_sample.tolist(),
                    'l2 error': se_sample.tolist(),
                }
            )

            logger.info(
                f"""network {network_idx} evaluated over {prediction_stats['samples']} samples:
                    - mse: {prediction_stats['mse']:.3f}
                    - rmse: {prediction_stats['rmse']:.3f}
                    - mean: {prediction_stats['latent_mean']:.3f}
                    - std: {prediction_stats['latent_std']:.3f}
                    - min: {prediction_stats['latent_min']:.3f}
                    - max: {prediction_stats['latent_max']:.3f}
                    """
            )
            logger.info(f"Random samples:\n{df.round(3).to_string()}")

    def _calculate_all_network_stats(self) -> List[Dict[str, Any]]:
        tasks = []
        for i, network in enumerate(self.network_model.network):
            network_name = getattr(network, 'name', f"Network_{i}")
            nb_points_in_eval = len(self.predict_at[i])

            yhat = self._yhats[i]
            gt = self._gtruths[i]
            x = self._x[i]

            _, dependent_output_pos, _, _ = get_reordered_protein_names(network)

            # Calculate dependent outputs for validation
            all_outputs = set(network.get_output_proteins())
            input_proteins = set(network.get_inverted_input_proteins())
            dependent_outputs = all_outputs - input_proteins
            n_dependent_outputs = len(dependent_outputs)

            network_info = {
                'network_name': network_name,
                'n_dependent_outputs': n_dependent_outputs,
            }

            # add extra_prediction_info if available
            if self.per_prediction_info is not None and i < len(self.per_prediction_info):
                network_info['extra_prediction_info'] = self.per_prediction_info[i]

            gridstats_params = {
                'hypercube_res': self.gridstats_hypercube_res,
                'hypercube_min': self.gridstats_hypercube_min,
                'hypercube_max': self.gridstats_hypercube_max,
                'k': self.gridstats_k,
                'radius': self.gridstats_radius,
                'min_points': self.gridstats_min_points,
            }

            tasks.append(
                {
                    'network_idx': i,
                    'yhat': yhat,
                    'gt': gt,
                    'x': x,
                    'dependent_output_pos': dependent_output_pos,
                    'nb_points_in_eval': nb_points_in_eval,
                    'rescaler': self.network_model.model.rescaler
                    if not self.already_latent
                    else IdentityRescaler(),
                    'gridstats_params': gridstats_params,
                    'network_info': network_info,
                    'enable_gridstats': self.enable_gridstats,
                }
            )

        all_stats = [None] * len(self.network_model.network)

        def get_info_dump(task, i):
            return {
                'n_networks': len(self.network_model.network),
                'shapes_yhats': [yh.shape for yh in self._yhats],
                'shapes_x': [x.shape for x in self._x],
                'input_order': self.input_order,
                'collection_points': self.collection_points,
                'task_network_idx': task['network_idx'],
                'task_network_name': task['network_info'].get('network_name', f"Network_{i}"),
                'task_yhat_shape': task['yhat'].shape,
                'task_gt_shape': None if task['gt'] is None else task['gt'].shape,
                'task_x_shape': task['x'].shape,
                'task_output_pos': task['dependent_output_pos'],
                'task_nb_points_in_eval': task['nb_points_in_eval'],
            }

        def process_and_log_stats(idx, task, result):
            all_stats[idx] = result
            if task.get('gt') is not None:
                rescaler = task['rescaler']
                latent_gt = rescaler.fwd(task['gt'])
                latent_yhat = rescaler.fwd(task['yhat'])
                if latent_gt.shape < latent_yhat.shape:
                    logger.debug(
                        f"ground truth shape {latent_gt.shape} is smaller than yhat shape {latent_yhat.shape}, "
                        "Assuming gt is only the dependent outputs."
                    )
                    # ensure dependent_output_pos is always a list for consistent indexing
                    output_pos = task['dependent_output_pos']
                    if isinstance(output_pos, int):
                        output_pos = [output_pos]
                    latent_yhat = latent_yhat[:, output_pos]
                latent_x = rescaler.fwd(task['x'])
                self._log_stats(idx, result, latent_gt, latent_yhat, latent_x)

        def handle_exception(idx, task, e):
            net_name = task['network_info'].get('network_name', f"Network_{idx}")
            logger.error(f"Error calculating stats for network {net_name}: {e}")
            logger.error(f"Task info: {get_info_dump(task, idx)}")
            logger.exception(e)
            all_stats[idx] = {'error': str(e)}

        if self.n_stats_workers > 1 and len(tasks) > 1:
            logger.info(
                f"Calculating network stats in parallel with {self.n_stats_workers} workers."
            )
            with PoolExecutor(max_workers=self.n_stats_workers) as executor:
                future_to_idx = {
                    executor.submit(_calculate_single_network_stats, **task): task['network_idx']
                    for task in tasks
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    task = tasks[idx]
                    try:
                        result = future.result()
                        process_and_log_stats(idx, task, result)
                    except Exception as e:
                        handle_exception(idx, task, e)

        else:
            if len(tasks) > 0:
                logger.info(
                    "Calculating network stats sequentially using the unified static method."
                )

            for task in tasks:
                idx = task['network_idx']
                try:
                    result = _calculate_single_network_stats(**task)
                    process_and_log_stats(idx, task, result)
                except Exception as e:
                    handle_exception(idx, task, e)

        return all_stats

    def get_network_stats(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ):
        """get statistics for all networks, computing if necessary"""
        if (
            not hasattr(self, '_network_stats')
            or self._network_stats is None
            or with_shared_params
            or with_local_params
        ):
            if (
                not hasattr(self, '_yhats')
                or self._yhats is None
                or with_shared_params
                or with_local_params
            ):
                self.compute_all_network_predictions(
                    with_shared_params=with_shared_params,
                    with_local_params=with_local_params,
                )
            else:
                # just calculate stats if predictions already exist
                self._network_stats = self._calculate_all_network_stats()

        return self._network_stats

    def get_shared_params(self) -> pr.ParameterTree:
        """get shared parameters from the network model"""
        return self.network_model.model.shared_params

    def get_local_params(self) -> pr.ParameterTree:
        """get local parameters from the network model"""
        # ensure the network model is built and parameters are initialized
        if (
            not hasattr(self.network_model, '_local_params')
            or self.network_model._local_params is None
        ):
            self.network_model.update_params()
        return self.network_model._local_params

    def save_csv(self):
        """save prediction statistics to a CSV file"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required to save prediction statistics to CSV")
            return

        stats = self.get_network_stats()
        df = pd.DataFrame(stats)

        # update df with content of self.extra_metadata
        for key, value in self.metadata.items():
            df[key] = value

        if self.verbose:
            logger.debug(f"prediction statistics DataFrame:\n{df.to_string()}")

        try:
            save_to = Path(self.save_csv_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_to, index=False)
            logger.info(f"saved prediction statistics to {save_to}")
        except Exception as e:
            logger.error(f"failed to save CSV to {self.save_csv_to}: {e}")

    def _normalize_input_order(self) -> List[Optional[List[int]]]:
        """normalize input_order to ensure it's consistent across networks"""
        assert self.network_model is not None
        assert isinstance(self.network_model.network, list)

        if self.input_order is None:
            logger.debug(
                f"input_order is None, using None for all {len(self.network_model.network)} networks"
            )
            return [None] * len(self.network_model.network)

        if isinstance(self.input_order, list):
            # single input order for all networks
            if all(isinstance(x, int) for x in self.input_order):
                logger.debug(f"using same input_order {self.input_order} for all networks")
                return [self.input_order] * len(self.network_model.network)

            if len(self.input_order) != len(self.network_model.network):
                # if it's a list of a single element, use it for all networks
                if len(self.input_order) == 1 and isinstance(self.input_order[0], list):
                    logger.debug(
                        f"input_order is a single list, using it for all {len(self.network_model.network)} networks"
                    )
                    return [self.input_order[0]] * len(self.network_model.network)
                raise ValueError(
                    f"input order list has {len(self.input_order)} items but there are "
                    f"{len(self.network_model.network)} networks"
                )
            return self.input_order

        raise ValueError(f"unexpected input_order type: {type(self.input_order)}")

    def get_data_lazy(
        self,
        rescale_latent=False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[LazyPlotData]:
        """get plot data in lazy evaluation mode"""
        logger.debug(f"getting data lazily for model {self.network_model.signature}")

        # handle collection points if provided
        if self.collection_points is not None and len(self.collection_points) > 0:
            return self._get_collection_data_lazy(with_shared_params, with_local_params)

        # otherwise, continue with normal network data
        plot_data_list = []
        input_order = self._normalize_input_order()
        logger.debug(f"normalized input_order: {input_order}")

        # create plot data for each network
        for i, network in enumerate(self.network_model.network):
            metadata = self._create_network_metadata(i, network)
            plot_data = self._extract_plot_data(
                i,
                network,
                input_order[i],
                metadata,
                rescale_latent=rescale_latent,
                with_shared_params=with_shared_params,
                with_local_params=with_local_params,
            )

            network_info = metadata['network_info']
            pretty_inputs = make_pretty_input_names(
                network_info['cotx'],
                plot_data.input_names,
            )
            plot_data.metadata['pretty_inputs'] = pretty_inputs

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} plot data objects")
        return plot_data_list

    def _get_collection_data_lazy(
        self,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[LazyPlotData]:
        """get plot data for collection points in lazy evaluation mode"""
        logger.debug(f"getting collection data lazily for model {self.network_model.signature}")

        assert self.collection_points is not None

        plot_data_list = []

        # compute index offsets for each collection point if needed
        if (
            not hasattr(self, '_collected_in')
            or self._collected_in is None
            or with_shared_params is not None
            or with_local_params is not None
        ):
            self.compute_all_network_predictions(
                with_shared_params=with_shared_params,
                with_local_params=with_local_params,
            )

        assert self._collected_in is not None
        assert len(self.collection_points) == len(self._input_shapes)
        assert self._collected_out is not None
        assert len(self.collection_points) == len(self._output_shapes)

        in_sizes, out_sizes = [], []
        for inshapes, outshapes in zip(self._input_shapes, self._output_shapes):
            in_sizes.append(np.sum([np.prod(s) for s in inshapes]))
            out_sizes.append(np.sum([np.prod(s) for s in outshapes]))
        in_offsets = np.cumsum([0] + in_sizes[:-1])
        out_offsets = np.cumsum([0] + out_sizes[:-1])

        collect_in_idx = [in_offsets[i] + np.arange(in_sizes[i]) for i in range(len(in_sizes))]
        collect_out_idx = [out_offsets[i] + np.arange(out_sizes[i]) for i in range(len(out_sizes))]

        # create plot data for each collection point
        for i, collection_point in enumerate(self.collection_points):

            def get_xy_fn(
                pdata: PlotData,
                collection_point_nodespec=collection_point,
                input_shapes=self._input_shapes[i],
                output_shapes=self._output_shapes[i],
                i=i,
            ) -> Tuple[NdArray, NdArray]:
                if (
                    not hasattr(self, '_collected_in')
                    or self._collected_in is None
                    or with_shared_params is not None
                    or with_local_params is not None
                ):
                    self.compute_all_network_predictions(
                        with_shared_params=with_shared_params,
                        with_local_params=with_local_params,
                    )
                assert self._collected_in is not None
                assert self._collected_out is not None

                flatx = self._collected_in[:, collect_in_idx[i]]
                flaty = self._collected_out[:, collect_out_idx[i]]

                pdata.metadata['collection_point_nodespec'] = collection_point_nodespec
                pdata.metadata['input_shapes'] = input_shapes
                pdata.metadata['output_shapes'] = output_shapes

                if self.verbose:
                    print(f"self._collected_in shape: {self._collected_in.shape}")
                    print(f"self._collected_out shape: {self._collected_out.shape}")
                    print(f"collection point = {collection_point_nodespec}")
                    print(f"collection point {i} collected_in_idx: {collect_in_idx[i]}")
                    print(f"collection point {i} collected_out_idx: {collect_out_idx[i]}")
                    print(f"collected node input shapes: {input_shapes}")
                    print(f"collected node output shapes: {output_shapes}")
                    print(f"flatx shape: {flatx.shape}")
                    print(f"flatx: {flatx}")
                    print(f"flaty shape: {flaty.shape}")
                    print(f"flaty: {flaty}")

                return flatx, flaty

            output_name = f"Node {collection_point.network_id}:{collection_point.node_id}"

            input_names = [f"In {i}" for i in range(len(self._input_shapes[i]))]

            metadata = self.metadata.copy()

            metadata.update(
                {
                    'datasource_type': 'collection',
                    'seed': self.seed,
                    'model_signature': self.network_model.model.signature,
                    'collection_point_index': i,
                    'network_id': collection_point.network_id,
                    'node_id': collection_point.node_id,
                }
            )

            plot_data = LazyPlotData(
                get_xy=get_xy_fn,
                input_names=input_names,
                output_name=output_name,
                metadata=metadata,
                disable_check_shapes=True,
            )

            plot_data_list.append(plot_data)

        logger.debug(f"returning {len(plot_data_list)} collection plot data objects")
        return plot_data_list

    def _create_network_metadata(self, network_idx: int, network) -> Dict[str, Any]:
        """create metadata dictionary for a network"""
        metadata = self.metadata.copy()

        from biocomp.network import generate_network_info

        network_info = generate_network_info(network)

        metadata.update(
            {
                'datasource_type': 'prediction',
                'seed': self.seed,
                'network': network.model_dump(),
                'model_signature': self.network_model.model.signature,
                'network_name': getattr(network, 'name', f"Network_{network_idx}"),
                'network_info': network_info,
                'network_index': network_idx,
                'prediction_settings': {
                    'use_output_as_input': self.use_output_as_input,
                    'z_value': self.z_value,
                    'disable_variational': self.disable_variational,
                    'max_evals': self.max_evals,
                    'n_predictions': len(self.predict_at[network_idx]),
                },
            }
        )

        if self.per_prediction_info is not None:
            assert len(self.per_prediction_info) >= network_idx
            metadata['extra_prediction_info'] = self.per_prediction_info[network_idx]
            if 'network_name' in metadata['extra_prediction_info']:
                assert (
                    metadata['extra_prediction_info']['network_name'] == metadata['network_name']
                ), (
                    f"network name mismatch: {metadata['extra_prediction_info']['network_name']} != {metadata['network_name']}"
                )

        return metadata

    def _extract_plot_data(
        self,
        network_idx: int,
        network,
        input_order: Optional[List[int]],
        metadata: Dict[str, Any],
        rescale_latent: bool = False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> LazyPlotData:
        """extract plot data from a network"""
        import biocomp.plotutils as pu

        # create function to get data for this network
        get_xy_fn = self._create_xy_function(
            network_idx,
            rescale_latent=rescale_latent,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
        )

        # extract plot data
        plot_data = pu.extract_lazy_plot_data_from_network(
            network,
            get_xy_fn,
            input_order=input_order,
            metadata=metadata,
        )

        logger.debug(
            f"extracted plot data for network {network_idx}: {plot_data.input_names=}, {plot_data.output_name=}"
        )

        return plot_data

    def get_data(
        self,
        rescale_latent=False,
        with_shared_params: Optional[pr.ParameterTree] = None,
        with_local_params: Optional[pr.ParameterTree] = None,
    ) -> List[PlotData]:
        """get fully-evaluated plot data"""
        lazy_data = self.get_data_lazy(
            rescale_latent=rescale_latent,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
        )

        # evaluate all plot data
        for i, data in enumerate(lazy_data):
            data.set_xy()

        return lazy_data


# 7.11 s
