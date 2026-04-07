from pydantic.functional_validators import BeforeValidator
from pydantic import ConfigDict
from typing import Any, Optional, List, Union, Dict, Annotated, Literal, Tuple, TypeAlias, Callable
import numpy as np
import jax.numpy as jnp
from biocomp.plotutils import PlotData, LazyPlotData, get_reordered_protein_names
from biocomp.datautils import IdentityRescaler
from biocomptools.toollib.common import make_pretty_input_names
from biocomptools.modelmodel import NetworkModel, BiocompModel, NodeSpec
from biocomptools.logging_config import get_logger
import biocomp.parameters as pr
from biocomptools.toollib.datasources import DataSource
from biocomptools.toollib.types import InputOrderElement
from biocomp.plotting.knn_utils_np import get_gaussian_weighted_knn
from biocomp.metric_utils import (
    grid_mse as compute_grid_mse,
    grid_r_squared as compute_grid_r_squared,
    grid_snr as compute_grid_snr,
    grid_kl_divergence as compute_grid_kl,
    compute_nrmse,
    noise_relative_error as compute_nre,
    SPLIT_HALF_SUBSET_SIZE,
    SPLIT_HALF_N_BOOTSTRAPS,
    GridStatsFields,
)
from pathlib import Path
import time

# from concurrent.futures import ProcessPoolExecutor as PoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed

logger = get_logger(__name__)

NdArray: TypeAlias = Union[np.ndarray, jnp.ndarray]


def _stats_task_dump(
    *,
    task: Dict[str, Any],
    idx: int,
    n_networks: int,
    yhats: List[NdArray] | None,
    xvals: List[NdArray] | None,
    input_order: Any,
    collection_points: Any,
) -> Dict[str, Any]:
    return {
        'n_networks': n_networks,
        'shapes_yhats': [yh.shape for yh in yhats] if yhats is not None else None,
        'shapes_x': [x.shape for x in xvals] if xvals is not None else None,
        'input_order': input_order,
        'collection_points': collection_points,
        'task_i': idx,
        'task': task,
    }


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


def _knn_mean_var_neff(tree, grid, y, k, min_points, max_radius, sigma_in_radius=3.0):
    """Combined KNN query + adaptive Gaussian weights + mean/var computation.

    Fused version of get_gaussian_weighted_knn(adaptive_sigma=True, normed_w=False)
    followed by normalization and get_knn_mean_and_variance.
    Returns (mean, variance, n_eff) without intermediate weight arrays.
    """
    eps = 1e-12
    distances, indices = tree.query(grid, k=k, workers=-1)
    finite_mask = np.isfinite(distances)
    max_finite = np.where(finite_mask, distances, -np.inf).max(axis=1)
    sigma = (max_finite / sigma_in_radius).reshape(-1, 1) + eps

    if max_radius is not None:
        valid_mask = finite_mask & (distances <= max_radius)
    else:
        valid_mask = finite_mask
    nb_points = valid_mask.sum(axis=1)

    Z = np.exp(-0.5 * (distances / sigma) ** 2)
    invalid = ~valid_mask
    Z[invalid] = 0.0
    indices_c = indices.copy()
    indices_c[invalid] = 0

    # n_eff from unnormalized weights (no NaN yet)
    n_eff = Z.sum(axis=1, keepdims=True)

    # normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        W = Z / n_eff

    # mean and variance (fused, no NaN in W yet for valid rows)
    y_nei = y[indices_c]
    w = W[..., None]
    mean = np.sum(w * y_nei, axis=1)
    diff = y_nei - mean[:, None, :]
    var_num = np.sum(w * (diff**2), axis=1)
    w2sum = np.sum(W**2, axis=1, keepdims=True)
    denom = np.maximum(1.0 - w2sum, eps)
    variance = var_num / denom

    # mark too-few-neighbors rows as NaN
    too_few = nb_points < min_points
    if np.any(too_few):
        mean[too_few] = np.nan
        variance[too_few] = np.nan
        n_eff[too_few] = np.nan

    return mean, variance, n_eff


def _compute_split_half_nrmse(
    latent_x: NdArray,
    latent_gt: NdArray,
    params: Dict[str, Any],
    n_bootstraps: int = SPLIT_HALF_N_BOOTSTRAPS,
    seed: int = 42,
    grid: NdArray | None = None,
) -> float:
    """Compute intrinsic noise floor via bootstrapped split-half nRMSE.

    For each bootstrap iteration, draws two independent subsets (with replacement)
    of fixed size, computes local means for each at shared grid points, then
    measures the nRMSE between them. Uses fixed subset size for fair comparison
    across datasets of different sizes.

    Note: dataquality.compute_split_half_nrmse uses a permutation-based approach
    which is conceptually similar but uses different sampling strategy. This
    bootstrap version is used for prediction noise estimation.
    """
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(seed)
    latent_gt = latent_gt.reshape(-1, 1) if latent_gt.ndim == 1 else latent_gt
    n_points = len(latent_x)

    if n_points < 200:
        return np.nan

    valid_mask = np.all(np.isfinite(latent_x), axis=1) & np.all(np.isfinite(latent_gt), axis=1)
    latent_x = np.asarray(latent_x[valid_mask])
    latent_gt = np.asarray(latent_gt[valid_mask])
    n_points = len(latent_x)

    if n_points < 200:
        return np.nan

    if grid is None:
        grid = make_hypercube(
            latent_x.shape[1],
            res=params['hypercube_res'],
            xmin=params['hypercube_min'],
            xmax=params['hypercube_max'],
        )

    subset_size = min(SPLIT_HALF_SUBSET_SIZE, n_points)
    k = params['k']
    min_points_param = params['min_points']
    max_radius = params['radius']

    # loop-invariant constants
    global_var = float(np.var(latent_gt))
    global_range = float(np.ptp(latent_gt))
    PRIOR_STRENGTH = 100.0
    ROBUST_EPSILON = max(0.01, 0.01 * global_range)
    WEIGHT_CAP = 0.1 * k

    scores = []
    for _ in range(n_bootstraps):
        idx_a = rng.choice(n_points, size=subset_size, replace=True)
        idx_b = rng.choice(n_points, size=subset_size, replace=True)

        x_a, y_a = latent_x[idx_a], latent_gt[idx_a]
        x_b, y_b = latent_x[idx_b], latent_gt[idx_b]

        # Use cKDTree directly (data is already finite from filtering above)
        tree_a = cKDTree(x_a)
        tree_b = cKDTree(x_b)

        mean_a, var_a, n_eff_a = _knn_mean_var_neff(tree_a, grid, y_a, k, min_points_param, max_radius)
        mean_b, var_b, n_eff_b = _knn_mean_var_neff(tree_b, grid, y_b, k, min_points_param, max_radius)

        pooled_var = (var_a + var_b) / 2.0
        sq_error = (mean_a - mean_b) ** 2

        n_eff_pooled = (n_eff_a + n_eff_b) / 2.0
        smoothed_var = (pooled_var * n_eff_pooled + global_var * PRIOR_STRENGTH) / (
            n_eff_pooled + PRIOR_STRENGTH
        )

        safe_denom = np.maximum(np.sqrt(smoothed_var), ROBUST_EPSILON)

        norm_sq_error = sq_error / (safe_denom**2)

        capped_weights = np.broadcast_to(np.minimum(n_eff_pooled, WEIGHT_CAP), norm_sq_error.shape)

        weights_flat = capped_weights.ravel()
        norm_sq_flat = norm_sq_error.ravel()
        mask = np.isfinite(norm_sq_flat) & (weights_flat > 0)

        if np.any(mask):
            nrmse = np.sqrt(np.average(norm_sq_flat[mask], weights=weights_flat[mask]))
            if np.isfinite(nrmse):
                scores.append(float(nrmse))

    return float(np.mean(scores)) if scores else np.nan


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


def validate_input_order(v: Any) -> Optional[List[List[InputOrderElement]]]:
    """Convert flat list of ints/strings to list of lists for broadcasting."""
    if v is None:
        return None
    if isinstance(v, list) and all(isinstance(x, (int, str)) for x in v):
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
            grid_stats, grid_cache = _calculate_grid_stats(
                latent_yhat, latent_gt, latent_x, gridstats_params
            )
            network_stats.update(grid_stats)

            # compute data noise floor (split-half nRMSE) reusing cached grid
            data_nrmse = _compute_split_half_nrmse(
                latent_x, latent_gt, gridstats_params, grid=grid_cache
            )
            network_stats['data_nrmse'] = data_nrmse
            network_stats['noise_relative_error'] = compute_nre(
                grid_stats['grid_nrmse'], data_nrmse
            )

    return network_stats


def _calculate_grid_stats(
    latent_yhat: NdArray,
    latent_gt: NdArray,
    latent_x: NdArray,
    params: Dict[str, Any],
) -> tuple[Dict[str, Any], NdArray]:
    """Calculate grid statistics with nRMSE and SNR. Returns (stats, grid) for reuse."""
    from scipy.spatial import cKDTree

    grid = make_hypercube(
        latent_x.shape[1],
        res=params['hypercube_res'],
        xmin=params['hypercube_min'],
        xmax=params['hypercube_max'],
    )

    latent_yhat = latent_yhat.reshape(-1, 1) if latent_yhat.ndim == 1 else latent_yhat
    latent_gt = latent_gt.reshape(-1, 1) if latent_gt.ndim == 1 else latent_gt

    valid_x_mask = np.all(np.isfinite(latent_x), axis=1)
    valid_latent_x = np.asarray(latent_x[valid_x_mask])
    valid_latent_yhat = np.asarray(latent_yhat[valid_x_mask])
    valid_latent_gt = np.asarray(latent_gt[valid_x_mask])
    # dedup: numpy-based unique rows
    if len(valid_latent_x) > 0:
        _, unique_idx = np.unique(valid_latent_x, axis=0, return_index=True)
        if len(unique_idx) < len(valid_latent_x):
            unique_idx.sort()
            valid_latent_x = valid_latent_x[unique_idx]
            valid_latent_yhat = valid_latent_yhat[unique_idx]
            valid_latent_gt = valid_latent_gt[unique_idx]

    global_var = np.var(valid_latent_gt) if valid_latent_gt.size > 0 else 1.0
    global_range = np.ptp(valid_latent_gt) if valid_latent_gt.size > 0 else 1.0

    # data already finite-filtered; use cKDTree directly
    tree = cKDTree(valid_latent_x)

    # raw gaussian weights, single tree query for both n_eff and knn_stats
    # Use adaptive sigma ("leashed balloon"): sigma scales with local density for fair 2D/3D comparison
    # max_radius acts as safety cutoff to prevent averaging over distant regions in voids
    indices, raw_weights = get_gaussian_weighted_knn(
        grid,
        tree=tree,
        k=params['k'],
        min_points=params['min_points'],
        adaptive_sigma=True,
        max_radius=params['radius'],  # hard cutoff for safety
        normed_w=False,
    )

    # effective sample size: sum of raw Gaussian weights (distance-aware confidence)
    n_eff = np.nansum(raw_weights, axis=1, keepdims=True)

    # normalize weights for mean/variance computation
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_weights = raw_weights / n_eff
    iw = (indices, norm_weights)

    from biocomp.plotting.knn_utils_np import get_knn_mean_and_variance
    gt_mean, gt_var = get_knn_mean_and_variance(grid, valid_latent_gt, iw=iw)
    yhat_mean, yhat_var = get_knn_mean_and_variance(grid, valid_latent_yhat, iw=iw)

    # compute metrics directly avoiding redundant sqrt/square roundtrips
    sq_error = (yhat_mean - gt_mean) ** 2
    local_var = gt_var  # gt_stdev**2 == gt_var
    gt_stdev = np.sqrt(gt_var)
    yhat_stdev = np.sqrt(yhat_var)

    mse_val = compute_grid_mse(yhat_mean, gt_mean)
    rmse_val = float(np.sqrt(mse_val))  # avoid recomputing nanmean
    r_squared_val = compute_grid_r_squared(yhat_mean, gt_mean)
    snr_val = compute_grid_snr(gt_mean, local_var)
    kl_mean, kl_similarity = compute_grid_kl(yhat_mean, yhat_stdev, gt_mean, gt_stdev)
    nrmse_val = compute_nrmse(
        sq_error,
        local_var,
        n_eff,
        global_var,
        global_range,
        gt_mean,
        k=params['k'],
    )

    return {
        'grid_gt_var': float(np.nanvar(gt_mean)),
        'grid_mse': mse_val,
        'grid_rmse': rmse_val,
        'grid_nrmse': nrmse_val,
        'grid_snr': snr_val,
        'grid_kl': kl_mean,
        'grid_kl_similarity': kl_similarity,
        'grid_r_squared': r_squared_val,
    }, grid


class NetworkPrediction(GridStatsFields, DataSource):
    """Performs predictions using a NetworkModel and prepares data for plotting.

    IMPORTANT - Space Handling (Latent vs Raw):
    ==========================================
    The biocomp model operates in "latent space" (normalized 0-1 range). Raw fluorescence
    data (values in millions) must be rescaled before prediction.

    There are two modes controlled by `already_latent`:

    1. already_latent=False (DEFAULT):
       - predict_at: expects RAW space (original fluorescence values, e.g., 1e6)
       - Internally calls NetworkModel.predict_unscaled() which:
         * Converts X to latent via rescaler.fwd(X)
         * Runs model prediction
         * Converts Y back to raw via rescaler.inv(Y)
       - get_data() returns: (X_raw, Y_raw)

    2. already_latent=True:
       - predict_at: expects LATENT space (0-1 normalized values)
       - Internally calls NetworkModel.predict() directly (no rescaling)
       - get_data() returns: (X_latent, Y_latent)
       - get_data(rescale_latent=True) returns: (X_raw, Y_raw)

    Common mistake: Passing latent data without already_latent=True will cause
    double-rescaling (latent values treated as raw, scaled again), producing garbage.

    Example usage:
        # From raw experimental data (DBSource, etc.)
        pred = NetworkPrediction(predict_at=[X_raw], network_model=nm)
        data = pred.get_data()  # Returns (X_raw, Y_raw)

        # From design targets (DataTarget.X is already latent)
        pred = NetworkPrediction(predict_at=[X_latent], network_model=nm, already_latent=True)
        data = pred.get_data(rescale_latent=True)  # Returns (X_raw, Y_raw)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    predict_at: Annotated[Union[NdArray, List[NdArray]], BeforeValidator(validate_predict_at)]
    network_model: NetworkModel
    input_order: Annotated[
        Optional[List[List[InputOrderElement]]], BeforeValidator(validate_input_order)
    ] = None
    ground_truth: Annotated[
        Optional[Union[NdArray, List[Optional[NdArray]]]],
        BeforeValidator(validate_ground_truth),
    ] = None

    per_prediction_info: Optional[list[Dict[str, Any]]] = None  # metadata for each prediction

    collection_points: Optional[List[NodeSpec]] = None  # collection points to examine inner nodes

    seed: int = 0
    max_evals: int = 300000
    use_output_as_input: bool = False
    z_value: Union[Literal['uniform', 'normal'], float] = 'uniform'
    z_normal_mean: float = 0.5
    z_normal_std: float = 0.2
    z_normal_clip: bool = True
    disable_variational: bool = True
    device: Literal['cpu', 'gpu'] = 'cpu'  # device preference for predictions

    save_csv_to: Optional[str] = None  # save prediction statistics to a CSV file

    already_latent: bool = (
        False  # Set True if predict_at is in latent space (0-1). See class docstring.
    )

    enable_gridstats: bool = True
    # gridstats_* fields inherited from GridStatsFields mixin

    shuffle_inputs: bool = False  # shuffle inputs before prediction (False preserves spatial structure)

    skip_input_reorder: bool = False  # skip input column reordering (for design visualization)

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
        for i, (x, net) in enumerate(zip(self.predict_at, self.network_model.network, strict=True)):
            expected_inputs = net.nb_inputs
            actual_inputs = x.shape[1] if x.ndim > 1 else 1
            if actual_inputs > expected_inputs:
                raise ValueError(
                    f"Network {i} ({net.name}): predict_at has {actual_inputs} columns but "
                    f"network expects {expected_inputs} inputs. "
                    "Explicitly provide correctly shaped inputs instead of relying on truncation."
                )
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
        for x, gt in zip(self.predict_at, self.ground_truth, strict=True):
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

        for i, (x, gt, _network) in enumerate(
            zip(self.predict_at, self.ground_truth, self.network_model.network, strict=True)
        ):
            if len(x) < effective_max_evals:
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
                "prediction params: "
                f"seed={self.seed}, disable_variational={self.disable_variational}, "
                f"z_value={self.z_value}, z_normal_mean={self.z_normal_mean}, "
                f"z_normal_std={self.z_normal_std}, z_normal_clip={self.z_normal_clip}, "
                f"device={self.device}"
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
            z_normal_mean=self.z_normal_mean,
            z_normal_std=self.z_normal_std,
            z_normal_clip=self.z_normal_clip,
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
        assert isinstance(self._aligned_x, list)

        for i, (network_output, aligned_x) in enumerate(
            zip(network_outputs, self._aligned_x, strict=True)
        ):
            effective_max_evals = min(max_evals, len(aligned_x))

            # store prediction results
            truncated_output = network_output[:effective_max_evals]
            self._yhats.append(np.asarray(truncated_output, dtype=np.float32))

            # store ground truth if available
            if self.ground_truth[i] is not None:
                truncated_gt = self._aligned_ground_truth[i][:effective_max_evals]
                self._gtruths.append(np.asarray(truncated_gt, dtype=np.float32))
            else:
                self._gtruths.append(None)

            # store inputs in NETWORK order (not display order)
            # extract_lazy_plot_data_from_network will apply input_order to convert to display
            truncated_x = aligned_x[:effective_max_evals]
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
                logger.debug(
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
                logger.debug(
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
                    logger.debug("applied inverse rescaling to x and yhat")

            if self.use_output_as_input:
                if self.verbose:
                    logger.debug("using output as input, returning yhat as both x and y")
                return yhat, yhat
            else:
                if self.verbose:
                    logger.debug("using original input, returning x and yhat")
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
            x = self._x[i]  # Already in network order from _prepare_inputs

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

            gridstats_params = self.get_gridstats_params()

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

        all_stats: List[Dict[str, Any] | None] = [None] * len(self.network_model.network)

        def process_and_log_stats(idx, task, result):
            all_stats[idx] = result
            if self.verbose and task.get('gt') is not None:
                rescaler = task['rescaler']
                latent_gt = rescaler.fwd(task['gt'])
                latent_yhat = rescaler.fwd(task['yhat'])
                if latent_gt.shape < latent_yhat.shape:
                    logger.debug(
                        f"ground truth shape {latent_gt.shape} is smaller than yhat shape {latent_yhat.shape}, "
                        "Assuming gt is only the dependent outputs."
                    )
                    output_pos = task['dependent_output_pos']
                    if isinstance(output_pos, int):
                        output_pos = [output_pos]
                    latent_yhat = latent_yhat[:, output_pos]
                latent_x = rescaler.fwd(task['x'])
                self._log_stats(idx, result, latent_gt, latent_yhat, latent_x)

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
                    except Exception as e:
                        net_name = task['network_info'].get('network_name', f"Network_{idx}")
                        raise RuntimeError(
                            f"Error calculating stats for network {net_name}: {e}; "
                            f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                        ) from e
                    process_and_log_stats(idx, task, result)

        else:
            if len(tasks) > 0:
                logger.info(
                    "Calculating network stats sequentially using the unified static method."
                )

            for task in tasks:
                idx = task['network_idx']
                try:
                    result = _calculate_single_network_stats(**task)
                except Exception as e:
                    net_name = task['network_info'].get('network_name', f"Network_{idx}")
                    raise RuntimeError(
                        f"Error calculating stats for network {net_name}: {e}; "
                        f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                    ) from e
                process_and_log_stats(idx, task, result)
        assert all(s is not None for s in all_stats), "all network stats must be populated"
        return [s for s in all_stats if s is not None]

    def compute_stats_from_outputs(self, network_outputs: List[NdArray]) -> List[Dict[str, Any]]:
        """Compute stats from externally-provided network outputs (e.g., from batched predictions).

        This is the canonical stats computation - uses same logic as _calculate_all_network_stats.
        Useful for hyperopt batched validation where outputs come from vmapped predictions.
        """
        from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed

        rescaler = (
            self.network_model.model.rescaler if not self.already_latent else IdentityRescaler()
        )
        gridstats_params = self.get_gridstats_params()

        # Build tasks
        tasks = []
        for i, network in enumerate(self.network_model.network):
            if i >= len(network_outputs):
                break
            yhat = network_outputs[i]
            gt = (
                self._aligned_ground_truth[i]
                if self._aligned_ground_truth and i < len(self._aligned_ground_truth)
                else None
            )
            x = self._aligned_x[i] if self._aligned_x and i < len(self._aligned_x) else None

            _, dependent_output_pos, _, _ = get_reordered_protein_names(network)
            network_name = getattr(network, 'name', f"Network_{i}")
            nb_points_in_eval = len(self.predict_at[i]) if self.predict_at else yhat.shape[0]

            all_outputs = set(network.get_output_proteins())
            input_proteins = set(network.get_inverted_input_proteins())
            n_dependent_outputs = len(all_outputs - input_proteins)

            network_info = {
                'network_name': network_name,
                'n_dependent_outputs': n_dependent_outputs,
            }
            if self.per_prediction_info is not None and i < len(self.per_prediction_info):
                network_info['extra_prediction_info'] = self.per_prediction_info[i]

            tasks.append(
                {
                    'network_idx': i,
                    'yhat': yhat,
                    'gt': gt,
                    'x': x,
                    'dependent_output_pos': dependent_output_pos,
                    'nb_points_in_eval': nb_points_in_eval,
                    'rescaler': rescaler,
                    'gridstats_params': gridstats_params,
                    'network_info': network_info,
                    'enable_gridstats': self.enable_gridstats,
                }
            )

        # Process tasks (parallel if n_stats_workers > 1)
        all_stats: List[Dict[str, Any] | None] = [None] * len(tasks)
        show_progress = self.verbose and len(tasks) > 3

        if self.n_stats_workers > 1 and len(tasks) > 1:
            from tqdm import tqdm

            with PoolExecutor(max_workers=self.n_stats_workers) as executor:
                future_to_idx = {
                    executor.submit(_calculate_single_network_stats, **task): task['network_idx']
                    for task in tasks
                }
                futures_iter = as_completed(future_to_idx)
                if show_progress:
                    futures_iter = tqdm(
                        futures_iter, total=len(tasks), desc="Stats", leave=False, ncols=80
                    )
                for future in futures_iter:
                    idx = future_to_idx[future]
                    task = tasks[idx]
                    try:
                        all_stats[idx] = future.result()
                    except Exception as e:
                        net_name = task['network_info'].get('network_name', f'Network_{idx}')
                        raise RuntimeError(
                            f"Stats computation failed for network {net_name}: {e}; "
                            f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                        ) from e
        else:
            from tqdm import tqdm

            task_iter = tasks
            if show_progress:
                task_iter = tqdm(tasks, desc="Stats", leave=False, ncols=80)
            for task in task_iter:
                idx = task['network_idx']
                try:
                    all_stats[idx] = _calculate_single_network_stats(**task)
                except Exception as e:
                    net_name = task['network_info'].get('network_name', f'Network_{idx}')
                    raise RuntimeError(
                        f"Stats computation failed for network {net_name}: {e}; "
                        f"task={_stats_task_dump(task=task, idx=idx, n_networks=len(self.network_model.network), yhats=self._yhats, xvals=self._x, input_order=self.input_order, collection_points=self.collection_points)}"
                    ) from e
        assert all(s is not None for s in all_stats), "all output stats must be populated"
        return [s for s in all_stats if s is not None]

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
        import pandas as pd

        stats = self.get_network_stats()
        df = pd.DataFrame(stats)

        # update df with content of self.extra_metadata
        for key, value in self.metadata.items():
            df[key] = value

        if self.verbose:
            logger.debug(f"prediction statistics DataFrame:\n{df.to_string()}")

        assert self.save_csv_to is not None, "save_csv_to must be set before saving CSV"
        save_to = Path(self.save_csv_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
        logger.info(f"saved prediction statistics to {save_to}")

    def _normalize_input_order(self) -> List[Optional[List[InputOrderElement]]]:
        """normalize input_order to ensure it's consistent across networks"""
        assert self.network_model is not None
        assert isinstance(self.network_model.network, list)

        if self.input_order is None:
            logger.debug(
                f"input_order is None, using None for all {len(self.network_model.network)} networks"
            )
            return [None] * len(self.network_model.network)

        if isinstance(self.input_order, list):
            # single input order for all networks (ints or protein names)
            if all(isinstance(x, (int, str)) for x in self.input_order):
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

        plot_data_list = []
        if self.skip_input_reorder:
            input_order = [list(range(net.nb_inputs)) for net in self.network_model.network]
            logger.debug(f"skip_input_reorder=True, using identity input_order: {input_order}")
        else:
            input_order = self._normalize_input_order()
            logger.debug(f"normalized input_order: {input_order}")

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
        for inshapes, outshapes in zip(self._input_shapes, self._output_shapes, strict=True):
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
                    logger.debug(
                        f"collection point {i}: {collection_point_nodespec}\n"
                        f"  in_shape={self._collected_in.shape}, out_shape={self._collected_out.shape}\n"
                        f"  in_idx={collect_in_idx[i]}, out_idx={collect_out_idx[i]}\n"
                        f"  node_input_shapes={input_shapes}, node_output_shapes={output_shapes}\n"
                        f"  flatx={flatx.shape}, flaty={flaty.shape}"
                    )

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
        input_order: Optional[List[InputOrderElement]],
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
        """Get fully-evaluated PlotData for each network.

        Args:
            rescale_latent: Only applies when already_latent=True. If True, converts
                output (X, Y) from latent space to raw space using rescaler.inv().
                Has no effect when already_latent=False (output is already raw).
            with_shared_params: Optional override for model shared parameters.
            with_local_params: Optional override for model local parameters.

        Returns:
            List of PlotData objects, one per network. Output space depends on settings:
            - already_latent=False: (X_raw, Y_raw)
            - already_latent=True, rescale_latent=False: (X_latent, Y_latent)
            - already_latent=True, rescale_latent=True: (X_raw, Y_raw)
        """
        lazy_data = self.get_data_lazy(
            rescale_latent=rescale_latent,
            with_shared_params=with_shared_params,
            with_local_params=with_local_params,
        )

        # evaluate all plot data
        for data in lazy_data:
            data.set_xy()

        return lazy_data

    def save_results(
        self,
        output_dir: Path | str,
        prediction_data: list[PlotData] | None = None,
        ground_truth: list[PlotData] | None = None,
    ) -> list[Path]:
        """Save per-network prediction results as Parquet files with metadata.

        Saves processed PlotData (from get_data()) which has correct column semantics.
        Ground truth saved separately (may differ in row count).

        Args:
            output_dir: Directory to save Parquet files.
            ground_truth: Optional ground truth PlotData list (for embedding in metadata).

        Returns:
            List of saved file paths.
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prediction_data is None:
            prediction_data = self.get_data()

        stats = self.get_network_stats() if self._yhats is not None else [{}] * len(prediction_data)
        saved = []

        for i, pred in enumerate(prediction_data):
            x_pred = np.asarray(pred.x)
            y_pred = np.asarray(pred.y)

            pred_cols = {f"x_{j}": x_pred[:, j] for j in range(x_pred.shape[1])}
            pred_cols.update({f"y_pred_{j}": y_pred[:, j] for j in range(y_pred.shape[1])})
            df = pd.DataFrame(pred_cols)
            table = pa.Table.from_pandas(df)

            network_name = pred.metadata.get('network_name', f'network_{i}')
            meta = {
                'network_name': network_name,
                'input_names': pred.input_names,
                'output_name': pred.output_name,
                'model_signature': self.network_model.signature,
                'stats': stats[i] if i < len(stats) else {},
            }
            import json
            existing_meta = table.schema.metadata or {}
            existing_meta[b'biocomp'] = json.dumps(meta, default=str).encode()
            table = table.replace_schema_metadata(existing_meta)

            safe_name = network_name.replace('/', '_').replace(' ', '_')
            fpath = output_dir / f"{safe_name}.parquet"
            pq.write_table(table, fpath)
            saved.append(fpath)

            if ground_truth is not None and i < len(ground_truth):
                gt = ground_truth[i]
                gt_x, gt_y = np.asarray(gt.x), np.asarray(gt.y)
                gt_cols = {f"x_{j}": gt_x[:, j] for j in range(gt_x.shape[1])}
                gt_cols.update({f"y_gt_{j}": gt_y[:, j] for j in range(gt_y.shape[1])})
                gt_table = pa.Table.from_pandas(pd.DataFrame(gt_cols))
                gt_meta_dict = {**meta, 'data_type': 'ground_truth', 'input_names': gt.input_names, 'output_name': gt.output_name}
                gt_existing = gt_table.schema.metadata or {}
                gt_existing[b'biocomp'] = json.dumps(gt_meta_dict, default=str).encode()
                gt_table = gt_table.replace_schema_metadata(gt_existing)
                pq.write_table(gt_table, output_dir / f"{safe_name}_gt.parquet")

        logger.info(f"Saved {len(saved)} prediction results to {output_dir}")
        return saved

    @staticmethod
    def load_results(output_dir: Path | str) -> list[tuple[PlotData, PlotData]]:
        """Load saved prediction results as (ground_truth, prediction) PlotData pairs.

        No model or GPU needed — pure file I/O. Ready for the auto-dispatch plot system.

        Returns:
            List of (gt_plotdata, pred_plotdata) tuples, one per network.
        """
        import pyarrow.parquet as pq
        import json

        output_dir = Path(output_dir)
        pairs = []

        for fpath in sorted(output_dir.glob("*.parquet")):
            if fpath.stem.endswith('_gt'):
                continue  # skip GT files, loaded alongside pred files

            table = pq.read_table(fpath)
            meta_bytes = table.schema.metadata.get(b'biocomp')
            meta = json.loads(meta_bytes.decode()) if meta_bytes else {}

            df = table.to_pandas()
            x_cols = sorted([c for c in df.columns if c.startswith('x_')])
            y_pred_cols = sorted([c for c in df.columns if c.startswith('y_pred_')])

            x_pred = df[x_cols].to_numpy()
            y_pred = df[y_pred_cols].to_numpy()

            n_outputs = len(y_pred_cols)
            input_names = meta.get('input_names', [f'x_{i}' for i in range(len(x_cols))])
            output_name = meta.get('output_name', 'output')
            if isinstance(output_name, str) and n_outputs > 1:
                output_name = [output_name] + [f'output_{j}' for j in range(1, n_outputs)]
            stats = meta.get('stats', {})
            network_name = meta.get('network_name', fpath.stem)

            pred_metadata = {
                'network_name': network_name,
                'model_signature': meta.get('model_signature', ''),
                'prediction_stats': stats,
            }
            pred_data = PlotData(
                xval=x_pred, yval=y_pred,
                input_names=input_names, output_name=output_name,
                metadata=pred_metadata,
            )

            # Load separate GT file if it exists
            gt_fpath = fpath.parent / f"{fpath.stem}_gt.parquet"
            if gt_fpath.exists():
                gt_df = pq.read_table(gt_fpath).to_pandas()
                gt_x_cols = sorted([c for c in gt_df.columns if c.startswith('x_')])
                gt_y_cols = sorted([c for c in gt_df.columns if c.startswith('y_gt_')])
                gt_data = PlotData(
                    xval=gt_df[gt_x_cols].to_numpy(),
                    yval=gt_df[gt_y_cols].to_numpy(),
                    input_names=input_names, output_name=output_name,
                    metadata={'network_name': network_name},
                )
            else:
                gt_data = PlotData(
                    xval=x_pred, yval=np.full_like(y_pred, np.nan),
                    input_names=input_names, output_name=output_name,
                    metadata={'network_name': network_name},
                )

            pairs.append((gt_data, pred_data))

        return pairs


# 7.11 s
