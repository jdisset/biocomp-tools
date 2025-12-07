"""Data quality analysis via split-half nRMSE (intrinsic noise floor estimation)."""

from typing import Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict

from biocomptools.toollib.networkprediction import _calculate_grid_stats

DEFAULT_GRIDSTATS_PARAMS = {
    'hypercube_res': 8,
    'hypercube_min': 0.0,
    'hypercube_max': 0.8,
    'k': 1024,
    'radius': 0.25,
    'min_points': 40,
}


def compute_split_half_nrmse(
    x: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_bootstraps: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute intrinsic noise floor via bootstrapped symmetric split-half nRMSE.

    This estimates the theoretical lower bound of prediction error - no model can
    reliably achieve a score lower than this.

    Args:
        x: Input features (n_samples, n_dims), already in latent space
        y: Target values (n_samples,) or (n_samples, 1), already in latent space
        params: Grid stats parameters (uses defaults if None)
        n_bootstraps: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with 'split_half_nrmse', 'std', 'n_bootstraps', 'n_points'
    """
    params = params or DEFAULT_GRIDSTATS_PARAMS
    rng = np.random.RandomState(seed)

    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n_points = len(x)

    if n_points < 100:
        return {'split_half_nrmse': np.nan, 'std': np.nan, 'n_bootstraps': 0, 'n_points': n_points}

    scores = []
    for _ in range(n_bootstraps):
        perm = rng.permutation(n_points)
        mid = n_points // 2
        idx_a, idx_b = perm[:mid], perm[mid:2*mid]

        x_a, y_a = x[idx_a], y[idx_a]
        x_b, y_b = x[idx_b], y[idx_b]

        # Symmetric evaluation: A->B and B->A
        stats_ab = _calculate_grid_stats(y_a, y_b, x_b, params)
        stats_ba = _calculate_grid_stats(y_b, y_a, x_a, params)

        score = (stats_ab['grid_nrmse'] + stats_ba['grid_nrmse']) / 2.0
        if np.isfinite(score):
            scores.append(score)

    if not scores:
        return {'split_half_nrmse': np.nan, 'std': np.nan, 'n_bootstraps': 0, 'n_points': n_points}

    return {
        'split_half_nrmse': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'n_bootstraps': len(scores),
        'n_points': n_points,
    }


class DataQualityReport(BaseModel):
    """Report of data quality metrics for a dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    network_name: str
    experiment_name: str = ""
    recipe_name: str = ""
    n_dims: int = 0
    n_points: int = 0
    split_half_nrmse: float = np.nan
    split_half_std: float = np.nan

    @property
    def quality_tier(self) -> str:
        """Interpret the split-half nRMSE score."""
        score = self.split_half_nrmse
        if np.isnan(score):
            return "unknown"
        elif score < 0.5:
            return "excellent"
        elif score < 1.0:
            return "acceptable"
        elif score < 1.5:
            return "noisy"
        else:
            return "problematic"
