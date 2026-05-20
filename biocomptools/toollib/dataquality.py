# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Data quality analysis via split-half nRMSE (intrinsic noise floor estimation)."""

from typing import Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict

from biocomptools.toollib.networkprediction import _compute_split_half_nrmse
from biocomp.metric_utils import DEFAULT_GRIDSTATS_PARAMS


def compute_split_half_nrmse(
    x: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    n_bootstraps: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute intrinsic noise floor via the SSOT split-half implementation.

    This estimates the theoretical lower bound of prediction error - no model can
    reliably achieve a score lower than this.

    Args:
        x: Input features (n_samples, n_dims), already in latent space
        y: Target values (n_samples,) or (n_samples, 1), already in latent space
        params: Grid stats parameters (uses defaults if None)
        n_bootstraps: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with 'split_half_nrmse', 'std', 'n_bootstraps', 'n_points'.
        `std` is `NaN` because the canonical implementation currently returns
        a single aggregated split-half estimate.
    """
    params = params or DEFAULT_GRIDSTATS_PARAMS
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n_points = len(x)
    score = _compute_split_half_nrmse(
        latent_x=x,
        latent_gt=y,
        params=params,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    return {
        'split_half_nrmse': float(score),
        'std': np.nan,
        'n_bootstraps': n_bootstraps if np.isfinite(score) else 0,
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
