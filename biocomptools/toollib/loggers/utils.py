"""Shared utilities for logger implementations."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_scalar(val: Any, default: float = float("nan")) -> float:
    """Convert JAX/numpy array or scalar to Python float, taking mean if multi-element.

    step_history values from jax.lax.scan have shape (batches_per_step, ...)
    so we need to handle multi-element arrays by taking the mean.
    Works with JAX arrays on any device (np.asarray handles device transfer).
    """
    if val is None:
        return default
    if hasattr(val, "shape"):
        arr = np.asarray(val)
        if arr.size == 0:
            return default
        if arr.size == 1:
            return float(arr.item())
        return float(np.nanmean(arr))
    if hasattr(val, "__float__"):
        return float(val)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return float(np.nanmean(val))
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


PENALTY_NAMES = [
    "l0_penalty",
    "tucount_penalty",
    "spread_penalty",
    "coupling_penalty",
    "ern_tying_penalty",
    "entropy_penalty",
    "commitment_penalty",
]


def extract_design_step_metrics(step_history: dict[str, Any]) -> dict[str, Any]:
    """Extract common scalar metrics from a design optimization step_history.

    Covers loss, sublosses, all_losses stats, TU stats, ratio stats, and
    regularization penalties. Each logger can extend the returned dict with
    logger-specific fields.
    """
    metrics: dict[str, Any] = {"loss": to_scalar(step_history.get("loss"))}

    all_losses = step_history.get("all_losses")
    if all_losses is not None:
        arr = np.asarray(all_losses)
        metrics["all_losses_mean"] = float(np.nanmean(arr))
        metrics["all_losses_min"] = float(np.nanmin(arr))
        metrics["all_losses_max"] = float(np.nanmax(arr))

    sublosses = step_history.get("sublosses", {})
    for key, val in sublosses.items():
        metrics[f"subloss_{key}"] = to_scalar(val)

    tu_stats = step_history.get("tu_stats", {})
    for key, val in tu_stats.items():
        metrics[f"tu_{key}"] = to_scalar(val)

    ratio_stats = step_history.get("ratio_stats", {})
    for key, val in ratio_stats.items():
        metrics[f"ratio_{key}"] = to_scalar(val)

    total_penalty = 0.0
    for pname in PENALTY_NAMES:
        val = step_history.get(pname)
        if val is not None:
            scalar_val = to_scalar(val)
            metrics[pname] = scalar_val
            total_penalty += scalar_val
    metrics["total_penalty"] = total_penalty
    loss = metrics["loss"]
    metrics["penalty_fraction"] = total_penalty / loss if loss > 0 else 0.0

    return metrics


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Rolling mean with edge-padded warmup."""
    arr = np.array(values, dtype=np.float64)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def has_nonzero(vals: list[float]) -> bool:
    """True if any value is non-negligible and non-NaN."""
    return any(abs(v) > 1e-12 for v in vals if not np.isnan(v))


def extract_best_network_metrics(step_history: dict[str, Any]) -> dict[str, Any]:
    """Extract loss and weighted subloss decomposition for the best network.

    Uses all_losses to find argmin, then indexes into subloss_per_network.
    Falls back to mean (aggregate) values when per-network data is unavailable.
    """
    metrics: dict[str, Any] = {}

    all_losses_raw = step_history.get("all_losses")
    sublosses = step_history.get("sublosses", {})

    weights: dict[str, float] = {}
    base_names = {k for k in sublosses if not k.endswith("_weighted") and "per_network" not in k}
    for name in base_names:
        uw = float(np.nanmean(np.asarray(sublosses[name])))
        w_key = f"{name}_weighted"
        if w_key in sublosses and abs(uw) > 1e-15:
            w = float(np.nanmean(np.asarray(sublosses[w_key])))
            weights[name] = w / uw
        else:
            weights[name] = 0.0

    if all_losses_raw is not None:
        al = np.asarray(all_losses_raw).ravel()
        best_idx = int(np.argmin(al))
        metrics["loss"] = float(al[best_idx])

        for name in base_names:
            pn_key = f"{name}_per_network"
            w = weights.get(name, 0.0)
            if pn_key in sublosses:
                pn = np.asarray(sublosses[pn_key]).ravel()
                if best_idx < len(pn):
                    metrics[f"subloss_{name}_weighted"] = float(pn[best_idx]) * w
                    continue
            w_key = f"{name}_weighted"
            if w_key in sublosses:
                metrics[f"subloss_{name}_weighted"] = to_scalar(sublosses[w_key])
    else:
        metrics["loss"] = to_scalar(step_history.get("loss"))
        for name in base_names:
            w_key = f"{name}_weighted"
            if w_key in sublosses:
                metrics[f"subloss_{name}_weighted"] = to_scalar(sublosses[w_key])

    total_penalty = 0.0
    for pname in PENALTY_NAMES:
        val = step_history.get(pname)
        if val is not None:
            s = to_scalar(val)
            metrics[pname] = s
            total_penalty += s
    metrics["total_penalty"] = total_penalty

    return metrics
