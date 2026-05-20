# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Quantile coverage benchmark utilities for distribution-aware model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence
import csv
import json
import time

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import ks_2samp
from dracon import load

from biocomp.plotutils import get_reordered_protein_names
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.toollib.datasources import DBSource
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.toollib.networkselector import (
    CleanupFilter,
    CustomFilter,
    NetworkFilter,
    NetworkSet,
    NetworkSetDifference,
    NetworkSetIntersection,
    NetworkSetUnion,
    Regex,
    UorfFilter,
    iRegex,
)

logger = get_logger(__name__)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    wsum = np.sum(w)
    if wsum <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / wsum)


def _mc_crps(samples: np.ndarray, targets: np.ndarray) -> float:
    """Monte Carlo CRPS from predictive samples.

    Args:
        samples: (N, M, D)
        targets: (N, D)
    """
    n, _m, d = samples.shape
    out = []
    for di in range(d):
        s = samples[:, :, di]
        y = targets[:, di][:, None]
        a = np.mean(np.abs(s - y), axis=1)
        b = np.mean(np.abs(s[:, :, None] - s[:, None, :]), axis=(1, 2))
        crps = np.mean(a - 0.5 * b)
        out.append(float(crps))
    if not out:
        return float("nan")
    return float(np.mean(out))


def compute_quantile_metrics_from_samples(
    samples: np.ndarray,
    targets: np.ndarray,
    quantiles: Sequence[float],
    coverage_interval: tuple[float, float],
) -> dict[str, Any]:
    """Compute quantile metrics for one network from predictive samples."""
    samples = np.asarray(samples, dtype=float)
    targets = _ensure_2d(np.asarray(targets, dtype=float))
    assert samples.ndim == 3, f"samples must be (N, M, D), got {samples.shape}"
    assert targets.ndim == 2, f"targets must be (N, D), got {targets.shape}"
    assert samples.shape[0] == targets.shape[0], "sample/target count mismatch"
    assert samples.shape[2] == targets.shape[1], "sample/target output-dim mismatch"

    qs = sorted(float(q) for q in quantiles)
    q_preds = {q: np.quantile(samples, q, axis=1) for q in qs}

    pinball = {}
    for q in qs:
        err = targets - q_preds[q]
        pin = np.maximum(q * err, (q - 1.0) * err)
        pinball[q] = float(np.mean(pin))

    qlow, qhigh = coverage_interval
    low = np.quantile(samples, qlow, axis=1)
    high = np.quantile(samples, qhigh, axis=1)
    in_interval = (targets >= low) & (targets <= high)
    coverage = float(np.mean(in_interval))
    target_coverage = float(qhigh - qlow)
    coverage_error = abs(coverage - target_coverage)

    median_q = min(qs, key=lambda q: abs(q - 0.5))
    median_pred = q_preds[median_q]
    rmse_q50 = float(np.sqrt(np.mean((median_pred - targets) ** 2)))

    samples_flat = samples.reshape(-1)
    targets_flat = targets.reshape(-1)
    ks = float(ks_2samp(samples_flat, targets_flat).statistic)

    return {
        "n_points": int(samples.shape[0]),
        "n_samples_per_point": int(samples.shape[1]),
        "n_outputs": int(samples.shape[2]),
        "coverage_interval": [float(qlow), float(qhigh)],
        "target_coverage": target_coverage,
        "coverage": coverage,
        "coverage_error": coverage_error,
        "pinball": {str(q): float(v) for q, v in pinball.items()},
        "mean_pinball": float(np.mean(list(pinball.values()))) if pinball else float("nan"),
        "rmse_q50": rmse_q50,
        "mc_crps": _mc_crps(samples, targets),
        "ks_marginal": ks,
    }


def _load_model(
    *,
    model: Any = None,
    model_path: str | None = None,
    model_name: str | None = None,
) -> BiocompModel:
    if model is not None:
        return model
    if model_name:
        return ModelSelector(name=model_name).get_model().load()
    if model_path:
        return BiocompModel.load(model_path)
    raise ValueError("One of model, model_path, or model_name must be provided.")


def _load_dataset(dataset_file: str):
    ctx = {
        "NetworkSet": NetworkSet,
        "NetworkSetUnion": NetworkSetUnion,
        "NetworkSetDifference": NetworkSetDifference,
        "NetworkSetIntersection": NetworkSetIntersection,
        "CleanupFilter": CleanupFilter,
        "NetworkFilter": NetworkFilter,
        "CustomFilter": CustomFilter,
        "Regex": Regex,
        "iRegex": iRegex,
        "DBSource": DBSource,
        "UorfFilter": UorfFilter,
    }
    dataset = load(dataset_file, context=ctx)
    if hasattr(dataset, "get_data"):
        return dataset.get_data()
    return DBSource(content=dataset).get_data()


def compute_quantile_coverage(
    *,
    dataset_file: str,
    model: Any = None,
    model_path: str | None = None,
    model_name: str | None = None,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    coverage_interval: tuple[float, float] = (0.1, 0.9),
    n_samples_per_point: int = 64,
    seed: int = 11,
    max_evals: int = 0,
    device: str = "gpu",
    disable_variational: bool = True,
    z_value: str | float = "uniform",
    z_normal_mean: float = 0.5,
    z_normal_std: float = 0.2,
    z_normal_clip: bool = True,
) -> dict[str, Any]:
    """Evaluate distribution recapitulation via quantile metrics."""
    quantiles = [float(q) for q in quantiles]
    coverage_interval = (float(coverage_interval[0]), float(coverage_interval[1]))
    n_samples_per_point = int(n_samples_per_point)
    seed = int(seed)
    max_evals = int(max_evals or 0)
    device = str(device)
    disable_variational = _as_bool(disable_variational)
    z_normal_clip = _as_bool(z_normal_clip)

    t0 = time.time()
    loaded_model = _load_model(model=model, model_path=model_path, model_name=model_name)
    ground_truth = _load_dataset(dataset_file)

    per_network: list[dict[str, Any]] = []
    for idx, pdata in enumerate(ground_truth):
        network = pdata.metadata["built_network"]
        network_name = pdata.metadata.get("network_name", f"Network_{idx}")
        nm = NetworkModel(model=loaded_model, network=[network])

        x = _ensure_2d(np.asarray(pdata.x, dtype=np.float32))
        y = _ensure_2d(np.asarray(pdata.y, dtype=np.float32))
        if max_evals and max_evals > 0:
            x = x[:max_evals]
            y = y[:max_evals]

        x_rep = np.repeat(x, n_samples_per_point, axis=0)
        yhat, _ = nm.predict_unscaled(
            x_rep,
            key=seed + idx,
            z_value=z_value,
            z_normal_mean=float(z_normal_mean),
            z_normal_std=float(z_normal_std),
            z_normal_clip=z_normal_clip,
            disable_variational=disable_variational,
            device=device,
        )

        yhat = _ensure_2d(np.asarray(yhat, dtype=np.float32))
        _, dep_output_pos, _, _ = get_reordered_protein_names(network)
        dep_output_pos = [dep_output_pos] if isinstance(dep_output_pos, int) else dep_output_pos

        if dep_output_pos and yhat.shape[1] > max(dep_output_pos):
            yhat_dep = yhat[:, dep_output_pos]
        else:
            yhat_dep = yhat

        if dep_output_pos and y.shape[1] > max(dep_output_pos):
            y_dep = y[:, dep_output_pos]
        elif y.shape[1] == yhat_dep.shape[1]:
            y_dep = y
        else:
            y_dep = y[:, : yhat_dep.shape[1]]

        samples = yhat_dep.reshape(x.shape[0], n_samples_per_point, yhat_dep.shape[1])
        metrics = compute_quantile_metrics_from_samples(
            samples=samples,
            targets=y_dep,
            quantiles=quantiles,
            coverage_interval=coverage_interval,
        )
        per_network.append(
            {
                "network_name": network_name,
                "network_index": idx,
                **metrics,
            }
        )

    weights = [m["n_points"] for m in per_network]
    aggregate = {
        "mean_pinball": _weighted_mean([m["mean_pinball"] for m in per_network], weights),
        "coverage": _weighted_mean([m["coverage"] for m in per_network], weights),
        "coverage_error": _weighted_mean([m["coverage_error"] for m in per_network], weights),
        "mc_crps": _weighted_mean([m["mc_crps"] for m in per_network], weights),
        "rmse_q50": _weighted_mean([m["rmse_q50"] for m in per_network], weights),
        "ks_marginal": _weighted_mean([m["ks_marginal"] for m in per_network], weights),
        "target_coverage": float(coverage_interval[1] - coverage_interval[0]),
    }

    return {
        "model_signature": loaded_model.signature,
        "dataset_file": dataset_file,
        "dataset_name": Path(dataset_file).stem,
        "seed": int(seed),
        "quantiles": [float(q) for q in quantiles],
        "coverage_interval": [float(coverage_interval[0]), float(coverage_interval[1])],
        "n_samples_per_point": int(n_samples_per_point),
        "disable_variational": bool(disable_variational),
        "z_value": z_value,
        "z_normal_mean": float(z_normal_mean),
        "z_normal_std": float(z_normal_std),
        "z_normal_clip": bool(z_normal_clip),
        "device": device,
        "aggregate": aggregate,
        "per_network": per_network,
        "runtime_seconds": float(time.time() - t0),
    }


def write_quantile_coverage_csv(result: dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantiles = [str(q) for q in result.get("quantiles", [])]
    headers = [
        "network_index",
        "network_name",
        "n_points",
        "coverage",
        "target_coverage",
        "coverage_error",
        "mean_pinball",
        "mc_crps",
        "rmse_q50",
        "ks_marginal",
    ] + [f"pinball_q{q}" for q in quantiles]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for m in result["per_network"]:
            row = {
                "network_index": m["network_index"],
                "network_name": m["network_name"],
                "n_points": m["n_points"],
                "coverage": m["coverage"],
                "target_coverage": m["target_coverage"],
                "coverage_error": m["coverage_error"],
                "mean_pinball": m["mean_pinball"],
                "mc_crps": m["mc_crps"],
                "rmse_q50": m["rmse_q50"],
                "ks_marginal": m["ks_marginal"],
            }
            for q in quantiles:
                row[f"pinball_q{q}"] = m["pinball"].get(q, float("nan"))
            writer.writerow(row)
    return str(path)


def write_quantile_coverage_json(result: dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return str(path)


def render_quantile_coverage_summary(
    *,
    ax: Axes,
    result: Optional[dict[str, Any]] = None,
    dataset_file: Optional[str] = None,
    model: Any = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    coverage_interval: tuple[float, float] = (0.1, 0.9),
    n_samples_per_point: int = 64,
    seed: int = 11,
    max_evals: int = 0,
    device: str = "gpu",
    disable_variational: bool = True,
    z_value: str | float = "uniform",
    z_normal_mean: float = 0.5,
    z_normal_std: float = 0.2,
    z_normal_clip: bool = True,
    save_csv_to: Optional[str] = None,
    save_json_to: Optional[str] = None,
    **_kwargs,
) -> dict[str, Any]:
    """Render quantile-coverage benchmark summary on a matplotlib axis."""
    if result is None:
        if dataset_file is None:
            raise ValueError("dataset_file is required when result is not provided.")
        result = compute_quantile_coverage(
            dataset_file=dataset_file,
            model=model,
            model_path=model_path,
            model_name=model_name,
            quantiles=quantiles,
            coverage_interval=coverage_interval,
            n_samples_per_point=n_samples_per_point,
            seed=seed,
            max_evals=max_evals,
            device=device,
            disable_variational=disable_variational,
            z_value=z_value,
            z_normal_mean=z_normal_mean,
            z_normal_std=z_normal_std,
            z_normal_clip=z_normal_clip,
        )

    if save_csv_to:
        saved_csv = write_quantile_coverage_csv(result, save_csv_to)
        logger.info(f"Saved quantile coverage CSV: {saved_csv}")
    if save_json_to:
        saved_json = write_quantile_coverage_json(result, save_json_to)
        logger.info(f"Saved quantile coverage JSON: {saved_json}")

    agg = result["aggregate"]
    rows = sorted(result["per_network"], key=lambda x: x["mean_pinball"])[:8]

    ax.axis("off")
    lines = [
        f"Quantile Coverage Benchmark",
        f"Model: {result['model_signature']}",
        f"Dataset: {result['dataset_name']}",
        f"Seed={result['seed']} | Samples/point={result['n_samples_per_point']} | Disable variational={result['disable_variational']}",
        f"z_value={result.get('z_value', 'uniform')} | z_normal_mean={result.get('z_normal_mean', 0.5)} | z_normal_std={result.get('z_normal_std', 0.2)} | z_normal_clip={result.get('z_normal_clip', True)}",
        "",
        f"Aggregate mean pinball: {agg['mean_pinball']:.5f}",
        f"Aggregate coverage: {agg['coverage']:.5f} (target {agg['target_coverage']:.2f})",
        f"Aggregate coverage error: {agg['coverage_error']:.5f}",
        f"Aggregate MC-CRPS: {agg['mc_crps']:.5f}",
        f"Aggregate RMSE@q50: {agg['rmse_q50']:.5f}",
        f"Aggregate KS marginal: {agg['ks_marginal']:.5f}",
        "",
        "Top networks by mean pinball:",
    ]
    for row in rows:
        lines.append(
            f"  {row['network_index']:03d} {row['network_name'][:70]} | pinball={row['mean_pinball']:.5f} | cov={row['coverage']:.3f}"
        )
    ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    return result
