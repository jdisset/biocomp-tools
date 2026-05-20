# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""σ_repeat: pairwise replicate-divergence metrics via Gaussian kNN smoothing.

For ordered pair (A, B):
    σ_RMSE(A->B) = RMSE(B.Y, kernel_A(B.X))
    σ_RRE(A->B)  = σ_RMSE(A->B) / RMSE(B.Y, kernel_B(B.X))

Two consumers downstream:

- `compute_group(...)` + `write_yaml(...)` - write the per-group metric YAML.
- `pair_panels(...)` - return a list of `mvp_panel`-compatible dicts (one per
  ordered (i,j) pair, including diagonal self-fits) that can be fed straight
  into `paper-jobs/plot/figures/autofig_dataset_row.yaml` rows. Bypasses the
  `MeasuredVsPredictedData` model (which is NetworkPrediction-shaped) by
  exposing a tiny duck-type `PairMVPData`.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray as NdArray
from pydantic import BaseModel, ConfigDict

from biocomp.plotting.knn_utils_np import make_tree, get_knn_mean_only


@dataclass
class KernelParams:
    k: int = 20
    min_points: int = 10
    radius: float = 0.3
    adaptive_sigma: bool = True
    sigma_in_radius: float = 3.0


@dataclass
class RescalingParams:
    """Log10 + clip-to-quantiles + global min-max -> [0,1] using constants
    pooled across all runs in the group."""

    floor: float = 1.0
    quantile_low: float = 0.01
    quantile_high: float = 0.99


@dataclass
class Run:
    label: str
    xp: str
    basename: str
    x: np.ndarray
    y: np.ndarray
    extra: dict = field(default_factory=dict)


@dataclass
class PairMetrics:
    a: str
    b: str
    kind: str
    sigma_rmse: float
    sigma_rre: float
    n_a: int
    n_b: int


@dataclass
class GroupMetrics:
    group_id: str
    short_name: str
    fingerprint: str
    kernel_params: KernelParams
    rescale: dict = field(default_factory=dict)
    pairs: list[PairMetrics] = field(default_factory=list)

    def aggregate(self, kind: str) -> dict:
        vals = [p for p in self.pairs if p.kind == kind]
        if not vals:
            return {"n_pairs": 0}
        rmse = np.array([p.sigma_rmse for p in vals], dtype=float)
        rre = np.array([p.sigma_rre for p in vals], dtype=float)
        return {
            "n_pairs": len(vals),
            "mean_sigma_rmse": float(np.nanmean(rmse)),
            "median_sigma_rmse": float(np.nanmedian(rmse)),
            "std_sigma_rmse": float(np.nanstd(rmse)),
            "mean_sigma_rre": float(np.nanmean(rre)),
            "median_sigma_rre": float(np.nanmedian(rre)),
            "std_sigma_rre": float(np.nanstd(rre)),
        }


class PairMVPData(BaseModel):
    """Drop-in duck type for `MeasuredVsPredictedData` consumed by
    `paper-jobs/plot/figures/tasks/mvp_panel.yaml`. Carries flat measured /
    predicted arrays plus the optional grid-overlay slots (always None here)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    measured: NdArray
    predicted: NdArray
    rescaler: Any = None
    grid_measured: NdArray | None = None
    grid_predicted: NdArray | None = None
    grid_weights: NdArray | None = None


def _y_d_out(y: np.ndarray) -> int:
    return y.shape[1] if y.ndim > 1 else 1


def _first_channel(y: np.ndarray) -> np.ndarray:
    return y[:, 0] if y.ndim == 2 else y


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for a in arrays:
        a2 = a if a.ndim > 1 else a.reshape(-1, 1)
        mask &= np.all(np.isfinite(a2), axis=1)
    return mask


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_true - y_pred).reshape(-1)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.nanmean(diff[finite] ** 2)))


def kernel_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, kp: KernelParams
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)
    train_mask = _finite_rows(x_train, y_train)
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    d_out = _y_d_out(y_train)
    if x_train.size == 0:
        return np.full((len(x_query), d_out), np.nan, dtype=np.float32)

    q_mask = _finite_rows(x_query)
    if not q_mask.any():
        return np.full((len(x_query), d_out), np.nan, dtype=np.float32)

    pred = get_knn_mean_only(
        x_query[q_mask],
        y_train,
        tree=make_tree(x_train),
        k=kp.k,
        min_points=kp.min_points,
        radius=kp.radius,
        adaptive_sigma=kp.adaptive_sigma,
        max_radius=kp.radius,
        sigma_in_radius=kp.sigma_in_radius,
    )
    out = np.full((len(x_query), pred.shape[1]), np.nan, dtype=pred.dtype)
    out[q_mask] = pred
    return out


def pair_metrics(a: Run, b: Run, kp: KernelParams, kind: str) -> PairMetrics:
    y_b_from_a = kernel_predict(a.x, a.y, b.x, kp)
    y_b_from_b = kernel_predict(b.x, b.y, b.x, kp)
    sigma_rmse = _rmse(b.y, y_b_from_a)
    sigma_self = _rmse(b.y, y_b_from_b)
    return PairMetrics(
        a=a.label,
        b=b.label,
        kind=kind,
        sigma_rmse=sigma_rmse,
        sigma_rre=sigma_rmse / sigma_self if sigma_self > 0 else float("nan"),
        n_a=int(a.x.shape[0]),
        n_b=int(b.x.shape[0]),
    )


def _classify(a: Run, b: Run) -> str | None:
    if a is b:
        return None
    if a.xp != b.xp:
        return "biological"
    if a.basename != b.basename:
        return "technical"
    return None


def _rescale_group_inplace(runs: list[Run], rp: RescalingParams) -> dict:
    floor = rp.floor

    def pool_log(attr: str) -> np.ndarray:
        arrs = [np.maximum(getattr(r, attr), floor) for r in runs]
        stacked = np.concatenate(arrs, axis=0)
        return np.log10(stacked[_finite_rows(stacked)])

    def quantiles(log_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo = np.quantile(log_arr, rp.quantile_low, axis=0)
        hi = np.quantile(log_arr, rp.quantile_high, axis=0)
        return lo, hi, np.maximum(hi - lo, 1e-9)

    qx_lo, qx_hi, rng_x = quantiles(pool_log("x"))
    qy_lo, qy_hi, rng_y = quantiles(pool_log("y"))

    def rescale(arr: np.ndarray, lo: np.ndarray, rng: np.ndarray) -> np.ndarray:
        return np.clip((np.log10(np.maximum(arr, floor)) - lo) / rng, 0.0, 1.0).astype(np.float32)

    for r in runs:
        r.x = rescale(np.asarray(r.x), qx_lo, rng_x)
        r.y = rescale(np.asarray(r.y), qy_lo, rng_y)

    return {
        "floor": float(floor),
        "log10_x_min": qx_lo.tolist(),
        "log10_x_max": qx_hi.tolist(),
        "log10_y_min": qy_lo.tolist(),
        "log10_y_max": qy_hi.tolist(),
    }


def compute_group(
    group_id: str,
    short_name: str,
    fingerprint: str,
    runs: list[Run],
    kp: KernelParams | None = None,
    rp: RescalingParams | None = None,
) -> GroupMetrics:
    kp = kp or KernelParams()
    rp = rp or RescalingParams()
    rescale_info = _rescale_group_inplace(runs, rp)
    out = GroupMetrics(
        group_id=group_id,
        short_name=short_name,
        fingerprint=fingerprint,
        kernel_params=kp,
        rescale=rescale_info,
    )
    for a in runs:
        for b in runs:
            kind = _classify(a, b)
            if kind is not None:
                out.pairs.append(pair_metrics(a, b, kp, kind))
    return out


def to_yaml_dict(gm: GroupMetrics) -> dict:
    return {
        "group_id": gm.group_id,
        "short_name": gm.short_name,
        "fingerprint": gm.fingerprint,
        "kernel_params": asdict(gm.kernel_params),
        "rescale": gm.rescale,
        "biological": {
            "aggregate": gm.aggregate("biological"),
            "pairs": [asdict(p) for p in gm.pairs if p.kind == "biological"],
        },
        "technical": {
            "aggregate": gm.aggregate("technical"),
            "pairs": [asdict(p) for p in gm.pairs if p.kind == "technical"],
        },
    }


def _xp_from_metadata(meta: dict) -> str:
    if meta.get("experiment_name"):
        return meta["experiment_name"]
    nw = meta.get("network")
    if isinstance(nw, dict):
        for k in ("xp", "experiment_name"):
            if nw.get(k):
                return nw[k]
    df = meta.get("datafile")
    if isinstance(df, dict):
        parts = df.get("file", "").split("/")
        if len(parts) > 1 and parts[0] == "Experiments":
            return parts[1]
    return "?"


def runs_from_plotdata(plotdata_list) -> list[Run]:
    out = []
    for d in plotdata_list:
        meta = d.metadata or {}
        xp = _xp_from_metadata(meta)
        base = meta.get("file_stem") or meta.get("network_name") or "?"
        out.append(
            Run(
                label=f"{xp}/{base}",
                xp=xp,
                basename=base,
                x=np.asarray(d.x),
                y=np.asarray(d.y),
                extra=dict(meta),
            )
        )
    return out


def write_yaml(gm: GroupMetrics, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(to_yaml_dict(gm), f, sort_keys=False)
    return str(out)


def pair_panels(
    runs: list[Run],
    gm: GroupMetrics,
    max_points_per_panel: int = 5000,
) -> list[dict]:
    """Square N×N grid of `mvp_panel`-shaped panels, flattened in row-major
    order with `axnum` baked in.

    Cell (i, j) shows scatter of (measured = run_j.Y, predicted = kernel_i(run_j.X)).
    Diagonal cells (self-fit) are kept so the figure has a square layout and
    the smoother's self-consistency is visible. Off-diagonal cells carry
    the σ_RMSE / σ_RRE for that ordered pair as `extra_metrics`.
    """
    rng = np.random.default_rng(0)
    pair_lookup = {(p.a, p.b): p for p in gm.pairs}
    panels: list[dict] = []
    axnum = 0
    for ri in runs:
        for rj in runs:
            y_pred = kernel_predict(ri.x, ri.y, rj.x, gm.kernel_params)
            yt = _first_channel(rj.y)
            yp = _first_channel(y_pred)
            mask = np.isfinite(yp) & np.isfinite(yt)
            yt, yp = yt[mask], yp[mask]
            if yt.size > max_points_per_panel:
                idx = rng.choice(yt.size, max_points_per_panel, replace=False)
                yt, yp = yt[idx], yp[idx]
            same = ri is rj
            p = pair_lookup.get((ri.label, rj.label))
            title = f"{ri.basename}\nself-fit" if same else f"{ri.basename} -> {rj.basename}"
            extras = None if same or p is None else {"σ_RMSE": p.sigma_rmse, "σ_RRE": p.sigma_rre}
            panels.append(
                {
                    "axnum": axnum,
                    "kind": "mvp",
                    "mvp_data": PairMVPData(measured=yt, predicted=yp),
                    "title": title,
                    "extra_metrics": extras,
                    "show_grid_overlay": False,
                }
            )
            axnum += 1
    return panels


def prepare_group(
    group_id: str,
    short_name: str,
    fingerprint: str,
    plotdata_list,
    output_dir: str | Path,
    kp: KernelParams | None = None,
    rp: RescalingParams | None = None,
) -> dict:
    """Side-effect: write `<output_dir>/<group_id>/sigma_repeat.yaml`.
    Returns: rows for the pair-MVP grid, plus run + metric counts. Used by
    `paper-jobs/special_study/replicate_metrics.yaml`."""
    out = Path(output_dir) / group_id
    runs = runs_from_plotdata(plotdata_list)
    if not runs:
        raise ValueError(f"no runs loaded for group {group_id}")
    gm = compute_group(group_id, short_name, fingerprint, runs, kp, rp)
    return {
        "sigma_repeat_yaml": write_yaml(gm, out / "sigma_repeat.yaml"),
        "panels": pair_panels(runs, gm),
        "run_labels": [r.label for r in runs],
        "n_runs": len(runs),
        "n_bio_pairs": sum(1 for p in gm.pairs if p.kind == "biological"),
        "n_tech_pairs": sum(1 for p in gm.pairs if p.kind == "technical"),
    }
