# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""σ_repeat: pairwise replicate-divergence metrics via Gaussian kNN smoothing.

For ordered pair (A, B):
    σ_RMSE(A->B) = RMSE(B.Y, kernel_A(B.X))                  # cross-fit
    σ_RRE(A->B)  = σ_RMSE(A->B) / RMSE(B.Y, kernel_B(B.X))   # relative
    eRMSE(A->B)  = sqrt(max(0, xMSE(A->B) - kMSE(B)))        # excess over B's floor

bioRMSE / techRMSE aggregate eRMSE in MSE space: sqrt(max(0, mean xMSE - mean kMSE))
over all ordered pairs of the kind. The data analog of model eRMSE (B's own kernel
replaces the model), so model vs biological reproducibility compare on one scale.

Two consumers downstream:

- `compute_group(...)` + `write_yaml(...)` - write the per-group metric YAML.
- `pair_panels(...)` - return a list of `mvp_panel`-compatible dicts (one per
  ordered (i,j) pair, including diagonal self-fits) that can be fed straight
  into `paper-jobs/plot/figures/autofig_dataset_row.yaml` rows. Bypasses the
  `MeasuredVsPredictedData` model (which is NetworkPrediction-shaped) by
  exposing a tiny duck-type `PairMVPData`.
"""

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from numpy.typing import NDArray as NdArray
from pydantic import BaseModel, ConfigDict

from biocomp.metric_utils import ermse, DEFAULT_GRIDSTATS_PARAMS
from biocomptools.toollib.kernel_floor import lattice_kernel_predict
from jeanplot.knn import make_tree, get_knn_mean_only

# A floor/cross predictor: (x_train, y_train, x_query) -> prediction at x_query.
Predict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# A rescale step: mutate runs in place to a target latent space, return info.
Rescale = Callable[[list["Run"]], dict]


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
    mse_cross: float
    mse_self: float
    ermse: float
    n_a: int
    n_b: int


@dataclass
class GroupMetrics:
    group_id: str
    short_name: str
    fingerprint: str
    estimator: str = "adaptive_knn"
    floor_params: dict = field(default_factory=dict)
    rescale: dict = field(default_factory=dict)
    pairs: list[PairMetrics] = field(default_factory=list)

    def aggregate(self, kind: str) -> dict:
        vals = [p for p in self.pairs if p.kind == kind]
        if not vals:
            return {"n_pairs": 0}
        rmse = np.array([p.sigma_rmse for p in vals], dtype=float)
        rre = np.array([p.sigma_rre for p in vals], dtype=float)
        cross = np.array([p.mse_cross for p in vals], dtype=float)
        floor = np.array([p.mse_self for p in vals], dtype=float)
        per_pair_ermse = np.array([p.ermse for p in vals], dtype=float)
        return {
            "n_pairs": len(vals),
            # cell-noise: pooled kernel self-fit floor (kRMSE). The term eRMSE subtracts.
            "cell_noise": float(np.sqrt(np.nanmean(floor))),
            # rep-noise: eRMSE aggregated in MSE space (bioRMSE for biological pairs).
            "ermse_pooled": ermse(float(np.nanmean(cross)), float(np.nanmean(floor))),
            "median_ermse": float(np.nanmedian(per_pair_ermse)),
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


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = (y_true - y_pred).reshape(-1)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.mean(diff[finite] ** 2))


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
    _assert_in_train_range(out, y_train)
    return out


def _assert_in_train_range(pred: np.ndarray, y_train: np.ndarray) -> None:
    """Kernel smoothers (kNN-mean, lattice) are convex combos of y_train, so a
    finite prediction outside its range is corruption (e.g. a stale neighbour
    cache). Cheap loud guard for the metrics pipeline."""
    finite = np.isfinite(pred)
    if not finite.any():
        return
    lo, hi = float(np.nanmin(y_train)), float(np.nanmax(y_train))
    vmin, vmax = float(pred[finite].min()), float(pred[finite].max())
    assert lo - 1e-3 <= vmin and vmax <= hi + 1e-3, (
        f"kernel prediction [{vmin:.4g},{vmax:.4g}] outside train y-range "
        f"[{lo:.4g},{hi:.4g}] -- corrupt neighbour cache?"
    )


@dataclass(frozen=True)
class FloorEstimator:
    """A kernel noise-floor predictor plus metadata for recording. ``predict``
    maps (x_train, y_train, x_query) -> prediction at x_query."""

    name: str
    predict: Predict
    params: dict


def adaptive_knn_floor(kp: KernelParams | None = None) -> FloorEstimator:
    """Direct adaptive-sigma Gaussian-kNN self/cross fit at the data points."""
    kp = kp or KernelParams()
    return FloorEstimator(
        "adaptive_knn", lambda xt, yt, xq: kernel_predict(xt, yt, xq, kp), asdict(kp)
    )


def lattice_floor(params: dict | None = None) -> FloorEstimator:
    """Model-side lattice kernel floor (same estimator + params as model eRMSE).
    Defaults to ``DEFAULT_GRIDSTATS_PARAMS`` so rep-noise lands on the model's
    ruler. Pair with `model_fwd_rescaler` to also share the latent space."""
    params = dict(params or DEFAULT_GRIDSTATS_PARAMS)
    predict = lambda xt, yt, xq: _bounded(lattice_kernel_predict(xt, yt, xq, params), yt)  # noqa: E731
    return FloorEstimator("lattice", predict, params)


def _bounded(pred: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    _assert_in_train_range(pred, y_train)
    return pred


def pair_metrics(a: Run, b: Run, predict: Predict, kind: str) -> PairMetrics:
    y_b_from_a = predict(a.x, a.y, b.x)
    y_b_from_b = predict(b.x, b.y, b.x)
    mse_cross = _mse(b.y, y_b_from_a)
    mse_self = _mse(b.y, y_b_from_b)
    sigma_rmse = float(np.sqrt(mse_cross))
    sigma_self = float(np.sqrt(mse_self))
    return PairMetrics(
        a=a.label,
        b=b.label,
        kind=kind,
        sigma_rmse=sigma_rmse,
        sigma_rre=sigma_rmse / sigma_self if sigma_self > 0 else float("nan"),
        mse_cross=mse_cross,
        mse_self=mse_self,
        ermse=ermse(mse_cross, mse_self),
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


def pooled_log_quantile_rescaler(rp: RescalingParams | None = None) -> Rescale:
    """Default model-independent rescaling: per-group pooled log10-quantile -> [0,1]."""
    rp = rp or RescalingParams()
    return lambda runs: _rescale_group_inplace(runs, rp)


def model_fwd_rescaler(rescaler) -> Rescale:
    """Map each run into a model's latent space via ``rescaler.fwd`` (raw->latent),
    the transform `NetworkModel.predict_unscaled` applies. Pair with `lattice_floor`
    to put rep-noise on the model eRMSE ruler (same floor + same latent space)."""

    def _apply(runs: list[Run]) -> dict:
        for r in runs:
            r.x = np.asarray(rescaler.fwd(np.asarray(r.x)), dtype=np.float32)
            r.y = np.asarray(rescaler.fwd(np.asarray(r.y)), dtype=np.float32)
        return {"rescaler": type(rescaler).__name__}

    return _apply


def identity_rescaler() -> Rescale:
    """No-op: runs already carry data in the target latent space."""
    return lambda runs: {"rescaler": "identity"}


def compute_group(
    group_id: str,
    short_name: str,
    fingerprint: str,
    runs: list[Run],
    estimator: FloorEstimator | None = None,
    rescale: Rescale | None = None,
) -> GroupMetrics:
    estimator = estimator or adaptive_knn_floor()
    rescale = rescale or pooled_log_quantile_rescaler()
    rescale_info = rescale(runs)
    out = GroupMetrics(
        group_id=group_id,
        short_name=short_name,
        fingerprint=fingerprint,
        estimator=estimator.name,
        floor_params=estimator.params,
        rescale=rescale_info,
    )
    for a in runs:
        for b in runs:
            kind = _classify(a, b)
            if kind is not None:
                out.pairs.append(pair_metrics(a, b, estimator.predict, kind))
    return out


def to_yaml_dict(gm: GroupMetrics) -> dict:
    return {
        "group_id": gm.group_id,
        "short_name": gm.short_name,
        "fingerprint": gm.fingerprint,
        "estimator": gm.estimator,
        "floor_params": gm.floor_params,
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
    predict: Predict,
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
            y_pred = predict(ri.x, ri.y, rj.x)
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
            extras = None if same or p is None else {"eRMSE": p.ermse, "σ_RRE": p.sigma_rre}
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
    estimator: FloorEstimator | None = None,
    rescale: Rescale | None = None,
) -> dict:
    """Side-effect: write `<output_dir>/<group_id>/sigma_repeat.yaml`.
    Returns: rows for the pair-MVP grid, plus run + metric counts. Used by
    `paper-jobs/special_study/replicate_metrics.yaml`."""
    out = Path(output_dir) / group_id
    estimator = estimator or adaptive_knn_floor()
    runs = runs_from_plotdata(plotdata_list)
    if not runs:
        raise ValueError(f"no runs loaded for group {group_id}")
    gm = compute_group(group_id, short_name, fingerprint, runs, estimator, rescale)
    bio, tech = gm.aggregate("biological"), gm.aggregate("technical")
    return {
        "sigma_repeat_yaml": write_yaml(gm, out / "sigma_repeat.yaml"),
        "panels": pair_panels(runs, gm, estimator.predict),
        "run_labels": [r.label for r in runs],
        "n_runs": len(runs),
        "n_bio_pairs": sum(1 for p in gm.pairs if p.kind == "biological"),
        "n_tech_pairs": sum(1 for p in gm.pairs if p.kind == "technical"),
        "bio_cell_noise": bio.get("cell_noise"),
        "bio_rep_noise": bio.get("ermse_pooled"),
        "tech_cell_noise": tech.get("cell_noise"),
        "tech_rep_noise": tech.get("ermse_pooled"),
    }


def summary_rows(metadata: dict, plots_dir: str | Path) -> list[dict]:
    """One row per group with biological replicates, reduced from the written
    `<gid>/sigma_repeat.yaml`. cell-noise = kRMSE floor, rep-noise = bioRMSE."""
    base = Path(plots_dir)
    rows = []
    for gid, meta in metadata.items():
        if not meta.get("has_bio"):
            continue
        f = base / gid / "sigma_repeat.yaml"
        if not f.exists():
            continue
        with open(f) as fh:
            agg = (yaml.safe_load(fh).get("biological") or {}).get("aggregate") or {}
        cn = agg.get("cell_noise")
        if cn is None or not np.isfinite(cn):  # stale (pre-rename) or failed group
            continue
        rows.append(
            {
                "group_id": gid,
                "short_name": meta.get("short_name", gid),
                "n_xps": meta.get("n_xps"),
                "n_runs": meta.get("n_total_runs"),
                "n_bio_pairs": agg.get("n_pairs"),
                "cell_noise": agg.get("cell_noise"),
                "rep_noise": agg.get("ermse_pooled"),
            }
        )
    return rows


def render_summary_markdown(rows: list[dict]) -> str:
    head = "| Network | n_xp | n_runs | cell-noise (kRMSE) | rep-noise (bioRMSE) |"
    sep = "|---|---|---|---|---|"

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "na"

    lines = [
        f"| {r['short_name']} | {r['n_xps']} | {r['n_runs']} | {fmt(r['cell_noise'])} | {fmt(r['rep_noise'])} |"
        for r in rows
    ]
    cn = [r["cell_noise"] for r in rows if isinstance(r["cell_noise"], (int, float))]
    rn = [r["rep_noise"] for r in rows if isinstance(r["rep_noise"], (int, float))]
    if cn and rn:
        lines.append(
            f"| **mean ({len(rows)} groups)** | | | **{np.mean(cn):.3f}** | **{np.mean(rn):.3f}** |"
        )
    return "\n".join([head, sep, *lines])


def write_summary(metadata: dict, plots_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    """Write `summary.csv` + `summary.md` from the per-group YAMLs; return the
    markdown plus the rows."""
    rows = summary_rows(metadata, plots_dir)
    md = render_summary_markdown(rows)
    out = Path(out_dir or plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["group_id", "short_name", "n_xps", "n_runs", "n_bio_pairs", "cell_noise", "rep_noise"]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(md + "\n")
    return {"markdown": md, "rows": rows, "n_groups": len(rows)}
