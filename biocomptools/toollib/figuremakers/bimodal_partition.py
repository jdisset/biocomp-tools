"""Class-partition scatters for 2-input circuits.

`BimodalPartitionData` pools `(X1, X2, y)` (raw + latent) once and assigns
each point an integer class id via a configurable rule. Each rule defines
its own class names and colors. Two render functions share the partition:
`bimodal_proj_scatter` for `(projection, y)` and `bimodal_xy_scatter` for
`(X1, X2)`.

Projection: `sum` → `rescaler.fwd(X1_raw + X2_raw)` (additive 2-TU);
            `diff` → signed-log `X2_raw - X1_raw` (single-ERN).

Rules:
  `sum_residual`         — 2 classes (main / off-trend) for additive 2-TU.
  `raw_diff_y_threshold` — 2 classes for single-ERN "should be ON, isn't".
  `ern_4_quadrant`       — 4 classes for single-ERN: X1-dominant suppressed,
                           ON (y high), X2-dominant off-trend, intermediate.
"""

from typing import Any, Callable, Literal, Sequence

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from biocomp.datautils import DataRescaler, IdentityRescaler
from biocomptools.toollib.figuremakers.latent_projection_density import (
    apply_symmetric_log_axis,
)


def _signed_fwd(x: np.ndarray, rescaler: DataRescaler) -> np.ndarray:
    return np.sign(x) * rescaler.fwd(np.abs(x))


ProjectionKind = Literal["sum", "diff"]
RuleKind = Literal["sum_residual", "raw_diff_y_threshold", "ern_partition"]


_BIMODAL_NAMES = ["A", "B"]
_BIMODAL_COLORS = ["#2A6FB8", "#D6422F"]
_ERN_NAMES = ["Zero", "A", "B", "C", "D", "E"]
_ERN_COLORS = ["#2A2A2A", "#E89635", "#4D8AC2", "#2A9D5C", "#D6422F", "#A0A0A0"]


class BimodalPartitionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plot_data_list: list[Any]
    rescaler: DataRescaler | None = None
    x1_idx: int = 0
    x2_idx: int = 1
    projection: ProjectionKind = "sum"
    rule: RuleKind = "sum_residual"
    residual_threshold: float = 0.08
    raw_diff_threshold: float = 3000.0
    raw_y_threshold: float = 3000.0
    diff_threshold: float = 3000.0
    y_threshold: float = 3000.0
    x1_threshold: float = 3000.0

    _x1_raw: np.ndarray = PrivateAttr()
    _x2_raw: np.ndarray = PrivateAttr()
    _y_raw: np.ndarray = PrivateAttr()
    _x1: np.ndarray = PrivateAttr()
    _x2: np.ndarray = PrivateAttr()
    _y: np.ndarray = PrivateAttr()
    _proj: np.ndarray = PrivateAttr()
    _class_ids: np.ndarray = PrivateAttr()
    _class_names: list[str] = PrivateAttr(default_factory=list)
    _class_colors: list[str] = PrivateAttr(default_factory=list)
    _input_names: list[str] = PrivateAttr(default_factory=list)
    _output_name: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _initialize(self):
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        x1r_l, x2r_l, yr_l = [], [], []
        x1l_l, x2l_l, yl_l, projl_l = [], [], [], []
        for pd in self.plot_data_list:
            x = np.asarray(pd.x, dtype=np.float32)
            y = np.asarray(pd.y, dtype=np.float32)
            assert x.shape[1] > max(self.x1_idx, self.x2_idx), (
                f"need indices x1={self.x1_idx} x2={self.x2_idx}, got {x.shape[1]} cols"
            )
            assert y.shape[1] == 1, "BimodalPartitionData expects a single output channel"
            x1r = x[:, self.x1_idx]
            x2r = x[:, self.x2_idx]
            yr = y[:, 0]
            x_lat = rescaler.fwd(x)
            y_lat = rescaler.fwd(y)
            if self.projection == "sum":
                proj_lat = rescaler.fwd(x1r + x2r)
            else:
                proj_lat = _signed_fwd(x2r - x1r, rescaler)
            x1r_l.append(x1r)
            x2r_l.append(x2r)
            yr_l.append(yr)
            x1l_l.append(x_lat[:, self.x1_idx])
            x2l_l.append(x_lat[:, self.x2_idx])
            yl_l.append(y_lat[:, 0])
            projl_l.append(proj_lat)
        x1_raw = np.concatenate(x1r_l).astype(np.float32)
        x2_raw = np.concatenate(x2r_l).astype(np.float32)
        y_raw = np.concatenate(yr_l).astype(np.float32)
        x1 = np.concatenate(x1l_l).astype(np.float32)
        x2 = np.concatenate(x2l_l).astype(np.float32)
        yv = np.concatenate(yl_l).astype(np.float32)
        proj = np.concatenate(projl_l).astype(np.float32)
        finite = (
            np.isfinite(x1) & np.isfinite(x2) & np.isfinite(yv)
            & np.isfinite(proj) & np.isfinite(x1_raw) & np.isfinite(x2_raw) & np.isfinite(y_raw)
        )
        self._x1_raw = x1_raw[finite]
        self._x2_raw = x2_raw[finite]
        self._y_raw = y_raw[finite]
        self._x1 = x1[finite]
        self._x2 = x2[finite]
        self._y = yv[finite]
        self._proj = proj[finite]
        self._class_ids, self._class_names, self._class_colors = self._classify()
        first = self.plot_data_list[0]
        self._input_names = list(getattr(first, "input_names", []))
        self._output_name = str(getattr(first, "output_name", ""))
        return self

    def _classify(self) -> tuple[np.ndarray, list[str], list[str]]:
        if self.rule == "sum_residual":
            ids = ((self._y - self._proj) < -self.residual_threshold).astype(np.int32)
            return ids, list(_BIMODAL_NAMES), list(_BIMODAL_COLORS)
        if self.rule == "raw_diff_y_threshold":
            diff = self._x2_raw - self._x1_raw
            ids = ((diff > self.raw_diff_threshold) & (self._y_raw < self.raw_y_threshold)).astype(np.int32)
            return ids, list(_BIMODAL_NAMES), list(_BIMODAL_COLORS)
        if self.rule == "ern_partition":
            diff = self._x2_raw - self._x1_raw
            zero = (self._x1_raw < self.x1_threshold) & (self._x2_raw < self.x1_threshold)
            x1_absent = (self._x1_raw < self.x1_threshold) & ~zero
            x1_dom = (diff < -self.diff_threshold) & ~zero & ~x1_absent
            on = (self._y_raw > self.y_threshold) & ~zero & ~x1_absent & ~x1_dom
            x2_dom = (diff > self.diff_threshold) & ~zero & ~x1_absent & ~x1_dom & ~on
            ids = np.full(self._y_raw.shape, 5, dtype=np.int32)
            ids[zero] = 0
            ids[x1_absent] = 1
            ids[x1_dom] = 2
            ids[on] = 3
            ids[x2_dom] = 4
            return ids, list(_ERN_NAMES), list(_ERN_COLORS)
        raise ValueError(f"unknown rule: {self.rule!r}")

    @property
    def x1(self) -> np.ndarray: return self._x1
    @property
    def x2(self) -> np.ndarray: return self._x2
    @property
    def y(self) -> np.ndarray: return self._y
    @property
    def proj(self) -> np.ndarray: return self._proj
    @property
    def class_ids(self) -> np.ndarray: return self._class_ids
    @property
    def class_names(self) -> list[str]: return self._class_names
    @property
    def class_colors(self) -> list[str]: return self._class_colors
    @property
    def n_classes(self) -> int: return len(self._class_names)
    @property
    def class_counts(self) -> list[int]:
        return [int((self._class_ids == i).sum()) for i in range(self.n_classes)]

    def class_mask(self, class_id: int) -> np.ndarray:
        return self._class_ids == class_id

    @property
    def main_mask(self) -> np.ndarray: return self.class_mask(0)
    @property
    def second_mask(self) -> np.ndarray: return self.class_mask(1)
    @property
    def n_main(self) -> int: return int(self.main_mask.sum())
    @property
    def n_second(self) -> int: return int(self.second_mask.sum())
    @property
    def input_names(self) -> list[str]: return self._input_names
    @property
    def output_name(self) -> str: return self._output_name
    @property
    def x1_name(self) -> str:
        return self._input_names[self.x1_idx] if self.x1_idx < len(self._input_names) else "X1"
    @property
    def x2_name(self) -> str:
        return self._input_names[self.x2_idx] if self.x2_idx < len(self._input_names) else "X2"


def _balanced_subsample(
    n_total: int, mask: np.ndarray, target: int | None, rng: np.random.Generator
) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return idx
    if target is None:
        return idx
    if target <= 0:
        return idx[:0]
    take = min(idx.size, max(1, int(round(target * idx.size / n_total))))
    return rng.choice(idx, size=take, replace=False)


def _scatter_classes(
    ax: Axes,
    data: BimodalPartitionData,
    xs: np.ndarray,
    ys: np.ndarray,
    n_sample: int | None,
    seed: int,
    classes: Sequence[int],
    marker_size: float,
    marker_alpha: float,
):
    rng = np.random.default_rng(seed)
    counts = data.class_counts
    n_total = sum(counts[i] for i in classes)
    base_kw = dict(s=marker_size, alpha=marker_alpha, linewidths=0, rasterized=True)
    for cls in classes:
        idx = _balanced_subsample(n_total, data.class_mask(cls), n_sample, rng)
        ax.scatter(
            xs[idx], ys[idx],
            color=data.class_colors[cls],
            label=f"{data.class_names[cls]} (n={counts[cls]})",
            **base_kw,
        )


def _apply_log_labels(
    ax: Axes,
    xlims: Sequence[float],
    ylims: Sequence[float],
    label_rescaler: DataRescaler | None,
    x_axis_symmetric: bool,
    y_axis_symmetric: bool,
):
    if label_rescaler is not None:
        apply_symmetric_log_axis(ax, "x", xlims, label_rescaler, symmetric=x_axis_symmetric)
        apply_symmetric_log_axis(ax, "y", ylims, label_rescaler, symmetric=y_axis_symmetric)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])


def _resolve_classes(
    data: BimodalPartitionData, classes: Sequence[int] | None
) -> Sequence[int]:
    return list(range(data.n_classes)) if classes is None else classes


def bimodal_proj_scatter(
    data: BimodalPartitionData,
    ax: Axes,
    n_sample: int | None = None,
    classes: Sequence[int] | None = None,
    label_rescaler: DataRescaler | None = None,
    marker_size: float = 3.0,
    marker_alpha: float = 0.35,
    xlims: Sequence[float] = (-0.05, 0.8),
    ylims: Sequence[float] = (-0.05, 0.8),
    x_axis_symmetric: bool = False,
    xtitle: str | None = None,
    ytitle: str | None = None,
    reference_curve_fn: Callable[[np.ndarray, Sequence[float]], np.ndarray] | None = None,
    reference_label: str = r"$y = X_1 + X_2$",
    reference_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    **_,
):
    """Scatter of pooled `(projection, y)` colored by class."""
    classes = _resolve_classes(data, classes)
    _scatter_classes(
        ax, data, data.proj, data.y, n_sample, seed,
        classes, marker_size, marker_alpha,
    )
    if reference_curve_fn is not None:
        ref_kw: dict[str, Any] = dict(
            color="#222222", lw=1.0, alpha=0.7, dashes=(5, 5), dash_capstyle="round",
            label=reference_label,
        )
        if reference_kwargs:
            ref_kw.update(reference_kwargs)
        if not ref_kw.get("dashes"):
            ref_kw.pop("dashes", None)
            ref_kw.pop("dash_capstyle", None)
        xs = np.linspace(xlims[0], xlims[1], 200)
        ax.plot(xs, reference_curve_fn(xs, ylims), **ref_kw)
    _apply_log_labels(
        ax, xlims, ylims, label_rescaler,
        x_axis_symmetric=x_axis_symmetric, y_axis_symmetric=False,
    )
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    ax.legend(loc="upper left", fontsize=7, frameon=False, markerscale=2.5)


def bimodal_xy_scatter(
    data: BimodalPartitionData,
    ax: Axes,
    n_sample: int | None = None,
    classes: Sequence[int] | None = None,
    label_rescaler: DataRescaler | None = None,
    marker_size: float = 3.0,
    marker_alpha: float = 0.35,
    xlims: Sequence[float] = (-0.05, 1.0),
    ylims: Sequence[float] = (-0.05, 1.0),
    xtitle: str | None = None,
    ytitle: str | None = None,
    seed: int = 0,
    **_,
):
    """Scatter of pooled `(X1, X2)` colored by class."""
    classes = _resolve_classes(data, classes)
    _scatter_classes(
        ax, data, data.x1, data.x2, n_sample, seed,
        classes, marker_size, marker_alpha,
    )
    _apply_log_labels(
        ax, xlims, ylims, label_rescaler,
        x_axis_symmetric=False, y_axis_symmetric=False,
    )
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    ax.legend(loc="upper left", fontsize=7, frameon=False, markerscale=2.5)
