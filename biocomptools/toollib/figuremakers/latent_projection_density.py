"""Density of `(projection(X), Y)` pooled across 2-input networks in latent space.

Generic plotting building block that subsumes the original "ERN diff density"
plot. The same machinery powers two paper figures:

- `projection="diff"`  → density of `(X2 - X1, Y)` for single-ERN circuits.
  A dashed `y = max(0, X2 - X1)` reference anchors the ReLU shape.
- `projection="sum"`   → density of `(X1 + X2, Y)` for two-TU additive circuits.
  A dashed `y = X1 + X2` reference anchors the linear-sum shape.

`TwoInputProjectionData` pools `(proj(x), y)` pairs across networks, mapping
both inputs and outputs through `rescaler.fwd()` into latent space before
projecting. Two preprocessing modes:

- `mode="raw"`: every raw datapoint becomes one (proj, y) entry.
- `mode="knn"`: per-network we first build a Gaussian-weighted KNN grid over
  the original (X1, X2) → Y latent surface, then feed the smoothed grid
  cells into the pool — filtering single-point outliers before projecting.

Rendering delegates to the canonical paper density primitive
(`biocomp.plotutils.histogram`), so style knobs auto-bind from
`histogram_params` in `default_plotconf_v2`. The thin `histogram` wrapper
relabels axes in raw fluorescence (MEFL) and overlays an arbitrary
reference curve `y = reference_curve_fn(x, ylims)`.
"""

from typing import Any, Callable, Literal, Sequence

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from biocomp.datautils import DataRescaler, IdentityRescaler
from biocomp.plotting.plotting_core import powers_of_ten
from biocomp.plotting.plotting_smooth import knn_grid as _canonical_knn_grid
from biocomp.plotutils import PlotData
from biocomp.plotutils import histogram as _canonical_histogram
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


_DEFAULT_KNN_STATS = {"k": 500, "min_points": 5, "radius": 0.1}

ProjectionKind = Literal["diff", "sum"]

ReferenceFn = Callable[[np.ndarray, Sequence[float]], np.ndarray]


def _project(x_lat: np.ndarray, x1_idx: int, x2_idx: int, kind: ProjectionKind) -> np.ndarray:
    x1 = x_lat[:, x1_idx : x1_idx + 1]
    x2 = x_lat[:, x2_idx : x2_idx + 1]
    if kind == "diff":
        return x2 - x1
    if kind == "sum":
        return x1 + x2
    raise ValueError(f"unknown projection kind: {kind!r}")


class TwoInputProjectionData(BaseModel):
    """Pool `proj(x_lat) vs y_lat` across 2-input PlotData objects.

    X and Y are mapped through `rescaler.fwd()` into latent space before the
    projection is applied. The resulting synthetic `PlotData` is ready to
    feed into `biocomp.plotutils.histogram` — pair it with an
    `IdentityRescaler` on the plot config to avoid double-transformation.

    When `mode="knn"`, each network's (X1, X2) → Y latent surface is first
    smoothed on a `grid_resolution` × `grid_resolution` grid via a
    Gaussian-weighted KNN mean. The grid cells then replace the raw points
    as inputs to the pool, filtering single-point noise before projecting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plot_data_list: list[Any]
    rescaler: DataRescaler | None = None
    projection: ProjectionKind = "diff"
    x1_idx: int = 0
    x2_idx: int = 1
    xlabel: str | None = None
    ylabel: str = "output"
    mode: Literal["raw", "knn"] = "raw"
    grid_resolution: int = 100
    latent_xlims: tuple[float, float] = (0.0, 1.0)
    latent_ylims: tuple[float, float] = (0.0, 1.0)
    knn_stats_params: dict = Field(default_factory=lambda: dict(_DEFAULT_KNN_STATS))

    _pooled: PlotData = PrivateAttr()

    @property
    def _default_xlabel(self) -> str:
        return {"diff": r"$X_2 - X_1$", "sum": r"$X_1 + X_2$"}[self.projection]

    def _knn_smooth(
        self, x_latent: np.ndarray, y_latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """KNN-smooth (X1, X2) → Y on a regular latent-space grid.

        Returns (pixel_coords (M, 2), pixel_values (M, 1)) — one row per
        grid cell with enough neighbors for a valid mean.
        """
        x_2d = x_latent[:, [self.x1_idx, self.x2_idx]]
        xy_grid, y_mean = _canonical_knn_grid(
            x=x_2d,
            y=y_latent,
            xlims=list(self.latent_xlims),
            ylims=list(self.latent_ylims),
            is_density_plot=False,
            grid_resolution=self.grid_resolution,
            knn_stats_params=dict(self.knn_stats_params),
        )
        y_mean = np.asarray(y_mean).reshape(-1)
        valid = np.isfinite(y_mean) & np.all(np.isfinite(xy_grid), axis=1)
        return xy_grid[valid], y_mean[valid].reshape(-1, 1)

    @model_validator(mode="after")
    def _initialize(self):
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        projs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        n_input_points = 0
        for pd in self.plot_data_list:
            x = np.asarray(pd.x, dtype=np.float32)
            y = np.asarray(pd.y, dtype=np.float32)
            assert x.shape[1] > max(self.x1_idx, self.x2_idx), (
                f"Network {pd.metadata.get('network_name', '?')} has {x.shape[1]} inputs, "
                f"need indices x1={self.x1_idx} x2={self.x2_idx}"
            )
            assert y.shape[1] == 1, (
                f"Network {pd.metadata.get('network_name', '?')} has {y.shape[1]} outputs, "
                "TwoInputProjectionData expects a single output channel"
            )
            n_input_points += x.shape[0]
            x_lat = rescaler.fwd(x)
            y_lat = rescaler.fwd(y)

            if self.mode == "knn":
                pixel_xy, pixel_y = self._knn_smooth(x_lat, y_lat)
                width = max(self.x1_idx, self.x2_idx) + 1
                x_lat = np.zeros((pixel_xy.shape[0], width), dtype=np.float32)
                x_lat[:, self.x1_idx] = pixel_xy[:, 0]
                x_lat[:, self.x2_idx] = pixel_xy[:, 1]
                y_lat = pixel_y

            proj = _project(x_lat, self.x1_idx, self.x2_idx, self.projection)
            mask = np.isfinite(proj).all(axis=1) & np.isfinite(y_lat).all(axis=1)
            projs.append(proj[mask])
            ys.append(y_lat[mask])

        assert projs, "TwoInputProjectionData received an empty plot_data_list"
        xpool = np.concatenate(projs, axis=0)
        ypool = np.concatenate(ys, axis=0)
        logger.info(
            f"TwoInputProjectionData[{self.projection}/{self.mode}]: pooled "
            f"{xpool.shape[0]} points across {len(self.plot_data_list)} networks "
            f"(from {n_input_points} raw input points; "
            f"{self.projection} range [{xpool.min():.3f}, {xpool.max():.3f}])"
        )
        self._pooled = PlotData(
            xval=xpool,
            yval=ypool,
            input_names=[self.xlabel or self._default_xlabel],
            output_name=self.ylabel,
            metadata={
                "projection": self.projection,
                "mode": self.mode,
                "n_networks": len(self.plot_data_list),
                "n_points": int(xpool.shape[0]),
                "n_raw_points": int(n_input_points),
                "grid_resolution": self.grid_resolution if self.mode == "knn" else None,
            },
        )
        return self

    @property
    def plot_data(self) -> PlotData:
        return self._pooled


def relu_reference(xs: np.ndarray, ylims: Sequence[float]) -> np.ndarray:
    """`y = max(0, x)` clipped to the y-range — matches the diff-projection plot."""
    return np.clip(xs, 0.0, ylims[1])


def identity_reference(xs: np.ndarray, ylims: Sequence[float]) -> np.ndarray:
    """`y = x` clipped to the y-range — matches the sum-projection plot."""
    return np.clip(xs, ylims[0], ylims[1])


def _format_power(signed_raw: float) -> str:
    if signed_raw == 0:
        return "0"
    e = int(round(np.log10(abs(signed_raw))))
    sign = "-" if signed_raw < 0 else ""
    return rf"${sign}10^{{{e}}}$"


def apply_symmetric_log_axis(
    ax: Axes,
    axis: str,
    lims: Sequence[float],
    rescaler: DataRescaler,
    symmetric: bool,
    tick_floor: float | None = None,
    min_separation: float = 0.05,
):
    """Place ticks at rescaler.fwd(±10^k) covering the requested latent range.

    `symmetric=True` mirrors powers-of-ten across zero (appropriate for the
    diff axis, which can be negative); `symmetric=False` uses only positive
    powers (appropriate for non-negative axes — sum, output).

    `tick_floor` excludes powers of ten below that raw-space magnitude.
    `min_separation` culls tick-label crowding near zero.
    """
    lims = list(lims)
    lims_inv = rescaler.inv(np.asarray(lims))
    max_abs = max(abs(float(lims_inv[0])), abs(float(lims_inv[1])))
    if tick_floor is None:
        input_range = getattr(rescaler, "input_range", None)
        tick_floor = float(input_range.min) if input_range is not None else 0.1
    p10 = powers_of_ten(tick_floor, max_abs)
    p10 = p10[(p10 >= tick_floor) & (p10 <= max_abs)]
    if symmetric:
        signed_raw = np.concatenate([-p10[::-1], [0.0], p10])
    else:
        signed_raw = p10
    latent_positions = rescaler.fwd(signed_raw)
    in_range = (latent_positions >= lims[0]) & (latent_positions <= lims[1])
    latent_positions = latent_positions[in_range]
    signed_raw = signed_raw[in_range]

    min_gap = min_separation * (lims[1] - lims[0])
    kept_positions: list[float] = []
    kept_raw: list[float] = []
    order = np.argsort(np.abs(latent_positions))
    for idx in order:
        pos = float(latent_positions[idx])
        if all(abs(pos - kp) >= min_gap for kp in kept_positions):
            kept_positions.append(pos)
            kept_raw.append(float(signed_raw[idx]))
    sort_idx = np.argsort(kept_positions)
    kept_positions = [kept_positions[i] for i in sort_idx]
    kept_raw = [kept_raw[i] for i in sort_idx]

    getattr(ax, f"set_{axis}lim")(lims[0], lims[1])
    getattr(ax, f"set_{axis}ticks")(kept_positions)
    getattr(ax, f"set_{axis}ticklabels")([_format_power(v) for v in kept_raw])
    ax.tick_params(axis=axis, which="minor", length=0)
    # grid_histogram enables major+minor gridlines at tick positions; a
    # vertical gridline at x=0 overlays the density as a visible streak.
    # Disable gridlines on the log-tick axis we just (re)placed.
    getattr(ax, f"{axis}axis").grid(False, which="both")


def histogram(
    plot_data: PlotData,
    ax: Axes,
    rescaler: DataRescaler | None = None,
    label_rescaler: DataRescaler | None = None,
    x_axis_symmetric: bool = True,
    reference_curve_fn: ReferenceFn | None = None,
    reference_curve_kwargs: dict | None = None,
    **histogram_kwargs: Any,
):
    """Latent-space (projection, Y) density plot for 2-input circuits.

    Delegates density rendering to `biocomp.plotutils.histogram` — the
    canonical paper primitive — so cmap, use_log_density, colorbar_params,
    noise_smooth, and vlims auto-bind from `histogram_params` in
    `default_plotconf_v2`. Rendering is identical for `raw` / `knn` modes
    of `TwoInputProjectionData` (the difference is what points feed in).

    `x_axis_symmetric=True` uses ±10^k MEFL ticks (diff projection); set
    `False` for non-negative projections (sum). `reference_curve_fn(xs,
    ylims)` returns y-values for an overlaid dashed reference line; pair
    with `reference_curve_kwargs` for line style and label.
    """
    rescaler = rescaler or IdentityRescaler()
    xlims = histogram_kwargs.get("xlims", ax.get_xlim())
    ylims = histogram_kwargs.get("ylims", ax.get_ylim())

    _canonical_histogram(plot_data=plot_data, ax=ax, rescaler=rescaler, **histogram_kwargs)

    if label_rescaler is not None:
        apply_symmetric_log_axis(ax, "x", xlims, label_rescaler, symmetric=x_axis_symmetric)
        apply_symmetric_log_axis(ax, "y", ylims, label_rescaler, symmetric=False)
    else:
        xt = [float(t) for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5) if xlims[0] <= t <= xlims[1]]
        yt = [float(t) for t in (0.0, 0.25, 0.5, 0.75, 1.0) if ylims[0] <= t <= ylims[1]]
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([f"{t:g}" for t in xt])
        ax.set_yticklabels([f"{t:g}" for t in yt])
        ax.tick_params(axis="both", which="minor", length=0)

    if reference_curve_fn is not None:
        ref_kw: dict[str, Any] = dict(
            color="#222222",
            lw=0.8,
            alpha=0.7,
            dashes=(4, 4),
            dash_capstyle="round",
        )
        if reference_curve_kwargs:
            ref_kw.update(reference_curve_kwargs)
        # `dashes=None` (or empty) means "solid": drop the kwarg so
        # matplotlib falls back to the default solid linestyle instead of
        # silently rendering nothing when handed a list like [None, None].
        if not ref_kw.get("dashes"):
            ref_kw.pop("dashes", None)
            ref_kw.pop("dash_capstyle", None)
        xs = np.linspace(xlims[0], xlims[1], 200)
        ys = reference_curve_fn(xs, ylims)
        ax.plot(xs, ys, **ref_kw)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
