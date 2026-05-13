"""Density of `(projection(X), Y)` pooled across 2-input networks in latent space.

`projection="diff"` (X2 - X1) anchors the single-ERN ReLU paper figure;
`projection="sum"` (X1 + X2) anchors the additive 2-TU paper figure.
Rendering delegates to the canonical `biocomp.plotutils.histogram` so
style auto-binds from `histogram_params` in `default_plotconf_v2`.
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


def _signed_fwd(x: np.ndarray, rescaler: DataRescaler) -> np.ndarray:
    return np.sign(x) * rescaler.fwd(np.abs(x))


def _project(
    x_lat: np.ndarray,
    x_raw: np.ndarray,
    rescaler: DataRescaler,
    x1_idx: int,
    x2_idx: int,
    kind: ProjectionKind,
    in_raw_space: bool,
) -> np.ndarray:
    if in_raw_space:
        x1r = x_raw[:, x1_idx : x1_idx + 1]
        x2r = x_raw[:, x2_idx : x2_idx + 1]
        if kind == "sum":
            return rescaler.fwd(x1r + x2r)
        if kind == "diff":
            return _signed_fwd(x2r - x1r, rescaler)
        raise ValueError(f"unknown projection kind: {kind!r}")
    x1 = x_lat[:, x1_idx : x1_idx + 1]
    x2 = x_lat[:, x2_idx : x2_idx + 1]
    if kind == "diff":
        return x2 - x1
    if kind == "sum":
        return x1 + x2
    raise ValueError(f"unknown projection kind: {kind!r}")


class TwoInputProjectionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plot_data_list: list[Any]
    rescaler: DataRescaler | None = None
    projection: ProjectionKind = "diff"
    x1_idx: int = 0
    x2_idx: int = 1
    xlabel: str | None = None
    ylabel: str = "output"
    mode: Literal["raw", "knn", "knn_uniform"] = "raw"
    grid_resolution: int = 100
    query_seed: int = 0
    latent_xlims: tuple[float, float] = (0.0, 1.0)
    latent_ylims: tuple[float, float] = (0.0, 1.0)
    knn_stats_params: dict = Field(default_factory=lambda: dict(_DEFAULT_KNN_STATS))
    max_centroid_offset_frac: float = 1.0
    # The latent rescaler is log-like, so combining two latent values does
    # arithmetic in *log* space — sum becomes product, diff becomes log
    # ratio. Set True to compute the projection in raw fluorescence then
    # rescale (signed-log for diff), so the axis is in real MEF units.
    project_in_raw_space: bool = False

    _pooled: PlotData = PrivateAttr()
    _density: np.ndarray | None = PrivateAttr(default=None)

    @property
    def _default_xlabel(self) -> str:
        return {"diff": r"$X_2 - X_1$", "sum": r"$X_1 + X_2$"}[self.projection]

    def _knn_smooth(
        self, x_latent: np.ndarray, y_latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_2d = x_latent[:, [self.x1_idx, self.x2_idx]]
        query_mode = "uniform" if self.mode == "knn_uniform" else "grid"
        knn_kw = dict(
            x=x_2d, y=y_latent,
            xlims=list(self.latent_xlims), ylims=list(self.latent_ylims),
            grid_resolution=self.grid_resolution,
            knn_stats_params=dict(self.knn_stats_params),
            max_centroid_offset_frac=self.max_centroid_offset_frac,
            query_mode=query_mode, query_seed=self.query_seed,
        )
        xy_grid, y_mean = _canonical_knn_grid(is_density_plot=False, **knn_kw)
        _, density = _canonical_knn_grid(is_density_plot=True, **knn_kw)
        y_mean = np.asarray(y_mean).reshape(-1)
        density = np.asarray(density).reshape(-1)
        valid = np.isfinite(y_mean) & np.all(np.isfinite(xy_grid), axis=1)
        width = max(self.x1_idx, self.x2_idx) + 1
        x_lat = np.zeros((int(valid.sum()), width), dtype=np.float32)
        x_lat[:, self.x1_idx] = xy_grid[valid, 0]
        x_lat[:, self.x2_idx] = xy_grid[valid, 1]
        return x_lat, y_mean[valid].reshape(-1, 1), density[valid]

    @model_validator(mode="after")
    def _initialize(self):
        rescaler = self.rescaler if self.rescaler is not None else IdentityRescaler()
        projs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        densities: list[np.ndarray] = []
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
            d_pernet: np.ndarray | None = None

            if self.mode in ("knn", "knn_uniform"):
                x_lat, y_lat, d_pernet = self._knn_smooth(x_lat, y_lat)

            x_raw = rescaler.inv(x_lat) if self.project_in_raw_space else x_lat
            proj = _project(
                x_lat, x_raw, rescaler,
                self.x1_idx, self.x2_idx, self.projection,
                self.project_in_raw_space,
            )
            mask = np.isfinite(proj).all(axis=1) & np.isfinite(y_lat).all(axis=1)
            projs.append(proj[mask])
            ys.append(y_lat[mask])
            if d_pernet is not None:
                densities.append(d_pernet[mask])

        assert projs, "TwoInputProjectionData received an empty plot_data_list"
        xpool = np.concatenate(projs, axis=0)
        ypool = np.concatenate(ys, axis=0)
        if densities:
            self._density = np.concatenate(densities, axis=0)
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
                "grid_resolution": self.grid_resolution if self.mode in ("knn", "knn_uniform") else None,
            },
        )
        return self

    @property
    def plot_data(self) -> PlotData:
        return self._pooled

    @property
    def density(self) -> np.ndarray | None:
        """Per-centroid density array (same length as `plot_data.x`), only
        populated in `knn`/`knn_uniform` mode. Returns `None` for `raw`."""
        return self._density


def relu_reference(xs: np.ndarray, ylims: Sequence[float]) -> np.ndarray:
    return np.clip(xs, 0.0, ylims[1])


def identity_reference(xs: np.ndarray, ylims: Sequence[float]) -> np.ndarray:
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
    """Place ticks at rescaler.fwd(±10^k) covering the requested latent range."""
    lims = list(lims)
    lims_inv = rescaler.inv(np.asarray(lims))
    max_abs = max(abs(float(lims_inv[0])), abs(float(lims_inv[1])))
    if tick_floor is None:
        input_range = getattr(rescaler, "input_range", None)
        tick_floor = float(input_range.min) if input_range is not None else 0.1
    p10 = powers_of_ten(tick_floor, max_abs)
    p10 = p10[(p10 >= tick_floor) & (p10 <= max_abs)]
    signed_raw = np.concatenate([-p10[::-1], [0.0], p10]) if symmetric else p10
    latent_positions = rescaler.fwd(signed_raw)
    in_range = (latent_positions >= lims[0]) & (latent_positions <= lims[1])
    latent_positions = latent_positions[in_range]
    signed_raw = signed_raw[in_range]

    min_gap = min_separation * (lims[1] - lims[0])
    kept_positions: list[float] = []
    kept_raw: list[float] = []
    for idx in np.argsort(np.abs(latent_positions)):
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

    Delegates rendering to `biocomp.plotutils.histogram`. When
    `label_rescaler` is given, axes are relabelled in raw fluorescence
    (powers of ten); `x_axis_symmetric` toggles between ±10^k (diff) and
    positive 10^k (sum, output). `reference_curve_fn(xs, ylims)` returns
    y-values for an overlaid dashed reference line.
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
            color="#222222", lw=0.8, alpha=0.7,
            dashes=(4, 4), dash_capstyle="round",
        )
        if reference_curve_kwargs:
            ref_kw.update(reference_curve_kwargs)
        if not ref_kw.get("dashes"):
            ref_kw.pop("dashes", None)
            ref_kw.pop("dash_capstyle", None)
        xs = np.linspace(xlims[0], xlims[1], 200)
        ax.plot(xs, reference_curve_fn(xs, ylims), **ref_kw)
        ax.legend(loc="upper left", fontsize=8, frameon=False)


def prep_log_axis(
    plot_data: PlotData | None = None,
    ax: Axes | None = None,
    rescaler: DataRescaler | None = None,
    xlims: Sequence[float] = (0.0, 1.0),
    ylims: Sequence[float] = (0.0, 1.0),
    x_symmetric: bool = False,
    y_symmetric: bool = False,
    apply_x: bool = True,
    apply_y: bool = False,
    hide_y_ticks: bool = False,
    hide_spines: Sequence[str] = (),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    **_kwargs: Any,
):
    """Empty plot method: set axis lims + signed-log tick labels.

    For panels that exist only to host overlays (e.g. an output
    distribution panel whose overlay draws the curve)."""
    assert ax is not None
    rescaler = rescaler or IdentityRescaler()
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    if apply_x:
        apply_symmetric_log_axis(ax, "x", xlims, rescaler, symmetric=x_symmetric)
    if apply_y:
        apply_symmetric_log_axis(ax, "y", ylims, rescaler, symmetric=y_symmetric)
    if hide_y_ticks:
        ax.set_yticks([])
    for s in hide_spines:
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


