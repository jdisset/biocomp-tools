"""Optional post-render overlay layer for `PlotTask`.

Overlays run *after* the base plot has been drawn on a `PlotTask`'s axes.
Each overlay can stamp extra geometry on the axes — typically tagged with
stable `gid` attributes for downstream tooling (the interactive HTML
viewer in `interactive_link.py` is the first consumer) — and returns
JSON-serializable metadata that callers harvest via
`PlotTask._overlay_results`.

The overlay layer is decoupled from base rendering: any `PlotTask` (a
canonical `smooth_2d`, `histogram`, `knn_cell_scatter`, ...) can opt in
by declaring `overlays: [...]` in YAML. Coordinate frames come from the
already-rendered axes (`ax.get_xlim()` etc.), so overlays inherit
whatever the base plot configured.

Overlays bin the *raw* `plot_data` (which is in MEF/raw space as
returned by `DBSource`). They rescale to latent via `plot_config.rescaler`
to match the rendered axes.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle
from pydantic import BaseModel, ConfigDict

from biocomp.datautils import DataRescaler


class Overlay(BaseModel):
    """Base class. Subclasses implement `apply(ax, plot_data, plot_config)`
    and return a JSON-serializable dict (the metadata downstream tooling
    keys off of). `name` becomes the SVG `gid` prefix.

    `rescaler` (optional) overrides `plot_config.rescaler` for the overlay's
    own use. Useful when the surrounding plot_method runs in already-latent
    space (e.g. with `IdentityRescaler` as in `ern_diff_density.yaml`) but
    the overlay needs the original RAW→latent map to bin raw plot_data."""

    name: str
    enabled: bool = True
    rescaler: DataRescaler | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _rescaler(self, plot_config):
        return self.rescaler if self.rescaler is not None else plot_config.rescaler

    def apply(self, ax: Axes, plot_data, plot_config) -> dict:
        raise NotImplementedError


# ---- helpers ---------------------------------------------------------------


def _rescale_xy(plot_data, rescaler):
    """Return (x_lat, y_lat, x_raw, y_raw). plot_data.x/y are RAW (MEF) as
    produced by `DBSource`; we forward through the given rescaler to land
    in latent (axes) space. Caller passes the rescaler explicitly so each
    overlay can decide whether to use its own or `plot_config.rescaler`."""
    x_raw = np.asarray(plot_data.x, dtype=np.float32)
    y_raw = np.asarray(plot_data.y, dtype=np.float32)
    if rescaler is not None:
        x = rescaler.fwd(x_raw)
        y = rescaler.fwd(y_raw)
    else:
        x, y = x_raw, y_raw
    return x.astype(np.float32), y.astype(np.float32), x_raw, y_raw


def _bin_2d(x_arr, y_arr, nx, ny, xmin, xmax, ymin, ymax):
    ix = np.floor((x_arr - xmin) / (xmax - xmin) * nx).astype(np.int32)
    iy = np.floor((y_arr - ymin) / (ymax - ymin) * ny).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(valid, iy * nx + ix, -1).astype(np.int32)


def _pixel_centers_2d(nx, ny, xmin, xmax, ymin, ymax):
    n = nx * ny
    ix = np.arange(n) % nx
    iy = np.arange(n) // nx
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    pixel_xy = np.column_stack([
        xmin + (ix + 0.5) * dx,
        ymin + (iy + 0.5) * dy,
    ]).astype(np.float32)
    return pixel_xy, float(dx), float(dy)


def _emit_rects(ax, name, pixel_count, pixel_xy, dx, dy, color, zorder):
    """Emit one Rectangle per non-empty pixel. alpha=1.0 is critical: it
    forces matplotlib to write `style="fill:<color>"` in SVG. CSS hides
    them by default; downstream JS sets inline `fillOpacity` per gid."""
    n = 0
    for idx in range(int(pixel_count.shape[0])):
        if pixel_count[idx] == 0:
            continue
        cx = float(pixel_xy[idx, 0])
        cy = float(pixel_xy[idx, 1])
        rect = Rectangle(
            (cx - dx / 2, cy - dy / 2), dx, dy,
            facecolor=color, edgecolor="none", linewidth=0,
            alpha=1.0, zorder=zorder,
        )
        rect.set_gid(f"ov-{name}-{idx}")
        ax.add_patch(rect)
        n += 1
    return n


def _grid_payload(name, kind, n_pixels, pixel_count, pixel_xy,
                  pixel_for_raw, xmin, xmax, ymin, ymax, n_emit, **extra):
    return {
        "kind": kind,
        "name": name,
        "n_pixels": int(n_pixels),
        "pixel_count": pixel_count.tolist(),
        "pixel_x": pixel_xy[:, 0].tolist(),
        "pixel_y": pixel_xy[:, 1].tolist(),
        "pixel_for_raw": pixel_for_raw.tolist(),
        "xmin": float(xmin), "xmax": float(xmax),
        "ymin": float(ymin), "ymax": float(ymax),
        "n_emitted_rects": int(n_emit),
        **extra,
    }


# ---- concrete overlays -----------------------------------------------------


class RegularGridOverlay(Overlay):
    """Bin plot_data on a uniform 2D grid; emit a rect per non-empty pixel.

    Defaults bin (x[:,x_col], y[:,0]) — i.e. one input column on the
    panel x-axis and the output on the y-axis. Set `y_col >= 0` to bin
    two input columns (e.g. for X1-vs-X2 smooth panels).
    """

    nx: int = 60
    ny: int = 60
    color: str = "#ff2d2d"
    zorder: int = 100
    x_col: int = 0
    y_col: int = -1  # -1 => use plot_data.y[:,0]
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    def apply(self, ax, plot_data, plot_config) -> dict:
        rescaler = self._rescaler(plot_config)
        x_lat, y_lat, _, _ = _rescale_xy(plot_data, rescaler)
        x_arr = x_lat[:, self.x_col]
        y_arr = y_lat[:, 0] if self.y_col < 0 else x_lat[:, self.y_col]
        xmin, xmax = self.xlim if self.xlim is not None else ax.get_xlim()
        ymin, ymax = self.ylim if self.ylim is not None else ax.get_ylim()
        pixel_for_raw = _bin_2d(x_arr, y_arr, self.nx, self.ny,
                                xmin, xmax, ymin, ymax)
        n_pixels = self.nx * self.ny
        pixel_count = np.bincount(
            pixel_for_raw[pixel_for_raw >= 0], minlength=n_pixels,
        ).astype(np.int32)
        pixel_xy, dx, dy = _pixel_centers_2d(self.nx, self.ny,
                                             xmin, xmax, ymin, ymax)
        n_emit = _emit_rects(ax, self.name, pixel_count, pixel_xy,
                             dx, dy, self.color, self.zorder)
        return _grid_payload(
            self.name, "regular_grid_2d", n_pixels, pixel_count, pixel_xy,
            pixel_for_raw, xmin, xmax, ymin, ymax, n_emit,
        )


class ProjDiffOverlay(Overlay):
    """Bin on a 2D grid where x = signed_log(raw_x[x2_col] - raw_x[x1_col]).

    For two-input subtraction projection plots ("X2 - X1 vs output"). The
    rendered axis is in signed-log raw space, so we project raw values
    forward via `_signed_fwd` to match. y-axis is plot_data.y in latent.
    """

    nx: int = 120
    ny: int = 120
    color: str = "#ff2d2d"
    zorder: int = 100
    x1_col: int = 0
    x2_col: int = 1
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    def apply(self, ax, plot_data, plot_config) -> dict:
        from biocomptools.toollib.figuremakers.latent_projection_density import (
            _signed_fwd,
        )

        rescaler = self._rescaler(plot_config)
        assert rescaler is not None, "ProjDiffOverlay needs a rescaler"
        _, y_lat, x_raw, _ = _rescale_xy(plot_data, rescaler)
        proj = _signed_fwd(
            x_raw[:, self.x2_col] - x_raw[:, self.x1_col], rescaler,
        ).astype(np.float32).ravel()
        y_arr = y_lat[:, 0]

        xmin, xmax = self.xlim if self.xlim is not None else ax.get_xlim()
        ymin, ymax = self.ylim if self.ylim is not None else ax.get_ylim()
        pixel_for_raw = _bin_2d(proj, y_arr, self.nx, self.ny,
                                xmin, xmax, ymin, ymax)
        n_pixels = self.nx * self.ny
        pixel_count = np.bincount(
            pixel_for_raw[pixel_for_raw >= 0], minlength=n_pixels,
        ).astype(np.int32)
        pixel_xy, dx, dy = _pixel_centers_2d(self.nx, self.ny,
                                             xmin, xmax, ymin, ymax)
        n_emit = _emit_rects(ax, self.name, pixel_count, pixel_xy,
                             dx, dy, self.color, self.zorder)
        return _grid_payload(
            self.name, "regular_grid_2d", n_pixels, pixel_count, pixel_xy,
            pixel_for_raw, xmin, xmax, ymin, ymax, n_emit,
        )


class KnnCellOverlay(Overlay):
    """KNN-cell-based selection grid. Each raw bins into the nearest cell
    centroid (computed via canonical `knn_grid`). Emits one rect per
    non-empty cell. Display-axis position uses
    `signed_log(raw_x2 - raw_x1)` when `proj_in_raw_space=True`."""

    color: str = "#ff2d2d"
    zorder: int = 100
    knn_grid_resolution: int = 60
    knn_radius: float = 0.1
    knn_k: int = 500
    knn_min_points: int = 5
    knn_centroid_frac: float = 1.0
    overlay_half_w: float = 0.012
    overlay_half_h: float = 0.012
    proj_in_raw_space: bool = True
    x1_col: int = 0
    x2_col: int = 1
    # KNN grid sampling range in (X1_lat, X2_lat) space. Default [0, 1] is
    # the [0,1]-normalized rescaler space (e.g. EBFP2_compressed_v2). For
    # raw log-space rescalers (e.g. LogPolyLog), widen to cover the data
    # range.
    latent_xlims: tuple[float, float] = (0.0, 1.0)
    latent_ylims: tuple[float, float] = (0.0, 1.0)

    def apply(self, ax, plot_data, plot_config) -> dict:
        from biocomp.plotting.plotting_smooth import knn_grid as canonical_knn_grid
        from biocomptools.toollib.figuremakers.latent_projection_density import (
            _signed_fwd,
        )
        from scipy.spatial import cKDTree

        rescaler = self._rescaler(plot_config)
        assert rescaler is not None, "KnnCellOverlay needs a rescaler"

        x_lat, y_lat, _, _ = _rescale_xy(plot_data, rescaler)
        x_2d = x_lat[:, [self.x1_col, self.x2_col]]
        y_col = y_lat[:, 0:1]

        knn_stats = {"k": self.knn_k, "min_points": self.knn_min_points,
                     "radius": self.knn_radius}
        xy_grid_lat, y_mean = canonical_knn_grid(
            x=x_2d, y=y_col,
            xlims=list(self.latent_xlims), ylims=list(self.latent_ylims),
            is_density_plot=False,
            grid_resolution=self.knn_grid_resolution,
            knn_stats_params=dict(knn_stats),
            max_centroid_offset_frac=self.knn_centroid_frac,
        )
        y_mean = np.asarray(y_mean).reshape(-1)
        valid = np.isfinite(y_mean) & np.all(np.isfinite(xy_grid_lat), axis=1)
        x1_v = xy_grid_lat[valid, 0].astype(np.float32)
        x2_v = xy_grid_lat[valid, 1].astype(np.float32)
        y_v = y_mean[valid].astype(np.float32)

        if self.proj_in_raw_space:
            x1_raw = rescaler.inv(x1_v)
            x2_raw = rescaler.inv(x2_v)
            display_x = _signed_fwd(
                x2_raw - x1_raw, rescaler,
            ).astype(np.float32).ravel()
        else:
            display_x = (x2_v - x1_v).astype(np.float32)

        tree = cKDTree(np.column_stack([x1_v, x2_v]))
        _, raw_to_cell = tree.query(x_2d)
        pixel_for_raw = raw_to_cell.astype(np.int32)
        n_pixels = int(valid.sum())
        pixel_count = np.bincount(pixel_for_raw, minlength=n_pixels).astype(np.int32)
        pixel_xy = np.column_stack([display_x, y_v]).astype(np.float32)
        dx = 2 * self.overlay_half_w
        dy = 2 * self.overlay_half_h
        n_emit = _emit_rects(ax, self.name, pixel_count, pixel_xy,
                             dx, dy, self.color, self.zorder)
        return _grid_payload(
            self.name, "knn_cell_2d", n_pixels, pixel_count, pixel_xy,
            pixel_for_raw,
            ax.get_xlim()[0], ax.get_xlim()[1],
            ax.get_ylim()[0], ax.get_ylim()[1],
            n_emit,
        )


class OutputDistributionOverlay(Overlay):
    """1D selection grid along the panel x-axis (= plot_data.y in latent).

    Emits a single placeholder polygon with gid `ov-{name}-curve`;
    downstream JS rewrites its `d` attribute on hover to draw the
    selection's density curve. Also exposes the 1D forward map
    raw->bin_idx for the JS payload builder."""

    n_bins: int = 100
    bandwidth_bins: float = 1.5
    color: str = "#ff2d2d"
    zorder: int = 100
    xlim: tuple[float, float] | None = None

    def apply(self, ax, plot_data, plot_config) -> dict:
        _, y_lat, _, _ = _rescale_xy(plot_data, self._rescaler(plot_config))
        y_arr = y_lat[:, 0]
        xmin, xmax = self.xlim if self.xlim is not None else ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        rng = xmax - xmin
        nb = int(self.n_bins)
        bins = np.floor((y_arr - xmin) / rng * nb).astype(np.int32)
        valid = (bins >= 0) & (bins < nb)
        pixel_for_raw = np.where(valid, bins, -1).astype(np.int32)
        pixel_count = np.bincount(bins[valid], minlength=nb).astype(np.int32)
        bin_centers = xmin + (np.arange(nb) + 0.5) * rng / nb

        placeholder = Polygon(
            [[xmin, ymin], [xmax, ymin]],
            closed=True, facecolor=self.color, edgecolor="none",
            linewidth=0, alpha=1.0, zorder=self.zorder,
        )
        placeholder.set_gid(f"ov-{self.name}-curve")
        ax.add_patch(placeholder)

        return {
            "kind": "output_distribution_1d",
            "name": self.name,
            "n_pixels": int(nb),
            "pixel_count": pixel_count.tolist(),
            "pixel_x": bin_centers.astype(np.float32).tolist(),
            "pixel_y": [0.0] * nb,
            "pixel_for_raw": pixel_for_raw.tolist(),
            "xmin": float(xmin), "xmax": float(xmax),
            "ymin": float(ymin), "ymax": float(ymax),
            "bandwidth_bins": float(self.bandwidth_bins),
            "n_emitted_rects": 1,
        }


class BrushHitRect(Overlay):
    """Invisible capture rectangle. Stamps a stable `hit-{name}` gid on
    the axes; downstream interactive viewers attach mouse handlers to it."""

    color: str = "white"
    zorder: int = 200

    def apply(self, ax, plot_data, plot_config) -> dict:
        hit = Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
            facecolor=self.color, edgecolor="none",
            alpha=1.0, zorder=self.zorder,
        )
        hit.set_gid(f"hit-{self.name}")
        ax.add_patch(hit)
        return {"kind": "hit_rect", "name": self.name}


OVERLAY_TYPES = [
    Overlay,
    RegularGridOverlay,
    ProjDiffOverlay,
    KnnCellOverlay,
    OutputDistributionOverlay,
    BrushHitRect,
]
