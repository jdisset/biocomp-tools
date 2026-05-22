# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any, Literal, Sequence

from jeanplot.core.container import Container
from jeanplot.core.models import LayoutConstraints, Size
from jeanplot.panels.smooth_1d import SmoothPanel1D
from jeanplot.panels.smooth_2d import SmoothPanel2D
from jeanplot.panels.smooth_3d import SmoothPanel3D

from biocomptools.jeanplot_panels.blurb import BlurbPanel
from biocomptools.jeanplot_panels.circuit import CircuitPanel
from biocomptools.jeanplot_panels.data import (
    NetworkPlotData,
    NetworkPredictedPlotData,
)
from biocomptools.jeanplot_panels.mvp_network import MVPNetworkPanel
from biocomptools.jeanplot_panels.network_diagram import NetworkDiagramPanel


# Default per-kind widths (inches). 1D/2D/3D for `data` are mode-dependent.
_DEFAULT_KIND_WIDTHS: dict[str, Any] = {
    "diagram": 5.0,
    "circuit": 5.0,
    "blurb": 4.0,
    "data": {1: 4.0, 2: 5.0, 3: 10.0},
    "mvp": 5.0,
    "mvp_floor": 5.0,
    "slices": 8.0,
}


def _resolve_data_dim(pd: Any) -> int:
    if hasattr(pd, "dimensions"):
        return int(pd.dimensions.input)
    x = pd.x if hasattr(pd, "x") else pd.xval
    return int(x.shape[1] if x.ndim > 1 else 1)


def _as_jeanplot_pd(pd: Any, rescaler: Any = None):
    from jeanplot.data.plot_data import PlotData as JeanplotPlotData
    from biocomptools.jeanplot_panels.data import _biocomp_to_jeanplot

    if isinstance(pd, (NetworkPlotData, NetworkPredictedPlotData)):
        return _biocomp_to_jeanplot(pd.source, rescaler=rescaler)
    if isinstance(pd, JeanplotPlotData):
        return pd
    return _biocomp_to_jeanplot(pd, rescaler=rescaler)


def _resolve_width(kind: str, kind_widths: dict, pd: Any | None = None) -> float:
    kw = kind_widths.get(kind)
    if kw is None:
        return 5.0
    if isinstance(kw, dict):
        assert pd is not None, f"width for {kind!r} is dim-keyed; pass plot_data"
        dim = _resolve_data_dim(pd)
        return float(kw[dim])
    return float(kw)


def _drop_none(d: dict | None) -> dict:
    # YAML knobs default to None to mean "absent"; strip them before splat
    # so Pydantic field defaults apply instead of clashing with non-Optional types.
    return {k: v for k, v in (d or {}).items() if v is not None}


def _build_data_panel(
    pd: Any,
    *,
    title: str | None,
    rescaler: Any | None,
    slice_grid_kwargs: dict,
) -> Container:
    jpd = _as_jeanplot_pd(pd, rescaler=rescaler)
    dim = _resolve_data_dim(jpd)
    if dim == 1:
        return SmoothPanel1D(plot_data=jpd, rescaler=rescaler, title=title)
    if dim == 2:
        return SmoothPanel2D(plot_data=jpd, rescaler=rescaler, title=title)
    if dim == 3:
        return SmoothPanel3D(
            plot_data=jpd, rescaler=rescaler, title=title, **slice_grid_kwargs
        )
    raise ValueError(f"unsupported data dim={dim}")


def _build_slices_only_panel(
    pd: Any,
    *,
    title: str | None,
    rescaler: Any | None,
    slice_grid_kwargs: dict,
) -> Container:
    import numpy as np

    from jeanplot.core.container import Container as _C
    from jeanplot.core.models import LayoutConstraints as _LC

    jpd = _as_jeanplot_pd(pd, rescaler=rescaler)
    rows, cols = slice_grid_kwargs.get("slice_grid", (3, 3))
    zslices = slice_grid_kwargs.get("zslices", [0.05, 0.4])
    n = rows * cols
    zs = np.linspace(float(zslices[0]), float(zslices[-1]), n)
    cells = []
    for i, z in enumerate(zs):
        r, c = i // cols, i % cols
        cell_title = f"{title}  z={z:.2f}" if (title and r == 0 and c == 0) else f"z={z:.2f}"
        cells.append(
            SmoothPanel2D(
                plot_data=jpd,
                rescaler=rescaler,
                zslice=[float(z)],
                title=cell_title,
                draw_colorbar=(c == cols - 1),
                draw_xlabel=(r == rows - 1),
                draw_ylabel=(c == 0),
            )
        )
    row_containers = [
        _C(
            layout=_LC(direction="row", gap=4),
            children=cells[r * cols : (r + 1) * cols],
        )
        for r in range(rows)
    ]
    return _C(layout=_LC(direction="column", gap=4), children=row_containers)


def _gap_spacer(width: float) -> Container:
    return Container(min_dimensions=Size(width=float(width), height=0.0))


def _wrap_cell(child: Container, width: float, height: float) -> Container:
    child.min_dimensions = Size(width=float(width), height=float(height))
    return child


def build_per_network_row(
    *,
    panels: Sequence[str],
    plot_data: Any,
    predicted_data: Any | None = None,
    mvp_data: Any | None = None,
    blurb_text: str | None = None,
    blurb_title: str | None = None,
    mvp_extras: dict | None = None,
    show_mvp_grid_overlay: bool = True,
    network: Any | None = None,
    layout: Literal["row", "stacked"] = "row",
    kind_widths: dict | None = None,
    row_height: float = 5.0,
    rescaler: Any | None = None,
    slice_grid_kwargs: dict | None = None,
) -> Container:
    """Build a per-network plotting row as a nested ``Container`` tree.

    ``panels`` is the canonical surface for what's shown - same shape as
    the old ``build_rows(panels=...)``. ``mvp_global`` is also accepted
    (single MVP cell over a global ``mvp_data``).

    ``layout="row"``: one Container(direction=row) with each cell.
    ``layout="stacked"``: a Container(direction=column) where each cell
    sits on its own row.
    """
    kw = dict(_DEFAULT_KIND_WIDTHS, **_drop_none(kind_widths))
    slice_grid_kwargs = _drop_none(slice_grid_kwargs)
    needs_pred_data = predicted_data is not None
    needs_mvp = mvp_data is not None

    if network is None and hasattr(plot_data, "metadata"):
        network = plot_data.metadata.get("built_network")

    cells: list[Container] = []

    def add_with_gap(cell: Container, gap_before: float | None, gap_after: float | None):
        if gap_before is not None and cells:
            cells.append(_gap_spacer(gap_before))
        cells.append(cell)
        if gap_after is not None:
            cells.append(_gap_spacer(gap_after))

    for kind in panels:
        if kind in ("ground_truth", "prediction") and kind == "prediction" and not needs_pred_data:
            continue
        if kind in ("mvp_row", "mvp_floor", "mvp_global") and not needs_mvp:
            continue

        if kind == "diagram":
            assert network is not None, (
                "diagram panel needs `network` (or plot_data.metadata['built_network'])"
            )
            cell = NetworkDiagramPanel(network=network, title="Network")
            add_with_gap(_wrap_cell(cell, _resolve_width("diagram", kw), row_height), None, None)
        elif kind == "circuit":
            assert network is not None, "circuit panel needs `network`"
            cell = CircuitPanel(network=network, title="Circuit")
            add_with_gap(_wrap_cell(cell, _resolve_width("circuit", kw), row_height), None, None)
        elif kind == "blurb":
            cell = BlurbPanel(text=blurb_text or "", title=blurb_title)
            add_with_gap(_wrap_cell(cell, _resolve_width("blurb", kw), row_height), 1.0, 2.0)
        elif kind == "ground_truth":
            cell = _build_data_panel(
                plot_data,
                title="Ground Truth",
                rescaler=rescaler,
                slice_grid_kwargs=slice_grid_kwargs,
            )
            add_with_gap(
                _wrap_cell(cell, _resolve_width("data", kw, plot_data), row_height), None, None
            )
        elif kind == "prediction":
            cell = _build_data_panel(
                predicted_data,
                title="Prediction",
                rescaler=rescaler,
                slice_grid_kwargs=slice_grid_kwargs,
            )
            add_with_gap(
                _wrap_cell(cell, _resolve_width("data", kw, predicted_data), row_height), 3.0, None
            )
        elif kind == "ground_truth_slices":
            cell = _build_slices_only_panel(
                plot_data, title="GT", rescaler=rescaler, slice_grid_kwargs=slice_grid_kwargs
            )
            add_with_gap(_wrap_cell(cell, _resolve_width("slices", kw), row_height), None, None)
        elif kind == "prediction_slices":
            cell = _build_slices_only_panel(
                predicted_data, title="Pred", rescaler=rescaler, slice_grid_kwargs=slice_grid_kwargs
            )
            add_with_gap(_wrap_cell(cell, _resolve_width("slices", kw), row_height), None, None)
        elif kind == "mvp_row":
            cell = MVPNetworkPanel(
                mvp_data=mvp_data,
                mode="mvp",
                title="Measured vs Predicted",
                extra_metrics=mvp_extras,
                show_grid_overlay=show_mvp_grid_overlay,
            )
            add_with_gap(_wrap_cell(cell, _resolve_width("mvp", kw), row_height), 0.75, 0.75)
        elif kind == "mvp_floor":
            cell = MVPNetworkPanel(
                mvp_data=mvp_data,
                mode="floor",
                title="Noise floor",
                show_grid_overlay=False,
            )
            add_with_gap(_wrap_cell(cell, _resolve_width("mvp_floor", kw), row_height), 3.0, None)
        elif kind == "mvp_global":
            cell = MVPNetworkPanel(
                mvp_data=mvp_data,
                mode="mvp",
                title="MVP (all networks)",
                show_grid_overlay=show_mvp_grid_overlay,
            )
            add_with_gap(_wrap_cell(cell, _resolve_width("mvp", kw), row_height), None, None)
        else:
            # unknown / unsupported panel kind: skip silently to match
            # the legacy `build_rows` behaviour for `mvp_global` etc.
            continue

    if not cells:
        return Container(layout=LayoutConstraints(direction="row"))

    if layout == "row":
        return Container(
            layout=LayoutConstraints(direction="row", gap=4, align_items="stretch"),
            children=cells,
            min_dimensions=Size(width=0.0, height=float(row_height)),
        )

    if layout == "stacked":
        # Each cell on its own row; drop pure gap spacers between rows.
        rows = [
            Container(
                layout=LayoutConstraints(direction="row", gap=4, align_items="stretch"),
                children=[c],
                min_dimensions=Size(width=0.0, height=float(row_height)),
            )
            for c in cells
            if not (isinstance(c, Container) and c.min_dimensions.height == 0.0 and not c.children)
        ]
        return Container(
            layout=LayoutConstraints(direction="column", gap=4, align_items="stretch"),
            children=rows,
        )

    raise ValueError(f"build_per_network_row: unknown layout {layout!r}")


__all__ = ["build_per_network_row"]
