# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import List, Literal, Optional, Any
import matplotlib.axes
from pydantic import ConfigDict

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.figuremakers.uorfmatrixfigure import (
    UORFMatrixFigure as _UORFMatrixFigure,
    GridPlotConfig,
)
from biocomp.plotutils import PlotData


# forwarded verbatim to the figuremaker
_FORWARDED = (
    "plot_config",
    "rescaler",
    "cell_kind",
    "slice_x_held_raw",
    "slice_y_held_raw",
    "slice_x_cmap",
    "slice_y_cmap",
    "slice_cmap_range",
    "slice_xlims",
    "slice_vlims",
    "slice_knn_stats_params",
    "slice_lineplot_props",
    "slice_res",
)


class UORFMatrixPanel(PlotPanel):
    plot_data: List[PlotData] = []
    plot_config: Any = None
    grid_plotconfigs: List[GridPlotConfig] = []
    rescaler: Any = None
    annotate: list[tuple[int, int]] = []
    show_individual_rmse: bool = True
    show_overall_rmse: bool = True
    overall_rmse_fontsize: int = 8
    rmse_fontsize: int = 8
    rmse_prefix: str = "RMSE: "

    cell_kind: Literal["heatmap", "slices"] = "heatmap"
    slice_x_held_raw: List[float] = []
    slice_y_held_raw: List[float] = []
    slice_x_cmap: str = "bc_reds"
    slice_y_cmap: str = "bc_greens"
    slice_cmap_range: tuple[float, float] = (0.35, 0.9)
    # None = auto-scale to data union; per-entry None falls back to auto
    slice_xlims: Optional[List[Optional[float]]] = None
    slice_vlims: Optional[List[Optional[float]]] = None
    slice_knn_stats_params: dict = {}
    slice_lineplot_props: dict = {"lw": 1.2, "marker": ""}
    slice_res: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self, ax: matplotlib.axes.Axes) -> None:
        kw = {
            "plot_data": self.plot_data,
            "grid_plotconfigs": self.grid_plotconfigs,
            "annotate": self.annotate,
            "show_individual_rmse": self.show_individual_rmse,
            "show_overall_rmse": self.show_overall_rmse,
            "overall_rmse_fontsize": self.overall_rmse_fontsize,
            "rmse_fontsize": self.rmse_fontsize,
            "rmse_prefix": self.rmse_prefix,
        }
        for name in _FORWARDED:
            v = getattr(self, name)
            if v is not None:
                kw[name] = v
        _UORFMatrixFigure(**kw).draw_into(ax)


UORFMatrixPanel.model_rebuild(force=True)
