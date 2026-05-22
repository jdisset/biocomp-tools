# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import List, Any
import matplotlib.axes
from pydantic import ConfigDict

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.figuremakers.uorfmatrixfigure import (
    UORFMatrixFigure as _UORFMatrixFigure,
    GridPlotConfig,
)
from biocomp.plotutils import PlotData


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
        if self.plot_config is not None:
            kw["plot_config"] = self.plot_config
        if self.rescaler is not None:
            kw["rescaler"] = self.rescaler
        _UORFMatrixFigure(**kw).draw_into(ax)


UORFMatrixPanel.model_rebuild(force=True)
