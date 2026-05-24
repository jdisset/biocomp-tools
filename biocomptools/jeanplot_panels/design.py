# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class DesignMetricsPanel(PlotPanel):
    plot_data: None = None
    result: Any

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.designutils import render_design_metrics

        render_design_metrics(ax=ax, result=self.result)
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


class LatticeHeatmapPanel(PlotPanel):
    plot_data: None = None
    result: Any
    heatmap_title: str = "Design View (Lattice)"
    cmap: str = "viridis"
    draw_colorbar: bool = False

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.designutils import render_lattice_heatmap

        render_lattice_heatmap(
            ax=ax,
            result=self.result,
            title=self.heatmap_title,
            cmap=self.cmap,
            draw_colorbar=self.draw_colorbar,
        )
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


DesignMetricsPanel.model_rebuild(force=True)
LatticeHeatmapPanel.model_rebuild(force=True)
