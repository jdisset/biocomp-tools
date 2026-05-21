# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class EmptyPanel(PlotPanel):
    plot_data: None = None
    text: str = ""

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.designutils import render_empty_panel

        render_empty_panel(ax=ax, text=self.text)


class ConstantTextPanel(PlotPanel):
    plot_data: None = None
    text: str = ""
    x: float = 0.5
    y: float = 0.5
    ha: str = "center"
    va: str = "center"
    fontsize: float = 10
    color: str = "#333"
    fontweight: str = "normal"
    transform_axes: bool = True

    def draw(self, ax: matplotlib.axes.Axes):
        ax.axis("off")
        kw = {
            "fontsize": self.fontsize,
            "ha": self.ha,
            "va": self.va,
            "color": self.color,
            "fontweight": self.fontweight,
        }
        if self.transform_axes:
            kw["transform"] = ax.transAxes
        ax.text(self.x, self.y, self.text, **kw)
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


EmptyPanel.model_rebuild(force=True)
ConstantTextPanel.model_rebuild(force=True)
