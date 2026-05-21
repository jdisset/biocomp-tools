# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Design-result panels: metrics box, lattice heatmap, full-width diagram."""

from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class DesignMetricsPanel(PlotPanel):
    """Render a design-result info panel (loss, rank, scaffold, fingerprint).

    Wraps ``biocomptools.toollib.figuremakers.designutils.render_design_metrics``.
    """

    plot_data: None = None
    result: Any

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.designutils import render_design_metrics

        render_design_metrics(ax=ax, result=self.result)
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


class LatticeHeatmapPanel(PlotPanel):
    """Pixel-perfect lattice heatmap of a design's optimized landscape.

    Wraps ``biocomptools.toollib.figuremakers.designutils.render_lattice_heatmap``.
    """

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


class NetworkDiagramFullWidthPanel(PlotPanel):
    """Full-width network diagram on a single axes.

    Unlike ``designutils.render_network_diagram_full_width`` (which spans
    two axes by removing them), this Panel renders the diagram on the
    laid-out cell directly via ``render_diagram_to_ax``. Layout-level
    spanning should be expressed by nesting in a wide ``Container``.
    """

    plot_data: None = None
    network: Any
    simplified: bool = True

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

        render_diagram_to_ax(
            network=self.network,
            ax=ax,
            simplified=self.simplified,
            title=self.title,
        )


DesignMetricsPanel.model_rebuild(force=True)
LatticeHeatmapPanel.model_rebuild(force=True)
NetworkDiagramFullWidthPanel.model_rebuild(force=True)
