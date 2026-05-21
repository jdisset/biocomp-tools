# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""NetworkDiagramPanel - jeanplot adapter for biocomp compute-graph diagrams."""

from typing import Any, Literal

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class NetworkDiagramPanel(PlotPanel):
    """Render a biocomp ``Network``'s compute-graph diagram on its axes.

    Thin wrapper around
    ``biocomptools.toollib.figuremakers.networkdiagram.render_diagram_to_ax``.
    """

    plot_data: None = None
    network: Any
    simplified: bool = True
    disabled_tu_ids: set[str] | None = None
    style_overrides: dict | None = None
    show_ratios: bool = False
    ratio_normalization: Literal["min", "sum"] = "sum"
    variable_thickness: bool = False
    show_edge_parts: bool = False
    thickness_range: tuple[float, float] = (0.5, 4.0)
    layout_spec: Any | None = None
    canvas_xlim: tuple[float, float] | None = None
    canvas_ylim: tuple[float, float] | None = None
    aspect: str = "equal"

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

        render_diagram_to_ax(
            network=self.network,
            ax=ax,
            simplified=self.simplified,
            disabled_tu_ids=self.disabled_tu_ids,
            style_overrides=self.style_overrides,
            title=self.title,
            show_ratios=self.show_ratios,
            ratio_normalization=self.ratio_normalization,
            variable_thickness=self.variable_thickness,
            show_edge_parts=self.show_edge_parts,
            thickness_range=self.thickness_range,
            layout_spec=self.layout_spec,
            canvas_xlim=self.canvas_xlim,
            canvas_ylim=self.canvas_ylim,
            aspect=self.aspect,
        )


NetworkDiagramPanel.model_rebuild(force=True)
