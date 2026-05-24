# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset

from typing import Any, Literal

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class CircuitPanel(PlotPanel):
    plot_data: None = None
    network: Any
    hide_marker_tus: bool = True
    hide_disabled_tus: bool = False
    disabled_tu_ids: set[str] | None = None
    show_tu_labels: bool = True
    axis_tags: dict[str, str] | None = None
    bias_axis_tag: str | None = None
    orientation: Literal["column", "row"] = "column"
    grid_gap: tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    style_overrides: dict | None = None
    canvas_xlim: tuple[float, float] | None = None
    canvas_ylim: tuple[float, float] | None = None
    aspect: str = "equal"

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

        render_circuit_to_ax(
            network=self.network,
            ax=ax,
            hide_marker_tus=self.hide_marker_tus,
            hide_disabled_tus=self.hide_disabled_tus,
            disabled_tu_ids=self.disabled_tu_ids,
            show_tu_labels=self.show_tu_labels,
            axis_tags=self.axis_tags,
            bias_axis_tag=self.bias_axis_tag,
            orientation=self.orientation,
            grid_gap=self.grid_gap,
            connection_style=self.connection_style,
            style_overrides=self.style_overrides,
            title=self.title,
            canvas_xlim=self.canvas_xlim,
            canvas_ylim=self.canvas_ylim,
            aspect=self.aspect,
        )


CircuitPanel.model_rebuild(force=True)
