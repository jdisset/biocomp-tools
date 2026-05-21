# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""CircuitPanel - jeanplot adapter for biocomp genetic-circuit schematics."""

from typing import Any, Literal

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class CircuitPanel(PlotPanel):
    """Render a biocomp ``Network``'s genetic-circuit schematic on its axes.

    Thin wrapper around
    ``biocomptools.toollib.figuremakers.geneticcircuit.render_circuit_to_ax``.
    All knobs of the underlying renderer are exposed as typed fields.
    """

    plot_data: None = None
    network: Any
    hide_marker_tus: bool = True
    hide_disabled_tus: bool = False
    disabled_tu_ids: set[str] | None = None
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
            grid_gap=self.grid_gap,
            connection_style=self.connection_style,
            style_overrides=self.style_overrides,
            title=self.title,
            canvas_xlim=self.canvas_xlim,
            canvas_ylim=self.canvas_ylim,
            aspect=self.aspect,
        )


CircuitPanel.model_rebuild(force=True)
