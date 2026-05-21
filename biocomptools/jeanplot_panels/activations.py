# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""ActivationsPanel - overlay of ReLU/GELU/SELU reference curves in latent space."""

from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class ActivationsPanel(PlotPanel):
    """Render ReLU/GELU/SELU curves on an existing latent-space axis.

    Wraps ``biocomptools.toollib.figuremakers.activation_reference.plot_activations``.
    """

    plot_data: None = None
    xlims: tuple[float, float] = (-0.7, 0.7)
    ylims: tuple[float, float] = (-0.05, 0.7)
    label_rescaler: Any | None = None
    activations: tuple[str, ...] = ("relu", "gelu", "selu")
    n_points: int = 400
    xtitle_label: str = r"$X_2 - X_1$ (fluorescence diff)"
    ytitle_label: str = "output fluorescence (MEF)"
    line_kwargs: dict | None = None

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.activation_reference import plot_activations

        plot_activations(
            ax=ax,
            xlims=self.xlims,
            ylims=self.ylims,
            label_rescaler=self.label_rescaler,
            activations=self.activations,
            n_points=self.n_points,
            xtitle=self.xtitle_label,
            ytitle=self.ytitle_label,
            line_kwargs=self.line_kwargs,
        )
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


ActivationsPanel.model_rebuild(force=True)
