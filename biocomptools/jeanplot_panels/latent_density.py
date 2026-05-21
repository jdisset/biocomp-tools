# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""LatentProjectionHistogramPanel - 2-input latent projection density histogram."""

from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class LatentProjectionHistogramPanel(PlotPanel):
    """Density histogram of a 2-input latent projection (e.g. ERN diff).

    Wraps ``biocomptools.toollib.figuremakers.latent_projection_density.histogram``.
    """

    plot_data: Any
    rescaler: Any | None = None
    label_rescaler: Any | None = None
    x_axis_symmetric: bool = True
    reference_curve_fn: Any | None = None
    reference_curve_kwargs: dict | None = None
    histogram_kwargs: dict | None = None

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.latent_projection_density import histogram

        histogram(
            plot_data=self.plot_data,
            ax=ax,
            rescaler=self.rescaler,
            label_rescaler=self.label_rescaler,
            x_axis_symmetric=self.x_axis_symmetric,
            reference_curve_fn=self.reference_curve_fn,
            reference_curve_kwargs=self.reference_curve_kwargs,
            **(self.histogram_kwargs or {}),
        )
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


LatentProjectionHistogramPanel.model_rebuild(force=True)
