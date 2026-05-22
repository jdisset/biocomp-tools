# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""FluoDensitiesPanel — multi-channel fluorescence KDE strip."""

from typing import Any

import numpy as np
from pydantic import Field

from jeanplot.data import PlotFunctionResult
from jeanplot.panels.base import PlotPanel

from biocomp.plotting.plotting_density import fluo_densities


class FluoDensitiesPanel(PlotPanel):
    rawdata: Any
    channel_names: list[str]
    logscale: bool = True
    res: int = 3000
    bw_method: float = 0.01
    show_quantiles: tuple[float, float] = (0.01, 0.99)
    zero_threshold: float = 4500
    n_inputs: int | None = None
    lpl_threshold: float = 200
    lpl_compression: float = 0.4
    rawdata2: Any | None = None

    plot_data: None = None
    axes_size: Any = Field(default_factory=lambda: __import__(
        "jeanplot.core.models", fromlist=["Size"]
    ).Size(width=1.5 * 4, height=10))

    def draw(self, ax) -> PlotFunctionResult | None:
        n = len(self.channel_names)
        sub_axes = [
            ax.inset_axes([i / n, 0.0, 1 / n, 1.0], transform=ax.transAxes)
            for i in range(n)
        ]
        ax.set_axis_off()

        fluo_densities(
            rawdata=np.asarray(self.rawdata),
            channel_names=list(self.channel_names),
            ax=sub_axes,
            logscale=self.logscale,
            res=self.res,
            bw_method=self.bw_method,
            show_quantiles=self.show_quantiles,
            rawdata2=np.asarray(self.rawdata2) if self.rawdata2 is not None else None,
            lpl_threshold=self.lpl_threshold,
            lpl_compression=self.lpl_compression,
            zero_threshold=self.zero_threshold,
            n_inputs=self.n_inputs,
        )
        if self.title:
            ax.set_title(self.title, **self.title_kwargs)
        return PlotFunctionResult(rendering=None, metadata={})


FluoDensitiesPanel.model_rebuild(force=True)
