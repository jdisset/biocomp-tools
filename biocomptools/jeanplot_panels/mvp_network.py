# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any, Literal

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class MVPNetworkPanel(PlotPanel):
    plot_data: None = None
    mvp_data: Any
    mode: Literal["mvp", "floor"] = "mvp"
    show_grid_overlay: bool = True
    extra_metrics: dict | None = None
    noise_floor: float | None = None

    def draw(self, ax: matplotlib.axes.Axes):
        mvp = self.mvp_data
        if self.mode == "floor":
            from biocomp.plotting.plotting_mvp import noise_floor_panel

            assert mvp.noise_floor_measured is not None, (
                "MVPNetworkPanel(mode='floor') requires compute_noise_floor=True"
            )
            noise_floor_panel(
                ax=ax,
                measured=mvp.noise_floor_measured,
                predicted=mvp.noise_floor_predicted,
                rescaler=mvp.rescaler,
                title=self.title or "Noise floor",
                extra_metrics=self.extra_metrics,
                grid_measured=mvp.grid_measured if self.show_grid_overlay else None,
                grid_predicted=mvp.grid_measured if self.show_grid_overlay else None,
                grid_weights=mvp.grid_weights if self.show_grid_overlay else None,
            )
            return

        from biocomp.plotting.plotting_mvp import measured_vs_predicted

        measured_vs_predicted(
            ax=ax,
            measured=mvp.measured,
            predicted=mvp.predicted,
            kernel_predicted=None,
            rescaler=mvp.rescaler,
            title=self.title,
            extra_metrics=self.extra_metrics,
            noise_floor=self.noise_floor,
            grid_measured=mvp.grid_measured if self.show_grid_overlay else None,
            grid_predicted=mvp.grid_predicted if self.show_grid_overlay else None,
            grid_weights=mvp.grid_weights if self.show_grid_overlay else None,
        )


MVPNetworkPanel.model_rebuild(force=True)
