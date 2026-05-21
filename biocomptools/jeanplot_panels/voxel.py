# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""SmoothVoxelPanel + BenchmarkDistributionPanel - smooth voxel violins."""

from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class SmoothVoxelPanel(PlotPanel):
    """Smooth voxel-conditioned violin (single or split) on one axes.

    Wraps ``biocomptools.toollib.figuremakers.smoothvoxel.render_smooth_voxel_example``.
    """

    plot_data: None = None
    dataset_file: str
    mode: str = "single"
    model: Any = None
    model_path: str | None = None
    model_name: str | None = None
    min_points_single: int = 60
    min_points_split: int = 80
    xlims: tuple[float, float] = (0.0, 0.7)
    ylims: tuple[float, float] = (0.0, 0.7)

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.smoothvoxel import render_smooth_voxel_example

        render_smooth_voxel_example(
            ax=ax,
            dataset_file=self.dataset_file,
            mode=self.mode,
            model=self.model,
            model_path=self.model_path,
            model_name=self.model_name,
            min_points_single=self.min_points_single,
            min_points_split=self.min_points_split,
            xlims=self.xlims,
            ylims=self.ylims,
            title=self.title,
        )


class BenchmarkDistributionPanel(PlotPanel):
    """Split smooth-voxel violin comparing ground truth vs prediction for one benchmark item.

    Wraps ``biocomptools.toollib.figuremakers.smoothvoxel.render_benchmark_distribution``.
    """

    plot_data: None = None
    item: Any
    bench: Any
    xlims: tuple[float, float] = (0.0, 0.7)
    ylims: tuple[float, float] = (0.0, 0.7)
    show_marginal_kde: bool = False
    tick_count: int = 5
    grid_resolution: int = 32
    draw_xlabel: bool = False
    draw_ylabel: bool = False

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.smoothvoxel import render_benchmark_distribution

        render_benchmark_distribution(
            ax=ax,
            item=self.item,
            bench=self.bench,
            xlims=self.xlims,
            ylims=self.ylims,
            show_marginal_kde=self.show_marginal_kde,
            tick_count=self.tick_count,
            grid_resolution=self.grid_resolution,
            draw_xlabel=self.draw_xlabel,
            draw_ylabel=self.draw_ylabel,
        )
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


SmoothVoxelPanel.model_rebuild(force=True)
BenchmarkDistributionPanel.model_rebuild(force=True)
