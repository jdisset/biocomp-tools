# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Benchmark figure panels - header, per-item metrics, slice grid composer.

Note on the "merged" panels: in the YAML world, ``render_circuit_merged_left``
and ``render_metrics_merged_right`` spanned multiple axes via ``axes_to_remove``.
In the jeanplot world, spanning is just layout - the caller nests
``BenchmarkCircuitMergedLeftPanel`` / ``BenchmarkMetricsMergedRightPanel``
inside a tall ``Container``, and the renderer hands the Panel its one
laid-out cell.
"""

from typing import Any

import matplotlib.axes

from jeanplot.core.container import Container
from jeanplot.core.models import LayoutConstraints
from jeanplot.panels.base import PlotPanel
from biocomptools.jeanplot_panels.circuit import CircuitPanel


class BenchmarkHeaderPanel(PlotPanel):
    plot_data: None = None
    bench: Any

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.benchmarkutils import render_summary_header

        render_summary_header(ax=ax, bench=self.bench)
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


class BenchmarkMetricsPanel(PlotPanel):
    plot_data: None = None
    item: Any
    bench: Any = None

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.benchmarkutils import render_metrics_panel

        render_metrics_panel(ax=ax, item=self.item, bench=self.bench)
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


class BenchmarkCircuitMergedLeftPanel(CircuitPanel):
    pass


class BenchmarkMetricsMergedRightPanel(BenchmarkMetricsPanel):
    pass

class BenchmarkSliceGridPanel(Container):
    plot_data: Any
    rows: int = 3
    cols: int = 3
    zlims: tuple[float, float] = (0.0, 0.6)
    title_text: str | None = None
    rescaler: Any | None = None
    is_drawable: bool = False

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.children:
            return

        import numpy as np
        from jeanplot.panels.smooth_2d import SmoothPanel2D

        n = self.rows * self.cols
        zs = np.linspace(self.zlims[0], self.zlims[1], n)
        cells = []
        for i, z in enumerate(zs):
            r, c = i // self.cols, i % self.cols
            cell_title = self.title_text if (i == 0 and self.title_text) else f"z={z:.2f}"
            cells.append(
                SmoothPanel2D(
                    plot_data=self.plot_data,
                    rescaler=self.rescaler,
                    zslice=[float(z)],
                    title=cell_title,
                    draw_colorbar=(c == self.cols - 1),
                    draw_xlabel=(r == self.rows - 1),
                    draw_ylabel=(c == 0),
                )
            )

        row_containers = [
            Container(
                layout=LayoutConstraints(direction="row", gap=4),
                children=cells[r * self.cols : (r + 1) * self.cols],
            )
            for r in range(self.rows)
        ]
        self.layout = LayoutConstraints(direction="column", gap=4)
        self.add_children(row_containers)


BenchmarkHeaderPanel.model_rebuild(force=True)
BenchmarkMetricsPanel.model_rebuild(force=True)
BenchmarkCircuitMergedLeftPanel.model_rebuild(force=True)
BenchmarkMetricsMergedRightPanel.model_rebuild(force=True)
BenchmarkSliceGridPanel.model_rebuild(force=True)
