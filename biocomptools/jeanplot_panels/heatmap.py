# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Literal

import matplotlib.axes
import pandas as pd
from pydantic import ConfigDict, Field

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.analysis.generalization.heatmap_figure import (
    DataMode, ClassDataMode, HeatmapConfig,
    draw_horizontal_heatmap_to_ax, draw_class_summary_to_ax,
    _resolve_horizontal_heatmap, _resolve_class_summary,
)
from biocomptools.toollib.analysis.generalization.pivot_build import load_metrics_csv
from biocomptools.toollib.analysis.generalization.heatmap_math import (
    build_class_pivot, build_network_pivot,
)
from biocomptools.toollib.analysis.generalization.views import GenViewConfig


class HorizontalHeatmapPanel(PlotPanel):
    plot_data: None = None
    view: GenViewConfig
    view_name: str = ""
    heatmap_conf: HeatmapConfig = Field(default_factory=HeatmapConfig)
    dataframe_path: str | None = None
    df: pd.DataFrame | None = None
    loss_filter: str | None = "regression"
    metric: str = "grid_nrmse"
    data_mode: DataMode = "abs"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self, ax: matplotlib.axes.Axes):
        df = self.df if self.df is not None else load_metrics_csv(self.dataframe_path or "")
        pivot, net_meta, conds = build_network_pivot(
            df, self.view, metric=self.metric, loss_filter=self.loss_filter,
        )
        data, row_order, stats, cmap, norm, title, cb_label = _resolve_horizontal_heatmap(
            pivot.values, net_meta, conds,
            view=self.view, view_name=self.view_name, data_mode=self.data_mode,
            loss_filter=self.loss_filter, cfg=self.heatmap_conf,
        )
        draw_horizontal_heatmap_to_ax(
            ax, data, net_meta, row_order, stats, cmap, norm, title, cb_label,
            self.view, self.heatmap_conf,
        )


class ClassSummaryHeatmapPanel(PlotPanel):
    plot_data: None = None
    view: GenViewConfig
    view_name: str = ""
    heatmap_conf: HeatmapConfig = Field(default_factory=HeatmapConfig)
    dataframe_path: str | None = None
    df: pd.DataFrame | None = None
    loss_filter: str | None = "regression"
    metric: str = "grid_nrmse"
    data_mode: ClassDataMode = "abs"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self, ax: matplotlib.axes.Axes):
        df = self.df if self.df is not None else load_metrics_csv(self.dataframe_path or "")
        pivot, net_meta, _ = build_network_pivot(
            df, self.view, metric=self.metric, loss_filter=self.loss_filter,
        )
        class_pivot, class_order, cond_order = build_class_pivot(pivot, net_meta, self.view)
        data, cmap, norm, title, cb_label, fold = _resolve_class_summary(
            class_pivot.values, cond_order,
            view=self.view, view_name=self.view_name, data_mode=self.data_mode,
            loss_filter=self.loss_filter, cfg=self.heatmap_conf,
        )
        draw_class_summary_to_ax(
            ax, data, class_order, cond_order, cmap, norm, cb_label, title,
            self.view, self.heatmap_conf, fold_colorbar=fold,
        )


HorizontalHeatmapPanel.model_rebuild(force=True)
ClassSummaryHeatmapPanel.model_rebuild(force=True)
