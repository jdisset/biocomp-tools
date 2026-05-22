# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import matplotlib.axes
import pandas as pd
from pydantic import ConfigDict, Field

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.analysis.generalization.shapley_figure import (
    ShapleyDetailConfig, ShapleyDetailFigure,
)
from biocomptools.toollib.analysis.generalization.views import ViewConfig


class ShapleyDetailPanel(PlotPanel):
    plot_data: None = None
    view: ViewConfig
    view_name: str = ""
    shapley_conf: ShapleyDetailConfig = Field(default_factory=ShapleyDetailConfig)
    dataframe_path: str | None = None
    df: pd.DataFrame | None = None
    loss_filter: str | None = "regression"
    metric: str = "grid_nrmse"
    loss_label: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self, ax: matplotlib.axes.Axes):
        fig = ShapleyDetailFigure(
            view=self.view, view_name=self.view_name,
            shapley_conf=self.shapley_conf,
            dataframe_path=self.dataframe_path, df=self.df,
            loss_filter=self.loss_filter, metric=self.metric,
            loss_label=self.loss_label,
        )
        shapley_mat, players, detailed = fig.compute()
        fig._draw(shapley_mat, players, detailed, parent_ax=ax)


ShapleyDetailPanel.model_rebuild(force=True)
