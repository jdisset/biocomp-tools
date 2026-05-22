# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from pathlib import Path
import matplotlib.axes
from pydantic import ConfigDict

from jeanplot.panels.base import PlotPanel
from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure
from biocomptools.modelmodel import BiocompModel


class InnerNodesPanel(PlotPanel):
    plot_data: None = None
    model: BiocompModel
    n_samples: int = 100_000
    print_summary: bool = True
    show_distribution: bool = False
    n_trendline_points: int = 200
    max_trajectory_points: int = 1000
    history_dir: Path | None = None
    embedding_trajectories: dict[str, list[tuple[float, ...]]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def draw(self, ax: matplotlib.axes.Axes) -> None:
        InnerNodesFigure(
            model=self.model,
            n_samples=self.n_samples,
            print_summary=self.print_summary,
            show_distribution=self.show_distribution,
            n_trendline_points=self.n_trendline_points,
            max_trajectory_points=self.max_trajectory_points,
            history_dir=self.history_dir,
            embedding_trajectories=self.embedding_trajectories,
        ).draw_into(ax)


InnerNodesPanel.model_rebuild(force=True)
