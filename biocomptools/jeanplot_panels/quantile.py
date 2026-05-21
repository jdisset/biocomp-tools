# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any

import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class QuantileCoveragePanel(PlotPanel):
    plot_data: None = None
    result: dict | None = None
    dataset_file: str | None = None
    model: Any = None
    model_path: str | None = None
    model_name: str | None = None
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    coverage_interval: tuple[float, float] = (0.1, 0.9)
    n_samples_per_point: int = 64
    seed: int = 11
    max_evals: int = 0
    device: str = "gpu"
    disable_variational: bool = True
    z_value: str | float = "uniform"
    z_normal_mean: float = 0.5
    z_normal_std: float = 0.2
    z_normal_clip: bool = True
    save_csv_to: str | None = None
    save_json_to: str | None = None

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.quantilecoverage import (
            render_quantile_coverage_summary,
        )

        render_quantile_coverage_summary(
            ax=ax,
            result=self.result,
            dataset_file=self.dataset_file,
            model=self.model,
            model_path=self.model_path,
            model_name=self.model_name,
            quantiles=self.quantiles,
            coverage_interval=self.coverage_interval,
            n_samples_per_point=self.n_samples_per_point,
            seed=self.seed,
            max_evals=self.max_evals,
            device=self.device,
            disable_variational=self.disable_variational,
            z_value=self.z_value,
            z_normal_mean=self.z_normal_mean,
            z_normal_std=self.z_normal_std,
            z_normal_clip=self.z_normal_clip,
            save_csv_to=self.save_csv_to,
            save_json_to=self.save_json_to,
        )
        if self.title:
            ax.set_title(self.title, **(self.title_kwargs or {}))


QuantileCoveragePanel.model_rebuild(force=True)
