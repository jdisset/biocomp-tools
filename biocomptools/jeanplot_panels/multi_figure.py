# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Render one figure per network in a dataset, loading the dataset once."""

from pydantic import BaseModel, ConfigDict

from jeanplot.cli import PlotJob
from biocomptools.jeanplot_panels.pipelines import load_paper_dataset


class PaperDatasetFigures(BaseModel):
    """Compose `figure_template` once per network in `dataset_file`, sharing one load.

    A broodmon `!DraconExec` program: the heavy `load_paper_dataset` runs once in
    the worker, then each network is rendered by injecting the loaded `D` into the
    template (which falls back to its own load when run standalone).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset_file: str
    figure_template: str
    output_dir: str
    swap_2d_axes: bool = False
    network_index: int | None = None  # render only this network; None = all

    def run(self) -> None:
        D = load_paper_dataset(self.dataset_file)
        base = {
            "D": D,
            "dataset_file": self.dataset_file,
            "output_dir": self.output_dir,
        }
        indices = range(len(D)) if self.network_index is None else [self.network_index]
        for i in indices:
            # Native orientation, then (for 2-input networks) the x<->y swap so
            # both orientations sit side by side. Swap is a display-only column
            # permutation handled inside the template.
            variants: list[dict] = [{}]
            if self.swap_2d_axes and int(D[i].x.shape[1]) == 2:
                variants.append({"swap_axes": True})
            for extra in variants:
                PlotJob.load(
                    self.figure_template,
                    context={**base, "network_index": i, **extra},
                ).run()


class PaperDatasetCount(BaseModel):
    """Print a dataset's network count so a broodmon edge `extract` can fan out."""

    dataset_file: str

    def run(self) -> None:
        print(f"NETWORKS={len(load_paper_dataset(self.dataset_file))}")
