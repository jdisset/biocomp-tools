# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Adapter bridging legacy biocomp `Figure` into jeanplot."""

from typing import Any

from jeanplot.panels.figure import Figure as JpFigure


class BiocompFigureAdapter(JpFigure):
    biocomp_figure: Any

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        bf = self.biocomp_figure
        spec = getattr(bf, "figure_spec", None)
        if spec is None:
            return
        if self.output_dir and self.output_dir != "./":
            spec.output_dir = self.output_dir
        else:
            self.output_dir = spec.output_dir
        if self.output_file and self.output_file != "unnamed.png":
            spec.output_file = self.output_file
        else:
            self.output_file = spec.output_file

    def render(self, **kwargs):
        bf = self.biocomp_figure
        spec = getattr(bf, "figure_spec", None)
        if spec is not None:
            if self.output_dir:
                spec.output_dir = self.output_dir
            if self.output_file:
                spec.output_file = self.output_file
        overwrite = kwargs.get("overwrite", True)
        bf.run(overwrite=overwrite)
        return bf


BiocompFigureAdapter.model_rebuild(force=True)
