# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""BiocompFigureAdapter - render a legacy biocomp Figure through the jeanplot CLI.

The jeanplot CLI takes ``figure: Figure`` where ``Figure`` is the jeanplot
``Container``-based Figure. Some figures in paper-jobs are full biocomp
``Figure`` subclasses (``UORFMatrixFigure``, ``InnerNodesFigure``) or plain
biocomp ``Figure`` instances composed of ``plot_tasks``. They own their own
layout / output path / save flow and don't fit jeanplot's container model.

This adapter is a ``jeanplot.Figure`` subclass that holds one biocomp Figure
and delegates ``render()`` to it. ``output_dir`` / ``output_file`` on the
adapter are mirrored down to the biocomp Figure's ``figure_spec`` so the
jeanplot CLI's ``-o`` / ``--output-file`` flags still work.
"""

from typing import Any

from jeanplot.panels.figure import Figure as JpFigure


class BiocompFigureAdapter(JpFigure):
    """Wrap a biocomp ``Figure`` (or subclass) as a jeanplot ``Figure``."""

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
