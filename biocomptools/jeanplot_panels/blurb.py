# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import matplotlib.axes

from jeanplot.panels.base import PlotPanel


class BlurbPanel(PlotPanel):
    plot_data: None = None
    text: str = ""
    fontsize: float = 9
    line_sep: float = 3.0
    block_sep: float = 8.0
    h1_fontsize: float | None = None
    h2_fontsize: float | None = None
    h3_fontsize: float | None = None
    bullet: str = "•"
    indent: str = "  "
    wrap: bool = True
    wrap_chars: int | None = None
    bullet_indent: int = 2

    def draw(self, ax: matplotlib.axes.Axes):
        from biocomptools.toollib.figuremakers.blurbpanel import render_blurb_to_ax

        render_blurb_to_ax(
            ax=ax,
            text=self.text,
            fontsize=self.fontsize,
            title=self.title,
            title_kwargs=self.title_kwargs or None,
            line_sep=self.line_sep,
            block_sep=self.block_sep,
            h1_fontsize=self.h1_fontsize,
            h2_fontsize=self.h2_fontsize,
            h3_fontsize=self.h3_fontsize,
            bullet=self.bullet,
            indent=self.indent,
            wrap=self.wrap,
            wrap_chars=self.wrap_chars,
            bullet_indent=self.bullet_indent,
        )


BlurbPanel.model_rebuild(force=True)
