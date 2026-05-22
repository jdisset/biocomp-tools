# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Bbox-bounded subfigure shim: mimics matplotlib SubFigure but carves a
fixed rectangle out of a parent figure via GridSpec(left/right/top/bottom).

This is the SSOT for "render a multi-axes block inside a panel's bbox" used
by UORFMatrixPanel, InnerNodesPanel, and the heatmap/shapley panels.
"""

from dataclasses import dataclass
from matplotlib import gridspec
from matplotlib.figure import Figure as MplFigure
from matplotlib.transforms import Bbox
import numpy as np


@dataclass
class BBoxSubFig:
    fig: MplFigure
    bbox: Bbox

    def gridspec(self, nrows: int, ncols: int, **kw) -> gridspec.GridSpec:
        return gridspec.GridSpec(
            nrows,
            ncols,
            figure=self.fig,
            left=self.bbox.x0,
            right=self.bbox.x1,
            bottom=self.bbox.y0,
            top=self.bbox.y1,
            **kw,
        )

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        width_ratios=None,
        height_ratios=None,
        gridspec_kw: dict | None = None,
    ):
        kw = dict(gridspec_kw or {})
        if width_ratios is not None:
            kw["width_ratios"] = width_ratios
        if height_ratios is not None:
            kw["height_ratios"] = height_ratios
        gs = self.gridspec(nrows, ncols, **kw)
        axes = np.array(
            [[self.fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
        )
        if nrows == 1 and ncols == 1:
            return axes[0, 0]
        if nrows == 1:
            return axes[0]
        if ncols == 1:
            return axes[:, 0]
        return axes

    def suptitle(self, s: str, *, y: float = 1.0, **kw) -> None:
        h = self.bbox.y1 - self.bbox.y0
        self.fig.text(
            (self.bbox.x0 + self.bbox.x1) / 2,
            self.bbox.y0 + y * h,
            s,
            ha="center",
            va="bottom",
            **kw,
        )

    def text(self, x: float, y: float, s: str, **kw) -> None:
        w = self.bbox.x1 - self.bbox.x0
        h = self.bbox.y1 - self.bbox.y0
        self.fig.text(self.bbox.x0 + x * w, self.bbox.y0 + y * h, s, **kw)

    def split_rows(
        self, n_rows: int, *, hspace: float = 0.25, height_ratios: list[float] | None = None
    ) -> list["BBoxSubFig"]:
        h = self.bbox.y1 - self.bbox.y0
        if height_ratios is None:
            height_ratios = [1.0] * n_rows
        tot = float(sum(height_ratios))
        row_h = [(r / tot) * h for r in height_ratios]
        usable = h - hspace * h * (n_rows - 1) / max(n_rows, 1)
        scale = usable / sum(row_h) if sum(row_h) > 0 else 1.0
        row_h = [r * scale for r in row_h]
        gap = hspace * h * (n_rows - 1) / max(n_rows, 1) / max(n_rows - 1, 1) if n_rows > 1 else 0.0
        result = []
        y_top = self.bbox.y1
        for r in row_h:
            y_bot = y_top - r
            result.append(
                BBoxSubFig(
                    fig=self.fig,
                    bbox=Bbox.from_extents(self.bbox.x0, y_bot, self.bbox.x1, y_top),
                )
            )
            y_top = y_bot - gap
        return result


def carve_bbox(ax) -> BBoxSubFig:
    """Hide ax and return a BBoxSubFig over its bbox."""
    bbox = ax.get_position()
    ax.set_axis_off()
    return BBoxSubFig(fig=ax.get_figure(), bbox=bbox)
