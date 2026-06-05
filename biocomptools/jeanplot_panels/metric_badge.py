# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Metric badge + rule panels for per-topology row/table figures.

`MetricBadgePanel` draws one pill-shaped two-cell badge (a colored ``label`` cell +
a lighter ``value`` cell, straight divider, single rounded outline) on its own axes.
Colors are jstyle-fillable fields keyed by ``kind`` (so the palette lives in the
theme, not here). `RulePanel` draws a horizontal booktabs rule across its axes -
table chrome must be a panel because the Figure renderer only draws PlotPanel leaves.
"""

from typing import Literal

import matplotlib.axes
import matplotlib.patches as mpatches

from jeanplot.panels.base import PlotPanel


def fmt_value(v: float) -> str:
    """0.054 -> '.054', 1.23 -> '1.23' (drop the leading 0 for sub-1 magnitudes)."""
    s = f"{v:.3f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _draw_pill(
    ax,
    x_left,
    y_center,
    height,
    label,
    value,
    label_bg,
    value_bg,
    *,
    font_size,
    dpi,
    outline,
    text_color,
    outline_w,
    divider_w,
    pad_in=0.07,
    dry=False,
):
    """Draw (or, if ``dry``, only measure) one pill anchored at its left edge.
    Coords are inches (axes data units with aspect='equal'). Returns total width."""
    r = height / 2.0
    rend = ax.figure.canvas.get_renderer()

    def text_w_in(s, weight):
        t = ax.text(
            0,
            -999,
            s,
            fontsize=font_size,
            fontweight=weight,
            ha="left",
            va="center",
            color=text_color,
        )
        w = t.get_window_extent(rend).width / dpi
        t.remove()
        return w

    lw_in = text_w_in(label, "normal")
    vw_in = text_w_in(value, "bold")
    label_cell_w = r * 0.8 + lw_in + pad_in
    value_cell_w = pad_in + vw_in + r * 0.8
    total_w = label_cell_w + value_cell_w
    if dry:
        return total_w

    dx = x_left + label_cell_w
    y0 = y_center - r
    pill = mpatches.FancyBboxPatch(
        (x_left, y0),
        total_w,
        height,
        boxstyle=mpatches.BoxStyle("round", pad=0.0, rounding_size=r),
        linewidth=0,
        facecolor=value_bg,
        edgecolor="none",
        zorder=1,
        mutation_aspect=1.0,
    )
    ax.add_patch(pill)
    rect = mpatches.Rectangle(
        (x_left, y0), label_cell_w, height, facecolor=label_bg, edgecolor="none", zorder=2
    )
    ax.add_patch(rect)
    rect.set_clip_path(pill)
    ax.plot(
        [dx, dx], [y0, y0 + height], color=outline, lw=divider_w, zorder=3, solid_capstyle="butt"
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x_left, y0),
            total_w,
            height,
            boxstyle=mpatches.BoxStyle("round", pad=0.0, rounding_size=r),
            linewidth=outline_w,
            facecolor="none",
            edgecolor=outline,
            zorder=4,
            mutation_aspect=1.0,
        )
    )
    ax.text(
        x_left + r * 0.8 + lw_in / 2.0,
        y_center,
        label,
        fontsize=font_size,
        fontweight="normal",
        ha="center",
        va="center",
        color=text_color,
        zorder=5,
    )
    ax.text(
        dx + pad_in + vw_in / 2.0,
        y_center,
        value,
        fontsize=font_size,
        fontweight="bold",
        ha="center",
        va="center",
        color=text_color,
        zorder=5,
    )
    return total_w


class MetricBadgePanel(PlotPanel):
    plot_data: None = None
    kind: str = ""  # theme key (ermse / noise_floor / rep_noise)
    label: str = ""
    value: float | None = None
    value_text: str | None = None  # explicit override; else fmt_value(value)
    label_color: str = "#E2E2E2"  # theme fills per-kind (jstyle cascade)
    value_color: str = "#F4F4F4"
    outline: str = "#0C0C0C"
    text_color: str = "#101010"
    badge_height: float = 0.22  # inches
    font_size: float = 7.5  # points
    outline_w: float = 1.0
    divider_w: float = 0.8
    align: Literal["left", "center", "right"] = "center"

    def draw(self, ax: matplotlib.axes.Axes):
        ax.axis("off")
        if self.value_text is not None:
            val = self.value_text
        elif self.value is not None:
            val = fmt_value(float(self.value))
        else:
            return
        fig = ax.figure
        fig.canvas.draw()
        dpi = fig.dpi
        pos = ax.get_position()
        w_in = pos.width * fig.get_figwidth()
        h_in = pos.height * fig.get_figheight()
        ax.set_xlim(0, w_in)
        ax.set_ylim(0, h_in)
        ax.set_aspect("equal")

        def pill(x_left, y_center, dry):
            return _draw_pill(
                ax,
                x_left,
                y_center,
                self.badge_height,
                self.label,
                val,
                self.label_color,
                self.value_color,
                font_size=self.font_size,
                dpi=dpi,
                outline=self.outline,
                text_color=self.text_color,
                outline_w=self.outline_w,
                divider_w=self.divider_w,
                dry=dry,
            )

        wpill = pill(0.0, 0.0, True)
        if self.align == "center":
            x = max(0.0, (w_in - wpill) / 2.0)
        elif self.align == "right":
            x = max(0.0, w_in - wpill)
        else:
            x = 0.0
        pill(x, h_in / 2.0, False)


class RulePanel(PlotPanel):
    """A horizontal booktabs-style rule spanning the panel width."""

    plot_data: None = None
    color: str = "#222222"
    lw: float = 1.0
    inset: float = 0.0  # axes-fraction inset at each end
    y: float = 0.5  # axes-fraction vertical position

    def draw(self, ax: matplotlib.axes.Axes):
        ax.axis("off")
        ax.plot(
            [self.inset, 1.0 - self.inset],
            [self.y, self.y],
            transform=ax.transAxes,
            color=self.color,
            lw=self.lw,
            solid_capstyle="butt",
        )


MetricBadgePanel.model_rebuild(force=True)
RulePanel.model_rebuild(force=True)
