"""Shared canvas helpers for jeanplot-rendered figures."""

from typing import Optional


def apply_canvas(
    ax,
    canvas_xlim: Optional[tuple[float, float]],
    canvas_ylim: Optional[tuple[float, float]],
):
    """Expand ax data lims to a fixed canvas, centred on the rendered content.

    The schematic is rendered first via jeanplot at its natural data extent
    (``adjust_lims=True``). When a fixed canvas is requested, we widen the
    ax data lims around that rendered content so small schematics appear
    small inside a uniformly-sized canvas while large ones fill it.

    Text sizing is left to jeanplot: components rendered with
    ``font_size_mode="data"`` (the default) re-evaluate their pixel size on
    every draw via the renderer's ``draw_event`` callback, so they stay
    proportional to the schematic's data-coord extent regardless of how the
    ax lims change after render.
    """
    if canvas_xlim is None and canvas_ylim is None:
        return

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cx = 0.5 * (cur_xlim[0] + cur_xlim[1])
    cy = 0.5 * (cur_ylim[0] + cur_ylim[1])
    if canvas_xlim is not None:
        half = 0.5 * (canvas_xlim[1] - canvas_xlim[0])
        ax.set_xlim(cx - half, cx + half)
    if canvas_ylim is not None:
        half = 0.5 * (canvas_ylim[1] - canvas_ylim[0])
        ax.set_ylim(cy - half, cy + half)
