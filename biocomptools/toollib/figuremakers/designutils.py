"""Design result visualization utilities.

DesignResult is a pure data holder - all values must be precomputed via DesignEvaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import matplotlib.axes
from matplotlib.patches import FancyBboxPatch

from biocomp.plotutils import PlotData
from biocomp.design import DataTarget

if TYPE_CHECKING:
    from biocomptools.toollib.design_eval import EvaluatedDesign

GOOD_COLOR = "#28a745"
BAD_COLOR = "#dc3545"
NEUTRAL_COLOR = "#333"
BASELINE_COLOR = "#6c757d"


@dataclass
class DesignResult:
    """Pure data holder for design result visualization.

    All data must be precomputed via DesignEvaluator.evaluate_designs().
    No lazy computation - all fields required at construction.
    """

    network: Any
    target: Any
    target_name: str
    rank: int
    replicate: int
    scaffold_network_name: str
    loss: float
    recipe_hash: str
    run_name: str
    model: Any

    gt_data: PlotData
    pred_data: PlotData
    lattice_data: PlotData | None
    lattice_grid: Any | None  # np.ndarray (yres, xres) for pixel-perfect rendering
    lattice_extent: tuple[float, float, float, float] | None  # (xmin, xmax, ymin, ymax)
    lattice_resolution: tuple[int, int] | None  # (xres, yres)
    design_nre: float | None
    baseline_nre: float | None
    exp_x_data: PlotData | None = None
    fingerprint: str | None = None

    @property
    def has_original_network(self) -> bool:
        return isinstance(self.target, DataTarget) and self.target.original_network is not None

    @classmethod
    def from_evaluated(cls, ev: EvaluatedDesign) -> DesignResult:
        """Create DesignResult from EvaluatedDesign."""
        return cls(
            network=ev.input.network,
            target=ev.input.target,
            target_name=ev.input.target_name,
            rank=ev.input.rank,
            replicate=ev.input.replicate,
            scaffold_network_name=ev.input.scaffold_network_name,
            loss=ev.input.loss,
            recipe_hash=ev.input.recipe_hash,
            run_name=ev.input.run_name,
            model=None,  # not needed after evaluation
            gt_data=ev.gt_data,
            pred_data=ev.pred_data,
            lattice_data=ev.lattice_data,
            lattice_grid=ev.lattice_grid,
            lattice_extent=ev.lattice_extent,
            lattice_resolution=ev.lattice_resolution,
            design_nre=ev.design_nre,
            baseline_nre=ev.baseline_nre,
            exp_x_data=ev.exp_x_data,
        )


def nre_color(design_nre: float | None, baseline_nre: float | None) -> str:
    """Determine color for NRE display based on quality."""
    if design_nre is None:
        return NEUTRAL_COLOR
    if baseline_nre is not None and design_nre <= baseline_nre * 1.5:
        return GOOD_COLOR
    if design_nre < 5.0:
        return NEUTRAL_COLOR
    return BAD_COLOR


def render_design_metrics(ax: matplotlib.axes.Axes, result: DesignResult, **_kwargs):
    """Render design metrics panel."""
    ax.axis('off')
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor='#EEEEEE',
            edgecolor='#ccc',
            linewidth=1,
            clip_on=False,
        )
    )

    loss_color = (
        GOOD_COLOR if result.loss < 0.5 else (BAD_COLOR if result.loss > 1.5 else NEUTRAL_COLOR)
    )
    ax.text(
        0.5,
        0.88,
        f"{result.loss:.4f}",
        transform=ax.transAxes,
        fontsize=22,
        va='center',
        ha='center',
        fontweight='bold',
        color=loss_color,
    )
    ax.text(
        0.5,
        0.76,
        "Design Loss",
        transform=ax.transAxes,
        fontsize=9,
        va='center',
        ha='center',
        color='gray',
    )

    if result.has_original_network:
        nre_y = 0.60
        color = nre_color(result.design_nre, result.baseline_nre)
        if result.design_nre is not None:
            ax.text(
                0.5,
                nre_y,
                f"NRE: {result.design_nre:.2f}",
                transform=ax.transAxes,
                fontsize=14,
                va='center',
                ha='center',
                fontweight='bold',
                color=color,
            )
        else:
            ax.text(
                0.5,
                nre_y,
                "NRE: N/A",
                transform=ax.transAxes,
                fontsize=14,
                va='center',
                ha='center',
                color='#aaa',
            )
        if result.baseline_nre is not None:
            ax.text(
                0.5,
                nre_y - 0.10,
                f"(baseline: {result.baseline_nre:.2f})",
                transform=ax.transAxes,
                fontsize=9,
                va='center',
                ha='center',
                color=BASELINE_COLOR,
            )
        info_y = 0.38
    else:
        info_y = 0.55

    ax.text(
        0.5,
        info_y,
        f"Rank: {result.rank}  |  Replicate: {result.replicate}",
        transform=ax.transAxes,
        fontsize=10,
        va='center',
        ha='center',
        family='monospace',
    )

    scaffold = (
        result.scaffold_network_name[:25] + '...'
        if len(result.scaffold_network_name) > 28
        else result.scaffold_network_name
    )
    ax.text(
        0.5,
        info_y - 0.12,
        f"Scaffold: {scaffold}",
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        family='monospace',
        color='#666',
    )
    ax.text(
        0.5,
        info_y - 0.22,
        f"Hash: {result.recipe_hash}",
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        family='monospace',
        color='#888',
    )
    if result.fingerprint:
        ax.text(
            0.5,
            info_y - 0.32,
            f"FP: {result.fingerprint}",
            transform=ax.transAxes,
            fontsize=8,
            va='center',
            ha='center',
            family='monospace',
            color='#888',
        )


def render_empty_panel(ax: matplotlib.axes.Axes, text: str = "", **_kwargs):
    """Render empty panel with optional text."""
    ax.axis('off')
    if text:
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=10,
            va='center',
            ha='center',
            color='#aaa',
        )


def render_lattice_heatmap(
    ax: matplotlib.axes.Axes,
    result: DesignResult,
    title: str = "Design View (Lattice)",
    cmap: str = "viridis",
    draw_colorbar: bool = False,
    **_kwargs,
):
    """Render pixel-perfect lattice heatmap showing exactly what design loss sees.

    Uses imshow with the exact grid resolution so each pixel corresponds to one
    lattice sample point used during design optimization.
    """
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if result.lattice_grid is None or result.lattice_extent is None:
        ax.axis('off')
        ax.text(0.5, 0.5, "No lattice data", transform=ax.transAxes, ha='center', va='center')
        return

    grid = np.asarray(result.lattice_grid)
    xmin, xmax, ymin, ymax = result.lattice_extent

    im = ax.imshow(
        grid,
        origin='lower',
        aspect='equal',
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        interpolation='nearest',
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    if title:
        ax.set_title(title)

    if draw_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, cax=cax)


def render_smooth_with_extent(
    ax: matplotlib.axes.Axes,
    plot_data: PlotData,
    extent: tuple[float, float, float, float] | None,
    title: str = "",
    draw_colorbar: bool = True,
    **kwargs,
):
    """Render smooth KNN plot with explicit axis limits from target extent."""
    from biocomp.plotutils import smooth
    from biocomp.datautils import DataRescaler

    rescaler = kwargs.pop('rescaler', DataRescaler())

    xlims = (extent[0], extent[1]) if extent else (None, None)
    ylims = (extent[2], extent[3]) if extent else (None, None)

    smooth(
        plot_data=plot_data,
        ax=ax,
        rescaler=rescaler,
        xlims=xlims,
        ylims=ylims,
        title=title,
        draw_colorbar=draw_colorbar,
        smooth_2d_params={'vlims': (None, None)},
        vlims=(None, None),
        **kwargs,
    )


def render_network_diagram_full_width(
    fig,
    axes_to_remove: list,
    network,
    title: str = "Network Diagram",
    **_kwargs,
):
    """Render network diagram spanning full width by replacing two axes.

    Removes the given axes and creates a new subplot spanning their positions.
    """
    from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

    positions = [a.get_position() for a in axes_to_remove]

    for a in axes_to_remove:
        a.remove()

    x0 = min(p.x0 for p in positions)
    y0 = min(p.y0 for p in positions)
    x1 = max(p.x1 for p in positions)
    y1 = max(p.y1 for p in positions)

    spanning_ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0])

    render_diagram_to_ax(network, spanning_ax, title=title)
