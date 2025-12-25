"""Figuremaker for side-by-side circuit + diagram rendering."""

from typing import Any
import matplotlib.axes

from biocomp.recipe import Recipe
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, PartsLibrary

from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax
from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax


def _ensure_library():
    if LibraryContext.get_library() is None:
        from biocomptools.toollib.common import config as bconfig
        LibraryContext.set_library(PartsLibrary.from_file(bconfig.paths.parts_library))


def render_circuit_diagram(
    recipe: Recipe,
    ax_left: matplotlib.axes.Axes,
    ax_right: matplotlib.axes.Axes,
    hide_marker_tus: bool = True,
    simplified: bool = True,
    **_kwargs,
):
    """Render network diagram (left) and genetic circuit (right) for a recipe."""
    _ensure_library()

    networks = recipe_to_networks(recipe, invert=True)
    if not networks:
        ax_left.text(0.5, 0.5, "No networks", ha="center", va="center")
        ax_right.text(0.5, 0.5, "No networks", ha="center", va="center")
        return

    network = networks[0]
    render_diagram_to_ax(network, ax_left, simplified=simplified, title="Network Diagram")
    render_circuit_to_ax(network, ax_right, hide_marker_tus=hide_marker_tus, title="Genetic Circuit")
