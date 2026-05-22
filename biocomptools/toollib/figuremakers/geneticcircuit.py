# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Genetic circuit rendering for biocomp networks using jeanplot."""

from typing import Optional, Any, Literal
import matplotlib.axes


_GENETIC_SCHEMATIC_THEME_CACHE: dict | None = None


def _load_genetic_schematic_theme(types):
    global _GENETIC_SCHEMATIC_THEME_CACHE
    if _GENETIC_SCHEMATIC_THEME_CACHE is None:
        from dracon import load, resolve_all_lazy
        import importlib.resources
        theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
            "genetic_schematic.yaml"
        )
        theme = load(str(theme_file), context={t.__name__: t for t in types}, raw_dict=True)
        resolve_all_lazy(theme)
        _GENETIC_SCHEMATIC_THEME_CACHE = theme
    return _GENETIC_SCHEMATIC_THEME_CACHE


def render_circuit_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    hide_marker_tus: bool = True,
    hide_disabled_tus: bool = False,
    disabled_tu_ids: Optional[set[str]] = None,
    grid_gap: tuple[float, float] = (40.0, 20.0),
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal",
    style_overrides: Optional[dict] = None,
    title: Optional[str] = None,
    canvas_xlim: Optional[tuple[float, float]] = None,
    canvas_ylim: Optional[tuple[float, float]] = None,
    aspect: str = "equal",
    **_kwargs,
):
    """Render a genetic circuit schematic to an existing matplotlib axes.

    ``aspect="equal"`` (default) preserves circuit aspect inside the cell;
    ``aspect="auto"`` stretches to fill (no margins).
    """
    from jeanplot.gene import GeneticSchematic
    from jeanplot import MatplotlibRenderer, jstyle, load_default_theme
    from jeanplot.core import Size, BoxStyle, LayoutConstraints, Offset, Shadow
    from jeanplot.core.svg import LineEndFlat

    load_default_theme()

    jeanplot_types = [Size, BoxStyle, LayoutConstraints, Offset, Shadow, LineEndFlat]
    theme = _load_genetic_schematic_theme(jeanplot_types)
    jstyle.update(theme)

    if style_overrides:
        jstyle.update(style_overrides)

    circuit_data = network.to_circuit_data(
        hide_markers=hide_marker_tus,
        disabled_tu_ids=disabled_tu_ids,
        hide_disabled=hide_disabled_tus,
    )
    schematic = GeneticSchematic.from_circuit(
        circuit_data,
        grid_gap=grid_gap,
        connection_style=connection_style,
    )
    jstyle.apply(schematic)

    ax.set_aspect(aspect)
    ax.axis("off")
    ax.set_facecolor("none")

    renderer = MatplotlibRenderer()
    renderer.render_component(ax, schematic, adjust_lims=True)

    # Optional fixed-canvas widening - schematic content is left at its
    # rendered data coords, lims are expanded around it so smaller recipes
    # appear small inside a uniformly-sized canvas. Text stays in data
    # space (jeanplot's data-unit auto-refresh handles re-sizing on draw).
    from biocomptools.toollib.figuremakers._jeanplot_canvas import apply_canvas

    apply_canvas(ax, canvas_xlim, canvas_ylim)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
