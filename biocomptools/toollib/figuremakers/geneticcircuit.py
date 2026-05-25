# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Genetic circuit rendering for biocomp networks using jeanplot."""

from typing import Optional, Any, Literal
import matplotlib.axes


_GENETIC_SCHEMATIC_THEME_CACHE: dict | None = None


def _load_genetic_schematic_theme(types):
    global _GENETIC_SCHEMATIC_THEME_CACHE
    if _GENETIC_SCHEMATIC_THEME_CACHE is None:
        from dracon import DraconLoader, resolve_all_lazy
        from jeanplot import DEFAULT_TYPES, make_context_from_types

        ctx = make_context_from_types(DEFAULT_TYPES + list(types))
        loader = DraconLoader(enable_interpolation=True, context=ctx, base_dict_type=dict)
        cfg = loader.stack(
            "pkg:jeanplot:resources/themes/default.yaml",
            "pkg:biocomptools.configs.themes:genetic_schematic.yaml",
        ).construct()
        resolve_all_lazy(cfg, except_for={"component"})
        _GENETIC_SCHEMATIC_THEME_CACHE = cfg["rules"]
    return _GENETIC_SCHEMATIC_THEME_CACHE


def render_circuit_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    hide_marker_tus: bool = True,
    hide_disabled_tus: bool = False,
    disabled_tu_ids: Optional[set[str]] = None,
    show_tu_labels: bool = True,
    axis_tags: Optional[dict[str, str]] = None,
    bias_axis_tag: Optional[str] = None,
    orientation: Literal["column", "row"] = "column",
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
    from jeanplot import MatplotlibRenderer, jstyle
    from jeanplot.core import Size, BoxStyle, LayoutConstraints, Offset, Shadow
    from jeanplot.core.svg import LineEndFlat

    # theme stack already includes default.yaml, so no separate load_default_theme
    jeanplot_types = [Size, BoxStyle, LayoutConstraints, Offset, Shadow, LineEndFlat]
    from jeanplot.core.style_engine import merge_jstyle_rules

    theme = _load_genetic_schematic_theme(jeanplot_types)
    jstyle.clear()
    if style_overrides:
        theme = merge_jstyle_rules(theme, style_overrides)
    jstyle.update(theme)

    circuit_data = network.to_circuit_data(
        hide_markers=hide_marker_tus,
        disabled_tu_ids=disabled_tu_ids,
        hide_disabled=hide_disabled_tus,
        show_tu_labels=show_tu_labels,
        axis_tags=axis_tags,
        bias_axis_tag=bias_axis_tag,
    )
    schematic = GeneticSchematic.from_circuit(
        circuit_data,
        grid_gap=grid_gap,
        connection_style=connection_style,
        orientation=orientation,
    )
    jstyle.apply(schematic)

    ax.axis("off")
    ax.set_facecolor("none")

    # pre-set: set_aspect after the draw doesn't stick on positioned axes
    ax.set_aspect(aspect)

    renderer = MatplotlibRenderer()
    # padding=0 fills the panel; set_aspect=False keeps our `aspect` (not forced "equal")
    renderer.render_component(
        ax, schematic, adjust_lims=True,
        adjust_lims_padding=0.0, adjust_lims_set_aspect=False,
    )

    # Optional fixed-canvas widening - schematic content is left at its
    # rendered data coords, lims are expanded around it so smaller recipes
    # appear small inside a uniformly-sized canvas. Text stays in data
    # space (jeanplot's data-unit auto-refresh handles re-sizing on draw).
    from biocomptools.toollib.figuremakers._jeanplot_canvas import apply_canvas

    apply_canvas(ax, canvas_xlim, canvas_ylim)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
