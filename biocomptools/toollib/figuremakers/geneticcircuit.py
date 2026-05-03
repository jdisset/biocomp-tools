"""Genetic circuit figure for biocomp networks using jeanplot."""

from typing import Optional, Any, Literal
from pydantic import Field
import matplotlib.pyplot as plt
import matplotlib.axes

from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


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

    # Optional fixed-canvas widening — schematic content is left at its
    # rendered data coords, lims are expanded around it so smaller recipes
    # appear small inside a uniformly-sized canvas. Text stays in data
    # space (jeanplot's data-unit auto-refresh handles re-sizing on draw).
    from biocomptools.toollib.figuremakers._jeanplot_canvas import apply_canvas

    apply_canvas(ax, canvas_xlim, canvas_ylim)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


class GeneticCircuitFigure(Figure):
    """Figure that renders a genetic circuit schematic using jeanplot."""

    network: Any = Field(description="biocomp Network object")
    hide_marker_tus: bool = True
    hide_disabled_tus: bool = False
    disabled_tu_ids: Optional[set[str]] = None
    grid_gap: tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    style_overrides: Optional[dict] = None

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        figsize = self.figure_spec.extra_args.get("figsize", (10, 8))
        dpi = self.figure_spec.extra_args.get("dpi", 150)

        fig, ax = plt.subplots(figsize=figsize)
        render_circuit_to_ax(
            network=self.network,
            ax=ax,
            hide_marker_tus=self.hide_marker_tus,
            hide_disabled_tus=self.hide_disabled_tus,
            disabled_tu_ids=self.disabled_tu_ids,
            grid_gap=self.grid_gap,
            connection_style=self.connection_style,
            style_overrides=self.style_overrides,
        )

        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved genetic circuit to {self.figure_spec.output_path}")
