"""biocomp-circuitplot: Generate circuit diagrams and network schematics from recipes"""

from typing import Optional, Literal, Annotated
from pathlib import Path
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

from dracon import dracon_program, Arg
from biocomp.recipe import Recipe
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, PartsLibrary
from biocomp.plotutils import FigureSpec

from biocomptools.toollib.figuremakers.geneticcircuit import (
    GeneticCircuitFigure,
    render_circuit_to_ax,
)
from biocomptools.toollib.figuremakers.networkdiagram import (
    NetworkDiagramFigure,
    render_diagram_to_ax,
)
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


@dracon_program(
    name='biocomp-circuitplot',
    description='Generate circuit diagrams and network schematics from recipes.',
    context_types=[Recipe, FigureSpec, GeneticCircuitFigure, NetworkDiagramFigure],
)
class CircuitPlotConfig(BaseModel):
    recipe: Optional[Recipe] = Field(default=None, description="Recipe object")
    recipe_file: Annotated[
        Optional[str], Arg(short="r", help="Path to recipe file (json5 or yaml)")
    ] = None
    output: Annotated[str, Arg(short="o", help="Output file path")] = "circuit.pdf"
    plot_type: Annotated[
        Literal["circuit", "diagram", "card", "all"],
        Arg(short="t", help="Type of plot: circuit (genetic), diagram (network), card, or all"),
    ] = "circuit"
    simplified: Annotated[bool, Arg(help="Hide inverse chains in diagram")] = True
    hide_marker_tus: Annotated[bool, Arg(help="Hide marker TUs in circuit")] = True
    show_recipe: Annotated[bool, Arg(help="Show recipe JSON in card view")] = False
    figsize: Annotated[tuple[float, float], Arg(help="Figure size (width, height)")] = (10, 8)
    dpi: Annotated[int, Arg(help="Output DPI")] = 150
    style: Annotated[Optional[dict], Arg(help="Custom style overrides")] = None
    invert: Annotated[bool, Arg(help="Apply network inversion")] = True

    def run(self):
        """Generate the circuit plot. Returns the output file path."""
        return run_circuitplot(self)


def load_recipe(config: CircuitPlotConfig) -> Recipe:
    if config.recipe is not None:
        return config.recipe

    if config.recipe_file:
        from biocomp.library import j5loads
        from biocomp.recipe import dict_to_recipe
        import dracon

        path = Path(config.recipe_file)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {path}")

        if path.suffix in (".yaml", ".yml"):
            return dracon.load(str(path))

        data = j5loads(path.read_text())
        if "content" in data and isinstance(data.get("content"), list):
            first = data["content"][0] if data["content"] else {}
            if "sources" in first:
                return dict_to_recipe(data)
        try:
            return Recipe.model_validate(data)
        except Exception:
            return dict_to_recipe(data)

    raise ValueError("Either recipe or recipe_file must be provided")


def ensure_library():
    lib = LibraryContext.get_library()
    if lib is None:
        from biocomptools.toollib.common import config as bconfig

        lib = PartsLibrary.from_file(bconfig.paths.parts_library)
        LibraryContext.set_library(lib)
    return lib


def run_circuitplot(config: CircuitPlotConfig):
    ensure_library()

    if hasattr(config, "_network") and config._network is not None:
        network = config._network
        recipe = None
    else:
        recipe = load_recipe(config)
        networks = recipe_to_networks(recipe, invert=config.invert)
        if not networks:
            raise ValueError("No networks generated from recipe")
        network = networks[0]
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_spec = FigureSpec(
        output_dir=str(output.parent),
        output_file=output.name,
        extra_args={"figsize": config.figsize, "dpi": config.dpi},
    )

    if config.plot_type == "circuit":
        figure = GeneticCircuitFigure(
            figure_spec=fig_spec,
            network=network,
            hide_marker_tus=config.hide_marker_tus,
            style_overrides=config.style,
        )
        figure.run()

    elif config.plot_type == "diagram":
        figure = NetworkDiagramFigure(
            figure_spec=fig_spec,
            network=network,
            simplified=config.simplified,
            style_overrides=config.style,
        )
        figure.run()

    elif config.plot_type == "all":
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
        render_diagram_to_ax(
            network, axes[0], simplified=config.simplified, style_overrides=config.style
        )
        render_circuit_to_ax(
            network, axes[1], hide_marker_tus=config.hide_marker_tus, style_overrides=config.style
        )
        axes[0].set_title("Network Diagram")
        axes[1].set_title("Genetic Circuit")
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved plot to {output}")

    elif config.plot_type == "card":
        if recipe is None:
            raise ValueError("Card plot requires a recipe - pass recipe or recipe_file")
        _render_card(network, recipe, output, config)

    else:
        raise ValueError(f"Unknown plot type: {config.plot_type}")

    return str(output)


def _render_card(network, recipe, output: Path, config: CircuitPlotConfig):
    from jeanplot import (
        Container,
        Text,
        Size,
        LayoutConstraints,
        BoxStyle,
        Shadow,
        Offset,
        SimpleBezierCurve,
        StraightCurve,
        OrthogonalCurve,
        LineEndFlat,
        LineEndCircle,
        LineEndArrow,
        MatplotlibRenderer,
        jstyle,
        load_default_theme,
    )
    from jeanplot.gene import GeneticSchematic
    from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram
    from dracon import load, resolve_all_lazy
    import importlib.resources

    load_default_theme()

    # Load the same themes used by standalone circuit/diagram renderers.
    network_theme_types = [
        Size,
        BoxStyle,
        LayoutConstraints,
        Offset,
        Shadow,
        SimpleBezierCurve,
        StraightCurve,
        OrthogonalCurve,
        LineEndFlat,
        LineEndCircle,
        LineEndArrow,
    ]
    network_theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
        "network_diagram.yaml"
    )
    network_theme = load(
        str(network_theme_file), context={t.__name__: t for t in network_theme_types}, raw_dict=True
    )
    resolve_all_lazy(network_theme)
    jstyle.update(network_theme)

    genetic_theme_types = [Size, BoxStyle, LayoutConstraints, Offset, Shadow, LineEndFlat]
    genetic_theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
        "genetic_schematic.yaml"
    )
    genetic_theme = load(
        str(genetic_theme_file), context={t.__name__: t for t in genetic_theme_types}, raw_dict=True
    )
    resolve_all_lazy(genetic_theme)
    jstyle.update(genetic_theme)

    if config.style:
        jstyle.update(config.style)

    diagram = NetworkDiagram(network=network, simplified=config.simplified)
    circuit_data = network.to_circuit_data(hide_markers=config.hide_marker_tus)
    schematic = GeneticSchematic.from_circuit(circuit_data)

    info_title = Container(
        children=[
            Text(
                text=network.name or "Network",
                font_size=8,
                color="#333",
                style_class=["info_title"],
                vertical_align="middle",
            )
        ],
        style_class=["info_title_box"],
        layout=LayoutConstraints(
            direction="column", gap=5, justify_content="center", align_items="center"
        ),
        style=BoxStyle(padding=(5, 5, 5, 5), corner_radius=3),
    )

    maincard = Container(
        children=[diagram, schematic],
        layout=LayoutConstraints(
            direction="column", gap=40, justify_content="space-around", align_items="center"
        ),
        style_class=["maincard"],
        style=BoxStyle(padding=(20, 20, 20, 20)),
    )

    body_children = [maincard]
    if config.show_recipe:
        recipe_text = recipe.model_dump_json(indent=2)
        recipebox = Container(
            children=[
                Text(
                    text=recipe_text,
                    font_size=5,
                    color="#787471",
                    style_class=["info_box"],
                    vertical_align="middle",
                ),
                Text(
                    text="Recipe",
                    font_size=8,
                    color="#787471",
                    style_class=["info_title"],
                    vertical_align="middle",
                ),
            ],
            layout=LayoutConstraints(
                direction="column", gap=10, justify_content="center", align_items="center"
            ),
            style_class=["recipebox"],
            style=BoxStyle(
                padding=(30, 50, 30, 30),
                background_color="#FEF9F2",
                corner_radius=3,
                border_color="#555",
                border_width=0.25,
            ),
        )
        body_children = [recipebox] + body_children

    body = Container(
        children=body_children,
        layout=LayoutConstraints(
            direction="row",
            gap=30 if config.show_recipe else 0,
            justify_content="start" if config.show_recipe else "center",
            align_items="stretch",
        ),
        z_index=-100,
        style=BoxStyle(
            padding=(10, 10, 10, 10),
            corner_radius=3,
            background_color="#fff",
            border_color="#222",
            border_width=0.25,
            shadow=Shadow(color="#aaa4", blur_radius=25),
        ),
    )

    root = Container(
        children=[body, info_title],
        layout=LayoutConstraints(
            direction="column", gap=10, justify_content="center", align_items="stretch"
        ),
    )

    jstyle.apply(root)
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    MatplotlibRenderer().render_component(ax, root, adjust_lims=True)
    fig.savefig(output, dpi=config.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved plot to {output}")


def main():
    CircuitPlotConfig.cli()


if __name__ == "__main__":
    main()
