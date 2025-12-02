"""biocomp-circuitplot: Generate circuit diagrams and network schematics from recipes"""

from typing import Optional, Literal, Annotated
from pathlib import Path
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

from dracon import make_program, Arg
from biocomp.recipe import Recipe
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, PartsLibrary
from biocomp.plotutils import FigureSpec

from biocomptools.toollib.figuremakers.geneticcircuit import GeneticCircuitFigure
from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagramFigure


class CircuitPlotConfig(BaseModel):
    recipe: Optional[Recipe] = Field(default=None, description="Recipe object (from dracon yaml)")
    recipe_file: Annotated[Optional[str], Arg(short="r", help="Path to recipe file (json5)")] = None
    output: Annotated[str, Arg(short="o", help="Output file path")] = "circuit.pdf"
    plot_type: Annotated[Literal["circuit", "diagram", "card", "all"], Arg(short="t", help="Type of plot: circuit (genetic), diagram (network), card, or all")] = "circuit"
    simplified: Annotated[bool, Arg(help="Hide inverse chains in diagram")] = True
    hide_marker_tus: Annotated[bool, Arg(help="Hide marker TUs in circuit")] = True
    show_recipe: Annotated[bool, Arg(help="Show recipe JSON in card view")] = False
    figsize: Annotated[tuple[float, float], Arg(help="Figure size (width, height)")] = (10, 8)
    dpi: Annotated[int, Arg(help="Output DPI")] = 150
    style: Annotated[Optional[dict], Arg(help="Custom style overrides")] = None
    invert: Annotated[bool, Arg(help="Apply network inversion")] = True


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
    recipe = load_recipe(config)
    ensure_library()

    networks = recipe_to_networks(recipe, invert=config.invert)
    if not networks:
        raise ValueError("No networks generated from recipe")

    network = networks[0]
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_spec = FigureSpec(
        output_dir=str(output.parent), output_file=output.name,
        extra_args={"figsize": config.figsize, "dpi": config.dpi}
    )

    if config.plot_type == "circuit":
        figure = GeneticCircuitFigure(
            figure_spec=fig_spec, network=network,
            hide_marker_tus=config.hide_marker_tus, style_overrides=config.style,
        )
        figure.run()

    elif config.plot_type == "diagram":
        figure = NetworkDiagramFigure(
            figure_spec=fig_spec, network=network,
            simplified=config.simplified, style_overrides=config.style,
        )
        figure.run()

    elif config.plot_type == "all":
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
        _render_diagram(network, axes[0], config)
        _render_circuit(network, axes[1], config)
        axes[0].set_title("Network Diagram")
        axes[1].set_title("Genetic Circuit")
        fig.savefig(output, dpi=config.dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Saved plot to {output}")

    elif config.plot_type == "card":
        _render_card(network, recipe, output, config)

    else:
        raise ValueError(f"Unknown plot type: {config.plot_type}")


def _render_circuit(network, ax, config: CircuitPlotConfig):
    from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
    from jeanplot.container import Container
    from jeanplot.models import LayoutConstraints
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if config.style:
        jstyle.update(config.style)

    schematic = NetworkGeneticSchematicV2(network=network, hide_marker_tus=config.hide_marker_tus)
    root = Container(children=[schematic], layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"))
    jstyle.apply(root)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    MatplotlibRenderer().render_component(ax, root, adjust_lims=True)


def _render_diagram(network, ax, config: CircuitPlotConfig):
    from jeanplot.network_diagram_v2 import NetworkDiagramV2
    from jeanplot.container import Container
    from jeanplot.models import LayoutConstraints
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if config.style:
        jstyle.update(config.style)

    diagram = NetworkDiagramV2(network=network, simplified=config.simplified)
    root = Container(children=[diagram], layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"))
    jstyle.apply(root)
    ax.set_aspect("equal")
    ax.axis("off")
    MatplotlibRenderer().render_component(ax, root, adjust_lims=True)


def _render_card(network, recipe, output: Path, config: CircuitPlotConfig):
    from jeanplot.network_diagram_v2 import NetworkDiagramV2
    from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
    from jeanplot.container import Container
    from jeanplot.text import Text
    from jeanplot.models import LayoutConstraints, BoxStyle, Shadow
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if config.style:
        jstyle.update(config.style)

    diagram = NetworkDiagramV2(network=network, simplified=config.simplified)
    schematic = NetworkGeneticSchematicV2(network=network, hide_marker_tus=config.hide_marker_tus)

    info_title = Container(
        children=[Text(text=network.name or "Network", font_size=8, color="#333", style_class=["info_title"], vertical_align="middle")],
        style_class=["info_title_box"],
        layout=LayoutConstraints(direction="column", gap=5, justify_content="center", align_items="center"),
        style=BoxStyle(padding=(5, 5, 5, 5), corner_radius=3),
    )

    maincard = Container(
        children=[diagram, schematic],
        layout=LayoutConstraints(direction="column", gap=40, justify_content="space-around", align_items="center"),
        style_class=["maincard"],
        style=BoxStyle(padding=(20, 20, 20, 20)),
    )

    body_children = [maincard]
    if config.show_recipe:
        recipe_text = recipe.model_dump_json(indent=2)
        recipebox = Container(
            children=[
                Text(text=recipe_text, font_size=5, color="#787471", style_class=["info_box"], vertical_align="middle"),
                Text(text="Recipe", font_size=8, color="#787471", style_class=["info_title"], vertical_align="middle"),
            ],
            layout=LayoutConstraints(direction="column", gap=10, justify_content="center", align_items="center"),
            style_class=["recipebox"],
            style=BoxStyle(padding=(30, 50, 30, 30), background_color="#FEF9F2", corner_radius=3, border_color="#555", border_width=0.25),
        )
        body_children = [recipebox] + body_children

    body = Container(
        children=body_children,
        layout=LayoutConstraints(direction="row", gap=30 if config.show_recipe else 0, justify_content="start" if config.show_recipe else "center", align_items="stretch"),
        z_index=-100,
        style=BoxStyle(padding=(10, 10, 10, 10), corner_radius=3, background_color="#fff", border_color="#222", border_width=0.25, shadow=Shadow(color="#aaa4", blur_radius=25)),
    )

    root = Container(
        children=[body, info_title],
        layout=LayoutConstraints(direction="column", gap=10, justify_content="center", align_items="stretch"),
    )

    jstyle.apply(root)
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    MatplotlibRenderer().render_component(ax, root, adjust_lims=True)
    fig.savefig(output, dpi=config.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved plot to {output}")


def main():
    import sys
    program = make_program(CircuitPlotConfig)
    config, _ = program.parse_args(sys.argv[1:])
    run_circuitplot(config)


if __name__ == "__main__":
    main()
