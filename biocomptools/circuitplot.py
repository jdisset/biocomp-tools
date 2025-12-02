"""biocomp-circuitplot: Generate circuit diagrams and network schematics from recipes"""

from typing import Optional, Literal, Annotated, Any
from pathlib import Path
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

from dracon import make_program, Arg
from biocomp.recipe import Recipe
from biocomp.network import recipe_to_networks
from biocomp.library import LibraryContext, PartsLibrary


class CircuitPlotConfig(BaseModel):
    """Configuration for circuit plot generation"""

    recipe: Optional[Recipe] = Field(
        default=None,
        description="Recipe object to plot (loaded from dracon yaml)",
    )
    recipe_file: Annotated[Optional[str], Arg(short="r", help="Path to recipe file (json5)")] = None
    output: Annotated[str, Arg(short="o", help="Output file path")] = "circuit.pdf"
    plot_type: Annotated[
        Literal["schematic", "diagram", "card", "all"],
        Arg(short="t", help="Type of plot to generate"),
    ] = "schematic"
    simplified: Annotated[bool, Arg(help="Hide inverse chains in diagram")] = True
    show_all_tus: Annotated[bool, Arg(help="Show all TUs including markers")] = False
    show_recipe: Annotated[bool, Arg(help="Show recipe JSON in card view")] = False
    figsize: Annotated[tuple[float, float], Arg(help="Figure size (width, height)")] = (10, 8)
    dpi: Annotated[int, Arg(help="Output DPI")] = 150
    style: Annotated[Optional[dict], Arg(help="Custom style overrides")] = None
    invert: Annotated[bool, Arg(help="Apply network inversion")] = True


def load_recipe(config: CircuitPlotConfig) -> Optional[Recipe]:
    """Load recipe from config (either directly or from file)

    Supports:
    - Direct Recipe object (config.recipe)
    - New dracon YAML files (.yaml, .yml)
    - Old JSON5 recipe files (.json5, .json) - auto-converted via dict_to_recipe
    """
    if config.recipe is not None:
        return config.recipe

    if config.recipe_file:
        from biocomp.library import j5loads
        from biocomp.recipe import dict_to_recipe
        path = Path(config.recipe_file)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {path}")
        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            # new dracon YAML format
            import dracon
            return dracon.load(str(path))
        else:
            # old JSON5 format - use dict_to_recipe for conversion
            data = j5loads(content)
            # check if it's already a new-style recipe (has Recipe-like structure)
            if "content" in data and isinstance(data.get("content"), list):
                first_content = data["content"][0] if data["content"] else {}
                if "sources" in first_content:
                    # old format with "sources" key - convert
                    return dict_to_recipe(data)
            # try direct validation (might be new format saved as json)
            try:
                return Recipe.model_validate(data)
            except Exception:
                # fallback to old format conversion
                return dict_to_recipe(data)

    raise ValueError("Either recipe or recipe_file must be provided")


def render_schematic(network: Any, ax: plt.Axes, style: Optional[dict] = None, **kwargs):
    """Render circuit schematic to axes"""
    from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
    from jeanplot.container import Container
    from jeanplot.models import LayoutConstraints
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if style:
        jstyle.update(style)

    schematic = NetworkGeneticSchematicV2(network=network, **kwargs)
    root = Container(
        children=[schematic],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)


def render_diagram(network: Any, ax: plt.Axes, style: Optional[dict] = None, **kwargs):
    """Render network compute diagram to axes"""
    from jeanplot.network_diagram_v2 import NetworkDiagramV2
    from jeanplot.container import Container
    from jeanplot.models import LayoutConstraints
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if style:
        jstyle.update(style)

    diagram = NetworkDiagramV2(network=network, **kwargs)
    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)
    ax.set_aspect("equal")
    ax.axis("off")
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)


def render_card(
    network: Any,
    recipe: Recipe,
    ax: plt.Axes,
    show_recipe: bool = False,
    style: Optional[dict] = None,
    **kwargs
):
    """Render complete network card with diagram, schematic, and info"""
    from jeanplot.network_diagram_v2 import NetworkDiagramV2
    from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
    from jeanplot.container import Container
    from jeanplot.text import Text
    from jeanplot.models import LayoutConstraints, BoxStyle, Shadow
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if style:
        jstyle.update(style)

    diagram = NetworkDiagramV2(network=network, **kwargs)
    schematic = NetworkGeneticSchematicV2(network=network, **kwargs)

    info_title = Container(
        children=[
            Text(
                text=network.name or "Network",
                font_size=8,
                color="#333",
                style_class=["info_title"],
                vertical_align="middle",
            ),
        ],
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

    if show_recipe:
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
        body = Container(
            children=[recipebox, maincard],
            layout=LayoutConstraints(direction="row", gap=30, justify_content="start", align_items="stretch"),
            z_index=-100,
            style=BoxStyle(padding=(10, 10, 10, 10), corner_radius=3, background_color="#fff", border_color="#222", border_width=0.25, shadow=Shadow(color="#aaa4", blur_radius=25)),
        )
    else:
        body = Container(
            children=[maincard],
            layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
            z_index=-100,
            style=BoxStyle(padding=(10, 10, 10, 10), corner_radius=3, background_color="#fff", border_color="#222", border_width=0.25, shadow=Shadow(color="#aaa4", blur_radius=25)),
        )

    root = Container(
        children=[body, info_title],
        layout=LayoutConstraints(direction="column", gap=10, justify_content="center", align_items="stretch"),
    )

    jstyle.apply(root)
    ax.set_aspect("equal")
    ax.axis("off")
    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)


def run_circuitplot(config: CircuitPlotConfig):
    """Main entry point for circuit plot generation"""
    recipe = load_recipe(config)

    lib = LibraryContext.get_library()
    if lib is None:
        from biocomptools.toollib.common import config as bconfig
        lib = PartsLibrary.from_file(bconfig.paths.parts_library)
        LibraryContext.set_library(lib)

    networks = recipe_to_networks(recipe, invert=config.invert)
    if not networks:
        raise ValueError("No networks generated from recipe")

    network = networks[0]
    output_path = Path(config.output)

    if config.plot_type == "all":
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 2, config.figsize[1]))
        render_diagram(network, axes[0], style=config.style, simplified=config.simplified)
        render_schematic(network, axes[1], style=config.style, show_all_tus=config.show_all_tus)
        axes[0].set_title("Compute Diagram")
        axes[1].set_title("Circuit Schematic")
    elif config.plot_type == "card":
        fig, ax = plt.subplots(figsize=config.figsize)
        render_card(
            network, recipe, ax,
            show_recipe=config.show_recipe,
            style=config.style,
            simplified=config.simplified,
            show_all_tus=config.show_all_tus,
        )
    elif config.plot_type == "diagram":
        fig, ax = plt.subplots(figsize=config.figsize)
        render_diagram(network, ax, style=config.style, simplified=config.simplified)
    else:
        fig, ax = plt.subplots(figsize=config.figsize)
        render_schematic(network, ax, style=config.style, show_all_tus=config.show_all_tus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    import sys
    program = make_program(CircuitPlotConfig)
    config, _ = program.parse_args(sys.argv[1:])
    run_circuitplot(config)


if __name__ == "__main__":
    main()
