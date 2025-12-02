"""Genetic circuit figure for biocomp networks"""

from typing import Optional, Any, Tuple, Literal
from pydantic import Field
import matplotlib.pyplot as plt

from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class GeneticCircuitFigure(Figure):
    """Figure that renders a genetic circuit schematic using jeanplot"""

    network: Any = Field(description="biocomp Network object")
    hide_marker_tus: bool = True
    grid_gap: Tuple[float, float] = (40.0, 20.0)
    connection_style: Literal["orthogonal", "bezier", "straight"] = "orthogonal"
    style_overrides: Optional[dict] = None

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        from jeanplot.network_schematic_v2 import NetworkGeneticSchematicV2
        from jeanplot.container import Container
        from jeanplot.models import LayoutConstraints
        from jeanplot.matplotlib_renderer import MatplotlibRenderer
        from jeanplot.style import jstyle

        if self.style_overrides:
            jstyle.update(self.style_overrides)

        schematic = NetworkGeneticSchematicV2(
            network=self.network,
            hide_marker_tus=self.hide_marker_tus,
            grid_gap=self.grid_gap,
            connection_style=self.connection_style,
        )

        root = Container(
            children=[schematic],
            layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
        )
        jstyle.apply(root)

        figsize = self.figure_spec.extra_args.get("figsize", (10, 8))
        dpi = self.figure_spec.extra_args.get("dpi", 150)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("none")

        renderer = MatplotlibRenderer()
        renderer.render_component(ax, root, adjust_lims=True)

        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved genetic circuit to {self.figure_spec.output_path}")
