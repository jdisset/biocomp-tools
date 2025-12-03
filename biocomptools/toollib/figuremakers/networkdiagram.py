"""Network diagram figure for biocomp networks"""

from typing import Any, Optional
from pydantic import Field
import matplotlib.pyplot as plt
import matplotlib.axes

from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def render_diagram_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    simplified: bool = True,
    style_overrides: Optional[dict] = None,
    title: Optional[str] = None,
    **_kwargs,  # absorb extra kwargs from PlotConfig
):
    """Render a network diagram to an existing matplotlib axes."""
    from jeanplot.network_diagram_v2 import NetworkDiagramV2
    from jeanplot.container import Container
    from jeanplot.models import LayoutConstraints
    from jeanplot.matplotlib_renderer import MatplotlibRenderer
    from jeanplot.style import jstyle

    if style_overrides:
        jstyle.update(style_overrides)

    diagram = NetworkDiagramV2(
        network=network,
        simplified=simplified,
    )

    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)

    ax.set_aspect("equal")
    ax.axis("off")

    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


class NetworkDiagramFigure(Figure):
    """Figure that renders a network compute diagram using jeanplot"""

    network: Any = Field(description="biocomp Network object")
    simplified: bool = True
    style_overrides: Optional[dict] = None

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        from jeanplot.network_diagram_v2 import NetworkDiagramV2
        from jeanplot.container import Container
        from jeanplot.models import LayoutConstraints
        from jeanplot.matplotlib_renderer import MatplotlibRenderer
        from jeanplot.style import jstyle

        if self.style_overrides:
            jstyle.update(self.style_overrides)

        diagram = NetworkDiagramV2(
            network=self.network,
            simplified=self.simplified,
        )

        root = Container(
            children=[diagram],
            layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
        )
        jstyle.apply(root)

        figsize = self.figure_spec.extra_args.get("figsize", (10, 8))
        dpi = self.figure_spec.extra_args.get("dpi", 150)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        renderer = MatplotlibRenderer()
        renderer.render_component(ax, root, adjust_lims=True)

        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved network diagram to {self.figure_spec.output_path}")
