"""Network compute diagram figure for biocomp networks using jeanplot primitives."""

from typing import Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import matplotlib.pyplot as plt
import matplotlib.axes

from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

NODE_STYLES = {
    "source": {"label": None, "style_class": ["node", "source"]},
    "transcription": {"label": "Tx", "style_class": ["node", "transcription"]},
    "translation": {"label": "Tl", "style_class": ["node", "translation"]},
    "sequestron_ERN": {"label": None, "style_class": ["node", "ern"]},
    "output": {"label": "Y", "style_class": ["node", "output"]},
    "aggregation": {"label": None, "style_class": ["node", "aggregation"]},
}


@dataclass
class DiagramNode:
    node_id: int
    node_type: str
    label: str | None
    layer: int
    style_classes: list[str] = field(default_factory=list)
    marker: str | None = None


@dataclass
class DiagramEdge:
    source_id: int
    target_id: int
    slot: int = 0


class ComputeDiagramData(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    network: Any
    simplified: bool = True

    _nodes: dict[int, DiagramNode] = PrivateAttr(default_factory=dict)
    _edges: list[DiagramEdge] = PrivateAttr(default_factory=list)
    _layers: list[list[int]] = PrivateAttr(default_factory=list)

    @model_validator(mode='after')
    def _extract_diagram(self):
        if not self.network or not self.network.compute_graph:
            return self

        graph = self.network.compute_graph
        net_info = self.network.generate_network_info()
        markers = set(net_info.get("markers", []))

        excluded = set()
        if self.simplified:
            for n in graph.nodes.values():
                if n.is_inverse_of is not None:
                    excluded.add(n.node_id)
                if n.node_type in ("input", "bias"):
                    excluded.add(n.node_id)

        for node in graph.nodes.values():
            if node.node_id in excluded:
                continue

            style_info = NODE_STYLES.get(node.node_type, {"label": "?", "style_class": ["node"]})
            label = style_info["label"]
            style_classes = list(style_info["style_class"])
            marker = None

            if node.node_type == "source":
                name = node.extra.get("name", "")
                marker = next((m for m in markers if m in name), None)
                if marker:
                    style_classes.append(marker)
                    label = marker

            elif node.node_type == "sequestron_ERN":
                ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
                label = ern_name
                if "_" in ern_name:
                    style_classes.append(ern_name.split("_")[0])

            self._nodes[node.node_id] = DiagramNode(
                node_id=node.node_id,
                node_type=node.node_type,
                label=label,
                layer=node.extra.get("layer_id", 0),
                style_classes=style_classes,
                marker=marker,
            )

        for edge in graph.edges.values():
            if edge.source_id in self._nodes and edge.target_id in self._nodes:
                src_type = self._nodes[edge.source_id].node_type
                tgt_type = self._nodes[edge.target_id].node_type
                if src_type == "aggregation" or tgt_type == "aggregation":
                    continue
                if tgt_type == "sequestron_ERN":
                    continue

                self._edges.append(
                    DiagramEdge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        slot=edge.to_input_slot,
                    )
                )

        node_ids = list(self._nodes.keys())
        self._layers = graph.topological_order(node_ids) if node_ids else []

        return self

    @property
    def nodes(self) -> dict[int, DiagramNode]:
        return self._nodes

    @property
    def edges(self) -> list[DiagramEdge]:
        return self._edges

    @property
    def layers(self) -> list[list[int]]:
        return self._layers


def build_diagram_component(data: ComputeDiagramData):
    from jeanplot import Container, Text, Connection, LayoutConstraints, BoxStyle, SimpleBezierCurve

    if not data.nodes:
        return Container(id="empty_diagram")

    node_components: dict[int, Container] = {}

    for node_id, node in data.nodes.items():
        node_box = Container(
            id=f"node_{node_id}",
            style_class=node.style_classes,
            layout=LayoutConstraints(align_items="center", justify_content="center"),
        )
        if node.label:
            node_box.add_child(
                Text(
                    text=node.label,
                    style_class=["node_label"],
                )
            )
        node_components[node_id] = node_box

    layer_containers = []
    for layer_idx, layer_node_ids in enumerate(data.layers):
        layer_nodes = [node_components[nid] for nid in layer_node_ids if nid in node_components]
        if not layer_nodes:
            continue

        layer = Container(
            id=f"layer_{layer_idx}",
            style_class=["layer"],
            children=layer_nodes,
            layout=LayoutConstraints(direction="column", gap=15, align_items="center"),
        )
        layer_containers.append(layer)

    root = Container(
        id="diagram",
        style_class=["compute_diagram"],
        children=layer_containers,
        layout=LayoutConstraints(
            direction="row", gap=40, align_items="center", justify_content="center"
        ),
        style=BoxStyle(padding=(20, 20, 20, 20)),
    )

    for edge in data.edges:
        if edge.source_id in node_components and edge.target_id in node_components:
            conn = Connection(
                id=f"edge_{edge.source_id}_{edge.target_id}",
                start_component=node_components[edge.source_id],
                end_component=node_components[edge.target_id],
                style_class=["edge", f"slot_{edge.slot}"],
                curve_type=SimpleBezierCurve(),
                is_overlay=True,
            )
            root.add_child(conn)

    return root


def render_diagram_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    simplified: bool = True,
    style_overrides: Optional[dict] = None,
    title: Optional[str] = None,
    use_legacy: bool = True,
    **_kwargs,
):
    """Render a network compute diagram to an existing matplotlib axes.

    Args:
        use_legacy: If True, use the deprecated NetworkDiagramV2 for full-featured rendering.
                   If False, use the new simplified implementation.
    """
    from jeanplot import Container, LayoutConstraints, MatplotlibRenderer, jstyle

    if style_overrides:
        jstyle.update(style_overrides)

    if use_legacy:
        from jeanplot._deprecated.network_diagram_v2 import NetworkDiagramV2

        diagram = NetworkDiagramV2(network=network, simplified=simplified)
        root = Container(
            children=[diagram],
            layout=LayoutConstraints(
                direction="row", justify_content="center", align_items="stretch"
            ),
        )
        jstyle.apply(root)
    else:
        data = ComputeDiagramData(network=network, simplified=simplified)
        root = build_diagram_component(data)

    ax.set_aspect("equal")
    ax.axis("off")

    renderer = MatplotlibRenderer()
    renderer.render_component(ax, root, adjust_lims=True)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


class NetworkDiagramFigure(Figure):
    """Figure that renders a network compute diagram using jeanplot."""

    network: Any = Field(description="biocomp Network object")
    simplified: bool = True
    style_overrides: Optional[dict] = None
    use_legacy: bool = True

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        figsize = self.figure_spec.extra_args.get("figsize", (10, 8))
        dpi = self.figure_spec.extra_args.get("dpi", 150)

        fig, ax = plt.subplots(figsize=figsize)
        render_diagram_to_ax(
            network=self.network,
            ax=ax,
            simplified=self.simplified,
            style_overrides=self.style_overrides,
            use_legacy=self.use_legacy,
        )

        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved network diagram to {self.figure_spec.output_path}")
