"""Network compute diagram figure for biocomp networks using jeanplot primitives."""

from typing import Any
from pydantic import Field, PrivateAttr
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.axes

from jeanplot.core.component import AnchorComponent
from jeanplot.core.container import Container
from jeanplot.core.models import BoxStyle, LayoutConstraints, Offset
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.svg import LineEndFlat
from jeanplot.core.text import Text

from biocomp.graphengine import is_inverse_node_type
from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class ComputeNode(Container):
    node_type: str = "unknown"
    node_label: str | None = None
    node_id: int | None = None
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(align_items="center", justify_content="center")
    )

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.style_class.append(f"node-type-{self.node_type}")
        if self.node_label:
            self.add_child(
                Text(
                    text=self.node_label,
                    id=f"lbl_{self.id}" if self.id else None,
                    style_class=["label"],
                    vertical_align="middle",
                    align="center",
                )
            )


class TranscriptionNode(ComputeNode):
    node_type: str = "transcription"
    node_label: str | None = "Tx"


class TranslationNode(ComputeNode):
    node_type: str = "translation"
    node_label: str | None = "Tl"


class ERNNode(ComputeNode):
    node_type: str = "sequestron_ERN"
    _tx_node: TranscriptionNode = PrivateAttr()
    _tl_node: TranslationNode = PrivateAttr()
    _out: AnchorComponent = PrivateAttr()
    _center: AnchorComponent = PrivateAttr()
    _tx_connector: Connection = PrivateAttr()
    _tl_connector: Connection = PrivateAttr()

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        def mk_id(p):
            return f"{p}_{self.id}" if self.id else None

        self._tx_node = TranscriptionNode(id=mk_id("tx"), is_overlay=True)
        self._tl_node = TranslationNode(id=mk_id("tl"), is_overlay=True)
        self._out = AnchorComponent(
            id=mk_id("out"), style_class=["ernout"], offset=Offset(reference_relative=(1.0, 0.5))
        )
        self._center = AnchorComponent(
            id=mk_id("center"),
            style_class=["erncenter"],
            offset=Offset(reference_relative=(0.5, 0.5)),
        )
        self._tx_connector = Connection(
            id=mk_id("txconn"),
            start_component=self._tx_node,
            end_component=self._out,
            style_class=["txconn"],
            curve_type=SimpleBezierCurve(),
            auto_route=False,
        )
        self._tl_connector = Connection(
            id=mk_id("tlconn"),
            start_component=self._tl_node,
            end_component=self._center,
            style_class=["tlconn"],
            curve_type=OrthogonalCurve(corner_radius=50, start_length=5, end_length=5),
            end_cap=LineEndFlat(),
            auto_route=False,
        )
        self.add_children(
            [
                self._tx_node,
                self._tl_node,
                self._out,
                self._center,
                self._tx_connector,
                self._tl_connector,
            ]
        )


class FluoNode(ComputeNode):
    node_type: str = "output"
    node_label: str | None = "Y"


class InvNode(ComputeNode):
    node_type: str = "inverted"
    node_label: str | None = "Inv"


class TUNode(ComputeNode):
    node_type: str = "source"


class AggregationNode(ComputeNode):
    node_type: str = "aggregation"
    collapsed: bool = False


class DeadEndNode(ComputeNode):
    node_type: str = "deadend"
    node_label: str | None = "X"


class InputNode(ComputeNode):
    node_type: str = "input"
    node_label: str | None = "In"


class BiasNode(ComputeNode):
    node_type: str = "bias"
    node_label: str | None = "B"


NODE_CLASSES = {
    "transcription": TranscriptionNode,
    "translation": TranslationNode,
    "output": FluoNode,
    "sequestron_ERN": ERNNode,
    "deadend": DeadEndNode,
    "source": TUNode,
    "input": InputNode,
    "bias": BiasNode,
    "aggregation": AggregationNode,
}


class NetworkDiagram(Container):
    network: Any = Field(description="biocomp Network object")
    simplified: bool = Field(default=True, description="collapsed aggregation view")
    disabled_tu_ids: set[str] = Field(
        default_factory=set, description="TU IDs to style as disabled"
    )
    layout: LayoutConstraints = Field(
        default_factory=lambda: LayoutConstraints(
            direction="row", gap=15, justify_content="center", align_items="stretch"
        )
    )
    style_class: list[str] = ["NetworkDiagram"]
    style: BoxStyle = Field(
        default_factory=lambda: BoxStyle(padding=(0, 0, 0, 0), margin=(0, 0, 0, 0))
    )

    _nodes: dict[int, ComputeNode] = PrivateAttr(default_factory=dict)
    _connections: list[Connection] = PrivateAttr(default_factory=list)
    _net_info: dict = PrivateAttr(default_factory=dict)
    _marker_tu_names: set[str] = PrivateAttr(default_factory=set)
    _marker_only_nodes: set[int] = PrivateAttr(default_factory=set)
    _cotx_marker_map: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if not self.network or self.network.compute_graph is None:
            raise ValueError("network with compute_graph is required")
        self._net_info = self.network.generate_network_info()
        self._marker_tu_names = self._find_marker_tu_names()
        self._marker_only_nodes = self._find_marker_only_nodes()
        self._cotx_marker_map = self._build_cotx_marker_map()
        self._build()

    @property
    def _graph(self):
        return self.network.compute_graph

    @property
    def _output_proteins(self) -> tuple:
        return self._net_info.get("output_proteins", ())

    def _find_marker_tu_names(self) -> set[str]:
        """Find TU name prefixes that produce marker proteins (inverted inputs) without ERN involvement."""
        # Use 'markers' from network info - these are the inverted input proteins
        marker_proteins = set(self._net_info.get("markers", ()))
        if not marker_proteins:
            return set()
        marker_tu_prefixes = set()
        all_parts = self._net_info.get("all_parts", {})
        for tu_full_name, parts in all_parts.items():
            # Check if this TU produces a marker protein but has no ERN parts
            has_marker_protein = bool(marker_proteins & set(parts.keys()))
            has_ern = any(cat.startswith("ERN") for cat in parts.values())
            if has_marker_protein and not has_ern:
                # Extract TU prefix (e.g., 'TU_0_1' -> 'TU_0', 'reporter_test_1' -> 'reporter')
                # tu_id on edges uses format like 'TU_0_cotx_1' or 'reporter_test_cotx'
                parts_split = tu_full_name.rsplit("_", 1)
                if len(parts_split) == 2:
                    marker_tu_prefixes.add(parts_split[0])  # 'TU_0' or 'reporter_test'
        return marker_tu_prefixes

    def _build_cotx_marker_map(self) -> dict[str, str]:
        """Map cotx_group names to their marker protein names (inverted inputs or fluo_bias)."""
        markers = set(self._net_info.get("markers", ()))
        all_parts = self._net_info.get("all_parts", {})
        cotx_to_marker: dict[str, str] = {}
        for node in self._graph.nodes.values():
            if node.node_type != "source":
                continue
            cotx = node.extra.get("cotx_group")
            if not cotx or cotx in cotx_to_marker:
                continue
            tu_name = node.extra.get("name", "")
            for tu_full_name, parts in all_parts.items():
                if not tu_full_name.startswith(tu_name):
                    continue
                for part_name in parts.keys():
                    if part_name in markers:
                        cotx_to_marker[cotx] = part_name
                        break
                if cotx in cotx_to_marker:
                    break
        for node in self._graph.nodes.values():
            if node.node_type != "bias":
                continue
            fluo_bias = node.extra.get("fluo_bias")
            if not fluo_bias or not isinstance(fluo_bias, dict):
                continue
            protein = fluo_bias.get("protein")
            if not protein:
                continue
            cotx = self._find_cotx_for_bias(node.node_id)
            if cotx and cotx not in cotx_to_marker:
                cotx_to_marker[cotx] = protein
        return cotx_to_marker

    _MARKER_COLORS: dict[str, str] = {
        "EBFP2": "#6cafc3",
        "EBFP": "#6cafc3",
        "MKO2": "#ef957d",
        "MKO": "#ef957d",
        "MKATE2": "#ef957d",
        "MKATE": "#ef957d",
        "TDTOMATO": "#ef957d",
        "MNEONGREEN": "#6ccb83",
        "MNG": "#6ccb83",
        "NEONGREEN": "#6ccb83",
        "EGFP": "#6ccb83",
        "EYFP": "#fad26d",
        "EYFPG5A": "#fad26d",
        "IRFP720": "#df9ae4",
        "IRFP": "#df9ae4",
        "MMAROON": "#d3a888",
        "MMAROON1": "#d3a888",
    }

    def _get_marker_color(self, marker: str) -> str:
        """Get the color for a marker protein name."""
        return self._MARKER_COLORS.get(marker, "#aaa")

    def _find_cotx_for_bias(self, node_id: int) -> str | None:
        """Find cotx_group for a bias node by traversing edges."""
        visited = set()
        current_id = node_id
        while current_id not in visited:
            visited.add(current_id)
            current = self._graph.nodes.get(current_id)
            if not current:
                break
            if current.node_type in ("aggregation", "source"):
                return current.extra.get("cotx_group")
            if is_inverse_node_type(current.node_type):
                orig = current.is_inverse_of
                if orig:
                    orig_node = self._graph.nodes.get(orig.node_id)
                    if orig_node and orig_node.extra.get("cotx_group"):
                        return orig_node.extra.get("cotx_group")
            outgoing = list(self._graph.get_outgoing_edges(current_id))
            if not outgoing:
                break
            current_id = outgoing[0].target_id
        return None

    def _is_marker_only_edge(self, edge) -> bool:
        """Check if edge carries only marker TU IDs."""
        tu_ids = edge.extra.get("tu_id", [])
        if not tu_ids:
            return False
        return all(
            any(tu_id.startswith(mtu + "_") for mtu in self._marker_tu_names) for tu_id in tu_ids
        )

    def _find_marker_only_nodes(self) -> set[int]:
        """Find nodes connected only by marker-only edges (except output/aggregation)."""
        if not self._marker_tu_names:
            return set()
        # For each node, check if ALL its edges are marker-only
        marker_nodes = set()
        protected_types = {"output", "aggregation", "input", "bias"}
        for node in self._graph.nodes.values():
            if node.node_type in protected_types or node.is_inverse_of is not None:
                continue
            in_edges = list(self._graph.get_incoming_edges(node.node_id))
            out_edges = list(self._graph.get_outgoing_edges(node.node_id))
            all_edges = in_edges + out_edges
            if all_edges and all(self._is_marker_only_edge(e) for e in all_edges):
                marker_nodes.add(node.node_id)
        return marker_nodes

    def _find_input_proteins(self, node_id: int, visited: set | None = None) -> list[str]:
        """Trace back from node to find input proteins feeding into it."""
        if visited is None:
            visited = set()
        if node_id in visited:
            return []
        visited.add(node_id)
        node = self._graph.nodes.get(node_id)
        if not node:
            return []
        if node.node_type == "input":
            idx = node.extra.get("input_from_output")
            if idx is not None and idx < len(self._output_proteins):
                return [self._output_proteins[idx]]
            return []
        proteins = []
        for e in self._graph.get_incoming_edges(node_id):
            proteins.extend(self._find_input_proteins(e.source_id, visited))
        return proteins

    def _is_hidden(self, node) -> bool:
        """In simplified mode, hide inputs, biases, inverse nodes, sources, numeric, and marker-only nodes."""
        if not self.simplified:
            return False
        if node.node_type in ("input", "bias", "source", "numeric"):
            return True
        if node.is_inverse_of is not None:
            return True
        if node.node_id in self._marker_only_nodes:
            return True
        return False

    def _make_node(self, node, nid: int) -> ComputeNode | None:
        kw = {"node_id": nid, "id": f"node_{nid}"}
        ntype = node.node_type

        if ntype not in NODE_CLASSES:
            return (
                InvNode(**kw)
                if is_inverse_node_type(ntype)
                else ComputeNode(node_type=ntype, node_label="?", **kw)
            )

        cls = NODE_CLASSES[ntype]

        if cls is AggregationNode:
            style = ["aggregation", "collapsed"] if self.simplified else ["aggregation"]
            cotx = node.extra.get("cotx_group")
            marker = None
            if cotx and cotx in self._cotx_marker_map:
                marker = self._cotx_marker_map[cotx].upper()
                style.append(marker)
            has_bias = any(
                self._graph.nodes.get(e.source_id)
                and self._graph.nodes[e.source_id].node_type == "bias"
                for e in self._graph.get_incoming_edges(nid)
            )
            if has_bias:
                style.append("bias_connected")
            agg = AggregationNode(style_class=style, collapsed=self.simplified, **kw)
            if self.simplified:
                for e in self._graph.get_outgoing_edges(nid):
                    src_node = self._graph.nodes.get(e.target_id)
                    if src_node and src_node.node_type == "source":
                        tu = TUNode(
                            node_id=e.target_id,
                            id=f"node_{e.target_id}",
                            style_class=["hidden_source"],
                            is_overlay=True,
                        )
                        agg.add_child(tu)
                        self._nodes[e.target_id] = tu
            return agg

        if cls is TUNode:
            name = node.extra.get("name", "")
            cotx = node.extra.get("cotx_group")
            style = ["source"]
            if name in self.disabled_tu_ids:
                style.append("disabled")
            if cotx and cotx in self._cotx_marker_map:
                style.append(self._cotx_marker_map[cotx].upper())
            return TUNode(style_class=style, **kw)

        if cls is FluoNode:
            # Use dependent_outputs for coloring (the non-marker output protein)
            proteins = self._net_info.get("dependent_outputs", ()) or self._net_info.get(
                "output_proteins", ()
            )
            f = FluoNode(**kw)
            if proteins:
                f.style_class.append(proteins[0].upper())
            return f

        if cls is ERNNode:
            ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
            e = ERNNode(**kw)
            e.add_child(Text(text=ern_name, style_class=["ern_name"]))
            return e

        if cls is InputNode:
            idx = node.extra.get("input_from_output")
            if idx is not None and idx < len(self._output_proteins):
                protein = self._output_proteins[idx]
                return InputNode(node_label=protein, style_class=["input", protein.upper()], **kw)
            return InputNode(**kw)

        if cls is BiasNode:
            cotx = self._find_cotx_for_bias(nid)
            style = ["bias"]
            if cotx and cotx in self._cotx_marker_map:
                style.append(self._cotx_marker_map[cotx].upper())
            return BiasNode(style_class=style, **kw)

        return cls(**kw)

    def _build(self):
        self._nodes.clear()
        for n in self._graph.nodes.values():
            if not self._is_hidden(n):
                if comp := self._make_node(n, n.node_id):
                    self._nodes[n.node_id] = comp

        self._connections = []
        for e in self._graph.edges.values():
            src, tgt = e.source_id, e.target_id
            if src not in self._nodes or tgt not in self._nodes:
                continue
            # Skip marker-only edges in simplified mode
            if self.simplified and self._is_marker_only_edge(e):
                continue
            src_comp, tgt_comp = self._nodes[src], self._nodes[tgt]
            if src_comp.node_type == "aggregation" or tgt_comp.node_type in (
                "aggregation",
                "sequestron_ERN",
            ):
                continue
            start = src_comp._out if isinstance(src_comp, ERNNode) else src_comp
            style = [
                "comp-connection",
                f"src-{src_comp.node_type}",
                f"dst-{tgt_comp.node_type}",
                f"slot-{e.to_input_slot}",
            ]
            self._connections.append(
                Connection(
                    id=f"conn_{src}_{tgt}_{e.to_input_slot}",
                    start_component=start,
                    end_component=tgt_comp,
                    line_width=1,
                    style_class=style,
                )
            )

        self.children = self._connections + self._create_layers()

    def _create_layers(self) -> list[Container]:
        layers, layed_out = [], set()
        dep_map = defaultdict(list)
        for e in self._graph.edges.values():
            dep_map[e.target_id].append(e.source_id)
        ern_ids = {n.node_id for n in self._graph.nodes.values() if n.node_type == "sequestron_ERN"}

        # Input layer: aggregations (which contain hidden sources)
        input_layer = Container(id="layer_input", style_class=["input_layer", "layer"])
        for nid, comp in sorted(self._nodes.items()):
            if comp.node_type == "aggregation":
                input_layer.add_child(comp)
                layed_out.add(nid)
                # Mark hidden source children as layed_out too
                for child in comp.children:
                    if hasattr(child, "node_id") and child.node_id is not None:
                        layed_out.add(child.node_id)
        if input_layer.children:
            layers.append(input_layer)

        # ERN layers
        ern_in_nodes = [e for e in ern_ids if e in self._nodes]
        for i, ern_layer in enumerate(self._graph.topological_order(ern_in_nodes)):
            if not ern_layer:
                continue
            lc = Container(style_class=["main_layer", f"main_layer_{i}", "layer"])
            lc.add_child(
                Text(
                    text=f"Layer {i + 1}",
                    font_size=5,
                    style_class=["layer_title"],
                    offset=Offset(reference_relative=(0.5, 1), relative=(-0.6, 1.5)),
                    is_overlay=True,
                    id=f"title_ern_{i}",
                )
            )
            for eid in sorted(ern_layer):
                if eid not in self._nodes:
                    continue
                ern = self._nodes[eid]
                lc.add_child(ern)
                layed_out.add(eid)
                for uid in dep_map.get(eid, []):
                    if uid in self._nodes and uid not in ern_ids:
                        up = self._nodes[uid]
                        attach = (
                            ern._tl_node
                            if isinstance(up, TranslationNode)
                            else (ern._tx_node if isinstance(up, TranscriptionNode) else None)
                        )
                        if attach:
                            up.attached_to, up.show = attach, False
                            lc.add_child(up)
                            layed_out.add(uid)
            if len([c for c in lc.children if not isinstance(c, Text)]) > 0:
                layers.append(lc)

        # Output layer
        out_ids = [n.node_id for n in self._graph.nodes.values() if n.node_type == "output"]
        if out_ids and (oid := out_ids[0]) in self._nodes:
            ol = Container(id="layer_output", style_class=["output_layer", "layer"])
            out = self._nodes[oid]
            ol.add_child(out)
            layed_out.add(oid)
            for uid in dep_map.get(oid, []):
                if uid in self._nodes and uid not in ern_ids:
                    up = self._nodes[uid]
                    if isinstance(up, (TranscriptionNode, TranslationNode)):
                        up.attached_to, up.attachment_offset, up.show = (
                            out,
                            Offset(absolute=(-40, 0)),
                            True,
                        )
                        ol.add_child(up)
                        layed_out.add(uid)
            if ol.children:
                layers.append(ol)

        # Auto layer for remaining nodes
        remaining = set(self._nodes) - layed_out - ern_ids - set(out_ids)
        if remaining:
            auto = []
            for i, al in enumerate(self._graph.topological_order(list(remaining))):
                if not al:
                    continue
                ac = Container(id=f"layer_auto_{i}", style_class=["auto_layer", "layer"])
                for nid in sorted(al, reverse=True):
                    if nid in self._nodes:
                        ac.add_child(self._nodes[nid])
                        layed_out.add(nid)
                if ac.children:
                    auto.append(ac)
            layers = layers[:1] + auto + layers[1:]

        return layers


_MARKER_COLORS_THEME = {
    "EBFP2": {"base": "#6cafc3", "bright": "#1AD5FF60"},
    "EBFP": {"base": "#6cafc3", "bright": "#1AD5FF60"},
    "MKO2": {"base": "#ef957d", "bright": "#FF2C4944"},
    "MKO": {"base": "#ef957d", "bright": "#FF2C4944"},
    "MKATE2": {"base": "#ef957d", "bright": "#FF2C4944"},
    "MKATE": {"base": "#ef957d", "bright": "#FF2C4944"},
    "TDTOMATO": {"base": "#ef957d", "bright": "#FF2C4944"},
    "MNEONGREEN": {"base": "#6ccb83", "bright": "#0EFF7377"},
    "MNG": {"base": "#6ccb83", "bright": "#0EFF7377"},
    "NEONGREEN": {"base": "#6ccb83", "bright": "#0EFF7377"},
    "EGFP": {"base": "#6ccb83", "bright": "#0EFF7377"},
    "EYFP": {"base": "#fad26d", "bright": "#FFC83Caa"},
    "EYFPG5A": {"base": "#fad26d", "bright": "#FFC83Caa"},
    "IRFP720": {"base": "#df9ae4", "bright": "#FF82FCaa"},
    "IRFP": {"base": "#df9ae4", "bright": "#FF82FCaa"},
    "MMAROON": {"base": "#d3a888", "bright": "#FF9853a0"},
    "MMAROON1": {"base": "#d3a888", "bright": "#FF9853a0"},
}


def _apply_marker_colors_to_collapsed_aggregations(diagram: "NetworkDiagram", Shadow):
    """Post-process collapsed aggregation nodes to apply marker colors."""

    def apply_to_component(comp):
        if isinstance(comp, AggregationNode) and comp.collapsed:
            for sc in comp.style_class:
                if sc in _MARKER_COLORS_THEME:
                    colors = _MARKER_COLORS_THEME[sc]
                    comp.style.background_color = colors["base"]
                    comp.style.shadow = Shadow(
                        color=colors["bright"], blur_radius=8, resolution=0.01, z_index=2
                    )
                    break
        if hasattr(comp, "children"):
            for child in comp.children:
                apply_to_component(child)

    apply_to_component(diagram)


def render_diagram_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    simplified: bool = True,
    disabled_tu_ids: set[str] | None = None,
    style_overrides: dict | None = None,
    title: str | None = None,
    **_kwargs,
):
    """Render a network compute diagram to an existing matplotlib axes."""
    from jeanplot import MatplotlibRenderer, jstyle
    from jeanplot.core import (
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
    )
    from dracon import load, resolve_all_lazy
    import importlib.resources

    types = [
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
    theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
        "network_diagram.yaml"
    )
    theme = load(str(theme_file), context={t.__name__: t for t in types}, raw_dict=True)
    resolve_all_lazy(theme)
    jstyle.update(theme)
    if style_overrides:
        jstyle.update(style_overrides)

    diagram = NetworkDiagram(
        network=network, simplified=simplified, disabled_tu_ids=disabled_tu_ids or set()
    )
    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)

    if simplified:
        _apply_marker_colors_to_collapsed_aggregations(diagram, Shadow)

    ax.set_aspect("equal")
    ax.axis("off")
    MatplotlibRenderer().render_component(ax, root, adjust_lims=True)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


class NetworkDiagramFigure(Figure):
    """Figure that renders a network compute diagram using jeanplot."""

    network: Any = Field(description="biocomp Network object")
    simplified: bool = True
    disabled_tu_ids: set[str] | None = None
    style_overrides: dict | None = None

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
            disabled_tu_ids=self.disabled_tu_ids,
            style_overrides=self.style_overrides,
        )
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved network diagram to {self.figure_spec.output_path}")
