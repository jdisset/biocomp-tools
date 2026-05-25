# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Network compute diagram figure for biocomp networks using jeanplot primitives."""

from typing import Any, Literal
from pydantic import BaseModel, Field, PrivateAttr
from collections import defaultdict
from statistics import median
import matplotlib.axes

from jeanplot.core.component import AnchorComponent, Component
from jeanplot.core.connection_label import ConnectionLabel
from jeanplot.core.container import Container
from jeanplot.core.ordering import min_crossing_permutation, relax_y
from jeanplot.core.models import BoxStyle, LayoutConstraints, Offset, Size
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.svg import LineEndFlat
from jeanplot.core.text import Text
from jeanplot.gene.elements import _format_ratio_multiplier

from biocomp.graphengine import is_inverse_node_type
from biocomp.ratio_schema import get_slot_entries
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class LayoutSpec(BaseModel):
    """Declarative alignment constraints for consistent network diagram layouts.

    Constrains the layout engine to produce consistent results across different
    networks, enabling side-by-side visual comparison.
    """

    canvas_size: Size | None = None
    ern_slot_order: list[str] | list[list[str]] | None = None
    max_ern_layers: int | None = None
    column_widths: dict[str, float] | None = None
    layer_min_height: float | None = None

    def get_slot_order_for_layer(self, layer_idx: int) -> list[str] | None:
        """Get the ERN slot order for a specific topological layer.

        - None -> no slot ordering (use default sort)
        - flat list[str] -> same order for all layers
        - list[list[str]] -> per-layer order; out-of-range layers get None
        """
        if self.ern_slot_order is None:
            return None
        if not self.ern_slot_order:
            return None
        if isinstance(self.ern_slot_order[0], list):
            per_layer = self.ern_slot_order
            return per_layer[layer_idx] if layer_idx < len(per_layer) else None
        return self.ern_slot_order

    @classmethod
    def from_networks(cls, networks: list[Any]) -> "LayoutSpec":
        """Compute a shared LayoutSpec from multiple networks.

        Inspects all networks to derive:
        - ern_slot_order: per-layer union of ERN type names (sorted alphabetically)
        - max_ern_layers: max ERN topological depth across all networks

        Does NOT auto-compute canvas_size - set that manually.
        """
        max_layers = 0
        per_layer_types: dict[int, set[str]] = {}

        for net in networks:
            graph = net.compute_graph
            if graph is None:
                continue
            ern_ids = [n.node_id for n in graph.nodes.values() if n.node_type == "sequestron_ERN"]
            if not ern_ids:
                continue
            layers = list(graph.topological_order(ern_ids))
            max_layers = max(max_layers, len(layers))
            for i, layer in enumerate(layers):
                for eid in layer:
                    node = graph.nodes[eid]
                    ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
                    if ern_name:
                        per_layer_types.setdefault(i, set()).add(ern_name)

        if not per_layer_types:
            return cls()

        ern_slot_order = [sorted(per_layer_types.get(i, set())) for i in range(max_layers)]

        return cls(
            ern_slot_order=ern_slot_order,
            max_ern_layers=max_layers if max_layers > 0 else None,
        )


def semantic_key(node: Any, graph: Any) -> str:
    """Compute a stable semantic identifier for a network graph node.

    These keys enable cross-network node correspondence for layout alignment.
    """
    ntype = node.node_type
    nid = node.node_id

    if ntype == "aggregation":
        cotx = node.extra.get("cotx_group", str(nid))
        return f"agg:{cotx}"

    if ntype == "sequestron_ERN":
        ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
        return f"ern:{ern_name or nid}"

    if ntype == "output":
        output_proteins = []
        for n in graph.nodes.values():
            if n.node_type == "output":
                output_proteins.append(n.node_id)
        idx = output_proteins.index(nid) if nid in output_proteins else 0
        return f"output:{idx}"

    if ntype == "input":
        idx = node.extra.get("input_from_output")
        return f"input:{idx}" if idx is not None else f"input:{nid}"

    if ntype == "bias":
        cotx = node.extra.get("cotx_group", str(nid))
        return f"bias:{cotx}"

    if ntype == "transcription":
        return f"tx:{nid}"

    if ntype == "translation":
        return f"tl:{nid}"

    return f"{ntype}:{nid}"


def _format_ratio_proportion(val: float) -> str:
    pct = val * 100
    if abs(pct - round(pct)) < 0.1:
        return f"({int(round(pct))}%)"
    return f"({pct:.1f}%)"


EMBEDDING_CATEGORY: dict[str, str] = {
    "tc_rate": "promoters",
    "tl_rate": "uORFs",
    "affinity": "ERNs",
}

IMPLICIT_EMPTY: dict[str, set[str]] = {
    "tl_rate": {"00_empty_tc"},
}


def _format_edge_parts(content_embedding_names: dict[str, tuple[str, ...]]) -> str:
    """Format edge embedding info: actual names for single parts, counts for multiple."""
    segments: list[str] = []
    for emb_name, part_names in content_embedding_names.items():
        empty = IMPLICIT_EMPTY.get(emb_name, set())
        real = [p for p in part_names if p and p not in empty]
        if not real:
            continue
        cat = EMBEDDING_CATEGORY.get(emb_name, emb_name)
        if len(real) == 1:
            segments.append(real[0])
        else:
            segments.append(f"{{{len(real)} {cat}}}")
    return " \u00b7 ".join(segments)


_EMBEDDING_FOR_TARGET: dict[str, set[str]] = {
    "transcription": {"tc_rate"},
    "translation": {"tl_rate"},
}


def _filter_embeddings_for_target(
    content_embedding_names: dict[str, tuple[str, ...]], target_type: str
) -> dict[str, tuple[str, ...]]:
    """Return only the embeddings relevant for the given target node type."""
    allowed = _EMBEDDING_FOR_TARGET.get(target_type)
    if allowed is not None:
        return {k: v for k, v in content_embedding_names.items() if k in allowed}
    # For other targets (output, etc.), show everything except tc/tl rate
    all_targeted = set().union(*_EMBEDDING_FOR_TARGET.values())
    return {k: v for k, v in content_embedding_names.items() if k not in all_targeted}


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
    show_ratios: bool = Field(default=False, description="Show ratio text labels on edges")
    ratio_normalization: Literal["min", "sum"] = Field(
        default="sum",
        description="'min' = divide by min (xN labels), 'sum' = divide by sum (% labels)",
    )
    variable_thickness: bool = Field(
        default=False, description="Scale edge width by ratio or embedding"
    )
    show_edge_parts: bool = Field(default=False, description="Show embedding part names on edges")
    thickness_range: tuple[float, float] = Field(
        default=(0.5, 4.0),
        description="(min, max) multiplier on base line_width for variable-thickness edges",
    )
    ratio_thickness_range: tuple[float, float] = Field(
        default=(0.25, 7.0),
        description="(min, max) multiplier on base line_width for ratio-driven edges.",
    )
    embedding_thickness_map: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "tc_rate": {"hef1a": 1},
            "tl_rate": {
                "None": 15,
                "1w_uorf": 10,
                "1x_uorf": 9,
                "2x_uorf": 8,
                "3x_uorf": 7,
                "4x_uorf": 6,
                "5x_uorf": 5,
                "6x_uorf": 4,
                "7x_uorf": 3,
                "8x_uorf": 2,
                "9x_uorf": 1,
            },
        },
        description="Embedding name -> (part name -> numeric value) for thickness. Case-insensitive.",
    )
    layout_spec: LayoutSpec | None = Field(
        default=None,
        description="Declarative layout constraints for consistent cross-diagram alignment",
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
    _connection_labels: list[ConnectionLabel] = PrivateAttr(default_factory=list)
    _thickness_raw: dict[str, list[tuple[str, float]]] = PrivateAttr(default_factory=dict)
    _net_info: dict = PrivateAttr(default_factory=dict)
    _marker_tu_ids: set[str] = PrivateAttr(default_factory=set)
    _marker_only_nodes: set[int] = PrivateAttr(default_factory=set)
    _cotx_marker_map: dict[str, str] = PrivateAttr(default_factory=dict)
    _collapsed_marker_proteins: tuple[str, ...] = PrivateAttr(default_factory=tuple)
    _ratio_map: dict[int, tuple[float, float]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if not self.network or self.network.compute_graph is None:
            raise ValueError("network with compute_graph is required")
        self._net_info = self.network.generate_network_info()
        self._collapsed_marker_proteins = tuple(
            self.network.get_inverted_input_proteins(include_biases=True)
        )
        self._marker_tu_ids = self._find_marker_tu_ids()
        self._cotx_marker_map = self._build_cotx_marker_map()
        self._marker_only_nodes = self._find_marker_only_nodes()
        self._build()
        if self.layout_spec and self.layout_spec.canvas_size:
            self.min_dimensions = self.layout_spec.canvas_size.model_copy()
            self.max_dimensions = self.layout_spec.canvas_size.model_copy()

    @property
    def _graph(self):
        return self.network.compute_graph

    @property
    def _output_proteins(self) -> tuple:
        return tuple(self.network.get_output_proteins())

    @property
    def _dependent_output_proteins(self) -> tuple:
        return tuple(self.network.get_dependent_output_proteins())

    @staticmethod
    def _edge_tu_ids(edge) -> tuple[str, ...]:
        tu_ids = edge.extra.get("tu_id", [])
        if tu_ids is None:
            return ()
        if isinstance(tu_ids, str):
            return (tu_ids,)
        return tuple(tu_ids)

    @staticmethod
    def _edge_proteins(edge) -> set[str]:
        proteins = set()
        for item in edge.content:
            proteins.add(item.name if hasattr(item, "name") else str(item))
        return proteins

    def _find_marker_tu_ids(self) -> set[str]:
        """Find TU IDs that correspond to collapsed marker inputs."""
        marker_proteins = set(self._collapsed_marker_proteins)
        if not marker_proteins:
            return set()
        marker_tu_ids = set()
        for source_node in self._graph.get_nodes_by_type("source"):
            for edge in self._graph.get_outgoing_edges(source_node.node_id):
                if edge.content_type != "DNA":
                    continue
                edge_tu_ids = self._edge_tu_ids(edge)
                if not edge_tu_ids:
                    continue
                if self._edge_proteins(edge) & marker_proteins:
                    marker_tu_ids.update(edge_tu_ids)
        return marker_tu_ids

    def _build_cotx_marker_map(self) -> dict[str, str]:
        """Map cotx_group names to marker proteins from inputs/biases."""
        markers = set(self._collapsed_marker_proteins)
        cotx_to_marker: dict[str, str] = {}
        for source_node in self._graph.get_nodes_by_type("source"):
            cotx = source_node.extra.get("cotx_group")
            if not cotx or cotx in cotx_to_marker:
                continue
            for edge in self._graph.get_outgoing_edges(source_node.node_id):
                if edge.content_type != "DNA":
                    continue
                marker_hits = self._edge_proteins(edge) & markers
                if marker_hits:
                    cotx_to_marker[cotx] = sorted(marker_hits)[0]
                    break

        for node in self._graph.get_nodes_by_type("bias"):
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

    def _find_cotx_for_node(self, node_id: int) -> str | None:
        """Find cotx_group for a node by traversing toward aggregation/source."""
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
                    if orig_node:
                        cotx = orig_node.extra.get("cotx_group")
                        if cotx:
                            return cotx
                        if orig_node.node_type == "source":
                            for edge in self._graph.get_incoming_edges(orig_node.node_id):
                                up = self._graph.nodes.get(edge.source_id)
                                if up and up.node_type == "aggregation":
                                    cotx = up.extra.get("cotx_group")
                                    if cotx:
                                        return cotx
            outgoing = list(self._graph.get_outgoing_edges(current_id))
            if not outgoing:
                break
            current_id = outgoing[0].target_id
        return None

    def _find_cotx_for_bias(self, node_id: int) -> str | None:
        """Find cotx_group for a bias node."""
        return self._find_cotx_for_node(node_id)

    def _find_aggregation_for_cotx(self, cotx: str | None) -> int | None:
        if not cotx:
            return None
        for node in self._graph.get_nodes_by_type("aggregation"):
            if node.extra.get("cotx_group") == cotx:
                return node.node_id
        return None

    def _is_marker_only_edge(self, edge) -> bool:
        """Check if edge carries only marker TU IDs."""
        tu_ids = self._edge_tu_ids(edge)
        if not tu_ids:
            return False
        marker_tu_ids = self._marker_tu_ids
        return all(tu_id in marker_tu_ids for tu_id in tu_ids)

    def _find_marker_only_nodes(self) -> set[int]:
        """Find nodes connected only by marker-only edges (except output/aggregation)."""
        if not self._marker_tu_ids:
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
        """In simplified mode, hide inverse/source internals and marker-only subgraphs."""
        if not self.simplified:
            return False
        if node.node_type in ("source", "numeric"):
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
            proteins = self._dependent_output_proteins or self._output_proteins
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

    def _build_ratio_map(self) -> dict[int, tuple[float, float]]:
        """Map source node_id -> (raw_ratio, normalized_ratio) via aggregation ratio schemas."""
        ratio_map: dict[int, tuple[float, float]] = {}
        # Build source_id (str) -> node_id mapping
        source_id_to_nid: dict[str, int] = {}
        for node in self._graph.get_nodes_by_type("source"):
            sid = node.extra.get("source_id")
            if sid is not None:
                source_id_to_nid[str(sid)] = node.node_id

        for agg_node in self._graph.get_nodes_by_type("aggregation"):
            entries = get_slot_entries(agg_node.extra, require=False)
            if not entries:
                continue
            raw_ratios = [float(e.get("ratio", 1.0)) for e in entries]
            positive = [r for r in raw_ratios if r > 0]
            if self.ratio_normalization == "sum":
                divisor = sum(positive) if positive else 1.0
            else:
                divisor = min(positive) if positive else 1.0
            for entry, raw_r in zip(entries, raw_ratios, strict=True):
                sid = str(entry["source_id"])
                nid = source_id_to_nid.get(sid)
                if nid is not None:
                    norm_r = raw_r / divisor if divisor > 0 else 1.0
                    ratio_map[nid] = (raw_r, norm_r)
        return ratio_map

    def _format_ratio_label(self, norm_r: float) -> str:
        if self.ratio_normalization == "sum":
            return _format_ratio_proportion(norm_r)
        return _format_ratio_multiplier(norm_r)

    def _get_marker_ratio(self, cotx: str) -> float | None:
        """Get normalized ratio for the marker source in a cotransfection group."""
        for node in self._graph.get_nodes_by_type("source"):
            if node.extra.get("cotx_group") != cotx:
                continue
            nid = node.node_id
            if nid not in self._ratio_map:
                continue
            for edge in self._graph.get_outgoing_edges(nid):
                if edge.content_type != "DNA":
                    continue
                tu_ids = self._edge_tu_ids(edge)
                if any(tid in self._marker_tu_ids for tid in tu_ids):
                    _, norm_r = self._ratio_map[nid]
                    return norm_r
        return None

    def _lookup_embedding_thickness(
        self, edge, target_type: str | None = None
    ) -> tuple[str, float] | None:
        """Find thickness group and value from edge embeddings.

        When target_type is given, prefer the embedding relevant for that node type
        (e.g. tl_rate for translation, tc_rate for transcription).
        """
        emb_names = getattr(edge, "content_embedding_names", None)
        if not emb_names:
            return None
        preferred = _EMBEDDING_FOR_TARGET.get(target_type) if target_type else None
        ordered = emb_names.items()
        if preferred:
            ordered = sorted(ordered, key=lambda kv: 0 if kv[0] in preferred else 1)
        for emb_name, part_names in ordered:
            lower_map = self._lower_embedding_maps.get(emb_name)
            if lower_map is None:
                continue
            implicit_empty = IMPLICIT_EMPTY.get(emb_name, set())
            for pname in part_names:
                key = "none" if (pname and pname in implicit_empty) else (pname or "None").lower()
                if key in lower_map:
                    return (emb_name, lower_map[key])
        return None

    def _build(self):
        self._nodes.clear()
        # Pre-compute case-insensitive embedding maps once
        self._lower_embedding_maps: dict[str, dict[str, float]] = {
            emb: {k.lower(): v for k, v in part_map.items()}
            for emb, part_map in self.embedding_thickness_map.items()
        }
        for n in self._graph.nodes.values():
            if not self._is_hidden(n):
                if comp := self._make_node(n, n.node_id):
                    self._nodes[n.node_id] = comp

        self._ratio_map = (
            self._build_ratio_map() if (self.show_ratios or self.variable_thickness) else {}
        )

        self._connections = []
        self._connection_labels = []
        self._thickness_raw.clear()
        labeled_sources: set[int] = set()

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

            conn_id = f"conn_{src}_{tgt}_{e.to_input_slot}"
            conn_classes = [
                "comp-connection",
                f"src-{src_comp.node_type}",
                f"dst-{tgt_comp.node_type}",
                f"slot-{e.to_input_slot}",
            ]
            self._connections.append(
                Connection(
                    id=conn_id,
                    start_component=start,
                    end_component=tgt_comp,
                    style_class=conn_classes,
                )
            )

            # Collect raw thickness values per group
            if self.variable_thickness:
                if src in self._ratio_map:
                    _, norm_r = self._ratio_map[src]
                    self._thickness_raw.setdefault("ratio", []).append((conn_id, norm_r))
                else:
                    result = self._lookup_embedding_thickness(e, tgt_comp.node_type)
                    if result is not None:
                        group, value = result
                        self._thickness_raw.setdefault(group, []).append((conn_id, value))

            # Derive marker classes for annotation labels
            src_node = self._graph.nodes.get(src)
            cotx = src_node.extra.get("cotx_group") if src_node else None
            marker = self._cotx_marker_map.get(cotx) if cotx else None
            marker_classes = [marker.upper()] if marker else []

            # Base style classes shared by all annotations on this edge
            base_label_classes = [
                "edge_annotation",
                f"src-{src_comp.node_type}",
                f"dst-{tgt_comp.node_type}",
                f"slot-{e.to_input_slot}",
                *marker_classes,
            ]

            # Ratio annotation (one per source node, avoid duplicates)
            if self.show_ratios and src in self._ratio_map and src not in labeled_sources:
                _, norm_r = self._ratio_map[src]
                self._connection_labels.append(
                    ConnectionLabel(
                        text=self._format_ratio_label(norm_r),
                        style_class=["edge_ratio", *base_label_classes],
                        connection=conn_id,
                        font_size=5.0,
                        font_weight="bold",
                        color="#666666",
                    )
                )
                labeled_sources.add(src)

            # Edge part name annotation - filter by target node type
            if self.show_edge_parts and e.content_embedding_names:
                filtered = _filter_embeddings_for_target(
                    e.content_embedding_names, tgt_comp.node_type
                )
                parts_text = _format_edge_parts(filtered)
                if parts_text:
                    embed_classes = [f"embed-{k}" for k in filtered]
                    cat_classes = [
                        f"cat-{EMBEDDING_CATEGORY[k]}" for k in filtered if k in EMBEDDING_CATEGORY
                    ]
                    self._connection_labels.append(
                        ConnectionLabel(
                            text=parts_text,
                            style_class=[
                                "edge_part",
                                *base_label_classes,
                                *embed_classes,
                                *cat_classes,
                            ],
                            connection=conn_id,
                            font_size=4.5,
                            font_weight="normal",
                            color="#888888",
                        )
                    )

        self.children = self._connections + self._connection_labels + self._create_layers()

    def apply_variable_line_widths(self) -> None:
        """Normalize raw thickness values and multiply onto connection line_width.

        Called after jstyle.apply() so that self.thickness_range reflects the
        theme value rather than the field default.
        """
        emb_min_t, emb_max_t = self.thickness_range
        ratio_min_t, ratio_max_t = self.ratio_thickness_range
        conn_multipliers: dict[str, float] = {}
        for group_name, entries in self._thickness_raw.items():
            part_map = self.embedding_thickness_map.get(group_name)
            if part_map is not None:
                # Embedding groups: normalize against the full map range
                all_values = list(part_map.values())
                min_v, max_v = min(all_values), max(all_values)
                for conn_id, raw_v in entries:
                    if max_v == min_v:
                        mult = (emb_min_t + emb_max_t) / 2
                    else:
                        t = (raw_v - min_v) / (max_v - min_v)
                        mult = emb_min_t + t * (emb_max_t - emb_min_t)
                    conn_multipliers[conn_id] = mult
                continue
            # Ratio group
            if self.ratio_normalization == "sum":
                # norm_r is already a proportion (sums to 1); scale proportionally
                for conn_id, proportion in entries:
                    conn_multipliers[conn_id] = max(ratio_min_t, proportion * ratio_max_t)
            else:
                values = [v for _, v in entries]
                min_v, max_v = min(values), max(values)
                for conn_id, raw_v in entries:
                    if max_v == min_v:
                        mult = (ratio_min_t + ratio_max_t) / 2
                    else:
                        t = (raw_v - min_v) / (max_v - min_v)
                        mult = ratio_min_t + t * (ratio_max_t - ratio_min_t)
                    conn_multipliers[conn_id] = mult
        for conn in self._connections:
            if conn.id and conn.id in conn_multipliers:
                conn.line_width *= conn_multipliers[conn.id]

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

        if not self.simplified:
            for nid, comp in sorted(self._nodes.items()):
                if comp.node_type in ("input", "bias"):
                    input_layer.add_child(comp)
                    layed_out.add(nid)

        # Attach simplified input/bias markers to the left of their source aggregation.
        if self.simplified:

            def legend_text_for(comp: ComputeNode, cotx: str | None) -> str:
                """Legend text: just the cotx name."""
                return cotx or ""

            marker_groups: dict[int, list[tuple[int, str, str, int, str | None, ComputeNode]]] = (
                defaultdict(list)
            )
            unanchored_markers: list[tuple[int, str, str, int, str | None, ComputeNode]] = []
            for nid, comp in sorted(self._nodes.items()):
                if comp.node_type not in ("input", "bias"):
                    continue
                cotx = self._find_cotx_for_node(nid)
                agg_id = self._find_aggregation_for_cotx(cotx)
                kind_rank = 0 if comp.node_type == "input" else 1
                entry = (
                    kind_rank,
                    (cotx or "").lower(),
                    (comp.node_label or "").lower(),
                    nid,
                    cotx,
                    comp,
                )
                if agg_id is None or agg_id not in self._nodes:
                    unanchored_markers.append(entry)
                    continue
                marker_groups[agg_id].append(entry)

            legend_anchor_x = -50
            for agg_id, entries in marker_groups.items():
                anchor = self._nodes[agg_id]
                entries.sort(key=lambda item: item[:4])
                n_entries = len(entries)
                y_step = 22
                for idx, (_kind, _cotx_key, _label_key, nid, cotx, comp) in enumerate(entries):
                    marker_classes = [
                        sc
                        for sc in comp.style_class
                        if sc not in ("input", "bias", "node-type-input", "node-type-bias")
                    ]
                    label = legend_text_for(comp, cotx)
                    if not label:
                        layed_out.add(nid)
                        continue
                    y = (idx - (n_entries - 1) / 2.0) * y_step
                    legend = Text(
                        text=label,
                        style_class=["legend_text", *marker_classes],
                        vertical_align="middle",
                        align="right",
                        is_overlay=True,
                        attached_to=anchor,
                        attachment_offset=Offset(relative=(-1, 0), absolute=(legend_anchor_x, y)),
                        id=f"legend_{nid}",
                    )
                    input_layer.add_child(legend)
                    layed_out.add(nid)

            unanchored_markers.sort(key=lambda item: item[:4])
            for _kind, _cotx_key, _label_key, nid, cotx, comp in unanchored_markers:
                marker_classes = [
                    sc
                    for sc in comp.style_class
                    if sc not in ("input", "bias", "node-type-input", "node-type-bias")
                ]
                label = legend_text_for(comp, cotx)
                if label:
                    legend = Text(
                        text=label,
                        style_class=["legend_text", *marker_classes],
                        vertical_align="middle",
                        align="right",
                        id=f"legend_{nid}",
                    )
                    input_layer.add_child(legend)
                layed_out.add(nid)

            # Marker color labels below each aggregation node
            for nid, comp in self._nodes.items():
                if comp.node_type != "aggregation":
                    continue
                g_node = self._graph.nodes.get(nid)
                if not g_node:
                    continue
                cotx = g_node.extra.get("cotx_group")
                if not cotx:
                    continue
                marker = self._cotx_marker_map.get(cotx)
                if not marker:
                    continue
                label = marker
                if self.show_ratios:
                    norm_r = self._get_marker_ratio(cotx)
                    if norm_r is not None:
                        label = f"{label} {self._format_ratio_label(norm_r)}"
                input_layer.add_child(
                    Text(
                        text=label,
                        style_class=["marker_color_label", marker.upper()],
                        is_overlay=True,
                        attached_to=comp,
                        id=f"marker_color_{nid}",
                    )
                )
        if input_layer.children:
            layers.append(input_layer)

        # ERN layers
        ern_in_nodes = [e for e in ern_ids if e in self._nodes]
        actual_ern_layers = list(self._graph.topological_order(ern_in_nodes))
        target_ern_count = (
            self.layout_spec.max_ern_layers
            if self.layout_spec and self.layout_spec.max_ern_layers
            else len(actual_ern_layers)
        )

        for i in range(target_ern_count):
            ern_layer = actual_ern_layers[i] if i < len(actual_ern_layers) else []
            slot_order = self.layout_spec.get_slot_order_for_layer(i) if self.layout_spec else None
            if not ern_layer and not slot_order:
                continue

            lc = Container(style_class=["main_layer", f"main_layer_{i}", "layer", f"ern_{i}"])
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

            if ern_layer and slot_order:
                ern_by_type: dict[str, int] = {}
                for eid in ern_layer:
                    if eid not in self._nodes:
                        continue
                    node = self._graph.nodes.get(eid)
                    if node:
                        ern_name = node.extra.get("seq_name", "").split("::")[-1].split("#")[0]
                        ern_by_type[ern_name] = eid

                for slot_name in slot_order:
                    if slot_name in ern_by_type:
                        eid = ern_by_type[slot_name]
                        ern = self._nodes[eid]
                        lc.add_child(ern)
                        layed_out.add(eid)
                        for uid in dep_map.get(eid, []):
                            if uid in self._nodes and uid not in ern_ids:
                                up = self._nodes[uid]
                                attach = (
                                    ern._tl_node
                                    if isinstance(up, TranslationNode)
                                    else (
                                        ern._tx_node if isinstance(up, TranscriptionNode) else None
                                    )
                                )
                                if attach:
                                    up.attached_to, up.show = attach, False
                                    lc.add_child(up)
                                    layed_out.add(uid)
                    else:
                        lc.add_child(
                            Container(
                                style_class=["ern_spacer"],
                                show=False,
                            )
                        )
            elif ern_layer:
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
            else:
                # Empty column spacer for padding to max_ern_layers
                for _ in slot_order or [""]:
                    lc.add_child(
                        Container(
                            style_class=["ern_spacer"],
                            show=False,
                        )
                    )

            has_real = any(not isinstance(c, Text) for c in lc.children)
            if has_real:
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

        # Apply layout_spec constraints to layer containers
        if self.layout_spec:
            for layer in layers:
                layer_key = self._layer_key(layer)
                if self.layout_spec.column_widths and layer_key in self.layout_spec.column_widths:
                    layer.min_dimensions = Size(
                        width=self.layout_spec.column_widths[layer_key],
                        height=layer.min_dimensions.height,
                    )
                if self.layout_spec.layer_min_height is not None:
                    layer.min_dimensions = Size(
                        width=layer.min_dimensions.width,
                        height=max(layer.min_dimensions.height, self.layout_spec.layer_min_height),
                    )

        return layers

    @staticmethod
    def _layer_key(layer: Container) -> str:
        """Derive a stable key for a layer container from its style classes."""
        classes = layer.style_class
        if "input_layer" in classes:
            return "input_layer"
        if "output_layer" in classes:
            return "output_layer"
        for cls in classes:
            if cls.startswith("ern_"):
                return cls
        if layer.id:
            return layer.id
        return "unknown"

    def optimize_input_order(self, renderer) -> None:
        """Reorder input-column groups to minimize outgoing-wire crossings."""
        input_layer = next(
            (c for c in self.children if "input_layer" in getattr(c, "style_class", [])),
            None,
        )
        if input_layer is None:
            return
        aggs = [c for c in input_layer.children if isinstance(c, AggregationNode)]
        if len(aggs) < 2:
            return

        self.measure_and_layout(renderer)
        idx_of = {id(a): k for k, a in enumerate(aggs)}

        def center_y(comp: Component) -> float:
            b = comp.get_world_bounds()
            return (b[1] + b[3]) / 2 if b else 0.0

        def group_of(comp: Component | None) -> int | None:
            while comp is not None:
                k = idx_of.get(id(comp))
                if k is not None:
                    return k
                comp = comp.parent
            return None

        targets: list[list[float]] = [[] for _ in aggs]
        for conn in self._connections:
            k = group_of(
                conn.start_component if isinstance(conn.start_component, Component) else None
            )
            if k is not None and isinstance(conn.end_component, Component):
                targets[k].append(center_y(conn.end_component))

        perm = min_crossing_permutation([center_y(a) for a in aggs], targets)
        if perm == list(range(len(aggs))):
            return
        ordered = iter(aggs[i] for i in perm)
        input_layer.children = [
            next(ordered) if isinstance(c, AggregationNode) else c for c in input_layer.children
        ]

    def relax_free_node_y(self, sweeps: int = 4) -> None:
        """Median-relax y of free Tx/Tl nodes (post-layout) to straighten edges."""
        owners = {id(c): c for c in self._nodes.values()}

        def owner_of(comp: Component | None) -> Component | None:
            while comp is not None:
                if id(comp) in owners:
                    return comp
                comp = comp.parent
            return None

        idx: dict[int, int] = {}
        comps: list[Component] = []
        cx: list[float] = []
        cy: list[float] = []
        ch: list[float] = []

        def reg(comp: Component) -> int | None:
            k = idx.get(id(comp))
            if k is None:
                b = comp.get_world_bounds()
                if b is None:
                    return None
                k = len(comps)
                idx[id(comp)] = k
                comps.append(comp)
                cx.append((b[0] + b[2]) / 2)
                cy.append((b[1] + b[3]) / 2)
                ch.append(b[3] - b[1])
            return k

        pairs: list[tuple[int, int]] = []
        for conn in self._connections:
            a = owner_of(
                conn.start_component if isinstance(conn.start_component, Component) else None
            )
            b = owner_of(conn.end_component if isinstance(conn.end_component, Component) else None)
            if a is None or b is None or a is b:
                continue
            ia, ib = reg(a), reg(b)
            if ia is not None and ib is not None:
                pairs.append((ia, ib))

        movable = [
            isinstance(c, (TranscriptionNode, TranslationNode)) and c.attached_to is None
            for c in comps
        ]
        if not any(movable):
            return

        neighbors: list[list[int]] = [[] for _ in comps]
        for ia, ib in pairs:
            neighbors[ia].append(ib)
            neighbors[ib].append(ia)

        gap = (median(ch) if ch else 0.0) * 1.25
        newy = relax_y(cy, neighbors, movable, [round(x) for x in cx], min_gap=gap, sweeps=sweeps)

        for i, c in enumerate(comps):
            if not movable[i] or abs(newy[i] - cy[i]) < 1e-6:
                continue
            yscale = c.parent.compute_world_matrix()[1][1] if c.parent else 1.0
            dy = (newy[i] - cy[i]) / yscale if yscale else (newy[i] - cy[i])
            c.offset = Offset(absolute=(c.offset.absolute[0], c.offset.absolute[1] + dy))


def render_diagram_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    simplified: bool = True,
    disabled_tu_ids: set[str] | None = None,
    style_overrides: dict | None = None,
    title: str | None = None,
    show_ratios: bool = False,
    ratio_normalization: Literal["min", "sum"] = "sum",
    variable_thickness: bool = False,
    show_edge_parts: bool = False,
    thickness_range: tuple[float, float] = (0.5, 4.0),
    layout_spec: LayoutSpec | None = None,
    canvas_xlim: tuple[float, float] | None = None,
    canvas_ylim: tuple[float, float] | None = None,
    aspect: str = "equal",
    **_kwargs,
):
    """Render a network compute diagram to an existing matplotlib axes.

    ``aspect="equal"`` (default) preserves the diagram's data aspect ratio
    inside the cell - produces whitespace bands when cell aspect ≠ content
    aspect. ``aspect="auto"`` stretches the diagram to fill the cell with
    no margins (boxes/text become slightly anisotropic; usually acceptable
    for schematic content).
    """
    import jeanplot
    from jeanplot import MatplotlibRenderer, jstyle, load_default_theme
    from jeanplot.core.style_engine import merge_jstyle_rules

    load_default_theme()
    if style_overrides:
        # jstyle.update *replaces* the cascade, so overrides must be merged onto
        # the default rules (just loaded into _DEFAULT_THEME_CACHE) or the layout
        # rules are wiped and the diagram collapses.
        jstyle.update(merge_jstyle_rules(jeanplot._DEFAULT_THEME_CACHE, style_overrides))

    diagram = NetworkDiagram(
        network=network,
        simplified=simplified,
        disabled_tu_ids=disabled_tu_ids or set(),
        show_ratios=show_ratios,
        ratio_normalization=ratio_normalization,
        variable_thickness=variable_thickness,
        show_edge_parts=show_edge_parts,
        thickness_range=thickness_range,
        layout_spec=layout_spec,
    )
    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)

    ax.set_aspect(aspect)
    ax.axis("off")
    renderer = MatplotlibRenderer()
    renderer.create_context(ax=ax)
    diagram.optimize_input_order(renderer)
    # pre_render callbacks run post-measure so they survive jstyle re-application:
    # line widths, then free-node y straightening (reads measured positions).
    renderer.pre_render_callbacks.append(lambda _ax: diagram.apply_variable_line_widths())
    renderer.pre_render_callbacks.append(lambda _ax: diagram.relax_free_node_y())
    # padding=0 fills the cell; set_aspect=False keeps our `aspect` (default 0.1 padding
    # was the margin around the diagram).
    renderer.render_component(
        ax, root, adjust_lims=True, adjust_lims_padding=0.0, adjust_lims_set_aspect=False
    )

    from biocomptools.toollib.figuremakers._jeanplot_canvas import apply_canvas

    apply_canvas(ax, canvas_xlim, canvas_ylim)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
