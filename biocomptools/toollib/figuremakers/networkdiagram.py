"""Network compute diagram figure for biocomp networks using jeanplot primitives."""

from typing import Any
from pydantic import Field, PrivateAttr
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.axes

from jeanplot.core.component import AnchorComponent
from jeanplot.core.connection_label import ConnectionLabel
from jeanplot.core.container import Container
from jeanplot.core.models import BoxStyle, LayoutConstraints, Offset
from jeanplot.core.connector import Connection, OrthogonalCurve, SimpleBezierCurve
from jeanplot.core.svg import LineEndFlat
from jeanplot.core.text import Text
from jeanplot.gene.elements import _format_ratio_multiplier

from biocomp.graphengine import is_inverse_node_type
from biocomp.ratio_schema import get_slot_entries
from biocomptools.toollib.plot import Figure
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

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
    variable_thickness: bool = Field(default=False, description="Scale edge width by ratio or embedding")
    show_edge_parts: bool = Field(default=False, description="Show embedding part names on edges")
    thickness_range: tuple[float, float] = Field(
        default=(0.5, 4.0),
        description="(min, max) multiplier on base line_width for variable-thickness edges",
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
            min_r = min(positive) if positive else 1.0
            for entry, raw_r in zip(entries, raw_ratios, strict=True):
                sid = str(entry["source_id"])
                nid = source_id_to_nid.get(sid)
                if nid is not None:
                    norm_r = raw_r / min_r if min_r > 0 else 1.0
                    ratio_map[nid] = (raw_r, norm_r)
        return ratio_map

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
                        text=_format_ratio_multiplier(norm_r),
                        style_class=["edge_ratio", *base_label_classes],
                        connection=conn_id,
                        font_size=5.0,
                        font_weight="bold",
                        color="#666666",
                    )
                )
                labeled_sources.add(src)

            # Edge part name annotation — filter by target node type
            if self.show_edge_parts and e.content_embedding_names:
                filtered = _filter_embeddings_for_target(
                    e.content_embedding_names, tgt_comp.node_type
                )
                parts_text = _format_edge_parts(filtered)
                if parts_text:
                    embed_classes = [f"embed-{k}" for k in filtered]
                    cat_classes = [
                        f"cat-{EMBEDDING_CATEGORY[k]}"
                        for k in filtered
                        if k in EMBEDDING_CATEGORY
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
        min_t, max_t = self.thickness_range
        conn_multipliers: dict[str, float] = {}
        for group_name, entries in self._thickness_raw.items():
            # For embedding groups, normalize against the full map range
            part_map = self.embedding_thickness_map.get(group_name)
            if part_map is not None:
                all_values = list(part_map.values())
                min_v, max_v = min(all_values), max(all_values)
            else:
                values = [v for _, v in entries]
                min_v, max_v = min(values), max(values)
            for conn_id, raw_v in entries:
                if max_v == min_v:
                    mult = (min_t + max_t) / 2
                else:
                    t = (raw_v - min_v) / (max_v - min_v)
                    mult = min_t + t * (max_t - min_t)
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
                        label = f"{label} {_format_ratio_multiplier(norm_r)}"
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


def render_diagram_to_ax(
    network: Any,
    ax: matplotlib.axes.Axes,
    simplified: bool = True,
    disabled_tu_ids: set[str] | None = None,
    style_overrides: dict | None = None,
    title: str | None = None,
    show_ratios: bool = False,
    variable_thickness: bool = False,
    show_edge_parts: bool = False,
    thickness_range: tuple[float, float] = (0.5, 4.0),
    **_kwargs,
):
    """Render a network compute diagram to an existing matplotlib axes."""
    from jeanplot import MatplotlibRenderer, jstyle, load_default_theme
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
    from jeanplot.core.models import TextHalo
    from dracon import load, resolve_all_lazy
    import importlib.resources

    load_default_theme()

    types = [
        Size,
        BoxStyle,
        LayoutConstraints,
        Offset,
        Shadow,
        TextHalo,
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
        network=network,
        simplified=simplified,
        disabled_tu_ids=disabled_tu_ids or set(),
        show_ratios=show_ratios,
        variable_thickness=variable_thickness,
        show_edge_parts=show_edge_parts,
        thickness_range=thickness_range,
    )
    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)

    ax.set_aspect("equal")
    ax.axis("off")
    renderer = MatplotlibRenderer()
    # Apply variable line widths via pre_render callback so they survive
    # jstyle re-application during measure_and_layout.
    renderer.pre_render_callbacks.append(lambda _ax: diagram.apply_variable_line_widths())
    renderer.render_component(ax, root, adjust_lims=True)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


class NetworkDiagramFigure(Figure):
    """Figure that renders a network compute diagram using jeanplot."""

    network: Any = Field(description="biocomp Network object")
    simplified: bool = True
    disabled_tu_ids: set[str] | None = None
    style_overrides: dict | None = None
    show_ratios: bool = False
    variable_thickness: bool = False
    show_edge_parts: bool = False
    thickness_range: tuple[float, float] = (0.5, 4.0)

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
            show_ratios=self.show_ratios,
            variable_thickness=self.variable_thickness,
            show_edge_parts=self.show_edge_parts,
            thickness_range=self.thickness_range,
        )
        self.figure_spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_spec.output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved network diagram to {self.figure_spec.output_path}")
