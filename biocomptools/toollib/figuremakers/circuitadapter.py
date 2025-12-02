"""Adapter for biocomp Network to circuit visualization data"""

from typing import Any, Optional
from pydantic import BaseModel
from collections import defaultdict
import numpy as np


class TUInfo(BaseModel):
    tu_id: str
    tu_name: str
    cotx_marker: Optional[str] = None
    is_marker: bool = False
    plasmid_name: str = ""
    source_id: str = ""
    in_l2: bool = False
    position_in_plasmid: int = 0
    number_of_tu_in_plasmid: int = 1
    aggregation_ratio: Optional[float] = None
    aggregation_node_id: Optional[int] = None
    in_aggregation: bool = False
    aggregation_ratio_norm: float = 1.0
    marker_ratio: Optional[float] = None
    aggregation_ratio_label: str = ""
    marker_in_l2: bool = False
    parts: list = []
    cotx_name: Optional[str] = None
    cotx_group: Optional[str] = None


class PartInfo(BaseModel):
    name: str
    category: str


class Interaction(BaseModel):
    src_tu_id: str
    src_part_name: str
    tgt_tu_id: str
    tgt_part_name: str
    type: str = "ERN"


def get_tu_informations(network: Any) -> dict[str, TUInfo]:
    """Extract TU information from GraphState-based Network"""
    if network.compute_graph is None:
        return {}

    graph = network.compute_graph
    net_info = network.generate_network_info()
    markers = set(net_info.get("markers", []))
    all_parts = net_info.get("all_parts", {})

    from biocomp.library import LibraryContext
    lib = LibraryContext.get_library()

    cat_order = {"insulator": 0, "promoter": 1, "uORF_group": 2, "ERN_recog_site_5p": 3,
                 "ERN": 4, "CDS": 5, "fluo_marker": 5, "terminator": 6}

    tus = {}
    aggr_to_tus = defaultdict(list)

    for node in graph.get_nodes_by_type("source"):
        source_id = node.extra.get("source_id", "")
        cotx_group = node.extra.get("cotx_group", "cotx_1")
        name = node.extra.get("name", "")

        outgoing = list(graph.get_outgoing_edges(node.node_id))
        n_outputs = len(set(e.from_output_slot for e in outgoing))

        for slot in range(max(1, n_outputs)):
            tu_id = f"{name}_{cotx_group}" if name else f"tu_{node.node_id}_{slot}"
            dna_edges = [e for e in outgoing if e.content_type == "DNA" and e.from_output_slot == slot]
            content = [p.name if hasattr(p, 'name') else str(p) for p in dna_edges[0].content] if dna_edges else []

            is_marker = any(item in markers for item in content)
            marker = next((item for item in content if item in markers), None)

            parts = []
            for ap_key, ap_parts in all_parts.items():
                base = ap_key.rsplit('_', 1)[0] if ap_key.rsplit('_', 1)[-1].isdigit() else ap_key
                if base == name or ap_key.startswith(name):
                    parts = [PartInfo(name=n, category=c) for n, c in ap_parts.items()]
                    break

            if not parts and lib:
                for pn in content:
                    if pn in lib.parts.index:
                        parts.append(PartInfo(name=pn, category=lib.parts.loc[pn].category))

            if lib and dna_edges:
                existing = {p.name for p in parts}
                for emb_parts in (dna_edges[0].content_embedding_names or {}).values():
                    for pn in emb_parts:
                        if pn and pn != "00_empty_tc" and pn not in existing and pn in lib.parts.index:
                            parts.append(PartInfo(name=pn, category=lib.parts.loc[pn].category))

            parts.sort(key=lambda p: cat_order.get(p.category, 99))

            tus[tu_id] = TUInfo(
                tu_id=tu_id, tu_name=name or tu_id, cotx_marker=marker, is_marker=is_marker,
                plasmid_name=source_id, source_id=source_id, in_l2=n_outputs > 1,
                position_in_plasmid=slot, number_of_tu_in_plasmid=n_outputs,
                parts=[p.model_dump() for p in parts], cotx_group=cotx_group,
            )

            for edge in graph.get_incoming_edges(node.node_id):
                upstream = graph.nodes.get(edge.source_id)
                if upstream and upstream.node_type == "aggregation":
                    tus[tu_id].aggregation_node_id = upstream.node_id
                    tus[tu_id].in_aggregation = True
                    aggr_to_tus[upstream.node_id].append(tu_id)
                    break

    for agg_id, tu_ids in aggr_to_tus.items():
        agg = graph.nodes.get(agg_id)
        if not agg:
            continue

        cotx_name = agg.extra.get("name")
        if cotx_name:
            for t in tu_ids:
                if t in tus:
                    tus[t].cotx_name = cotx_name

        raw_ratios, members = agg.extra.get("ratios", []), agg.extra.get("members", [])
        if not raw_ratios or not members:
            continue

        src_ratio = dict(zip(members, raw_ratios)) if len(members) == len(raw_ratios) else {}
        for t in tu_ids:
            if t in tus and (r := src_ratio.get(tus[t].source_id)) is not None:
                tus[t].aggregation_ratio = r

        if src_ratio:
            ratios = np.array(list(src_ratio.values()))
            normed = np.round(ratios / max(ratios.min(), 1e-6), 2)
            label = ":".join(str(int(round(r))) if np.isclose(r, round(r)) else str(r) for r in normed)
            for t in tu_ids:
                if t in tus:
                    tus[t].aggregation_ratio_label = label

    for tu_ids in aggr_to_tus.values():
        if marker_tu := next((tus[t] for t in tu_ids if t in tus and tus[t].is_marker), None):
            for t in tu_ids:
                if t in tus:
                    tus[t].cotx_marker = marker_tu.cotx_marker

    return tus


def get_interactions(network: Any) -> list[Interaction]:
    """Extract ERN interactions from GraphState-based Network"""
    if network.compute_graph is None:
        return []

    graph = network.compute_graph
    interactions = []

    def find_sources(edge, target: str | None = None, visited: set | None = None) -> list[str]:
        if visited is None:
            visited = set()
        if edge.source_id in visited:
            return []
        visited.add(edge.source_id)

        node = graph.nodes.get(edge.source_id)
        if not node:
            return []
        if node.node_type == "source":
            name, cotx = node.extra.get("name", ""), node.extra.get("cotx_group", "cotx_1")
            return [f"{name}_{cotx}" if name else f"tu_{node.node_id}"]

        results = []
        if node.node_type == "sequestron_ERN":
            for e in graph.get_incoming_edges(node.node_id):
                names = [p.name for p in e.content] if e.content else []
                if target and target in names:
                    results.extend(find_sources(e, target, visited))
            if not results:
                for e in graph.get_incoming_edges(node.node_id):
                    results.extend(find_sources(e, target, visited))
        else:
            for e in graph.get_incoming_edges(node.node_id):
                results.extend(find_sources(e, target, visited))
        return results

    for ern in graph.get_nodes_by_type("sequestron_ERN"):
        incoming = list(graph.get_incoming_edges(ern.node_id))
        pos_edges = [e for e in incoming if e.to_input_slot == 0]
        neg_edges = [e for e in incoming if e.to_input_slot == 1]

        if not pos_edges or not neg_edges:
            continue

        pos_edge, neg_edge = pos_edges[0], neg_edges[0]
        ern_part = pos_edge.content[0].name if pos_edge.content else None
        rec_parts = [p.name for p in neg_edge.content] if neg_edge.content else []
        rec_part = next((p for p in rec_parts if ern_part and ern_part in p), None)

        if not ern_part or not rec_part:
            continue

        for src in find_sources(pos_edge, ern_part):
            for tgt in find_sources(neg_edge, rec_part):
                interactions.append(Interaction(
                    src_tu_id=src, src_part_name=ern_part,
                    tgt_tu_id=tgt, tgt_part_name=rec_part, type="ERN",
                ))

    return interactions


def get_source_id(tu_id: str, tu_infos: dict[str, TUInfo]) -> str:
    info = tu_infos.get(tu_id)
    if not info:
        return "unknown_source"
    return f"plasmid_{info.plasmid_name}" if info.in_l2 else f"source_{info.cotx_marker}_{info.aggregation_node_id}_{info.aggregation_ratio_label}"
