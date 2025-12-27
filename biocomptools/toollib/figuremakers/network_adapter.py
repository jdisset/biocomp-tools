"""Adapter for biocomp Network to visualization data structures."""

from typing import Any
from pydantic import BaseModel
from collections import defaultdict
import numpy as np


class TUInfo(BaseModel):
    tu_id: str
    tu_name: str
    cotx_marker: str | None = None
    is_marker: bool = False
    plasmid_name: str = ""
    source_id: str = ""
    in_l2: bool = False
    position_in_plasmid: int = 0
    number_of_tu_in_plasmid: int = 1
    aggregation_ratio: float | None = None
    aggregation_node_id: int | None = None
    in_aggregation: bool = False
    aggregation_ratio_norm: float = 1.0
    marker_ratio: float | None = None
    aggregation_ratio_label: str = ""
    marker_in_l2: bool = False
    parts: list = []
    cotx_name: str | None = None
    cotx_group: str | None = None


class PartInfo(BaseModel):
    name: str
    category: str


def get_tu_informations(network: Any) -> dict[str, TUInfo]:
    """Extract TU information from GraphState-based Network."""
    if network.compute_graph is None:
        return {}

    graph = network.compute_graph
    net_info = network.generate_network_info()
    markers = set(net_info.get("markers", []))
    all_parts = net_info.get("all_parts", {})

    from biocomp.library import LibraryContext
    lib = LibraryContext.get_library()

    cat_order = {
        "insulator": 0, "promoter": 1, "uORF_group": 2, "ERN_recog_site_5p": 3,
        "ERN": 4, "CDS": 5, "fluo_marker": 5, "terminator": 6,
    }

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
            content = ([p.name if hasattr(p, "name") else str(p) for p in dna_edges[0].content]
                       if dna_edges else [])

            is_marker = any(item in markers for item in content)
            marker = next((item for item in content if item in markers), None)

            parts = []
            for ap_key, ap_parts in all_parts.items():
                base = ap_key.rsplit("_", 1)[0] if ap_key.rsplit("_", 1)[-1].isdigit() else ap_key
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

    return tus
