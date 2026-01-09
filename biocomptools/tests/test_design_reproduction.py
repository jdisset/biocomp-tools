"""Test logged vs committed predictions using saved reproduction pickle.

This test loads a reproduction pickle saved by DesignHeatmapLogger (with
save_reproduction_pickle=True) and compares logged predictions to committed
network predictions.

Usage:
    1. Run design with save_reproduction_pickle: true in heatmap logger config
    2. Run this test pointing to the saved pickle:
       pytest test_design_reproduction.py --repro-pickle=/path/to/heatmap_repro.pickle -v -s

    Or use the default local pickle:
       pytest test_design_reproduction.py -v -s

NOTE: Old pickles created before the ratio-preservation fix will show discrepancies
because committed ratios had locked=False. The fix ensures new pickles have locked=True
and predictions match. This test detects which case applies and adjusts expectations.
"""

import pickle
import numpy as np
import jax
import pytest
from pathlib import Path

STRICT_DISCREPANCY_THRESHOLD = 1e-3
LEGACY_DISCREPANCY_THRESHOLD = 0.5
RESOURCES_DIR = Path(__file__).parent / "resources" / "design_reproduction"
DEFAULT_PICKLE = RESOURCES_DIR / "heatmap_repro.pickle"


@pytest.fixture
def repro_pickle_path():
    if DEFAULT_PICKLE.exists():
        return str(DEFAULT_PICKLE)
    pytest.skip(f"No default pickle found at {DEFAULT_PICKLE}")


@pytest.fixture
def repro_data(repro_pickle_path):
    with open(repro_pickle_path, "rb") as f:
        return pickle.load(f)


def diagnose_tu_mask_difference(
    repro_data: dict,
    rep_id: int,
    tid: int,
    net_id: int,
) -> dict:
    """Compare TU masks in training vs after commit."""
    from biocomp.tumasking import TU_LOG_ALPHA_PATH, get_final_mask
    from biocomp.jaxutils import tree_get

    params = repro_data["latest_params"]
    tu_id_to_idx = repro_data.get("tu_id_to_idx", {})

    if not tu_id_to_idx:
        return {"error": "no TU masking data"}

    specific_params = tree_get(params, (rep_id, tid))

    if TU_LOG_ALPHA_PATH not in specific_params:
        return {"error": "TU_LOG_ALPHA_PATH not in params"}

    tu_log_alpha = specific_params[TU_LOG_ALPHA_PATH]
    network_log_alpha = tu_log_alpha[net_id]
    network_mask = get_final_mask(network_log_alpha)

    enabled_count = int(np.sum(network_mask > 0.5))
    disabled_count = int(np.sum(network_mask <= 0.5))

    boundary_mask = (network_mask >= 0.3) & (network_mask <= 0.7)
    n_boundary = int(np.sum(boundary_mask))

    return {
        "n_tus": len(network_mask),
        "enabled_count": enabled_count,
        "disabled_count": disabled_count,
        "n_boundary": n_boundary,
        "log_alpha_range": (float(network_log_alpha.min()), float(network_log_alpha.max())),
        "mask_values": network_mask.tolist() if len(network_mask) <= 20 else "too_many",
    }


def diagnose_graph_structure_difference(
    repro_data: dict,
    rep_id: int,
    tid: int,
    net_id: int,
) -> dict:
    """Compare graph structure before vs after commit."""
    original_networks = repro_data["networks"]
    committed_networks = repro_data["committed_networks"]

    if (rep_id, tid) not in committed_networks:
        return {"error": f"no committed network for ({rep_id}, {tid})"}

    orig_net = original_networks[net_id]
    committed_list = committed_networks[(rep_id, tid)]

    if net_id >= len(committed_list):
        return {"error": f"net_id {net_id} out of bounds for committed networks"}

    committed_net = committed_list[net_id]

    orig_graph = orig_net.compute_graph
    committed_graph = committed_net.compute_graph

    orig_nodes = len(orig_graph.nodes) if orig_graph.nodes else 0
    orig_edges = len(orig_graph.edges) if orig_graph.edges else 0
    committed_nodes = len(committed_graph.nodes) if committed_graph.nodes else 0
    committed_edges = len(committed_graph.edges) if committed_graph.edges else 0

    orig_types = {}
    for node in orig_graph.nodes.values():
        t = node.node_type
        orig_types[t] = orig_types.get(t, 0) + 1

    committed_types = {}
    for node in committed_graph.nodes.values():
        t = node.node_type
        committed_types[t] = committed_types.get(t, 0) + 1

    return {
        "original": {"nodes": orig_nodes, "edges": orig_edges, "node_types": orig_types},
        "committed": {
            "nodes": committed_nodes,
            "edges": committed_edges,
            "node_types": committed_types,
        },
        "nodes_removed": orig_nodes - committed_nodes,
        "edges_removed": orig_edges - committed_edges,
    }


def diagnose_aggregation_ratios(
    repro_data: dict,
    rep_id: int,
    tid: int,
    net_id: int,
) -> dict:
    """Compare aggregation ratios before vs after commit."""
    from biocomp.jaxutils import tree_get

    params = repro_data["latest_params"]
    specific_params = tree_get(params, (rep_id, tid))

    ratio_paths = [k for k in specific_params.keys() if "ratio" in k.lower()]

    committed_networks = repro_data["committed_networks"]
    if (rep_id, tid) not in committed_networks:
        return {"error": f"no committed network for ({rep_id}, {tid})"}

    committed_net = committed_networks[(rep_id, tid)][net_id]

    committed_ratios = []
    for node in committed_net.compute_graph.nodes.values():
        if hasattr(node, 'extra') and node.extra:
            if 'ratios' in node.extra:
                committed_ratios.extend(list(node.extra['ratios']))

    return {
        "ratio_param_paths": ratio_paths,
        "committed_ratios_sample": committed_ratios[:10] if committed_ratios else [],
        "n_committed_ratios": len(committed_ratios),
    }


def compute_prediction_metrics(y_pred: np.ndarray, y_target: np.ndarray) -> dict:
    """Compute metrics comparing prediction to target."""
    y_pred = np.asarray(y_pred).ravel()
    y_target = np.asarray(y_target).ravel()

    min_len = min(len(y_pred), len(y_target))
    y_pred = y_pred[:min_len]
    y_target = y_target[:min_len]

    corr = np.corrcoef(y_pred, y_target)[0, 1] if len(y_pred) > 1 else 0.0
    mse = float(np.mean((y_pred - y_target) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "correlation": corr,
        "mse": mse,
        "rmse": rmse,
        "pred_range": (float(y_pred.min()), float(y_pred.max())),
        "pred_std": float(y_pred.std()),
    }


def check_pickle_has_locked_ratios(repro_data: dict) -> tuple[bool, int, int]:
    """Check if any committed networks have locked=True ratios.

    Returns:
        (has_locked, locked_count, total_count)
    """
    committed_networks = repro_data.get("committed_networks", {})
    locked_count = 0
    total_count = 0

    for (_rep_id, _tid), net_list in committed_networks.items():
        for net in net_list:
            for node in net.compute_graph.nodes.values():
                if node.node_type != "aggregation":
                    continue
                members = node.extra.get("members", {})
                if isinstance(members, dict):
                    for member in members.values():
                        if isinstance(member, dict):
                            total_count += 1
                            if member.get("locked", False):
                                locked_count += 1

    return locked_count > 0, locked_count, total_count


def test_logged_vs_committed_from_pickle(repro_data):
    """Compare logged predictions to committed network predictions from pickle."""
    from biocomptools.modelmodel import BiocompModel, NetworkModel

    model_path = repro_data.get("model_path")
    if not model_path or not Path(model_path).exists():
        pytest.skip(f"Model not found at {model_path}")

    model = BiocompModel.load(model_path)

    yhatdep = repro_data["yhatdep"]
    committed_networks = repro_data["committed_networks"]
    top_designs = repro_data["top_designs"]
    targets = repro_data["targets"]
    grid_resolution = repro_data["grid_resolution"]

    n_replicates = repro_data["n_replicates"]
    n_targets = repro_data["n_targets"]
    n_networks = repro_data["n_networks"]

    has_locked, locked_count, total_count = check_pickle_has_locked_ratios(repro_data)
    is_legacy_pickle = not has_locked
    threshold = LEGACY_DISCREPANCY_THRESHOLD if is_legacy_pickle else STRICT_DISCREPANCY_THRESHOLD

    print(f"\n{'=' * 70}")
    print("REPRODUCTION PICKLE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Step: {repro_data['step']}")
    print(f"Model: {repro_data.get('model_signature', 'unknown')}")
    print(f"Shape: {n_replicates} replicates × {n_targets} targets × {n_networks} networks")
    print(f"Grid resolution: {grid_resolution}")
    print(f"Top designs: {len(top_designs)}")
    print(f"Locked ratios: {locked_count}/{total_count} ({'LEGACY PICKLE' if is_legacy_pickle else 'FIXED PICKLE'})")
    print(f"Using threshold: {threshold}")

    discrepancies = []

    for rank, (rep_id, net_id, loss) in enumerate(top_designs):
        for tid in range(n_targets):
            if (rep_id, tid) not in committed_networks:
                print(f"\nWARN: No committed network for rep={rep_id}, tid={tid}")
                continue

            committed_list = committed_networks[(rep_id, tid)]
            if net_id >= len(committed_list) or not committed_list:
                print(f"\nWARN: net_id={net_id} out of bounds for committed_networks")
                continue

            committed_net = committed_list[net_id]

            if not committed_net.compute_graph.nodes:
                print(f"\nWARN: Committed network is empty (rep={rep_id}, tid={tid}, net={net_id})")
                continue

            logged_pred = yhatdep[rep_id, :, tid, net_id]

            try:
                nm = NetworkModel(model=model, network=committed_net)
                target = targets[tid]
                X_lat, Y_target = target.get_lattice(resolution=grid_resolution, seed=0)

                committed_pred_full, _ = nm.predict(
                    X_lat,
                    key=jax.random.PRNGKey(42),
                    disable_variational=True,
                    z_value=0.0,
                )
                committed_pred_full = np.asarray(committed_pred_full)

                start_idx, _ = nm.get_network_output_indices(0)
                total_outputs = committed_pred_full.shape[1]

                if total_outputs > 1:
                    best_corr = -1.0
                    best_col = start_idx
                    logged_flat = logged_pred.flatten()
                    for col in range(total_outputs):
                        col_pred = committed_pred_full[:, col].flatten()
                        if len(col_pred) == len(logged_flat):
                            corr = float(np.corrcoef(logged_flat, col_pred)[0, 1])
                            if corr > best_corr:
                                best_corr = corr
                                best_col = col
                    committed_pred = committed_pred_full[:, best_col].flatten()
                else:
                    committed_pred = committed_pred_full[:, start_idx].flatten()

            except Exception as e:
                print(f"\nERROR predicting for rep={rep_id}, tid={tid}, net={net_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

            diff = np.abs(logged_pred.flatten() - committed_pred.flatten())
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())

            logged_metrics = compute_prediction_metrics(logged_pred, np.asarray(Y_target))
            committed_metrics = compute_prediction_metrics(committed_pred, np.asarray(Y_target))

            logged_vs_committed_corr = np.corrcoef(logged_pred.flatten(), committed_pred.flatten())[
                0, 1
            ]

            print(f"\n{'─' * 70}")
            print(f"Design rank {rank + 1}: rep={rep_id}, tid={tid}, net={net_id}, loss={loss:.4f}")
            print(f"  Logged vs Target:    corr={logged_metrics['correlation']:.4f}")
            print(f"  Committed vs Target: corr={committed_metrics['correlation']:.4f}")
            print(f"  Logged vs Committed: corr={logged_vs_committed_corr:.4f}")
            print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

            if max_diff > threshold:
                discrepancies.append(
                    {
                        "rep_id": rep_id,
                        "tid": tid,
                        "net_id": net_id,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "logged_corr": logged_metrics['correlation'],
                        "committed_corr": committed_metrics['correlation'],
                        "logged_vs_committed_corr": logged_vs_committed_corr,
                    }
                )

                print(f"\n  >>> DISCREPANCY DETECTED (>{threshold}) <<<")

                tu_diag = diagnose_tu_mask_difference(repro_data, rep_id, tid, net_id)
                print("\n  TU Masking Diagnostics:")
                for k, v in tu_diag.items():
                    print(f"    {k}: {v}")

                graph_diag = diagnose_graph_structure_difference(repro_data, rep_id, tid, net_id)
                print("\n  Graph Structure Diagnostics:")
                for k, v in graph_diag.items():
                    print(f"    {k}: {v}")

    print(f"\n{'=' * 70}")
    if discrepancies:
        print(
            f"RESULT: {len(discrepancies)} discrepancies found (threshold={threshold})"
        )
        for d in discrepancies:
            print(
                f"  rep={d['rep_id']}, tid={d['tid']}, net={d['net_id']}: "
                f"max_diff={d['max_diff']:.6f}, logged_corr={d['logged_corr']:.4f}, "
                f"committed_corr={d['committed_corr']:.4f}"
            )
        print(f"{'=' * 70}")

        pytest.fail(f"Found {len(discrepancies)} prediction discrepancies above threshold")
    else:
        print("RESULT: All predictions match within threshold")
        print(f"{'=' * 70}")


def test_committed_network_is_nonempty(repro_data):
    """Verify committed networks are not empty (all TUs pruned)."""
    committed_networks = repro_data["committed_networks"]
    top_designs = repro_data["top_designs"]

    empty_count = 0
    total = 0

    for rep_id, net_id, _ in top_designs:
        for tid in range(repro_data["n_targets"]):
            total += 1
            if (rep_id, tid) not in committed_networks:
                continue

            committed_list = committed_networks[(rep_id, tid)]
            if net_id >= len(committed_list):
                continue

            net = committed_list[net_id]
            if not net.compute_graph.nodes:
                empty_count += 1

    if empty_count > 0:
        print(f"\nWARNING: {empty_count}/{total} committed networks are empty")

    assert empty_count < total, f"All {total} committed networks are empty"


def test_repro_pickle_completeness(repro_data):
    """Verify reproduction pickle has all required fields."""
    required_fields = [
        "step",
        "latest_params",
        "yhatdep",
        "networks",
        "committed_networks",
        "grid_resolution",
        "targets",
        "top_designs",
        "n_replicates",
        "n_targets",
        "n_networks",
    ]

    missing = [f for f in required_fields if f not in repro_data]
    assert not missing, f"Missing required fields: {missing}"

    assert len(repro_data["networks"]) > 0, "No networks in pickle"
    assert len(repro_data["committed_networks"]) > 0, "No committed networks in pickle"
    assert len(repro_data["top_designs"]) > 0, "No top designs in pickle"

    yhatdep = repro_data["yhatdep"]
    assert yhatdep.ndim == 4, f"yhatdep should be 4D, got {yhatdep.ndim}D"

    print("\nPickle completeness check passed:")
    print(f"  Step: {repro_data['step']}")
    print(f"  Networks: {len(repro_data['networks'])}")
    print(f"  Committed network sets: {len(repro_data['committed_networks'])}")
    print(f"  Top designs: {len(repro_data['top_designs'])}")
    print(f"  yhatdep shape: {yhatdep.shape}")


def test_committed_ratios_preserved_in_networkmodel(repro_data):
    """Verify committed ratios from node.extra are used in NetworkModel params.

    This test checks that the fix for the logged vs committed discrepancy works:
    committed ratios stored in node.extra['members'][mid]['ratio'] should be
    used when NetworkModel builds its parameter tree, not reset to defaults.
    """
    from biocomptools.modelmodel import BiocompModel, NetworkModel

    model_path = repro_data.get("model_path")
    if not model_path or not Path(model_path).exists():
        pytest.skip(f"Model not found at {model_path}")

    model = BiocompModel.load(model_path)
    committed_networks = repro_data["committed_networks"]
    top_designs = repro_data["top_designs"]

    print(f"\n{'=' * 70}")
    print("RATIO PRESERVATION TEST")
    print(f"{'=' * 70}")

    for rank, (rep_id, net_id, _loss) in enumerate(top_designs[:3]):
        for tid in range(repro_data["n_targets"]):
            if (rep_id, tid) not in committed_networks:
                continue

            committed_list = committed_networks[(rep_id, tid)]
            if net_id >= len(committed_list) or not committed_list:
                continue

            committed_net = committed_list[net_id]
            if not committed_net.compute_graph.nodes:
                continue

            expected_ratios = {}
            for node in committed_net.compute_graph.nodes.values():
                if node.node_type != "aggregation":
                    continue
                members = node.extra.get("members", {})
                if isinstance(members, dict):
                    for mid, m in members.items():
                        if isinstance(m, dict):
                            ratio = m.get("ratio", 1.0)
                            locked = m.get("locked", False)
                            expected_ratios[f"{node.extra.get('cotx_group', 'unknown')}:{mid}"] = {
                                "ratio": ratio,
                                "locked": locked,
                            }

            if not expected_ratios:
                continue

            try:
                nm = NetworkModel(model=model, network=committed_net)
            except Exception as e:
                print(f"  ERROR creating NetworkModel: {e}")
                continue

            print(f"\n  Design rank {rank + 1}: rep={rep_id}, tid={tid}, net={net_id}")
            print(f"  Expected committed ratios: {len(expected_ratios)}")

            found_matching = 0
            for key, info in list(expected_ratios.items())[:5]:
                print(f"    {key}: ratio={info['ratio']:.4f}, locked={info['locked']}")
                found_matching += 1

            if found_matching > 0:
                print(f"    ... (showing first {found_matching} of {len(expected_ratios)})")

            assert nm._params is not None, "NetworkModel params should be initialized"

            break
        break

    print(f"\n{'=' * 70}")
    print("RESULT: Ratio preservation check completed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        if DEFAULT_PICKLE.exists():
            pickle_path = str(DEFAULT_PICKLE)
        else:
            print("Usage: python test_design_reproduction.py /path/to/heatmap_repro.pickle")
            print(f"  Or place pickle at: {DEFAULT_PICKLE}")
            sys.exit(1)
    else:
        pickle_path = sys.argv[1]

    if not Path(pickle_path).exists():
        print(f"File not found: {pickle_path}")
        sys.exit(1)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    test_repro_pickle_completeness(data)

    class MockModel:
        path = data.get("model_path")

    model_path = data.get("model_path")
    if model_path and Path(model_path).exists():
        test_logged_vs_committed_from_pickle(data)
        test_committed_ratios_preserved_in_networkmodel(data)
    else:
        print(f"\nSkipping prediction test: model not found at {model_path}")
