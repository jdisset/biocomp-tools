"""Test that committed network predictions match evaluation predictions.

This test verifies that the design optimization output is consistent between:
1. Predictions made during evaluation (using optimized params + TU masks)
2. Predictions made from committed networks (rebuilt from recipe)

The correlation between these should be within 5% of each other.
"""
import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from biocomp.design import DesignManager, DesignConfig
from biocomp.design_targets import SVGTarget
from biocomp.compute import ComputeStack
from biocomp.tumasking import TU_LOG_ALPHA_PATH

pytestmark = pytest.mark.skipif(
    not os.environ.get("BIOCOMP_DESIGNER_MODEL"),
    reason="BIOCOMP_DESIGNER_MODEL not set",
)


@pytest.fixture
def design_setup():
    from biocomptools.modelmodel import BiocompModel
    from biocomp.recipe import Recipe
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    model = BiocompModel.load(os.environ["BIOCOMP_DESIGNER_MODEL"])

    svg_path = os.path.join(
        os.environ.get("BIOCOMP_ROOT", "."),
        "biocomp-jobs/design/targets/MIT_T.svg"
    )
    if not os.path.exists(svg_path):
        pytest.skip("MIT_T.svg not found")

    target = SVGTarget(
        name="test_target",
        path=svg_path,
        lattice_x_extent=[0.0, 0.5],
        lattice_y_extent=[0.0, 0.5],
    )

    scaffold_path = os.path.join(
        os.environ.get("BIOCOMP_ROOT", "."),
        "biocomp-jobs/design/architectures/T_2_fully_unlocked.yaml"
    )
    if not os.path.exists(scaffold_path):
        pytest.skip("T_2_fully_unlocked.yaml not found")

    recipe = Recipe.from_yaml(scaffold_path)
    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="shortest")
    assert len(networks) >= 1

    return {
        "model": model,
        "target": target,
        "networks": networks,
        "recipe": recipe,
    }


def test_committed_network_matches_evaluation(design_setup):
    """Committed network predictions should match evaluation predictions.

    This is the core invariant: after commit, predictions should be consistent.
    """
    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction
    from biocomp.parameters import ParameterTree, load_params
    import biocomp.biorules as br

    model = design_setup["model"]
    target = design_setup["target"]
    networks = design_setup["networks"]
    network = networks[0]

    resolution = (24, 24)
    X_lat, Y_target = target.get_lattice(resolution=resolution, seed=0)
    n_samples = X_lat.shape[0]
    n_replicates, n_targets, n_networks = 2, 1, 1

    dmanager = DesignManager(
        targets=[target],
        networks=[network],
        grid_resolution=resolution,
    )
    dconf = DesignConfig(n_replicates=n_replicates)
    stack = dmanager.build_stack(model, unlock_ratios=True)

    key = random.PRNGKey(42)
    init_params = stack.init(key)

    if TU_LOG_ALPHA_PATH not in init_params:
        pytest.skip("No TU masking in this network")

    orig_tu_log_alpha = init_params[TU_LOG_ALPHA_PATH]
    n_tus = orig_tu_log_alpha.shape[-1]
    assert n_tus > 0, "No TUs to mask"

    # simulate optimization: disable some TUs (log_alpha < 0 means disabled)
    tu_log_alpha_optimized = jnp.where(
        jnp.arange(n_tus) % 2 == 0,
        jnp.full_like(orig_tu_log_alpha, 3.0),   # enabled
        jnp.full_like(orig_tu_log_alpha, -3.0),  # disabled
    )
    tu_log_alpha_4d = jnp.broadcast_to(
        tu_log_alpha_optimized,
        (n_replicates, n_targets, n_networks, n_tus)
    )

    optimized_params = init_params.copy()
    optimized_params[TU_LOG_ALPHA_PATH] = tu_log_alpha_4d

    # 1. make prediction using original stack + optimized params (evaluation path)
    rep_params = jax.tree.map(lambda x: x[0, 0], optimized_params)
    num_z_val = int(rep_params["global/number_of_random_variables"][0, 0].squeeze())
    z_batch = random.uniform(key, (n_samples, num_z_val))
    keys = random.split(key, n_samples)

    def predict_single(x, z, k):
        return stack.apply(rep_params, x, z, k, tu_enabled_random_vars=None)

    yhat_eval, _ = jax.vmap(predict_single)(X_lat, z_batch, keys)
    dep_mask = stack.get_dependent_output_mask()
    Y_eval = jnp.compress(dep_mask, yhat_eval, axis=-1)[:, 0]

    # 2. commit the network and make prediction from committed network
    committed_networks = stack.commit(optimized_params)
    assert len(committed_networks) == 1
    committed_net = committed_networks[0]

    if len(committed_net.compute_graph.nodes) == 0:
        pytest.skip("Committed network is empty (all TUs pruned)")

    nm = NetworkModel(model=model, network=committed_net)
    pred = NetworkPrediction(predict_at=[X_lat], network_model=nm, already_latent=True)
    data = pred.get_data(rescale_latent=False)[0]
    Y_committed = np.asarray(data.y)

    # 3. compare predictions
    Y_eval_np = np.asarray(Y_eval)

    assert Y_eval_np.shape == Y_committed.shape, (
        f"Shape mismatch: eval={Y_eval_np.shape}, committed={Y_committed.shape}"
    )

    eval_corr_with_target = np.corrcoef(Y_eval_np.flatten(), np.asarray(Y_target).flatten())[0, 1]
    committed_corr_with_target = np.corrcoef(Y_committed.flatten(), np.asarray(Y_target).flatten())[0, 1]

    print(f"\nEval prediction: range=[{Y_eval_np.min():.4f}, {Y_eval_np.max():.4f}], corr={eval_corr_with_target:.4f}")
    print(f"Committed prediction: range=[{Y_committed.min():.4f}, {Y_committed.max():.4f}], corr={committed_corr_with_target:.4f}")

    eval_vs_committed_corr = np.corrcoef(Y_eval_np.flatten(), Y_committed.flatten())[0, 1]
    print(f"Eval vs Committed correlation: {eval_vs_committed_corr:.4f}")

    assert not np.isnan(eval_corr_with_target), "Eval correlation is NaN"
    assert not np.isnan(committed_corr_with_target), "Committed correlation is NaN"

    is_flat = Y_committed.std() < 0.01
    assert not is_flat, (
        f"BUG: Committed prediction is flat (std={Y_committed.std():.6f}). "
        f"This is the symptom of the design commit bug."
    )

    corr_diff = abs(eval_corr_with_target - committed_corr_with_target)
    assert corr_diff < 0.05, (
        f"BUG: Correlation mismatch between eval ({eval_corr_with_target:.4f}) "
        f"and committed ({committed_corr_with_target:.4f}). "
        f"Difference: {corr_diff:.4f} > 0.05 threshold. "
        f"This indicates the commit process is corrupting the network."
    )


def test_committed_ratios_preserved(design_setup):
    """Verify that ratios are correctly transferred during commit."""
    from biocomp.parameters import ParameterTree
    import biocomp.biorules as br

    model = design_setup["model"]
    target = design_setup["target"]
    networks = design_setup["networks"]
    network = networks[0]

    resolution = (16, 16)
    n_replicates, n_targets, n_networks = 1, 1, 1

    dmanager = DesignManager(
        targets=[target],
        networks=[network],
        grid_resolution=resolution,
    )
    stack = dmanager.build_stack(model, unlock_ratios=True)
    key = random.PRNGKey(42)
    init_params = stack.init(key)

    # find ratio paths
    ratio_paths = [p for p in init_params.data.flatten_keys() if "ratio" in p.lower()]
    print(f"\nRatio paths in params: {ratio_paths}")

    if not ratio_paths:
        pytest.skip("No ratio parameters found")

    if TU_LOG_ALPHA_PATH not in init_params:
        pytest.skip("No TU masking in this network")

    # keep all TUs enabled for this test
    orig_tu_log_alpha = init_params[TU_LOG_ALPHA_PATH]
    n_tus = orig_tu_log_alpha.shape[-1]
    tu_log_alpha_enabled = jnp.full_like(orig_tu_log_alpha, 3.0)
    tu_log_alpha_4d = jnp.broadcast_to(
        tu_log_alpha_enabled,
        (n_replicates, n_targets, n_networks, n_tus)
    )
    optimized_params = init_params.copy()
    optimized_params[TU_LOG_ALPHA_PATH] = tu_log_alpha_4d

    # get ratios before commit
    orig_ratios = {}
    for path in ratio_paths:
        if "ratio_min" not in path and "ratio_max" not in path:
            orig_ratios[path] = np.asarray(optimized_params[path])

    # commit
    committed_networks = stack.commit(optimized_params)
    assert len(committed_networks) == 1
    committed_net = committed_networks[0]

    agg_nodes = [n for n in committed_net.compute_graph.nodes.values() if n.node_type == "aggregation"]
    print(f"\nCommitted network aggregation nodes: {len(agg_nodes)}")
    for node in agg_nodes:
        print(f"  Node {node.node_id}: ratios={node.extra.get('ratios')}, members={node.extra.get('members')}")

    # rebuild stack from committed network and check ratios
    from biocomptools.modelmodel import NetworkModel
    nm = NetworkModel(model=model, network=committed_net)
    new_params = nm._params

    new_ratio_paths = [p for p in new_params.data.flatten_keys() if "ratio" in p.lower()]
    print(f"\nRatio paths in new params: {new_ratio_paths}")

    # check that ratios are not all equal (which would indicate they were reset to 1.0)
    for path in new_ratio_paths:
        if "ratio_min" not in path and "ratio_max" not in path:
            ratios = np.asarray(new_params[path])
            print(f"  {path}: shape={ratios.shape}, values={ratios.flatten()[:6]}")
            if ratios.size > 1:
                assert not np.allclose(ratios, ratios.flat[0]), (
                    f"BUG: All ratios in {path} are equal ({ratios.flat[0]:.4f}). "
                    f"This suggests ratios were reset during commit."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
