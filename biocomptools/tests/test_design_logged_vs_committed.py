"""Test that logged predictions match committed network predictions.

This test captures the specific bug where heatmap logger shows good predictions
during design optimization, but the final committed network prediction looks
completely different (and worse).

The test runs actual design optimization (not synthetic params) to reproduce
the exact scenario users encounter.
"""
import os
import pytest
import numpy as np
from pathlib import Path

import dracon as dr
from biocomp.design import DesignManager, DesignConfig, start
from biocomp.design_targets import SVGTarget, LatticeSampling
from biocomp.jaxutils import tree_get
from biocomp.logger_dispatch import LoggerDispatch
from biocomp.recipe import Recipe

pytestmark = pytest.mark.skipif(
    not os.environ.get("BIOCOMP_DESIGNER_MODEL"),
    reason="BIOCOMP_DESIGNER_MODEL not set",
)


@pytest.fixture
def design_setup():
    from biocomptools.modelmodel import BiocompModel
    from biocomp.network import recipe_to_networks
    import biocomp.biorules as br

    model = BiocompModel.load(os.environ["BIOCOMP_DESIGNER_MODEL"])

    svg_path = os.path.join(
        os.environ.get("BIOCOMP_ROOT", "."),
        "Designs/MIT_T.svg"
    )
    if not os.path.exists(svg_path):
        pytest.skip("MIT_T.svg not found")

    target = SVGTarget(
        name="test_target",
        path=svg_path,
        lattice_x_extent=[0.0, 0.5],
        lattice_y_extent=[0.0, 0.5],
    )

    scaffold_path = Path(__file__).parent.parent.parent.parent / "biocomp-jobs/design/architectures/T_2_fully_unlocked.yaml"
    if not os.path.exists(scaffold_path):
        pytest.skip("T_2_fully_unlocked.yaml not found")

    data = dr.load(str(scaffold_path), context={"Recipe": Recipe})
    recipe = data["recipe"]
    networks = recipe_to_networks(recipe, br.ALL_RULES, invert=True, inversion_mode="shortest")
    assert len(networks) >= 1

    return {
        "model": model,
        "target": target,
        "networks": networks[:1],
        "recipe": recipe,
    }


def compute_metrics(y_pred, y_target):
    """Compute various metrics for comparison."""
    y_pred = np.asarray(y_pred).ravel()
    y_target = np.asarray(y_target).ravel()

    corr = np.corrcoef(y_pred, y_target)[0, 1] if len(y_pred) > 1 else 0.0
    mse = float(np.mean((y_pred - y_target) ** 2))
    rmse = float(np.sqrt(mse))

    y_target_mean = y_target.mean()
    ss_tot = np.sum((y_target - y_target_mean) ** 2)
    nrmse = rmse / np.sqrt(ss_tot / len(y_target)) if ss_tot > 0 else float('inf')

    return {
        "correlation": corr,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mean": float(y_pred.mean()),
        "std": float(y_pred.std()),
        "min": float(y_pred.min()),
        "max": float(y_pred.max()),
    }


def print_side_by_side_text_plot(y_logged, y_committed, y_target, resolution=(24, 24)):
    """Print side-by-side text plots for visual comparison."""
    chars = " ░▒▓█"

    def to_grid(y, res):
        y = np.asarray(y).ravel()
        n = res[0] * res[1]
        if len(y) != n:
            return None
        return y.reshape(res)

    def normalize_grid(g):
        if g is None:
            return None
        gmin, gmax = g.min(), g.max()
        if gmax - gmin < 1e-6:
            return np.zeros_like(g)
        return (g - gmin) / (gmax - gmin)

    def grid_to_chars(g, width=20):
        if g is None:
            return ["?" * width] * 10
        g = normalize_grid(g)
        h, w = g.shape
        scale_h = max(1, h // 10)
        scale_w = max(1, w // width)
        lines = []
        for i in range(0, h, scale_h):
            line = ""
            for j in range(0, w, scale_w):
                val = g[i:i+scale_h, j:j+scale_w].mean()
                char_idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
                line += chars[char_idx]
            lines.append(line[:width])
        return lines[:10]

    g_logged = to_grid(y_logged, resolution)
    g_committed = to_grid(y_committed, resolution)
    g_target = to_grid(y_target, resolution)

    lines_logged = grid_to_chars(g_logged)
    lines_committed = grid_to_chars(g_committed)
    lines_target = grid_to_chars(g_target)

    print("\n" + "=" * 70)
    print(f"{'TARGET':^20} | {'LOGGED':^20} | {'COMMITTED':^20}")
    print("-" * 70)
    for i in range(min(len(lines_target), len(lines_logged), len(lines_committed))):
        print(f"{lines_target[i]:^20} | {lines_logged[i]:^20} | {lines_committed[i]:^20}")
    print("=" * 70)


def test_logged_yhatdep_matches_committed_prediction(design_setup):
    """Logged yhatdep from optimization should match committed network prediction.

    This is THE core test for the reported bug: heatmap logger shows good predictions
    during design, but final committed network prediction looks completely different.
    """
    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    model = design_setup["model"]
    target = design_setup["target"]
    networks = design_setup["networks"]

    resolution = (24, 24)
    n_replicates = 2
    n_epochs = 10

    dmanager = DesignManager(
        targets=[target],
        networks=networks,
        sampling=LatticeSampling(resolution=resolution, jitter_std=0.0, noise_std=0.0),
        enable_tu_masking=True,
    )

    dconf = DesignConfig(
        n_replicates=n_replicates,
        n_epochs=n_epochs,
        batch_size=1,
        n_batches_per_epoch=10,
        batches_per_step=1,
        keep_in_history="all",
        reshuffle_batches=False,
        seed=42,
    )

    captured_yhatdep = []
    captured_step = []
    captured_loss = []

    def capture_logger(step, config, step_history=None, stack=None, **kwargs):
        """Custom logger to capture yhatdep from step_history."""
        if step_history is None:
            return
        if "yhatdep" in step_history:
            captured_yhatdep.append(np.asarray(step_history["yhatdep"]))
            captured_step.append(step)
        if "loss" in step_history:
            captured_loss.append(np.asarray(step_history["loss"]))

    loggers = [
        (10, capture_logger),
        (-1, capture_logger),
    ]

    class PeriodicCallbackDispatch(LoggerDispatch):
        def __init__(self, callbacks):
            self.callbacks = callbacks

        def on_start(self, config: object, stack: object) -> None:
            for period, callback in self.callbacks:
                if period == 0:
                    callback(0, config, step_history={}, stack=stack)

        def on_step(self, step: int, config: object, step_history: dict, stack: object) -> None:
            for period, callback in self.callbacks:
                if period is not None and period > 0 and step % period == 0:
                    callback(step, config, step_history=step_history, stack=stack)

        def on_end(self, step: int, config: object, step_history: dict, stack: object) -> None:
            for period, callback in self.callbacks:
                if period is None or period == -1:
                    callback(step, config, step_history=step_history, stack=stack)

        def needs_params_sync(self, step: int) -> bool:
            return False

    dispatch = PeriodicCallbackDispatch(loggers)

    print("\n" + "=" * 70)
    print("RUNNING DESIGN OPTIMIZATION")
    print("=" * 70)

    try:
        final_params, loss_history, step_history = start(
            dmanager=dmanager,
            dconf=dconf,
            model=model,
            dispatch=dispatch,
        )
    except AssertionError as exc:
        if "No TUs in stack" in str(exc):
            pytest.skip(f"Design setup has no TUs for masking: {exc}")
        raise

    assert len(captured_yhatdep) > 0, "No yhatdep captured from loggers"

    final_logged_yhatdep = captured_yhatdep[-1]
    final_step = captured_step[-1]

    print(f"\nCaptured {len(captured_yhatdep)} yhatdep snapshots")
    print(f"Final snapshot at step {final_step}")
    print(f"Final logged yhatdep shape: {final_logged_yhatdep.shape}")

    n_networks = len(networks)
    n_targets = dmanager.n_targets

    stack = dmanager.build_stack(model, unlock_ratios=True)

    X_lat, Y_target = target.get_lattice(resolution=resolution, seed=0)
    X_lat.shape[0]

    print(f"\nX_lat shape: {X_lat.shape}, Y_target shape: {Y_target.shape}")

    rep_id, tid, net_id = 0, 0, 0

    bparams = tree_get(final_params, (rep_id, tid))

    print(f"\nCommitting network for rep={rep_id}, target={tid}...")
    committed_networks = stack.commit(bparams)
    assert len(committed_networks) == n_networks, f"Expected {n_networks} networks, got {len(committed_networks)}"

    committed_net = committed_networks[net_id]

    if len(committed_net.compute_graph.nodes) == 0:
        pytest.skip("Committed network is empty (all TUs pruned)")

    print(f"Committed network has {len(committed_net.compute_graph.nodes)} nodes")

    nm = NetworkModel(model=model, network=committed_net)
    pred = NetworkPrediction(
        predict_at=[X_lat],
        network_model=nm,
        already_latent=True,
    )
    data = pred.get_data(rescale_latent=False)[0]
    Y_committed = np.asarray(data.y).ravel()

    print(f"\nCommitted prediction shape: {Y_committed.shape}")
    print(f"Committed prediction range: [{Y_committed.min():.4f}, {Y_committed.max():.4f}]")

    Y_logged = None
    (n_replicates, final_logged_yhatdep.shape[1], n_targets, n_networks)

    if final_logged_yhatdep.ndim == 4:
        print(f"yhatdep is 4D: {final_logged_yhatdep.shape}")
        Y_logged = final_logged_yhatdep[rep_id, :, tid, net_id]
    elif final_logged_yhatdep.ndim == 3:
        print(f"yhatdep is 3D: {final_logged_yhatdep.shape}")
        Y_logged = final_logged_yhatdep[rep_id, :, tid] if final_logged_yhatdep.shape[2] == n_targets else final_logged_yhatdep[rep_id, :, net_id]
    elif final_logged_yhatdep.ndim == 2:
        print(f"yhatdep is 2D: {final_logged_yhatdep.shape}")
        Y_logged = final_logged_yhatdep[rep_id, :]
    else:
        print(f"yhatdep is 5D: {final_logged_yhatdep.shape}")
        Y_logged = final_logged_yhatdep[rep_id, 0, :, tid, net_id]

    Y_logged = np.asarray(Y_logged).ravel()
    Y_target_flat = np.asarray(Y_target).ravel()

    print(f"\nLogged prediction shape: {Y_logged.shape}")
    print(f"Logged prediction range: [{Y_logged.min():.4f}, {Y_logged.max():.4f}]")

    if len(Y_logged) != len(Y_committed):
        print(f"WARNING: Shape mismatch - logged has {len(Y_logged)} samples, committed has {len(Y_committed)}")
        min_len = min(len(Y_logged), len(Y_committed), len(Y_target_flat))
        Y_logged = Y_logged[:min_len]
        Y_committed = Y_committed[:min_len]
        Y_target_flat = Y_target_flat[:min_len]

    logged_metrics = compute_metrics(Y_logged, Y_target_flat)
    committed_metrics = compute_metrics(Y_committed, Y_target_flat)

    print("\n" + "=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'Logged':>15} {'Committed':>15} {'Difference':>15}")
    print("-" * 70)
    for key in ["correlation", "mse", "rmse", "nrmse", "mean", "std"]:
        logged_val = logged_metrics[key]
        committed_val = committed_metrics[key]
        diff = abs(logged_val - committed_val)
        print(f"{key:<20} {logged_val:>15.4f} {committed_val:>15.4f} {diff:>15.4f}")

    logged_vs_committed_corr = np.corrcoef(Y_logged, Y_committed)[0, 1]
    print(f"\n{'Logged vs Committed correlation:':>35} {logged_vs_committed_corr:.4f}")

    print_side_by_side_text_plot(Y_logged, Y_committed, Y_target_flat, resolution)

    logged_corr = logged_metrics["correlation"]
    committed_corr = committed_metrics["correlation"]
    corr_diff = abs(logged_corr - committed_corr)

    assert not np.isnan(logged_corr), "Logged correlation is NaN"
    assert not np.isnan(committed_corr), "Committed correlation is NaN"

    is_flat = Y_committed.std() < 0.01
    assert not is_flat, (
        f"BUG: Committed prediction is flat (std={Y_committed.std():.6f}). "
        f"This is a symptom of the design commit bug - predictions are being corrupted."
    )

    assert corr_diff < 0.05, (
        f"BUG: Correlation difference between logged ({logged_corr:.4f}) and "
        f"committed ({committed_corr:.4f}) is {corr_diff:.4f} > 0.05 threshold.\n"
        f"This confirms the reported bug: logged predictions during design look good, "
        f"but committed network predictions are significantly different."
    )

    assert logged_vs_committed_corr > 0.95, (
        f"BUG: Direct correlation between logged and committed predictions is only "
        f"{logged_vs_committed_corr:.4f} < 0.95 threshold.\n"
        f"The predictions should be nearly identical but they're not."
    )

    for metric_name in ["nrmse", "rmse"]:
        logged_val = logged_metrics[metric_name]
        committed_val = committed_metrics[metric_name]
        diff = abs(logged_val - committed_val)
        rel_diff = diff / max(logged_val, 1e-6)
        assert rel_diff < 0.05, (
            f"BUG: {metric_name} differs by {rel_diff*100:.1f}% between logged and committed. "
            f"Logged: {logged_val:.4f}, Committed: {committed_val:.4f}"
        )

    print("\n" + "=" * 70)
    print("TEST PASSED: Logged predictions match committed predictions")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
