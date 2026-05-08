"""Validation loss computation for hyperopt."""

import numpy as np
from biocomptools.logging_config import get_logger
from biocomp.metric_utils import (
    compute_validation_objective,
    extract_metric_values,
)

logger = get_logger(__name__)


def print_stats_barplot(
    names: list[str],
    stats: list[dict],
    metric: str = 'grid_nrmse',
    width: int = 160,
    height: int = 30,
    title: str | None = None,
    best_stats: list[dict] | None = None,
    trial_num: int | None = None,
    best_trial_num: int | None = None,
):
    """Print horizontal bar chart with 2-column layout: current trial (left) and best trial (right)."""
    import plotext as plt

    values = []
    best_values = []
    valid_names = []
    for i, (name, s) in enumerate(zip(names, stats, strict=True)):
        val = s.get(metric)
        if val is not None and np.isfinite(val):
            values.append(float(val))
            short_name = name[:18] + '..' if len(name) > 20 else name
            valid_names.append(short_name)
            if best_stats and i < len(best_stats):
                bv = best_stats[i].get(metric)
                best_values.append(float(bv) if bv is not None and np.isfinite(bv) else 0.0)
            else:
                best_values.append(0.0)

    if not values:
        print(f"  No valid {metric} values to display")
        return

    # sort by current value ascending (worst at top visually)
    indices = sorted(range(len(values)), key=lambda i: values[i])
    sorted_names = [valid_names[i] for i in indices]
    sorted_vals = [values[i] for i in indices]
    sorted_best = [best_values[i] for i in indices] if best_stats else None

    try:
        plt.clf()
        plt.theme("matrix")
        plt.plot_size(width, height)

        if sorted_best and any(v > 0 for v in sorted_best):
            # 2-column layout: current trial on left, best trial on right
            plt.subplots(1, 2)

            # Left: current trial
            plt.subplot(1, 1)
            plt.bar(sorted_names, sorted_vals, orientation='h', width=0.8)
            plt.xlabel(metric)
            current_title = "Current Trial" + (f" #{trial_num}" if trial_num else "")
            plt.title(current_title)

            # Right: best trial overall
            plt.subplot(1, 2)
            plt.bar(sorted_names, sorted_best, orientation='h', width=0.8)
            plt.xlabel(metric)
            best_title = "Best Trial" + (f" #{best_trial_num}" if best_trial_num else "")
            plt.title(best_title)
        else:
            # Single column if no best stats
            plt.bar(sorted_names, sorted_vals, orientation='h', width=0.8)
            plt.xlabel(metric)
            if title:
                plt.title(title)
        plt.show()
    except Exception as e:
        print(f"  (barplot failed: {e})")


def _safe_float(v) -> float:
    """Convert value to float, returning nan for non-numeric types."""
    if v is None:
        return float('nan')
    try:
        f = float(v)
        return f if np.isfinite(f) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def print_trial_summary(
    trial_num: int,
    loss: float,
    stats: list[dict],
    names: list[str],
    metric: str = 'grid_nrmse',
    show_barplot: bool = True,
    top_n: int = 30,
    best_stats: list[dict] | None = None,
    best_loss: float | None = None,
    best_trial_number: int | None = None,
):
    """Print a summary of trial results with optional barplot and rich table."""
    values = [_safe_float(s.get(metric)) for s in stats]
    valid = [v for v in values if np.isfinite(v)]

    # try fallback metrics if primary not available
    fallback_metrics = ['grid_nrmse', 'rmse', 'mae']
    if not valid:
        for fb in fallback_metrics:
            values = [_safe_float(s.get(fb)) for s in stats]
            valid = [v for v in values if np.isfinite(v)]
            if valid:
                metric = fb
                break

    if valid:
        mean_v = np.mean(valid)
        max_v = np.max(valid)
        min_v = np.min(valid)

        # Print header with best trial info
        print(f"\n{'═' * 80}")
        is_new_best = best_trial_number == trial_num
        best_marker = " ★ NEW BEST ★" if is_new_best else ""
        print(f"Trial {trial_num}: loss={loss:.6f}{best_marker}")
        if best_loss is not None and best_trial_number is not None:
            print(f"Best so far: Trial {best_trial_number} with loss={best_loss:.6f}")
        print(f"{metric}: mean={mean_v:.4f}, min={min_v:.4f}, max={max_v:.4f}")
        print(f"{'─' * 80}")

        # Rich table for worst networks
        if len(valid) > 0:
            sorted_pairs = sorted(
                zip(names, values, stats, strict=True),
                key=lambda x: -x[1] if np.isfinite(x[1]) else float('-inf'),
            )
            _print_stats_table(sorted_pairs[:top_n], metric, best_stats, names)

        if show_barplot and len(valid) > 0:
            top_names = [p[0] for p in sorted_pairs[:top_n]]
            top_stats = [p[2] for p in sorted_pairs[:top_n]]
            top_best = None
            if best_stats:
                name_to_best = {
                    n: best_stats[i] for i, n in enumerate(names) if i < len(best_stats)
                }
                top_best = [name_to_best.get(n, {}) for n in top_names]
            print_stats_barplot(
                top_names,
                top_stats,
                metric=metric,
                best_stats=top_best,
                trial_num=trial_num,
                best_trial_num=best_trial_number,
            )
    else:
        all_keys = set()
        for s in stats:
            all_keys.update(
                k for k, v in s.items() if isinstance(v, (int, float)) and np.isfinite(v)
            )
        print(f"\nTrial {trial_num}: loss={loss:.4f} (no valid {metric} values)")
        if all_keys:
            print(f"  Available metrics: {sorted(all_keys)}")


def _print_stats_table(
    sorted_pairs: list[tuple[str, float, dict]],
    metric: str,
    best_stats: list[dict] | None,
    all_names: list[str],
):
    """Print a rich table of network stats."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fallback to simple print if rich not available
        print(f"  {'Network':<40} {metric:>10} {'best':>10} {'delta':>10}")
        print(f"  {'-' * 70}")
        name_to_best = {}
        if best_stats:
            name_to_best = {
                n: best_stats[i] for i, n in enumerate(all_names) if i < len(best_stats)
            }
        for name, val, _ in sorted_pairs[:15]:
            best_val = name_to_best.get(name, {}).get(metric)
            best_str = f"{best_val:.4f}" if best_val is not None and np.isfinite(best_val) else "-"
            delta = val - best_val if best_val is not None and np.isfinite(best_val) else None
            delta_str = f"{delta:+.4f}" if delta is not None else "-"
            print(f"  {name[:40]:<40} {val:>10.4f} {best_str:>10} {delta_str:>10}")
        return

    console = Console()
    table = Table(title=f"Top {len(sorted_pairs)} worst networks by {metric}", show_lines=False)
    table.add_column("Network", style="cyan", no_wrap=True, max_width=45)
    table.add_column("Current", justify="right", style="yellow")
    table.add_column("Best", justify="right", style="green")
    table.add_column("Δ", justify="right")

    name_to_best = {}
    if best_stats:
        name_to_best = {n: best_stats[i] for i, n in enumerate(all_names) if i < len(best_stats)}

    for name, val, _ in sorted_pairs:
        best_val = name_to_best.get(name, {}).get(metric)
        best_str = f"{best_val:.4f}" if best_val is not None and np.isfinite(best_val) else "-"

        delta = val - best_val if best_val is not None and np.isfinite(best_val) else None
        if delta is not None:
            if delta > 0.01:
                delta_str = f"[red]+{delta:.4f}[/red]"
            elif delta < -0.01:
                delta_str = f"[green]{delta:.4f}[/green]"
            else:
                delta_str = f"{delta:+.4f}"
        else:
            delta_str = "-"

        short_name = name[:43] + ".." if len(name) > 45 else name
        table.add_row(short_name, f"{val:.4f}", best_str, delta_str)

    console.print(table)


def _extract_metric(stats: list[dict], key: str, positive_only: bool = False) -> np.ndarray:
    """Extract finite metric values from stats list.

    delegates to biocomp.metric_utils.extract_metric_values
    """
    return extract_metric_values(stats, key, positive_only=positive_only)


def compute_loss_from_stats(
    stats: list[dict],
    objective: str,
    softmax_alpha: float = 5.0,
) -> float:
    """Compute scalar loss from network statistics.

    delegates to biocomp.metric_utils.compute_validation_objective
    """
    return compute_validation_objective(
        stats, objective, softmax_alpha=softmax_alpha,
    )


class ValidationRunner:
    """Handles validation loss computation with optional batched predictions."""

    def __init__(
        self,
        predictor,  # NetworkPrediction
        objective: str = "geomean_nrmse",
        softmax_alpha: float = 5.0,
    ):
        self.predictor = predictor
        self.objective = objective
        self.softmax_alpha = softmax_alpha
        self._batched_predict = None

    def compute_loss(self, params) -> float:
        """Compute validation loss for single params (extracts first replicate if batched)."""
        import jax

        try:
            single_params = jax.tree.map(lambda x: x[0], params)
        except (IndexError, TypeError):
            single_params = params
        stats = self.predictor.get_network_stats(with_shared_params=single_params)
        return compute_loss_from_stats(stats, self.objective, self.softmax_alpha)

    def compute_loss_single(self, single_params) -> float:
        """Compute validation loss for already-single params."""
        stats = self.predictor.get_network_stats(with_shared_params=single_params)
        return compute_loss_from_stats(stats, self.objective, self.softmax_alpha)

    def compute_loss_with_stats(self, params) -> tuple[float, list[dict]]:
        """Compute validation loss and return per-network stats (using NetworkPrediction as source of truth)."""
        import jax

        try:
            single_params = jax.tree.map(lambda x: x[0], params)
        except (IndexError, TypeError):
            single_params = params

        # use NetworkPrediction's canonical stats computation
        stats = self.predictor.get_network_stats(with_shared_params=single_params)
        loss = compute_loss_from_stats(stats, self.objective, self.softmax_alpha)
        return loss, stats

    def get_network_names(self) -> list[str]:
        """Get names of networks being validated."""
        nets = self.predictor.network_model.network
        if not isinstance(nets, list):
            nets = [nets]
        return [getattr(n, 'name', f'net_{i}') for i, n in enumerate(nets)]

    def compute_losses_batched(
        self, all_params, verbose: bool = False, return_best_stats: bool = False
    ) -> list[float] | tuple[list[float], int, list[dict]]:
        """Compute validation losses for batched params using vmapped prediction."""
        import jax
        import jax.numpy as jnp
        import time
        from biocomp import parameters as pr
        from biocomptools.modelmodel import get_shared_params

        def get_n_trials(params):
            for leaf in jax.tree.leaves(params):
                if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
                    return leaf.shape[0]
            return 1

        n_trials = get_n_trials(all_params)
        predictor = self.predictor
        network_model = predictor.network_model
        stack = network_model._stack

        stacked_x = np.column_stack(predictor._aligned_x)
        n_samples = len(predictor._aligned_x[0])
        num_z = int(all_params["global/number_of_random_variables"][0])

        if verbose:
            print(f"[batched-val] {n_trials} trials × {n_samples} samples")

        key = jax.random.PRNGKey(predictor.seed or 42)
        Z = (
            jax.random.uniform(key, (n_samples, num_z))
            if predictor.z_value == 'uniform'
            else jnp.ones((n_samples, num_z)) * predictor.z_value
        )
        keys = jax.random.split(key, n_samples)

        logstd_path = 'shared/quantization/logstdevs'
        if logstd_path in all_params:
            logstd = all_params[logstd_path]
            for path, value in logstd.iter_leaves():
                logstd[path] = jnp.ones_like(value) * -100

        if self._batched_predict is None:
            sample_vmap = jax.vmap(stack.apply, in_axes=(None, 0, 0, 0))
            trial_vmap = jax.vmap(sample_vmap, in_axes=(0, None, None, None))
            try:
                self._batched_predict = jax.jit(trial_vmap, device=jax.devices('gpu')[0])
            except (RuntimeError, IndexError):
                self._batched_predict = jax.jit(trial_vmap)

        def ensure_batch_dim(params, n_trials, already_batched=False):
            def proc(x):
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    if len(x.shape) == 0:
                        return jnp.broadcast_to(x, (n_trials,))
                    return x if already_batched else jnp.broadcast_to(x, (n_trials,) + x.shape)
                arr = jnp.asarray(x)
                if arr.shape == ():
                    return jnp.broadcast_to(arr, (n_trials,))
                return arr if already_batched else jnp.broadcast_to(arr, (n_trials,) + arr.shape)

            return jax.tree.map(proc, params)

        shared_params = ensure_batch_dim(get_shared_params(all_params), n_trials, True)
        local_params = ensure_batch_dim(network_model._local_params, n_trials, False)
        merged = pr.ParameterTree.merge(shared_params, local_params)

        t0 = time.time()
        out, _ = self._batched_predict(merged, jnp.array(stacked_x), Z, keys)
        out.block_until_ready()
        if verbose:
            print(f"[batched-val] prediction: {time.time() - t0:.1f}s")

        t0 = time.time()
        losses = []
        all_stats = [] if return_best_stats else None

        # Use rich progress for stats computation
        if verbose:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[batched-val] stats", total=n_trials)
                for trial_idx in range(n_trials):
                    trial_out = np.asarray(out[trial_idx])
                    network_outputs = network_model.split_outputs_per_network(trial_out, n_samples)
                    stats = predictor.compute_stats_from_outputs(network_outputs)
                    loss = compute_loss_from_stats(
                        stats, self.objective, self.softmax_alpha
                    )
                    losses.append(loss)
                    if return_best_stats:
                        all_stats.append(stats)
                    progress.update(task, advance=1)
            print(f"[batched-val] stats: {time.time() - t0:.1f}s")
        else:
            for trial_idx in range(n_trials):
                trial_out = np.asarray(out[trial_idx])
                network_outputs = network_model.split_outputs_per_network(trial_out, n_samples)
                stats = predictor.compute_stats_from_outputs(network_outputs)
                loss = compute_loss_from_stats(
                    stats, self.objective, self.softmax_alpha
                )
                losses.append(loss)
                if return_best_stats:
                    all_stats.append(stats)

        if return_best_stats:
            best_idx = int(np.argmin(losses))
            return losses, best_idx, all_stats[best_idx]
        return losses
