"""Design Heatmap Logger: Rich ASCII visualization of target vs prediction during optimization."""

import numpy as np
from typing import Any

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomp.plotting.ascii_heatmap import heatmap
from biocomp.designutils import (
    side_by_side_txt_plot,
    LOSS_ORDER,
)

logger = get_logger(__name__)

PENALTY_ORDER = [
    "l0_penalty",
    "spread_penalty",
    "coupling_penalty",
    "tucount_penalty",
    "ern_tying_penalty",
]


def _to_scalar(val: Any, default: float = 0.0) -> float:
    """Convert array-like (numpy/JAX) or scalar to float, averaging if multi-element."""
    if val is None:
        return default
    if hasattr(val, 'shape'):
        size = getattr(val, 'size', 1)
        if size > 1:
            return float(np.mean(val))
        if size == 1:
            return float(np.asarray(val).item())
    try:
        return float(val) if val else default
    except (TypeError, ValueError):
        return default


def _extract_at_indices(arr: Any, rid: int, tid: int, nid: int) -> float:
    """Extract scalar value from array at replicate rid, target tid, network nid."""
    if arr is None:
        return 0.0
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[min(nid, len(arr) - 1)])
    if arr.ndim == 2:
        # (n_targets, n_networks)
        return float(arr[min(tid, arr.shape[0] - 1), min(nid, arr.shape[1] - 1)])
    if arr.ndim == 3:
        # (n_replicates, n_targets, n_networks)
        return float(
            arr[min(rid, arr.shape[0] - 1), min(tid, arr.shape[1] - 1), min(nid, arr.shape[2] - 1)]
        )
    if arr.ndim == 4:
        # (n_replicates, n_batches, n_targets, n_networks) -> take last batch
        return float(
            arr[
                min(rid, arr.shape[0] - 1),
                -1,
                min(tid, arr.shape[2] - 1),
                min(nid, arr.shape[3] - 1),
            ]
        )
    return float(np.mean(arr))


def _format_loss_table_comparison(
    training_sublosses: dict,
    eval_losses: dict | None,
    penalties: dict,
    loss_weights: dict,
    show_penalties: bool = True,
) -> list[str]:
    """Format loss table with Training vs Eval comparison (3-column with delta)."""
    lines = []
    lines.append("┌─────────────────┬──────────┬──────────┬─────────┐")
    lines.append("│ Grid Losses     │ Training │     Eval │   Delta │")
    lines.append("├─────────────────┼──────────┼──────────┼─────────┤")

    train_grid_total = 0.0
    eval_grid_total = 0.0

    for name in LOSS_ORDER:
        train_key = f"{name}_weighted"
        train_val = training_sublosses.get(train_key, training_sublosses.get(name, 0.0))
        if hasattr(train_val, 'ndim') and train_val.ndim > 0:
            train_val = float(np.mean(np.asarray(train_val)))
        elif train_val:
            train_val = float(train_val)
        else:
            train_val = 0.0

        eval_val = eval_losses.get(name, 0.0) if eval_losses else None
        if eval_val is not None:
            if hasattr(eval_val, 'ndim') and eval_val.ndim > 0:
                eval_val = float(np.mean(np.asarray(eval_val)))
            weight = loss_weights.get(f"w_{name}", loss_weights.get(name, 1.0))
            if hasattr(weight, 'ndim') and weight.ndim > 0:
                weight = float(np.mean(np.asarray(weight)))
            eval_val = float(eval_val) * float(weight)

        if train_val > 1e-8 or (eval_val is not None and eval_val > 1e-8):
            train_grid_total += train_val
            if eval_val is not None:
                eval_grid_total += eval_val
                delta = abs(train_val - eval_val)
                lines.append(f"│ {name:15} │ {train_val:8.4f} │ {eval_val:8.4f} │ {delta:7.4f} │")
            else:
                lines.append(f"│ {name:15} │ {train_val:8.4f} │      n/a │         │")

    lines.append("├─────────────────┼──────────┼──────────┼─────────┤")
    if eval_losses:
        delta_total = abs(train_grid_total - eval_grid_total)
        lines.append(
            f"│ {'GRID TOTAL':15} │ {train_grid_total:8.4f} │ {eval_grid_total:8.4f} │ {delta_total:7.4f} │"
        )
    else:
        lines.append(f"│ {'GRID TOTAL':15} │ {train_grid_total:8.4f} │      n/a │         │")

    if show_penalties and penalties:
        lines.append("├─────────────────┼──────────┼──────────┼─────────┤")
        penalty_sum = 0.0
        for pen_name in PENALTY_ORDER:
            pen_val = _to_scalar(penalties.get(pen_name, 0.0))
            if pen_val > 1e-8:
                penalty_sum += pen_val
                lines.append(f"│ {pen_name:15} │ {pen_val:8.4f} │      n/a │         │")

        if penalty_sum > 1e-8:
            lines.append("├─────────────────┼──────────┼──────────┼─────────┤")
            total_loss = train_grid_total + penalty_sum
            lines.append(f"│ {'TOTAL LOSS':15} │ {total_loss:8.4f} │      n/a │         │")

    lines.append("└─────────────────┴──────────┴──────────┴─────────┘")
    return lines


def _format_loss_table_eval_only(
    eval_losses: dict,
    penalties: dict,
    loss_weights: dict,
    show_penalties: bool = True,
) -> list[str]:
    """Format loss table with eval-only column (for NSGA2 mode where no training sublosses)."""
    lines = []
    lines.append("┌─────────────────┬──────────┐")
    lines.append("│ Grid Losses     │     Eval │")
    lines.append("├─────────────────┼──────────┤")

    grid_total = 0.0

    for name in LOSS_ORDER:
        raw_val = _to_scalar(eval_losses.get(name, 0.0))
        weight = _to_scalar(loss_weights.get(f"w_{name}", loss_weights.get(name, 1.0)), default=1.0)
        weighted_val = raw_val * weight

        if weighted_val > 1e-8:
            grid_total += weighted_val
            lines.append(f"│ {name:15} │ {weighted_val:8.4f} │")

    lines.append("├─────────────────┼──────────┤")
    lines.append(f"│ {'GRID TOTAL':15} │ {grid_total:8.4f} │")

    if show_penalties and penalties:
        lines.append("├─────────────────┼──────────┤")
        penalty_sum = 0.0
        for pen_name in PENALTY_ORDER:
            pen_val = _to_scalar(penalties.get(pen_name, 0.0))
            if pen_val > 1e-8:
                penalty_sum += pen_val
                lines.append(f"│ {pen_name:15} │ {pen_val:8.4f} │")

        if penalty_sum > 1e-8:
            lines.append("├─────────────────┼──────────┤")
            total_loss = grid_total + penalty_sum
            lines.append(f"│ {'TOTAL':15} │ {total_loss:8.4f} │")

    lines.append("└─────────────────┴──────────┘")
    return lines


def _format_loss_table_simple(
    training_sublosses: dict,
    penalties: dict,
    loss_weights: dict,
    show_penalties: bool = True,
) -> list[str]:
    """Format loss table with Unweighted vs Weighted columns (2-column)."""
    lines = []
    lines.append("┌─────────────────┬──────────┬──────────┐")
    lines.append("│ Grid Losses     │ Unweight │ Weighted │")
    lines.append("├─────────────────┼──────────┼──────────┤")

    weighted_total = 0.0

    for name in LOSS_ORDER:
        raw_val = training_sublosses.get(name, 0.0)
        weighted_val = training_sublosses.get(f"{name}_weighted", 0.0)

        if hasattr(raw_val, 'ndim') and raw_val.ndim > 0:
            raw_val = float(np.mean(np.asarray(raw_val)))
        elif raw_val:
            raw_val = float(raw_val)
        else:
            raw_val = 0.0

        if hasattr(weighted_val, 'ndim') and weighted_val.ndim > 0:
            weighted_val = float(np.mean(np.asarray(weighted_val)))
        elif weighted_val:
            weighted_val = float(weighted_val)
        else:
            weighted_val = 0.0

        if raw_val > 1e-8 or weighted_val > 1e-8:
            weighted_total += weighted_val
            lines.append(f"│ {name:15} │ {raw_val:8.4f} │ {weighted_val:8.4f} │")

    lines.append("├─────────────────┼──────────┼──────────┤")
    lines.append(f"│ {'GRID TOTAL':15} │          │ {weighted_total:8.4f} │")

    if show_penalties and penalties:
        lines.append("├─────────────────┼──────────┼──────────┤")
        penalty_sum = 0.0
        for pen_name in PENALTY_ORDER:
            pen_val = _to_scalar(penalties.get(pen_name, 0.0))
            if pen_val > 1e-8:
                penalty_sum += pen_val
                lines.append(f"│ {pen_name:15} │          │ {pen_val:8.4f} │")

        if penalty_sum > 1e-8:
            lines.append("├─────────────────┼──────────┼──────────┤")
            total_loss = weighted_total + penalty_sum
            lines.append(f"│ {'TOTAL LOSS':15} │          │ {total_loss:8.4f} │")

    lines.append("└─────────────────┴──────────┴──────────┘")
    return lines


def _format_loss_table_per_design(
    training_sublosses: dict,
    penalties: dict,
    loss_weights: dict,
    rid: int,
    tid: int,
    nid: int,
    show_penalties: bool = True,
) -> list[str]:
    """Format loss table extracting values for a specific (replicate, target, network)."""
    lines = []
    lines.append("┌─────────────────┬──────────┬──────────┐")
    lines.append("│ Grid Losses     │ Unweight │ Weighted │")
    lines.append("├─────────────────┼──────────┼──────────┤")

    weighted_total = 0.0

    for name in LOSS_ORDER:
        raw_val = _extract_at_indices(training_sublosses.get(name), rid, tid, nid)
        weighted_val = _extract_at_indices(
            training_sublosses.get(f"{name}_weighted"), rid, tid, nid
        )

        if raw_val > 1e-8 or weighted_val > 1e-8:
            weighted_total += weighted_val
            lines.append(f"│ {name:15} │ {raw_val:8.4f} │ {weighted_val:8.4f} │")

    lines.append("├─────────────────┼──────────┼──────────┤")
    lines.append(f"│ {'GRID TOTAL':15} │          │ {weighted_total:8.4f} │")

    if show_penalties and penalties:
        lines.append("├─────────────────┼──────────┼──────────┤")
        penalty_sum = 0.0
        for pen_name in PENALTY_ORDER:
            pen_val = _to_scalar(penalties.get(pen_name, 0.0))
            if pen_val > 1e-8:
                penalty_sum += pen_val
                lines.append(f"│ {pen_name:15} │          │ {pen_val:8.4f} │")

        if penalty_sum > 1e-8:
            lines.append("├─────────────────┼──────────┼──────────┤")
            total_loss = weighted_total + penalty_sum
            lines.append(f"│ {'TOTAL LOSS':15} │          │ {total_loss:8.4f} │")

    lines.append("└─────────────────┴──────────┴──────────┘")
    return lines


class DesignHeatmapLogger(Logger):
    """Logger that prints rich ASCII heatmaps of target vs prediction side-by-side.

    Features:
    - Side-by-side TARGET vs PREDICTION heatmaps
    - Box-drawing loss tables with grid losses and penalties
    - Optional Training vs Eval comparison mode for debugging discrepancies
    - Per-target breakdown and ratio statistics
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    required_arrays: list[str] = ["yhatdep", "latest_params"]
    async_ok: bool = False
    xres: int = Field(default=40, description="Horizontal resolution for ASCII heatmap")
    yres: int = Field(default=20, description="Vertical resolution for ASCII heatmap")
    show_stats: bool = Field(default=True, description="Show statistics below heatmaps")
    target_idx: int = Field(default=0, description="Which target to visualize")
    network_idx: int | None = Field(default=None, description="Which network (None=best)")

    show_comparison: bool = Field(
        default=False, description="Show Training vs Eval losses side-by-side"
    )
    show_penalties: bool = Field(default=True, description="Show penalty breakdown in table")
    show_ratio_stats: bool = Field(default=True, description="Show ratio min/max/mean")
    show_local_params: bool = Field(
        default=True, description="Show local parameter values (ratios)"
    )
    top_k: int = Field(default=3, description="Number of top networks to display (by loss)")

    _dmanager: Any = None
    _model: Any = None
    _stack: Any = None
    _grid_resolution: tuple[int, int] | None = None
    _cached_target_grid: np.ndarray | None = None
    _total_steps: int = 0
    _loss_weights: dict = {}
    _n_targets: int = 1
    _network_names: list[str] = []

    use_fresh_predictions: bool = Field(
        default=True,
        description="Compute fresh predictions from latest_params (ensures display matches fingerprint)",
    )
    fingerprint_only_at_end: bool = Field(
        default=True,
        description="Only compute fingerprint at final step (saves time during training)",
    )

    save_reproduction_pickle: bool = Field(
        default=False,
        description="Save pickle with data for debugging logged vs committed prediction discrepancies",
    )
    reproduction_pickle_path: str | None = Field(
        default=None,
        description="Path to save reproduction pickle (default: output_dir/heatmap_repro.pickle)",
    )

    _output_dir: str | None = None
    _last_step_history: dict | None = None
    _model_path: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dmanager = None
        self._model = None
        self._stack = None
        self._grid_resolution = None
        self._cached_target_grid = None
        self._total_steps = 0
        self._loss_weights = {}
        self._n_targets = 1
        self._network_names = []
        self._output_dir = None
        self._last_step_history = None
        self._model_path = None

    def initialize(self, training_program=None):
        if training_program:
            if hasattr(training_program, '_model'):
                self._model = training_program._model
            if hasattr(training_program, '_save_dir'):
                self._output_dir = str(training_program._save_dir)
            if hasattr(training_program, 'model_name') and training_program.model_name:
                self._model_path = training_program.model_name
            elif self._model and hasattr(self._model, 'path'):
                self._model_path = str(self._model.path)
            if hasattr(training_program, '_dmanager'):
                self._dmanager = training_program._dmanager
                if self._dmanager:
                    if (
                        hasattr(self._dmanager, 'grid_resolution')
                        and self._dmanager.grid_resolution
                    ):
                        self._grid_resolution = self._dmanager.grid_resolution
                    if hasattr(self._dmanager, 'n_targets'):
                        self._n_targets = self._dmanager.n_targets
                    if hasattr(self._dmanager, 'networks'):
                        self._network_names = [n.name for n in self._dmanager.networks]

            if hasattr(training_program, 'design_conf'):
                dc = training_program.design_conf
                if hasattr(dc, 'n_epochs') and hasattr(dc, 'n_batches_per_epoch'):
                    self._total_steps = dc.n_epochs * dc.n_batches_per_epoch

                # Extract loss weights from loss_function.kwargs (primary source)
                if hasattr(dc, 'loss_function'):
                    lf = dc.loss_function
                    if hasattr(lf, 'kwargs') and lf.kwargs:
                        for k, v in lf.kwargs.items():
                            if k.startswith('w_'):
                                self._loss_weights[k] = v

                # Also check direct attributes as fallback
                for attr in [
                    'w_sinkhorn',
                    'w_lncc',
                    'w_rmse',
                    'w_mse',
                    'w_simse',
                    'w_spectral',
                    'w_gradient',
                    'w_contrast',
                    'w_zncc',
                ]:
                    if attr not in self._loss_weights and hasattr(dc, attr):
                        self._loss_weights[attr] = getattr(dc, attr)

        if self._dmanager and self._model:
            try:
                self._stack = self._dmanager.build_stack(self._model)
            except Exception as e:
                logger.warning(f"Failed to build stack for introspection: {e}")

        if self._dmanager and self._grid_resolution:
            targets = self._dmanager.targets
            if self.target_idx < len(targets):
                target = targets[self.target_idx]
                try:
                    _, Y_grid = target.get_lattice(resolution=self._grid_resolution, seed=0)
                    self._cached_target_grid = Y_grid
                except Exception as e:
                    logger.warning(f"Failed to cache target grid: {e}")

    def _compute_fresh_prediction(
        self, params, rep_idx: int, target_idx: int, network_idx: int
    ) -> np.ndarray | None:
        """Compute fresh prediction from params to ensure display matches fingerprint.

        Uses deterministic z_value=0.0 (same as fingerprint computation) on the canonical
        lattice grid. This fixes the timing mismatch where logged yhatdep is from pre-update
        params but latest_params is post-update.
        """
        if self._model is None or self._dmanager is None or self._grid_resolution is None:
            return None

        try:
            import jax
            from biocomp.jaxutils import tree_get
            from biocomptools.modelmodel import NetworkModel

            stack = self._dmanager.build_stack(self._model)
            specific_params = tree_get(params, (rep_idx, target_idx))
            committed_networks = stack.commit(specific_params)

            if network_idx >= len(committed_networks):
                return None

            committed_net = committed_networks[network_idx]
            if not committed_net.compute_graph.nodes:
                return None

            nm = NetworkModel(model=self._model, network=committed_net)
            target = self._dmanager.targets[target_idx]
            X_lat, _ = target.get_lattice(resolution=self._grid_resolution, seed=0)
            X_lat = np.asarray(X_lat)

            Y_pred, _ = nm.predict(
                X_lat,
                key=jax.random.PRNGKey(42),
                disable_variational=True,
                z_value=0.0,
            )
            Y_pred = np.asarray(Y_pred).flatten()
            xres, yres = self._grid_resolution
            return Y_pred.reshape(yres, xres)

        except Exception as e:
            logger.debug(
                f"Fresh prediction failed (rid={rep_idx}, tid={target_idx}, nid={network_idx}): {e}"
            )
            return None

    def _get_top_k_designs(
        self, step_history: dict, tid: int, n_replicates: int, n_networks: int
    ) -> list[tuple[int, int, float]]:
        """Get top k (replicate, network) pairs sorted by loss. Returns [(rep_idx, net_idx, loss), ...]."""
        all_losses = step_history.get("all_losses")
        if all_losses is None:
            return [(r, n, float('nan')) for r in range(n_replicates) for n in range(n_networks)][
                : self.top_k
            ]

        arr = np.asarray(all_losses)

        # Build list of (rep_idx, net_idx, loss) for all designs
        designs: list[tuple[int, int, float]] = []

        if arr.ndim == 0:
            designs = [(0, 0, float(arr))]
        elif arr.ndim == 1:
            # Shape: (n_networks,) - single replicate
            for nid in range(len(arr)):
                designs.append((0, nid, float(arr[nid])))
        elif arr.ndim == 2:
            # Shape: (n_targets, n_networks) - single replicate
            tid_safe = min(tid, arr.shape[0] - 1)
            for nid in range(arr.shape[1]):
                designs.append((0, nid, float(arr[tid_safe, nid])))
        elif arr.ndim == 3:
            # Shape: (n_replicates, n_targets, n_networks)
            tid_safe = min(tid, arr.shape[1] - 1)
            for rid in range(arr.shape[0]):
                for nid in range(arr.shape[2]):
                    designs.append((rid, nid, float(arr[rid, tid_safe, nid])))
        elif arr.ndim == 4:
            # Shape: (n_replicates, n_batches, n_targets, n_networks)
            # Take last batch (most recent)
            tid_safe = min(tid, arr.shape[2] - 1)
            for rid in range(arr.shape[0]):
                for nid in range(arr.shape[3]):
                    loss = float(arr[rid, -1, tid_safe, nid])
                    designs.append((rid, nid, loss))
        else:
            # Fallback: average extra dims
            designs = [(0, 0, float(np.mean(arr)))]

        # Sort by loss (ascending) and take top k
        designs.sort(key=lambda x: x[2])
        return designs[: self.top_k]

    def _render_single_design(
        self,
        step: int,
        step_history: dict,
        yhatdep: np.ndarray,
        tid: int,
        rid: int,
        nid: int,
        rank: int,
        loss: float,
        Y_target_grid: np.ndarray | None,
        n_total_designs: int,
        is_final: bool = False,
    ) -> list[str]:
        """Render heatmap and loss table for a single (replicate, network) design."""
        xres, yres = self._grid_resolution  # type: ignore
        net_name = self._network_names[nid] if nid < len(self._network_names) else f"Net {nid}"

        Y_pred_grid = None
        params = step_history.get('latest_params')
        if self.use_fresh_predictions and params is not None:
            Y_pred_grid = self._compute_fresh_prediction(params, rid, tid, nid)

        if Y_pred_grid is None:
            Y_pred_grid = yhatdep[rid, :, tid, nid].reshape(yres, xres)

        tu_stats = step_history.get("tu_stats", {})
        tu_str = ""
        if tu_stats:
            # Use per-network counts for accurate display
            enabled_per_net = tu_stats.get("enabled_count_per_network")
            n_tus = tu_stats.get("n_tus", tu_stats.get("total_count", 1))
            if enabled_per_net is not None:
                arr = np.asarray(enabled_per_net)
                if arr.ndim == 4:  # (n_replicates, n_batches, n_targets, n_networks)
                    rid_safe = min(rid, arr.shape[0] - 1)
                    tid_safe = min(tid, arr.shape[2] - 1)
                    nid_safe = min(nid, arr.shape[3] - 1)
                    enabled = float(arr[rid_safe, -1, tid_safe, nid_safe])
                elif arr.ndim == 3:  # (n_replicates, n_targets, n_networks)
                    rid_safe = min(rid, arr.shape[0] - 1)
                    tid_safe = min(tid, arr.shape[1] - 1)
                    nid_safe = min(nid, arr.shape[2] - 1)
                    enabled = float(arr[rid_safe, tid_safe, nid_safe])
                elif arr.ndim == 2:  # (n_targets, n_networks)
                    enabled = float(arr[min(tid, arr.shape[0] - 1), min(nid, arr.shape[1] - 1)])
                elif arr.ndim == 1:  # (n_networks,)
                    enabled = float(arr[min(nid, len(arr) - 1)])
                else:  # scalar (ndim == 0)
                    enabled = float(arr)
            else:
                enabled = float(np.asarray(tu_stats.get("enabled_count", 0)))
            n_tus_arr = np.asarray(n_tus)
            if n_tus_arr.ndim == 0:
                total = float(n_tus_arr)
            elif n_tus_arr.ndim == 1:
                total = float(n_tus_arr[min(nid, len(n_tus_arr) - 1)])
            else:
                total = float(n_tus_arr.flat[0])
            tu_pct = 100 * enabled / max(total, 1)
            tu_str = f" │ TUs: {int(enabled)}/{int(total)} ({tu_pct:.0f}%)"

        fp_str = ""
        compute_fp = is_final or not self.fingerprint_only_at_end
        params = step_history.get('latest_params')
        if compute_fp and self._model and self._dmanager and params is not None:
            try:
                from biocomp.fingerprint import compute_fingerprint_from_params

                stack = self._dmanager.build_stack(self._model)
                fingerprint = compute_fingerprint_from_params(
                    stack=stack,
                    params=params,
                    model=self._model,
                    rep_id=rid,
                    target_id=tid,
                    network_idx=nid,
                )
                fp_str = f" │ FP: {fingerprint}"
            except Exception as e:
                logger.debug(f"Fingerprint computation failed: {e}")

        header = f" Rep {rid} {net_name} (rank {rank + 1}/{n_total_designs}, loss={loss:.4f}){tu_str}{fp_str}"

        def _extract_scalar(val, rid, nid):
            """Extract scalar from potentially multi-dim array using (rid, nid) indices."""
            if val is None or (isinstance(val, (int, float)) and val == 0.0):
                return 0.0
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            elif arr.ndim == 1:
                return float(arr[min(nid, len(arr) - 1)])
            elif arr.ndim == 2:
                return float(arr[min(rid, arr.shape[0] - 1), min(nid, arr.shape[1] - 1)])
            else:
                return float(np.mean(arr))

        penalties = {
            "l0_penalty": _extract_at_indices(
                step_history.get("l0_penalty_per_network"), rid, tid, nid
            ),
            "spread_penalty": _extract_scalar(step_history.get("spread_penalty", 0.0), rid, nid),
            "coupling_penalty": _extract_scalar(
                step_history.get("coupling_penalty", 0.0), rid, nid
            ),
            "tucount_penalty": _extract_scalar(step_history.get("tucount_penalty", 0.0), rid, nid),
            "ern_tying_penalty": _extract_scalar(
                step_history.get("ern_tying_penalty", 0.0), rid, nid
            ),
        }

        width = self.xres * 2 + 13

        lines = [f"{'─' * width}", header, f"{'─' * width}"]

        if Y_target_grid is not None:
            Y_target_unflipped = np.flipud(Y_target_grid)
            txt_output, metrics = side_by_side_txt_plot(
                Y_target_unflipped,
                Y_pred_grid,
                height=self.yres,
                width=self.xres,
                loss_weights=self._loss_weights,
                training_penalties=penalties if self.show_penalties else None,
                shared_colorbar=False,
                show_axes=True,
                compute_metrics=True,
            )
            lines.append(txt_output)

            if self.show_stats:
                pred_range = metrics.get("pred_range", (0, 0))
                corr = metrics.get("correlation", 0.0)
                lines.append(f"Pred: [{pred_range[0]:.2f}, {pred_range[1]:.2f}] │ Corr: {corr:.4f}")
        else:
            Y_pred_flipped = np.flipud(Y_pred_grid)
            pred_heatmap = heatmap(
                Y_pred_flipped,
                xres=self.xres,
                yres=self.yres,
                show_colorbar=True,
            )
            lines.append(pred_heatmap)

            if self.show_stats:
                lines.append(f"Pred: [{Y_pred_grid.min():.2f}, {Y_pred_grid.max():.2f}]")

        if self.show_local_params:
            if params is None:
                logger.warning(
                    f"[DesignHeatmapLogger] Cannot show params: latest_params not in step_history. "
                    f"Keys available: {list(step_history.keys())}"
                )
                lines.append("")
                lines.append("[!] Cannot show params: latest_params not in step_history")
            elif self._stack is None:
                logger.warning(
                    f"[DesignHeatmapLogger] Cannot introspect: stack not built. "
                    f"model={self._model is not None}, dmanager={self._dmanager is not None}"
                )
                lines.append("")
                lines.append(
                    f"[!] Cannot introspect: stack not built "
                    f"(model={self._model is not None}, dmanager={self._dmanager is not None})"
                )
            else:
                try:
                    from biocomp.jaxutils import tree_get
                    from biocomp.paramintrospect import format_network_params_rich
                    from io import StringIO
                    from rich.console import Console

                    specific_params = tree_get(params, (rid, tid))

                    string_io = StringIO()
                    console = Console(file=string_io, force_terminal=True, width=100)
                    format_network_params_rich(self._stack, specific_params, nid, console)
                    param_str = string_io.getvalue()

                    if param_str:
                        lines.append("")
                        for line in param_str.split("\n"):
                            lines.append(f"  {line}")
                    else:
                        logger.warning(
                            f"[DesignHeatmapLogger] format_network_params_rich returned empty for network {nid}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[DesignHeatmapLogger] Introspection failed for network {nid}: {e}"
                    )
                    lines.append("")
                    lines.append(f"[!] Introspection failed for network {nid}: {e}")

        return lines

    def _render_heatmaps(self, step: int, step_history: dict, is_final: bool = False) -> str | None:
        if self._grid_resolution is None or self._dmanager is None:
            return None

        yhatdep = step_history.get("yhatdep")
        if yhatdep is None:
            return None

        yhatdep = np.asarray(yhatdep)
        xres, yres = self._grid_resolution

        # Normalize yhatdep to shape: (n_replicates, batch_size, n_targets, n_networks)
        # Input can be 3D, 4D, or 5D
        if yhatdep.ndim == 5:
            # (n_replicates, n_batches, batch_size, n_targets, n_networks) -> take last batch
            yhatdep = yhatdep[:, -1, :, :, :]
        elif yhatdep.ndim == 4:
            # Could be (n_replicates, batch_size, n_targets, n_networks) - already correct
            # OR (n_batches, batch_size, n_targets, n_networks) if no replicates
            # Check if first dim looks like replicates (small) vs batch_size (large)
            if yhatdep.shape[1] == xres * yres:
                pass  # Already (n_replicates, batch_size, n_targets, n_networks)
            else:
                # Assume single replicate: (n_batches, batch_size, n_targets, n_networks)
                yhatdep = yhatdep[-1:, :, :, :]  # Take last batch, add replicate dim
        elif yhatdep.ndim == 3:
            # (batch_size, n_targets, n_networks) -> add replicate dim
            yhatdep = yhatdep[np.newaxis, :, :, :]
        else:
            return None

        n_replicates, batch_size, n_targets, n_networks = yhatdep.shape
        if batch_size != xres * yres:
            return None

        tid = min(self.target_idx, n_targets - 1)
        Y_target_grid = (
            np.flipud(self._cached_target_grid) if self._cached_target_grid is not None else None
        )

        n_total_designs = n_replicates * n_networks
        if self.network_idx is not None:
            # Specific network requested - show all replicates for that network
            top_designs = [(r, self.network_idx, float('nan')) for r in range(n_replicates)]
        else:
            top_designs = self._get_top_k_designs(step_history, tid, n_replicates, n_networks)

        width = self.xres * 2 + 8
        all_lines = [f"{'═' * width}"]

        progress = step / max(self._total_steps, 1) * 100
        target_str = f" │ Target {tid}/{self._n_targets}" if self._n_targets > 1 else ""
        main_header = f" Step {step}/{self._total_steps} ({progress:.1f}%){target_str} │ Top {len(top_designs)} of {n_total_designs} designs"
        all_lines.append(main_header)
        all_lines.append(f"{'═' * width}")

        for rank, (rid, nid, loss) in enumerate(top_designs):
            rid = min(rid, n_replicates - 1)
            nid = min(nid, n_networks - 1)
            design_lines = self._render_single_design(
                step,
                step_history,
                yhatdep,
                tid,
                rid,
                nid,
                rank,
                loss,
                Y_target_grid,
                n_total_designs,
                is_final=is_final,
            )
            all_lines.extend(design_lines)
            all_lines.append("")

        if self.show_ratio_stats:
            ratio_stats = step_history.get("ratio_stats", {})
            if ratio_stats:
                r_min = _to_scalar(ratio_stats.get("min", 0.0))
                r_max = _to_scalar(ratio_stats.get("max", 0.0))
                r_mean = _to_scalar(ratio_stats.get("mean", 0.0))
                all_lines.append(f"Ratios: [{r_min:.2f}, {r_max:.2f}] μ={r_mean:.2f}")

        return '\n'.join(all_lines)

    def _save_reproduction_pickle(self, step: int, step_history: dict):
        """Save pickle with data needed to reproduce logged vs committed predictions."""
        if not self.save_reproduction_pickle:
            return

        import pickle
        from pathlib import Path
        from copy import deepcopy
        from biocomp.jaxutils import tree_get, tree_to_np

        if self._dmanager is None or self._model is None:
            logger.warning("Cannot save reproduction pickle: dmanager or model not available")
            return

        params = step_history.get("latest_params")
        yhatdep = step_history.get("yhatdep")
        if params is None or yhatdep is None:
            logger.warning("Cannot save reproduction pickle: params or yhatdep not in step_history")
            return

        yhatdep = np.asarray(yhatdep)
        xres, yres = self._grid_resolution or (32, 32)

        if yhatdep.ndim == 5:
            yhatdep = yhatdep[:, -1, :, :, :]
        elif yhatdep.ndim == 4:
            if yhatdep.shape[1] != xres * yres:
                yhatdep = yhatdep[-1:, :, :, :]
        elif yhatdep.ndim == 3:
            yhatdep = yhatdep[np.newaxis, :, :, :]

        n_replicates, batch_size, n_targets, n_networks = yhatdep.shape

        top_designs = self._get_top_k_designs(
            step_history, tid=0, n_replicates=n_replicates, n_networks=n_networks
        )

        needed_pairs: set[tuple[int, int]] = set()
        for rep_id, _net_id, _ in top_designs:
            for tid in range(n_targets):
                needed_pairs.add((rep_id, tid))

        committed_networks: dict[tuple[int, int], list] = {}
        stack = self._dmanager.build_stack(self._model)

        logger.info(f"Reproduction pickle: Committing {len(needed_pairs)} (rep, target) pairs...")
        for rep_id, tid in sorted(needed_pairs):
            try:
                specific_params = tree_get(params, (rep_id, tid))
                committed = stack.commit(specific_params)
                committed_networks[(rep_id, tid)] = deepcopy(committed)
            except Exception as e:
                logger.warning(
                    f"Reproduction pickle: Commit failed for (rep={rep_id}, tid={tid}): {e}"
                )
                committed_networks[(rep_id, tid)] = []

        target_grids = {}
        for tid, target in enumerate(self._dmanager.targets):
            try:
                _, Y_grid = target.get_lattice(resolution=self._grid_resolution, seed=0)
                target_grids[tid] = np.asarray(Y_grid)
            except Exception as e:
                logger.warning(f"Reproduction pickle: Failed to get target grid for tid={tid}: {e}")

        repro_data = {
            "step": step,
            "latest_params": tree_to_np(params),
            "yhatdep": yhatdep,
            "networks": deepcopy(self._dmanager.networks),
            "committed_networks": committed_networks,
            "tu_id_to_idx": getattr(stack, "tu_id_to_idx", None),
            "grid_resolution": self._grid_resolution,
            "targets": self._dmanager.targets,
            "model_path": self._model_path,
            "model_signature": self._model.signature if hasattr(self._model, 'signature') else None,
            "top_designs": top_designs,
            "loss_weights": self._loss_weights,
            "target_grids": target_grids,
            "n_replicates": n_replicates,
            "n_targets": n_targets,
            "n_networks": n_networks,
        }

        if self.reproduction_pickle_path:
            pickle_path = Path(self.reproduction_pickle_path)
        elif self._output_dir:
            pickle_path = Path(self._output_dir) / "heatmap_repro.pickle"
        else:
            pickle_path = Path("heatmap_repro.pickle")

        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(repro_data, f)

        logger.info(f"Reproduction pickle: Saved reproduction data to {pickle_path}")
        logger.info(
            f"Reproduction pickle: Contains {len(top_designs)} top designs, "
            f"{len(committed_networks)} committed network sets"
        )

    def get_callbacks(self, training_program=None):
        def callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None:
                return
            output = self._render_heatmaps(step, step_history, is_final=False)
            if output:
                print(output)
                print()

        def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None:
                return
            output = self._render_heatmaps(step, step_history, is_final=True)
            if output:
                print("\n" + "═" * 60)
                print(" FINAL RESULT ".center(60, "═"))
                print("═" * 60)
                print(output)
                print()

            if self.save_reproduction_pickle:
                self._save_reproduction_pickle(step, step_history)

        return [(self.periods, callback), (-1, final_callback)]

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        return None

    def finalize(self):
        pass
