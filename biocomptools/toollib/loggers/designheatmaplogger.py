"""Design Heatmap Logger: Rich ASCII visualization of target vs prediction during optimization."""

import numpy as np
from typing import Any

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomp.plotting.ascii_heatmap import heatmap

logger = get_logger(__name__)

LOSS_ORDER = [
    "sinkhorn",
    "lncc",
    "rmse",
    "mse",
    "simse",
    "spectral",
    "gradient",
    "contrast",
    "zncc",
]
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
    if hasattr(val, 'shape') and getattr(val, 'size', 1) > 1:
        return float(np.mean(val))
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


def _side_by_side(left: str, right: str, gap: int = 4) -> str:
    """Join two multi-line strings side by side."""
    left_lines = left.split('\n')
    right_lines = right.split('\n')
    max_left = max(len(line) for line in left_lines) if left_lines else 0
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [''] * (max_lines - len(left_lines))
    right_lines += [''] * (max_lines - len(right_lines))
    return '\n'.join(
        f"{left_line:<{max_left}}{' ' * gap}{right_line}"
        for left_line, right_line in zip(left_lines, right_lines, strict=True)
    )


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
    top_k: int = Field(default=3, description="Number of top networks to display (by loss)")

    _dmanager: Any = None
    _grid_resolution: tuple[int, int] | None = None
    _cached_target_grid: np.ndarray | None = None
    _total_steps: int = 0
    _loss_weights: dict = {}
    _n_targets: int = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dmanager = None
        self._grid_resolution = None
        self._cached_target_grid = None
        self._total_steps = 0
        self._loss_weights = {}
        self._n_targets = 1

    def initialize(self, training_program=None):
        if training_program:
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

            if hasattr(training_program, 'design_conf'):
                dc = training_program.design_conf
                if hasattr(dc, 'n_epochs') and hasattr(dc, 'n_batches_per_epoch'):
                    self._total_steps = dc.n_epochs * dc.n_batches_per_epoch

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
                    if hasattr(dc, attr):
                        self._loss_weights[attr] = getattr(dc, attr)

        if self._dmanager and self._grid_resolution:
            targets = self._dmanager.targets
            if self.target_idx < len(targets):
                target = targets[self.target_idx]
                try:
                    _, Y_grid = target.get_lattice(resolution=self._grid_resolution, seed=0)
                    self._cached_target_grid = Y_grid
                except Exception as e:
                    logger.warning(f"Failed to cache target grid: {e}")

    def _compute_eval_losses(self, Y_target_grid: np.ndarray, Y_pred_grid: np.ndarray) -> dict:
        """Compute eval-mode grid losses via compute_grid_losses()."""
        try:
            import jax.numpy as jnp
            from biocomp.designloss import compute_grid_losses

            result = compute_grid_losses(
                jnp.array(Y_pred_grid),
                jnp.array(Y_target_grid),
                w_sinkhorn=1.0,
                w_lncc=1.0,
                w_mse=1.0,
                w_rmse=1.0,
                w_simse=1.0,
                w_spectral=1.0,
                w_gradient=1.0,
                w_contrast=1.0,
            )
            return result.to_dict()
        except Exception as e:
            logger.warning(f"Failed to compute eval losses: {e}")
            return {}

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
    ) -> list[str]:
        """Render heatmap and loss table for a single (replicate, network) design."""
        xres, yres = self._grid_resolution  # type: ignore

        # yhatdep shape: (n_replicates, batch_size, n_targets, n_networks)
        Y_pred_grid = np.flipud(yhatdep[rid, :, tid, nid].reshape(yres, xres))

        vmin = min(
            Y_pred_grid.min(),
            Y_target_grid.min() if Y_target_grid is not None else Y_pred_grid.min(),
        )
        vmax = max(
            Y_pred_grid.max(),
            Y_target_grid.max() if Y_target_grid is not None else Y_pred_grid.max(),
        )

        pred_heatmap = heatmap(
            Y_pred_grid, vmin=vmin, vmax=vmax, xres=self.xres, yres=self.yres, show_colorbar=False
        )
        if Y_target_grid is not None:
            target_heatmap = heatmap(
                Y_target_grid,
                vmin=vmin,
                vmax=vmax,
                xres=self.xres,
                yres=self.yres,
                show_colorbar=False,
            )
            combined = _side_by_side(target_heatmap, pred_heatmap, gap=4)
        else:
            combined = pred_heatmap

        width = self.xres * 2 + 8

        is_nsga2 = (
            step_history.get("gen_best_loss") is not None
            or step_history.get("pareto_fitness") is not None
        )

        tu_stats = step_history.get("tu_stats", {})
        tu_str = ""
        if tu_stats:
            enabled = tu_stats.get("enabled_count", 0)
            total = tu_stats.get("total_count", 1)
            if hasattr(enabled, 'shape') and enabled.shape:
                enabled = float(
                    enabled[min(nid, len(enabled) - 1)] if enabled.ndim == 1 else np.mean(enabled)
                )
            if hasattr(total, 'shape') and total.shape:
                total = float(
                    total[min(nid, len(total) - 1)] if total.ndim == 1 else np.mean(total)
                )
            tu_pct = 100 * float(enabled) / max(float(total), 1)
            tu_str = f" │ TUs: {int(enabled)}/{int(total)} ({tu_pct:.0f}%)"

        if is_nsga2:
            header = (
                f" Rep {rid} Net {nid} (rank {rank + 1}/{n_total_designs}, loss={loss:.4f}){tu_str}"
            )
        else:
            header = (
                f" Rep {rid} Net {nid} (rank {rank + 1}/{n_total_designs}, loss={loss:.4f}){tu_str}"
            )

        lines = [
            f"{'─' * width}",
            header,
            f"{'─' * width}",
            f"{'TARGET':^{self.xres}}    {'PREDICTION':^{self.xres}}",
            combined,
            f"{vmin:.2f} {'░▒▓█' * (self.xres // 4)} {vmax:.2f}".center(width),
        ]

        sublosses = step_history.get("sublosses", {})
        penalties = {
            "l0_penalty": step_history.get("l0_penalty", 0.0),
            "spread_penalty": step_history.get("spread_penalty", 0.0),
            "coupling_penalty": step_history.get("coupling_penalty", 0.0),
            "tucount_penalty": step_history.get("tucount_penalty", 0.0),
            "ern_tying_penalty": step_history.get("ern_tying_penalty", 0.0),
        }

        lines.append("")
        if sublosses:
            table_lines = _format_loss_table_per_design(
                sublosses, penalties, self._loss_weights, rid, tid, nid, self.show_penalties
            )
            lines.extend(table_lines)
        elif Y_target_grid is not None:
            Y_pred_unflipped = np.flipud(Y_pred_grid)
            Y_target_unflipped = np.flipud(Y_target_grid)
            eval_losses = self._compute_eval_losses(Y_target_unflipped, Y_pred_unflipped)
            if eval_losses:
                table_lines = _format_loss_table_eval_only(
                    eval_losses, penalties, self._loss_weights, self.show_penalties
                )
                lines.extend(table_lines)

        if self.show_stats:
            footer_parts = [f"Pred: [{Y_pred_grid.min():.2f}, {Y_pred_grid.max():.2f}]"]
            if Y_target_grid is not None:
                corr = np.corrcoef(Y_target_grid.ravel(), Y_pred_grid.ravel())[0, 1]
                footer_parts.append(f"Corr: {corr:.4f}")
            lines.append(" │ ".join(footer_parts))

        return lines

    def _render_heatmaps(self, step: int, step_history: dict) -> str | None:
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

    def get_callbacks(self, training_program=None):
        def callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None:
                return
            output = self._render_heatmaps(step, step_history)
            if output:
                print(output)
                print()

        def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None:
                return
            output = self._render_heatmaps(step, step_history)
            if output:
                print("\n" + "═" * 60)
                print(" FINAL RESULT ".center(60, "═"))
                print("═" * 60)
                print(output)
                print()

        return [(self.periods, callback), (-1, final_callback)]

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        return None

    def finalize(self):
        pass
