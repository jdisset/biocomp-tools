"""Design Heatmap Logger: ASCII visualization of target vs prediction during optimization."""

import numpy as np
from typing import Any

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomp.plotting.ascii_heatmap import heatmap

logger = get_logger(__name__)


def _side_by_side(left: str, right: str, gap: int = 4) -> str:
    """Join two multi-line strings side by side."""
    left_lines = left.split('\n')
    right_lines = right.split('\n')
    max_left = max(len(line) for line in left_lines) if left_lines else 0
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [''] * (max_lines - len(left_lines))
    right_lines += [''] * (max_lines - len(right_lines))
    return '\n'.join(
        f"{left:<{max_left}}{' ' * gap}{right}"
        for left, right in zip(left_lines, right_lines, strict=True)
    )


class DesignHeatmapLogger(Logger):
    """Logger that prints ASCII heatmaps of target vs prediction side-by-side."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async_ok: bool = False  # must be sync to print during optimization loop
    xres: int = Field(default=40, description="Horizontal resolution for ASCII heatmap")
    yres: int = Field(default=20, description="Vertical resolution for ASCII heatmap")
    show_stats: bool = Field(default=True, description="Show statistics below heatmaps")
    target_idx: int = Field(default=0, description="Which target to visualize")
    network_idx: int | None = Field(default=None, description="Which network (None=best)")

    _dmanager: Any = None
    _grid_resolution: tuple[int, int] | None = None
    _cached_target_grid: np.ndarray | None = None
    _total_steps: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dmanager = None
        self._grid_resolution = None
        self._cached_target_grid = None
        self._total_steps = 0

    def initialize(self, training_program=None):
        if training_program:
            if hasattr(training_program, '_dmanager'):
                self._dmanager = training_program._dmanager
                if self._dmanager and hasattr(self._dmanager, 'grid_resolution') and self._dmanager.grid_resolution:
                    self._grid_resolution = self._dmanager.grid_resolution
            if hasattr(training_program, 'design_conf'):
                dc = training_program.design_conf
                if hasattr(dc, 'n_epochs') and hasattr(dc, 'n_batches_per_epoch'):
                    self._total_steps = dc.n_epochs * dc.n_batches_per_epoch

        if self._dmanager and self._grid_resolution:
            targets = self._dmanager.targets
            if self.target_idx < len(targets):
                target = targets[self.target_idx]
                try:
                    _, Y_grid = target.get_lattice(resolution=self._grid_resolution, seed=0)
                    self._cached_target_grid = Y_grid
                except Exception as e:
                    logger.warning(f"Failed to cache target grid: {e}")

    def _render_heatmaps(self, step: int, step_history: dict) -> str | None:
        if self._grid_resolution is None or self._dmanager is None:
            return None

        yhatdep = step_history.get("yhatdep")
        if yhatdep is None:
            return None

        yhatdep = np.asarray(yhatdep)
        xres, yres = self._grid_resolution

        if yhatdep.ndim == 5:
            yhatdep = yhatdep[0, 0]
        elif yhatdep.ndim == 4:
            yhatdep = yhatdep[0]
        elif yhatdep.ndim != 3:
            return None

        batch_size, n_targets, n_networks = yhatdep.shape
        if batch_size != xres * yres:
            return None

        tid = min(self.target_idx, n_targets - 1)

        nid = self.network_idx
        if nid is None:
            all_losses = step_history.get("all_losses")
            if all_losses is not None:
                arr = np.asarray(all_losses)
                if arr.ndim >= 4:
                    net_losses = arr[0, 0, tid, :] if arr.ndim == 4 else arr[0, 0, 0, tid, :]
                    nid = int(np.argmin(net_losses))
                else:
                    nid = 0
            else:
                nid = 0

        nid = min(nid, n_networks - 1)

        Y_pred_grid = np.flipud(yhatdep[:, tid, nid].reshape(yres, xres))
        Y_target_grid = np.flipud(self._cached_target_grid) if self._cached_target_grid is not None else None

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

        progress = step / max(self._total_steps, 1) * 100
        loss = step_history.get("loss", float('nan'))
        loss_val = float(np.mean(np.asarray(loss))) if hasattr(loss, '__len__') else float(loss)

        lines = [
            f"{'─' * (self.xres * 2 + 8)}",
            f"Step {step}/{self._total_steps} ({progress:.1f}%) | Loss: {loss_val:.6f} | Net: {nid}",
            f"{'─' * (self.xres * 2 + 8)}",
            f"{'TARGET':^{self.xres}}    {'PREDICTION':^{self.xres}}",
            combined,
            f"{vmin:.2f} {'░▒▓█' * (self.xres // 4)} {vmax:.2f}",
        ]

        if self.show_stats and Y_target_grid is not None:
            corr = np.corrcoef(Y_target_grid.ravel(), Y_pred_grid.ravel())[0, 1]
            mse = np.mean((Y_pred_grid - Y_target_grid) ** 2)
            lines.append(
                f"Corr: {corr:.4f} | MSE: {mse:.6f} | Pred range: [{Y_pred_grid.min():.3f}, {Y_pred_grid.max():.3f}]"
            )

        tu_stats = step_history.get("tu_stats", {})
        if tu_stats:
            enabled = tu_stats.get("enabled_count", 0)
            total = tu_stats.get("total_count", 1)
            if hasattr(enabled, '__len__'):
                enabled = float(np.sum(np.asarray(enabled)))
            if hasattr(total, '__len__'):
                total = float(np.sum(np.asarray(total)))
            lines.append(
                f"TUs: {int(enabled)}/{int(total)} enabled ({100 * enabled / max(total, 1):.1f}%)"
            )

        return '\n'.join(lines)

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
                print("\n" + "=" * 60)
                print("FINAL RESULT")
                print("=" * 60)
                print(output)
                print()

        return [(self.periods, callback), (-1, final_callback)]

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        return None

    def finalize(self):
        pass
