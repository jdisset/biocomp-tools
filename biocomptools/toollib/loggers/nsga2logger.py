# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""NSGA2 Design Logger: tracks multi-objective optimization with pareto front visualization."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class NSGA2DesignLogger(Logger):
    """Logger for NSGA2 multi-objective design optimization.

    Tracks:
    - Pareto front evolution (pattern loss vs TU count)
    - Best designs at each generation
    - ASCII visualization of pareto front and top predictions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: str | None = None
    print_pareto: bool = Field(default=True, description="Print ASCII pareto front")
    print_predictions: bool = Field(default=True, description="Print ASCII predictions for top designs")
    save_pareto: bool = Field(default=True, description="Save pareto front data")
    prediction_resolution: tuple[int, int] = Field(default=(48, 24), description="Resolution for prediction plots")
    n_top_to_visualize: int = Field(default=3, description="Number of top designs to visualize")

    _history: list[dict[str, Any]] = []
    _save_dir: Path | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []

    def initialize(self, training_program=None):
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / 'nsga2_logs'
        else:
            self._save_dir = Path('nsga2_logs')

        self._save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"NSGA2DesignLogger initialized: {self._save_dir}")

    def _format_pareto_ascii(
        self,
        pareto_fitness: np.ndarray,
        width: int = 60,
        height: int = 20,
    ) -> str:
        """Create ASCII scatter plot of pareto front (loss vs TU count)."""
        if pareto_fitness is None or len(pareto_fitness) == 0:
            return "  [No pareto front data]"

        losses = pareto_fitness[:, 0]
        tu_counts = pareto_fitness[:, 1]

        loss_min, loss_max = float(np.min(losses)), float(np.max(losses))
        tu_min, tu_max = float(np.min(tu_counts)), float(np.max(tu_counts))

        loss_range = max(loss_max - loss_min, 1e-6)
        tu_range = max(tu_max - tu_min, 1)

        grid = [[' ' for _ in range(width)] for _ in range(height)]

        for loss, tu in zip(losses, tu_counts, strict=False):
            x = int((loss - loss_min) / loss_range * (width - 1))
            y = int((tu - tu_min) / tu_range * (height - 1))
            y = height - 1 - y
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            grid[y][x] = '●'

        lines = []
        lines.append(f"  Pareto Front ({len(pareto_fitness)} solutions)")
        lines.append(f"  {'─' * (width + 2)}")

        tu_labels = np.linspace(tu_max, tu_min, min(5, height))
        label_positions = np.linspace(0, height - 1, len(tu_labels)).astype(int)

        for i, row in enumerate(grid):
            label = ""
            if i in label_positions:
                idx = list(label_positions).index(i)
                label = f"{tu_labels[idx]:>4.0f}"
            else:
                label = "    "
            lines.append(f"{label} │{''.join(row)}│")

        lines.append(f"     └{'─' * width}┘")

        loss_labels = [f"{v:.3f}" for v in np.linspace(loss_min, loss_max, 5)]
        loss_label_line = "      " + loss_labels[0].ljust(width // 4)
        for lbl in loss_labels[1:-1]:
            loss_label_line += lbl.center(width // 4)
        loss_label_line += loss_labels[-1].rjust(width // 4)
        lines.append(loss_label_line)

        lines.append("      " + "Pattern Loss ->".center(width))
        lines.append("  ↑ TU Count")

        return "\n".join(lines)

    def _format_prediction_ascii(
        self,
        prediction: np.ndarray,
        target: np.ndarray | None = None,
        title: str = "",
        xres: int = 48,
        yres: int = 24,
    ) -> str:
        """Create ASCII heatmap of prediction grid."""
        from biocomp.plotting.ascii_heatmap import heatmap_with_labels

        pred_grid = prediction
        if pred_grid.ndim == 1:
            side = int(np.sqrt(len(pred_grid)))
            if side * side == len(pred_grid):
                pred_grid = pred_grid.reshape(side, side)
            else:
                return f"  [Cannot reshape prediction of len {len(pred_grid)}]"

        pred_grid = np.flipud(pred_grid)

        lines = []
        if title:
            lines.append(f"  {title}")
            lines.append("")

        hm = heatmap_with_labels(
            pred_grid,
            xlabel="x",
            ylabel="y",
            xres=min(xres, pred_grid.shape[1]),
            yres=min(yres, pred_grid.shape[0]),
            show_colorbar=True,
        )
        lines.append(hm)

        if target is not None:
            target_grid = target
            if target_grid.ndim == 1:
                side = int(np.sqrt(len(target_grid)))
                if side * side == len(target_grid):
                    target_grid = target_grid.reshape(side, side)

            if target_grid.ndim == 2:
                lines.append("")
                lines.append("  Target:")
                target_grid = np.flipud(target_grid)
                hm_target = heatmap_with_labels(
                    target_grid,
                    xlabel="x",
                    ylabel="y",
                    xres=min(xres, target_grid.shape[1]),
                    yres=min(yres, target_grid.shape[0]),
                    show_colorbar=True,
                )
                lines.append(hm_target)

        return "\n".join(lines)

    def _extract_metrics(self, step_history: dict) -> dict[str, Any]:
        """Extract NSGA2-specific metrics from step_history."""
        metrics = {"step": step_history.get("step", -1)}

        for key in [
            "gen_best_loss",
            "gen_best_tu_count",
            "gen_mean_loss",
            "gen_mean_tu_count",
            "pareto_size",
            "pareto_min_loss",
            "pareto_min_tu",
        ]:
            val = step_history.get(key)
            if val is not None:
                metrics[key] = float(val) if hasattr(val, 'item') else val

        return metrics

    def _print_generation_summary(self, step: int, metrics: dict, pareto_fitness: np.ndarray | None):
        """Print a one-line generation summary."""
        best_loss = metrics.get("gen_best_loss", float("nan"))
        best_tu = metrics.get("gen_best_tu_count", float("nan"))
        pareto_size = metrics.get("pareto_size", 0)
        pareto_min_loss = metrics.get("pareto_min_loss", float("nan"))

        logger.info(
            f"Gen {step:4d} | best_loss={best_loss:.4f} TU={best_tu:.0f} | "
            f"pareto_size={pareto_size} min_loss={pareto_min_loss:.4f}"
        )

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        step_history = view.to_step_history()

        metrics = self._extract_metrics(step_history)
        metrics["step"] = step
        self._history.append(metrics)

        pareto_fitness = step_history.get("pareto_fitness")
        if pareto_fitness is not None:
            pareto_fitness = np.asarray(pareto_fitness)

        self._print_generation_summary(step, metrics, pareto_fitness)

        if self.print_pareto and pareto_fitness is not None and step % 10 == 0:
            print(self._format_pareto_ascii(pareto_fitness))

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        batch = view.latest()
        if batch is None:
            return
        step_history = view.to_step_history()

        pareto_fitness = step_history.get("pareto_fitness")
        pareto_front = step_history.get("pareto_front")

        if pareto_fitness is not None:
            pareto_fitness = np.asarray(pareto_fitness)
            print("\n" + "=" * 60)
            print("FINAL PARETO FRONT")
            print("=" * 60)
            print(self._format_pareto_ascii(pareto_fitness, width=70, height=25))

            sorted_idx = np.argsort(pareto_fitness[:, 0])
            print("\nTop solutions by pattern loss:")
            print("-" * 50)
            for i, idx in enumerate(sorted_idx[:10]):
                loss, tu = pareto_fitness[idx]
                print(f"  {i + 1:2d}. Loss={loss:.4f}, TU_count={tu:.0f}")

        if self._save_dir and pareto_fitness is not None:
            self._save_pareto_data(pareto_front, pareto_fitness)

        if self.print_predictions and step_history.get("yhatdep") is not None:
            self._print_top_predictions(step_history, pareto_fitness)

    def _save_pareto_data(self, pareto_front: np.ndarray | None, pareto_fitness: np.ndarray):
        """Save pareto front data to files."""
        if self._save_dir is None:
            return

        np.savez_compressed(
            self._save_dir / "pareto_front.npz",
            population=np.asarray(pareto_front) if pareto_front is not None else np.array([]),
            fitness=pareto_fitness,
        )

        summary = {
            "n_solutions": int(len(pareto_fitness)),
            "min_loss": float(np.min(pareto_fitness[:, 0])),
            "max_loss": float(np.max(pareto_fitness[:, 0])),
            "min_tu_count": float(np.min(pareto_fitness[:, 1])),
            "max_tu_count": float(np.max(pareto_fitness[:, 1])),
            "solutions": [
                {"loss": float(f[0]), "tu_count": float(f[1])}
                for f in pareto_fitness
            ],
        }

        (self._save_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved pareto data to {self._save_dir}")

    def _print_top_predictions(self, step_history: dict, pareto_fitness: np.ndarray | None):
        """Print ASCII predictions for top designs."""
        yhatdep = step_history.get("yhatdep")
        if yhatdep is None:
            return

        yhatdep = np.asarray(yhatdep)
        n_to_show = min(self.n_top_to_visualize, len(yhatdep))

        print("\n" + "=" * 60)
        print(f"TOP {n_to_show} DESIGN PREDICTIONS")
        print("=" * 60)

        for i in range(n_to_show):
            pred = yhatdep[i] if yhatdep.ndim > 1 else yhatdep
            loss = pareto_fitness[i, 0] if pareto_fitness is not None else float("nan")
            tu_count = pareto_fitness[i, 1] if pareto_fitness is not None else float("nan")

            title = f"Design {i + 1}: loss={loss:.4f}, TU={tu_count:.0f}"
            xres, yres = self.prediction_resolution
            print(self._format_prediction_ascii(pred, title=title, xres=xres, yres=yres))
            print()

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        if not self._history:
            return None
        return {
            "generations_tracked": len(self._history),
            "final_best_loss": self._history[-1].get("gen_best_loss"),
            "final_pareto_size": self._history[-1].get("pareto_size"),
        }

    def finalize(self):
        logger.info(f"NSGA2DesignLogger finalized with {len(self._history)} generations")

        if self._save_dir and self._history:
            (self._save_dir / "history.json").write_text(
                json.dumps(self._history, indent=2, default=str)
            )
