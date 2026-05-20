# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""CMA-ES Logger: EC-specific monitoring for evolutionary design optimization.

Tracks and visualizes:
- Generation-level loss (best and mean)
- CMA-ES sigma (step size) evolution
- Population validity statistics
- Convergence indicators
"""

from typing import Literal

import numpy as np

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class CMAESLogger(Logger):
    """Logger for CMA-ES evolutionary optimization with plotext terminal graphs.

    Displays EC-specific metrics: generation losses, sigma evolution, population validity.
    Uses plotext for inline terminal visualization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plot_height: int = Field(default=15, description="Height of plotext graphs")
    plot_width: int = Field(default=80, description="Width of plotext graphs")
    show_table: bool = Field(default=True, description="Show rich table with stats")
    show_plots: bool = Field(default=True, description="Show plotext graphs")
    execution_mode: Literal["inline", "thread", "process"] = "inline"

    _history: list[dict] = []
    _best_loss_ever: float = float("inf")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._best_loss_ever = float("inf")

    def _extract_ec_metrics(self, step: int, step_history: dict) -> dict:
        gen_best = step_history.get("gen_best_loss")
        gen_mean = step_history.get("gen_mean_loss")
        sigma = step_history.get("sigma")
        n_valid = step_history.get("n_valid")
        phase = step_history.get("phase")

        loss_data = step_history.get("loss")
        best_loss = (
            loss_data[0][0]
            if isinstance(loss_data, list) and loss_data
            else step_history.get("loss", float("nan"))
        )

        return {
            "step": step,
            "gen_best_loss": float(gen_best) if gen_best is not None else float("nan"),
            "gen_mean_loss": float(gen_mean) if gen_mean is not None else float("nan"),
            "sigma": float(sigma) if sigma is not None else float("nan"),
            "n_valid": int(n_valid) if n_valid is not None else 0,
            "phase": int(phase) if phase is not None else 0,
            "best_loss": float(best_loss) if best_loss is not None else float("nan"),
        }

    def _print_ec_stats(self, metrics: dict):
        from rich.console import Console
        from rich.table import Table

        console = Console()

        step = metrics["step"]
        gen_best = metrics["gen_best_loss"]
        gen_mean = metrics["gen_mean_loss"]
        sigma = metrics["sigma"]
        n_valid = metrics["n_valid"]
        best_loss = metrics["best_loss"]

        if best_loss < self._best_loss_ever:
            self._best_loss_ever = best_loss
            improved = True
        else:
            improved = False

        sigma_status = (
            "[green]healthy[/green]"
            if 0.01 < sigma < 1.0
            else "[yellow]low[/yellow]"
            if sigma <= 0.01
            else "[red]high[/red]"
        )

        table = Table(title=f"[bold cyan]CMA-ES Generation {step}[/bold cyan]", expand=False)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        improvement_pct = (
            (gen_mean - gen_best) / abs(gen_mean) * 100 if abs(gen_mean) > 1e-10 else 0
        )

        table.add_row(
            "Gen Best Loss",
            f"{gen_best:.6f}",
            "[green]↓[/green]" if improved else "",
        )
        table.add_row(
            "Gen Mean Loss",
            f"{gen_mean:.6f}",
            f"[dim]+{improvement_pct:.1f}% vs best[/dim]",
        )
        table.add_row(
            "Global Best",
            f"[bold green]{self._best_loss_ever:.6f}[/bold green]",
            "[green]★[/green]" if improved else "",
        )
        table.add_row("σ (Step Size)", f"{sigma:.6f}", sigma_status)
        table.add_row(
            "Valid Pop", f"{n_valid}", "[green]ok[/green]" if n_valid > 0 else "[red]x[/red]"
        )

        console.print(table)

    def _plot_ec_history(self):
        import plotext as plt

        if len(self._history) < 2:
            return

        steps = [h["step"] for h in self._history]
        gen_best = [h["gen_best_loss"] for h in self._history]
        gen_mean = [h["gen_mean_loss"] for h in self._history]
        sigmas = [h["sigma"] for h in self._history]
        best_losses = [h["best_loss"] for h in self._history]

        valid_gen_best = all(np.isfinite(v) for v in gen_best)
        valid_gen_mean = all(np.isfinite(v) for v in gen_mean)
        valid_sigmas = all(np.isfinite(v) for v in sigmas)
        valid_best = all(np.isfinite(v) for v in best_losses)

        plt.clf()
        plt.theme("dark")
        plt.subplots(1, 2)
        plt.plot_size(self.plot_width, self.plot_height)

        plt.subplot(1, 1)
        if valid_best:
            plt.plot(steps, best_losses, marker="braille", color="green", label="Global Best")
        if valid_gen_best:
            plt.plot(steps, gen_best, marker="braille", color="cyan", label="Gen Best")
        if valid_gen_mean:
            plt.plot(steps, gen_mean, marker="braille", color="yellow", label="Gen Mean")

        all_positive = all(v > 0 for v in gen_best + gen_mean + best_losses if np.isfinite(v))
        if all_positive:
            plt.yscale("log")
        plt.title("CMA-ES Loss Convergence")
        plt.xlabel("Generation")
        plt.ylabel("Loss")

        plt.subplot(1, 2)
        if valid_sigmas:
            plt.plot(steps, sigmas, marker="braille", color="magenta", label="σ")
            if all(s > 0 for s in sigmas):
                plt.yscale("log")
        plt.title("Step Size (σ) Evolution")
        plt.xlabel("Generation")
        plt.ylabel("σ")

        plt.show()

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step_history = view.to_step_history()
        if "sigma" not in step_history and "gen_best_loss" not in step_history:
            return

        metrics = self._extract_ec_metrics(context.current_step, step_history)
        self._history.append(metrics)

        if self.show_table:
            self._print_ec_stats(metrics)

        if self.show_plots:
            self._plot_ec_history()

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        if not self._history:
            return

        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        first = self._history[0]
        last = self._history[-1]

        summary_lines = [
            "[bold]CMA-ES Optimization Summary[/bold]",
            "",
            f"Total Generations: {len(self._history)}",
            f"Initial Loss: {first['gen_best_loss']:.6f}",
            f"Final Loss: {last['best_loss']:.6f}",
        ]

        if first["gen_best_loss"] > 0:
            improvement = (
                (first["gen_best_loss"] - last["best_loss"]) / first["gen_best_loss"] * 100
            )
            summary_lines.append(f"Improvement: {improvement:.1f}%")

        summary_lines.extend(
            [
                "",
                f"Final σ: {last['sigma']:.6f}",
                f"σ Range: [{min(h['sigma'] for h in self._history):.4f}, {max(h['sigma'] for h in self._history):.4f}]",
            ]
        )

        console.print(Panel("\n".join(summary_lines), title="[cyan]CMA-ES Complete[/cyan]"))

        if self.show_plots:
            self._plot_ec_history()

    def get_metrics(self, replicate: int | None = None) -> dict | None:
        if not self._history:
            return None
        return {
            "generations": len(self._history),
            "best_loss": self._best_loss_ever,
            "final_sigma": self._history[-1]["sigma"] if self._history else None,
        }

    def finalize(self):
        logger.info(f"CMAESLogger finalized: {len(self._history)} generations tracked")
