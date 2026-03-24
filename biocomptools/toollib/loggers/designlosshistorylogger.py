"""Design Loss History Logger: single-panel loss history for the best network.

Plots total loss, weighted sublosses, and active penalties for the best-performing
network at each step (the one with minimum total loss), matching what the final
winning design actually experienced.
"""

import numpy as np
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.utils import (
    PENALTY_NAMES,
    extract_best_network_metrics,
    has_nonzero,
    rolling_mean,
)
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DesignLossHistoryLogger(Logger):
    """Single-panel loss history for the best network at each step.

    Plots total loss, weighted sublosses (weight > 0), and active penalties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_window: int | None = 1
    required_metrics: list[str] = ["loss", "sublosses", "all_losses", *PENALTY_NAMES]

    output_dir: str | None = None
    smoothing_window: int = Field(default=1, description="Rolling mean window (1 = no smoothing)")
    plot_interval: int = Field(default=500, description="Generate interim plots every N steps")
    phase_boundaries: list[float] | None = Field(
        default=None,
        description="Phase boundary fractions (e.g. [0.4, 0.75]). Auto-detected if None.",
    )
    dpi: int = Field(default=150, description="Figure DPI")

    _history: list[dict[str, Any]] = []
    _save_dir: Path | None = None
    _total_steps_hint: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history = []
        self._save_dir = None
        self._total_steps_hint = None

    def initialize(self, training_program: object = None) -> None:
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, "_save_dir"):
            self._save_dir = training_program._save_dir / "loss_history"
        else:
            self._save_dir = Path("loss_history")
        self._save_dir.mkdir(parents=True, exist_ok=True)

        if training_program and hasattr(training_program, "design_conf"):
            dc = training_program.design_conf
            if hasattr(dc, "phase1_frac") and hasattr(dc, "phase2_frac"):
                self.phase_boundaries = [dc.phase1_frac, dc.phase2_frac]
            if hasattr(dc, "n_steps"):
                self._total_steps_hint = dc.n_steps

        logger.info(f"DesignLossHistoryLogger initialized: {self._save_dir}")

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        step_history = view.to_step_history()
        metrics = extract_best_network_metrics(step_history)
        metrics["step"] = step
        self._history.append(metrics)

        if (
            self._save_dir
            and self.plot_interval > 0
            and step > 0
            and step % self.plot_interval == 0
        ):
            self._generate_figure(
                self._save_dir / f"loss_history_step{step:06d}.png",
                title_suffix=f" (step {step})",
            )

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        step_history = view.to_step_history()
        metrics = extract_best_network_metrics(step_history)
        metrics["step"] = step
        if not self._history or self._history[-1]["step"] != step:
            self._history.append(metrics)

        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._generate_figure(
                self._save_dir / "loss_history_final.png",
                title_suffix=" (final)",
            )
            logger.info(
                f"DesignLossHistoryLogger: saved final figure "
                f"({len(self._history)} steps) to {self._save_dir}"
            )

    def _generate_figure(self, output_path: Path, title_suffix: str = "") -> None:
        if not self._history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping loss history figure")
            return

        steps = [h["step"] for h in self._history]
        sw = self.smoothing_window

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            f"Design Loss History — best network{title_suffix}", fontsize=13, fontweight="bold"
        )

        losses = [h.get("loss", np.nan) for h in self._history]
        ax.plot(steps, rolling_mean(losses, sw), "-", color="black", linewidth=2.0, label="total")

        weighted_keys = sorted(
            {
                k
                for h in self._history
                for k in h
                if k.startswith("subloss_") and k.endswith("_weighted")
            }
        )
        for key in weighted_keys:
            vals = [h.get(key, 0.0) for h in self._history]
            if not has_nonzero(vals):
                continue
            label = key.removeprefix("subloss_").removesuffix("_weighted")
            ax.plot(steps, rolling_mean(vals, sw), linewidth=1.3, label=label)

        for pname in PENALTY_NAMES:
            vals = [h.get(pname, 0.0) for h in self._history]
            if not has_nonzero(vals):
                continue
            label = pname.removesuffix("_penalty")
            ax.plot(steps, rolling_mean(vals, sw), "--", linewidth=1.3, label=label)

        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, ncol=2)

        self._draw_phase_boundaries(steps, [ax])

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _draw_phase_boundaries(self, steps: list[int], axes: list) -> None:
        boundaries = self.phase_boundaries
        if not boundaries or not steps:
            return
        max_step = max(steps)
        if max_step <= 0:
            return

        total = self._total_steps_hint or max_step
        for frac in boundaries:
            step_val = int(frac * total)
            if step_val <= 0 or step_val > max_step:
                continue
            for ax in axes:
                ax.axvline(step_val, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        if not self._history:
            return None
        return {
            "steps_tracked": len(self._history),
            "final_loss": self._history[-1].get("loss"),
        }

    def finalize(self) -> None:
        logger.info(f"DesignLossHistoryLogger finalized with {len(self._history)} entries")
