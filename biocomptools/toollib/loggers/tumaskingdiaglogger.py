# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""TU Masking Diagnostic Logger: Per-network convergence diagnostics for TU masking optimization.

Tracks metrics identified by ML experts as diagnostic for TU masking issues:
- Mask entropy (exploration indicator) - per network
- Boundary TUs (0.3-0.7 probability) - per network
- Revival rate (TUs crossing floor upward) - per network
- Below-floor count (TUs in the "graveyard") - per network

See biocomp-doc/for-ml-collaborators/tu_masking_introduction.md for context.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections import deque
from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

L0_PENALTY_FLOOR_PROB = 0.2


@dataclass
class NetworkDiag:
    """Per-network TU diagnostics."""

    name: str
    entropy: float
    enabled: int
    n_tus: int
    boundary: int
    below_floor: int
    revivals: int
    prob_mean: float
    prob_std: float


@dataclass
class StepDiag:
    """Full step diagnostics with per-network breakdown."""

    step: int
    networks: list[NetworkDiag] = field(default_factory=list)
    total_entropy: float = 0.0
    total_enabled: int = 0
    total_tus: int = 0
    total_boundary: int = 0
    total_below_floor: int = 0
    total_revivals: int = 0
    revivals_last_100: int = 0
    no_tu_stats: bool = False


class TUMaskingDiagLogger(Logger):
    """TU masking convergence diagnostics with per-network breakdown.

    Tracks metrics the ML experts identified as diagnostic:
    - Mask entropy: High = exploring, Low = committed. <0.2 in Phase 1 = premature commitment
    - Boundary TUs: Count with prob in [0.3, 0.7]. <5% early = decisions too fast
    - Below-floor count: TUs with prob < 0.2 (in the "graveyard")
    - Revival rate: TUs crossing floor upward. =0 = floor graveyard problem

    Usage:
        loggers:
          - !TUMaskingDiagLogger
            periods: 10
            output_dir: ${output_dir}/tu_diag
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call_at_interval: int = Field(default=10, description="Steps between diagnostic output")
    output_dir: str | None = Field(default=None, description="Directory for summary plots")
    history_len: int = Field(default=500, description="Number of steps to retain in history")
    console_output: bool = Field(default=True, description="Print diagnostic lines to console")
    track_revivals: bool = Field(default=True, description="Track TU revivals (requires state)")

    _history: list[StepDiag] = []
    _prev_probs: np.ndarray | None = None
    _revival_count_window: deque = deque(maxlen=100)
    _save_dir: Path | None = None
    _total_revivals: int = 0
    _legend_printed: bool = False
    _network_names: list[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._prev_probs = None
        self._revival_count_window = deque(maxlen=100)
        self._total_revivals = 0
        self._legend_printed = False
        self._network_names = []

    def initialize(self, training_program=None):
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / 'tu_masking_diag'
        else:
            self._save_dir = Path('tu_masking_diag')

        self._save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TUMaskingDiagLogger initialized: {self._save_dir}")

    def _compute_entropy(
        self, probs: np.ndarray, axis: int | None = None, epsilon: float = 1e-6
    ) -> np.ndarray | float:
        """Compute binary entropy of TU mask probabilities.

        H = -mean[p*log(p) + (1-p)*log(1-p)]
        Range: 0 (all committed) to 1 (all uncertain at 0.5)
        """
        p = np.clip(probs, epsilon, 1 - epsilon)
        entropy_per_tu = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        normalized = entropy_per_tu / np.log(2)
        if axis is not None:
            return np.mean(normalized, axis=axis)
        return float(np.mean(normalized))

    def _count_revivals_per_network(self, probs: np.ndarray) -> np.ndarray:
        """Count TUs that crossed the floor threshold upward since last step, per network.

        Args:
            probs: shape (n_networks, n_tus)

        Returns:
            array of shape (n_networks,) with revival counts
        """
        assert probs.ndim == 2, f"expected (n_networks, n_tus), got {probs.shape}"
        if self._prev_probs is None or probs.shape != self._prev_probs.shape:
            return np.zeros(probs.shape[0], dtype=int)

        floor = L0_PENALTY_FLOOR_PROB
        was_below = self._prev_probs < floor
        now_above = probs >= floor
        revivals = np.sum(was_below & now_above, axis=1).astype(int)
        return revivals

    def _extract_diagnostics(self, step: int, tu_stats: dict, step_history: dict) -> StepDiag:
        """Extract per-network diagnostic metrics from tu_stats."""
        log_alpha = tu_stats.get("log_alpha")
        if log_alpha is None:
            return StepDiag(step=step, no_tu_stats=True)

        arr = np.asarray(log_alpha)

        # log_alpha shape in design mode (from nested vmaps/scans):
        #   5D: (batches_per_step, n_replicates, n_targets, n_networks, n_tus)
        #   4D: (n_replicates, n_targets, n_networks, n_tus)
        #   3D: (n_targets, n_networks, n_tus)
        #   2D: (n_networks, n_tus) - already reduced
        # We take replicate=0, target=0, last batch to get (n_networks, n_tus)
        while arr.ndim > 2:
            arr = arr[-1] if arr.ndim == 5 else arr[0]  # last batch for batches dim, first for rep/target
        assert arr.ndim == 2, f"expected 2D (n_networks, n_tus), got shape {arr.shape}"

        n_networks, n_tus = arr.shape
        probs = 1 / (1 + np.exp(-arr))  # sigmoid per element

        # get network names
        names = step_history.get("network_names")
        if names is None or len(names) != n_networks:
            names = [f"net_{i}" for i in range(n_networks)]
        self._network_names = list(names)

        # per-network metrics
        entropies = self._compute_entropy(probs, axis=1)
        enabled_counts = np.sum(probs >= 0.5, axis=1).astype(int)
        boundary_counts = np.sum((probs >= 0.3) & (probs <= 0.7), axis=1).astype(int)
        below_floor_counts = np.sum(probs < L0_PENALTY_FLOOR_PROB, axis=1).astype(int)
        prob_means = np.mean(probs, axis=1)
        prob_stds = np.std(probs, axis=1)

        # revival tracking per network
        if self.track_revivals:
            revival_counts = self._count_revivals_per_network(probs)
            total_step_revivals = int(np.sum(revival_counts))
            self._revival_count_window.append(total_step_revivals)
            self._total_revivals += total_step_revivals
            self._prev_probs = probs.copy()
        else:
            revival_counts = np.zeros(n_networks, dtype=int)

        # build per-network diagnostics
        network_diags = []
        for i in range(n_networks):
            network_diags.append(
                NetworkDiag(
                    name=names[i],
                    entropy=float(entropies[i]),
                    enabled=int(enabled_counts[i]),
                    n_tus=n_tus,
                    boundary=int(boundary_counts[i]),
                    below_floor=int(below_floor_counts[i]),
                    revivals=int(revival_counts[i]),
                    prob_mean=float(prob_means[i]),
                    prob_std=float(prob_stds[i]),
                )
            )

        # aggregates
        total_entropy = self._compute_entropy(probs.flatten())

        return StepDiag(
            step=step,
            networks=network_diags,
            total_entropy=total_entropy,
            total_enabled=int(np.sum(enabled_counts)),
            total_tus=n_networks * n_tus,
            total_boundary=int(np.sum(boundary_counts)),
            total_below_floor=int(np.sum(below_floor_counts)),
            total_revivals=self._total_revivals,
            revivals_last_100=sum(self._revival_count_window),
        )

    def _entropy_bar(self, entropy: float) -> str:
        """Return bar character for entropy level."""
        bars = "▁▂▃▄▅▆▇█"
        return bars[min(int(entropy * 8), 7)]

    def _format_console_output(self, diag: StepDiag) -> str:
        """Format diagnostic as multi-line table."""
        if diag.no_tu_stats:
            return f"[TU:{diag.step:5d}] no tu_stats available"

        lines = []

        # print legend once
        if not self._legend_printed:
            lines.append(
                "TU Masking: Enabled=TUs on, Entropy(0=committed,1=undecided), Bound=[0.3-0.7], Floor(<0.2)"
            )
            self._legend_printed = True

        # header
        sep = "─" * 60
        lines.append(f"[TU:{diag.step:5d}] {sep}")
        lines.append(f"           {'Network':<20} {'Enabled':>7}  {'Entropy':>7}  {'Bound':>5}  {'Floor':>5}")

        # per-network rows
        for nd in diag.networks:
            bar = self._entropy_bar(nd.entropy)
            lines.append(
                f"           {nd.name:<20} {nd.enabled:2d}/{nd.n_tus:<4d}  "
                f"{nd.entropy:.2f}{bar}    {nd.boundary:5d}  {nd.below_floor:5d}"
            )

        # separator + totals
        lines.append(f"           {sep}")
        total_bar = self._entropy_bar(diag.total_entropy)
        lines.append(
            f"           {'TOTAL':<20} {diag.total_enabled:2d}/{diag.total_tus:<4d}  "
            f"{diag.total_entropy:.2f}{total_bar}    {diag.total_boundary:5d}  {diag.total_below_floor:5d}  "
            f"rev={diag.revivals_last_100}/100"
        )

        return "\n".join(lines)

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        step_history = view.to_step_history()
        tu_stats = step_history.get("tu_stats", {})
        diag = self._extract_diagnostics(step, tu_stats, step_history)

        self._history.append(diag)
        if len(self._history) > self.history_len:
            self._history = self._history[-self.history_len :]

        if self.console_output:
            print(self._format_console_output(diag))

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        batch = view.latest()
        if batch is not None:
            step_history = view.to_step_history()
            tu_stats = step_history.get("tu_stats", {})
            diag = self._extract_diagnostics(step, tu_stats, step_history)
            self._history.append(diag)
            if self.console_output:
                print(self._format_console_output(diag))

        if self._save_dir:
            self._save_summary()
            self._save_history_pickle()
            if len(self._history) > 10:
                self._generate_plots()

    def _save_summary(self):
        """Save text summary of TU masking diagnostics with per-network breakdown."""
        if not self._history:
            return

        lines = [
            "TU MASKING DIAGNOSTIC SUMMARY",
            "=" * 60,
            "",
            f"Total steps logged: {len(self._history)}",
            f"Total TU revivals: {self._total_revivals}",
            f"Networks: {len(self._network_names)} ({', '.join(self._network_names)})",
            "",
        ]

        final = self._history[-1]
        if not final.no_tu_stats:
            lines.extend(
                [
                    "FINAL STATE (per network):",
                    "-" * 40,
                ]
            )

            for nd in final.networks:
                bar = self._entropy_bar(nd.entropy)
                lines.append(
                    f"  {nd.name}: {nd.enabled}/{nd.n_tus} enabled, "
                    f"entropy={nd.entropy:.3f}{bar}, boundary={nd.boundary}, floor={nd.below_floor}"
                )

            lines.extend(
                [
                    "",
                    f"TOTALS: {final.total_enabled}/{final.total_tus} enabled, "
                    f"entropy={final.total_entropy:.3f}, boundary={final.total_boundary}, floor={final.total_below_floor}",
                    "",
                    "HEALTH INDICATORS:",
                ]
            )

            # per-network health checks
            for nd in final.networks:
                issues = []
                if nd.entropy < 0.2:
                    issues.append("low entropy (committed early)")
                if nd.boundary == 0 and nd.entropy < 0.1:
                    issues.append("fully committed")
                if nd.below_floor > nd.n_tus * 0.8:
                    issues.append("most TUs in graveyard")

                status = "  " + ("WARNING " if issues else "OK      ") + f"{nd.name}: "
                if issues:
                    status += ", ".join(issues)
                else:
                    status += f"{nd.enabled}/{nd.n_tus} enabled, healthy entropy"
                lines.append(status)

            # global health
            lines.append("")
            if self._total_revivals == 0 and final.total_below_floor > 0:
                lines.append(
                    "  GLOBAL WARNING: No revivals despite TUs below floor - graveyard problem"
                )
            elif self._total_revivals > 0:
                lines.append(f"  GLOBAL OK: {self._total_revivals} TU revivals occurred")

        summary_path = self._save_dir / "tu_masking_summary.txt"
        summary_path.write_text("\n".join(lines))
        logger.info(f"Saved TU masking summary to {summary_path}")

    def _save_history_pickle(self):
        """Save raw history as pickle for analysis."""
        import dill as pickle

        history_path = self._save_dir / "tu_masking_history.pickle"
        with open(history_path, "wb") as f:
            pickle.dump(self._history, f)
        logger.info(f"Saved TU masking history to {history_path}")

    def _generate_plots(self):
        """Generate diagnostic plots with per-network curves."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return

        valid_history = [h for h in self._history if not h.no_tu_stats]
        if not valid_history or not valid_history[0].networks:
            return

        steps = [h.step for h in valid_history]
        n_networks = len(valid_history[0].networks)
        network_names = [nd.name for nd in valid_history[0].networks]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "TU Masking Convergence Diagnostics (Per-Network)", fontsize=14, fontweight="bold"
        )

        colors = plt.cm.tab10(np.linspace(0, 1, max(n_networks, 10)))

        # 1. Entropy over time (per network)
        ax = axes[0, 0]
        for i, name in enumerate(network_names):
            entropy = [h.networks[i].entropy for h in valid_history]
            ax.plot(steps, entropy, color=colors[i], linewidth=1.5, label=name, alpha=0.8)
        total_entropy = [h.total_entropy for h in valid_history]
        ax.plot(steps, total_entropy, "k--", linewidth=2, label="TOTAL", alpha=0.9)
        ax.axhline(0.3, color="g", linestyle=":", alpha=0.5, label="Healthy")
        ax.axhline(0.2, color="r", linestyle=":", alpha=0.5, label="Warning")
        ax.fill_between(steps, 0, 0.2, alpha=0.1, color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mask Entropy")
        ax.set_title("Mask Entropy (Per Network)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # 2. Enabled TU count (per network)
        ax = axes[0, 1]
        for i, name in enumerate(network_names):
            enabled = [h.networks[i].enabled for h in valid_history]
            ax.plot(steps, enabled, color=colors[i], linewidth=1.5, label=name, alpha=0.8)
        n_tus_per = valid_history[0].networks[0].n_tus
        ax.axhline(n_tus_per, color="gray", linestyle="--", alpha=0.5, label=f"Max ({n_tus_per})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Enabled TUs")
        ax.set_title("Enabled TU Count (Per Network)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # 3. Below floor count (per network)
        ax = axes[1, 0]
        for i, name in enumerate(network_names):
            below = [h.networks[i].below_floor for h in valid_history]
            ax.plot(steps, below, color=colors[i], linewidth=1.5, label=name, alpha=0.8)
        ax.axhline(0, color="g", linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("TUs Below Floor")
        ax.set_title("TUs in Graveyard (prob < 0.2)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # 4. Boundary count (per network)
        ax = axes[1, 1]
        for i, name in enumerate(network_names):
            boundary = [h.networks[i].boundary for h in valid_history]
            ax.plot(steps, boundary, color=colors[i], linewidth=1.5, label=name, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Boundary TUs")
        ax.set_title("Boundary TUs (prob 0.3-0.7)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self._save_dir / "tu_masking_diagnostics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved TU masking diagnostic plots to {plot_path}")

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        """Return aggregate metrics for MLFlow/hyperopt compatibility."""
        if not self._history:
            return None
        final = self._history[-1]
        if final.no_tu_stats:
            return None
        return {
            "final_entropy": final.total_entropy,
            "total_revivals": self._total_revivals,
            "steps_tracked": len(self._history),
            "final_enabled": final.total_enabled,
            "final_boundary": final.total_boundary,
            "final_below_floor": final.total_below_floor,
        }

    def finalize(self):
        logger.info(f"TUMaskingDiagLogger finalized with {len(self._history)} entries")
