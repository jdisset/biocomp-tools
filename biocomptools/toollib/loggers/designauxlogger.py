"""Design Aux Logger: Captures and visualizes auxiliary data from design optimization runs.

Produces:
- Visual summary plots (loss curves, subloss breakdowns, TU stats, ratio evolution)
- Textual summary with key metrics and insights
- Pickle files with raw aux data history for later analysis
"""

import numpy as np
import json
import dill as pickle
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any, Dict
from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.utils import extract_design_step_metrics
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DesignAuxLogger(Logger):
    """Logger that captures comprehensive aux data from design optimization.

    Tracks:
    - Subloss components (sinkhorn, lncc, mse, etc.) over time
    - TU statistics (enabled count, mean probability) over time
    - Ratio statistics (min, max, mean) over time
    - Regularization penalties over time
    - Produces visual summary and textual analysis
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: Optional[str] = None
    save_pickle: bool = Field(default=True, description="Save raw aux history as pickle")
    save_json: bool = Field(default=True, description="Save summary metrics as JSON")
    generate_plots: bool = Field(default=True, description="Generate visual summary plots")
    plot_period: int = Field(default=100, description="How often to generate interim plots")

    _history: List[Dict[str, Any]] = []
    _step_count: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._step_count = 0

    _save_dir: Optional[Path] = None

    def initialize(self, training_program=None):
        # get output_dir from training_program if not provided
        if self.output_dir:
            self._save_dir = Path(self.output_dir)
        elif training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / 'aux_logs'
        else:
            self._save_dir = Path('aux_logs')

        self._save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DesignAuxLogger initialized: {self._save_dir}")

    def _extract_aux_metrics(self, step_history: Dict) -> Dict[str, Any]:
        """Extract scalar metrics from step_history for tracking."""
        return extract_design_step_metrics(step_history)

    def _update_history(self, step: int, step_history: Dict):
        metrics = self._extract_aux_metrics(step_history)
        metrics["step"] = step
        self._history.append(metrics)

    def _generate_visual_summary(self, output_path: Path, title_suffix: str = ""):
        """Generate visual summary plots of the design optimization."""
        if not self._history:
            return

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping visual summary")
            return

        steps = [h["step"] for h in self._history]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"Design Optimization Summary{title_suffix}", fontsize=14, fontweight='bold')

        # 1. Total loss curve
        ax = axes[0, 0]
        losses = [h.get("loss", np.nan) for h in self._history]
        ax.plot(steps, losses, 'b-', linewidth=1.5, label='Total Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Subloss breakdown
        ax = axes[0, 1]
        subloss_keys = [
            k for k in self._history[0].keys() if k.startswith("subloss_") and "weighted" not in k
        ]
        for key in subloss_keys:
            vals = [h.get(key, np.nan) for h in self._history]
            label = key.replace("subloss_", "")
            ax.plot(steps, vals, linewidth=1.2, label=label)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss Component')
        ax.set_title('Subloss Components (unweighted)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. TU statistics
        ax = axes[1, 0]
        tu_enabled = [h.get("tu_enabled_count", np.nan) for h in self._history]
        tu_total = [h.get("tu_total_count", 1) for h in self._history]
        tu_pct = [
            100 * e / t if t > 0 else np.nan for e, t in zip(tu_enabled, tu_total, strict=True)
        ]
        ax.plot(steps, tu_pct, 'g-', linewidth=1.5, label='% TU Enabled')
        ax.set_xlabel('Step')
        ax.set_ylabel('% Enabled')
        ax.set_title('TU Masking: % Enabled')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # TU mean probability on secondary axis
        ax2 = ax.twinx()
        tu_prob = [h.get("tu_mean_prob", np.nan) for h in self._history]
        ax2.plot(steps, tu_prob, 'r--', linewidth=1, alpha=0.7, label='Mean P(enabled)')
        ax2.set_ylabel('Mean Probability', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # 4. Ratio statistics
        ax = axes[1, 1]
        ratio_min = [h.get("ratio_min", np.nan) for h in self._history]
        ratio_max = [h.get("ratio_max", np.nan) for h in self._history]
        ratio_mean = [h.get("ratio_mean", np.nan) for h in self._history]
        ax.fill_between(steps, ratio_min, ratio_max, alpha=0.3, label='Range')
        ax.plot(steps, ratio_mean, 'b-', linewidth=1.5, label='Mean')
        ax.set_xlabel('Step')
        ax.set_ylabel('Ratio Value')
        ax.set_title('Ratio Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Regularization penalties
        ax = axes[2, 0]
        penalty_names = ["l0_penalty", "tucount_penalty", "spread_penalty", "coupling_penalty"]
        for pname in penalty_names:
            vals = [h.get(pname, np.nan) for h in self._history]
            if not all(np.isnan(v) if isinstance(v, float) else False for v in vals):
                ax.plot(steps, vals, linewidth=1.2, label=pname.replace("_penalty", ""))
        ax.set_xlabel('Step')
        ax.set_ylabel('Penalty')
        ax.set_title('Regularization Penalties')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('symlog', linthresh=1e-6)

        # 6. Summary text
        ax = axes[2, 1]
        ax.axis('off')
        summary_text = self._generate_text_summary()
        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved visual summary to {output_path}")

    def _generate_text_summary(self) -> str:
        """Generate textual summary of design optimization."""
        if not self._history:
            return "No data collected"

        first, last = self._history[0], self._history[-1]
        n_steps = last["step"]

        lines = [
            "DESIGN OPTIMIZATION SUMMARY",
            "=" * 30,
            f"Total steps: {n_steps}",
            "",
            "LOSS PROGRESSION:",
            f"  Initial: {first.get('loss', 'N/A'):.6f}",
            f"  Final:   {last.get('loss', 'N/A'):.6f}",
        ]

        if first.get("loss") and last.get("loss"):
            improvement = (first["loss"] - last["loss"]) / first["loss"] * 100
            lines.append(f"  Improvement: {improvement:.1f}%")

        lines.extend(["", "SUBLOSS BREAKDOWN (final):"])
        for key in sorted(last.keys()):
            if key.startswith("subloss_") and "weighted" in key:
                val = last[key]
                lines.append(f"  {key.replace('subloss_', '')}: {val:.6f}")

        lines.extend(["", "TU MASKING (final):"])
        tu_enabled = last.get("tu_enabled_count", "N/A")
        tu_total = last.get("tu_total_count", "N/A")
        lines.append(f"  Enabled: {tu_enabled}/{tu_total}")
        lines.append(f"  Mean P(enabled): {last.get('tu_mean_prob', 'N/A'):.3f}")

        lines.extend(["", "RATIO STATS (final):"])
        lines.append(
            f"  Range: [{last.get('ratio_min', 'N/A'):.4f}, {last.get('ratio_max', 'N/A'):.4f}]"
        )
        lines.append(f"  Mean: {last.get('ratio_mean', 'N/A'):.4f}")

        return "\n".join(lines)

    def _save_pickle_history(self, output_path: Path):
        """Save raw history as pickle for later analysis."""
        with open(output_path, 'wb') as f:
            pickle.dump(self._history, f)
        logger.info(f"Saved aux history pickle to {output_path}")

    def _save_json_summary(self, output_path: Path):
        """Save summary metrics as JSON."""
        if not self._history:
            return

        summary = {
            "total_steps": self._step_count,
            "initial_metrics": self._history[0] if self._history else {},
            "final_metrics": self._history[-1] if self._history else {},
            "history_length": len(self._history),
        }

        # compute trajectories for key metrics
        summary["trajectories"] = {
            "steps": [h["step"] for h in self._history],
            "loss": [h.get("loss") for h in self._history],
            "tu_enabled_pct": [
                100 * h.get("tu_enabled_count", 0) / max(h.get("tu_total_count", 1), 1)
                for h in self._history
            ],
        }

        output_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Saved JSON summary to {output_path}")

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
        def periodic_callback(step, training_config, step_history=None, stack=None, **kwargs):
            self._step_count = step
            logger.debug(
                f"DesignAuxLogger periodic_callback: step={step}, "
                f"step_history_keys={list(step_history.keys()) if step_history else None}"
            )
            if step_history is None:
                logger.warning(f"DesignAuxLogger: step_history is None at step {step}")
                return
            self._update_history(step, step_history)

            # generate interim plots if requested
            if self.generate_plots and self._save_dir and step % self.plot_period == 0:
                self._generate_visual_summary(
                    self._save_dir / f"aux_summary_step{step:06d}.png",
                    title_suffix=f" (Step {step})",
                )

        def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
            self._step_count = step
            logger.info(
                f"DesignAuxLogger final_callback: step={step}, "
                f"step_history={'None' if step_history is None else f'keys={list(step_history.keys())}'}, "
                f"current_history_len={len(self._history)}"
            )
            if step_history is not None:
                self._update_history(step, step_history)
            else:
                logger.warning("DesignAuxLogger: step_history is None in final_callback")

            if self._save_dir:
                self._save_dir.mkdir(parents=True, exist_ok=True)

                if self.save_pickle:
                    self._save_pickle_history(self._save_dir / "aux_history.pickle")

                if self.save_json:
                    self._save_json_summary(self._save_dir / "aux_summary.json")

                if self.generate_plots:
                    self._generate_visual_summary(
                        self._save_dir / "aux_summary_final.png", title_suffix=" (Final)"
                    )

                # save text summary
                text_summary = self._generate_text_summary()
                (self._save_dir / "aux_summary.txt").write_text(text_summary)
                logger.info(f"Saved text summary to {self._save_dir / 'aux_summary.txt'}")

        callbacks = []
        if self.call_at_interval is not None:
            callbacks.append((self.call_at_interval, periodic_callback))
        if -1 in self.call_at:
            callbacks.append((-1, final_callback))
        return callbacks

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not self._history:
            return None
        return {
            "steps_tracked": len(self._history),
            "final_loss": self._history[-1].get("loss") if self._history else None,
        }

    def finalize(self):
        logger.info(f"DesignAuxLogger finalized with {len(self._history)} entries")


