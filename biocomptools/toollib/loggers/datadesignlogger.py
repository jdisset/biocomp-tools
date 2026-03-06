"""Logger for data-driven design optimization - saves metrics and loss history."""

import numpy as np
import json
from pathlib import Path
from typing import Any
from pydantic import ConfigDict

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DataDesignLogger(Logger):
    """
    Logger for data-driven design optimization runs.

    Tracks:
    - Design loss over time (per target, per replicate, per network)
    - Comparison with baseline model prediction loss
    - Summary statistics saved as JSON for later visualization
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: str | None = None
    baseline_r2: float | None = None
    baseline_rmse: float | None = None
    baseline_loss: float | None = None
    top_k: int = 5
    save_interval: int = 100

    _loss_history: list[tuple[int, np.ndarray]] = []
    _best_loss_per_target: dict[int, float] = {}
    _best_config_per_target: dict[int, tuple[int, int]] = {}
    _final_params: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loss_history = []
        self._best_loss_per_target = {}
        self._best_config_per_target = {}
        self._final_params = None
        self._step_count = 0

    def _update_history(self, step: int, all_losses: np.ndarray):
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 2:
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)

        self._loss_history.append((step, all_losses.copy()))

        n_replicates, n_targets, n_networks = all_losses.shape
        for target_id in range(n_targets):
            target_losses = all_losses[:, target_id, :]
            min_loss = float(np.min(target_losses))

            if (
                target_id not in self._best_loss_per_target
                or min_loss < self._best_loss_per_target[target_id]
            ):
                self._best_loss_per_target[target_id] = min_loss
                flat_idx = np.argmin(target_losses)
                rep_id = flat_idx // n_networks
                net_id = flat_idx % n_networks
                self._best_config_per_target[target_id] = (int(rep_id), int(net_id))

    def _save_loss_history(self, output_path: Path):
        """Save loss history as JSON for later visualization via biocomp-plot."""
        if not self._loss_history:
            return

        _, first_losses = self._loss_history[0]
        n_replicates, n_targets, n_networks = first_losses.shape

        data = {
            'steps': [s for s, _ in self._loss_history],
            'shape': {
                'n_replicates': n_replicates,
                'n_targets': n_targets,
                'n_networks': n_networks,
            },
            'baseline': {
                'loss': self.baseline_loss,
                'r2': self.baseline_r2,
                'rmse': self.baseline_rmse,
            },
            'mean_losses': [float(np.mean(losses)) for _, losses in self._loss_history],
            'min_losses': [float(np.min(losses)) for _, losses in self._loss_history],
            'best_per_target': {
                str(k): {'loss': v, 'config': list(self._best_config_per_target.get(k, (0, 0)))}
                for k, v in self._best_loss_per_target.items()
            },
        }

        # per-target top-k trajectories
        if self._loss_history:
            _, final_losses = self._loss_history[-1]
            per_target = {}
            for target_id in range(n_targets):
                target_final = final_losses[:, target_id, :].reshape(-1)
                top_k_idx = np.argsort(target_final)[: self.top_k]
                trajectories = []
                for flat_idx in top_k_idx:
                    rep_id, net_id = flat_idx // n_networks, flat_idx % n_networks
                    history = [
                        float(losses[rep_id, target_id, net_id]) for _, losses in self._loss_history
                    ]
                    trajectories.append(
                        {
                            'rep_id': int(rep_id),
                            'net_id': int(net_id),
                            'final_loss': float(target_final[flat_idx]),
                            'history': history,
                        }
                    )
                per_target[str(target_id)] = trajectories
            data['per_target_topk'] = per_target

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved design loss history to {output_path}")

    def _save_final_summary(self, output_path: Path):
        """Save final summary as JSON."""
        if not self._loss_history:
            return

        _, final_losses = self._loss_history[-1]
        _, initial_losses = self._loss_history[0]
        n_replicates, n_targets, n_networks = final_losses.shape

        summary = {
            'total_steps': self._step_count,
            'shape': {
                'n_replicates': n_replicates,
                'n_targets': n_targets,
                'n_networks': n_networks,
            },
            'initial': {
                'mean_loss': float(np.mean(initial_losses)),
                'best_loss': float(np.min(initial_losses)),
            },
            'final': {
                'mean_loss': float(np.mean(final_losses)),
                'best_loss': float(np.min(final_losses)),
            },
            'baseline': {
                'loss': self.baseline_loss,
                'r2': self.baseline_r2,
                'rmse': self.baseline_rmse,
            },
            'best_config_per_target': {
                str(t): {'rep': rep, 'net': net, 'loss': self._best_loss_per_target.get(t)}
                for t, (rep, net) in self._best_config_per_target.items()
            },
        }

        if self.baseline_loss is not None and self.baseline_loss > 0:
            summary['improvement_vs_baseline_pct'] = (
                1 - np.min(final_losses) / self.baseline_loss
            ) * 100

        output_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved design summary to {output_path}")

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        self._step_count = step
        step_history = view.to_step_history()
        all_losses = step_history.get('all_losses')
        if all_losses is None:
            return
        self._update_history(step, all_losses)
        self._final_params = step_history.get('latest_params')

        if self.output_dir and step % self.save_interval == 0:
            output_path = Path(self.output_dir)
            self._save_loss_history(output_path / f'loss_history_step{step:06d}.json')

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        step = context.current_step
        self._step_count = step
        batch = view.latest()
        if batch is not None:
            step_history = view.to_step_history()
            all_losses = step_history.get('all_losses')
            if all_losses is not None:
                self._update_history(step, all_losses)
            self._final_params = step_history.get('latest_params')

        if self.output_dir:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._save_loss_history(output_path / 'final_loss_history.json')
            self._save_final_summary(output_path / 'final_summary.json')

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        if not self._loss_history:
            return None

        _, final_losses = self._loss_history[-1]
        metrics = {
            'final_mean_loss': float(np.mean(final_losses)),
            'final_best_loss': float(np.min(final_losses)),
            'total_steps': self._step_count,
            'best_configs': dict(self._best_config_per_target),
            'best_losses': dict(self._best_loss_per_target),
        }

        if self.baseline_loss is not None:
            metrics['baseline_loss'] = self.baseline_loss
            metrics['improvement_vs_baseline'] = (
                1 - np.min(final_losses) / self.baseline_loss
            ) * 100

        if self.baseline_r2 is not None:
            metrics['baseline_r2'] = self.baseline_r2
        if self.baseline_rmse is not None:
            metrics['baseline_rmse'] = self.baseline_rmse

        return metrics

    def finalize(self):
        pass
