"""Design Checkpoint Logger: Saves intermediate design optimization states as pickle files.

Enables:
- Resuming from intermediate checkpoints
- Post-hoc analysis of parameter evolution
- Debugging design optimization issues
"""

import dill as pickle
import jax
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any
from pydantic import ConfigDict, Field

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DesignCheckpointLogger(Logger):
    """Logger that saves design optimization checkpoints as pickle files.

    Saves:
    - Parameters (all replicates, all targets)
    - Optimizer state (for resuming)
    - Loss history
    - Aux data history if enabled
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: str = Field(default="checkpoints", description="Subdirectory for checkpoint files")
    save_optimizer_state: bool = Field(
        default=True, description="Include optimizer state in checkpoint"
    )
    save_aux_history: bool = Field(default=True, description="Include aux data in checkpoint")
    max_checkpoints: int = Field(
        default=5, description="Maximum checkpoints to keep (0 = unlimited)"
    )

    _save_dir: Optional[Path] = None
    _aux_history: List[Dict] = []
    _saved_checkpoints: List[Path] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aux_history = []
        self._saved_checkpoints = []

    def initialize(self, training_program=None):
        if training_program and hasattr(training_program, '_save_dir'):
            self._save_dir = training_program._save_dir / self.output_dir
        elif self.output_dir:
            self._save_dir = Path(self.output_dir)

        if self._save_dir:
            self._save_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"DesignCheckpointLogger saving to {self._save_dir}")

    def _save_checkpoint(self, step: int, step_history: Dict, training_config: Any):
        if self._save_dir is None:
            return

        params = step_history.get('latest_params')
        if params is None:
            # expected when checkpoint period doesn't align with sync points in defer_sync mode
            logger.debug(f"Skipping checkpoint at step {step}: not a sync point")
            return

        checkpoint = {
            'step': step,
            'params': jax.device_get(params),
            'loss': step_history.get('loss'),
            'all_losses': jax.device_get(step_history.get('all_losses')),
            'config': {
                'n_replicates': training_config.n_replicates,
                'n_epochs': training_config.n_epochs,
                'n_batches_per_epoch': training_config.n_batches_per_epoch,
            },
        }

        if self.save_optimizer_state:
            opt_state = step_history.get('opt_state')
            if opt_state is not None:
                checkpoint['opt_state'] = jax.device_get(opt_state)

        if self.save_aux_history and self._aux_history:
            checkpoint['aux_history'] = self._aux_history.copy()

        # add current step's aux data
        aux_keys = [
            'sublosses',
            'tu_stats',
            'ratio_stats',
            'l0_penalty',
            'tucount_penalty',
            'spread_penalty',
            'coupling_penalty',
        ]
        checkpoint['current_aux'] = {
            k: jax.device_get(step_history.get(k)) for k in aux_keys if k in step_history
        }

        checkpoint_path = self._save_dir / f"checkpoint_step{step:06d}.pickle"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        self._saved_checkpoints.append(checkpoint_path)

        # cleanup old checkpoints if max_checkpoints is set
        if self.max_checkpoints > 0 and len(self._saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self._saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def _collect_aux_entry(self, step: int, step_history: Dict):
        """Collect condensed aux data for history tracking."""
        import numpy as np

        def _to_scalar(val) -> float:
            """Convert array to scalar, taking mean if multi-element."""
            if val is None:
                return float('nan')
            arr = jax.device_get(val)
            if hasattr(arr, 'shape'):
                arr = np.asarray(arr)
                if arr.size == 0:
                    return float('nan')
                return float(np.nanmean(arr))
            return float(arr) if arr is not None else float('nan')

        entry = {'step': step}

        # scalar metrics
        for key in ['loss', 'l0_penalty', 'tucount_penalty', 'spread_penalty', 'coupling_penalty']:
            val = step_history.get(key)
            if val is not None:
                entry[key] = _to_scalar(val)

        # sublosses
        sublosses = step_history.get('sublosses', {})
        if sublosses:
            for k, v in sublosses.items():
                entry[f'subloss_{k}'] = _to_scalar(v) if v is not None else None

        # TU stats
        tu_stats = step_history.get('tu_stats', {})
        if tu_stats:
            for k, v in tu_stats.items():
                entry[f'tu_{k}'] = _to_scalar(v) if v is not None else None

        # ratio stats
        ratio_stats = step_history.get('ratio_stats', {})
        if ratio_stats:
            for k, v in ratio_stats.items():
                entry[f'ratio_{k}'] = _to_scalar(v) if v is not None else None

        self._aux_history.append(entry)

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
        def periodic_callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None or step == 0:
                return

            # collect aux data every step for history
            self._collect_aux_entry(step, step_history)

            # save checkpoint at period intervals
            self._save_checkpoint(step, step_history, training_config)

        def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
            if step_history is None:
                return

            self._collect_aux_entry(step, step_history)
            self._save_checkpoint(step, step_history, training_config)

            # save final aux history separately
            if self._save_dir and self._aux_history:
                history_path = self._save_dir / "aux_history_full.pickle"
                with open(history_path, 'wb') as f:
                    pickle.dump(self._aux_history, f)
                logger.info(f"Saved full aux history to {history_path}")

        # use self.periods for periodic, -1 for end
        return [(self.periods, periodic_callback), (-1, final_callback)]

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        return {
            'checkpoints_saved': len(self._saved_checkpoints),
            'aux_history_entries': len(self._aux_history),
        }

    def finalize(self):
        logger.info(
            f"DesignCheckpointLogger finalized: "
            f"{len(self._saved_checkpoints)} checkpoints, "
            f"{len(self._aux_history)} aux entries"
        )


def load_design_checkpoint(checkpoint_path: str | Path) -> Dict[str, Any]:
    """Load a design checkpoint from pickle file.

    Returns dict with:
    - 'step': optimization step
    - 'params': ParameterTree
    - 'loss': loss at checkpoint
    - 'all_losses': per-target/network losses
    - 'opt_state': optimizer state (if saved)
    - 'aux_history': list of aux data dicts (if saved)
    - 'current_aux': aux data at checkpoint step
    """
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def load_aux_history(history_path: str | Path) -> List[Dict[str, Any]]:
    """Load aux history from pickle file.

    Returns list of dicts, each containing:
    - 'step': optimization step
    - 'loss', 'l0_penalty', etc.: scalar metrics
    - 'subloss_*': subloss components
    - 'tu_*': TU masking statistics
    - 'ratio_*': ratio statistics
    """
    with open(history_path, 'rb') as f:
        return pickle.load(f)
