import pickle
from pathlib import Path
from typing import List, Tuple, Callable, Optional
from pydantic import Field
import jax

from biocomp.jaxutils import tree_get
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class CheckpointLogger(Logger):
    """Saves model checkpoints for each replicate at specified intervals."""

    output_dir: str = Field(
        default="checkpoints", description="Subdirectory to save checkpoint files."
    )
    save_optimizer_state: bool = Field(
        default=True,
        description="If True, saves parameters and optimizer state for resuming training.",
    )

    _save_dir: Optional[Path] = None
    _replicate_model_factory: Optional[Callable] = None

    def initialize(self, training_program):
        """Initializes the logger by setting up the output directory."""
        self._save_dir = training_program._save_dir / self.output_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._replicate_model_factory = training_program.get_replicate_model_func()
        logger.debug(f"CheckpointLogger saving to {self._save_dir}")

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Returns the callback for saving checkpoints."""

        def save_checkpoint(
            step: int, training_config, step_history: Optional[dict] = None, **kwargs
        ):
            try:
                if step == 0 or step_history is None:
                    return

                if 'latest_params' not in step_history:
                    logger.warning("No 'latest_params' in step_history for CheckpointLogger.")
                    return

                params = step_history['latest_params']
                opt_state = step_history.get('opt_state')
                losses = step_history.get('loss', [])
                n_replicates = training_config.n_replicates

                logger.info(f"Saving checkpoint for step {step}...")

                for i in range(n_replicates):
                    rep_params = tree_get(params, i)
                    # safe check for None - avoid "truth value of array" error
                    if rep_params is None:
                        continue

                    # save the full BiocompModel
                    assert self._replicate_model_factory is not None, (
                        "CheckpointLogger._replicate_model_factory is not set. "
                        "Ensure that the training program provides a model factory."
                    )
                    model = self._replicate_model_factory(
                        all_params=params, all_losses=losses, replicate_id=i
                    )
                    # safe boolean check for model existence
                    if model is not None:
                        model_path = self._save_dir / f"step_{step:06d}_rep_{i}.model.pickle"
                        model.save(model_path)
                    else:
                        logger.warning(f"Failed to create model for replicate {i} at step {step}")

                    # save the raw training state for resuming
                    if self.save_optimizer_state and opt_state is not None:
                        rep_opt_state = tree_get(opt_state, i)
                        state_dict = {
                            'params': rep_params,
                            'opt_state': rep_opt_state,
                            'step': step,
                        }
                        state_path = self._save_dir / f"step_{step:06d}_rep_{i}.state.pickle"
                        with open(state_path, 'wb') as f:
                            pickle.dump(jax.device_get(state_dict), f)
            except Exception as e:
                logger.error(f"Error saving checkpoint at step {step}: {e}")
                logger.exception(e)

        return [(self.periods, save_checkpoint)]
