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
            if step == 0 or step_history is None:
                return

            if 'latest_params' not in step_history:
                logger.warning("No 'latest_params' in step_history for CheckpointLogger.")
                return

            params = step_history['latest_params']
            opt_state = step_history.get('opt_state')
            n_replicates = training_config.n_replicates

            logger.info(f"Saving checkpoint for step {step}...")

            for i in range(n_replicates):
                rep_params = tree_get(params, i)
                if rep_params is None:
                    continue

                # save the full BiocompModel
                model = self._replicate_model_factory(all_params=params, replicate_id=i)
                if model:
                    model_path = self._save_dir / f"step_{step:06d}_rep_{i}.model.pickle"
                    model.save(model_path)

                # save the raw training state for resuming
                if self.save_optimizer_state and opt_state:
                    rep_opt_state = tree_get(opt_state, i)
                    state_dict = {
                        'params': rep_params,
                        'opt_state': rep_opt_state,
                        'step': step,
                    }
                    state_path = self._save_dir / f"step_{step:06d}_rep_{i}.state.pickle"
                    with open(state_path, 'wb') as f:
                        pickle.dump(jax.device_get(state_dict), f)

        return [(self.periods, save_checkpoint)]
