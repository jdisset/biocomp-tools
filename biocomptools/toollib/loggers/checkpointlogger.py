# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import dill as pickle
from pathlib import Path
from typing import Callable
from pydantic import Field, PrivateAttr
import jax

from biocomp.jaxutils import tree_get
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
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
    required_arrays: list[str] = ["latest_params"]

    _save_dir: Path | None = PrivateAttr(default=None)
    _replicate_model_factory: Callable[..., object] | None = PrivateAttr(default=None)

    def initialize(self, training_program):
        """Initializes the logger by setting up the output directory."""
        self._save_dir = training_program._save_dir / self.output_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._replicate_model_factory = training_program.get_replicate_model_func()
        logger.debug(f"CheckpointLogger saving to {self._save_dir}")

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        if context.current_step == 0:
            return
        self._save(view, context)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self._save(view, context)

    def _save(self, view: HistoryView, context: LoggerContext) -> None:
        latest = view.latest()
        if latest is None:
            return
        step_history = view.to_step_history()

        if "latest_params" not in step_history:
            logger.debug(f"Skipping checkpoint at step {context.current_step}: not a sync point")
            return

        try:
            params = step_history["latest_params"]
            opt_state = step_history.get("opt_state")
            losses = step_history.get("loss", [])
            n_replicates = context.training_config.n_replicates
            step = context.current_step

            logger.info(f"Saving checkpoint for step {step}...")

            for i in range(n_replicates):
                rep_params = tree_get(params, i)
                if rep_params is None:
                    continue

                assert self._replicate_model_factory is not None, (
                    "CheckpointLogger._replicate_model_factory is not set. "
                    "Ensure that the training program provides a model factory."
                )
                model = self._replicate_model_factory(
                    all_params=params, all_losses=losses, replicate_id=i
                )
                if model is not None:
                    model_path = self._save_dir / f"step_{step:06d}_rep_{i}.model.pickle"
                    model.save(model_path)
                else:
                    logger.warning(f"Failed to create model for replicate {i} at step {step}")

                if self.save_optimizer_state and opt_state is not None:
                    rep_opt_state = tree_get(opt_state, i)
                    state_dict = {
                        "params": rep_params,
                        "opt_state": rep_opt_state,
                        "step": step,
                    }
                    state_path = self._save_dir / f"step_{step:06d}_rep_{i}.state.pickle"
                    with open(state_path, "wb") as f:
                        pickle.dump(jax.device_get(state_dict), f)
        except Exception as e:
            logger.error(f"Error saving checkpoint at step {context.current_step}: {e}")
            logger.exception(e)
