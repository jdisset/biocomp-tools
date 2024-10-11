## {{{                          --     imports     --

import dracon as dr
import logging
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import wandb as wb
from pathlib import Path
import numpy as np
from numpy import ndarray as ndArray
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Annotated
from pydantic import Field, BaseModel
from biocomptools.toollib.common import config
from biocomptools.toollib.networkselector import NetworkSet, NetworkSelector, build_data_manager
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from dracon.lazy import LazyDraconModel
import pandas as pd
import time
from dracon.resolvable import Resolvable
from dracon.commandline import Program, make_program, Arg
import biocomp as bc
from biocomp.train import TrainingConfig
from biocomp.library import PartsLibrary

from biocomp.utils import (
    ArbitraryModel,
    load_lib,
    save,
    EncodedPartialFunction,
    PartialFunction,
    PartialFunctionResult,
)

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG, DataManager
import re
from sqlmodel import select, Session, col
from tqdm import tqdm

import biocomptools.toollib.models as md


logging.getLogger('dracon.commandline').setLevel(logging.DEBUG)
##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     Loggers     --


class Logger(BaseModel):
    periods: Union[int, List[int]] = 1  # Number of steps between logs or list of periods

    def initialize(self, training_program):
        """Optional initialization before training starts."""
        pass

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Return a list of (period, callback_function) tuples for the training loop."""
        raise NotImplementedError

    def finalize(self):
        """Optional cleanup after training ends."""
        pass


class WandBLogger(Logger):
    entity: str
    project: str
    run_name: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    save_period: int = 1  # Period for saving checkpoints
    prediction_period: int = 5  # Period for logging predictions
    output_dir: Optional[str] = None  # Optional output directory
    validation_samples: int = 100  # Number of samples for validation predictions

    _wandb_run: Any = None

    def initialize(self, training_program):
        import wandb

        self._wandb_run = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.run_name,
            config=training_program.model_dump(),
        )

        # Determine save directory
        today = time.strftime('%Y-%m-%d', time.localtime())
        training_run_name = (
            f'{today}_{self._wandb_run.project}_{self._wandb_run.id}_{self._wandb_run.name}'
        )

        if not self.output_dir:
            self._save_dir = Path(training_program.outputdir) / 'wandb' / training_run_name
        else:
            self._save_dir = Path(self.output_dir) / training_run_name

        self._save_dir.mkdir(exist_ok=True, parents=True)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        callbacks = []

        # Loss logging callback
        def wandb_loss_logger(step, training_config, step_history=None, **kwargs):
            if self._wandb_run and step_history is not None:
                losses = step_history.get('loss')
                if losses is not None:
                    # Handle array of losses
                    if isinstance(losses, (list, np.ndarray)):
                        loss_value = np.mean(losses)
                    else:
                        loss_value = losses
                    self._wandb_run.log({'loss': loss_value}, step=step)

        callbacks.append(
            (self.periods if isinstance(self.periods, int) else self.periods[0], wandb_loss_logger)
        )

        # Model checkpoint saving callback
        def save_checkpoint(step, training_config, step_history=None, params=None, **kwargs):
            if step % self.save_period == 0 and params is not None:
                model_path = self._save_dir / 'model_checkpoints' / f'model_step_{step}.pickle'
                model_path.parent.mkdir(exist_ok=True, parents=True)
                save(params, model_path)
                self._wandb_run.save(str(model_path), base_path=str(self._save_dir))

        callbacks.append((self.save_period, save_checkpoint))

        # Prediction logging callback
        self._validation_dman = None

        def log_predictions(step, training_config, step_history=None, params=None, **kwargs):
            if params is None:
                logging.warning('No params provided for prediction logging')
                return

            validation_set = training_program.validation_set

            if not validation_set.content:
                return  # No validation set provided

            if self._validation_dman is None:
                self._validation_dman = build_data_manager(
                    training_program.parts_library,
                    training_program.db_session,
                    training_program.path_prefix,
                    data_conf=training_program.data_conf,
                    dataset=validation_set,
                )

            # Generate predictions and plot them
            predictions_dir = self._save_dir / 'training' / 'predictions' / f'step_{step}'
            predictions_dir.mkdir(exist_ok=True, parents=True)

            images = self.generate_and_save_predictions(
                params, self._validation_dman, predictions_dir
            )
            # Log images to WandB
            self._wandb_run.log({'validation_predictions': images}, step=step)

        # callbacks.append((self.prediction_period, log_predictions))

        return callbacks

    def generate_and_save_predictions(self, params, data_manager, predictions_dir):
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO


        images = []
        networks = data_manager.get_networks()

        # Generate plots for each network
        for network in tqdm(networks, desc='Generating predictions'):
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

            fig = self.plot_prediction(network)
            image_path = predictions_dir / f'{network.name}_prediction.png'
            fig.savefig(image_path)
            plt.close(fig)
            # Create WandB image
            wandb_image = wandb.Image(str(image_path), caption=network.name)
            images.append(wandb_image)

        return images

    def plot_prediction(self, network):
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return fig

    def finalize(self):
        if self._wandb_run:
            self._wandb_run.finish()


##────────────────────────────────────────────────────────────────────────────}}}
