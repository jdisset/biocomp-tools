## {{{                          --     imports     --

import dracon as dr
from dracon.deferred import DeferredNode
import logging
from scipy.ndimage import gaussian_filter1d
from labellines import labelLine, labelLines
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
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
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


class PredictionLogger(Logger):
    figure_template: DeferredNode[Figure]

    def initialize(self, training_program):
        self._dman = training_program._training_dman

    def run(self, params):
        # TODO:
        # [ ] get stack from training_dman
        # [ ] generate (batched) predictions (c.f. biocomp/trainutils.py:wandb_plot_pred)
        # [ ] repeatedly construct figure_template with ground_truth and predicted as context

        networks = self._dman.get_networks()
        stack = self._dman.get_compute_stack()

    

        # figure = self.figure_template.construct(context={"D": ...})
        # figure.run()


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

## {{{              --     saving and plotting best model     --


def get_best_smoothed_loss_id(all_losses: ndArray, sigma: float = 12.0) -> Tuple[int, np.ndarray]:
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma, mode='nearest')
    endval = smoothed_losses[:, -1]
    endval[np.isnan(endval)] = np.inf
    best_loss_id = int(np.argmin(endval))
    return best_loss_id, smoothed_losses


def ffill(arr, mask=None):
    if mask is None:
        mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def plot_loss(loss_history: List[np.ndarray]):
    all_losses = np.hstack(loss_history)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    nan_mask = np.isnan(all_losses)
    filled_losses = ffill(all_losses)
    best_loss_id, smoothed_losses = get_best_smoothed_loss_id(filled_losses)

    yrange = np.nanmax(all_losses) - np.nanmin(all_losses)

    # Plot non-nan values as blue solid lines
    for i in range(all_losses.shape[0]):
        non_nan_indices = ~nan_mask[i]
        ax.plot(
            np.arange(all_losses.shape[1])[non_nan_indices],
            all_losses[i, non_nan_indices],
            color='#AAA',
            linestyle='-',
            linewidth=1,
            alpha=0.5,
        )

        nan_boundaries = np.where(np.diff(non_nan_indices))[0]
        # plot red cross
        for boundary in nan_boundaries:
            ax.plot(
                boundary,
                all_losses[i, boundary],
                'x',
                linewidth=2,
                color='red',
                alpha=0.5,
                markersize=5,
            )
            offsetx = 0.01 * all_losses.shape[1]
            offsety = 0.00 * yrange
            ax.text(
                boundary + offsetx,
                all_losses[i, boundary] + offsety,
                f'rep {i}',
                fontsize=7,
                color='red',
                ha='left',
                va='center',
            )

        valid_propotion = non_nan_indices.sum() / all_losses.shape[1]

        if valid_propotion > 0.2:
            ax.plot(
                np.arange(all_losses.shape[1])[non_nan_indices],
                smoothed_losses[i, non_nan_indices],
                linewidth=1,
                label=f'rep {i}',
            )
        # only plot smoothed for the valid

    try:
        labelLines(ax.get_lines(), zorder=2.5)
    except Exception as e:
        pass

    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax


##────────────────────────────────────────────────────────────────────────────}}}
