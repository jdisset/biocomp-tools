## {{{                          --     imports     --

import dracon as dr
from dracon.deferred import DeferredNode
import logging
import biocomp.plotutils as pu
from scipy.ndimage import gaussian_filter1d
from labellines import labelLine, labelLines
import matplotlib.pyplot as plt
import biocomp as bc
import wandb as wb
from pathlib import Path
import numpy as np
from numpy import ndarray as ndArray
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Annotated, Literal
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


class Logger(ArbitraryModel):
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
    """Generates prediction visualizations during training"""

    figure_template: DeferredNode[Figure]
    batch_size: int = 2000
    n_chunks: int = 5
    seed: int = 0

    # Which dataset to use for predictions
    data_source: Literal["training", "validation", "custom"] = "validation"
    custom_dataset: Optional[NetworkSet] = None  # Only used if data_source is "custom"

    def initialize(self, training_program):
        # Select the appropriate DataManager based on data_source
        if self.data_source == "training":
            self._dman = training_program._training_dman
        elif self.data_source == "validation":
            if not training_program.validation_set.content:
                raise ValueError("Validation set is empty but data_source is 'validation'")
            self._dman = build_data_manager(
                training_program.parts_library,
                training_program.db_session,
                training_program.path_prefix,
                training_program.data_conf,
                training_program.validation_set,
                network_cache=config.paths.cache.networks,
            )
        elif self.data_source == "custom":
            if not self.custom_dataset:
                raise ValueError("Custom dataset not provided but data_source is 'custom'")
            self._dman = build_data_manager(
                training_program.parts_library,
                training_program.db_session,
                training_program.path_prefix,
                training_program.data_conf,
                self.custom_dataset,
                network_cache=config.paths.cache.networks,
            )
        else:
            raise ValueError(f"Unknown data_source: {self.data_source}")

        # Create the predictions directory
        self.predictions_dir = Path(training_program.outputdir) / 'predictions' / self.data_source
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(self, step: int, predictions, groundtruth):
        """Save prediction plots for each network"""
        prediction_path = self.predictions_dir / f'step_{step}'
        prediction_path.mkdir(exist_ok=True)

        for i, (pred, truth) in enumerate(zip(predictions, groundtruth)):
            # Construct figure with context variables that can be used in the config
            fig = self.figure_template.construct(
                context={
                    "pred_data": pred,
                    "ground_truth_data": truth,
                    "network_id": i,
                    "step": step,
                    "prediction_path": prediction_path.as_posix(),
                    "data_source": self.data_source,
                }
            )
            fig.run()

        return prediction_path

    def generate_predictions(self, params):
        import jax
        from biocomp import compute as bcmp

        networks = self._dman.get_networks()
        stack = self._dman.get_compute_stack()
        assert isinstance(stack, bcmp.ComputeStack)

        n_samples_total = self.batch_size * self.n_chunks
        key = jax.random.PRNGKey(self.seed)

        # Get uniform samples across the input space
        X, Y = self._dman.get_uniform_samples(key, n_samples_total)
        X = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in X]
        Y = [np.expand_dims(arr, axis=1) if arr.ndim == 1 else arr for arr in Y]

        # Concatenate all inputs
        all_x = np.concatenate(X, axis=1)
        x_chunks = np.split(all_x, self.n_chunks, axis=0)

        @jax.jit
        def compute(params, XX, Q, keys):
            res, _ = stack.apply(params, XX, Q, keys)  # type: ignore
            return res

        # Process chunks
        predictions = []
        for chunk_id, XX in enumerate(x_chunks):
            Q = jax.random.uniform(key, (self.batch_size, stack.total_nb_of_outputs))
            keys = jax.random.split(key, self.batch_size)
            key = keys[-1]
            chunk_pred = jax.vmap(compute, in_axes=(None, 0, 0, 0))(params, XX, Q, keys)
            predictions.append(np.array(chunk_pred))

        all_predictions = np.concatenate(predictions, axis=0)

        # Split predictions by network
        network_predictions = []
        network_groundtruth = []
        for i, network in enumerate(networks):
            out_id = stack.get_network_global_output_id(i)
            n_out = network.get_nb_outputs()
            x, y = X[i], Y[i]
            yhat = all_predictions[: x.shape[0], out_id : out_id + n_out]

            # Create PlotData objects for ground truth and predictions
            metadata = {
                'network': network,
                'prediction_error': np.abs(y - yhat).mean(),
                'source_type': 'prediction',
            }

            pred_data = pu.PlotData(
                xval=x,
                yval=yhat,
                input_names=network.get_input_proteins(),
                output_name=network.get_output_proteins()[0],
                metadata=metadata,
            )

            truth_data = pu.PlotData(
                xval=x,
                yval=y,
                input_names=network.get_input_proteins(),
                output_name=network.get_output_proteins()[0],
                metadata=metadata,
            )

            network_predictions.append(pred_data)
            network_groundtruth.append(truth_data)

        return network_predictions, network_groundtruth

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def log_predictions(step, training_config, params=None, **kwargs):
            if params is None:
                return

            predictions, groundtruth = self.generate_predictions(params)
            prediction_path = self.save_predictions(step, predictions, groundtruth)

            # Return the path so WandBLogger can find and upload the images
            return {"prediction_path": prediction_path}

        return [(self.periods, log_predictions)]


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


funny_words = dr.load('pkg:biocomptools:configs/funnywords.yaml')

def generate_unique_name(directory: Path | str, prefix: str = '', suffix: str = '') -> str:
    """Generate a unique name for a file or directory in the given directory."""
    directory = Path(directory)
    adj = np.random.choice(funny_words['adjectives'])
    noun = np.random.choice(funny_words['nouns'])
    name = f'{prefix}{adj}-{noun}{suffix}'
    # add a number to the name if it already exists
    i = 1
    while (directory / name).exists():
        name = f'{prefix}{adj}-{noun}-{i}{suffix}'
        i += 1
    return name


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

    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])

    nan_mask = np.isnan(all_losses)
    filled_losses = ffill(all_losses)
    best_loss_id, smoothed_losses = get_best_smoothed_loss_id(filled_losses)

    yrange = np.nanmax(all_losses) - np.nanmin(all_losses)

    # plot non-nan values as blue solid lines
    colormap = plt.cm.get_cmap('tab10')
    lines = []
    for i in range(all_losses.shape[0]):
        non_nan_indices = ~nan_mask[i]
        l = ax.plot(
            np.arange(all_losses.shape[1])[non_nan_indices],
            all_losses[i, non_nan_indices],
            color='#AAA',
            linestyle='-',
            linewidth=1,
            alpha=0.5,
        )
        lines.append(l)

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
                color=colormap(i % 20),
            )

    ax.set_title(
        f'Loss history. Best loss with replicate {best_loss_id}, ~ {smoothed_losses[best_loss_id, -1]:.4f}'
    )

    try:
        labelLines(ax.get_lines(), zorder=2.5)
    except Exception as e:
        pass

    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')

    return fig, ax


def add_metadata(fig, ax, metadata: dict, run_name: str):
    """Add metadata to the figure in a clean, formatted way"""
    fig.suptitle(f'Run "{run_name}"')

    ax_meta = fig.add_subplot(fig.add_gridspec(1, 2, width_ratios=[3, 1])[1])
    ax_meta.set_axis_off()

    meta_text = '\n'.join(f'{k}: {v}' for k, v in metadata.items())
    ax_meta.text(0, 1, meta_text, va='top', ha='left', fontsize=8)

    plt.tight_layout()

    return fig


##────────────────────────────────────────────────────────────────────────────}}}
