## {{{                          --     imports     --

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel, get_shared_params
from biocomptools.plot import NetworkPrediction
from biocomp.jaxutils import tree_get
from biocomptools.toollib.networkselector import NetworkSet, build_data_manager
from biocomptools.toollib.datasources import DBSource
import biocomptools.toollib.models as md
from biocomp.compute import ComputeConfig
from biocomptools.run_training import TrainingProgram
from biocomptools.toollib.common import config
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
from rich.console import Console
from rich.table import Table
from biocomp.utils import load_lib
from biocomp.datautils import DataConfig, DataManager
import matplotlib.pyplot as plt
from collections import defaultdict
from biocomptools.trainutils import ffill
from biocomptools.toollib.loggers.paramgradlogger import get_plot_rows_and_columns

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class ValidationLossLogger(Logger):
    """
    Logs validation loss during training by evaluating the model on a validation set.
    Generates a summary plot of validation loss history at each logging step.
    """

    # General configuration
    name: Optional[str] = None
    validation_set: Optional[NetworkSet] = None
    n_evals: int = 2048
    enable_gridstats: bool = False
    seed: int = 42
    predictor_n_stats_workers: int = 8
    plot_training_losses: bool = False

    # Plotting configuration
    save_plots: bool = True
    plot_dpi: int = 200

    # Required components (can be auto-filled from TrainingProgram)
    compute_conf: Optional[Union[ComputeConfig, Dict]] = None
    data_conf: Optional[Union[DataConfig, Dict]] = None
    n_replicates: int = 1

    # Internal state
    _training_program: Optional[TrainingProgram] = None
    _console: Optional[Console] = None
    _history: List[Dict[str, Any]] = []
    _predictor: Optional[NetworkPrediction] = None
    _xynetworks: Optional[Tuple] = None
    _plot_save_dir: Optional[Path] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._console = Console()

    def find_myself(self, training_program: Optional[TrainingProgram] = None):
        idx = 0
        if training_program:
            for i, logger in enumerate(training_program.loggers):
                if logger is self:
                    idx = i
                    break
        return idx

    def initialize(self, training_program):
        """Initialize from training program or standalone configuration."""

        if self.name is None:
            if training_program:
                idx = self.find_myself(training_program)
                self.name = f"loss_{idx}"
            else:
                self.name = "loss"

        if training_program:
            self._training_program = training_program
            self.compute_conf = training_program.compute_conf
            self.data_conf = training_program.data_conf
            self.n_replicates = training_program.training_conf.n_replicates
            if self.save_plots:
                self._plot_save_dir = Path(training_program._save_dir) / f"plots/val_{self.name}"
                self._plot_save_dir.mkdir(exist_ok=True, parents=True)
        elif self.compute_conf is None or self.data_conf is None:
            raise ValueError("In standalone mode, compute_conf and data_conf must be provided.")

        self._initialize_predictor()

    def _initialize_predictor(self):
        if self._xynetworks is None:
            assert isinstance(self.validation_set, NetworkSet)
            db_path = Path(config.db.sqlite.path).expanduser().resolve()
            engine = md.get_biocompdb_sqlite_engine(db_path)
            with md.Session(bind=engine) as session:
                self.validation_set.run_selectors(session)
                dman = build_data_manager(
                    lib=load_lib(),
                    db_session=session,
                    path_prefix=Path(config.paths.root).expanduser().resolve(),
                    data_conf=self.data_conf,
                    dataset=self.validation_set,
                )
                self._xynetworks = dman.get_per_network_xy_samples(self.n_evals)
                xs, ys, networks = self._xynetworks
                # print network names for debugging:
                logger.debug(
                    f"ValidationLossLogger {self.name} has {len(networks)} networks: {networks}"
                )

        xs, ys, networks = self._xynetworks
        assert isinstance(self.compute_conf, ComputeConfig) and isinstance(
            self.data_conf, DataConfig
        )
        model = BiocompModel(compute_config=self.compute_conf, rescaler=self.data_conf.rescaler)
        network_model = NetworkModel(model=model, network=networks)

        self._predictor = NetworkPrediction(
            predict_at=xs,
            network_model=network_model,
            ground_truth=ys,
            seed=self.seed,
            disable_variational=True,
            max_evals=self.n_evals,
            already_latent=True,
            n_stats_workers=self.predictor_n_stats_workers,
            enable_gridstats=self.enable_gridstats,
        )

        self.metadata = {
            'validation_name': self.name,
            'validation_set': {
                'content': self.validation_set.content,
                'name': self.validation_set.name,
            },
        }

    def _get_replicate_metrics(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to format metrics for a single replicate."""
        result = {'RMSE': float(metrics_dict.get('avg_rmse', np.nan))}
        if self.enable_gridstats and 'avg_grid_rmse' in metrics_dict:
            result['grid_RMSE'] = float(metrics_dict['avg_grid_rmse'])

        per_network_data = metrics_dict.get('per_network', {})
        if per_network_data:
            result['per_network'] = {
                name: {'RMSE': stats['rmse']} for name, stats in per_network_data.items()
            }
        return result

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Return the latest validation metrics.
        - If replicate is None, returns a list of metrics for all replicates.
        - If replicate is an int, returns metrics for that specific replicate.
        """
        if not self._history:
            return None

        latest_run = self._history[-1]
        latest_metrics_list = latest_run.get('metrics')
        if not latest_metrics_list:
            return None

        if replicate is not None:
            if replicate < len(latest_metrics_list):
                rep_metrics = self._get_replicate_metrics(latest_metrics_list[replicate])
                return {f'{self.name}_validation_loss': rep_metrics}
            else:
                logger.warning(
                    f"Replicate index {replicate} out of bounds for ValidationLossLogger."
                )
                return None
        else:
            all_reps_metrics = [self._get_replicate_metrics(m) for m in latest_metrics_list]
            return {f'validation::{self.name}': all_reps_metrics}

    def _compute_validation_metrics(self, params) -> Tuple[Optional[List[Dict]], float]:
        from time import time

        assert self._predictor is not None
        all_metrics = []
        t0 = time()
        for i in range(self.n_replicates):
            stats = self._predictor.get_network_stats(with_shared_params=tree_get(params, i))
            valid_stats = [s for s in stats if s.get('rmse') is not None]
            if not valid_stats:
                logger.warning(f"No valid statistics computed for replicate {i}")
                continue

            metrics = {
                'avg_rmse': float(np.mean([s['rmse'] for s in valid_stats])),
                'per_network': {s['network_name']: {'rmse': s['rmse']} for s in valid_stats},
                'n_evaluated': len(valid_stats),
            }
            if self.enable_gridstats:
                grid_rmses = [
                    s.get('grid_rmse') for s in valid_stats if s.get('grid_rmse') is not None
                ]
                if grid_rmses:
                    metrics['avg_grid_rmse'] = float(np.mean(grid_rmses))
            all_metrics.append(metrics)

        eval_time = time() - t0
        return (all_metrics, eval_time) if all_metrics else (None, eval_time)

    def _print_validation_stats(self, step: int, metrics_list: List[Dict], eval_time: float):
        table = Table(
            title=f"{self.name.title()} Loss - Step {step} ({metrics_list[0]['n_evaluated']} networks) in {eval_time:.2f}s"
        )
        table.add_column("Replicate", style="cyan", justify="right")
        table.add_column("Avg RMSE", style="green", justify="right")
        if self.enable_gridstats:
            table.add_column("Avg Grid RMSE", style="yellow", justify="right")

        for i, metrics in enumerate(metrics_list):
            row = [str(i), f"{metrics['avg_rmse']:.4f}"]
            if self.enable_gridstats:
                row.append(f"{metrics.get('avg_grid_rmse', np.nan):.4f}")
            table.add_row(*row)
        self._console.print(table)

        if len(self._history) > 1:
            prev_item = self._history[-2]
            curr_avg = np.nanmean([m['avg_rmse'] for m in metrics_list])
            prev_avg = np.nanmean([m['avg_rmse'] for m in prev_item['metrics']])
            if prev_avg > 0 and not np.isnan(prev_avg):
                improvement = (prev_avg - curr_avg) / prev_avg * 100
                if abs(improvement) > 0.01:
                    color = "green" if improvement > 0 else "red"
                    symbol = "▲" if improvement > 0 else "▼"
                    self._console.print(
                        f"[{color}]  {symbol} {improvement:+.2f}% (vs step {prev_item['step']})[/{color}]"
                    )

    def _plot_history(self, step: int):
        if not self._history or self._plot_save_dir is None:
            return

        steps = [h['step'] for h in self._history]
        all_net_names = sorted(
            {name for h in self._history for r in h['metrics'] for name in r.get('per_network', {})}
        )
        n_nets = len(all_net_names)
        loss_data = np.full((n_nets + 1, self.n_replicates, len(steps)), np.nan)

        for s_idx, h in enumerate(self._history):
            for r_idx, r_metrics in enumerate(h['metrics']):
                loss_data[0, r_idx, s_idx] = r_metrics.get('avg_rmse')
                for n_idx, n_name in enumerate(all_net_names):
                    loss_data[n_idx + 1, r_idx, s_idx] = (
                        r_metrics.get('per_network', {}).get(n_name, {}).get('rmse')
                    )

        n_rows, n_cols = get_plot_rows_and_columns(n_nets, ideal_ratio=1.0)
        fig = plt.figure(figsize=(5 * n_cols, 2.5 * (n_rows + 2)), dpi=self.plot_dpi)
        gs = fig.add_gridspec(n_rows + 2, n_cols, hspace=0.5, wspace=0.5)
        fig.suptitle(f'{self.name.title()} Loss History - Step {step}', fontsize=14)

        ax_main = fig.add_subplot(gs[0:2, 0:2])
        self._plot_single_metric(
            ax_main,
            loss_data[0],
            steps,
            'Overall Average RMSE',
            is_main=True,
            training_steps=steps,
            training_history=self._history,
        )

        axes_flat = [fig.add_subplot(gs[r + 2, c]) for r in range(n_rows) for c in range(n_cols)]
        for i, ax in enumerate(axes_flat):
            if i >= n_nets:
                ax.set_visible(False)
                continue
            self._plot_single_metric(ax, loss_data[i + 1], steps, all_net_names[i])

        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self._plot_save_dir / f"history_step_{step:05d}.png")
        plt.close(fig)

    def _plot_single_metric(
        self, ax, data, steps, title, is_main=False, training_steps=None, training_history=None
    ):
        filled_data = ffill(data)
        cmap = plt.cm.get_cmap('tab10')

        # Plot validation loss lines (colored)
        for i in range(self.n_replicates):
            ax.plot(
                steps,
                data[i, :],
                color=cmap(i % 10),
                linestyle='-',
                linewidth=1.5 if is_main else 0.5,
                alpha=1.0,
                label=f'Val Rep {i}' if is_main else None,
            )

        # Add training loss line for main plot (grey)
        if is_main and training_history is not None and self.plot_training_losses:
            training_losses = []
            train_steps = []
            for h in training_history:
                if h.get('training_loss') is not None:
                    train_steps.append(h['step'])
                    # Average training loss across replicates
                    loss_array = h['training_loss']
                    if hasattr(loss_array, 'mean'):
                        avg_loss = float(loss_array.mean())
                    else:
                        avg_loss = float(np.mean(loss_array))
                    training_losses.append(avg_loss)

            if training_losses:
                ax.plot(
                    train_steps,
                    training_losses,
                    color='grey',
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.7,
                    label='Training Loss',
                )

        ax.set_title(title, fontsize=11 if is_main else 7)
        ax.set_yscale('log')
        ax.set_xlabel('Step')
        ax.set_ylabel('RMSE (log scale)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if is_main and (self.n_replicates > 1 or training_history is not None):
            ax.legend(fontsize='small')

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        self.initialize(training_program)

        def log_validation_loss(step, training_config, step_history=None, **kwargs):
            if step_history is None or 'latest_params' not in step_history:
                if step == 0:
                    logger.debug("No latest params available for validation at step 0 (expected)")
                else:
                    logger.warning(f"No latest params available for validation at step {step}")
                return

            logger.info(f"Computing {self.name} loss at step {step}...")
            metrics_list, eval_time = self._compute_validation_metrics(
                step_history['latest_params']
            )

            if metrics_list is None:
                logger.warning(f"Could not compute {self.name} metrics")
                return

            # Store training loss alongside validation metrics
            training_loss = step_history.get('loss')
            self._history.append(
                {'step': step, 'metrics': metrics_list, 'training_loss': training_loss}
            )
            self._print_validation_stats(step, metrics_list, eval_time)
            if self.save_plots:
                self._plot_history(step)

        return [(self.periods, log_validation_loss)]

    def finalize(self):
        """Create video from validation loss history plots using ffmpeg."""
        if self._plot_save_dir is None:
            return

        from biocomptools.toollib.video_utils import create_video_from_plots

        logger.info("ValidationLossLogger: Creating video from validation plots...")

        video_path = self._plot_save_dir / "validation_history_video.mp4"
        video_created = create_video_from_plots(
            plot_dir=self._plot_save_dir,
            output_path=video_path,
            plot_pattern="history_step_*.png",
        )

        if video_created:
            logger.info(f"ValidationLossLogger: Created validation video: {video_path}")
        else:
            logger.debug("ValidationLossLogger: No video created (insufficient plots or errors)")
