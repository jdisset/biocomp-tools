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
import time
import jax.numpy as jnp

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
    predictor_n_stats_workers: int = 1
    plot_training_losses: bool = False

    update_xynetworks: bool = True

    # Plotting configuration
    save_plots: bool = True
    plot_dpi: int = 200

    # Required components (can be auto-filled from TrainingProgram)
    compute_conf: Optional[Union[ComputeConfig, Dict]] = None
    data_conf: Optional[Union[DataConfig, Dict]] = None
    n_replicates: int = 1

    # Internal state
    _dman: Optional[DataManager] = None
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
        
        # Initialize predictor during async initialization phase
        logger.info(f"ValidationLossLogger {self.name}: Starting async initialization of predictor and NetworkModel")
        start_time = time.time()
        self._initialize_predictor()
        init_time = time.time() - start_time
        logger.info(f"ValidationLossLogger {self.name}: Completed async initialization in {init_time:.2f} seconds")


    def _initialize_predictor(self, force_reinit: bool = False):
        if self._predictor is not None and not force_reinit and self._xynetworks is not None:
            logger.debug(f"ValidationLossLogger {self.name} already initialized.")
            return
        if self._dman is None:
            assert isinstance(self.validation_set, NetworkSet)
            db_path = Path(config.db.sqlite.path).expanduser().resolve()
            engine = md.get_biocompdb_sqlite_engine(db_path)
            with md.Session(bind=engine) as session:
                self.validation_set.run_selectors(session)
                self._dman = build_data_manager(
                    lib=load_lib(),
                    db_session=session,
                    path_prefix=Path(config.paths.root).expanduser().resolve(),
                    data_conf=self.data_conf,
                    dataset=self.validation_set,
                    jax_sampling=False,
                )
            self._xynetworks = self._dman.get_per_network_xy_samples(self.n_evals)

        assert self._xynetworks is not None, "No xynetworks available for validation."
        xs, ys, networks = self._xynetworks
        xshapes = [x.shape for x in xs]
        yshapes = [y.shape for y in ys]
        logger.info(
            f"ValidationLossLogger {self.name} initialized with {len(networks)} networks, "
            f"input shapes: {xshapes}, output shapes: {yshapes}"
        )

        assert isinstance(self.compute_conf, ComputeConfig) and isinstance(
            self.data_conf, DataConfig
        )

        model = BiocompModel(compute_config=self.compute_conf, rescaler=self.data_conf.rescaler)
        network_model = NetworkModel(model=model, network=networks)

        # prepare per_prediction_info with networkdatapair information
        per_prediction_info = []
        for i, network in enumerate(networks):
            network_info = {
                'network_name': network.name,
                'networkdatapair': {
                    'network_name': network.name,
                    'datafile_path': network.metadata.get('data_file', 'unknown'),
                },
            }
            per_prediction_info.append(network_info)

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
            per_prediction_info=per_prediction_info,
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

        per_network_data = metrics_dict.get('per_network', [])
        if per_network_data:
            # convert to list format with networkdatapair info
            result['per_network'] = []
            for network_data in per_network_data:
                network_metric = {
                    'network_name': network_data['network_name'],
                    'RMSE': network_data['rmse'],
                }
                if 'networkdatapair' in network_data:
                    network_metric['networkdatapair'] = network_data['networkdatapair']
                if self.enable_gridstats and 'grid_rmse' in network_data:
                    network_metric['grid_RMSE'] = network_data['grid_rmse']
                result['per_network'].append(network_metric)
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
            replicate_start = time()
            stats = self._predictor.get_network_stats(with_shared_params=tree_get(params, i))
            replicate_time = time() - replicate_start

            valid_stats = [s for s in stats if s.get('rmse') is not None]
            if not valid_stats:
                logger.warning(f"No valid statistics computed for replicate {i}")
                continue

            # overall metrics
            metrics = {
                'avg_rmse': float(np.mean([s['rmse'] for s in valid_stats])),
                'n_evaluated': len(valid_stats),
            }

            # per-network list with networkdatapair info
            per_network_list = []
            for s in valid_stats:
                network_metric = {
                    'rmse': s['rmse'],
                    'network_name': s['network_name'],
                }

                # add networkdatapair info from extra_prediction_info
                if 'extra_prediction_info' in s and 'networkdatapair' in s['extra_prediction_info']:
                    network_metric['networkdatapair'] = s['extra_prediction_info'][
                        'networkdatapair'
                    ]
                    # assertion to verify network_name consistency
                    assert network_metric['networkdatapair']['network_name'] == s['network_name'], (
                        f"Network name mismatch: {network_metric['networkdatapair']['network_name']} != {s['network_name']}"
                    )

                per_network_list.append(network_metric)

            metrics['per_network'] = per_network_list

            if self.enable_gridstats:
                grid_rmses = [
                    s.get('grid_rmse') for s in valid_stats if s.get('grid_rmse') is not None
                ]
                if grid_rmses:
                    metrics['avg_grid_rmse'] = float(np.mean(grid_rmses))
                    # also add grid_rmse to per_network_list
                    for j, s in enumerate(valid_stats):
                        if s.get('grid_rmse') is not None and j < len(per_network_list):
                            per_network_list[j]['grid_rmse'] = s['grid_rmse']

            all_metrics.append(metrics)

            logger.info(
                f"ValidationLossLogger {self.name}: Replicate {i} validation completed in {replicate_time:.2f} seconds"
            )

        eval_time = time() - t0
        logger.info(
            f"ValidationLossLogger {self.name}: Total validation time: {eval_time:.2f} seconds for {self.n_replicates} replicates"
        )
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

        # Debug: print per-network validation details for first replicate
        if metrics_list and metrics_list[0].get('per_network'):
            per_net = metrics_list[0]['per_network']
            net_table = Table(title=f"Per-Network Validation (Replicate 0) - Step {step}")
            net_table.add_column("Network", style="cyan")
            net_table.add_column("RMSE", style="green", justify="right")

            for net_metrics in per_net:
                net_table.add_row(
                    net_metrics['network_name'][:50] + "..." if len(net_metrics['network_name']) > 50 else net_metrics['network_name'],
                    f"{net_metrics['rmse']:.6f}",
                )

            self._console.print(net_table)

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
        """Plot validation history using improved plotting functionality."""
        if not self._history or self._plot_save_dir is None:
            return
    
        from time import time
        from biocomptools.toollib.loggers.plotting_utils import MetricsPlotter
    
        plot_start_time = time()
    
        # Use the improved MetricsPlotter for consistent styling
        output_path = self._plot_save_dir / f"val_{self.name}_{step:05d}.png"
        
        # Get training_id from training program if available
        training_id = getattr(self._training_program, 'training_id', None) if self._training_program else None
        
        MetricsPlotter.plot_validation_history(
            self._history,
            f"Validation Loss ({self.name.title()})",
            output_path, 
            self.name,
            training_id=training_id
        )
    
        plot_time = time() - plot_start_time
        logger.info(
            f"ValidationLossLogger {self.name}: Plot generation completed in {plot_time:.2f} seconds"
        )



    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        self.initialize(training_program)

        def log_validation_loss(step, training_config, step_history=None, **kwargs):
            total_start_time = time.time()

            # Check if predictor was already initialized during async phase
            if self._predictor is None:
                init_start = time.time()
                self._initialize_predictor()
                init_time = time.time() - init_start
                logger.info(
                    f"ValidationLossLogger {self.name}: Late initialization took {init_time:.2f} seconds"
                )
            else:
                logger.debug(f"ValidationLossLogger {self.name}: Using predictor initialized during async phase")

            if step_history is None or 'latest_params' not in step_history:
                if step == 0:
                    logger.debug("No latest params available for validation at step 0 (expected)")
                else:
                    logger.warning(f"No latest params available for validation at step {step}")
                return

            logger.info(
                f"ValidationLossLogger {self.name}: Computing validation loss at step {step}..."
            )

            eval_start = time.time()
            metrics_list, eval_time = self._compute_validation_metrics(
                step_history['latest_params']
            )
            eval_total_time = time.time() - eval_start
            logger.info(
                f"ValidationLossLogger {self.name}: Evaluation phase took {eval_total_time:.2f} seconds"
            )

            if metrics_list is None:
                logger.warning(f"Could not compute {self.name} metrics")
                return

            # Store training loss alongside validation metrics
            training_loss = step_history.get('loss')
            self._history.append(
                {'step': step, 'metrics': metrics_list, 'training_loss': training_loss}
            )

            print_start = time.time()
            self._print_validation_stats(step, metrics_list, eval_time)
            print_time = time.time() - print_start
            logger.info(
                f"ValidationLossLogger {self.name}: Stats printing took {print_time:.2f} seconds"
            )

            if self.save_plots:
                plot_start = time.time()
                self._plot_history(step)
                plot_time = time.time() - plot_start
                logger.info(
                    f"ValidationLossLogger {self.name}: Plot generation took {plot_time:.2f} seconds"
                )

            total_time = time.time() - total_start_time
            logger.info(
                f"ValidationLossLogger {self.name}: Total callback time {total_time:.2f} seconds"
            )

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

