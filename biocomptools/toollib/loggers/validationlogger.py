## {{{                          --     imports     --

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel, get_shared_params
from biocomptools.plot import NetworkPrediction
from biocomp.jaxutils import tree_get, tree_to_np
from biocomptools.toollib.networkselector import NetworkSet, build_data_manager
from biocomptools.toollib.datasources import DBSource
import biocomptools.toollib.models as md
from biocomp.compute import ComputeConfig
from biocomptools.run_training import TrainingProgram
from biocomptools.toollib.common import config
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
from sqlmodel import Field, SQLModel, create_engine, Relationship, Session
import pickle
from rich.console import Console
from rich.table import Table
from biocomp.utils import PartialFunction, load_lib, save
from biocomp.datautils import (
    DataConfig,
    DEFAULT_DATA_CONFIG,
    DataManager,
)

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class ValidationLossLogger(Logger):
    """
    Logs validation loss during training by evaluating model on validation set

    Can work in two modes:
    1. With TrainingProgram: Automatically extracts validation_set
    2. Standalone: Manually provide validation_set and configs
    """

    # validation configuration
    n_evals: int = 1500  # limit samples for efficiency
    batch_sampling_ratio: float = 0.1  # fraction of validation data to use each time

    # required components
    validation_set: Optional[NetworkSet] = None
    compute_conf: Optional[Union[ComputeConfig, Dict]] = None
    data_conf: Optional[Union[DataConfig, Dict]] = None
    n_replicates: int = 1  # default for standalone mode

    seed: int = 42

    # internal state
    _training_program: Optional[TrainingProgram] = None
    _console: Optional[Console] = None
    _history: Dict[int, List[Dict[str, float]]] = {}
    _base_model: Optional[NetworkModel] = None
    _predictor: Optional[NetworkPrediction] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._console = Console()

    def initialize(self, training_program):
        """Initialize from training program (traditional mode)"""
        if training_program is None:
            if self.validation_set is None:
                raise ValueError("In standalone mode, validation_set must be provided")
            if self.compute_conf is None or self.data_conf is None:
                raise ValueError("In standalone mode, compute_conf and data_conf must be provided")
            logger.info("ValidationLossLogger running in standalone mode")

        else:
            self._training_program = training_program
            self.compute_conf = training_program.compute_conf
            self.data_conf = training_program.data_conf
            self.n_replicates = training_program.training_conf.n_replicates

            if self.validation_set is None:
                self.validation_set = training_program.validation_set

        if self.validation_set is None:
            logger.warning("No validation set provided, ValidationLossLogger will be inactive")
            return

        self._initialize_predictor()

    def _initialize_predictor(self):
        assert isinstance(self.validation_set, NetworkSet)

        db_path = Path(config.db.sqlite.path).expanduser().resolve()
        engine = md.get_biocompdb_sqlite_engine(db_path)
        with Session(bind=engine) as session:
            self.validation_set.run_selectors(session)
            dman = build_data_manager(
                lib=load_lib(),
                db_session=session,
                path_prefix=Path(config.paths.root).expanduser().resolve(),
                data_conf=self.data_conf,
                dataset=self.validation_set,
            )
            xs, ys, networks = dman.get_per_network_xy_samples(self.n_evals)

        for x, n in zip(xs, networks):
            if x.shape[1] != n.get_nb_inputs():
                raise ValueError(
                    f"Network {n.name} has {n.get_nb_inputs()} inputs, "
                    f"but data has {x.shape[1]} features."
                )

        assert isinstance(self.compute_conf, ComputeConfig)
        assert isinstance(self.data_conf, DataConfig)

        model = BiocompModel(
            compute_config=self.compute_conf,
            rescaler=self.data_conf.rescaler,
        )
        self._base_model = NetworkModel(
            model=model,
            network=networks,
        )

        assert isinstance(self._base_model, NetworkModel)

        self._predictor = NetworkPrediction(
            predict_at=xs,
            network_model=self._base_model,
            ground_truth=ys,
            seed=self.seed,
            disable_variational=True,
            max_evals=self.n_evals,
        )

    def _compute_validation_metrics(self, params):
        """Compute validation metrics for current parameters"""

        assert isinstance(self._predictor, NetworkPrediction)

        all_metrics = []
        for i in range(self.n_replicates):
            stats = self._predictor.get_network_stats(with_shared_params=tree_get(params, i))

            valid_stats = [s for s in stats if s.get('rmse') is not None]

            if not valid_stats:
                logger.warning("No valid statistics computed")
                return None

            metrics = {
                'avg_rmse': float(np.mean([s['rmse'] for s in valid_stats])),
                'avg_mse': float(np.mean([s['mse'] for s in valid_stats])),
                'min_rmse': float(np.min([s['rmse'] for s in valid_stats])),
                'max_rmse': float(np.max([s['rmse'] for s in valid_stats])),
                'std_rmse': float(np.std([s['rmse'] for s in valid_stats])),
                'min_grid_rmse': float(np.min([s['grid_rmse'] for s in valid_stats])),
                'max_grid_rmse': float(np.max([s['grid_rmse'] for s in valid_stats])),
                'avg_grid_rmse': float(np.mean([s['grid_rmse'] for s in valid_stats])),
                'std_grid_rmse': float(np.std([s['grid_rmse'] for s in valid_stats])),
                'n_evaluated': len(valid_stats),
                'n_total': len(stats),
            }

            # add per-network stats if not too many
            if len(valid_stats) <= 20:
                metrics['per_network'] = {
                    s['network_name']: {'rmse': s['rmse'], 'mse': s['mse']} for s in valid_stats
                }

            all_metrics.append(metrics)

        return all_metrics

    def _print_validation_stats(self, step: int, metrics_list: List[Dict[str, float]]):
        """Print validation statistics in a nice table format"""
        n_replicates = len(metrics_list)

        if n_replicates == 1:
            # single replicate - show detailed stats
            metrics = metrics_list[0]
            table = Table(title=f"Validation Loss - Step {step}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Average RMSE", f"{metrics['avg_rmse']:.4f}")
            table.add_row("Average MSE", f"{metrics['avg_mse']:.4f}")
            table.add_row("Std RMSE", f"{metrics['std_rmse']:.4f}")
            table.add_row("Min RMSE", f"{metrics['min_rmse']:.4f}")
            table.add_row("Max RMSE", f"{metrics['max_rmse']:.4f}")
            table.add_row("Avg Grid RMSE", f"{metrics['avg_grid_rmse']:.4f}")
            table.add_row("Std Grid RMSE", f"{metrics['std_grid_rmse']:.4f}")
            table.add_row("Networks Evaluated", f"{metrics['n_evaluated']}/{metrics['n_total']}")
        else:
            # multiple replicates - show summary across replicates
            table = Table(title=f"Validation Loss - Step {step} ({n_replicates} replicates)")
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", style="green")
            table.add_column("Std", style="yellow")
            table.add_column("Min", style="blue")
            table.add_column("Max", style="red")

            # aggregate metrics across replicates
            metric_names = ['avg_rmse', 'avg_mse', 'avg_grid_rmse', 'std_rmse', 'std_grid_rmse']
            for metric_name in metric_names:
                values = [m[metric_name] for m in metrics_list]
                display_name = metric_name.replace('_', ' ').title()
                if metric_name == 'avg_rmse':
                    display_name = "Average RMSE"
                elif metric_name == 'avg_mse':
                    display_name = "Average MSE"
                elif metric_name == 'avg_grid_rmse':
                    display_name = "Avg Grid RMSE"
                elif metric_name == 'std_rmse':
                    display_name = "Std RMSE"
                elif metric_name == 'std_grid_rmse':
                    display_name = "Std Grid RMSE"

                table.add_row(
                    display_name,
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                )

            # show networks evaluated (should be same for all replicates)
            table.add_row(
                "Networks Evaluated",
                f"{metrics_list[0]['n_evaluated']}/{metrics_list[0]['n_total']}",
                "-",
                "-",
                "-",
            )

        self._console.print(table)

        # print improvement if we have history
        if len(self._history) > 0:
            prev_steps = sorted(self._history.keys())
            prev_metrics_list = self._history[prev_steps[-1]]

            # compare average RMSE across replicates
            curr_avg_rmse = np.mean([m['avg_rmse'] for m in metrics_list])
            prev_avg_rmse = np.mean([m['avg_rmse'] for m in prev_metrics_list])

            improvement = (prev_avg_rmse - curr_avg_rmse) / prev_avg_rmse * 100

            if improvement > 0:
                self._console.print(
                    f"[green]↓ Improvement: {improvement:.2f}% from step {prev_steps[-1]}[/green]"
                )
            else:
                self._console.print(
                    f"[red]↑ Degradation: {-improvement:.2f}% from step {prev_steps[-1]}[/red]"
                )

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        # initialize if needed
        self.initialize(training_program)

        def log_validation_loss(step, training_config, step_history=None, **kwargs):
            if step_history is None or 'latest_params' not in step_history:
                logger.warning("No latest params available for validation")
                return

            logger.info(f"Computing validation loss at step {step}...")

            params = step_history['latest_params']
            metrics_list = self._compute_validation_metrics(params)

            if metrics_list is None:
                logger.warning("Could not compute validation metrics")
                return

            self._history[step] = metrics_list
            self._print_validation_stats(step, metrics_list)

            # log detailed stats if requested (only for first replicate)
            if len(metrics_list) > 0:
                first_metrics = metrics_list[0]
                if 'per_network' in first_metrics and len(first_metrics['per_network']) <= 10:
                    logger.debug("Per-network validation stats (replicate 0):")
                    for network_name, stats in first_metrics['per_network'].items():
                        logger.debug(
                            f"  {network_name}: RMSE={stats['rmse']:.4f}, MSE={stats['mse']:.4f}"
                        )

        return [(self.periods, log_validation_loss)]

    def finalize(self):
        """Print final validation summary"""
        if not self._history:
            return

        # find best step based on average RMSE across all replicates
        best_step = min(
            self._history, key=lambda s: np.mean([m['avg_rmse'] for m in self._history[s]])
        )
        best_metrics_list = self._history[best_step]
        best_avg_rmse = np.mean([m['avg_rmse'] for m in best_metrics_list])

        self._console.print("\n[bold]Final Validation Summary[/bold]")
        self._console.print(f"Best validation RMSE: {best_avg_rmse:.4f} at step {best_step}")

        # show best replicate if multiple
        if len(best_metrics_list) > 1:
            best_replicate = min(
                range(len(best_metrics_list)), key=lambda i: best_metrics_list[i]['avg_rmse']
            )
            self._console.print(
                f"Best replicate: {best_replicate} with RMSE: "
                f"{best_metrics_list[best_replicate]['avg_rmse']:.4f}"
            )

        if len(self._history) > 1:
            table = Table(title="Validation Loss Trajectory")
            table.add_column("Step", style="cyan")
            table.add_column("Avg RMSE", style="green")
            table.add_column("Avg Grid RMSE", style="magenta")
            table.add_column("Std RMSE", style="yellow")
            table.add_column("Best Rep", style="blue")
            table.add_column("Networks", style="white")

            for step in sorted(self._history.keys()):
                metrics_list = self._history[step]
                rmse_values = [m['avg_rmse'] for m in metrics_list]
                grid_rmse_values = [m['avg_grid_rmse'] for m in metrics_list]
                avg_rmse = np.mean(rmse_values)
                avg_grid_rmse = np.mean(grid_rmse_values)
                std_rmse = np.std(rmse_values) if len(rmse_values) > 1 else 0.0
                best_rep = min(range(len(rmse_values)), key=lambda i: rmse_values[i])

                table.add_row(
                    str(step),
                    f"{avg_rmse:.4f}",
                    f"{avg_grid_rmse:.4f}",
                    f"{std_rmse:.4f}",
                    str(best_rep),
                    f"{metrics_list[0]['n_evaluated']}/{metrics_list[0]['n_total']}",
                )

            self._console.print(table)
