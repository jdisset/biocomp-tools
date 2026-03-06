from __future__ import annotations

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.plot import NetworkPrediction
from biocomp.jaxutils import tree_get
from biocomptools.toollib.networkselector import NetworkSet, build_data_manager
import biocomptools.toollib.models as md
from sqlmodel import Session
from biocomp.compute import ComputeConfig
from biocomptools.toollib.common import config
from biocomp.metric_utils import GridStatsFields
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from pydantic import PrivateAttr
from rich.console import Console
from rich.table import Table
from biocomp.library import load_lib
from biocomp.datautils import DataConfig, DataManager
import time

if TYPE_CHECKING:
    from biocomptools.run_training import TrainingProgram

logger = get_logger(__name__)


class ValidationLossLogger(GridStatsFields, Logger):
    name: str | None = None
    validation_set: NetworkSet | None = None
    n_evals: int = 2048
    enable_gridstats: bool = False
    seed: int = 42
    predictor_n_stats_workers: int = 1
    plot_training_losses: bool = False
    device: Literal["cpu", "gpu"] = "cpu"
    update_xynetworks: bool = True
    save_plots: bool = True
    plot_dpi: int = 200
    compute_conf: ComputeConfig | dict | None = None
    data_conf: DataConfig | dict | None = None
    n_replicates: int = 1
    execution_mode: Literal["inline", "thread", "process"] = "inline"
    required_arrays: list[str] = ["latest_params"]

    _dman: DataManager | None = PrivateAttr(default=None)
    _training_program: TrainingProgram | None = PrivateAttr(default=None)
    _console: Console | None = PrivateAttr(default=None)
    _history: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _predictor: NetworkPrediction | None = PrivateAttr(default=None)
    _xynetworks: tuple | None = PrivateAttr(default=None)
    _plot_save_dir: Path | None = PrivateAttr(default=None)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._console = Console()

    # find_myself() inherited from Logger base class

    def initialize(self, training_program):
        if self.name is None:
            self.name = f"loss_{self.find_myself(training_program)}" if training_program else "loss"

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

        logger.info(f"ValidationLossLogger {self.name}: initializing predictor")
        t0 = time.time()
        self._initialize_predictor()
        logger.info(f"ValidationLossLogger {self.name}: initialized in {time.time() - t0:.2f}s")

    def _initialize_predictor(self, force_reinit: bool = False):
        if self._predictor is not None and not force_reinit and self._xynetworks is not None:
            return

        if self._dman is None:
            assert isinstance(self.validation_set, NetworkSet)
            db_path = Path(config.db.sqlite.path).expanduser().resolve()
            engine = md.get_biocompdb_sqlite_engine(db_path)
            with Session(bind=engine) as session:
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

        assert self._xynetworks is not None
        xs, ys, networks = self._xynetworks
        assert isinstance(self.compute_conf, ComputeConfig) and isinstance(
            self.data_conf, DataConfig
        )

        model = BiocompModel(compute_config=self.compute_conf, rescaler=self.data_conf.rescaler)
        network_model = NetworkModel(model=model, network=networks)

        per_prediction_info = [
            {
                "network_name": n.name,
                "networkdatapair": {
                    "network_name": n.name,
                    "datafile_path": n.metadata.get("data_file", "unknown"),
                },
            }
            for n in networks
        ]

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
            device=self.device,
            gridstats_hypercube_res=self.gridstats_hypercube_res,
            gridstats_hypercube_min=self.gridstats_hypercube_min,
            gridstats_hypercube_max=self.gridstats_hypercube_max,
            gridstats_k=self.gridstats_k,
            gridstats_radius=self.gridstats_radius,
            gridstats_min_points=self.gridstats_min_points,
        )

        self.metadata = {
            "validation_name": self.name,
            "validation_set": {
                "content": self.validation_set.content,
                "name": self.validation_set.name,
            },
        }

    def _get_replicate_metrics(self, metrics_dict: dict[str, Any]) -> dict[str, Any]:
        result = {"RMSE": float(metrics_dict.get("avg_rmse", np.nan))}
        if self.enable_gridstats:
            for k, v in [
                ("avg_grid_rmse", "grid_RMSE"),
                ("mean_nrmse", "mean_nRMSE"),
                ("geomean_nrmse", "geomean_nRMSE"),
                ("softmax_nrmse", "softmax_nRMSE"),
                ("avg_grid_snr", "grid_SNR"),
            ]:
                if k in metrics_dict:
                    result[v] = float(metrics_dict[k])

        per_network_data = metrics_dict.get("per_network", [])
        if per_network_data:
            result["per_network"] = []
            for nd in per_network_data:
                nm = {"network_name": nd["network_name"], "RMSE": nd["rmse"]}
                if "networkdatapair" in nd:
                    nm["networkdatapair"] = nd["networkdatapair"]
                if self.enable_gridstats:
                    for k, v in [
                        ("grid_rmse", "grid_RMSE"),
                        ("grid_nrmse", "grid_nRMSE"),
                        ("grid_snr", "grid_SNR"),
                    ]:
                        if k in nd:
                            nm[v] = nd[k]
                result["per_network"].append(nm)
        return result

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        if not self._history:
            return None
        latest = self._history[-1].get("metrics")
        if not latest:
            return None
        if replicate is not None:
            if replicate < len(latest):
                return {
                    f"{self.name}_validation_loss": self._get_replicate_metrics(latest[replicate])
                }
            return None
        return {f"validation::{self.name}": [self._get_replicate_metrics(m) for m in latest]}

    def _compute_validation_metrics(self, params) -> tuple[list[dict] | None, float]:
        from scipy.stats import gmean

        assert self._predictor is not None
        all_metrics = []
        t0 = time.time()

        for i in range(self.n_replicates):
            stats = self._predictor.get_network_stats(with_shared_params=tree_get(params, i))
            valid_stats = [s for s in stats if s.get("rmse") is not None]
            if not valid_stats:
                continue

            metrics = {
                "avg_rmse": float(np.mean([s["rmse"] for s in valid_stats])),
                "n_evaluated": len(valid_stats),
            }
            per_network_list = []
            for s in valid_stats:
                nm = {"rmse": s["rmse"], "network_name": s["network_name"]}
                if "extra_prediction_info" in s and "networkdatapair" in s["extra_prediction_info"]:
                    nm["networkdatapair"] = s["extra_prediction_info"]["networkdatapair"]
                per_network_list.append(nm)
            metrics["per_network"] = per_network_list

            if self.enable_gridstats:
                grid_rmses = [
                    s.get("grid_rmse") for s in valid_stats if s.get("grid_rmse") is not None
                ]
                grid_nrmses = np.array(
                    [
                        s.get("grid_nrmse")
                        for s in valid_stats
                        if s.get("grid_nrmse") is not None and np.isfinite(s.get("grid_nrmse"))
                    ]
                )
                grid_snrs = [
                    s.get("grid_snr") for s in valid_stats if s.get("grid_snr") is not None
                ]

                if grid_rmses:
                    metrics["avg_grid_rmse"] = float(np.mean(grid_rmses))
                if len(grid_nrmses) > 0:
                    metrics["mean_nrmse"] = float(np.mean(grid_nrmses))
                    positive = grid_nrmses[grid_nrmses > 0]
                    if len(positive) > 0:
                        metrics["geomean_nrmse"] = float(gmean(positive))
                    alpha = 5.0
                    max_val = np.max(grid_nrmses)
                    metrics["softmax_nrmse"] = float(
                        max_val
                        + (1 / alpha) * np.log(np.sum(np.exp(alpha * (grid_nrmses - max_val))))
                    )
                    metrics["avg_grid_nrmse"] = metrics["mean_nrmse"]
                if grid_snrs:
                    metrics["avg_grid_snr"] = float(np.mean(grid_snrs))

                for j, s in enumerate(valid_stats):
                    if j < len(per_network_list):
                        for k in ["grid_rmse", "grid_nrmse", "grid_snr"]:
                            if s.get(k) is not None:
                                per_network_list[j][k] = s[k]

            all_metrics.append(metrics)

        return (all_metrics, time.time() - t0) if all_metrics else (None, time.time() - t0)

    def _print_validation_stats(self, step: int, metrics_list: list[dict], eval_time: float):
        table = Table(
            title=f"{self.name.title()} Loss - Step {step} ({metrics_list[0]['n_evaluated']} networks) in {eval_time:.2f}s"
        )
        table.add_column("Rep", style="cyan", justify="right")
        table.add_column("RMSE", style="green", justify="right")
        if self.enable_gridstats:
            for col in ["gRMSE", "mean", "geomean", "softmax", "SNR"]:
                table.add_column(
                    col,
                    style="magenta"
                    if col not in ["gRMSE", "SNR"]
                    else ("yellow" if col == "gRMSE" else "blue"),
                    justify="right",
                )

        for i, m in enumerate(metrics_list):
            row = [str(i), f"{m['avg_rmse']:.4f}"]
            if self.enable_gridstats:
                row += [
                    f"{m.get('avg_grid_rmse', np.nan):.4f}",
                    f"{m.get('mean_nrmse', np.nan):.3f}",
                    f"{m.get('geomean_nrmse', np.nan):.3f}",
                    f"{m.get('softmax_nrmse', np.nan):.3f}",
                    f"{m.get('avg_grid_snr', np.nan):.1f}",
                ]
            table.add_row(*row)
        self._console.print(table)

        if metrics_list and metrics_list[0].get("per_network"):
            net_table = Table(title=f"Per-Network Validation (Replicate 0) - Step {step}")
            net_table.add_column("Network", style="cyan")
            net_table.add_column("RMSE", style="green", justify="right")
            for nm in metrics_list[0]["per_network"]:
                name = (
                    nm["network_name"][:50] + "..."
                    if len(nm["network_name"]) > 50
                    else nm["network_name"]
                )
                net_table.add_row(name, f"{nm['rmse']:.6f}")
            self._console.print(net_table)

        if len(self._history) > 1:
            prev = self._history[-2]
            curr_avg = np.nanmean([m["avg_rmse"] for m in metrics_list])
            prev_avg = np.nanmean([m["avg_rmse"] for m in prev["metrics"]])
            if prev_avg > 0 and not np.isnan(prev_avg):
                improvement = (prev_avg - curr_avg) / prev_avg * 100
                if abs(improvement) > 0.01:
                    color = "green" if improvement > 0 else "red"
                    symbol = "▲" if improvement > 0 else "▼"
                    self._console.print(
                        f"[{color}]  {symbol} {improvement:+.2f}% (vs step {prev['step']})[/{color}]"
                    )

    def _plot_history(self, step: int):
        if not self._history or self._plot_save_dir is None:
            return
        from biocomptools.toollib.loggers.plotting_utils import MetricsPlotter

        output_path = self._plot_save_dir / f"val_{self.name}_{step:05d}.png"
        training_id = (
            getattr(self._training_program, "training_id", None) if self._training_program else None
        )
        MetricsPlotter.plot_validation_history(
            self._history,
            f"Validation Loss ({self.name.title()})",
            output_path,
            self.name,
            training_id=training_id,
        )

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        if self._predictor is None:
            self._initialize_predictor()

        step_history = view.to_step_history()
        if "latest_params" not in step_history:
            return

        step = context.current_step
        logger.info(f"ValidationLossLogger {self.name}: computing validation at step {step}")
        metrics_list, eval_time = self._compute_validation_metrics(step_history["latest_params"])
        if metrics_list is None:
            return

        self._history.append(
            {"step": step, "metrics": metrics_list, "training_loss": step_history.get("loss")}
        )
        self._print_validation_stats(step, metrics_list, eval_time)

        if self.save_plots:
            self._plot_history(step)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self.on_batch(view, context)

    def finalize(self):
        if self._plot_save_dir is None:
            return
        from biocomptools.toollib.video_utils import create_video_from_plots

        video_path = self._plot_save_dir / "validation_history_video.mp4"
        create_video_from_plots(self._plot_save_dir, video_path, "history_step_*.png")
