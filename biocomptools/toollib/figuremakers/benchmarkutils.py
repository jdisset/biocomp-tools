"""Benchmark summary utilities."""

from dataclasses import dataclass
from typing import Literal, Optional, Any, Union
import matplotlib.axes
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
from scipy.stats import gmean

from biocomp.plotutils import PlotData
from biocomp.metric_utils import GridStatsFields
from biocomptools.toollib.networkprediction import PredictionSamplingConfig

IN_TRAINING_COLOR = "#EEEEEE"
NOT_IN_TRAINING_COLOR = "#EEEEEE"
GOOD_COLOR = "#28a745"
BAD_COLOR = "#dc3545"
RMSE_THRESHOLD = 0.1
NRMSE_THRESHOLD = 1.0


@dataclass
class BenchmarkItem:
    idx: int
    gt_data: PlotData
    pred_data: PlotData
    network: object
    network_name: str
    rmse: Optional[float]
    nrmse: Optional[float]
    snr: Optional[float]
    in_training: bool

    @property
    def file_prefix(self) -> str:
        return f"{str(self.idx + 1).zfill(2)}_"


class BenchmarkData(GridStatsFields, BaseModel):
    """Benchmark computation for a model on a dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Optional[str] = None
    model_name: Optional[str] = None
    model: Optional[Any] = None
    dataset_file: Optional[str] = None
    max_items: Optional[int] = None
    device: str = 'gpu'
    max_evals: int = 250000
    # gridstats_* fields inherited from GridStatsFields mixin

    # z_value controls the latent noise distribution for distributional models
    z_value: Union[Literal['uniform', 'normal'], float] = 'uniform'
    z_normal_mean: float = 0.5
    z_normal_std: float = 0.2
    z_normal_clip: bool = True
    disable_variational: bool = True

    # Grouped sampling config (wins over scalar fields above if supplied).
    sampling: Optional[PredictionSamplingConfig] = None

    _model: Any = PrivateAttr(default=None)
    _items: list[BenchmarkItem] = PrivateAttr(default_factory=list)
    _dataset_name: str = PrivateAttr(default="")
    _aggregate_stats: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode='after')
    def _initialize_after(self):
        self._do_initialize()
        return self

    def _do_initialize(self):
        from biocomptools.modelmodel import BiocompModel, NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction
        from biocomptools.toollib.datasources import DBSource
        from biocomptools.toollib.networkselector import (
            NetworkSet,
            NetworkSetUnion,
            NetworkSetDifference,
            NetworkSetIntersection,
            CleanupFilter,
            NetworkFilter,
            Regex,
            iRegex,
        )
        from dracon import load

        if self.model is not None:
            self._model = self.model
        else:
            self._model = BiocompModel.resolve(
                name=self.model_name or None, path=self.model_path
            )

        self._dataset_name = (
            self.dataset_file.split('/')[-1].rsplit('.', 1)[0] if self.dataset_file else "unknown"
        )

        ctx = {
            'NetworkSet': NetworkSet,
            'NetworkSetUnion': NetworkSetUnion,
            'NetworkSetDifference': NetworkSetDifference,
            'NetworkSetIntersection': NetworkSetIntersection,
            'CleanupFilter': CleanupFilter,
            'NetworkFilter': NetworkFilter,
            'Regex': Regex,
            'iRegex': iRegex,
            'DBSource': DBSource,
        }
        dataset = load(self.dataset_file, context=ctx)
        ground_truth = (
            dataset.get_data()
            if hasattr(dataset, 'get_data')
            else DBSource(content=dataset).get_data()
        )

        max_n = int(self.max_items) if self.max_items else None
        items_to_plot = ground_truth[:max_n] if max_n else ground_truth

        networks = [d.metadata['built_network'] for d in items_to_plot]
        network_model = NetworkModel(model=self._model, network=networks)

        # d.x is in display order (alphabetical); NetworkPrediction expects network order.
        # input_order maps network→display, so argsort inverts it back. column_proteins
        # rides along through the same inversion so the boundary assertion in
        # NetworkPrediction can independently verify the restored wiring.
        predict_at_network_order = []
        predict_at_column_proteins: list[list[str] | None] = []
        for d in items_to_plot:
            x_display = np.asarray(d.x)
            io = d.metadata.get('input_order')
            if io is not None:
                inv = np.argsort(io)
                predict_at_network_order.append(x_display[:, inv])
                cp = d.column_proteins
                predict_at_column_proteins.append(
                    [cp[int(i)] for i in inv] if cp is not None else None
                )
            else:
                predict_at_network_order.append(x_display)
                predict_at_column_proteins.append(
                    list(d.column_proteins) if d.column_proteins is not None else None
                )

        predictor = NetworkPrediction(
            predict_at=predict_at_network_order,
            predict_at_column_proteins=predict_at_column_proteins,
            ground_truth=[d.y for d in items_to_plot],
            per_prediction_info=[d.metadata for d in items_to_plot],
            max_evals=self.max_evals,
            network_model=network_model,
            enable_gridstats=True,
            device=self.device,
            gridstats_hypercube_res=self.gridstats_hypercube_res,
            gridstats_hypercube_min=self.gridstats_hypercube_min,
            gridstats_hypercube_max=self.gridstats_hypercube_max,
            gridstats_k=self.gridstats_k,
            gridstats_radius=self.gridstats_radius,
            gridstats_min_points=self.gridstats_min_points,
            z_value=self.z_value,
            z_normal_mean=self.z_normal_mean,
            z_normal_std=self.z_normal_std,
            z_normal_clip=self.z_normal_clip,
            disable_variational=self.disable_variational,
            sampling=self.sampling,
        )

        prediction_data = predictor.get_data()
        network_stats = predictor.get_network_stats()

        training_content = self._model.metadata.get('training_set', {}).get('content', [])
        training_names = {
            item.get('network_name', '') for item in training_content if isinstance(item, dict)
        }

        self._items = []
        all_nrmses = []
        all_snrs = []
        for i, (gt, pred, stats) in enumerate(
            zip(items_to_plot, prediction_data, network_stats, strict=True)
        ):
            net_name = gt.metadata.get('network_name', f'Item_{i}')
            rmse = stats.get('grid_rmse') or stats.get('rmse')
            nrmse = stats.get('grid_nrmse')
            snr = stats.get('grid_snr')
            if nrmse is not None and np.isfinite(nrmse):
                all_nrmses.append(nrmse)
            if snr is not None and np.isfinite(snr):
                all_snrs.append(snr)

            # Ensure gt_data uses the same x points as pred_data (which may be truncated by max_evals)
            pred_n = len(pred.x)
            if len(gt.x) > pred_n:
                gt_truncated = PlotData(
                    xval=gt.x[:pred_n],
                    yval=gt.y[:pred_n] if gt.y is not None else None,
                    input_names=gt.input_names,
                    output_name=gt.output_name,
                    column_proteins=gt.column_proteins,
                    metadata=gt.metadata,
                )
            else:
                gt_truncated = gt

            self._items.append(
                BenchmarkItem(
                    idx=i,
                    gt_data=gt_truncated,
                    pred_data=pred,
                    network=gt.metadata.get('built_network'),
                    network_name=net_name,
                    rmse=rmse,
                    nrmse=nrmse,
                    snr=snr,
                    in_training=net_name in training_names,
                )
            )

        if all_nrmses:
            arr = np.array(all_nrmses)
            positive = arr[arr > 0]
            alpha = 5.0
            max_val = np.max(arr)
            self._aggregate_stats = {
                'mean_nrmse': float(np.mean(arr)),
                'geomean_nrmse': float(gmean(positive)) if len(positive) > 0 else None,
                'softmax_nrmse': float(
                    max_val + (1 / alpha) * np.log(np.sum(np.exp(alpha * (arr - max_val))))
                ),
            }
        if all_snrs:
            self._aggregate_stats['mean_snr'] = float(np.mean(all_snrs))

    @property
    def loaded_model(self):
        return self._model

    @property
    def model_signature(self) -> str:
        return self._model.signature

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def items(self) -> list[BenchmarkItem]:
        return self._items

    @property
    def n_items(self) -> int:
        return len(self._items)

    @property
    def all_rmses(self) -> list[float]:
        return [item.rmse for item in self._items if item.rmse is not None]

    @property
    def all_nrmses(self) -> list[float]:
        return [
            item.nrmse for item in self._items if item.nrmse is not None and np.isfinite(item.nrmse)
        ]

    @property
    def mean_rmse(self) -> Optional[float]:
        rmses = self.all_rmses
        return sum(rmses) / len(rmses) if rmses else None

    @property
    def mean_nrmse(self) -> Optional[float]:
        return self._aggregate_stats.get('mean_nrmse')

    @property
    def geomean_nrmse(self) -> Optional[float]:
        return self._aggregate_stats.get('geomean_nrmse')

    @property
    def softmax_nrmse(self) -> Optional[float]:
        return self._aggregate_stats.get('softmax_nrmse')

    @property
    def all_snrs(self) -> list[float]:
        return [item.snr for item in self._items if item.snr is not None and np.isfinite(item.snr)]

    @property
    def mean_snr(self) -> Optional[float]:
        return self._aggregate_stats.get('mean_snr')

    @property
    def network_names(self) -> list[str]:
        return [item.network_name[:20] for item in self._items]

    @property
    def is_in_training(self) -> list[bool]:
        return [item.in_training for item in self._items]

    @property
    def training_set_name(self) -> str:
        return self._model.metadata.get('training_set', {}).get('name', 'Unknown')

    _MAX_AGGREGATE_POINTS: int = 50000

    @property
    def all_measured(self) -> np.ndarray:
        """Flattened 1D array of all ground-truth y-values, subsampled if needed."""
        m, _ = self._get_aggregate_arrays()
        return m

    @property
    def all_predicted(self) -> np.ndarray:
        """Flattened 1D array of all predicted y-values, subsampled if needed."""
        _, p = self._get_aggregate_arrays()
        return p

    def _get_aggregate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        measured_parts: list[np.ndarray] = []
        predicted_parts: list[np.ndarray] = []
        for item in self._items:
            gt_y = item.gt_data.yval
            pr_y = item.pred_data.yval
            if gt_y is None or pr_y is None:
                continue
            gt_flat = np.asarray(gt_y).ravel()
            pr_flat = np.asarray(pr_y).ravel()
            n = min(len(gt_flat), len(pr_flat))
            measured_parts.append(gt_flat[:n])
            predicted_parts.append(pr_flat[:n])
        if not measured_parts:
            return np.array([]), np.array([])
        measured = np.concatenate(measured_parts)
        predicted = np.concatenate(predicted_parts)
        if len(measured) > self._MAX_AGGREGATE_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(measured), self._MAX_AGGREGATE_POINTS, replace=False)
            measured, predicted = measured[idx], predicted[idx]
        return measured, predicted


def render_summary_header(ax: matplotlib.axes.Axes, bench: BenchmarkData, **_kwargs):
    """Render summary header with model info, metrics, and per-item barplot."""
    ax.axis('off')

    info_lines = [
        f"Model: {bench.model_signature}",
        f"Trained on: {bench.training_set_name}",
        f"Benchmark: {bench.dataset_name}",
        f"N items: {bench.n_items}",
    ]
    ax.text(
        0.02,
        0.95,
        '\n'.join(info_lines),
        transform=ax.transAxes,
        fontsize=10,
        va='top',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
    )

    # Main metrics display - geomean nRMSE
    y_pos = 0.75
    if bench.geomean_nrmse is not None:
        color = GOOD_COLOR if bench.geomean_nrmse < NRMSE_THRESHOLD else BAD_COLOR
        ax.text(
            0.32,
            y_pos,
            f"{bench.geomean_nrmse:.3f}",
            transform=ax.transAxes,
            fontsize=28,
            va='center',
            ha='center',
            fontweight='bold',
            color=color,
        )
        ax.text(
            0.32,
            y_pos - 0.12,
            "Geomean nRMSE",
            transform=ax.transAxes,
            fontsize=10,
            va='center',
            ha='center',
            color='gray',
        )

    # Secondary stats
    stats_parts = []
    if bench.geomean_nrmse is not None:
        stats_parts.append(f"nRMSE={bench.geomean_nrmse:.3f}")
    if bench.mean_snr is not None:
        stats_parts.append(f"SNR={bench.mean_snr:.1f}dB")
    if stats_parts:
        ax.text(
            0.32,
            0.38,
            "  ".join(stats_parts),
            transform=ax.transAxes,
            fontsize=9,
            va='center',
            ha='center',
            family='monospace',
            color='#555',
        )

    if bench.mean_rmse is not None:
        ax.text(
            0.32,
            0.25,
            f"RMSE={bench.mean_rmse:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            va='center',
            ha='center',
            family='monospace',
            color='#555',
        )

    ax.text(
        0.32,
        0.12,
        "NRE: 1.0=perfect (noise floor), lower=better",
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        color='gray',
    )

    bar_data = bench.all_nrmses
    bar_label = 'nRMSE'
    bar_threshold = NRMSE_THRESHOLD
    if bar_data and bench.network_names:
        inset = ax.inset_axes([0.5, 0.1, 0.48, 0.8])
        colors = [IN_TRAINING_COLOR if it else NOT_IN_TRAINING_COLOR for it in bench.is_in_training]
        y_pos = np.arange(len(bar_data))
        inset.barh(y_pos, bar_data, color=colors, edgecolor='#666', linewidth=0.5)
        inset.set_yticks(y_pos)
        inset.set_yticklabels(bench.network_names[: len(bar_data)], fontsize=7)
        inset.set_xlabel(bar_label, fontsize=9)
        inset.axvline(x=bar_threshold, color=GOOD_COLOR, linestyle='--', alpha=0.7, linewidth=1)
        inset.set_xlim(0, max(bar_data) * 1.1 if bar_data else 1)
        inset.invert_yaxis()
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)
        inset.legend(
            handles=[
                Patch(facecolor=IN_TRAINING_COLOR, edgecolor='#666', label='In training'),
                Patch(facecolor=NOT_IN_TRAINING_COLOR, edgecolor='#666', label='Not in training'),
            ],
            loc='lower right',
            fontsize=7,
        )


def render_metrics_panel(
    ax: matplotlib.axes.Axes, item: BenchmarkItem, bench: 'BenchmarkData' = None, **_kwargs
):
    """Render metrics panel for a single benchmark item."""
    ax.axis('off')

    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor=NOT_IN_TRAINING_COLOR,
            edgecolor='#ccc',
            linewidth=1,
            clip_on=False,
        )
    )

    avg_nrmse = bench.mean_nrmse if bench else None

    if item.nrmse is not None:
        ncolor = (
            GOOD_COLOR
            if (avg_nrmse and item.nrmse < avg_nrmse)
            else (BAD_COLOR if avg_nrmse else '#333')
        )
        ax.text(
            0.5,
            0.78,
            f"{item.nrmse:.3f}",
            transform=ax.transAxes,
            fontsize=18,
            va='center',
            ha='center',
            fontweight='bold',
            color=ncolor,
        )
        ax.text(
            0.5,
            0.65,
            "nRMSE",
            transform=ax.transAxes,
            fontsize=7,
            va='center',
            ha='center',
            color='gray',
        )

    # Secondary metrics
    y_offset = 0.50
    if item.nrmse is not None:
        ax.text(
            0.5,
            y_offset,
            f"nRMSE={item.nrmse:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            va='center',
            ha='center',
            family='monospace',
            color='#555',
        )
        y_offset -= 0.12

    if item.snr is not None:
        ax.text(
            0.5,
            y_offset,
            f"SNR={item.snr:.1f}dB",
            transform=ax.transAxes,
            fontsize=8,
            va='center',
            ha='center',
            family='monospace',
            color='#555',
        )
        y_offset -= 0.12

    # Dimensionality
    ndim = item.gt_data.x.shape[1] if item.gt_data and item.gt_data.x is not None else None
    if ndim is not None:
        ax.text(
            0.5,
            y_offset,
            f"{ndim}D",
            transform=ax.transAxes,
            fontsize=8,
            va='center',
            ha='center',
            family='monospace',
            color='#555',
        )

    status = "In training" if item.in_training else "Not in training"
    ax.text(
        0.5,
        0.08,
        status,
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        style='italic',
    )
