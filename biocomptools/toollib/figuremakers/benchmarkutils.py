"""Benchmark summary utilities."""

from dataclasses import dataclass
from typing import Optional, Any
import matplotlib.axes
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from biocomp.plotutils import PlotData

IN_TRAINING_COLOR = "#fff3cd"
NOT_IN_TRAINING_COLOR = "#d4edda"
GOOD_COLOR = "#28a745"
BAD_COLOR = "#dc3545"
RMSE_THRESHOLD = 0.1


@dataclass
class BenchmarkItem:
    """Single benchmark item with ground truth, prediction, and stats."""
    idx: int
    gt_data: PlotData
    pred_data: PlotData
    network: object
    network_name: str
    rmse: Optional[float]
    in_training: bool

    @property
    def file_prefix(self) -> str:
        return f"{str(self.idx + 1).zfill(2)}_"


class BenchmarkData(BaseModel):
    """Encapsulates benchmark computation for a model on a dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Optional[str] = None
    model_name: Optional[str] = None
    dataset_file: Optional[str] = None
    max_items: Optional[int] = None

    _model: Any = PrivateAttr(default=None)
    _items: list[BenchmarkItem] = PrivateAttr(default_factory=list)
    _dataset_name: str = PrivateAttr(default="")

    @model_validator(mode='after')
    def _initialize_after(self):
        self._do_initialize()
        return self

    def _do_initialize(self):
        from biocomptools.modelmodel import BiocompModel, NetworkModel
        from biocomptools.toollib.modelselector import ModelSelector
        from biocomptools.toollib.networkprediction import NetworkPrediction
        from biocomptools.toollib.datasources import DBSource
        from biocomptools.toollib.networkselector import (
            NetworkSet, NetworkSetUnion, NetworkSetDifference, NetworkSetIntersection,
            CleanupFilter, NetworkFilter, Regex, iRegex,
        )
        from dracon import load

        # Load model
        if self.model_name:
            self._model = ModelSelector(name=self.model_name).get_model().load()
        else:
            self._model = BiocompModel.load(self.model_path)

        # Derive dataset name
        self._dataset_name = self.dataset_file.split('/')[-1].rsplit('.', 1)[0] if self.dataset_file else "unknown"

        # Load ground truth data
        ctx = {
            'NetworkSet': NetworkSet, 'NetworkSetUnion': NetworkSetUnion,
            'NetworkSetDifference': NetworkSetDifference, 'NetworkSetIntersection': NetworkSetIntersection,
            'CleanupFilter': CleanupFilter, 'NetworkFilter': NetworkFilter,
            'Regex': Regex, 'iRegex': iRegex, 'DBSource': DBSource,
        }
        dataset = load(self.dataset_file, context=ctx)
        if hasattr(dataset, 'get_data'):
            ground_truth = dataset.get_data()
        else:
            db_source = DBSource(content=dataset)
            ground_truth = db_source.get_data()

        # Limit items
        max_n = int(self.max_items) if self.max_items else None
        items_to_plot = ground_truth[:max_n] if max_n else ground_truth

        # Build network model and predictor
        networks = [d.metadata['built_network'] for d in items_to_plot]
        network_model = NetworkModel(model=self._model, network=networks)

        predictor = NetworkPrediction(
            predict_at=[d.x for d in items_to_plot],
            ground_truth=[d.y for d in items_to_plot],
            per_prediction_info=[d.metadata for d in items_to_plot],
            max_evals=250000,
            network_model=network_model,
            enable_gridstats=True,
        )

        prediction_data = predictor.get_data()
        network_stats = predictor.get_network_stats()

        # Training set membership
        training_content = self._model.metadata.get('training_set', {}).get('content', [])
        training_names = {
            item.get('network_name', '') for item in training_content if isinstance(item, dict)
        }

        # Build items
        self._items = []
        for i, (gt, pred, stats) in enumerate(zip(items_to_plot, prediction_data, network_stats)):
            net_name = gt.metadata.get('network_name', f'Item_{i}')
            rmse = stats.get('grid_rmse') or stats.get('rmse')
            self._items.append(BenchmarkItem(
                idx=i,
                gt_data=gt,
                pred_data=pred,
                network=gt.metadata.get('built_network'),
                network_name=net_name,
                rmse=rmse,
                in_training=net_name in training_names,
            ))

    @property
    def model(self):
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
    def mean_rmse(self) -> Optional[float]:
        rmses = self.all_rmses
        return sum(rmses) / len(rmses) if rmses else None

    @property
    def network_names(self) -> list[str]:
        return [item.network_name[:20] for item in self._items]

    @property
    def is_in_training(self) -> list[bool]:
        return [item.in_training for item in self._items]

    @property
    def training_set_name(self) -> str:
        return self._model.metadata.get('training_set', {}).get('name', 'Unknown')


def render_summary_header(
    ax: matplotlib.axes.Axes,
    bench: BenchmarkData,
    **_kwargs,
):
    """Render summary header with model info, mean RMSE, and per-item barplot."""
    ax.axis('off')

    info_text = (
        f"Model: {bench.model_signature}\n"
        f"Trained on: {bench.training_set_name}\n"
        f"Benchmark: {bench.dataset_name}\n"
        f"N items: {bench.n_items}"
    )
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    if bench.mean_rmse is not None:
        color = GOOD_COLOR if bench.mean_rmse < RMSE_THRESHOLD else BAD_COLOR
        ax.text(0.35, 0.5, f"{bench.mean_rmse:.4f}", transform=ax.transAxes,
                fontsize=36, va='center', ha='center', fontweight='bold', color=color)
        ax.text(0.35, 0.15, "Mean RMSE", transform=ax.transAxes,
                fontsize=12, va='center', ha='center', color='gray')

    if bench.all_rmses and bench.network_names:
        inset = ax.inset_axes([0.5, 0.1, 0.48, 0.8])
        colors = [IN_TRAINING_COLOR if it else NOT_IN_TRAINING_COLOR for it in bench.is_in_training]
        y_pos = np.arange(len(bench.all_rmses))

        inset.barh(y_pos, bench.all_rmses, color=colors, edgecolor='#666', linewidth=0.5)
        inset.set_yticks(y_pos)
        inset.set_yticklabels(bench.network_names, fontsize=7)
        inset.set_xlabel('RMSE', fontsize=9)
        inset.axvline(x=RMSE_THRESHOLD, color=GOOD_COLOR, linestyle='--', alpha=0.7, linewidth=1)
        inset.set_xlim(0, max(bench.all_rmses) * 1.1 if bench.all_rmses else 1)
        inset.invert_yaxis()
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)

        legend_elements = [
            Patch(facecolor=IN_TRAINING_COLOR, edgecolor='#666', label='In training'),
            Patch(facecolor=NOT_IN_TRAINING_COLOR, edgecolor='#666', label='Not in training'),
        ]
        inset.legend(handles=legend_elements, loc='lower right', fontsize=7)


def render_metrics_panel(
    ax: matplotlib.axes.Axes,
    item: BenchmarkItem,
    **_kwargs,
):
    """Render metrics panel for a single benchmark item."""
    ax.axis('off')

    bg_color = IN_TRAINING_COLOR if item.in_training else NOT_IN_TRAINING_COLOR
    rect = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                          boxstyle="round,pad=0.02", facecolor=bg_color,
                          edgecolor='#ccc', linewidth=1, clip_on=False)
    ax.add_patch(rect)

    if item.rmse is not None:
        color = GOOD_COLOR if item.rmse < RMSE_THRESHOLD else BAD_COLOR
        ax.text(0.5, 0.75, f"{item.rmse:.3f}", transform=ax.transAxes,
                fontsize=20, va='center', ha='center', fontweight='bold', color=color)

    display_name = item.network_name[:25] if len(item.network_name) > 25 else item.network_name
    ax.text(0.5, 0.45, display_name, transform=ax.transAxes,
            fontsize=8, va='center', ha='center', family='monospace')

    status = "In training" if item.in_training else "Not in training"
    ax.text(0.5, 0.2, status, transform=ax.transAxes,
            fontsize=8, va='center', ha='center', style='italic')
