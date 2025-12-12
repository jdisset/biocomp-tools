"""Design result data holder and metrics figuremaker."""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import matplotlib.axes
from matplotlib.patches import FancyBboxPatch

from biocomp.plotutils import PlotData

GOOD_COLOR = "#28a745"
BAD_COLOR = "#dc3545"
NEUTRAL_COLOR = "#333"


@dataclass
class DesignResult:
    """Minimal data holder for a single design result (like BenchmarkItem)."""

    network: Any
    target: Any
    target_name: str
    rank: int
    replicate: int
    scaffold_network_name: str
    loss: float
    recipe_hash: str
    run_name: str = ""
    model: Any = None

    _gt_data: Optional[PlotData] = field(default=None, repr=False)
    _pred_data: Optional[PlotData] = field(default=None, repr=False)
    _lattice_data: Optional[PlotData] = field(default=None, repr=False)

    @property
    def gt_data(self) -> PlotData:
        """Ground truth / target data for plotting."""
        if self._gt_data is None:
            from biocomp.design import DataTarget

            if isinstance(self.target, DataTarget):
                X, Y = self.target.X, np.atleast_1d(self.target.Y.squeeze())
                if len(X) > 20000:  # subsample
                    idx = np.random.default_rng(42).choice(len(X), 20000, replace=False)
                    X, Y = X[idx], Y[idx]
            else:
                X, Y = self.target.sample_uniform(10000, seed=42)
                Y = Y.ravel()
            self._gt_data = PlotData(
                xval=X, yval=Y,
                input_names=[f'X{i+1}' for i in range(X.shape[1])],
                output_name='Y',
            )
        return self._gt_data

    @property
    def pred_data(self) -> PlotData:
        """Prediction data for plotting (lazily computed)."""
        if self._pred_data is None:
            from biocomptools.modelmodel import NetworkModel
            from biocomptools.toollib.networkprediction import NetworkPrediction

            X = self.gt_data.x
            predictor = NetworkPrediction(
                predict_at=[X],
                max_evals=50000,
                network_model=NetworkModel(model=self.model, network=[self.network]),
                device='gpu',
            )
            pred = predictor.get_data()
            self._pred_data = PlotData(
                xval=X, yval=pred[0].y if pred else self.gt_data.y,
                input_names=[f'X{i+1}' for i in range(X.shape[1])],
                output_name='Y',
            )
        return self._pred_data

    @property
    def lattice_data(self) -> PlotData:
        """Target sampled on lattice for design view."""
        if self._lattice_data is None:
            X, Y = self.target.get_lattice((48, 48), seed=0)
            self._lattice_data = PlotData(
                xval=X, yval=Y.ravel(),
                input_names=[f'X{i+1}' for i in range(X.shape[1])],
                output_name='Y',
            )
        return self._lattice_data


def render_design_metrics(ax: matplotlib.axes.Axes, result: DesignResult, **_kwargs):
    """Render metrics panel for design result."""
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes, boxstyle="round,pad=0.02",
        facecolor='#EEEEEE', edgecolor='#ccc', linewidth=1, clip_on=False,
    ))

    loss_color = GOOD_COLOR if result.loss < 0.5 else (BAD_COLOR if result.loss > 1.5 else NEUTRAL_COLOR)
    ax.text(0.5, 0.80, f"{result.loss:.4f}", transform=ax.transAxes,
            fontsize=24, va='center', ha='center', fontweight='bold', color=loss_color)
    ax.text(0.5, 0.65, "Design Loss", transform=ax.transAxes,
            fontsize=10, va='center', ha='center', color='gray')

    ax.text(0.5, 0.45, f"Rank: {result.rank}  |  Replicate: {result.replicate}",
            transform=ax.transAxes, fontsize=10, va='center', ha='center', family='monospace')

    scaffold = result.scaffold_network_name[:25] + '...' if len(result.scaffold_network_name) > 28 else result.scaffold_network_name
    ax.text(0.5, 0.30, f"Scaffold: {scaffold}", transform=ax.transAxes,
            fontsize=8, va='center', ha='center', family='monospace', color='#666')

    ax.text(0.5, 0.15, f"Hash: {result.recipe_hash}", transform=ax.transAxes,
            fontsize=8, va='center', ha='center', family='monospace', color='#888')


def render_empty_panel(ax: matplotlib.axes.Axes, text: str = "", **_kwargs):
    """Render empty placeholder panel."""
    ax.axis('off')
    if text:
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                fontsize=10, va='center', ha='center', color='#aaa')
