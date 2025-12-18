from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import matplotlib.axes
from matplotlib.patches import FancyBboxPatch

from biocomp.plotutils import PlotData
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

GOOD_COLOR = "#28a745"
BAD_COLOR = "#dc3545"
NEUTRAL_COLOR = "#333"
BASELINE_COLOR = "#6c757d"


@dataclass
class DesignResult:
    """Data holder for design result. All data properties return RAW space for plotting."""

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
    _baseline_nre_value: Optional[float] = field(default=None, repr=False)
    _design_nre_value: Optional[float] = field(default=None, repr=False)

    def _to_raw_space(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            return X, Y
        if np.abs(X).max() > 100 or np.abs(Y).max() > 100:
            logger.warning(
                f"DesignResult._to_raw_space: Input data appears to be in RAW space already "
                f"(X max={np.abs(X).max():.0f}, Y max={np.abs(Y).max():.0f}). "
                f"Expected LATENT space (values roughly -2 to 2). "
                f"Applying rescaler.inv() to raw data will produce invalid results. "
                f"This usually indicates DataTarget.X/Y wasn't properly rescaled."
            )
        return self.model.rescaler.inv(X), self.model.rescaler.inv(Y.reshape(-1, 1)).ravel()

    @property
    def gt_data(self) -> PlotData:
        if self._gt_data is None:
            from biocomp.design import DataTarget

            if isinstance(self.target, DataTarget):
                X, Y = self.target.X, np.atleast_1d(self.target.Y.squeeze())
                if len(X) > 20000:
                    idx = np.random.default_rng(42).choice(len(X), 20000, replace=False)
                    X, Y = X[idx], Y[idx]
            else:
                X, Y = self.target.sample_uniform(10000, seed=42)
                Y = Y.ravel()
            X, Y = self._to_raw_space(X, Y)
            self._gt_data = PlotData(
                xval=X,
                yval=Y,
                input_names=[f'X{i + 1}' for i in range(X.shape[1])],
                output_name='Y',
            )
        return self._gt_data

    @property
    def pred_data(self) -> PlotData:
        if self._pred_data is None:
            from biocomptools.modelmodel import NetworkModel
            from biocomptools.toollib.networkprediction import NetworkPrediction
            from biocomp.design import DataTarget

            if isinstance(self.target, DataTarget):
                X_latent = self.target.X
                if len(X_latent) > 20000:
                    idx = np.random.default_rng(42).choice(len(X_latent), 20000, replace=False)
                    X_latent = X_latent[idx]
            else:
                X_latent, _ = self.target.sample_uniform(10000, seed=42)

            predictor = NetworkPrediction(
                predict_at=[X_latent],
                max_evals=50000,
                network_model=NetworkModel(model=self.model, network=[self.network]),
                already_latent=True,
                device='gpu',
            )
            pred = predictor.get_data(rescale_latent=True)
            X_pred = pred[0].x if pred else X_latent
            Y_pred = pred[0].y if pred else np.zeros(len(X_pred))
            if self.model is not None and not pred:
                X_pred = self.model.rescaler.inv(X_pred)
            self._pred_data = PlotData(
                xval=X_pred,
                yval=Y_pred,
                input_names=[f'X{i + 1}' for i in range(X_pred.shape[1])],
                output_name='Y',
            )
        return self._pred_data

    @property
    def lattice_data(self) -> PlotData:
        if self._lattice_data is None:
            X, Y = self.target.get_lattice((48, 48), seed=0)
            X, Y_flat = self._to_raw_space(X, Y.ravel())
            self._lattice_data = PlotData(
                xval=X,
                yval=Y_flat,
                input_names=[f'X{i + 1}' for i in range(X.shape[1])],
                output_name='Y',
            )
        return self._lattice_data

    @property
    def has_original_network(self) -> bool:
        from biocomp.design import DataTarget

        return isinstance(self.target, DataTarget) and self.target.original_network is not None

    def _compute_nre_for_network(self, network: Any, max_evals: int = 50000) -> Optional[float]:
        from biocomp.design import DataTarget

        if self.model is None or not isinstance(self.target, DataTarget):
            return None
        try:
            from biocomptools.modelmodel import NetworkModel
            from biocomptools.toollib.networkprediction import NetworkPrediction

            X, Y = self.target.X, self.target.Y
            if len(X) > max_evals:
                idx = np.random.default_rng(42).choice(len(X), max_evals, replace=False)
                X, Y = X[idx], Y[idx]

            predictor = NetworkPrediction(
                predict_at=[X],
                ground_truth=[Y.reshape(-1, 1) if Y.ndim == 1 else Y],
                max_evals=max_evals,
                network_model=NetworkModel(model=self.model, network=[network]),
                already_latent=True,
                enable_gridstats=True,
                device='gpu',
                verbose=False,
            )
            stats = predictor.get_network_stats()
            return stats[0].get('noise_relative_error') if stats else None
        except Exception as e:
            logger.warning(f"Failed to compute NRE: {e}")
            return None

    @property
    def baseline_nre(self) -> Optional[float]:
        if self._baseline_nre_value is not None:
            return self._baseline_nre_value
        if not self.has_original_network:
            return None
        if not hasattr(self, '_baseline_nre_computed'):
            self._baseline_nre_computed = self._compute_nre_for_network(
                self.target.original_network
            )
        return self._baseline_nre_computed

    @property
    def design_nre(self) -> Optional[float]:
        if self._design_nre_value is not None:
            return self._design_nre_value
        if not hasattr(self, '_design_nre_computed'):
            from biocomp.design import DataTarget

            self._design_nre_computed = (
                self._compute_nre_for_network(self.network)
                if isinstance(self.target, DataTarget)
                else None
            )
        return self._design_nre_computed


def render_design_metrics(ax: matplotlib.axes.Axes, result: DesignResult, **_kwargs):
    ax.axis('off')
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.02",
            facecolor='#EEEEEE',
            edgecolor='#ccc',
            linewidth=1,
            clip_on=False,
        )
    )

    loss_color = (
        GOOD_COLOR if result.loss < 0.5 else (BAD_COLOR if result.loss > 1.5 else NEUTRAL_COLOR)
    )
    ax.text(
        0.5,
        0.88,
        f"{result.loss:.4f}",
        transform=ax.transAxes,
        fontsize=22,
        va='center',
        ha='center',
        fontweight='bold',
        color=loss_color,
    )
    ax.text(
        0.5,
        0.76,
        "Design Loss",
        transform=ax.transAxes,
        fontsize=9,
        va='center',
        ha='center',
        color='gray',
    )

    if result.has_original_network:
        baseline_nre, design_nre = result.baseline_nre, result.design_nre
        nre_y = 0.60
        if design_nre is not None:
            if baseline_nre is not None and design_nre <= baseline_nre * 1.5:
                nre_color = GOOD_COLOR
            elif design_nre < 5.0:
                nre_color = NEUTRAL_COLOR
            else:
                nre_color = BAD_COLOR
            ax.text(
                0.5,
                nre_y,
                f"NRE: {design_nre:.2f}",
                transform=ax.transAxes,
                fontsize=14,
                va='center',
                ha='center',
                fontweight='bold',
                color=nre_color,
            )
        else:
            ax.text(
                0.5,
                nre_y,
                "NRE: N/A",
                transform=ax.transAxes,
                fontsize=14,
                va='center',
                ha='center',
                color='#aaa',
            )
        if baseline_nre is not None:
            ax.text(
                0.5,
                nre_y - 0.10,
                f"(baseline: {baseline_nre:.2f})",
                transform=ax.transAxes,
                fontsize=9,
                va='center',
                ha='center',
                color=BASELINE_COLOR,
            )
        info_y = 0.38
    else:
        info_y = 0.55

    ax.text(
        0.5,
        info_y,
        f"Rank: {result.rank}  |  Replicate: {result.replicate}",
        transform=ax.transAxes,
        fontsize=10,
        va='center',
        ha='center',
        family='monospace',
    )
    scaffold = (
        result.scaffold_network_name[:25] + '...'
        if len(result.scaffold_network_name) > 28
        else result.scaffold_network_name
    )
    ax.text(
        0.5,
        info_y - 0.12,
        f"Scaffold: {scaffold}",
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        family='monospace',
        color='#666',
    )
    ax.text(
        0.5,
        info_y - 0.22,
        f"Hash: {result.recipe_hash}",
        transform=ax.transAxes,
        fontsize=8,
        va='center',
        ha='center',
        family='monospace',
        color='#888',
    )


def render_empty_panel(ax: matplotlib.axes.Axes, text: str = "", **_kwargs):
    ax.axis('off')
    if text:
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=10,
            va='center',
            ha='center',
            color='#aaa',
        )
