"""Target vs Prediction comparison utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.axes
from pydantic import BaseModel, model_validator

from biocomp.plotutils import PlotData
from biocomp.metric_utils import RegressionStats
from biocomp.designloss import GridLossResult, compute_grid_losses

logger = logging.getLogger(__name__)


class load_and_compute(BaseModel):
    """Pydantic wrapper for loading and computing target vs prediction comparison.

    Used by Dracon YAML: !biocomptools.toollib.figuremakers.targetpredictutils.load_and_compute
    """

    target_file: str
    recipe_file: str
    model_path: str
    lattice_resolution: int = 64
    pred_resolution: int = 224

    _result: TargetPredictionData | None = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def _compute(self):
        import os
        from pathlib import Path
        import dracon as dr
        from biocomp.recipe import Recipe
        from biocomp.network import recipe_to_networks
        from biocomp.design_targets import SVGTarget, DataTarget
        from biocomptools.modelmodel import BiocompModel

        model = BiocompModel.load(self.model_path)

        biocomp_root = Path(os.environ.get('BIOCOMP_ROOT', ''))
        target_context = {
            'SVGTarget': SVGTarget,
            'DataTarget': DataTarget,
            'BIOCOMP_ROOT': biocomp_root,
        }
        target_data = dr.load(self.target_file, context=target_context)
        if isinstance(target_data, list):
            target = target_data[0]
        elif hasattr(target_data, 'get'):
            target = target_data.get('target') or target_data.get('targets', [target_data])[0]
        else:
            target = target_data

        with open(self.recipe_file, 'r') as f:
            content = f.read()
        if '\n!biocomp.recipe.Recipe' in content:
            idx = content.index('\n!biocomp.recipe.Recipe')
            recipe_part = content[idx + 1:]
        elif content.startswith('!biocomp.recipe.Recipe'):
            recipe_part = content
        else:
            recipe_part = content
        recipe_context = {'Recipe': Recipe, 'biocomp.recipe.Recipe': Recipe}
        recipe = dr.loads(recipe_part, context=recipe_context)
        if not hasattr(recipe, 'content'):
            if hasattr(recipe, 'get'):
                recipe = recipe.get('recipe', recipe)

        networks = recipe_to_networks(recipe, invert=True, inversion_mode='main')
        network = networks[0]

        self._result = compute_target_prediction(
            target=target,
            network=network,
            model=model,
            lattice_resolution=(self.lattice_resolution, self.lattice_resolution),
            pred_resolution=self.pred_resolution,
        )
        return self

    @property
    def target_name(self) -> str:
        return self._result.target_name

    @property
    def recipe_name(self) -> str:
        return self._result.recipe_name

    @property
    def target_data(self) -> PlotData:
        return self._result.target_data

    @property
    def pred_data(self) -> PlotData:
        return self._result.pred_data

    @property
    def stats(self) -> RegressionStats:
        return self._result.stats

    @property
    def lattice_extent(self) -> tuple[float, float, float, float]:
        return self._result.lattice_extent

    @property
    def grid_losses(self) -> GridLossResult:
        return self._result.grid_losses


@dataclass
class TargetPredictionData:
    """Precomputed data for target vs prediction comparison."""

    target_name: str
    recipe_name: str
    target_data: PlotData
    pred_data: PlotData
    stats: RegressionStats
    grid_losses: GridLossResult
    lattice_extent: tuple[float, float, float, float]


def compute_target_prediction(
    target: Any,
    network: Any,
    model: Any,
    lattice_resolution: tuple[int, int] = (64, 64),
    pred_resolution: int = 224,
) -> TargetPredictionData:
    """Compute target and prediction data for comparison."""
    import jax.numpy as jnp
    from biocomptools.modelmodel import NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    target_name = getattr(target, 'name', 'Target') or 'Target'
    recipe_name = getattr(network, 'name', 'Network') or 'Network'

    x_ext = getattr(target, 'latent_x', (0.0, 1.0))
    y_ext = getattr(target, 'latent_y', (0.0, 1.0))
    extent = (x_ext[0], x_ext[1], y_ext[0], y_ext[1])

    X_lattice, Y_lattice = target.get_lattice(lattice_resolution, seed=0)
    Y_target_grid = np.asarray(Y_lattice)
    Y_flat = Y_target_grid.ravel()
    target_plot = PlotData(
        xval=model.rescaler.inv(X_lattice),
        yval=model.rescaler.inv(Y_flat.reshape(-1, 1)).ravel(),
        input_names=['X1', 'X2'],
        output_name='Y',
    )

    xv = np.linspace(x_ext[0], x_ext[1], pred_resolution)
    yv = np.linspace(y_ext[0], y_ext[1], pred_resolution)
    xx, yy = np.meshgrid(xv, yv)
    X_pred = np.column_stack([xx.ravel(), yy.ravel()])

    nm = NetworkModel(model=model, network=[network])
    predictor = NetworkPrediction(
        predict_at=[X_pred],
        network_model=nm,
        already_latent=True,
        device='gpu',
        verbose=False,
        skip_input_reorder=True,
    )
    pred_list = predictor.get_data(rescale_latent=True)
    pred_plot = pred_list[0]

    Y_target_interp = _interpolate_to_pred_grid(
        X_lattice, Y_flat, X_pred, lattice_resolution
    )
    Y_pred_latent = model.rescaler.fwd(pred_plot.yval.reshape(-1, 1)).ravel()
    stats = RegressionStats.compute(Y_target_interp, Y_pred_latent, validate=False)

    Y_pred_grid = Y_pred_latent.reshape(pred_resolution, pred_resolution)
    Y_target_grid_resized = _interpolate_to_pred_grid(
        X_lattice, Y_flat, X_pred, lattice_resolution
    ).reshape(pred_resolution, pred_resolution)
    grid_losses = compute_grid_losses(
        jnp.array(Y_pred_grid),
        jnp.array(Y_target_grid_resized),
        w_sinkhorn=1.0,
        w_lncc=0.5,
        w_mse=1.0,
        w_simse=1.0,
        w_spectral=1.0,
    )

    logger.info(
        f"Target vs Prediction metrics for {target_name} / {recipe_name}:\n"
        f"  Total: {grid_losses.total:.4f}  |  Sinkhorn: {grid_losses.sinkhorn:.4f}  |  LNCC: {grid_losses.lncc:.4f}\n"
        f"  MSE: {grid_losses.mse:.4f}  |  SIMSE: {grid_losses.simse:.4f}  |  Spectral: {grid_losses.spectral:.4f}"
    )

    return TargetPredictionData(
        target_name=target_name,
        recipe_name=recipe_name,
        target_data=target_plot,
        pred_data=pred_plot,
        stats=stats,
        grid_losses=grid_losses,
        lattice_extent=extent,
    )


def _interpolate_to_pred_grid(
    X_lattice: np.ndarray,
    Y_flat: np.ndarray,
    X_pred: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Interpolate lattice Y values to prediction grid coordinates."""
    from scipy.interpolate import RegularGridInterpolator

    x_unique = np.unique(X_lattice[:, 0])
    y_unique = np.unique(X_lattice[:, 1])
    Y_grid = Y_flat.reshape(resolution[1], resolution[0])
    interp = RegularGridInterpolator(
        (y_unique, x_unique), Y_grid, method='linear', bounds_error=False
    )
    return interp((X_pred[:, 1], X_pred[:, 0]))


def render_metrics_subtitle(
    ax: matplotlib.axes.Axes,
    data: TargetPredictionData | load_and_compute,
    **_kwargs,
):
    """Render metrics as a subtitle below the figure."""
    ax.axis('off')
    g = data.grid_losses
    line1 = f"Total: {g.total:.4f}  |  Sinkhorn: {g.sinkhorn:.4f}  |  LNCC: {g.lncc:.4f}"
    line2 = f"MSE: {g.mse:.4f}  |  SIMSE: {g.simse:.4f}  |  Spectral: {g.spectral:.4f}"
    metrics = f"{line1}\n{line2}"
    ax.text(
        0.5, 0.5, metrics,
        transform=ax.transAxes,
        fontsize=9,
        ha='center', va='center',
        family='monospace',
        linespacing=1.5,
    )
