"""Design evaluation utilities - batched prediction.

Single source of truth for all design evaluation. All networks are batched into
ONE NetworkModel to avoid repeated JIT compilation (~25s savings per batch).
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Any

from biocomp.plotutils import PlotData
from biocomptools.logging_config import get_logger
from biocomptools.toollib.design_data import prepare_target_data

logger = get_logger(__name__)


def is_valid_network(network: Any) -> bool:
    """Check if network has valid structure for evaluation."""
    if network is None:
        return False
    cg = getattr(network, 'compute_graph', None)
    if cg is None or not cg.nodes:
        return False
    return sum(1 for n in cg.nodes.values() if n.node_type == "output") == 1


@dataclass
class DesignInput:
    """Input for design evaluation - minimal required fields."""

    network: Any
    target: Any
    target_name: str
    rank: int
    replicate: int
    scaffold_network_name: str
    loss: float
    recipe_hash: str
    run_name: str = ""
    design_dir: str = ""


@dataclass
class EvaluatedDesign:
    """Complete evaluated design with all computed data."""

    input: DesignInput
    gt_data: PlotData
    pred_data: PlotData
    lattice_data: PlotData | None
    lattice_grid: np.ndarray | None  # (yres, xres) for pixel-perfect rendering
    lattice_extent: tuple[float, float, float, float] | None  # (xmin, xmax, ymin, ymax)
    lattice_resolution: tuple[int, int] | None  # (xres, yres)
    exp_x_data: PlotData | None = None
    is_valid: bool = True


class DesignEvaluator:
    """Batched design evaluation - single source of truth for predictions."""

    def __init__(self, model: Any, max_evals: int = 50000, fail_fast: bool = True):
        assert model is not None, "model required for evaluation"
        self.model = model
        self.max_evals = max_evals
        self.fail_fast = fail_fast

    def evaluate_designs(self, inputs: list[DesignInput]) -> list[EvaluatedDesign]:
        """Evaluate all designs in ONE batched call. Returns fully populated results."""
        if not inputs:
            return []

        valid_inputs = [(i, inp) for i, inp in enumerate(inputs) if is_valid_network(inp.network)]
        invalid_indices = {i for i in range(len(inputs))} - {i for i, _ in valid_inputs}

        if invalid_indices:
            logger.info(f"Skipping {len(invalid_indices)} invalid networks")

        if not valid_inputs:
            return [self._make_invalid_result(inp) for inp in inputs]

        precomputed = self._batch_compute(valid_inputs)

        results = []
        for i, inp in enumerate(inputs):
            if i in invalid_indices:
                results.append(self._make_invalid_result(inp))
            else:
                key = (inp.target_name, id(inp.network))
                if key not in precomputed:
                    results.append(self._make_invalid_result(inp))
                    continue
                pred_data = precomputed[key]['pred_data']
                exp_x_data = precomputed[key].get('exp_x_data')
                assert exp_x_data is not None, (
                    f"exp_x_data missing for {inp.target_name} "
                    + f"(key has: {list(precomputed[key].keys())})"
                )
                gt_data = self._compute_gt_data(inp)
                lattice_data, lattice_grid, lattice_extent, lattice_resolution = (
                    self._compute_lattice_data(inp)
                )

                results.append(
                    EvaluatedDesign(
                        input=inp,
                        gt_data=gt_data,
                        pred_data=pred_data,
                        lattice_data=lattice_data,
                        lattice_grid=lattice_grid,
                        lattice_extent=lattice_extent,
                        lattice_resolution=lattice_resolution,
                        exp_x_data=exp_x_data,
                    )
                )
        return results

    def _batch_compute(
        self, valid_inputs: list[tuple[int, DesignInput]]
    ) -> dict[tuple, dict]:
        """Batch compute predictions for all valid inputs."""
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction

        n = len(valid_inputs)
        logger.info(f"Batching predictions for {n} designs...")
        start = time.time()

        networks, predict_at, keys = [], [], []
        for _, inp in valid_inputs:
            td = prepare_target_data(inp.target, max_samples=self.max_evals, seed=42)
            networks.append(inp.network)
            predict_at.append(td.X)
            keys.append((inp.target_name, id(inp.network)))

        network_model = NetworkModel(model=self.model, network=networks)

        predictor = NetworkPrediction(
            predict_at=predict_at,
            max_evals=self.max_evals,
            network_model=network_model,
            already_latent=True,
            enable_gridstats=False,
            device='gpu',
            verbose=False,
            skip_input_reorder=True,  # design: X was passed positionally during optimization
            shuffle_inputs=True,  # shuffle before truncation to avoid biased subsampling
        )

        preds = predictor.get_data(rescale_latent=True)
        assert len(preds) == n, f"prediction count {len(preds)} != {n}"

        precomputed = {}
        for i, pred in enumerate(preds):
            pred_data = PlotData(
                xval=pred.x,
                yval=pred.y,
                input_names=[f'X{j + 1}' for j in range(pred.x.shape[1])],
                output_name='Y',
            )
            precomputed[keys[i]] = {'pred_data': pred_data}

        from biocomptools.toollib.typical_experimental_distribution import sample_latent

        n_inputs = networks[0].nb_inputs
        assert n_inputs > 0, f"network has no inputs: {networks[0]}"
        logger.debug(f"Computing exp_x predictions for {n} networks, n_inputs={n_inputs}")
        exp_x_samples = sample_latent(150000, n_inputs, seed=42)
        logger.debug(f"exp_x_samples shape: {exp_x_samples.shape}")
        exp_predictor = NetworkPrediction(
            predict_at=[exp_x_samples] * n,
            network_model=network_model,
            already_latent=True,
            device='gpu',
            verbose=False,
            skip_input_reorder=True,
        )
        exp_preds = exp_predictor.get_data(rescale_latent=True)
        assert len(exp_preds) == n, f"exp_x prediction count {len(exp_preds)} != {n}"
        for i, exp_pred in enumerate(exp_preds):
            exp_x_data = PlotData(
                xval=exp_pred.x,
                yval=exp_pred.y,
                input_names=[f'X{j + 1}' for j in range(exp_pred.x.shape[1])],
                output_name='Y',
            )
            precomputed[keys[i]]['exp_x_data'] = exp_x_data
        logger.debug(f"Added exp_x_data for {len(exp_preds)} networks")

        logger.info(f"Batched {n} predictions (incl. exp X) in {time.time() - start:.2f}s")

        return precomputed

    def _compute_gt_data(self, inp: DesignInput) -> PlotData:
        """Compute ground truth data for plotting."""
        seed = hash((inp.rank, inp.replicate, inp.target_name)) % (2**31)
        td = prepare_target_data(inp.target, max_samples=20000, seed=seed)
        return td.to_plot_data(model=self.model)

    def _compute_lattice_data(
        self, inp: DesignInput
    ) -> tuple[PlotData | None, np.ndarray | None, tuple | None, tuple | None]:
        """Compute lattice visualization data.

        Returns:
            lattice_data: PlotData for smooth plotting (flattened)
            lattice_grid: 2D array (yres, xres) for pixel-perfect rendering (latent space)
            lattice_extent: (xmin, xmax, ymin, ymax) in latent space
            lattice_resolution: (xres, yres) for reference
        """
        if not hasattr(inp.target, 'get_lattice'):
            return None, None, None, None

        resolution = (48, 48)
        x_ext = getattr(inp.target, 'latent_x', (0.0, 1.0))
        y_ext = getattr(inp.target, 'latent_y', (0.0, 1.0))

        X_grid, Y_grid = inp.target.get_lattice(resolution, seed=0)
        lattice_extent = (x_ext[0], x_ext[1], y_ext[0], y_ext[1])

        Y_grid_2d = Y_grid if Y_grid.ndim == 2 else Y_grid.reshape(resolution[1], resolution[0])

        td = prepare_target_data(
            inp.target, max_samples=48 * 48, seed=0, grid_resolution=resolution
        )
        lattice_data = td.to_plot_data(model=self.model)

        return lattice_data, Y_grid_2d, lattice_extent, resolution

    def _make_invalid_result(self, inp: DesignInput) -> EvaluatedDesign:
        """Create placeholder result for invalid network."""
        empty = PlotData(
            xval=np.zeros((0, 2)),
            yval=np.zeros(0),
            input_names=['X1', 'X2'],
            output_name='Y',
        )
        return EvaluatedDesign(
            input=inp,
            gt_data=empty,
            pred_data=empty,
            lattice_data=None,
            lattice_grid=None,
            lattice_extent=None,
            lattice_resolution=None,
            exp_x_data=empty,
            is_valid=False,
        )
