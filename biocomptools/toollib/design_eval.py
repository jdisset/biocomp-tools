"""Design evaluation utilities - batched prediction and baseline NRE computation.

All committed networks are batched into ONE NetworkModel to avoid repeated JIT compilation.
This module uses defensive assertions to ensure correct indexing throughout the pipeline.
"""

import numpy as np
import time
import traceback
from typing import Optional, Any

from biocomp.plotutils import PlotData
from biocomp.design import DataTarget
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DesignEvaluator:
    """Evaluates design candidates via batched predictions for efficiency.

    Key optimization: ALL committed networks are batched into ONE NetworkModel,
    avoiding repeated JIT compilation (~25s per batch).
    """

    def __init__(self, model: Any, max_evals: int = 50000):
        self.model = model
        self.max_evals = max_evals

    def _prepare_target_data(
        self, target: Any, network: Any = None, max_samples: int = 20000, seed: int = 42
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare X/Y data for a target, subsampling if needed.

        Args:
            target: The design target (DataTarget, SVGTarget, etc.)
            network: If provided and target is DataTarget, reorder X columns to match
                     the network's expected input order.
            max_samples: Maximum samples to return (subsample if needed)
            seed: Random seed for subsampling

        Returns:
            (X_latent, Y_gt) tuple where X columns are ordered to match network's inputs
        """
        if isinstance(target, DataTarget):
            if network is not None:
                X_latent = target.get_reordered_X(network)
            else:
                X_latent = target.X
            Y_gt = target.Y
            if len(X_latent) > max_samples:
                idx = np.random.default_rng(seed).choice(len(X_latent), max_samples, replace=False)
                X_latent = X_latent[idx]
                Y_gt = Y_gt[idx] if Y_gt is not None else None
        else:
            X_latent, _ = target.sample_uniform(min(10000, max_samples), seed=seed)
            Y_gt = None
        return X_latent, Y_gt

    def precompute_for_design_results(
        self, design_results: list[dict]
    ) -> tuple[dict[tuple, dict], dict[int, Optional[float]]]:
        """Precompute predictions for ALL design results in ONE batched call.

        Optimization: Instead of building a separate NetworkModel per target group,
        we build ONE NetworkModel with ALL committed networks across all targets.
        This avoids ~25s JIT compilation per target group.

        Returns:
            (precomputed, baseline_cache) where:
            - precomputed: dict[(target_name, net_id)] -> {'pred_data': ..., 'design_nre': ...}
            - baseline_cache: dict[id(original_network)] -> baseline_nre
        """
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction

        if not design_results:
            return {}, {}

        n_results = len(design_results)
        logger.info(f"Batching predictions for {n_results} design results in ONE call...")
        start_time = time.time()

        all_networks = []
        all_predict_at = []
        all_ground_truth = []
        result_indices = []  # tracks (target_name, network_id(obj)) for each position

        for i, r in enumerate(design_results):
            network = r['network']
            target = r['target']
            target_name = r['target_name']

            assert network is not None, f"design_results[{i}] has None network"
            assert target is not None, f"design_results[{i}] has None target"

            X, Y = self._prepare_target_data(target, network=network)
            assert X.ndim == 2, f"X must be 2D, got {X.ndim}D for result {i}"

            all_networks.append(network)
            all_predict_at.append(X)
            if isinstance(target, DataTarget) and Y is not None:
                Y_shaped = Y.reshape(-1, 1) if Y.ndim == 1 else Y
                all_ground_truth.append(Y_shaped)
            else:
                all_ground_truth.append(None)

            result_indices.append((target_name, id(network)))

        assert len(all_networks) == n_results, f"network count mismatch: {len(all_networks)} != {n_results}"
        assert len(result_indices) == n_results, f"index count mismatch: {len(result_indices)} != {n_results}"

        logger.info(f"Building batched NetworkModel for {n_results} networks (all targets)...")
        model_start = time.time()
        network_model = NetworkModel(model=self.model, network=all_networks)
        logger.info(f"Batched NetworkModel built in {time.time() - model_start:.2f}s")

        has_any_gt = any(gt is not None for gt in all_ground_truth)
        enable_gridstats = has_any_gt

        predictor = NetworkPrediction(
            predict_at=all_predict_at,
            ground_truth=all_ground_truth if has_any_gt else None,
            max_evals=self.max_evals,
            network_model=network_model,
            already_latent=True,
            enable_gridstats=enable_gridstats,
            device='gpu',
            verbose=False,
        )

        pred_results = predictor.get_data(rescale_latent=True)
        nre_stats = predictor.get_network_stats() if enable_gridstats else None

        assert len(pred_results) == n_results, (
            f"prediction count mismatch: {len(pred_results)} != {n_results}"
        )

        precomputed: dict[tuple, dict] = {}
        for i, pred in enumerate(pred_results):
            key = result_indices[i]
            assert key not in precomputed, f"duplicate key {key} at index {i}"

            pred_data = PlotData(
                xval=pred.x,
                yval=pred.y,
                input_names=[f'X{j + 1}' for j in range(pred.x.shape[1])],
                output_name='Y',
            )
            nre = nre_stats[i].get('noise_relative_error') if nre_stats and i < len(nre_stats) else None

            precomputed[key] = {'pred_data': pred_data, 'design_nre': nre}

        assert len(precomputed) == n_results, (
            f"precomputed count mismatch: {len(precomputed)} != {n_results}"
        )

        logger.info(f"All {n_results} predictions computed in {time.time() - start_time:.2f}s")

        baseline_cache = self._compute_baseline_nres(design_results)
        return precomputed, baseline_cache

    def _compute_baseline_nres(self, design_results: list[dict]) -> dict[int, Optional[float]]:
        """Compute baseline NRE for original networks (batched)."""
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction

        baseline_groups: dict[int, tuple] = {}
        for r in design_results:
            target = r['target']
            if not isinstance(target, DataTarget) or target.original_network is None:
                continue
            orig_net = target.original_network
            net_key = id(orig_net)
            if net_key not in baseline_groups:
                baseline_groups[net_key] = (orig_net, target)

        if not baseline_groups:
            return {}

        baseline_items = list(baseline_groups.items())
        n_baselines = len(baseline_items)
        networks = [item[1][0] for item in baseline_items]
        targets = [item[1][1] for item in baseline_items]

        logger.info(f"Computing baseline NRE for {n_baselines} unique original networks...")
        start_time = time.time()

        assert len(networks) == len(targets), f"networks/targets length mismatch: {len(networks)} != {len(targets)}"

        result_map = {}
        try:
            network_model = NetworkModel(model=self.model, network=networks)

            predict_at, ground_truth = [], []
            for net, target in zip(networks, targets, strict=True):
                X, Y = self._prepare_target_data(target, network=net)
                predict_at.append(X)
                ground_truth.append(Y.reshape(-1, 1) if Y.ndim == 1 else Y)

            predictor = NetworkPrediction(
                predict_at=predict_at,
                ground_truth=ground_truth,
                max_evals=self.max_evals,
                network_model=network_model,
                already_latent=True,
                enable_gridstats=True,
                device='gpu',
                verbose=False,
            )
            stats = predictor.get_network_stats()

            assert stats is not None, "expected stats from baseline prediction"
            assert len(stats) == n_baselines, f"stats count {len(stats)} != {n_baselines}"

            for i, (net_key, _) in enumerate(baseline_items):
                result_map[net_key] = stats[i].get('noise_relative_error')

            logger.info(f"Baseline NRE computed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.warning(f"Batched baseline NRE computation failed: {e}")
            logger.debug(traceback.format_exc())

        return result_map

    # kept for backward compatibility but prefer precompute_for_design_results
    def compute_baseline_nres(self, design_results: list[dict]) -> dict[int, Optional[float]]:
        return self._compute_baseline_nres(design_results)
