"""Design evaluation utilities - batched prediction and baseline NRE computation."""

import numpy as np
import traceback
from typing import Optional, Any
from collections import defaultdict

from biocomp.plotutils import PlotData
from biocomp.design import DataTarget
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class DesignEvaluator:
    """Evaluates design candidates via batched predictions for efficiency."""

    def __init__(self, model: Any, max_evals: int = 50000):
        self.model = model
        self.max_evals = max_evals

    def batch_predictions_for_target(
        self, group: list[dict], target: Any = None
    ) -> tuple[dict[int, PlotData], dict[int, Optional[float]]]:
        """
        Batch predictions for multiple networks sharing a target.

        Args:
            group: List of design result dicts with 'network' and 'target' keys
            target: Optional target override (uses group[0]['target'] if None)

        Returns:
            (pred_data_map, nre_map) keyed by group index
        """
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction
        import time

        if not group:
            return {}, {}

        target = target or group[0]['target']
        networks = [r['network'] for r in group]
        n_networks = len(networks)

        if isinstance(target, DataTarget):
            X_latent = target.X
            Y_gt = target.Y
            if len(X_latent) > 20000:
                idx = np.random.default_rng(42).choice(len(X_latent), 20000, replace=False)
                X_latent, Y_gt = X_latent[idx], Y_gt[idx] if Y_gt is not None else None
        else:
            X_latent, _ = target.sample_uniform(10000, seed=42)
            Y_gt = None

        start_time = time.time()
        logger.info(f"Building batched NetworkModel for {n_networks} networks...")
        network_model = NetworkModel(model=self.model, network=networks)
        logger.info(f"Batched NetworkModel built in {time.time() - start_time:.2f}s")

        predict_at = [X_latent] * n_networks
        ground_truth = None
        if isinstance(target, DataTarget) and Y_gt is not None:
            gt_shaped = Y_gt.reshape(-1, 1) if Y_gt.ndim == 1 else Y_gt
            ground_truth = [gt_shaped] * n_networks

        predictor = NetworkPrediction(
            predict_at=predict_at,
            ground_truth=ground_truth,
            max_evals=self.max_evals,
            network_model=network_model,
            already_latent=True,
            enable_gridstats=isinstance(target, DataTarget),
            device='gpu',
            verbose=False,
        )

        pred_results = predictor.get_data(rescale_latent=True)
        nre_stats = predictor.get_network_stats() if isinstance(target, DataTarget) else None

        pred_data_map, nre_map = {}, {}
        for i, pred in enumerate(pred_results):
            pred_data_map[i] = PlotData(
                xval=pred.x,
                yval=pred.y,
                input_names=[f'X{j + 1}' for j in range(pred.x.shape[1])],
                output_name='Y',
            )
            if nre_stats and i < len(nre_stats):
                nre_map[i] = nre_stats[i].get('noise_relative_error')

        return pred_data_map, nre_map

    def compute_baseline_nres(self, design_results: list[dict]) -> dict[int, Optional[float]]:
        """
        Compute baseline NRE for original networks.

        Args:
            design_results: List of design result dicts with 'target' containing DataTarget

        Returns:
            Dict mapping id(original_network) -> NRE value
        """
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction
        import time

        baseline_groups: dict[int, tuple] = {}
        for r in design_results:
            target = r['target']
            if not isinstance(target, DataTarget) or target.original_network is None:
                continue
            orig_net = target.original_network
            net_id = id(orig_net)
            if net_id not in baseline_groups:
                baseline_groups[net_id] = (orig_net, target)

        if not baseline_groups:
            return {}

        baseline_items = list(baseline_groups.items())
        networks = [item[1][0] for item in baseline_items]
        targets = [item[1][1] for item in baseline_items]

        logger.info(f"Computing baseline NRE for {len(networks)} unique original networks...")
        start_time = time.time()

        result_map = {}
        try:
            network_model = NetworkModel(model=self.model, network=networks)
            predict_at, ground_truth = [], []
            for target in targets:
                X, Y = target.X, target.Y
                if len(X) > self.max_evals:
                    idx = np.random.default_rng(42).choice(len(X), self.max_evals, replace=False)
                    X, Y = X[idx], Y[idx]
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

            for i, (net_id, _) in enumerate(baseline_items):
                if stats and i < len(stats):
                    result_map[net_id] = stats[i].get('noise_relative_error')

            logger.info(f"Baseline NRE computed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.warning(f"Batched baseline NRE computation failed: {e}")
            logger.debug(traceback.format_exc())

        return result_map

    def precompute_for_design_results(
        self, design_results: list[dict]
    ) -> tuple[dict[tuple, dict], dict[int, Optional[float]]]:
        """
        Precompute predictions and baselines for all design results.

        Args:
            design_results: List of design result dicts

        Returns:
            (precomputed, baseline_cache) where:
            - precomputed: dict[(target_name, net_id)] -> {'pred_data': ..., 'design_nre': ...}
            - baseline_cache: dict[id(original_network)] -> baseline_nre
        """
        by_target: dict[str, list[dict]] = defaultdict(list)
        for r in design_results:
            by_target[r['target_name']].append(r)

        precomputed: dict[tuple, dict] = {}
        for target_name, group in by_target.items():
            try:
                pred_data_map, nre_map = self.batch_predictions_for_target(group)
                for i, r in enumerate(group):
                    net_key = id(r['network'])
                    precomputed[(target_name, net_key)] = {
                        'pred_data': pred_data_map.get(i),
                        'design_nre': nre_map.get(i),
                    }
            except Exception as e:
                logger.warning(f"Batched prediction failed for target {target_name}: {e}")
                logger.debug(traceback.format_exc())

        baseline_cache = self.compute_baseline_nres(design_results)
        return precomputed, baseline_cache
