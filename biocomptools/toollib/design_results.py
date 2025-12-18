import json
import numpy as np
from pathlib import Path
from typing import Optional, Any, Union
from dataclasses import dataclass, field, asdict

from biocomptools.logging_config import get_logger
from biocomp.metric_utils import RegressionStats, DistributionStats

logger = get_logger(__name__)


@dataclass
class RegressionMetrics:
    """regression metrics for design evaluation.

    wraps biocomp.metric_utils.RegressionStats with design-specific field names.
    """

    rmse: float
    mae: float
    r2: float
    pearson_r: float
    pearson_p: float
    max_error: float
    p95_error: float

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "RegressionMetrics":
        stats = RegressionStats.compute(y_true, y_pred)
        return cls(
            rmse=stats.rmse,
            mae=stats.mae,
            r2=stats.r2,
            pearson_r=stats.pearson_r,
            pearson_p=stats.pearson_p,
            max_error=stats.max_error,
            p95_error=stats.p95_error,
        )


@dataclass
class DistributionMetrics:
    """distribution metrics for design evaluation.

    wraps biocomp.metric_utils.DistributionStats with design-specific field names.
    """

    target_mean: float
    target_std: float
    target_min: float
    target_max: float
    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "DistributionMetrics":
        stats = DistributionStats.compute(y_true, y_pred)
        return cls(
            target_mean=stats.target_mean,
            target_std=stats.target_std,
            target_min=stats.target_min,
            target_max=stats.target_max,
            prediction_mean=stats.pred_mean,
            prediction_std=stats.pred_std,
            prediction_min=stats.pred_min,
            prediction_max=stats.pred_max,
        )


@dataclass
class LossComponents:
    total: float
    sinkhorn: Optional[float] = None
    lncc: Optional[float] = None
    spectral: Optional[float] = None
    over1_penalty: Optional[float] = None


@dataclass
class NREMetrics:
    design_nre: Optional[float] = None
    baseline_nre: Optional[float] = None
    design_nrmse: Optional[float] = None
    baseline_nrmse: Optional[float] = None
    data_nrmse: Optional[float] = None


@dataclass
class DesignMetrics:
    target_name: str
    network_name: str
    replicate_id: int
    network_id: int
    rank: int
    step: int
    loss: LossComponents
    regression: RegressionMetrics
    distribution: DistributionMetrics
    recipe_summary: dict = field(default_factory=dict)
    nre: Optional[NREMetrics] = None

    def to_dict(self) -> dict:
        result = {
            'target_name': self.target_name,
            'network_name': self.network_name,
            'replicate_id': self.replicate_id,
            'network_id': self.network_id,
            'rank': self.rank,
            'step': self.step,
            'loss': asdict(self.loss),
            'regression': asdict(self.regression),
            'distribution': asdict(self.distribution),
            'recipe_summary': self.recipe_summary,
        }
        if self.nre is not None:
            result['nre'] = asdict(self.nre)
        return result

    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DesignResultsManager:
    def __init__(self, base_dir: Union[str, Path], step: Optional[int] = None):
        self.base_dir, self.step = Path(base_dir), step
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ('targets', 'checkpoints', 'comparison'):
            (self.base_dir / subdir).mkdir(exist_ok=True)

    def get_target_dir(self, target_name: str) -> Path:
        d = self.base_dir / 'targets' / self._sanitize_name(target_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_rank_dir(self, target_name: str, rank: int, step: Optional[int] = None) -> Path:
        step_dir = self.get_target_dir(target_name) / (
            'final' if step is None else f'steps/step_{step:06d}'
        )
        rank_dir = step_dir / f'rank_{rank:02d}'
        rank_dir.mkdir(parents=True, exist_ok=True)
        return rank_dir

    def get_checkpoint_dir(self, step: int) -> Path:
        d = self.base_dir / 'checkpoints' / f'step_{step:06d}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_comparison_dir(self) -> Path:
        return self.base_dir / 'comparison'

    @staticmethod
    def _sanitize_name(name: str) -> str:
        for c in '/ \\ : * ? " < > |'.split():
            name = name.replace(c, '_')
        return name[:200]

    def save_target_summary(self, target_name: str, summary: dict):
        (self.get_target_dir(target_name) / 'target_summary.json').write_text(
            json.dumps(summary, indent=2)
        )

    def save_rankings(self, target_name: str, rankings: list, step: Optional[int] = None):
        out_dir = self.get_target_dir(target_name) / (
            'final' if step is None else f'steps/step_{step:06d}'
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'rankings.json').write_text(
            json.dumps(
                [
                    {'rank': i + 1, 'replicate_id': r, 'network_id': n, 'loss': float(loss)}
                    for i, (r, n, loss) in enumerate(rankings)
                ],
                indent=2,
            )
        )


def compute_nre_for_network(
    target: Any, network: Any, model: Any, max_evals: int = 50000
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    from biocomp.design import DataTarget

    if not isinstance(target, DataTarget) or model is None or network is None:
        return None, None, None
    try:
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction

        X, Y = target.X, target.Y
        if len(X) > max_evals:
            idx = np.random.default_rng(42).choice(len(X), max_evals, replace=False)
            X, Y = X[idx], Y[idx]

        predictor = NetworkPrediction(
            predict_at=[X],
            ground_truth=[Y.reshape(-1, 1) if Y.ndim == 1 else Y],
            max_evals=max_evals,
            network_model=NetworkModel(model=model, network=[network]),
            already_latent=True,
            enable_gridstats=True,
            device='gpu',
            verbose=False,
        )
        s = predictor.get_network_stats()
        return (
            (s[0].get('noise_relative_error'), s[0].get('grid_nrmse'), s[0].get('data_nrmse'))
            if s
            else (None, None, None)
        )
    except Exception as e:
        logger.warning(f"Failed to compute NRE: {e}")
        return None, None, None


def compute_design_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_value: float,
    target_name: str,
    network_name: str,
    replicate_id: int,
    network_id: int,
    rank: int,
    step: int,
    loss_components: Optional[dict] = None,
    recipe_info: Optional[dict] = None,
    nre_metrics: Optional[NREMetrics] = None,
) -> DesignMetrics:
    lc = loss_components or {}
    return DesignMetrics(
        target_name,
        network_name,
        replicate_id,
        network_id,
        rank,
        step,
        LossComponents(
            float(loss_value),
            lc.get('sinkhorn'),
            lc.get('lncc'),
            lc.get('spectral'),
            lc.get('over1_penalty'),
        ),
        RegressionMetrics.compute(y_true, y_pred),
        DistributionMetrics.compute(y_true, y_pred),
        recipe_info or {},
        nre_metrics,
    )


def extract_recipe_summary(network: Any, params: Any = None) -> dict:
    summary = {
        'network_name': getattr(network, 'name', 'unknown'),
        'uorfs': [],
        'ratios': {},
        'parts': [],
    }
    try:
        if hasattr(network, 'graph'):
            for nid, ndata in network.graph.nodes.items():
                extra = ndata.get('extra', {})
                if 'uorf' in str(ndata.get('type', '')).lower():
                    summary['uorfs'].append(
                        {'node_id': str(nid), 'value': extra.get('part_name', 'unknown')}
                    )
                if ndata.get('type') == 'aggregation' and 'ratios' in extra:
                    summary['ratios'][extra.get('cotx_group', 'unknown')] = list(
                        map(float, extra['ratios'])
                    )
        if (
            hasattr(network, 'source_recipe')
            and network.source_recipe
            and hasattr(network.source_recipe, 'content')
        ):
            for cotx in network.source_recipe.content:
                for tu in getattr(cotx, 'units', []):
                    for slot in getattr(tu, 'slots', []):
                        if 'uORF' in str(slot):
                            summary['uorfs'].append(str(slot))
    except Exception as e:
        logger.warning(f"Failed to extract recipe summary: {e}")
    return summary
