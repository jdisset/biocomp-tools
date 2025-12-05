import json
import numpy as np
from pathlib import Path
from typing import Optional, Any, Union
from dataclasses import dataclass, field, asdict
from scipy import stats

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    pearson_r: float
    pearson_p: float
    max_error: float
    p95_error: float

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "RegressionMetrics":
        y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(valid):
            return cls(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        y_true, y_pred = y_true[valid], y_pred[valid]
        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        ss_res, ss_tot = np.sum(errors**2), np.sum((y_true - np.mean(y_true))**2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred) if len(y_true) > 2 else (np.nan, np.nan)

        return cls(
            rmse=float(np.sqrt(np.mean(errors**2))),
            mae=float(np.mean(abs_errors)),
            r2=r2,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            max_error=float(np.max(abs_errors)),
            p95_error=float(np.percentile(abs_errors, 95)),
        )


@dataclass
class DistributionMetrics:
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
        y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        return cls(
            float(np.nanmean(y_true)), float(np.nanstd(y_true)),
            float(np.nanmin(y_true)), float(np.nanmax(y_true)),
            float(np.nanmean(y_pred)), float(np.nanstd(y_pred)),
            float(np.nanmin(y_pred)), float(np.nanmax(y_pred)),
        )


@dataclass
class LossComponents:
    total: float
    sinkhorn: Optional[float] = None
    lncc: Optional[float] = None
    spectral: Optional[float] = None
    over1_penalty: Optional[float] = None


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

    def to_dict(self) -> dict:
        return {
            'target_name': self.target_name, 'network_name': self.network_name,
            'replicate_id': self.replicate_id, 'network_id': self.network_id,
            'rank': self.rank, 'step': self.step,
            'loss': asdict(self.loss), 'regression': asdict(self.regression),
            'distribution': asdict(self.distribution), 'recipe_summary': self.recipe_summary,
        }

    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DesignResultsManager:
    def __init__(self, base_dir: Union[str, Path], step: Optional[int] = None):
        self.base_dir = Path(base_dir)
        self.step = step
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ('targets', 'checkpoints', 'comparison'):
            (self.base_dir / subdir).mkdir(exist_ok=True)

    def get_target_dir(self, target_name: str) -> Path:
        target_dir = self.base_dir / 'targets' / self._sanitize_name(target_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_rank_dir(self, target_name: str, rank: int, step: Optional[int] = None) -> Path:
        target_dir = self.get_target_dir(target_name)
        step_dir = target_dir / ('final' if step is None else f'steps/step_{step:06d}')
        rank_dir = step_dir / f'rank_{rank:02d}'
        rank_dir.mkdir(parents=True, exist_ok=True)
        return rank_dir

    def get_checkpoint_dir(self, step: int) -> Path:
        checkpoint_dir = self.base_dir / 'checkpoints' / f'step_{step:06d}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def get_comparison_dir(self) -> Path:
        return self.base_dir / 'comparison'

    @staticmethod
    def _sanitize_name(name: str) -> str:
        for old, new in {'/' : '_', '\\': '_', ':': '_', '*': '_', '?': '_', '"': '_', '<': '_', '>': '_', '|': '_', ' ': '_'}.items():
            name = name.replace(old, new)
        return name[:200]

    def save_target_summary(self, target_name: str, summary: dict):
        with open(self.get_target_dir(target_name) / 'target_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def save_rankings(self, target_name: str, rankings: list, step: Optional[int] = None):
        out_dir = self.get_target_dir(target_name) / ('final' if step is None else f'steps/step_{step:06d}')
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'rankings.json', 'w') as f:
            json.dump([{'rank': i+1, 'replicate_id': r, 'network_id': n, 'loss': float(l)} for i, (r, n, l) in enumerate(rankings)], f, indent=2)


def compute_design_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, loss_value: float,
    target_name: str, network_name: str, replicate_id: int, network_id: int,
    rank: int, step: int, loss_components: Optional[dict] = None, recipe_info: Optional[dict] = None,
) -> DesignMetrics:
    lc = loss_components or {}
    return DesignMetrics(
        target_name=target_name, network_name=network_name,
        replicate_id=replicate_id, network_id=network_id, rank=rank, step=step,
        loss=LossComponents(float(loss_value), lc.get('sinkhorn'), lc.get('lncc'), lc.get('spectral'), lc.get('over1_penalty')),
        regression=RegressionMetrics.compute(y_true, y_pred),
        distribution=DistributionMetrics.compute(y_true, y_pred),
        recipe_summary=recipe_info or {},
    )


def extract_recipe_summary(network: Any, params: Any = None) -> dict:
    summary = {'network_name': getattr(network, 'name', 'unknown'), 'uorfs': [], 'ratios': {}, 'parts': []}
    try:
        if hasattr(network, 'graph'):
            for node_id, node_data in network.graph.nodes.items():
                extra = node_data.get('extra', {})
                if 'uorf' in str(node_data.get('type', '')).lower():
                    summary['uorfs'].append({'node_id': str(node_id), 'value': extra.get('part_name', 'unknown')})
                if node_data.get('type') == 'aggregation' and 'ratios' in extra:
                    summary['ratios'][extra.get('cotx_group', 'unknown')] = list(map(float, extra['ratios']))
        if hasattr(network, 'source_recipe') and network.source_recipe:
            recipe = network.source_recipe
            if hasattr(recipe, 'content'):
                for cotx in recipe.content:
                    if hasattr(cotx, 'units'):
                        for tu in cotx.units:
                            if hasattr(tu, 'slots'):
                                for slot in tu.slots:
                                    if 'uORF' in str(slot):
                                        summary['uorfs'].append(str(slot))
    except Exception as e:
        logger.warning(f"Failed to extract recipe summary: {e}")
    return summary
