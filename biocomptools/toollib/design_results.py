"""Design metrics and results management.

Data structures for design evaluation metrics. Metric computation is handled by
DesignEvaluator in design_eval.py - this module only defines data structures.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

from biocomp.metric_utils import RegressionStats, DistributionStats
from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from biocomp.network import Network
    import biocomp.parameters as pr

logger = get_logger(__name__)


@dataclass
class LossComponents:
    """Design loss breakdown."""

    total: float
    sinkhorn: float | None = None
    lncc: float | None = None
    spectral: float | None = None
    tucount_penalty: float | None = None


@dataclass
class NREMetrics:
    """NRE-specific metrics for design evaluation."""

    design_nre: float | None = None
    baseline_nre: float | None = None
    design_nrmse: float | None = None
    baseline_nrmse: float | None = None
    data_nrmse: float | None = None


@dataclass
class DesignMetrics:
    """Complete design result metrics."""

    target_name: str
    network_name: str
    replicate_id: int
    network_id: int
    rank: int
    step: int
    loss: LossComponents
    regression: RegressionStats
    distribution: DistributionStats
    recipe_summary: dict = field(default_factory=dict)
    nre: NREMetrics | None = None
    fingerprint: str | None = None

    def to_dict(self) -> dict:
        return {
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
            **(({'nre': asdict(self.nre)}) if self.nre else {}),
            **(({'fingerprint': self.fingerprint}) if self.fingerprint else {}),
        }

    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DesignResultsManager:
    """File I/O for design results."""

    def __init__(self, base_dir: str | Path, step: int | None = None):
        self.base_dir = Path(base_dir)
        self.step = step
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ('targets', 'checkpoints', 'comparison'):
            (self.base_dir / subdir).mkdir(exist_ok=True)

    def get_target_dir(self, target_name: str) -> Path:
        d = self.base_dir / 'targets' / self._sanitize_name(target_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_rank_dir(self, target_name: str, rank: int, step: int | None = None) -> Path:
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

    def save_rankings(self, target_name: str, rankings: list, step: int | None = None):
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
    loss_components: dict | None = None,
    recipe_info: dict | None = None,
    nre_metrics: NREMetrics | None = None,
    fingerprint: str | None = None,
) -> DesignMetrics:
    """Create DesignMetrics from evaluation data."""
    lc = loss_components or {}
    return DesignMetrics(
        target_name=target_name,
        network_name=network_name,
        replicate_id=replicate_id,
        network_id=network_id,
        rank=rank,
        step=step,
        loss=LossComponents(
            total=float(loss_value),
            sinkhorn=lc.get('sinkhorn'),
            lncc=lc.get('lncc'),
            spectral=lc.get('spectral'),
            tucount_penalty=lc.get('tucount_penalty'),
        ),
        regression=RegressionStats.compute(y_true, y_pred),
        distribution=DistributionStats.compute(y_true, y_pred),
        recipe_summary=recipe_info or {},
        nre=nre_metrics,
        fingerprint=fingerprint,
    )


def extract_recipe_summary(network: "Network", params: "pr.ParameterTree | None" = None) -> dict[str, object]:
    """Extract recipe information from a network.

    Returns a summary dict with network_name, uorfs list, ratios dict, and parts list.
    On extraction failure, logs a warning and returns partial results.
    """
    summary: dict[str, object] = {
        'network_name': getattr(network, 'name', 'unknown'),
        'uorfs': [],
        'ratios': {},
        'parts': [],
    }

    uorfs: list[dict[str, str] | str] = []
    ratios: dict[str, list[float]] = {}

    if hasattr(network, 'graph') and network.graph is not None:
        for nid, ndata in network.graph.nodes.items():
            if not isinstance(ndata, dict):
                continue

            node_type = ndata.get('type', '')
            extra = ndata.get('extra')
            if extra is None:
                extra = {}

            if not isinstance(extra, dict):
                logger.warning(f"Node {nid} has non-dict extra field: {type(extra)}")
                continue

            # Extract uORF info
            if 'uorf' in str(node_type).lower():
                part_name = extra.get('part_name')
                if part_name is None:
                    logger.debug(f"Node {nid} is uorf type but missing part_name")
                    part_name = 'unknown'
                uorfs.append({'node_id': str(nid), 'value': str(part_name)})

            # Extract aggregation ratios
            if node_type == 'aggregation':
                members_data = extra.get('members')
                cotx_group = extra.get('cotx_group')

                if members_data is not None and isinstance(members_data, dict) and members_data:
                    ratio_vals: list[float] = []
                    for m in sorted(members_data.keys()):
                        member = members_data[m]
                        if isinstance(member, dict):
                            ratio_vals.append(float(member.get("ratio", 1.0)))
                        else:
                            ratio_vals.append(1.0)

                    if cotx_group is None:
                        logger.debug(f"Node {nid} aggregation missing cotx_group")
                        cotx_group = 'unknown'
                    ratios[str(cotx_group)] = ratio_vals

    # Extract from source_recipe if available
    if hasattr(network, 'source_recipe') and network.source_recipe is not None:
        source_recipe = network.source_recipe
        if hasattr(source_recipe, 'content') and source_recipe.content is not None:
            for cotx in source_recipe.content:
                for tu in getattr(cotx, 'units', []):
                    for slot in getattr(tu, 'slots', []):
                        if 'uORF' in str(slot):
                            uorfs.append(str(slot))

    summary['uorfs'] = uorfs
    summary['ratios'] = ratios
    return summary
