"""Database utilities for querying Metric table.

For metric calculations (MSE, RMSE, nRMSE, etc.), use biocomp.metric_utils directly.
"""

from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select, and_

from biocomptools.toollib.models import Metric


def get_valid_metrics(
    session: Session,
    metric_name: Optional[str] = None,
    trained_model_name: Optional[str] = None,
    exclude_nulls: bool = True,
) -> list[Metric]:
    """Query metrics with optional filtering for NULL values."""
    conditions = []
    if metric_name:
        conditions.append(Metric.name == metric_name)
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)
    if exclude_nulls:
        conditions.append(Metric.value.is_not(None))

    query = select(Metric).where(and_(*conditions)) if conditions else select(Metric)
    return list(session.exec(query).all())


def get_metrics_with_nulls(
    session: Session, metric_name: Optional[str] = None, trained_model_name: Optional[str] = None
) -> list[Metric]:
    """Get metrics with NULL values (indicating NaN/inf in original data)."""
    conditions = [Metric.value.is_(None)]
    if metric_name:
        conditions.append(Metric.name == metric_name)
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)

    return list(session.exec(select(Metric).where(and_(*conditions))).all())


def count_null_metrics(session: Session) -> dict[str, int]:
    """Count metrics with NULL values grouped by metric name."""
    counts: dict[str, int] = {}
    for m in get_metrics_with_nulls(session):
        counts[m.name] = counts.get(m.name, 0) + 1
    return counts


def summarize_metric_values(
    session: Session, metric_name: str, trained_model_name: Optional[str] = None
) -> dict:
    """Get summary statistics for a metric, including NULL count."""
    conditions = [Metric.name == metric_name]
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)

    total = session.exec(select(func.count(Metric.id)).where(and_(*conditions))).one()

    valid_conditions = conditions + [Metric.value.is_not(None)]
    valid_count, min_val, max_val, avg_val = session.exec(
        select(
            func.count(Metric.id),
            func.min(Metric.value),
            func.max(Metric.value),
            func.avg(Metric.value),
        ).where(and_(*valid_conditions))
    ).one()

    return {
        'total': total,
        'valid': valid_count,
        'null': total - valid_count,
        'min': min_val,
        'max': max_val,
        'avg': avg_val,
    }
