"""
Utilities for working with metrics that may contain NULL values (from NaN/inf).
"""
from typing import List, Optional, Union
from sqlmodel import Session, select, and_, or_
from biocomptools.toollib.models import Metric


def get_valid_metrics(
    session: Session,
    metric_name: Optional[str] = None,
    trained_model_name: Optional[str] = None,
    exclude_nulls: bool = True
) -> List[Metric]:
    """
    Query metrics with optional filtering for NULL values.
    
    Args:
        session: SQLModel database session
        metric_name: Filter by metric name
        trained_model_name: Filter by model name
        exclude_nulls: If True, exclude metrics where value is NULL
        
    Returns:
        List of Metric objects
    """
    query = select(Metric)
    
    conditions = []
    if metric_name:
        conditions.append(Metric.name == metric_name)
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)
    if exclude_nulls:
        conditions.append(Metric.value.is_not(None))
    
    if conditions:
        query = query.where(and_(*conditions))
    
    return session.exec(query).all()


def get_metrics_with_nulls(
    session: Session,
    metric_name: Optional[str] = None,
    trained_model_name: Optional[str] = None
) -> List[Metric]:
    """
    Get metrics that have NULL values (indicating NaN/inf in original data).
    
    Args:
        session: SQLModel database session
        metric_name: Filter by metric name
        trained_model_name: Filter by model name
        
    Returns:
        List of Metric objects with NULL values
    """
    query = select(Metric).where(Metric.value.is_(None))
    
    conditions = [Metric.value.is_(None)]
    if metric_name:
        conditions.append(Metric.name == metric_name)
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)
    
    query = select(Metric).where(and_(*conditions))
    
    return session.exec(query).all()


def count_null_metrics(session: Session) -> dict:
    """
    Count metrics with NULL values grouped by metric name.
    
    Returns:
        Dict mapping metric name to count of NULL values
    """
    from sqlalchemy import func
    
    # Get all metrics with NULL values
    null_metrics = get_metrics_with_nulls(session)
    
    # Group by name
    counts = {}
    for metric in null_metrics:
        counts[metric.name] = counts.get(metric.name, 0) + 1
    
    return counts


def summarize_metric_values(
    session: Session,
    metric_name: str,
    trained_model_name: Optional[str] = None
) -> dict:
    """
    Get summary statistics for a metric, including NULL count.
    
    Returns:
        Dict with 'total', 'valid', 'null', 'min', 'max', 'avg'
    """
    from sqlalchemy import func
    
    conditions = [Metric.name == metric_name]
    if trained_model_name:
        conditions.append(Metric.trained_model_name == trained_model_name)
    
    # Total count
    total_query = select(func.count(Metric.id)).where(and_(*conditions))
    total = session.exec(total_query).one()
    
    # Valid (non-NULL) stats
    valid_conditions = conditions + [Metric.value.is_not(None)]
    valid_query = select(
        func.count(Metric.id),
        func.min(Metric.value),
        func.max(Metric.value),
        func.avg(Metric.value)
    ).where(and_(*valid_conditions))
    
    valid_count, min_val, max_val, avg_val = session.exec(valid_query).one()
    
    return {
        'total': total,
        'valid': valid_count,
        'null': total - valid_count,
        'min': min_val,
        'max': max_val,
        'avg': avg_val
    }


# Example usage:
"""
from sqlmodel import create_engine, Session
from biocomptools.toollib.metric_utils import get_valid_metrics, summarize_metric_values

engine = create_engine("sqlite:///biocompdb.sqlite")
with Session(engine) as session:
    # Get all valid RMSE metrics
    valid_rmse = get_valid_metrics(session, metric_name="RMSE")
    
    # Get summary including NULLs
    summary = summarize_metric_values(session, "MSE")
    print(f"MSE metrics: {summary['valid']} valid, {summary['null']} NULL/NaN")
"""