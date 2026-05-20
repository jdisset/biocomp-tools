# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
## {{{                          --     imports     --

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

##────────────────────────────────────────────────────────────────────────────}}}


class NetworkDataPairMetrics(BaseModel):
    """Metrics for a single network-datafile pair."""
    network_name: str
    networkdatapair: Dict[str, Any]
    RMSE: float
    MSE: float
    n_samples: int
    grid_RMSE: Optional[float] = None
    grid_MSE: Optional[float] = None


class ReplicateMetrics(BaseModel):
    """Metrics for a single training replicate."""
    replicate: int
    overall_RMSE: float
    overall_MSE: float
    n_samples: int
    per_networkdatapair: List[NetworkDataPairMetrics] = Field(default_factory=list)
    avg_grid_rmse: Optional[float] = None
    avg_grid_mse: Optional[float] = None
    sublosses: Optional[Dict[str, float]] = None


class StepMetrics(BaseModel):
    """Metrics for a single training/validation step."""
    step: int
    metrics: List[ReplicateMetrics]
    training_loss: Optional[Any] = None  # Can be array or scalar
    eval_time: Optional[float] = None


class LoggerMetricsHistory(BaseModel):
    """Complete history of metrics from a logger."""
    logger_name: str
    logger_type: str
    history: List[StepMetrics] = Field(default_factory=list)

    def get_latest_metrics(self, replicate: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get the latest metrics, optionally for a specific replicate."""
        if not self.history:
            return None

        latest_step = self.history[-1]

        if replicate is not None:
            if replicate < len(latest_step.metrics):
                return {f'{self.logger_name}_{self.logger_type}': latest_step.metrics[replicate]}
            else:
                return None
        else:
            return {f'{self.logger_type}::{self.logger_name}': latest_step.metrics}

    def add_step_metrics(self, step: int, metrics: List[ReplicateMetrics],
                        training_loss: Optional[Any] = None, eval_time: Optional[float] = None):
        """Add metrics for a new step."""
        step_metrics = StepMetrics(
            step=step,
            metrics=metrics,
            training_loss=training_loss,
            eval_time=eval_time
        )
        self.history.append(step_metrics)
