"""Logging utilities for hyperopt."""

from typing import Callable
from tqdm import tqdm
import optuna


class TqdmProgressLogger:
    """Progress bar for hyperopt trials."""

    def __init__(self, n_trials: int, desc: str = "Hyperopt"):
        self.pbar = tqdm(total=n_trials, desc=desc)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.update()


class OptunaPruningLogger:
    """Log pruning decisions for debugging."""

    def __init__(self, log_fn: Callable[[str], None] = print):
        self.log = log_fn

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.log(f"Trial {trial.number} pruned at step {trial.intermediate_values}")


def create_pruner(
    pruning_method: str = "median",
    n_warmup_steps: int = 5,
    n_startup_trials: int = 10,
    percentile: float = 25.0,
) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner."""
    if pruning_method == "none":
        return optuna.pruners.NopPruner()
    if pruning_method == "median":
        return optuna.pruners.MedianPruner(
            n_warmup_steps=n_warmup_steps,
            n_startup_trials=n_startup_trials,
        )
    if pruning_method == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=percentile,
            n_warmup_steps=n_warmup_steps,
            n_startup_trials=n_startup_trials,
        )
    raise ValueError(f"Unknown pruner: {pruning_method}")
