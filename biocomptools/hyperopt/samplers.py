# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Optuna sampler creation utilities."""

from typing import Optional
from optuna.samplers import BaseSampler, TPESampler, CmaEsSampler, QMCSampler, RandomSampler

try:
    import optunahub

    _OPTUNAHUB = True
except ImportError:
    _OPTUNAHUB = False


def create_sampler(
    sampler_type: str = "tpe",
    seed: Optional[int] = None,
    n_startup_trials: int = 10,
    # CMA-ES params
    cmaes_restart_strategy: Optional[str] = None,
    cmaes_with_margin: bool = False,
    cmaes_popsize: Optional[int] = None,
    cmaes_sigma0: float = 0.5,
    cmaes_source_trials: Optional[list] = None,
    cmaes_warn_independent_sampling: bool = False,
) -> BaseSampler:
    """Create an Optuna sampler based on configuration."""
    if sampler_type == "tpe":
        return TPESampler(seed=seed, n_startup_trials=n_startup_trials)

    if sampler_type == "cmaes":
        restart = cmaes_restart_strategy if cmaes_restart_strategy != "none" else None
        # use optunahub for restart strategies (ipop/bipop)
        if restart in ("ipop", "bipop") and _OPTUNAHUB:
            try:
                mod = optunahub.load_module("samplers/restart_cmaes")
                return mod.RestartCmaEsSampler(
                    seed=seed,
                    n_startup_trials=n_startup_trials,
                    popsize=cmaes_popsize,
                    restart_strategy=restart,
                    warn_independent_sampling=cmaes_warn_independent_sampling,
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"optunahub restart_cmaes failed: {e}. Falling back to standard CmaEsSampler.",
                    stacklevel=2,
                )
        # standard CmaEsSampler - source_trials and sigma0 cannot be used together
        kwargs = dict(
            seed=seed,
            restart_strategy=restart,
            with_margin=cmaes_with_margin,
            popsize=cmaes_popsize,
            warn_independent_sampling=cmaes_warn_independent_sampling,
        )
        if cmaes_source_trials:
            kwargs['source_trials'] = cmaes_source_trials
        else:
            kwargs['sigma0'] = cmaes_sigma0
        return CmaEsSampler(**kwargs)

    if sampler_type == "qmc":
        return QMCSampler(seed=seed, scramble=True)

    if sampler_type == "random":
        return RandomSampler(seed=seed)

    raise ValueError(f"Unknown sampler: {sampler_type}")
