"""Shared base classes and utilities for hyperparameter optimization.

This module provides:
- HyperparamSpec: Hyperparameter specification for Optuna
- BaseHyperoptProgram: Abstract base with common hyperopt functionality
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Annotated, Literal

import optuna
from optuna.samplers import BaseSampler
from pydantic import BaseModel, Field
from tqdm import tqdm

from dracon.commandline import Arg

from biocomptools.logging_config import get_logger
from .samplers import create_sampler


logger = get_logger(__name__)


class HyperparamSpec(BaseModel):
    """Hyperparameter specification for Optuna optimization."""

    name: str
    type: Literal['float', 'log_float', 'int', 'categorical']
    low: float | int | None = None
    high: float | int | None = None
    choices: list[str] | None = None
    step: float | None = None
    target_path: str | None = None

    def suggest(self, trial: optuna.Trial) -> float | int | str:
        if self.type == 'float':
            assert self.low is not None and self.high is not None
            return trial.suggest_float(self.name, float(self.low), float(self.high), step=self.step)
        if self.type == 'log_float':
            assert self.low is not None and self.high is not None
            return trial.suggest_float(self.name, float(self.low), float(self.high), log=True)
        if self.type == 'int':
            assert self.low is not None and self.high is not None
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        if self.type == 'categorical':
            assert self.choices is not None
            return trial.suggest_categorical(self.name, self.choices)
        raise ValueError(f"Unknown hyperparameter type: {self.type}")


class BaseHyperoptProgram(BaseModel, ABC):
    """Abstract base class for hyperparameter optimization programs.

    Provides shared functionality:
    - Study management (load, save, resume)
    - Sampler creation with CMA-ES support
    - Dashboard and show-best modes
    - Progress tracking

    Subclasses must implement:
    - _prepare(): Setup domain-specific resources
    - _run_single_trial(trial): Execute one trial and return loss
    """

    study_name: Annotated[str, Arg(help='Study name for persistence')] = "hyperopt"
    output_dir: Annotated[str, Arg(help='Output directory')] = "./hyperopt_output"
    n_trials: Annotated[int, Arg(help='Number of trials')] = 100
    seed: Annotated[int | None, Arg(help='Random seed')] = None

    sampler: Annotated[str, Arg(help='Sampler: tpe, cmaes, hybrid, qmc, random')] = "tpe"
    n_startup_trials: int = 10
    pruning: Annotated[bool, Arg(help='Enable pruning')] = False

    cmaes_restart_strategy: str | None = "bipop"
    cmaes_popsize: int | None = 32
    cmaes_sigma0: float = 0.5
    cmaes_warm_start: bool = True
    cmaes_with_margin: bool = True
    cmaes_warn_independent_sampling: bool = False
    hybrid_switch_trial: int = 50

    hyperparams: list[HyperparamSpec] = Field(default_factory=list)

    show_best: Annotated[bool, Arg(help='Show best trial and exit')] = False
    dashboard: Annotated[bool, Arg(help='Launch Optuna dashboard')] = False
    dashboard_port: int = 8080
    verbose: bool = True

    _best_loss: float = float('inf')
    _best_trial_number: int | None = None

    model_config = {'arbitrary_types_allowed': True}

    @property
    def _storage_path(self) -> str:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{Path(self.output_dir) / (self.study_name + '.db')}"

    def _load_existing(self) -> optuna.Study | None:
        try:
            return optuna.load_study(study_name=self.study_name, storage=self._storage_path)
        except KeyError:
            return None

    def _get_completed_trials(self, study: optuna.Study | None) -> list[optuna.trial.FrozenTrial]:
        if not study:
            return []
        return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def _save_results(self, study: optuna.Study):
        out = Path(self.output_dir) / self.study_name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "best_hyperparams.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)
        logger.info(f"Results saved to {out}")

    def _create_sampler(
        self, n_complete: int = 0, source_trials: list[optuna.trial.FrozenTrial] | None = None
    ) -> BaseSampler:
        sampler_type = self.sampler
        if sampler_type == "hybrid":
            sampler_type = "tpe" if n_complete < self.hybrid_switch_trial else "cmaes"

        return create_sampler(
            sampler_type=sampler_type,
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            cmaes_restart_strategy=self.cmaes_restart_strategy,
            cmaes_with_margin=self.cmaes_with_margin,
            cmaes_popsize=self.cmaes_popsize,
            cmaes_sigma0=self.cmaes_sigma0,
            cmaes_source_trials=source_trials,
            cmaes_warn_independent_sampling=self.cmaes_warn_independent_sampling,
        )

    def _suggest_hyperparams(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters from trial.

        Returns dict of {name: value} for all hyperparams.
        """
        hp: dict[str, Any] = {spec.name: spec.suggest(trial) for spec in self.hyperparams}
        hp['seed'] = self.seed if self.seed else trial.number
        return hp

    @abstractmethod
    def _prepare(self):
        """Prepare domain-specific resources before optimization.

        Called once before the optimization loop starts.
        """
        ...

    @abstractmethod
    def _run_single_trial(self, trial: optuna.Trial) -> float:
        """Run a single optimization trial.

        Args:
            trial: Optuna trial object for suggesting hyperparameters

        Returns:
            Loss value (lower is better)
        """
        ...

    async def _run_optimization_loop(self):
        """Run the main optimization loop."""
        completed = self._get_completed_trials(self._load_existing())
        sampler = self._create_sampler(len(completed), completed if self.cmaes_warm_start else None)
        pruner = (
            optuna.pruners.MedianPruner(n_warmup_steps=5)
            if self.pruning
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self._storage_path,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        pbar = tqdm(total=self.n_trials, desc="Hyperopt")

        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pbar.update()
                pbar.set_postfix(best=f"{study.best_value:.4f}")

        study.optimize(
            self._run_single_trial,
            n_trials=self.n_trials,
            callbacks=[callback],
        )
        pbar.close()

        self._save_results(study)

        if self.verbose:
            logger.info(f"\nBest trial: #{study.best_trial.number}")
            logger.info(f"Best loss: {study.best_value:.6f}")
            logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    async def run(self):
        """Main entry point for hyperopt program."""
        if self.dashboard:
            import optuna_dashboard

            logger.info(f"Launching dashboard at http://localhost:{self.dashboard_port}")
            optuna_dashboard.run_server(self._storage_path, port=self.dashboard_port)
            return

        if self.show_best:
            study = self._load_existing()
            if study:
                completed = self._get_completed_trials(study)
                logger.info(f"Study: {self.study_name} ({len(completed)} complete)")
                if completed:
                    logger.info(f"Best: #{study.best_trial.number} loss={study.best_value:.6f}")
                    logger.info(f"Best params: {study.best_params}")
            return

        self._prepare()
        await self._run_optimization_loop()


SCHEDULE_SUFFIXES = ('_phase1_value', '_phase2_end_value', '_phase3_end_value')
PHASE_FRAC_SUFFIXES = ('_phase1_frac', '_phase2_frac')


def expand_schedule_hyperparams(
    hp: dict[str, Any],
    phase1_frac: float = 0.4,
    phase2_frac: float = 0.75,
) -> dict[str, Any]:
    """Expand schedule hyperparams to full 3-phase specification.

    Handles three cases for each schedule:
    1. Constant: Just the base name (e.g., 'w_sinkhorn') → all phases get same value
    2. Linear: phase1_value + phase3_end_value → computes phase2_end_value via interpolation
    3. Full 3-phase: All three values explicitly provided → used as-is

    Args:
        hp: Raw hyperparams dict from trial.suggest()
        phase1_frac: Global phase1 fraction (from hyperparams or default)
        phase2_frac: Global phase2 fraction (from hyperparams or default)

    Returns:
        Expanded dict with all 3-phase values filled in.

    Raises:
        ValueError: If schedule spec is ambiguous or invalid.
    """
    result = dict(hp)
    phase1_frac = hp.get('phase1_frac', phase1_frac)
    phase2_frac = hp.get('phase2_frac', phase2_frac)

    assert 0 < phase1_frac < phase2_frac < 1, (
        f"Invalid phase fractions: phase1_frac={phase1_frac}, phase2_frac={phase2_frac}. "
        f"Must satisfy 0 < phase1_frac < phase2_frac < 1."
    )

    schedule_names: set[str] = set()
    for name in hp.keys():
        if name in ('phase1_frac', 'phase2_frac', 'seed'):
            continue
        for suffix in SCHEDULE_SUFFIXES + PHASE_FRAC_SUFFIXES:
            if name.endswith(suffix):
                schedule_names.add(name[: -len(suffix)])
                break
        else:
            schedule_names.add(name)

    for sched_name in schedule_names:
        p1_key = f"{sched_name}_phase1_value"
        p2_key = f"{sched_name}_phase2_end_value"
        p3_key = f"{sched_name}_phase3_end_value"

        has_p1 = p1_key in hp
        has_p2 = p2_key in hp
        has_p3 = p3_key in hp
        has_base = sched_name in hp and not any(sched_name.endswith(s) for s in SCHEDULE_SUFFIXES)

        if has_base and not (has_p1 or has_p2 or has_p3):
            # Case 1: Constant schedule
            val = hp[sched_name]
            result[p1_key] = val
            result[p2_key] = val
            result[p3_key] = val
            logger.debug(f"Schedule '{sched_name}': constant={val}")

        elif has_p1 and has_p3 and not has_p2:
            # Case 2: Linear schedule - compute phase2 via interpolation
            p1_val = hp[p1_key]
            p3_val = hp[p3_key]
            # Linear interpolation: at phase2_frac, value is between p1 and p3
            # Assuming phase1 holds p1_val, then linear ramp to p3_val
            # phase2 is at fraction phase2_frac, phase3 ends at 1.0
            # Interpolate: p2 = p1 + (p3 - p1) * (phase2_frac - phase1_frac) / (1 - phase1_frac)
            if phase2_frac > phase1_frac:
                t = (phase2_frac - phase1_frac) / (1.0 - phase1_frac)
                p2_val = p1_val + (p3_val - p1_val) * t
            else:
                p2_val = p1_val
            result[p2_key] = p2_val
            logger.debug(
                f"Schedule '{sched_name}': linear p1={p1_val:.4f} → p2={p2_val:.4f} → p3={p3_val:.4f}"
            )

        elif has_p1 and has_p2 and has_p3:
            # Case 3: Full 3-phase - use as-is
            logger.debug(
                f"Schedule '{sched_name}': 3-phase p1={hp[p1_key]:.4f}, "
                f"p2={hp[p2_key]:.4f}, p3={hp[p3_key]:.4f}"
            )

        elif has_p1 or has_p2 or has_p3:
            # Partial specification - this is an error
            present = [k for k in [p1_key, p2_key, p3_key] if k in hp]
            missing = [k for k in [p1_key, p2_key, p3_key] if k not in hp]
            raise ValueError(
                f"Schedule '{sched_name}' has incomplete spec. "
                f"Present: {present}, Missing: {missing}. "
                f"Provide either: (1) just '{sched_name}' for constant, "
                f"(2) '{p1_key}' + '{p3_key}' for linear, or "
                f"(3) all three for full 3-phase."
            )

    return result


def get_schedule_param_names(hyperparams: list[HyperparamSpec]) -> dict[str, list[str]]:
    """Analyze hyperparams to identify schedule structure.

    Returns:
        Dict mapping schedule names to list of suffixes provided.
        E.g., {'w_sinkhorn': ['_phase1_value', '_phase3_end_value'], 'lambda_l0': ['_phase1_value', '_phase2_end_value', '_phase3_end_value']}
    """
    schedules: dict[str, list[str]] = {}
    for spec in hyperparams:
        name = spec.name
        if name in ('phase1_frac', 'phase2_frac', 'seed'):
            continue
        for suffix in SCHEDULE_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                schedules.setdefault(base, []).append(suffix)
                break
        else:
            schedules.setdefault(name, []).append('')  # empty string = base name (constant)
    return schedules


def verify_hyperparam_propagation(
    params_at_start: dict[str, Any],
    params_at_use: dict[str, Any],
    expected_hyperparams: dict[str, Any],
    context: str = "",
) -> list[str]:
    """Verify that hyperparameters were correctly propagated to execution.

    Args:
        params_at_start: Parameters as suggested by trial
        params_at_use: Parameters as actually used (from keep_in_history)
        expected_hyperparams: Expected hyperparam values
        context: Description for error messages

    Returns:
        List of discrepancies found (empty if all match)
    """
    discrepancies = []
    for name, expected in expected_hyperparams.items():
        if name in ('seed',):
            continue
        actual = params_at_use.get(name)
        if actual is None:
            discrepancies.append(f"{context}: {name} not found in params")
        elif abs(float(actual) - float(expected)) > 1e-6:
            discrepancies.append(f"{context}: {name} expected={expected}, got={actual}")
    return discrepancies
