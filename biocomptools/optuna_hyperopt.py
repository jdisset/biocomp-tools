"""
Optuna-based hyperparameter optimization for biocompiler training.

This tool integrates with the Dracon configuration system, allowing you to define
hyperparameter search spaces and training settings in YAML config files.

Progress is automatically saved to SQLite after each trial, so you can:
- Stop the script at any time (Ctrl+C) and resume later
- Run multiple sessions that accumulate results
- Query the database directly for analysis

Usage:
    # Run with a config file
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset

    # Override parameters from CLI
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.n_trials 500

    # Check status of existing study
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.show_best true

    # Launch Optuna dashboard web interface
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.dashboard true

    # Launch dashboard on custom port
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.dashboard true --define.dashboard_port 9000

    # Resume existing study (just run the same command again)
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset
"""

from __future__ import annotations

import json
import pickle
import time
import asyncio
import sys
from pathlib import Path
from typing import Any, Annotated

import numpy as np
import optuna
from optuna.samplers import TPESampler
from pydantic import Field, ConfigDict, BaseModel

from dracon.commandline import Arg, make_program
from dracon.deferred import DeferredNode

from biocomp.compute import ComputeConfig
from biocomp.datautils import DataConfig
from biocomp.train import TrainingConfig, start
from biocomp.library import load_lib
from biocomptools.toollib.common import config
from biocomptools.toollib.networkselector import build_data_manager, NetworkSet
from biocomptools.optimtools import make_context_from_types, DEFAULT_TYPES
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class OptunaPruningLogger(Logger):
    """Logger that reports losses to Optuna for pruning bad trials early."""

    trial: optuna.Trial | None = None
    async_ok: bool = False  # must run synchronously to check pruning
    _step_losses: list[float] = []

    def model_post_init(self, __context):
        self._step_losses = []

    def get_callbacks(self, training_program):
        def report_loss(step, training_config, step_history, stack):
            if self.trial is None:
                return

            loss = step_history.get('loss')
            if loss is None:
                return

            mean_loss = float(np.asarray(loss).mean())
            self._step_losses.append(mean_loss)

            self.trial.report(mean_loss, step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return [(1, report_loss)]

    def get_final_loss(self) -> float:
        """Get smoothed final loss (last 5% average)."""
        if not self._step_losses:
            return float('inf')
        n_final = max(1, len(self._step_losses) // 20)
        return float(np.mean(self._step_losses[-n_final:]))


class HyperparamSpec(BaseModel):
    """Specification for a hyperparameter to optimize."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str  # "float", "int", "categorical", "log_float"
    low: float | None = None
    high: float | None = None
    choices: list | None = None

    def suggest(self, trial: optuna.Trial) -> Any:
        if self.type == "float":
            return trial.suggest_float(self.name, self.low, self.high)
        elif self.type == "log_float":
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        elif self.type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        elif self.type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        raise ValueError(f"Unknown hyperparameter type: {self.type}")


class HyperoptProgram(BaseModel):
    """Optuna hyperparameter optimization program for biocompiler."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # study configuration
    study_name: Annotated[str, Arg(help='Study name for persistence')] = "biocomp_hyperopt"
    n_trials: Annotated[int, Arg(help='Number of trials to run')] = 100
    n_startup_trials: Annotated[int, Arg(help='Random trials before TPE')] = 20
    pruning: Annotated[bool, Arg(help='Enable early stopping of bad trials')] = True
    seed: Annotated[int | None, Arg(help='Random seed')] = None

    # output configuration
    output_dir: Annotated[str, Arg(help='Directory to save results')] = Field(
        default_factory=lambda: str(Path(config.paths.root) / "hyperopt_results")
    )

    # training configuration (deferred - reconstructed per trial with hyperparameters)
    training_conf: Annotated[DeferredNode[TrainingConfig], Arg(help='Base training config')]
    compute_conf: Annotated[DeferredNode[ComputeConfig], Arg(help='Base compute config')]
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = Field(default_factory=DataConfig)
    training_set: Annotated[NetworkSet, Arg(help='Networks in training set')] = Field(
        default_factory=NetworkSet
    )

    # hyperparameters to optimize
    hyperparams: Annotated[list[HyperparamSpec], Arg(help='Hyperparameters to optimize')] = Field(
        default_factory=list
    )

    # info-only modes
    show_best: Annotated[bool, Arg(help='Show best results and exit')] = False
    export_only: Annotated[bool, Arg(help='Export results and exit')] = False
    dashboard: Annotated[bool, Arg(help='Launch Optuna dashboard web interface and exit')] = False
    dashboard_port: Annotated[int, Arg(help='Port for Optuna dashboard')] = 8080

    # internal state
    _lib: Any = None
    _training_dman: Any = None

    def model_post_init(self, __context):
        self._lib = load_lib()
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())

    @property
    def db_session(self):
        from biocomptools.toollib.models import get_biocompdb_sqlite_engine
        from sqlmodel import Session

        engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)
        return Session(engine)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    def _prepare_data_manager(self):
        """Build data manager (done once, reused across trials)."""
        if self._training_dman is not None:
            return

        print("Preparing data manager (this takes a moment)...")
        with self.db_session as session:
            self.training_set.run_selectors(session)
            self._training_dman = build_data_manager(
                lib=self._lib,
                db_session=session,
                path_prefix=self.path_prefix,
                data_conf=self.data_conf,
                dataset=self.training_set,
            )
        print("Data manager ready.")

    def _get_storage_path(self) -> str:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{output_path / (self.study_name + '.db')}"

    def _load_existing_study(self) -> optuna.Study | None:
        try:
            return optuna.load_study(
                study_name=self.study_name,
                storage=self._get_storage_path(),
            )
        except KeyError:
            return None

    def _show_study_status(self) -> bool:
        study = self._load_existing_study()
        if study is None:
            print(f"No existing study found with name '{self.study_name}'")
            return False

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        print(f"\n{'=' * 60}")
        print(f"Study: {self.study_name}")
        print(f"Storage: {self._get_storage_path()}")
        print(f"{'=' * 60}")
        print(f"Total trials: {len(study.trials)}")
        print(f"  - Completed: {n_complete}")
        print(f"  - Pruned: {n_pruned}")
        print(f"  - Failed: {n_failed}")

        if n_complete > 0:
            print(f"\nBest trial: #{study.best_trial.number}")
            print(f"Best loss: {study.best_value:.6f}")
            print("\nBest hyperparameters:")
            for k, v in study.best_params.items():
                print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'=' * 60}\n")
        return True

    def _set_nested_value(self, obj: Any, path: list[str], value: Any):
        """Set a nested attribute value using a path like ['node_functions', 'translation', 'kwargs', 'rate_dim']."""
        current = obj
        for key in path[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict):
                current = current[key]
            else:
                raise ValueError(f"Cannot navigate path {'.'.join(path)} in {type(obj).__name__}")

        # set the final value
        final_key = path[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
        else:
            raise ValueError(f"Cannot set {final_key} in {type(current).__name__}")

    def _launch_dashboard(self):
        """Launch Optuna dashboard web interface."""
        storage = self._get_storage_path()

        try:
            import optuna_dashboard

            print(f"\n{'=' * 60}")
            print("Launching Optuna Dashboard")
            print(f"Storage: {storage}")
            print(f"URL: http://localhost:{self.dashboard_port}")
            print("Press Ctrl+C to stop")
            print(f"{'=' * 60}\n")

            optuna_dashboard.run_server(storage, port=self.dashboard_port)

        except ImportError:
            print("ERROR: optuna-dashboard not installed")
            print("Install with: pip install optuna-dashboard")
            print("\nAlternatively, run manually:")
            print(f"  optuna-dashboard {storage}")
            sys.exit(1)

    def _save_results(self, study: optuna.Study):
        results_dir = Path(self.output_dir) / self.study_name
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "best_hyperparams.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        with open(results_dir / "best_hyperparams.pkl", "wb") as f:
            pickle.dump(study.best_params, f)

        df = study.trials_dataframe()
        df.to_csv(results_dir / "trials.csv", index=False)

        summary = {
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_complete": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")
        print(f"Best loss: {study.best_value:.6f}")

    def _run_single_trial(self, trial: optuna.Trial) -> float:
        """Run a single training trial with sampled hyperparameters."""
        # sample hyperparameters
        hyperparams = {spec.name: spec.suggest(trial) for spec in self.hyperparams}

        # apply ordering constraints
        if hyperparams.get("initial_learning_rate", 0) > hyperparams.get("peak_learning_rate", 1):
            hyperparams["initial_learning_rate"] = hyperparams["peak_learning_rate"] * 0.1
        if hyperparams.get("final_learning_rate", 0) > hyperparams.get("peak_learning_rate", 1):
            hyperparams["final_learning_rate"] = hyperparams["peak_learning_rate"] * 0.01

        # seed per trial
        hyperparams['seed'] = self.seed if self.seed is not None else trial.number

        try:
            # construct configs with hyperparameters as context variables
            training_conf = self.training_conf.construct(context=hyperparams)
            compute_conf = self.compute_conf.construct(context=hyperparams)

            # handle path-based hyperparameters (e.g., "compute_conf.node_functions.translation.kwargs.rate_dim")
            for spec in self.hyperparams:
                if '.' in spec.name and spec.name.startswith('compute_conf.'):
                    path_parts = spec.name.split('.')[1:]  # skip 'compute_conf' prefix
                    value = hyperparams[spec.name]
                    self._set_nested_value(compute_conf, path_parts, value)
                elif '.' in spec.name and spec.name.startswith('training_conf.'):
                    path_parts = spec.name.split('.')[1:]  # skip 'training_conf' prefix
                    value = hyperparams[spec.name]
                    self._set_nested_value(training_conf, path_parts, value)

            # create pruning logger
            pruning_logger = OptunaPruningLogger(trial=trial) if self.pruning else None
            logger_callbacks = []
            if pruning_logger:
                pruning_logger.initialize(None)
                logger_callbacks = pruning_logger.get_callbacks(None)

            t0 = time.time()
            params, loss_history, _ = start(
                dman=self._training_dman,
                training_config=training_conf,
                compute_config=compute_conf,
                loggers=logger_callbacks,
            )
            elapsed = time.time() - t0

            # get final loss
            if pruning_logger:
                loss = pruning_logger.get_final_loss()
            elif loss_history:
                losses = np.asarray(loss_history)
                mean_losses = losses.mean(axis=(1, 2))
                n_final = max(1, len(mean_losses) // 20)
                loss = float(mean_losses[-n_final:].mean())
            else:
                loss = float('inf')

            print(f"Trial {trial.number}: loss={loss:.6f} ({elapsed:.1f}s)")
            return loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            import traceback

            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            return float('inf')

    async def run(self):
        """Main entry point."""
        # handle info-only modes
        if self.dashboard:
            self._launch_dashboard()
            return

        if self.show_best:
            self._show_study_status()
            return

        if self.export_only:
            study = self._load_existing_study()
            if study is None:
                print(f"No existing study found with name '{self.study_name}'")
                return
            self._save_results(study)
            self._show_study_status()
            return

        # prepare data manager
        self._prepare_data_manager()

        # check for existing study
        existing = self._load_existing_study()
        if existing is not None:
            n_complete = len(
                [t for t in existing.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            print(
                f"Resuming existing study with {len(existing.trials)} trials ({n_complete} complete)"
            )
            if n_complete > 0:
                print(f"Current best loss: {existing.best_value:.6f}")

        # create study
        sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed,
        )
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=5)
            if self.pruning
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self._get_storage_path(),
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        print(f"\nStarting optimization: {self.n_trials} new trials")
        print(f"Study: {self.study_name}")
        print(f"Hyperparameters: {[h.name for h in self.hyperparams]}")
        print("(Progress saved after each trial - safe to interrupt with Ctrl+C)\n")

        try:
            study.optimize(self._run_single_trial, n_trials=self.n_trials, show_progress_bar=True)
        except KeyboardInterrupt:
            print("\n\nOptimization interrupted. Progress saved.")

        self._save_results(study)


async def main_async():
    cliprog = make_program(
        HyperoptProgram,
        name='biocomp-hyperopt',
        description='Optuna hyperparameter optimization for biocompiler.',
    )

    context = {
        **make_context_from_types(DEFAULT_TYPES),
        'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
        'HyperparamSpec': HyperparamSpec,
    }

    program, _ = cliprog.parse_args(
        sys.argv[1:],
        deferred_paths=['/training_conf', '/compute_conf'],
        context=context,
        capture_globals=False,
        enable_shorthand_vars=False,
    )
    assert isinstance(program, HyperoptProgram)
    await program.run()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
