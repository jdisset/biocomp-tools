# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Optuna-based hyperparameter optimization for biocomp training.

Progress saved to SQLite after each trial for resume support.
Run multiple processes on different GPUs for parallelism.

Usage:
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.n_trials 500
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.show_best true
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset --define.dashboard true
"""

from __future__ import annotations

import pickle
import time
import asyncio
import threading
from pathlib import Path
from typing import Any, Annotated

import numpy as np
import optuna
from tqdm import tqdm
from pydantic import Field

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode
from biocomp.compute import ComputeConfig
from biocomp.datautils import DataConfig
from biocomp.train import TrainingConfig, start, compile_training_step
from biocomp.library import load_lib
import jax
from biocomptools.toollib.common import config
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.toollib.networkselector import build_data_manager, NetworkSet
from biocomptools.optimtools import DEFAULT_TYPES, make_context_from_types
from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.toollib.networkprediction import NetworkPrediction
from biocomp.metric_utils import DEFAULT_GRIDSTATS_PARAMS
from biocomptools.hyperopt.base import BaseHyperoptProgram, HyperparamSpec

logger = get_logger(__name__)


@dracon_program(
    name='biocomp-hyperopt',
    description='Run hyperparameter optimization for biocomp models.',
    context_types=DEFAULT_TYPES + [HyperparamSpec],
    context={'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve()},
)
class HyperoptProgram(BaseHyperoptProgram):
    """Training hyperopt with validation loss and vmap trials support.

    Extends BaseHyperoptProgram with:
    - Training configuration
    - Validation with NRE/nRMSE metrics
    - vmap-batched trials for parallelism
    - Dataset weight optimization
    """

    # Additional CMA-ES options not in base
    cmaes_x0: str | None = None
    sparse_sampling: bool = False
    sparse_alpha: float = 0.1

    # Training configuration
    training_conf: Annotated[
        DeferredNode[TrainingConfig] | TrainingConfig, Arg(help='Training config')
    ]
    compute_conf: Annotated[DeferredNode[ComputeConfig] | ComputeConfig, Arg(help='Compute config')]
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = Field(default_factory=DataConfig)
    training_set: Annotated[NetworkSet | DeferredNode[NetworkSet], Arg(help='Training set')] = (
        Field(default_factory=NetworkSet)
    )

    # Validation - defaults from metric_utils.DEFAULT_GRIDSTATS_PARAMS
    # Note: uses validation_ prefix to distinguish from potential training-time gridstats
    use_validation_loss: bool = False
    validation_set: NetworkSet | None = None
    n_validation_evals: int = 32000
    validation_objective: str = "geomean_nrmse"
    validation_enable_gridstats: bool = True
    validation_gridstats_res: int = DEFAULT_GRIDSTATS_PARAMS["hypercube_res"]
    validation_gridstats_min: float = DEFAULT_GRIDSTATS_PARAMS["hypercube_min"]
    validation_gridstats_max: float = DEFAULT_GRIDSTATS_PARAMS["hypercube_max"]
    validation_gridstats_k: int = DEFAULT_GRIDSTATS_PARAMS["k"]
    validation_gridstats_radius: float = DEFAULT_GRIDSTATS_PARAMS["radius"]
    validation_gridstats_min_points: int = DEFAULT_GRIDSTATS_PARAMS["min_points"]
    validation_softmax_alpha: float = 5.0
    validation_powermean_p: float = 2.0

    # Dataset weight optimization
    rebuild_dman_per_trial: bool = False

    # Model saving
    n_top_models: int = 10

    # Additional modes not in base
    export_only: bool = False
    n_jobs: int = 1
    vmap_trials: bool = False
    verbose_stats: bool = True
    stats_top_n: int = 30

    # Internal state
    _lib: Any = None
    _training_dman: Any = None
    _cached_step: Any = None
    _vmap_cached_step: Any = None
    _cached_batches: Any = None
    _compile_lock: Any = None
    _validation_predictor: Any = None
    _validation_runner: Any = None
    _network_weight_mapping: list | None = None
    _network_to_dataset: dict | None = None
    _weight_name_to_ndp: dict | None = None
    _best_stats: list[dict] | None = None
    db_session: Any = None
    path_prefix: Path | None = None

    def model_post_init(self, _):
        self._lib = load_lib()
        self.path_prefix = Path(config.paths.root).expanduser().resolve()
        self._compile_lock = threading.Lock()

    def _prepare(self):
        """Prepare data manager for training hyperopt."""
        self._prepare_data_manager()

    def _prepare_data_manager(self, hyperparams: dict | None = None):
        """Build or update DataManager. Fast path for weight-only changes.

        When rebuild_dman_per_trial=True, uses _update_weights_fast to apply new
        hyperparams weights without full DataManager reconstruction. The
        _network_to_dataset mapping enables matching hyperparam names
        (dataweight_X) to network dataset memberships.
        """
        if self._training_dman is not None:
            if self.rebuild_dman_per_trial and self._network_weight_mapping:
                self._update_weights_fast(hyperparams or {})
                return

        from biocomptools.toollib.models import get_biocompdb_sqlite_engine
        from sqlmodel import Session

        engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)

        with Session(engine) as session:
            self.db_session = session

            # resolve training set
            if isinstance(self.training_set, DeferredNode):
                ctx = make_context_from_types(DEFAULT_TYPES)
                ctx.update(hyperparams or {})
                resolved = self.training_set.construct(context=ctx)
            else:
                resolved = self.training_set

            resolved.run_selectors(session)
            logger.info(f"Building DataManager for {len(resolved.content)} networks...")

            self._training_dman = build_data_manager(
                lib=self._lib,
                db_session=session,
                path_prefix=self.path_prefix,
                data_conf=self.data_conf,
                dataset=resolved,
            )

            # cache weight mapping for fast updates
            if self.rebuild_dman_per_trial:
                self._build_weight_mapping(resolved, session)

            logger.info(f"DataManager ready ({len(resolved.content)} pairs)")

    def _build_weight_mapping(self, dataset: NetworkSet, session):
        """Build mapping from network indices to weight parameter names."""
        net_data = dataset.get_networks_and_data(session)
        networks, _ = zip(*net_data, strict=True)
        self._network_weight_mapping = []
        self._weight_name_to_ndp = {}
        self._network_to_dataset = {}

        for ndp in dataset.content:
            self._weight_name_to_ndp[ndp.network_name] = ndp
            # After run_selectors, NDPs retain the dataset_name from their parent filter
            if hasattr(ndp, 'dataset_name') and ndp.dataset_name:
                self._network_to_dataset[ndp.network_name] = ndp.dataset_name

        for n in networks:
            n.build(self._lib)
            network_list = n._network if isinstance(n._network, list) else [n._network]
            for _ in network_list:
                self._network_weight_mapping.append(n.name)

    def _update_weights_fast(self, hyperparams: dict):
        """Update weights directly without re-resolving deferred nodes. KEY OPTIMIZATION."""
        new_weights = []
        for name in self._network_weight_mapping:
            # Use dataset name for hyperparam lookup (matches hyperparam naming convention)
            dataset_name = self._network_to_dataset.get(name)
            hp_name = f"dataweight_{dataset_name}" if dataset_name else None
            if hp_name and hp_name in hyperparams:
                new_weights.append(hyperparams[hp_name])
            else:
                # fallback to default weight from original NDPs
                ndp = self._weight_name_to_ndp.get(name)
                new_weights.append(ndp.weight if ndp else 1.0)
        self._training_dman.set_weights(new_weights)

    def _build_per_trial_weights_fast(self, all_hyperparams: list[dict]) -> Any:
        """Build weight matrix for vmap-trials. Vectorized for speed."""
        import jax.numpy as jnp

        n_trials = len(all_hyperparams)
        # Use dataset names (from _network_to_dataset) instead of network names for hyperparam lookup
        hp_names = []
        for n in self._network_weight_mapping:
            dataset_name = self._network_to_dataset.get(n)
            hp_names.append(f"dataweight_{dataset_name}" if dataset_name else None)

        defaults = np.array(
            [
                self._weight_name_to_ndp.get(n, type('o', (), {'weight': 1.0})).weight
                for n in self._network_weight_mapping
            ]
        )
        weights = np.tile(defaults, (n_trials, 1))
        for j, hp in enumerate(all_hyperparams):
            for i, name in enumerate(hp_names):
                if name and name in hp:
                    weights[j, i] = hp[name]
        networks = self._training_dman.get_networks()
        expanded = np.repeat(weights, [n.nb_outputs for n in networks], axis=1)
        return jnp.array(expanded)

    def _compile_cached_step(self, training_conf: TrainingConfig, compute_conf: ComputeConfig):
        # Thread-safe compilation: only first thread compiles, others wait and reuse
        with self._compile_lock:
            if self._cached_step is not None:
                return
            logger.info("Compiling training step...")
            self._cached_step = compile_training_step(
                dman=self._training_dman,
                training_config=training_conf,
                compute_config=compute_conf,
            )

    def _prepare_validation(self, compute_conf: ComputeConfig):
        # Thread-safe validation preparation
        with self._compile_lock:
            if self._validation_runner is not None or self.validation_set is None:
                return

            from biocomptools.hyperopt.validation import ValidationRunner
            from biocomptools.toollib.models import get_biocompdb_sqlite_engine
            from sqlmodel import Session

            needs_gridstats = self.validation_enable_gridstats and self.validation_objective in (
                "softmax_nrmse",
                "geomean_nrmse",
            )

            logger.info(f"Preparing validation (objective={self.validation_objective})...")
            engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)

            with Session(engine) as session:
                self.validation_set.run_selectors(session)
                val_dman = build_data_manager(
                    lib=self._lib,
                    db_session=session,
                    path_prefix=self.path_prefix,
                    data_conf=self.data_conf,
                    dataset=self.validation_set,
                    jax_sampling=False,
                )

            xs, ys, networks = val_dman.get_per_network_xy_samples(self.n_validation_evals)
            model = BiocompModel(compute_config=compute_conf, rescaler=self.data_conf.rescaler)
            network_model = NetworkModel(model=model, network=networks)

            self._validation_predictor = NetworkPrediction(
                predict_at=xs,
                network_model=network_model,
                ground_truth=ys,
                seed=self.seed or 42,
                disable_variational=True,
                max_evals=self.n_validation_evals,
                already_latent=True,
                n_stats_workers=8,
                device='gpu',
                per_prediction_info=[{'network_name': n.name} for n in networks],
                enable_gridstats=needs_gridstats,
                gridstats_hypercube_res=self.validation_gridstats_res,
                gridstats_hypercube_min=self.validation_gridstats_min,
                gridstats_hypercube_max=self.validation_gridstats_max,
                gridstats_k=self.validation_gridstats_k,
                gridstats_radius=self.validation_gridstats_radius,
                gridstats_min_points=self.validation_gridstats_min_points,
            )

            self._validation_runner = ValidationRunner(
                self._validation_predictor,
                objective=self.validation_objective,
                softmax_alpha=self.validation_softmax_alpha,
                powermean_p=self.validation_powermean_p,
            )
            logger.info(f"Validation ready ({len(networks)} networks)")

    def _run_single_trial(self, trial: optuna.Trial) -> float:
        """Run single training trial."""
        hp = {spec.name: spec.suggest(trial) for spec in self.hyperparams}
        hp['seed'] = self.seed if self.seed else trial.number

        try:
            training_conf = (
                self.training_conf.construct(context=hp)
                if isinstance(self.training_conf, DeferredNode)
                else self.training_conf
            )
            training_conf = training_conf.model_copy()
            training_conf.clear_source_data = False  # keep data for reuse across trials
            compute_conf = (
                self.compute_conf.construct(context=hp)
                if isinstance(self.compute_conf, DeferredNode)
                else self.compute_conf
            )

            if self.rebuild_dman_per_trial:
                self._prepare_data_manager(hp)

            use_cached = self.rebuild_dman_per_trial and all(
                s.name.startswith("dataweight_") for s in self.hyperparams
            )
            if use_cached:
                self._compile_cached_step(training_conf, compute_conf)

            if self.use_validation_loss and self._validation_runner is None:
                self._prepare_validation(compute_conf)

            t0 = time.time()
            params, loss_history, _ = start(
                dman=self._training_dman,
                training_config=training_conf,
                compute_config=compute_conf,
                cached_step=self._cached_step if use_cached else None,
                skip_loss_history=self.use_validation_loss,
            )
            train_time = time.time() - t0

            if self.use_validation_loss:
                t0 = time.time()
                loss, stats = self._validation_runner.compute_loss_with_stats(params)
                val_time = time.time() - t0
                logger.info(
                    f"Trial {trial.number}: val={loss:.6f} (train={train_time:.1f}s, val={val_time:.1f}s)"
                )
                # update best stats tracking
                if loss < self._best_loss:
                    self._best_loss = loss
                    self._best_stats = stats
                    self._best_trial_number = trial.number
                if self.verbose_stats:
                    try:
                        from biocomptools.hyperopt.validation import print_trial_summary

                        names = self._validation_runner.get_network_names()
                        print_trial_summary(
                            trial.number,
                            loss,
                            stats,
                            names,
                            top_n=self.stats_top_n,
                            best_stats=self._best_stats,
                            best_loss=self._best_loss,
                            best_trial_number=self._best_trial_number,
                        )
                    except Exception as plot_err:
                        logger.warning(f"Failed to print stats: {plot_err}")
            elif loss_history:
                losses = np.asarray(loss_history)
                n_final = max(1, len(losses) // 20)
                loss = float(losses[-n_final:].mean(axis=(0, 1, 2)))
                logger.info(f"Trial {trial.number}: train={loss:.6f} ({train_time:.1f}s)")
            else:
                loss = float('inf')

            self._save_model_if_top(trial.number, loss, params, compute_conf, hp)
            return loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed: {e}")
            return float('inf')

    def _run_vmap_trials_batch(
        self,
        trials: list[optuna.Trial],
        training_conf,
        compute_conf,
        batch_idx: int = 0,
        total_batches: int = 1,
    ):
        """Run multiple trials as vmapped pseudo-replicates."""
        import jax

        n_trials = len(trials)
        trial_nums = [t.number for t in trials]
        print(
            f"\n[vmap] Batch {batch_idx + 1}/{total_batches} ({n_trials} trials, optuna ids: {trial_nums[0]}-{trial_nums[-1]})"
        )

        all_hp = [{spec.name: spec.suggest(t) for spec in self.hyperparams} for t in trials]
        for i, hp in enumerate(all_hp):
            hp['seed'] = (self.seed or 0) + trials[i].number

        batch_conf = training_conf.model_copy()
        batch_conf.n_replicates = n_trials
        batch_conf.clear_source_data = False  # keep data for reuse across trials

        # Thread-safe vmap compilation
        with self._compile_lock:
            if self._vmap_cached_step is None:
                print("[vmap] Compiling...")
                t0 = time.time()
                self._vmap_cached_step = compile_training_step(
                    dman=self._training_dman,
                    training_config=batch_conf,
                    compute_config=compute_conf,
                )
                print(f"[vmap] Compiled ({time.time() - t0:.1f}s)")

        # Use unique seeds per trial (based on trial numbers) to avoid position-based bias across batches
        trial_seeds = np.array([hp['seed'] for hp in all_hp], dtype=np.uint32)
        init_keys = jax.vmap(jax.random.PRNGKey)(trial_seeds)
        params = jax.vmap(self._vmap_cached_step.stack.init)(init_keys)

        weights = self._build_per_trial_weights_fast(all_hp)
        params.at("global/per_output_weights", weights, tags=["non_grad", "local"], overwrite=True)

        # Cache batches on CPU for reuse across trial batches (major speedup)
        # Keeping on CPU avoids GPU memory pressure during validation
        n_batches_needed = batch_conf.n_batches
        batch_size = batch_conf.batch_size
        if self._cached_batches is None:
            print("[vmap] Generating batches (caching on CPU for reuse)...")
            t0 = time.time()
            # Generate enough for max_replicates (first batch size) - subsequent can slice
            flat_n = n_trials * n_batches_needed
            batch_key = jax.random.PRNGKey(self.seed or 42)
            xflat, yflat = self._training_dman.get_batches(flat_n, batch_size, batch_key)
            # Move to CPU for caching (saves GPU memory)
            cpu = jax.devices("cpu")[0]
            xflat_cpu = jax.device_put(xflat, cpu)
            yflat_cpu = jax.device_put(yflat, cpu)
            del xflat, yflat  # free GPU copies
            self._cached_batches = (xflat_cpu, yflat_cpu, n_trials)
            print(f"[vmap] Batch generation done ({time.time() - t0:.1f}s)")

        xflat, yflat, cached_n_reps = self._cached_batches
        if n_trials <= cached_n_reps:
            # Slice and reshape cached batches, then move to GPU for training
            flat_n = n_trials * n_batches_needed
            gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
            xy_batches = (
                jax.device_put(xflat[:flat_n].reshape(n_trials, n_batches_needed, *xflat.shape[1:]), gpu),
                jax.device_put(yflat[:flat_n].reshape(n_trials, n_batches_needed, *yflat.shape[1:]), gpu),
            )
        else:
            # Rare case: more trials than cached - regenerate
            xy_batches = None

        print(f"[vmap] Training {n_trials} trials...")
        t0 = time.time()

        # Create hyperopt training logger if rich is available
        hyperopt_loggers = []
        try:
            from biocomptools.hyperopt.training_logger import HyperoptTrainingLogger

            hyperopt_logger = HyperoptTrainingLogger(n_replicates=n_trials)
            hyperopt_loggers = hyperopt_logger.get_callbacks(None)
        except ImportError:
            pass

        params, loss_history, _ = start(
            dman=self._training_dman,
            training_config=batch_conf,
            compute_config=compute_conf,
            init_params=params,
            skip_weight_init=True,
            cached_step=self._vmap_cached_step,
            loggers=hyperopt_loggers,
            skip_loss_history=self.use_validation_loss,
            xy_batches=xy_batches,
        )
        print(f"[vmap] Training done ({time.time() - t0:.1f}s)")

        # Free GPU copy of batches before validation (CPU cache remains for next batch)
        del xy_batches

        best_stats = None
        if self.use_validation_loss:
            print("[vmap] Validation...")
            t0 = time.time()
            if self.verbose_stats:
                losses, best_idx, best_stats = self._validation_runner.compute_losses_batched(
                    params, verbose=True, return_best_stats=True
                )
            else:
                losses = self._validation_runner.compute_losses_batched(params, verbose=True)
            print(f"[vmap] Validation done ({time.time() - t0:.1f}s)")
        else:
            loss_arr = np.asarray(loss_history)
            n_final = max(1, loss_arr.shape[0] // 20)
            losses = [float(loss_arr[-n_final:, i, :].mean()) for i in range(n_trials)]

        return losses, params, all_hp, compute_conf, best_stats

    async def _run_vmap_optimization(self):
        """Run optimization with vmap-trials batching."""
        existing_study = self._load_existing()
        previously_completed = self._get_completed_trials(existing_study)
        n_previously_complete = len(previously_completed)

        sampler = self._create_sampler(
            n_previously_complete, previously_completed if self.cmaes_warm_start else None
        )
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self._storage_path,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )

        training_conf = (
            self.training_conf.construct(context={})
            if isinstance(self.training_conf, DeferredNode)
            else self.training_conf
        )
        compute_conf = (
            self.compute_conf.construct(context={})
            if isinstance(self.compute_conf, DeferredNode)
            else self.compute_conf
        )

        if self.use_validation_loss and self._validation_runner is None:
            self._prepare_validation(compute_conf)

        batch_size = self.n_jobs
        total_batches = (self.n_trials + batch_size - 1) // batch_size
        db_path = Path(self.output_dir).resolve() / (self.study_name + '.db')

        # show resume info
        if n_previously_complete > 0:
            print(f"\n{'═' * 70}")
            print(f"[vmap] Resuming study '{self.study_name}'")
            print(f"[vmap] Database: {db_path}")
            print(f"[vmap] Previously completed: {n_previously_complete} trials")
            if existing_study and existing_study.best_trial:
                print(
                    f"[vmap] Previous best: {existing_study.best_value:.6f} (trial #{existing_study.best_trial.number})"
                )
            print(
                f"[vmap] New trials to run: {self.n_trials} (in {total_batches} batches of {batch_size})"
            )
            print(f"{'═' * 70}\n")
        else:
            print(f"\n{'═' * 70}")
            print(f"[vmap] Starting new study '{self.study_name}'")
            print(f"[vmap] Database: {db_path}")
            print(
                f"[vmap] Trials to run: {self.n_trials} (in {total_batches} batches of {batch_size})"
            )
            print(f"{'═' * 70}\n")

        remaining = self.n_trials
        batch_idx = 0
        pbar = tqdm(total=self.n_trials, desc="Hyperopt", unit="trial")

        try:
            while remaining > 0:
                trials = [study.ask() for _ in range(min(batch_size, remaining))]
                try:
                    losses, params, all_hp, comp_conf, best_stats = self._run_vmap_trials_batch(
                        trials,
                        training_conf,
                        compute_conf,
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                    )
                    best_idx = int(np.argmin(losses))
                    for i, (trial, loss, hp) in enumerate(zip(trials, losses, all_hp, strict=True)):
                        study.tell(trial, loss)
                        idx = i
                        single_params = jax.tree.map(lambda x, j=idx: x[j], params)
                        self._save_model_if_top(trial.number, loss, single_params, comp_conf, hp)

                    # update best stats tracking and show summary
                    batch_best_loss = losses[best_idx]
                    batch_best_trial = trials[best_idx].number
                    if batch_best_loss < self._best_loss:
                        self._best_loss = batch_best_loss
                        self._best_stats = best_stats
                        self._best_trial_number = batch_best_trial
                    if self.verbose_stats and self.use_validation_loss and best_stats is not None:
                        try:
                            from biocomptools.hyperopt.validation import print_trial_summary

                            names = self._validation_runner.get_network_names()
                            print_trial_summary(
                                batch_best_trial,
                                batch_best_loss,
                                best_stats,
                                names,
                                top_n=self.stats_top_n,
                                best_stats=self._best_stats,
                                best_loss=self._best_loss,
                                best_trial_number=self._best_trial_number,
                            )
                        except Exception as plot_err:
                            logger.warning(f"Failed to print stats: {plot_err}")
                except Exception as e:
                    logger.exception(f"Batch failed: {e}")
                    for trial in trials:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)

                pbar.update(len(trials))
                remaining -= len(trials)
                batch_idx += 1

                # show progress summary
                all_completed = [
                    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                ]
                new_completed = len(all_completed) - n_previously_complete
                pbar.set_postfix(
                    best=f"{study.best_value:.4f}", new=new_completed, total=len(all_completed)
                )

        except KeyboardInterrupt:
            print("\n[vmap] Interrupted by user")
        finally:
            pbar.close()

        # final summary
        all_completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        new_completed = len(all_completed) - n_previously_complete
        print(f"\n{'═' * 70}")
        print(f"[vmap] Study complete: {new_completed} new trials ({len(all_completed)} total)")
        print(f"[vmap] Best: {study.best_value:.6f} (trial #{study.best_trial.number})")
        print(f"{'═' * 70}\n")

        self._save_results(study)

    def _save_model_if_top(self, trial_num: int, loss: float, params, compute_conf, hp: dict):
        if self.n_top_models <= 0 or loss == float('inf'):
            return

        from biocomp.jaxutils import tree_to_np
        from biocomptools.run_training import get_shared_params

        models_dir = Path(self.output_dir) / self.study_name / "top_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            [
                (float(f.stem.split("_loss_")[1]), f)
                for f in models_dir.glob("*.pickle")
                if "_loss_" in f.stem
            ],
            key=lambda x: x[0],
        )

        if len(existing) >= self.n_top_models and loss >= existing[-1][0]:
            return

        try:
            single = jax.tree.map(lambda x: x[0], params)
        except (IndexError, TypeError):
            single = params

        shared = get_shared_params(pickle.loads(pickle.dumps(single)))
        if shared is None:
            return

        model = BiocompModel(
            compute_config=compute_conf,
            rescaler=self.data_conf.rescaler,
            shared_params=tree_to_np(shared),
            metadata={'trial': trial_num, 'loss': loss, 'study': self.study_name, 'hp': hp},
        )
        model.save(models_dir / f"trial_{trial_num:04d}_loss_{loss:.6f}.pickle")

        existing.append((loss, models_dir / f"trial_{trial_num:04d}_loss_{loss:.6f}.pickle"))
        existing.sort(key=lambda x: x[0])
        while len(existing) > self.n_top_models:
            _, worst = existing.pop()
            worst.unlink()

    async def run(self):
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
            return

        if self.export_only:
            study = self._load_existing()
            if study:
                self._save_results(study)
            return

        self._prepare_data_manager()

        if self.vmap_trials:
            await self._run_vmap_optimization()
            return

        # standard sequential optimization
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

        study.optimize(
            self._run_single_trial, n_trials=self.n_trials, callbacks=[callback], n_jobs=self.n_jobs
        )
        pbar.close()
        self._save_results(study)


async def _main_async():
    setup_logging()
    await HyperoptProgram.cli()


def main():
    asyncio.run(_main_async())


if __name__ == '__main__':
    main()
