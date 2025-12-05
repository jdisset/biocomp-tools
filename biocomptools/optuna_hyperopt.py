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

    # Dataset weight optimization with validation-based objective
    biocomp-hyperopt +biocomp-jobs/hyperopt/dataset_weights --use_validation_loss true

    # Disable model saving (default keeps top 10)
    biocomp-hyperopt +biocomp-jobs/hyperopt/fullset ++n_top_models 0

Model Saving:
    By default, the top 10 models (by loss) are saved as BiocompModel pickle files
    in {output_dir}/{study_name}/top_models/. This allows you to use the best
    hyperparameter configurations without re-training. Worse models are automatically
    evicted as better ones are found. Set n_top_models=0 to disable.

Parallel Execution:
    # Run multiple processes manually (each in separate terminal/tmux pane)
    # They coordinate via shared SQLite storage automatically
    biocomp-hyperopt +hyperopt/fullset
    biocomp-hyperopt +hyperopt/fullset  # in another terminal

    # For multi-GPU setups
    CUDA_VISIBLE_DEVICES=0 biocomp-hyperopt +hyperopt/fullset
    CUDA_VISIBLE_DEVICES=1 biocomp-hyperopt +hyperopt/fullset

    Note: Set streaming_batches=true in your training config to reduce GPU memory
    and enable multiple processes on the same GPU.

CMA-ES Configuration:
    When using sampler cmaes, the following defaults are applied to avoid local minima:
    - cmaes_restart_strategy: bipop (bidirectional population restarts)
    - cmaes_popsize: 32 (larger than default 4+3*log(n) for better exploration)
    - cmaes_warm_start: true (resume from existing trials instead of random init)
    - cmaes_with_margin: true (prevents discrete params from collapsing)

Notes on restart strategies:
    - bipop (default): Alternates between small (exploitation) and large (exploration) populations
    - ipop: Restarts with 2x population each time (simpler, good for escaping local minima)
    - lr_adapt: Adapts learning rate for noisy/multimodal landscapes (incompatible with restarts)
    - Requires: pip install optunahub (for ipop/bipop)
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
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, BaseSampler
from optuna.distributions import FloatDistribution
from tqdm import tqdm
from pydantic import Field, ConfigDict, BaseModel
from typing import Literal

try:
    import optunahub

    _OPTUNAHUB_AVAILABLE = True
except ImportError:
    _OPTUNAHUB_AVAILABLE = False

from dracon.commandline import Arg, make_program
from dracon.deferred import DeferredNode

from biocomp.compute import ComputeConfig
from biocomp.datautils import DataConfig
from biocomp.train import TrainingConfig, start, CompiledTrainingStep, compile_training_step
from biocomp.library import load_lib
from biocomp.jaxutils import tree_get
from biocomptools.toollib.common import config
from biocomptools.toollib.networkselector import build_data_manager, NetworkSet, NetworkDataPair
from biocomptools.optimtools import make_context_from_types, DEFAULT_TYPES
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.toollib.networkprediction import NetworkPrediction

logger = get_logger(__name__)


class TqdmProgressLogger(Logger):
    """Simple tqdm progress bar for training steps."""

    trial_number: int = 0
    total_steps: int = 0
    async_ok: bool = False
    _pbar: Any = None

    def get_callbacks(self, training_program):
        def on_start(step, training_config, step_history, stack):
            self.total_steps = int(
                training_config.n_epochs
                * training_config.n_batches
                / training_config.batches_per_step
            )
            self._pbar = tqdm(
                total=self.total_steps,
                desc=f"Trial {self.trial_number}",
                unit="step",
                leave=False,
            )

        def on_step(step, training_config, step_history, stack):
            if self._pbar is not None:
                loss = step_history.get('loss')
                if loss is not None:
                    mean_loss = float(np.asarray(loss).mean())
                    self._pbar.set_postfix(loss=f"{mean_loss:.4f}")
                self._pbar.update(1)

        def on_end(step, training_config, step_history, stack):
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None

        return [(0, on_start), (1, on_step), (-1, on_end)]


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


class SparseSampler(BaseSampler):
    """Sampler that uses beta distribution biased toward low values (sparse weights).

    For float parameters, samples from Beta(alpha, 2) scaled to [low, high].
    With alpha=0.5, most samples will be near 0 (the low end).
    """

    def __init__(self, alpha: float = 0.5, seed: int | None = None):
        self._alpha = alpha
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        if isinstance(param_distribution, FloatDistribution):
            # Beta(alpha, 2) is skewed toward 0 when alpha < 1
            sample = self._rng.beta(self._alpha, 2.0)
            low, high = param_distribution.low, param_distribution.high
            return low + sample * (high - low)
        # Fall back to uniform for other types
        return self._rng.uniform(param_distribution.low, param_distribution.high)


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
    n_jobs: Annotated[
        int, Arg(help='Number of parallel trials (-1 = CPU count, 1 = sequential)')
    ] = 1
    n_startup_trials: Annotated[int, Arg(help='Random trials before sampler kicks in')] = 20
    pruning: Annotated[bool, Arg(help='Enable early stopping of bad trials')] = True
    seed: Annotated[int | None, Arg(help='Random seed')] = None

    # sampler configuration
    sampler: Annotated[
        Literal["tpe", "cmaes", "hybrid", "sparse", "sparse_cmaes"],
        Arg(
            help='Sampler: tpe, cmaes, hybrid (TPE->CMA-ES), sparse (beta toward 0), sparse_cmaes (sparse startup then CMA-ES)'
        ),
    ] = "tpe"
    hybrid_switch_trial: Annotated[
        int, Arg(help='For hybrid sampler: switch from TPE to CMA-ES after this many trials')
    ] = 100

    # CMA-ES specific configuration (defaults optimized to avoid local minima)
    cmaes_restart_strategy: Annotated[
        Literal["none", "ipop", "bipop"] | None,
        Arg(help='CMA-ES restart strategy: bipop (default), ipop, or none'),
    ] = "bipop"
    cmaes_popsize: Annotated[
        int | None, Arg(help='CMA-ES population size (default 32, larger = more exploration)')
    ] = 32
    cmaes_warm_start: Annotated[
        bool, Arg(help='Warm start CMA-ES from existing trials when resuming a study')
    ] = True
    cmaes_inc_popsize: Annotated[
        int, Arg(help='Population size multiplier for restarts (ipop/bipop)')
    ] = 2
    cmaes_sigma0: Annotated[
        float | None, Arg(help='Initial step size for CMA-ES (None = auto, typically min_range/6)')
    ] = None
    cmaes_lr_adapt: Annotated[
        bool,
        Arg(
            help='Use learning rate adaptation for multimodal/noisy problems (incompatible with restarts)'
        ),
    ] = False
    cmaes_with_margin: Annotated[
        bool, Arg(help='Use CMA-ES with margin (prevents discrete params from collapsing)')
    ] = True
    cmaes_x0: Annotated[
        Literal["center", "sparse", "dense"] | None,
        Arg(
            help='CMA-ES initial point: center (default), sparse (most weights near 0), dense (most weights near 1)'
        ),
    ] = None
    sparse_sampling: Annotated[
        bool,
        Arg(
            help='Use sparse random sampling (beta distribution biased toward 0) instead of uniform for startup trials'
        ),
    ] = False
    sparse_alpha: Annotated[
        float,
        Arg(
            help='Alpha parameter for beta distribution when sparse_sampling=True (lower = more sparse, default 0.5)'
        ),
    ] = 0.5

    # output configuration
    output_dir: Annotated[str, Arg(help='Directory to save results')] = Field(
        default_factory=lambda: str(Path(config.paths.root) / "hyperopt_results")
    )

    # training configuration (deferred - reconstructed per trial with hyperparameters)
    training_conf: Annotated[DeferredNode[TrainingConfig], Arg(help='Base training config')]
    compute_conf: Annotated[DeferredNode[ComputeConfig], Arg(help='Base compute config')]
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = Field(default_factory=DataConfig)
    # training_set can be either a NetworkSet (for standard use) or DeferredNode (for weight optimization)
    training_set: Annotated[
        NetworkSet | DeferredNode[NetworkSet], Arg(help='Networks in training set')
    ] = Field(default_factory=NetworkSet)

    # validation configuration (for validation-based objective)
    validation_set: Annotated[NetworkSet | None, Arg(help='Networks for validation')] = None
    use_validation_loss: Annotated[bool, Arg(help='Use validation loss as objective')] = False
    n_validation_evals: Annotated[int, Arg(help='Number of validation samples per network')] = 2048

    # per-trial dataset weight support (for optimizing dataset weights)
    rebuild_dman_per_trial: Annotated[bool, Arg(help='Rebuild data manager each trial')] = False

    # hyperparameters to optimize
    hyperparams: Annotated[list[HyperparamSpec], Arg(help='Hyperparameters to optimize')] = Field(
        default_factory=list
    )

    # model saving configuration
    n_top_models: Annotated[int, Arg(help='Number of top models to keep (0 to disable)')] = 10

    # info-only modes
    show_best: Annotated[bool, Arg(help='Show best results and exit')] = False
    export_only: Annotated[bool, Arg(help='Export results and exit')] = False
    dashboard: Annotated[bool, Arg(help='Launch Optuna dashboard web interface and exit')] = False
    dashboard_port: Annotated[int, Arg(help='Port for Optuna dashboard')] = 8080

    # internal state
    _lib: Any = None
    _training_dman: Any = None
    _validation_predictor: Any = None
    _validation_xynetworks: Any = None
    _network_weight_mapping: list[str] | None = None  # network_idx -> ndp.network_name
    _cached_step: CompiledTrainingStep | None = None  # cached compiled step for JIT reuse

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

    def _prepare_data_manager(self, hyperparams: dict | None = None):
        """Build data manager. If rebuild_dman_per_trial, update weights only (fast path)."""
        # fast path: if we have a DataManager and just need to update weights
        if (
            self._training_dman is not None
            and self.rebuild_dman_per_trial
            and hyperparams is not None
        ):
            if self._network_weight_mapping is not None:
                # update weights using the mapping
                self._update_weights_from_hyperparams(hyperparams)
                return

        if self._training_dman is not None and not self.rebuild_dman_per_trial:
            return

        logger.info("Preparing data manager...")

        # resolve training_set if it's a DeferredNode
        if isinstance(self.training_set, DeferredNode):
            # construct with hyperparameters as context for weight interpolation
            context = hyperparams or {}
            resolved_training_set = self.training_set.construct(context=context)
        else:
            resolved_training_set = self.training_set

        with self.db_session as session:
            resolved_training_set.run_selectors(session)
            self._training_dman = build_data_manager(
                lib=self._lib,
                db_session=session,
                path_prefix=self.path_prefix,
                data_conf=self.data_conf,
                dataset=resolved_training_set,
            )
            # build weight mapping for fast updates on subsequent trials
            if self.rebuild_dman_per_trial:
                self._build_weight_mapping(resolved_training_set, session)
        logger.info(f"Data manager ready ({len(resolved_training_set.content)} network-data pairs)")

    def _build_weight_mapping(self, dataset: NetworkSet, session):
        """Build mapping from network indices to NetworkDataPair names for weight updates."""
        net_data = dataset.get_networks_and_data(session)
        networks, _ = zip(*net_data)

        # map network_idx -> ndp.network_name (respecting network splitting)
        self._network_weight_mapping = []
        for n in networks:
            n.build(self._lib)  # ensure built (should be cached)
            network_list = n._network if isinstance(n._network, list) else [n._network]
            for _ in network_list:
                self._network_weight_mapping.append(n.name)

    def _update_weights_from_hyperparams(self, hyperparams: dict):
        """Update DataManager weights based on hyperparameters (fast path)."""
        # find which hyperparam controls each network's weight
        # hyperparams keys are like "dataweight_1_Single_CasE_no_ratios"
        # we need to match these to network names

        # rebuild the ndp_weights by resolving the training_set with new context
        if isinstance(self.training_set, DeferredNode):
            resolved = self.training_set.construct(context=hyperparams)
            with self.db_session as session:
                resolved.run_selectors(session)
                ndp_weights = {ndp.network_name: ndp.weight for ndp in resolved.content}
        else:
            # no deferred node, weights don't change
            return

        # apply weights using the mapping
        new_weights = [ndp_weights.get(name, 1.0) for name in self._network_weight_mapping]
        self._training_dman.set_weights(new_weights)

    def _compile_cached_step(self, training_conf: TrainingConfig, compute_conf: ComputeConfig):
        """Compile and cache the training step for reuse across trials.

        This is called once on the first trial. The compiled step can then be
        reused across subsequent trials since only the weights change (not the
        network structure or training configuration).
        """
        if self._cached_step is not None:
            return  # already compiled

        logger.info("Compiling training step (first trial only)...")

        self._cached_step = compile_training_step(
            dman=self._training_dman,
            training_config=training_conf,
            compute_config=compute_conf,
        )
        logger.info("Training step compiled")

    def _prepare_validation(self, compute_conf: ComputeConfig):
        """Initialize validation predictor (done once)."""
        if self._validation_predictor is not None:
            return
        if self.validation_set is None:
            return

        logger.info("Preparing validation predictor...")
        with self.db_session as session:
            self.validation_set.run_selectors(session)
            val_dman = build_data_manager(
                lib=self._lib,
                db_session=session,
                path_prefix=self.path_prefix,
                data_conf=self.data_conf,
                dataset=self.validation_set,
                jax_sampling=False,
            )

        self._validation_xynetworks = val_dman.get_per_network_xy_samples(self.n_validation_evals)
        xs, ys, networks = self._validation_xynetworks

        model = BiocompModel(compute_config=compute_conf, rescaler=self.data_conf.rescaler)
        network_model = NetworkModel(model=model, network=networks)

        per_prediction_info = [
            {'network_name': n.name, 'networkdatapair': {'network_name': n.name}} for n in networks
        ]

        self._validation_predictor = NetworkPrediction(
            predict_at=xs,
            network_model=network_model,
            ground_truth=ys,
            seed=self.seed or 42,
            disable_variational=True,
            max_evals=self.n_validation_evals,
            already_latent=True,
            n_stats_workers=1,
            per_prediction_info=per_prediction_info,
            device='cpu',
            enable_gridstats=False,  # skip expensive KNN grid stats for hyperopt validation
        )
        logger.info(f"Validation predictor ready ({len(networks)} networks)")

    def _compute_validation_loss(self, params) -> float:
        """Compute validation loss using trained params."""
        if self._validation_predictor is None:
            return float('inf')

        # get stats for first replicate
        stats = self._validation_predictor.get_network_stats(with_shared_params=tree_get(params, 0))
        valid_stats = [s for s in stats if s.get('rmse') is not None]

        if not valid_stats:
            return float('inf')

        return float(np.mean([s['rmse'] for s in valid_stats]))

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
            logger.info(f"No existing study found with name '{self.study_name}'")
            return False

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        lines = [
            f"Study: {self.study_name} | Storage: {self._get_storage_path()}",
            f"Trials: {len(study.trials)} total ({n_complete} complete, {n_pruned} pruned, {n_failed} failed)",
        ]
        if n_complete > 0:
            params = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in study.best_params.items())
            lines.append(f"Best: #{study.best_trial.number} loss={study.best_value:.6f} [{params}]")
        logger.info("\n".join(lines))
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

    def _compute_x0(self) -> dict[str, float] | None:
        """Compute initial point for CMA-ES based on cmaes_x0 setting.

        Returns:
            Dictionary mapping parameter names to initial values, or None for default (center).
        """
        if self.cmaes_x0 is None or self.cmaes_x0 == "center":
            return None

        x0 = {}
        for hp in self.hyperparams:
            if hp.type in ("float", "log_float", "int"):
                low, high = hp.low, hp.high
                if self.cmaes_x0 == "sparse":
                    # Start near the low end (10% of range)
                    x0[hp.name] = low + 0.1 * (high - low)
                elif self.cmaes_x0 == "dense":
                    # Start near the high end (90% of range)
                    x0[hp.name] = low + 0.9 * (high - low)
            # categorical params don't get x0
        return x0 if x0 else None

    def _create_cmaes_sampler(
        self,
        n_startup: int = 1,
        source_trials: list | None = None,
        independent_sampler: BaseSampler | None = None,
    ) -> optuna.samplers.BaseSampler:
        """Create CMA-ES sampler with configured restart strategy.

        Uses OptunaHub's RestartCmaEsSampler for ipop/bipop strategies, which are
        designed to escape local minima by restarting with modified population sizes.

        - ipop: Restart with increasing population size (better exploration)
        - bipop: Alternate between small (exploitation) and large (exploration) populations
        - lr_adapt: Learning rate adaptation for multimodal/noisy problems
        - with_margin: Prevents discrete parameters from collapsing to single values
        - source_trials: Warm start from existing trials (estimates initial distribution)
        - x0: Initial point (sparse/dense/center)
        """
        restart_strategy = self.cmaes_restart_strategy
        if restart_strategy == "none":
            restart_strategy = None

        # compute x0 if specified
        x0 = self._compute_x0()
        if x0:
            logger.info(
                f"Using {self.cmaes_x0} initialization for CMA-ES (x0 values near {list(x0.values())[0]:.2f})"
            )

        # use OptunaHub for restart strategies (ipop/bipop)
        if restart_strategy in ("ipop", "bipop"):
            if not _OPTUNAHUB_AVAILABLE:
                logger.warning(
                    "optunahub not installed, falling back to standard CMA-ES without restarts. "
                    "Install with: pip install optunahub"
                )
                restart_strategy = None
            else:
                # RestartCmaEsSampler doesn't support with_margin, lr_adapt, source_trials, or x0
                if self.cmaes_with_margin:
                    logger.info(
                        f"Note: with_margin not supported with {restart_strategy} restarts, "
                        "using restart strategy (more important for avoiding local minima)"
                    )
                if source_trials:
                    logger.info(
                        f"Note: warm start (source_trials) not supported with {restart_strategy} restarts"
                    )
                if x0:
                    logger.info(
                        f"Note: x0 not supported with {restart_strategy} restarts, "
                        "x0 is sampled uniformly within search space for each restart"
                    )
                module = optunahub.load_module("samplers/restart_cmaes")
                RestartCmaEsSampler = module.RestartCmaEsSampler
                logger.info(
                    f"Using RestartCmaEsSampler with {restart_strategy} strategy, popsize={self.cmaes_popsize}"
                )
                return RestartCmaEsSampler(
                    n_startup_trials=n_startup,
                    seed=self.seed,
                    restart_strategy=restart_strategy,
                    popsize=self.cmaes_popsize,
                    inc_popsize=self.cmaes_inc_popsize,
                    sigma0=self.cmaes_sigma0,
                    warn_independent_sampling=False,
                )

        # standard CMA-ES (no restarts, but can use lr_adapt, with_margin, source_trials, and x0)
        if source_trials:
            logger.info(f"Warm starting CMA-ES from {len(source_trials)} existing trials")
            if x0:
                logger.info("Note: x0 is ignored when using source_trials (warm start)")
                x0 = None
        return CmaEsSampler(
            x0=x0,
            n_startup_trials=n_startup,
            seed=self.seed,
            popsize=self.cmaes_popsize,
            sigma0=self.cmaes_sigma0,
            lr_adapt=self.cmaes_lr_adapt,
            with_margin=self.cmaes_with_margin,
            source_trials=source_trials,
            independent_sampler=independent_sampler,
            warn_independent_sampling=False,
        )

    def _create_sampler(
        self, n_completed_trials: int = 0, source_trials: list | None = None
    ) -> optuna.samplers.BaseSampler:
        """Create sampler based on configuration.

        For hybrid mode, returns TPE if below switch threshold, CMA-ES otherwise.
        CMA-ES is particularly effective for continuous hyperparameters in high dimensions
        once we have some good samples to initialize from.
        """
        # Use sparse sampler for startup trials if requested
        if self.sparse_sampling:
            logger.info(
                f"Using sparse sampling (beta alpha={self.sparse_alpha}) for independent samples"
            )

        if self.sampler == "sparse":
            # Pure sparse random sampling (no CMA-ES)
            return SparseSampler(alpha=self.sparse_alpha, seed=self.seed)
        elif self.sampler == "tpe":
            return TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        elif self.sampler == "cmaes":
            # For CMA-ES with sparse_sampling, use SparseSampler as independent_sampler
            independent_sampler = (
                SparseSampler(alpha=self.sparse_alpha, seed=self.seed)
                if self.sparse_sampling
                else None
            )
            return self._create_cmaes_sampler(
                n_startup=self.n_startup_trials,
                source_trials=source_trials if self.cmaes_warm_start else None,
                independent_sampler=independent_sampler,
            )
        elif self.sampler == "hybrid":
            if n_completed_trials < self.hybrid_switch_trial:
                logger.info(
                    f"Hybrid mode: using TPE (trial {n_completed_trials} < {self.hybrid_switch_trial})"
                )
                return TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
            else:
                logger.info(
                    f"Hybrid mode: switched to CMA-ES (trial {n_completed_trials} >= {self.hybrid_switch_trial})"
                )
                return self._create_cmaes_sampler(
                    n_startup=0,  # don't need random trials, we have TPE samples
                    source_trials=source_trials if self.cmaes_warm_start else None,
                )
        elif self.sampler == "sparse_cmaes":
            # Sparse random startup, then CMA-ES optimization from sparse region
            if n_completed_trials < self.n_startup_trials:
                logger.info(
                    f"sparse_cmaes: using sparse sampling (trial {n_completed_trials} < {self.n_startup_trials})"
                )
                return SparseSampler(alpha=self.sparse_alpha, seed=self.seed)
            else:
                logger.info(
                    f"sparse_cmaes: switched to CMA-ES (trial {n_completed_trials} >= {self.n_startup_trials})"
                )
                # Use sparse trials to warm-start CMA-ES (it will learn the sparse distribution)
                return self._create_cmaes_sampler(
                    n_startup=0,  # don't need more random trials
                    source_trials=source_trials if self.cmaes_warm_start else None,
                    independent_sampler=SparseSampler(alpha=self.sparse_alpha, seed=self.seed),
                )
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

    def _launch_dashboard(self):
        """Launch Optuna dashboard web interface."""
        storage = self._get_storage_path()

        try:
            import optuna_dashboard
            logger.info(f"Launching Optuna Dashboard at http://localhost:{self.dashboard_port} (Ctrl+C to stop)")
            optuna_dashboard.run_server(storage, port=self.dashboard_port)
        except ImportError:
            logger.error(f"optuna-dashboard not installed. Install with: pip install optuna-dashboard")
            logger.error(f"Or run manually: optuna-dashboard {storage}")
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

        logger.info(f"Results saved to {results_dir} (best loss: {study.best_value:.6f})")

    def _get_top_models_dir(self) -> Path:
        """Get directory for top models."""
        return Path(self.output_dir) / self.study_name / "top_models"

    def _save_model_if_top(
        self,
        trial_number: int,
        loss: float,
        params,
        compute_conf: ComputeConfig,
        hyperparams: dict,
    ):
        """Save model if it's among the top N, evicting the worst if needed."""
        if self.n_top_models <= 0 or loss == float('inf'):
            return

        from biocomp.jaxutils import tree_to_np
        from biocomptools.run_training import get_shared_params

        models_dir = self._get_top_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        # get existing models and their losses
        existing = []
        for f in models_dir.glob("*.pickle"):
            try:
                # parse loss from filename: trial_NNN_loss_X.XXXXXX.pickle
                parts = f.stem.split("_")
                idx = parts.index("loss")
                file_loss = float(parts[idx + 1])
                existing.append((file_loss, f))
            except (ValueError, IndexError):
                continue

        existing.sort(key=lambda x: x[0])  # sort by loss ascending

        # check if this model should be saved
        if len(existing) >= self.n_top_models and loss >= existing[-1][0]:
            return  # not good enough

        # create model
        shared_params = get_shared_params(pickle.loads(pickle.dumps(tree_get(params, 0))))
        if shared_params is None:
            logger.warning(f"Trial {trial_number}: could not extract shared params")
            return

        model = BiocompModel(
            compute_config=compute_conf,
            rescaler=self.data_conf.rescaler,
            shared_params=tree_to_np(shared_params),
            metadata={
                'hyperopt_trial': trial_number,
                'hyperopt_loss': loss,
                'hyperopt_study': self.study_name,
                'hyperparams': {k: v for k, v in hyperparams.items() if not k.startswith('_')},
            },
        )

        # save model
        fname = models_dir / f"trial_{trial_number:04d}_loss_{loss:.6f}.pickle"
        model.save(fname)
        logger.debug(f"Saved model {model.signature} (trial {trial_number}, loss {loss:.6f})")

        # evict worst if over limit
        existing.append((loss, fname))
        existing.sort(key=lambda x: x[0])
        while len(existing) > self.n_top_models:
            _, worst_file = existing.pop()
            worst_file.unlink()
            logger.debug(f"Evicted model {worst_file.name}")

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

            # update weights if needed (for dataset weight optimization)
            # _prepare_data_manager uses fast path when DataManager exists
            if self.rebuild_dman_per_trial:
                self._prepare_data_manager(hyperparams)

            # compile and cache step on first trial (for dataset weight optimization)
            # only do this when rebuild_dman_per_trial is True (dataset weights only change)
            # and hyperparams are only dataset weights (no structural hyperparams)
            use_cached_step = self.rebuild_dman_per_trial and all(
                spec.name.startswith("dataweight_") for spec in self.hyperparams
            )
            if use_cached_step:
                self._compile_cached_step(training_conf, compute_conf)

            # init validation predictor on first trial if needed
            if self.use_validation_loss and self._validation_predictor is None:
                self._prepare_validation(compute_conf)

            # create loggers
            pruning_logger = None
            if self.pruning and not self.use_validation_loss:
                pruning_logger = OptunaPruningLogger(trial=trial)

            progress_logger = TqdmProgressLogger(trial_number=trial.number)

            logger_callbacks = []
            # add progress bar callbacks
            progress_logger.initialize(None)
            logger_callbacks.extend(progress_logger.get_callbacks(None))
            # add pruning callbacks
            if pruning_logger:
                pruning_logger.initialize(None)
                logger_callbacks.extend(pruning_logger.get_callbacks(None))

            t0 = time.time()
            params, loss_history, _ = start(
                dman=self._training_dman,
                training_config=training_conf,
                compute_config=compute_conf,
                loggers=logger_callbacks,
                cached_step=self._cached_step if use_cached_step else None,
            )
            train_elapsed = time.time() - t0

            # determine final loss
            if self.use_validation_loss:
                # use validation RMSE as objective
                t_val = time.time()
                loss = self._compute_validation_loss(params)
                val_elapsed = time.time() - t_val
                logger.info(f"Trial {trial.number}: val_loss={loss:.6f} (train={train_elapsed:.1f}s, val={val_elapsed:.1f}s)")
            elif pruning_logger:
                loss = pruning_logger.get_final_loss()
                logger.info(f"Trial {trial.number}: train_loss={loss:.6f} ({train_elapsed:.1f}s)")
            elif loss_history:
                losses = np.asarray(loss_history)
                mean_losses = losses.mean(axis=(1, 2))
                n_final = max(1, len(mean_losses) // 20)
                loss = float(mean_losses[-n_final:].mean())
                logger.info(f"Trial {trial.number}: train_loss={loss:.6f} ({train_elapsed:.1f}s)")
            else:
                loss = float('inf')
                logger.warning(f"Trial {trial.number}: loss=inf (no data)")

            # save model if among top N
            self._save_model_if_top(trial.number, loss, params, compute_conf, hyperparams)

            return loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"Trial {trial.number} failed: {e}")
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
                logger.warning(f"No existing study found with name '{self.study_name}'")
                return
            self._save_results(study)
            self._show_study_status()
            return

        # prepare data manager
        self._prepare_data_manager()

        # check for existing study and count completed trials
        existing = self._load_existing_study()
        n_complete = 0
        source_trials = None
        if existing is not None:
            completed_trials = [
                t for t in existing.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            n_complete = len(completed_trials)
            logger.info(f"Resuming study: {len(existing.trials)} trials ({n_complete} complete)")
            if n_complete > 0:
                logger.info(f"Current best loss: {existing.best_value:.6f}")
                # use completed trials for warm starting CMA-ES
                if self.cmaes_warm_start and self.sampler in ("cmaes", "hybrid"):
                    source_trials = completed_trials

        # create sampler based on config and current progress
        sampler = self._create_sampler(n_complete, source_trials=source_trials)
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

        sampler_info = self.sampler
        if self.sampler == "hybrid":
            sampler_info += f" (switch at {self.hybrid_switch_trial})"
        if self.sampler in ("cmaes", "hybrid"):
            cmaes_opts = []
            if self.cmaes_restart_strategy and self.cmaes_restart_strategy != "none":
                cmaes_opts.append(f"restart={self.cmaes_restart_strategy}")
            if self.cmaes_lr_adapt:
                cmaes_opts.append("lr_adapt")
            if self.cmaes_with_margin:
                cmaes_opts.append("with_margin")
            if self.cmaes_popsize:
                cmaes_opts.append(f"popsize={self.cmaes_popsize}")
            if cmaes_opts:
                sampler_info += f" [{', '.join(cmaes_opts)}]"
        parallel_info = ""
        if self.n_jobs != 1:
            parallel_info = f" | {'all CPUs' if self.n_jobs == -1 else f'{self.n_jobs} workers'}"
        logger.info(
            f"Starting optimization: {self.n_trials} trials | {self.study_name} | {sampler_info}{parallel_info}\n"
            f"Hyperparameters: {[h.name for h in self.hyperparams]}"
        )

        # show_progress_bar doesn't work well with n_jobs > 1
        show_progress = self.n_jobs == 1

        try:
            if self.sampler in ("hybrid", "sparse_cmaes"):
                # run in batches to allow switching sampler mid-optimization
                switch_at = (
                    self.hybrid_switch_trial if self.sampler == "hybrid" else self.n_startup_trials
                )
                remaining = self.n_trials
                while remaining > 0:
                    completed_trials = [
                        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                    n_complete = len(completed_trials)
                    # get source trials for warm start
                    warm_source = completed_trials if self.cmaes_warm_start else None
                    # check if we need to switch sampler
                    new_sampler = self._create_sampler(n_complete, source_trials=warm_source)
                    if type(new_sampler) != type(study.sampler):
                        logger.info(
                            f"Switching sampler from {type(study.sampler).__name__} to {type(new_sampler).__name__}"
                        )
                        # recreate study with new sampler (will load existing trials)
                        study = optuna.create_study(
                            study_name=self.study_name,
                            storage=self._get_storage_path(),
                            load_if_exists=True,
                            direction="minimize",
                            sampler=new_sampler,
                            pruner=pruner,
                        )
                    # run batch
                    batch_size = min(remaining, max(1, switch_at - n_complete))
                    study.optimize(
                        self._run_single_trial,
                        n_trials=batch_size,
                        n_jobs=self.n_jobs,
                        show_progress_bar=show_progress,
                    )
                    remaining -= batch_size
            else:
                study.optimize(
                    self._run_single_trial,
                    n_trials=self.n_trials,
                    n_jobs=self.n_jobs,
                    show_progress_bar=show_progress,
                )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted. Progress saved.")

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
        deferred_paths=['/training_conf', '/compute_conf', '/training_set'],
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
