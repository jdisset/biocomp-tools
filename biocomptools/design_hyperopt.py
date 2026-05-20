# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Hyperparameter optimization for design runs using CMA-ES.

Uses JAX-native schedule evaluation to avoid recompilation between trials.

Usage:
    biocomp-design-hyperopt +biocomp-jobs/hyperopt/design_lattice
"""

from __future__ import annotations

import time
import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any, Annotated

import numpy as np
import optuna
from pydantic import Field
import jax
import jax.numpy as jnp
from jax import vmap

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode

from biocomp.design import (
    DesignManager,
    DesignConfig,
    initialize_params,
    get_ratio_paths_and_sources,
    normalize_ratio_source_arrays,
    sample_for_evaluation,
)
from biocomp.design_targets import TargetUnion, SamplingConfigUnion, UniformSampling
from biocomp.ratio_utils import normalize_ratios_for_pruning as normalize_ratios_prune
from biocomp.designloss import HYPEROPT_SCHEDULE_NAMESPACE
from biocomp.recipe import Recipe
from biocomp.tumasking import TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX
from biocomp.tumasking_strategy import get_full_log_alpha
from biocomp.optimutils import make_training_step, per_replicate_step, optimize, compile_step
from biocomp.parameters import ParameterTree

from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.toollib.common import config
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.optimtools import DEFAULT_TYPES
from biocomptools.hyperopt.base import (
    BaseHyperoptProgram,
    HyperparamSpec,
    expand_schedule_hyperparams,
    get_schedule_param_names,
    SCHEDULE_SUFFIXES,
)

logger = get_logger(__name__)


def _build_design_manager(
    targets: list[TargetUnion],
    scaffolds: list[Recipe],
    sampling: SamplingConfigUnion,
    model: BiocompModel,
    enable_tu_masking: bool,
) -> DesignManager:
    """Build DesignManager from scaffolds."""
    from biocomp.network import recipe_to_networks

    networks = []
    for recipe in scaffolds:
        nets = recipe_to_networks(recipe, invert=True, inversion_mode="main")
        networks.extend(nets)

    return DesignManager(
        targets=targets,
        networks=networks,
        sampling=sampling,
        enable_tu_masking=enable_tu_masking,
    )


@dracon_program(
    name='biocomp-design-hyperopt',
    description='Run hyperparameter optimization for design runs.',
    context_types=DEFAULT_TYPES + [HyperparamSpec],
    context={'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve()},
)
class DesignHyperoptProgram(BaseHyperoptProgram):
    """Design hyperopt with JAX-native schedule injection for recompilation-free trials.

    Extends BaseHyperoptProgram with:
    - Design configuration and scaffold management
    - JAX-native schedule parameter injection
    - RMSE-based evaluation
    """

    study_name: Annotated[str, Arg(help='Study name for persistence')] = "design_hyperopt"
    output_dir: Annotated[str, Arg(help='Output directory')] = "./design_hyperopt_output"
    sampler: Annotated[str, Arg(help='Sampler: tpe, cmaes')] = "cmaes"
    n_startup_trials: int = 5
    cmaes_popsize: int | None = 16
    cmaes_sigma0: float = 0.3

    # Design configuration
    design_conf: Annotated[DeferredNode[DesignConfig] | DesignConfig, Arg(help='Design config')] = (
        Field(default_factory=DesignConfig)
    )

    targets: Annotated[list[TargetUnion] | TargetUnion, Arg(help='Design targets')] = Field(
        default_factory=list
    )

    scaffolds: Annotated[list[Recipe] | Recipe, Arg(help='Scaffold recipes')] = Field(
        default_factory=list
    )

    sampling: Annotated[SamplingConfigUnion, Arg(help='Sampling strategy')] = Field(
        default_factory=UniformSampling
    )

    model_name: Annotated[str | None, Arg(help='Model path or name')] = None
    model_selector: Annotated[ModelSelector | None, Arg(help='Model selector')] = None

    # Evaluation configuration
    eval_n_samples: Annotated[int, Arg(help='Samples for RMSE evaluation')] = 10000
    eval_top_k: Annotated[int, Arg(help='Best K networks to average')] = 3
    use_eval_loss: Annotated[bool, Arg(help='Use RMSE eval instead of training loss')] = True
    n_top_models: int = 5

    # Internal state
    _model: Any = None
    _dmanager: Any = None
    _cached_step: Any = None
    _cached_stack: Any = None
    _cached_eval_fn: Any = None
    _initial_params: Any = None
    _static_params: Any = None
    _ratio_paths: list | None = None
    _total_steps: int | None = None
    _design_conf: Any = None
    _tu_masking_strategy: Any = None

    def model_post_init(self, _):
        if isinstance(self.targets, TargetUnion):
            self.targets = [self.targets]
        if isinstance(self.scaffolds, Recipe):
            self.scaffolds = [self.scaffolds]

    def _prepare(self):
        """Prepare design manager and compile design step."""
        dconf = (
            self.design_conf.construct(context={})
            if isinstance(self.design_conf, DeferredNode)
            else self.design_conf
        )
        self._design_conf = dconf

        self._prepare_design_manager()
        self._compile_design_step(dconf)

    def _load_model(self):
        """Load the BiocompModel."""
        if self._model is not None:
            return

        if self.model_name:
            model_path = Path(self.model_name)
            if model_path.exists() and model_path.suffix == '.pickle':
                self._model = BiocompModel.load(model_path)
                logger.info(f"Loaded model from {model_path}")
                return

        if self.model_selector:
            from biocomptools.toollib.models import get_biocompdb_sqlite_engine
            from sqlmodel import Session

            engine = get_biocompdb_sqlite_engine(config.db.sqlite.path)
            with Session(engine) as session:
                selected = self.model_selector.get_model(session)
                if selected:
                    self._model = selected.load()
                    logger.info(f"Loaded model: {selected.name}")

        if self._model is None:
            raise ValueError("No model specified. Use model_name or model_selector.")

    def _prepare_design_manager(self):
        """Build DesignManager from config."""
        if self._dmanager is not None:
            return

        self._load_model()

        self._dmanager = _build_design_manager(
            targets=self.targets,
            scaffolds=self.scaffolds,
            sampling=self.sampling,
            model=self._model,
            enable_tu_masking=self._design_conf.enable_tu_masking,
        )
        logger.info(
            f"DesignManager ready: {len(self._dmanager.networks)} networks, "
            f"{self._dmanager.n_targets} targets"
        )

    def _prepopulate_hyperopt_schedules(
        self,
        params: ParameterTree,
        n_replicates: int,
        n_targets: int,
        loss_fn_defaults: dict[str, Any] | None = None,
    ) -> None:
        """Pre-populate params tree with hyperopt schedule paths before JIT compilation.

        JAX pytree structure is fixed at compile time - paths must exist before compilation.
        Values have shape (n_replicates, n_targets) to match other non_grad params.

        Handles three schedule specification modes:
        1. Constant: Just base name (e.g., 'w_sinkhorn') -> prepopulates all 3 phase values
        2. Linear: phase1_value + phase3_end_value -> prepopulates all 3 (phase2 computed at injection)
        3. Full 3-phase: All three phase values explicitly provided

        Also prepopulates non-optimized params from loss_fn_defaults as constants.
        """
        assert self.hyperparams, (
            "hyperparams must be set before calling _prepopulate_hyperopt_schedules"
        )
        ns = HYPEROPT_SCHEDULE_NAMESPACE
        shape = (n_replicates, n_targets)

        schedule_info = get_schedule_param_names(self.hyperparams)
        optimized_names = set(schedule_info.keys())

        for sched_name, _suffixes in schedule_info.items():
            # Always prepopulate all 3 phase values for every schedule
            # This ensures the params tree structure is complete before JIT
            for suffix in SCHEDULE_SUFFIXES:
                path = f"{ns}/{sched_name}{suffix}"
                params.at(
                    path,
                    jnp.zeros(shape, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )

            # Also prepopulate phase fractions
            params.at(
                f"{ns}/{sched_name}_phase1_frac",
                jnp.full(shape, 0.4, dtype=jnp.float32),
                tags=["non_grad", "hyperopt"],
                overwrite=True,
            )
            params.at(
                f"{ns}/{sched_name}_phase2_frac",
                jnp.full(shape, 0.75, dtype=jnp.float32),
                tags=["non_grad", "hyperopt"],
                overwrite=True,
            )

        # Also prepopulate non-optimized params from loss function defaults as constants
        # This prevents "schedule not found" warnings for params we intentionally don't optimize
        if loss_fn_defaults:
            for name, default_val in loss_fn_defaults.items():
                if name in optimized_names or not isinstance(default_val, (int, float)):
                    continue
                val = float(default_val)
                for suffix in SCHEDULE_SUFFIXES:
                    path = f"{ns}/{name}{suffix}"
                    params.at(
                        path,
                        jnp.full(shape, val, dtype=jnp.float32),
                        tags=["non_grad", "hyperopt"],
                        overwrite=True,
                    )
                params.at(
                    f"{ns}/{name}_phase1_frac",
                    jnp.full(shape, 0.4, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )
                params.at(
                    f"{ns}/{name}_phase2_frac",
                    jnp.full(shape, 0.75, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )
            logger.debug(
                f"Pre-populated {len(loss_fn_defaults) - len(optimized_names)} non-optimized schedule defaults"
            )

        logger.debug(
            f"Pre-populated {len(schedule_info)} hyperopt schedules in params tree (shape={shape})"
        )

    def _compile_design_step(self, dconf: DesignConfig):
        """Compile the design step function once, to be reused across trials."""
        if self._cached_step is not None:
            return

        logger.info("Compiling design step (will be reused for all trials)...")
        t0 = time.time()

        pkey, bkey, loop_key = jax.random.split(dconf.seed_key, 3)

        stack = self._dmanager.build_stack(self._model)
        self._cached_stack = stack

        strategy = dconf.build_tu_masking_strategy()
        self._tu_masking_strategy = strategy
        n_tus = self._dmanager.n_tus if strategy.has_masking else 0
        n_networks = len(self._dmanager.networks)

        initial_params = initialize_params(
            stack,
            dconf.n_replicates,
            self._dmanager.n_targets,
            self._model.shared_params,
            pkey,
            strategy=strategy,
            n_tus=n_tus,
            n_networks=n_networks,
            no_masking_tu_ids=stack.no_masking_tu_ids,
            tu_id_to_idx=stack.tu_id_to_idx,
        )

        loss_fn_defaults = getattr(dconf.loss_function, 'kwargs', None) or {}
        self._prepopulate_hyperopt_schedules(
            initial_params, dconf.n_replicates, self._dmanager.n_targets, loss_fn_defaults
        )

        static, dynamic = initial_params.filter_by_tag(["non_grad", "shared"])
        self._initial_params = initial_params
        self._static_params = static

        num_z = static["global/number_of_random_variables"]
        num_z = (self._dmanager.n_targets, int(num_z.ravel()[0].squeeze()))

        steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)
        self._total_steps = int(dconf.n_epochs * steps_per_epoch)

        direct_ratio_paths, source_ratio_paths = get_ratio_paths_and_sources(initial_params)
        self._ratio_paths = direct_ratio_paths

        def norm_ratios_hook(params, *a, **kw):
            if direct_ratio_paths:
                params = params.update_leaves_by_path(direct_ratio_paths, normalize_ratios_prune)
            if source_ratio_paths:
                params = normalize_ratio_source_arrays(
                    params, source_ratio_paths, normalize_ratios_prune
                )
            # Clip TU log_alpha params if strategy uses direct mode
            if strategy.has_masking:
                for path in strategy.param_paths:
                    if 'log_alpha' in path and path in params:
                        params = params.update_leaves_by_path(
                            [path], lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX)
                        )
            return params

        loss_func = dconf.loss_function.get_impl()(
            stack,
            dconf,
            self._dmanager,
            num_z=num_z,
            ratio_paths=self._ratio_paths,
            hyperopt_schedule_ns=HYPEROPT_SCHEDULE_NAMESPACE,
            hyperopt_total_steps=self._total_steps,
        )

        step_fn = make_training_step(
            loss_func,
            dconf.optimizer,
            dconf.keep_in_history,
            scannable=True,
            post_update_hook=norm_ratios_hook,
            updates_need_vmap=True,
            static_tags=["non_grad", "shared"],
            sanitize_grads=True,
        )

        effective_batch_size = dconf.batch_size
        if self._dmanager.is_lattice_mode:
            xres, yres = self._dmanager.grid_resolution
            effective_batch_size *= xres * yres

        def step(params: ParameterTree, opt_state, step_key, xs, ys):
            keys = jax.random.split(step_key, dconf.n_replicates)
            return jax.vmap(
                jax.tree_util.Partial(
                    per_replicate_step, num_z=num_z, training_config=dconf, scannable_step=step_fn
                )
            )(params, opt_state, keys, xs, ys)

        xbatches_sample, ybatches_sample = self._dmanager.get_samples(
            (
                len(self._dmanager.networks),
                1,
                dconf.n_replicates,
                dconf.batches_per_step,
                dconf.batch_size,
            ),
            jax.random.PRNGKey(42),
        )
        xb_sample = jnp.concatenate(xbatches_sample, axis=-1)[0]
        yb_sample = ybatches_sample[0][0]

        initial_optimizer_state = vmap(vmap(dconf.optimizer.init))(dynamic)

        sample_args = (
            initial_params,
            initial_optimizer_state,
            jax.random.PRNGKey(0),
            xb_sample,
            yb_sample,
        )
        self._cached_step = compile_step(step, sample_args)

        # Pre-compile evaluation function to avoid recompilation per trial
        if self.use_eval_loss:
            self._compile_eval_fn(dconf)

        logger.info(f"Design step compiled in {time.time() - t0:.1f}s")

    def _inject_hyperparams(self, params: ParameterTree, hp: dict) -> None:
        """Inject hyperparams into params tree in-place. Paths must already exist (pre-populated).

        Uses expand_schedule_hyperparams to handle:
        1. Constant schedules: base name -> all 3 phases get same value
        2. Linear schedules: phase1 + phase3 -> phase2 computed via interpolation
        3. Full 3-phase: all values used as-is
        """
        ns = HYPEROPT_SCHEDULE_NAMESPACE
        phase1_frac = hp.get('phase1_frac', 0.4)
        phase2_frac = hp.get('phase2_frac', 0.75)
        if phase1_frac >= phase2_frac:
            phase2_frac = min(phase1_frac + 0.1, 0.95)
            logger.warning(
                f"phase1_frac >= phase2_frac ({hp.get('phase1_frac')} >= {hp.get('phase2_frac')}), "
                f"adjusted phase2_frac to {phase2_frac}"
            )

        # Expand linear schedules to full 3-phase
        expanded_hp = expand_schedule_hyperparams(hp, phase1_frac, phase2_frac)

        def set_param(path: str, val: float) -> None:
            assert path in params, (
                f"Path {path} not in params - was _prepopulate_hyperopt_schedules called?"
            )
            shape = params[path].shape
            params.at(
                path,
                jnp.full(shape, val, dtype=jnp.float32),
                tags=["non_grad", "hyperopt"],
                overwrite=True,
            )

        schedule_names: set[str] = set()
        for name, value in expanded_hp.items():
            if name in ('phase1_frac', 'phase2_frac', 'seed'):
                continue
            for suffix in SCHEDULE_SUFFIXES:
                if name.endswith(suffix):
                    schedule_names.add(name[: -len(suffix)])
                    set_param(f"{ns}/{name}", value)
                    break

        for sched_name in schedule_names:
            set_param(f"{ns}/{sched_name}_phase1_frac", phase1_frac)
            set_param(f"{ns}/{sched_name}_phase2_frac", phase2_frac)

    def _compile_eval_fn(self, dconf: DesignConfig) -> None:
        """Pre-compile evaluation function to avoid recompilation per trial."""
        if self._cached_eval_fn is not None:
            return

        stack = self._cached_stack

        def apply_with_tu_mask(params, x_batch, z_batch, keys, tu_mask):
            def apply_single(x, z, k):
                return stack.apply(params, x, z, k, tu_enabled_random_vars=tu_mask)

            return vmap(apply_single)(x_batch, z_batch, keys)

        self._cached_eval_fn = jax.jit(apply_with_tu_mask)
        self._eval_dep_mask = stack.get_dependent_output_mask()
        self._eval_num_z = int(
            self._static_params["global/number_of_random_variables"][0, 0].squeeze()
        )
        logger.debug("Evaluation function pre-compiled")

    def _evaluate_cached(
        self,
        final_params: ParameterTree,
        xraw: jnp.ndarray,
        yraw: jnp.ndarray,
        key: jax.Array,
        max_eval_size: int = 64,
    ) -> jnp.ndarray:
        """Evaluate design using pre-compiled function. Returns MSE losses per network."""
        dconf = self._design_conf
        n_networks = len(self._dmanager.networks)
        n_replicates = dconf.n_replicates
        n_targets = self._dmanager.n_targets
        n_samples = xraw.shape[2]

        x_combined = xraw.transpose(1, 2, 3, 0, 4).reshape(n_replicates, n_samples, n_targets, -1)
        y_combined = yraw[0]

        has_tu_masking = self._tu_masking_strategy.has_masking
        all_losses = []

        for rep_idx in range(n_replicates):
            rep_losses = []
            for tid in range(n_targets):
                rep_params = jax.tree.map(lambda x, r=rep_idx, t=tid: x[r, t], final_params)
                x_slice = x_combined[rep_idx, :, tid, :]
                y_slice = y_combined[rep_idx, :, tid, :]

                if has_tu_masking:
                    log_alpha = get_full_log_alpha(rep_params)
                    tu_mask = jax.nn.sigmoid(log_alpha) if log_alpha is not None else None
                else:
                    tu_mask = None

                yhats = []
                for start in range(0, n_samples, max_eval_size):
                    end = min(start + max_eval_size, n_samples)
                    z_batch = jax.random.uniform(key, (end - start, self._eval_num_z))
                    yhat, _ = self._cached_eval_fn(
                        rep_params,
                        x_slice[start:end],
                        z_batch,
                        jax.random.split(key, end - start),
                        tu_mask,
                    )
                    yhats.append(yhat)

                yhat_dep = jnp.compress(
                    self._eval_dep_mask, jnp.concatenate(yhats, axis=0), axis=-1
                )
                rep_losses.append(
                    jnp.mean((yhat_dep - jnp.tile(y_slice, (1, n_networks))) ** 2, axis=0).tolist()
                )
            all_losses.append(rep_losses)

        return jnp.array(all_losses)

    def _run_single_trial(self, trial: optuna.Trial) -> float:
        """Run a single design trial."""
        hp = self._suggest_hyperparams(trial)
        dconf = self._design_conf
        assert dconf is not None, "_prepare() must be called before _run_single_trial()"

        try:
            pkey, bkey, loop_key, eval_key, tu_key = jax.random.split(
                jax.random.PRNGKey(hp['seed']), 5
            )

            params = deepcopy(self._initial_params)
            self._inject_hyperparams(params, hp)

            # Re-initialize TU params if hyperparams specify different init values
            if self._tu_masking_strategy.has_masking and (
                'tu_init_mean' in hp or 'tu_init_std' in hp
            ):
                mean = hp.get('tu_init_mean', dconf.tu_masking.init_mean)
                std = hp.get('tu_init_std', dconf.tu_masking.init_std)
                # Only handle direct log_alpha mode - other modes have more complex init
                if TU_LOG_ALPHA_PATH in params:
                    old_shape = params[TU_LOG_ALPHA_PATH].shape
                    new_log_alpha = mean + std * jax.random.normal(tu_key, shape=old_shape)
                    new_log_alpha = jnp.clip(new_log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX)
                    params.at(TU_LOG_ALPHA_PATH, new_log_alpha, overwrite=True)

            static, dynamic = params.filter_by_tag(["non_grad", "shared"])
            opt_state = vmap(vmap(dconf.optimizer.init))(dynamic)

            steps_per_epoch = max(1, dconf.n_batches_per_epoch // dconf.batches_per_step)

            xbatches_list, ybatches_list = self._dmanager.get_samples(
                (
                    len(self._dmanager.networks),
                    steps_per_epoch,
                    dconf.n_replicates,
                    dconf.batches_per_step,
                    dconf.batch_size,
                ),
                bkey,
            )
            xbatches = jnp.concatenate(xbatches_list, axis=-1)
            ybatches = ybatches_list[0]

            t0 = time.time()
            final_params, loss_history, step_history = optimize(
                self._cached_step,
                params,
                opt_state,
                xbatches=xbatches,
                ybatches=ybatches,
                config=dconf,
                n_total_steps=self._total_steps,
                steps_per_epoch=steps_per_epoch,
                key=loop_key,
                stack=self._cached_stack,
                loggers=None,
                verbose=False,
                precompiled=True,
            )
            train_time = time.time() - t0

            if self.use_eval_loss:
                xraw, yraw = sample_for_evaluation(
                    self._dmanager, dconf, final_params, self.eval_n_samples, eval_key
                )
                mse_losses = self._evaluate_cached(final_params, xraw, yraw, eval_key)
                rmse_per_network = np.sqrt(np.mean(np.array(mse_losses), axis=(0, 1)))
                top_k_indices = np.argsort(rmse_per_network)[: self.eval_top_k]
                loss = float(np.mean(rmse_per_network[top_k_indices]))
            else:
                loss = float(np.mean(loss_history[-1])) if loss_history else float('inf')

            if self.verbose:
                logger.info(f"Trial {trial.number}: loss={loss:.6f} ({train_time:.1f}s)")

            return loss

        except Exception as e:
            logger.exception(f"Trial {trial.number} failed: {e}")
            return float('inf')


async def _main_async():
    setup_logging()
    await DesignHyperoptProgram.cli()


def main():
    asyncio.run(_main_async())


if __name__ == '__main__':
    main()
