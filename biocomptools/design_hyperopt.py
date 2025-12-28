"""Hyperparameter optimization for design runs using CMA-ES.

Uses JAX-native schedule evaluation to avoid recompilation between trials.

Usage:
    biocomp-design-hyperopt +biocomp-jobs/hyperopt/design_lattice
"""

from __future__ import annotations
import json
import time
import asyncio
from pathlib import Path
from typing import Any, Annotated
import numpy as np
import optuna
from optuna.samplers import BaseSampler
from tqdm import tqdm
from pydantic import Field, BaseModel

from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode
import jax
import jax.numpy as jnp
from jax import vmap

from biocomp.design import (
    DesignManager,
    DesignConfig,
    TargetUnion,
    SamplingConfigUnion,
    UniformSampling,
    initialize_params,
    get_ratio_paths_and_sources,
    normalize_ratios_prune,
    normalize_ratio_source_arrays,
    sample_for_evaluation,
    evaluate_design,
)
from biocomp.designloss import HYPEROPT_SCHEDULE_NAMESPACE
from biocomp.recipe import Recipe
from biocomp.tumasking import TU_LOG_ALPHA_PATH, LOG_ALPHA_MIN, LOG_ALPHA_MAX
from biocomp.optimutils import make_training_step, per_replicate_step, optimize, compile_step
from biocomp.parameters import ParameterTree

from biocomptools.modelmodel import BiocompModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.toollib.common import config
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.optimtools import DEFAULT_TYPES
from biocomptools.optuna_hyperopt import HyperparamSpec

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
        nets = recipe_to_networks(recipe, model.library, unlock_ratios=True)
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
class DesignHyperoptProgram(BaseModel):
    """Design hyperparameter optimization using CMA-ES with recompilation-free trials."""

    study_name: Annotated[str, Arg(help='Study name for persistence')] = "design_hyperopt"
    output_dir: Annotated[str, Arg(help='Output directory')] = "./design_hyperopt_output"
    n_trials: Annotated[int, Arg(help='Number of trials')] = 100
    seed: Annotated[int | None, Arg(help='Random seed')] = None

    sampler: Annotated[str, Arg(help='Sampler: tpe, cmaes')] = "cmaes"
    n_startup_trials: int = 5

    cmaes_restart_strategy: str | None = "bipop"
    cmaes_popsize: int | None = 16
    cmaes_sigma0: float = 0.3
    cmaes_warm_start: bool = True
    cmaes_with_margin: bool = True
    cmaes_warn_independent_sampling: bool = False

    # Design config
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

    disable_tu_masking: Annotated[bool, Arg(help='Disable TU masking')] = False

    # Hyperparams
    hyperparams: list[HyperparamSpec] = []

    # Evaluation config
    eval_n_samples: Annotated[int, Arg(help='Samples for RMSE evaluation')] = 10000
    eval_top_k: Annotated[int, Arg(help='Best K networks to average')] = 3
    use_eval_loss: Annotated[bool, Arg(help='Use RMSE eval instead of training loss')] = True

    # Modes
    show_best: bool = False
    dashboard: bool = False
    dashboard_port: int = 8080
    n_top_models: int = 5
    verbose: bool = True

    # Internal state
    _model: Any = None
    _dmanager: Any = None
    _cached_step: Any = None
    _cached_stack: Any = None
    _initial_params: Any = None
    _static_params: Any = None
    _ratio_paths: list | None = None
    _total_steps: int | None = None

    model_config = {'arbitrary_types_allowed': True}

    def model_post_init(self, _):
        if isinstance(self.targets, TargetUnion):
            self.targets = [self.targets]
        if isinstance(self.scaffolds, Recipe):
            self.scaffolds = [self.scaffolds]

    @property
    def _storage_path(self) -> str:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{Path(self.output_dir) / (self.study_name + '.db')}"

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
            enable_tu_masking=not self.disable_tu_masking,
        )
        logger.info(
            f"DesignManager ready: {len(self._dmanager.networks)} networks, "
            f"{self._dmanager.n_targets} targets"
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

        n_tus = self._dmanager.n_tus if self._dmanager.enable_tu_masking else 0
        n_networks = len(self._dmanager.networks)

        initial_params = initialize_params(
            stack,
            dconf.n_replicates,
            self._dmanager.n_targets,
            self._model.shared_params,
            pkey,
            n_tus=n_tus,
            n_networks=n_networks,
            tu_log_alpha_init_mean=dconf.tu_log_alpha_init_mean,
            tu_log_alpha_init_std=dconf.tu_log_alpha_init_std,
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
            if TU_LOG_ALPHA_PATH in params:
                params = params.update_leaves_by_path(
                    [TU_LOG_ALPHA_PATH], lambda x: jnp.clip(x, LOG_ALPHA_MIN, LOG_ALPHA_MAX)
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

        logger.info(f"Design step compiled in {time.time() - t0:.1f}s")

    def _inject_hyperparams(self, params: ParameterTree, hp: dict) -> ParameterTree:
        """Inject hyperparameters into the params tree as schedule parameters.

        Hyperparams named like 'lambda_l0_phase1_value' are injected as:
          hyperopt_schedules/lambda_l0_phase1_value

        Global phase fracs are applied to all schedules.
        """
        ns = HYPEROPT_SCHEDULE_NAMESPACE
        phase1_frac = hp.get('phase1_frac', 0.4)
        phase2_frac = hp.get('phase2_frac', 0.75)

        schedule_names = set()
        for name, value in hp.items():
            if name in ('phase1_frac', 'phase2_frac', 'seed'):
                continue
            for suffix in (
                '_phase1_value',
                '_phase2_end_value',
                '_phase3_end_value',
                '_phase1_frac',
                '_phase2_frac',
            ):
                if name.endswith(suffix):
                    sched_name = name[: -len(suffix)]
                    schedule_names.add(sched_name)
                    path = f"{ns}/{name}"
                    params = params.at(
                        path,
                        jnp.array(value, dtype=jnp.float32),
                        tags=["non_grad", "hyperopt"],
                        overwrite=True,
                    )
                    break
            else:
                path = f"{ns}/{name}_phase1_value"
                params = params.at(
                    path,
                    jnp.array(value, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )
                params = params.at(
                    f"{ns}/{name}_phase2_end_value",
                    jnp.array(value, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )
                params = params.at(
                    f"{ns}/{name}_phase3_end_value",
                    jnp.array(value, dtype=jnp.float32),
                    tags=["non_grad", "hyperopt"],
                    overwrite=True,
                )
                schedule_names.add(name)

        for sched_name in schedule_names:
            params = params.at(
                f"{ns}/{sched_name}_phase1_frac",
                jnp.array(phase1_frac, dtype=jnp.float32),
                tags=["non_grad", "hyperopt"],
                overwrite=True,
            )
            params = params.at(
                f"{ns}/{sched_name}_phase2_frac",
                jnp.array(phase2_frac, dtype=jnp.float32),
                tags=["non_grad", "hyperopt"],
                overwrite=True,
            )

        return params

    def _run_single_trial(self, trial: optuna.Trial, dconf: DesignConfig) -> float:
        """Run a single design trial."""
        hp = {spec.name: spec.suggest(trial) for spec in self.hyperparams}
        hp['seed'] = self.seed if self.seed else trial.number

        try:
            pkey, bkey, loop_key, eval_key = jax.random.split(
                jax.random.PRNGKey(hp['seed']), 4
            )

            params = self._inject_hyperparams(self._initial_params, hp)

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
            final_params, history = optimize(
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
            )
            train_time = time.time() - t0

            if self.use_eval_loss:
                xraw, yraw = sample_for_evaluation(
                    self._dmanager, dconf, final_params, self.eval_n_samples, eval_key
                )
                _, mse_losses = evaluate_design(
                    self._dmanager,
                    dconf,
                    self._model,
                    final_params,
                    xraw,
                    yraw,
                    eval_key,
                    store_predictions=False,
                )
                rmse_per_network = np.sqrt(np.mean(np.array(mse_losses), axis=(0, 1)))
                top_k_indices = np.argsort(rmse_per_network)[: self.eval_top_k]
                loss = float(np.mean(rmse_per_network[top_k_indices]))
            else:
                losses = np.array(history[-1]['all_losses'])
                loss = float(np.mean(losses))

            if self.verbose:
                logger.info(f"Trial {trial.number}: loss={loss:.6f} ({train_time:.1f}s)")

            return loss

        except Exception as e:
            logger.exception(f"Trial {trial.number} failed: {e}")
            return float('inf')

    def _create_sampler(self, n_complete: int = 0, source_trials=None) -> BaseSampler:
        """Create Optuna sampler."""
        from biocomptools.hyperopt.samplers import create_sampler

        return create_sampler(
            sampler_type=self.sampler,
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            cmaes_restart_strategy=self.cmaes_restart_strategy,
            cmaes_with_margin=self.cmaes_with_margin,
            cmaes_popsize=self.cmaes_popsize,
            cmaes_sigma0=self.cmaes_sigma0,
            cmaes_source_trials=source_trials,
            cmaes_warn_independent_sampling=self.cmaes_warn_independent_sampling,
        )

    def _load_existing(self) -> optuna.Study | None:
        try:
            return optuna.load_study(study_name=self.study_name, storage=self._storage_path)
        except KeyError:
            return None

    def _get_completed_trials(self, study: optuna.Study | None) -> list:
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
                    logger.info(f"Best params: {study.best_params}")
            return

        self._prepare_design_manager()

        dconf = (
            self.design_conf.construct(context={})
            if isinstance(self.design_conf, DeferredNode)
            else self.design_conf
        )

        self._compile_design_step(dconf)

        completed = self._get_completed_trials(self._load_existing())
        sampler = self._create_sampler(len(completed), completed if self.cmaes_warm_start else None)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self._storage_path,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )

        pbar = tqdm(total=self.n_trials, desc="Design Hyperopt")

        def callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pbar.update()
                pbar.set_postfix(best=f"{study.best_value:.4f}")

        study.optimize(
            lambda trial: self._run_single_trial(trial, dconf),
            n_trials=self.n_trials,
            callbacks=[callback],
        )
        pbar.close()

        self._save_results(study)

        logger.info(f"\nBest trial: #{study.best_trial.number}")
        logger.info(f"Best loss: {study.best_value:.6f}")
        logger.info(f"Best params: {json.dumps(study.best_params, indent=2)}")


async def _main_async():
    setup_logging()
    await DesignHyperoptProgram.cli()


def main():
    asyncio.run(_main_async())


if __name__ == '__main__':
    main()
