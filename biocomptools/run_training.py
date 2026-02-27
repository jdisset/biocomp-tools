from biocomptools.optimtools import (
    BaseOptimizationProgram,
    run_optimization_program,
    Logger,
)
from biocomptools.modelmodel import BiocompModel, get_shared_params
from biocomptools.trainutils import (
    get_best_smoothed_loss_replicate_id,
    get_latest_avg_loss,
    make_json_ready,
)
from biocomptools.logging_config import get_logger
from biocomptools.toollib.networkselector import build_data_manager, NetworkSet

from biocomp.compute import ComputeConfig, DEFAULT_COMPUTE_CONFIG
from biocomp.datautils import DataConfig, DEFAULT_DATA_CONFIG
from biocomp.train import TrainingConfig
from functools import partial

from dracon.commandline import Arg
import asyncio

import sys
import numpy as np
from typing import Annotated, Any, Callable, Optional
from pydantic import Field

logger = get_logger(__name__)


def _finite_fraction(tree) -> float:
    from jax.tree_util import tree_leaves

    total = 0
    finite = 0
    for leaf in tree_leaves(tree):
        arr = np.asarray(leaf)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        total += arr.size
        finite += int(np.isfinite(arr).sum())
    if total == 0:
        return 0.0
    return finite / total


def _extract_replicate_params(all_params, replicate_id: int):
    from biocomp.jaxutils import tree_get

    candidates = []
    seen = set()
    for idx in [replicate_id, (replicate_id,), (-1, replicate_id), (replicate_id, -1), (-1,)]:
        try:
            key = str(idx)
            if key in seen:
                continue
            seen.add(key)
            cand = tree_get(all_params, idx)
            score = _finite_fraction(cand)
            candidates.append((score, idx, cand))
        except Exception:
            continue

    if not candidates:
        return None
    score, idx, params = max(candidates, key=lambda x: x[0])
    logger.debug(f"Selected params index {idx} with finite fraction {score:.4f}")
    return params


class TrainingProgram(BaseOptimizationProgram):
    training_conf: Annotated[TrainingConfig, Arg(help='Training config')] = Field(
        default_factory=lambda: TrainingConfig()
    )
    compute_conf: Annotated[ComputeConfig, Arg(help='Compute config')] = DEFAULT_COMPUTE_CONFIG
    data_conf: Annotated[DataConfig, Arg(help='Data config')] = DEFAULT_DATA_CONFIG

    training_set: Annotated[NetworkSet, Arg(help='Networks in training set')] = Field(
        default_factory=NetworkSet
    )

    use_jax_sampling: bool = True
    uniform_weights: Annotated[bool, Arg(help='Override all dataset weights to 1.0')] = False

    _training_dman: Any = None
    _training_id: Optional[str] = None

    @property
    def training_id(self) -> str:
        if self._training_id is None:
            self._training_id = self.unique_id
        return self._training_id

    def get_output_subdir(self) -> str:
        return 'training'

    def initialize_context(self):
        with self.db_session as session:
            self.training_set.run_selectors(session)
            session.expunge_all()
            session.close()

    def _get_logger_context(self) -> dict:
        context = super()._get_logger_context()
        context.update(
            {
                'training_conf': self.training_conf,
                'compute_conf': self.compute_conf,
                'data_conf': self.data_conf,
                'training_set': self.training_set,
            }
        )
        return context

    def _build_dman(self):
        self._training_dman = build_data_manager(
            lib=self.parts_library,
            db_session=self.db_session,
            path_prefix=self.path_prefix,
            data_conf=self.data_conf,
            dataset=self.training_set,
        )
        self._training_dman.jax_sampling = self.use_jax_sampling
        if self.uniform_weights:
            n = len(self._training_dman.get_networks())
            self._training_dman.set_weights([1.0] * n)
            logger.info(f"Uniform weights: set all {n} network weights to 1.0")

    def enrich_metadata(self):
        self._build_dman()
        assert self._training_dman is not None
        dman = self._training_dman

        dataman_info = {
            "network_names": [n.name for n in dman.get_networks()],
            "input_dimensions": [x.shape[1] for x in dman.get_X()],
            "output_dimensions": [y.shape[1] for y in dman.get_Y()],
            "data_config": dman.data_cfg.model_dump(),
        }

        self._metadata.update(
            {
                'training_id': self.training_id,
                'run_name': self._run_name,
                'experiment_name': self.experiment_name,
                'training_set': {
                    'content': self.training_set.content,
                    'name': self.training_set.name,
                },
                'training_conf': self.training_conf,
                'compute_conf': self.compute_conf,
                'data_conf': self.data_conf,
                'data_manager_info': dataman_info,
                'final_model_dump': self._modeldump,
            }
        )

    async def execute_optimization(self, dispatch):
        from biocomp.train import start

        all_params, all_losses, step_history = start(
            self._training_dman,
            self.training_conf,
            self.compute_conf,
            dispatch=dispatch,
        )

        return all_params, all_losses, step_history

    def save_outputs(self, all_params, all_losses, step_history=None):
        save_dir = self._save_dir / self.get_output_subdir()

        try:
            from jax.tree_util import tree_leaves

            leaves = tree_leaves(all_params)
            if leaves:
                arr = np.asarray(leaves[0])
                logger.info(
                    f"save_outputs: all_params leaf0 shape={arr.shape}, dtype={arr.dtype}"
                )
        except Exception:
            logger.info(f"save_outputs: all_params type={type(all_params)}")
        if "debug" in self._run_name:
            import pickle

            with open(save_dir / "debug_all_params.pkl", "wb") as f:
                pickle.dump(all_params, f)

        self.save_best(all_params, all_losses, save_dir)

        np.save(save_dir / 'loss_history.npy', all_losses)

        logger_metrics = [
            m
            for m in (
                lg.get_metrics(replicate=None) for lg in self.loggers if isinstance(lg, Logger)
            )
            if m
        ]
        if logger_metrics:
            self._metadata['logger_metrics_all_replicates'] = make_json_ready(logger_metrics)

        try:
            self.save_loss_plot(all_losses, save_dir)
        except Exception as e:
            logger.error(f"Failed to save loss plot: {e}")
        self.save_metadata(save_dir)

    def get_replicate_model_func(self):
        from copy import deepcopy

        compute_conf = deepcopy(self.compute_conf)
        data_conf = deepcopy(self.data_conf)
        base_metadata = make_json_ready(self._metadata)

        return partial(
            create_replicate_model,
            compute_conf=compute_conf,
            rescaler=data_conf.rescaler,
            base_metadata=base_metadata,
            loggers=self.loggers,
        )

    def get_best_model_func(self):
        replicate_model_factory = self.get_replicate_model_func()
        return partial(get_best_model, model_factory=replicate_model_factory)

    def save_best(self, all_params, all_losses: list[np.ndarray], save_dir, name=None):
        model_factory = self.get_best_model_func()
        model = model_factory(all_params=all_params, all_losses=all_losses)
        if model is None:
            logger.warning("Falling back to replicate 0 model export.")
            model = create_replicate_model(
                all_params=all_params,
                all_losses=all_losses,
                replicate_id=0,
                compute_conf=self.compute_conf,
                rescaler=self.data_conf.rescaler,
                base_metadata=make_json_ready(self._metadata),
                loggers=self.loggers,
            )
        if model is None:
            logger.error("!!!!!! No best model found !!!!!")
            return
        if name is None:
            name = f"{model.signature}.bestmodel"
        fname = save_dir / f'{name}.pickle'
        model.save(fname)
        logger.debug(f"Saved best model to {fname}")


def create_replicate_model(
    all_params, all_losses, replicate_id, compute_conf, rescaler, base_metadata, loggers
):
    from biocomp.jaxutils import tree_to_np
    import pickle

    params = _extract_replicate_params(all_params, replicate_id)
    # Fallback for already-sliced parameter trees (no leading replicate axis).
    if params is None and replicate_id == 0:
        params = all_params
        logger.debug("Falling back to full all_params for replicate-0 export.")
    if params is None:
        logger.warning(f"No parameters found for replicate {replicate_id}.")
        return None

    shared_params = get_shared_params(pickle.loads(pickle.dumps(params)))
    if shared_params is None:
        return None

    local_metadata = base_metadata.copy()
    local_metadata['replicate_number'] = replicate_id

    latest_loss = get_latest_avg_loss(all_losses, replicate_id)
    if not np.isnan(latest_loss):
        local_metadata['training_loss'] = latest_loss

    rep_metrics = [
        m for m in (logger.get_metrics(replicate=replicate_id) for logger in loggers) if m
    ]
    if rep_metrics:
        local_metadata['logger_metrics'] = make_json_ready(rep_metrics)

    model = BiocompModel(
        compute_config=compute_conf,
        rescaler=rescaler,
        shared_params=tree_to_np(shared_params),
        metadata=make_json_ready(local_metadata),
    )
    return model


def get_best_model(all_params, all_losses, model_factory: Callable):
    if not isinstance(all_losses, list):
        all_losses = [all_losses]

    if all_params is not None and len(all_losses) > 0:
        try:
            first_loss = all_losses[0]
            if hasattr(first_loss, 'shape') and len(first_loss.shape) > 0:
                n_replicates_from_loss = first_loss.shape[0]

                if hasattr(all_params, 'iter_leaves'):
                    for _path, param in all_params.iter_leaves():
                        if hasattr(param, 'shape') and len(param.shape) > 0:
                            n_replicates_from_params = param.shape[0]
                            if n_replicates_from_params != n_replicates_from_loss:
                                logger.warning(
                                    f"Dimension mismatch: params have {n_replicates_from_params} replicates, losses have {n_replicates_from_loss}"
                                )
                            break
        except Exception as e:
            logger.debug(f"Could not check parameter dimensions: {e}")

    best_model_id, _, _ = get_best_smoothed_loss_replicate_id(all_losses)
    if best_model_id == -1:
        n_replicates = 0
        if hasattr(all_params, 'iter_leaves'):
            try:
                for _path, param in all_params.iter_leaves():
                    if hasattr(param, 'shape') and len(param.shape) > 0:
                        n_replicates = int(param.shape[0])
                        break
            except Exception as e:
                logger.debug(f"Could not infer replicate count from params: {e}")

        if n_replicates > 0:
            logger.warning(
                "Could not determine best model from losses. Falling back to replicate 0."
            )
            best_model_id = 0
        else:
            logger.warning("Could not determine best model.")
            return None

    logger.debug(f"Best model is replicate number {best_model_id}")
    return model_factory(all_params=all_params, all_losses=all_losses, replicate_id=best_model_id)


async def main_async():
    await run_optimization_program(
        TrainingProgram,
        'biocomp-train',
        'Start training biocomp models.',
        sys.argv[1:],
    )


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
