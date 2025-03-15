## {{{                          --     imports     --

from dracon.deferred import DeferredNode
from typing import List, Tuple, Callable
from biocomptools.plot import plot_extra_context
from biocomptools.plot import PlotJob
from biocomptools.toollib.loggers.logger import Logger
from dracon.utils import ser_debug
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class PlotLogger(Logger):
    jobs: List[DeferredNode[PlotJob]] = []

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        get_best_model = training_program.get_best_model_func()

        def plot_callback(step, training_config, step_history=None, **kwargs):
            logger.debug(f"\n==== PlotLogger callback at step {step} ====")
            best_model = None
            if step_history is not None:
                losses = step_history.get('loss')
                params = step_history.get('latest_params')
                best_model = get_best_model(params, losses)

                if best_model is not None:
                    logger.debug(f"Got best model with signature: {best_model.signature()}")

            logger.info(f'Plotting logger called at step {step}')
            logger.debug(f'Plotting logger has {len(self.jobs)} jobs')

            import dracon as dr
            from dracon.asizeof import asizeof

            for job in self.jobs:
                j = job.copy(reroot=True)

                # nr = dr.utils.node_repr(
                #     j,
                #     enable_colors=True,
                #     show_biggest_context=5,
                # )
                # print(nr)
                # print(f'Job size: {asizeof(j) / 1e6:.2f} MB')
                # print(f'Job.context.size: {asizeof(j.context) / 1e6:.2f} MB')
                # print(f'Job._full_composition.size: {asizeof(j._full_composition) / 1e6:.2f} MB')
                # print(f'Job._loader.size: {asizeof(j._loader) / 1e6:.2f} MB')
                # ser_debug(j._loader, 'sizeof', max_size_mb=1)
                # ser_debug(j._loader.context, 'sizeof', max_size_mb=1)
                # ser_debug(j._full_composition, 'sizeof', max_size_mb=1)

                try:
                    if best_model is not None:
                        constructed_job = j.construct(
                            context={
                                **plot_extra_context,
                                'best_model': best_model,
                                'step': step,
                            },
                        )
                        if not isinstance(constructed_job, PlotJob):
                            constructed_job = PlotJob(**constructed_job)
                        constructed_job.run()
                    else:
                        logger.info('Skipping prediction plots - no best model available yet')
                except Exception as e:
                    logger.error(f'Error plotting job: {e}')
                    logger.exception(e)
                    continue

        return [(self.periods, plot_callback)]
