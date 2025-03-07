## {{{                          --     imports     --

import dracon as dr
from dracon.deferred import DeferredNode
from typing import Dict, List, Optional, Tuple, Callable, Union, Annotated, Literal, TypeVar
from biocomptools.plot import plot_extra_context
from biocomptools.plot import PlotJob
from biocomptools.toollib.loggers.logger import Logger

from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class PlotLogger(Logger):
    jobs: List[DeferredNode[PlotJob]] = []

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        def plot_callback(step, training_config, step_history=None, **kwargs):
            logger.debug(f"\n==== PlotLogger callback at step {step} ====")
            best_model = None
            if step_history is not None:
                losses = step_history.get('loss')
                params = step_history.get('latest_params')
                best_model = training_program.get_best_model(params, losses)

                if best_model is not None:
                    logger.debug(f"Got best model with signature: {best_model.signature()}")

            logger.info(f'Plotting at step {step}')

            for job in self.jobs:
                j = job.copy()
                jstr = dr.node_repr(j, context_paths=['**.d', '**.all_predicted_data'])
                try:
                    if best_model is not None:
                        constructed_job = j.construct(
                            context={
                                **plot_extra_context,
                                'training_program': training_program,
                                'best_model': best_model,
                                'step': step,
                            },
                            # deferred_paths=['/**.figures.*'],
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
