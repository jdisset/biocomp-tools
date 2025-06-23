## {{{                          --     imports     --
import memray

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
    
    def initialize(self, training_program):
        """Store save_dir for later use in job construction."""
        super().initialize(training_program)
        # Store save_dir from training program for use in plot job contexts
        self._save_dir = getattr(training_program, '_save_dir', None)

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        get_best_model = training_program.get_best_model_func()

        def plot_callback(
            step,
            training_config,
            step_history=None,
            stack=None,
            xbatches=None,
            ybatches=None,
            **kwargs,
        ):
            # with memray.Tracker("./plotlogger_memray_profile.bin"):
            logger.debug(f"\n==== PlotLogger callback at step {step} ====")
            best_model = None
            if step_history is not None:
                losses = step_history.get('loss')
                params = step_history.get('latest_params')
                best_model = get_best_model(params, losses)

                if best_model is not None:
                    logger.debug(f"Got best model with signature: {best_model.signature}")

            logger.info(f'Plotting logger called at step {step}')
            logger.debug(f'Plotting logger has {len(self.jobs)} jobs')

            import dracon as dr
            from dracon.asizeof import asizeof

            for job in self.jobs:
                j = job.copy(reroot=True)
                try:
                    if best_model is not None:
                        constructed_job = j.construct(
                            context={
                                **plot_extra_context,
                                'best_model': best_model,
                                'step': step,
                                'save_dir': self._save_dir,  # Include save_dir in context
                                # 'step_history': step_history,
                                # 'stack': stack,
                                # 'xbatches': xbatches,
                                # 'ybatches': ybatches,
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
