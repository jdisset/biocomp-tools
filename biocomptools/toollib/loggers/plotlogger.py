## {{{                          --     imports     --
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues in threads

from dracon.deferred import DeferredNode
from typing import List, Tuple, Callable
from biocomptools.plot import DEFAULT_CONTEXT
from biocomptools.plot import PlotJob
from biocomptools.toollib.loggers.logger import Logger
from dracon.utils import ser_debug
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class PlotLogger(Logger):
    jobs: List[DeferredNode[PlotJob]] = []
    parallel_ok: bool = True

    def initialize(self, training_program):
        """Store save_dir for later use in job construction."""
        super().initialize(training_program)
        # Store save_dir from training program for use in plot job contexts
        self._save_dir = getattr(training_program, '_save_dir', None)

    def do_plot(self, step, best_model, job_idx, job: DeferredNode[PlotJob]):
        j = job.copy(reroot=True)
        if best_model is not None:
            # Create minimal context for inner nodes plot to avoid training_set conflicts
            from biocomptools.toollib.figuremakers.innernodes import (
                InnerNodesFigure,
                InnerNodesFigureSpec,
            )
            from biocomp.plotutils import FigureSpec, FigureLayout

            minimal_context = {
                'InnerNodesFigure': InnerNodesFigure,
                'InnerNodesFigureSpec': InnerNodesFigureSpec,
                'FigureSpec': FigureSpec,
                'FigureLayout': FigureLayout,
                'best_model': best_model,
                'step': step,
                'save_dir': self._save_dir,
            }

            construction_context = minimal_context
            logger.debug(
                f'Constructing job {job_idx + 1} with context keys: {list(construction_context.keys())}'
            )
            logger.debug(
                f'Context step={step}, best_model signature={best_model.signature[:16]}..., save_dir={self._save_dir}'
            )
            try:
                constructed_job = j.construct(context=construction_context)
            except Exception as construct_error:
                logger.error(
                    f'Construction failed for job {job_idx + 1} at step {step}: {construct_error}'
                )
                logger.debug(f'Construction context keys: {list(construction_context.keys())}')
                logger.debug(f'Job being constructed: {str(j)[:1000]}')
                logger.exception(construct_error)
                return False
            logger.debug(
                f'Job {job_idx + 1} constructed successfully: {type(constructed_job).__name__}'
            )
            if not isinstance(constructed_job, PlotJob):
                constructed_job = PlotJob(**constructed_job)
            logger.debug(f'Running job {job_idx + 1}...')
            constructed_job.run()
            logger.debug(f'Job {job_idx + 1} completed successfully')
        else:
            logger.info('Skipping prediction plots - no best model available yet')

        return True

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

            for job_idx, job in enumerate(self.jobs):
                logger.debug(f'Processing job {job_idx + 1}/{len(self.jobs)}: {type(job).__name__}')
                logger.debug(f'Job content preview: {str(job)[:200]}...')
                N_RETRIES = 2
                while N_RETRIES > 0:
                    try:
                        if not self.do_plot(step, best_model, job_idx, job):
                            logger.warning(f'Job {job_idx + 1} failed to run, retrying...')
                            N_RETRIES -= 1
                        else:
                            logger.debug(f'Job {job_idx + 1} completed successfully')
                            break
                    except Exception as e:
                        logger.error(f'Error running job {job_idx + 1}: {e}')
                        logger.debug(f'Job details: {str(job)[:500]}')
                        logger.exception(e)
                        N_RETRIES -= 1

        return [(self.periods, plot_callback)]
