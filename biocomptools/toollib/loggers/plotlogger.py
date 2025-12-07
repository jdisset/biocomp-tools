import matplotlib
matplotlib.use('Agg')

from dracon.deferred import DeferredNode
from typing import List, Tuple, Callable
from biocomptools.plot import PlotJob
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class PlotLogger(Logger):
    jobs: List[DeferredNode[PlotJob]] = []
    parallel_ok: bool = True
    extra_context: dict = {}

    def initialize(self, training_program):
        super().initialize(training_program)
        self._save_dir = getattr(training_program, '_save_dir', None)

    def do_plot(self, step, best_model, job_idx, job: DeferredNode[PlotJob]):
        j = job.copy(reroot=True)
        if best_model is None:
            logger.info('Skipping plot - no best model available')
            return True

        from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure, InnerNodesFigureSpec
        from biocomptools.toollib.figuremakers.benchmarkutils import (
            BenchmarkData, BenchmarkItem, render_summary_header, render_metrics_panel,
            generate_benchmark_summary,
        )
        from biocomp.plotutils import FigureSpec, FigureLayout

        ctx = {
            'InnerNodesFigure': InnerNodesFigure, 'InnerNodesFigureSpec': InnerNodesFigureSpec,
            'BenchmarkData': BenchmarkData, 'BenchmarkItem': BenchmarkItem,
            'render_summary_header': render_summary_header, 'render_metrics_panel': render_metrics_panel,
            'generate_benchmark_summary': generate_benchmark_summary,
            'FigureSpec': FigureSpec, 'FigureLayout': FigureLayout,
            'best_model': best_model, 'step': step, 'save_dir': self._save_dir,
            **self.extra_context,
        }

        try:
            logger.info(f'Job {job_idx + 1}: constructing deferred node')
            constructed = j.construct(context=ctx)
            if not isinstance(constructed, PlotJob):
                logger.info(f'Job {job_idx + 1}: converting dict to PlotJob')
                constructed = PlotJob(**constructed)
            logger.info(f'Job {job_idx + 1}: running PlotJob with {len(constructed.figures)} figures')
            constructed.run()
            logger.info(f'Job {job_idx + 1} completed successfully')
        except Exception as e:
            logger.error(f'Job {job_idx + 1} failed: {e}')
            logger.exception(e)
            return False
        return True

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        get_best_model = training_program.get_best_model_func()

        def plot_callback(step, training_config, step_history=None, **kwargs):
            best_model = None
            if step_history is not None:
                losses = step_history.get('loss')
                params = step_history.get('latest_params')
                best_model = get_best_model(params, losses)

            logger.info(f'PlotLogger at step {step}, {len(self.jobs)} jobs')

            for job_idx, job in enumerate(self.jobs):
                for attempt in range(2):
                    try:
                        if self.do_plot(step, best_model, job_idx, job):
                            break
                    except Exception as e:
                        logger.error(f'Job {job_idx + 1} attempt {attempt + 1} failed: {e}')

        return [(self.periods, plot_callback)]
