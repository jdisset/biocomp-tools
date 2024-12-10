## {{{                          --     imports     --
import os

os.environ.setdefault('RAY_DEDUP_LOGS', '0')

from memory_profiler import profile

import logging.config
import sys
import time
import ray
from pathlib import Path
from typing import List, Annotated, Dict, Any, Optional
import dracon as dr
from dracon.lazy import LazyDraconModel, resolve_all_lazy
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode

from biocomp.utils import PartialFunction
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout
from biocomptools.toollib.datasources import DataSource, DBSource, NetworkPrediction
from biocomptools.toollib.common import config
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
from biocomptools.toollib.figuremakers.uorfmatrixfigure import uORFMatrixFigure, bundle_uorf_data

from biocomptools.toollib.networkselector import (
    NetworkSelector,
    Regex,
    NetworkSet,
    NetworkDataId,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
)

from dracon.utils import dict_like
from biocomptools.logging_config import get_logger, setup_logging

import warnings

# Suppress the fork warning from jax + ray (it's not a problem in this case)
warnings.filterwarnings("ignore", message="os.fork()", module="subprocess")

log = get_logger(__name__)

DEFAULT_TYPES = [
    Figure,
    PlotConfig,
    PlotTask,
    DataSource,
    DBSource,
    SimpleLayout,
    Regex,
    NetworkPrediction,
    NetworkSelector,
    NetworkSet,
    NetworkDataId,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    PartialFunction,
    DataRescaler,
    uORFMatrixFigure,
    bundle_uorf_data,
]

##────────────────────────────────────────────────────────────────────────────}}}


def make_context_from_types(types):
    return {t.__name__: t for t in types}


def get_pretty_axis_label(i: int, d: DataSource) -> str:
    if "pretty_inputs" in d.metadata:
        return f'$\\mathbf{{X_{i+1} ({d.input_names[i]}}})$\n{d.metadata["pretty_inputs"][i]}'
    return f'$\\mathbf{{X_{i+1}}}$ ({d.input_names[i]})'


def _make_figure(figure: DeferredNode[Figure], i: int, total: int, **kw):
    t0 = time.time()
    try:
        f = figure.construct(deferred_paths=['/figures.*.plot_tasks.*'])

        if dict_like(f):
            f = Figure(**f)  # type: ignore
        f.run(**kw)
    except Exception as e:
        log.error(f"Error making figure: {e}")
        # show the traceback
        log.exception(e)
        return

    import matplotlib.pyplot as plt
    import gc

    plt.close('all')
    gc.collect()

    t1 = time.time()
    opath = f.figure_spec.output_path
    print(f"[{i}/{total}] Completed {opath} in {t1 - t0:.2f}s")


@ray.remote
class PlotWorker:
    def __init__(self):
        setup_logging(default_level=logging.DEBUG)

    def process_figure(self, figure: DeferredNode[Figure], i: int, total: int, **kw):
        _make_figure(figure, i, total, **kw)
        return i


class PlotJob(LazyDraconModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworkers: Annotated[int, Arg(help='Number of workers (processes) to use')] = 8
    skip_existing: Annotated[bool, Arg(help='Overwrite existing figures')] = False

    def run(self):
        self._overwrite = not self.skip_existing
        total_figures = len(self.figures)
        log.info(f"Going to create {total_figures} figures using {self.nworkers} workers")

        if total_figures == 0:
            log.warning("No figures to create")
            return

        # Single figure or single worker case
        if total_figures == 1 or self.nworkers <= 1:
            log.info("Running in single-threaded mode")
            for i, fig in enumerate(self.figures):
                _make_figure(fig, i + 1, total_figures, overwrite=self._overwrite)
            return

        if not ray.is_initialized():
            ray.init(num_cpus=self.nworkers)

        workers = [PlotWorker.remote() for _ in range(self.nworkers)]

        pending_tasks = []
        unassigned_figures = list(enumerate(self.figures))

        # Initially fill the worker pool
        while workers and unassigned_figures:
            worker = workers.pop(0)
            idx, figure = unassigned_figures.pop(0)
            pending_tasks.append(
                (
                    worker,
                    worker.process_figure.remote(
                        figure, idx + 1, total_figures, overwrite=self._overwrite
                    ),
                )
            )

        while pending_tasks:
            # Wait for the next task to complete
            done_refs = [task_ref for _, task_ref in pending_tasks]
            done_id, _ = ray.wait(done_refs, num_returns=1)

            # Find which worker completed
            for i, (worker, task_ref) in enumerate(pending_tasks):
                if task_ref in done_id:
                    pending_tasks.pop(i)

                    # Assign new work to the freed worker if any remains
                    if unassigned_figures:
                        idx, figure = unassigned_figures.pop(0)
                        pending_tasks.append(
                            (
                                worker,
                                worker.process_figure.remote(
                                    figure, idx + 1, total_figures, overwrite=self._overwrite
                                ),
                            )
                        )
                    break

        ray.shutdown()


def main():
    setup_logging(default_level=logging.DEBUG)

    prog = make_program(
        PlotJob,
        name='biocomp-plot',
        description='Make plots.',
    )

    pj, args = prog.parse_args(
        sys.argv[1:],
        deferred_paths=[
            '/figures.*',
        ],

        context={
            **make_context_from_types(DEFAULT_TYPES),
            'get_pretty_axis_label': get_pretty_axis_label,
            'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
        },

    )

    dmp = dr.dump(pj)

    log.debug(dmp)

    pj.run()


if __name__ == '__main__':
    main()
