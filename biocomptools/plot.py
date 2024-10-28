## {{{                          --     imports     --
import os

os.environ['RAY_DEDUP_LOGS'] = '0'
import logging.config
import sys
import time
import ray
from typing import List, Annotated, Dict, Any, Optional
from dracon.lazy import LazyDraconModel
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode

from biocomp.utils import PartialFunction
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout
from biocomptools.toollib.datasources import DataSource, DBSource, NetworkPrediction
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
from biocomptools.toollib.networkselector import (
    NetworkSelector,
    Regex,
    NetworkSet,
    NetworkDataId,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
)

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
]

##────────────────────────────────────────────────────────────────────────────}}}


def make_context_from_types(types):
    return {t.__name__: t for t in types}


def _construct_and_run(figure: DeferredNode[Figure], i, total):
    t0 = time.time()
    f = figure.construct(deferred_paths=['/figures.*.plot_tasks.*'])
    if isinstance(f, dict):
        f = Figure(**f)
    f.run()
    t1 = time.time()
    opath = f.figure_spec._output_path
    print(f"[{i}/{total}] Completed: {opath} in {t1 - t0:.2f}s")


@ray.remote
def make_figure(f, i, total):
    setup_logging(default_level=logging.INFO)
    _construct_and_run(f, i, total)


class PlotJob(LazyDraconModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworker: Annotated[int, Arg(help='Number of Ray workers to use')] = 8

    def run(self):
        if not ray.is_initialized():
            ray.init(num_cpus=self.nworker)

        total_figures = len(self.figures)
        remote_tasks = [
            make_figure.remote(fig, i + 1, total_figures) for i, fig in enumerate(self.figures)
        ]

        ray.get(remote_tasks)


def main():
    setup_logging(default_level=logging.INFO)

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
        context=make_context_from_types(DEFAULT_TYPES),
    )

    pj.run()


if __name__ == '__main__':
    main()
