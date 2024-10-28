## {{{                          --     imports     --
import logging.config
from pydantic import BaseModel, Field, BeforeValidator
import sys
import ray
import matplotlib as mpl
from typing import List, Annotated, Dict, Any, Optional

import dracon as dr
from dracon.lazy import LazyDraconModel
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode


from biocomp.utils import PartialFunction, ArbitraryModel
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout
from biocomp import utils as ut


from biocomptools.toollib.datasources import DataSource, DBSource, NetworkPrediction
from biocomptools.toollib.common import maybetqdm
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
from biocomptools.toollib.networkselector import NetworkSelector, Regex

from biocomptools.logging_config import get_logger, setup_logging

log = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     PlotJob     --

DEFAULT_CONTEXT = {
    'Figure': Figure,
    'DBSource': DBSource,
    'NetworkPrediction': NetworkPrediction,
    'SimpleLayout': SimpleLayout,
    'Regex': Regex,
}


def _construct_and_run(figure):
    f = figure.construct(deferred_paths=['/figures.*.plot_tasks.*'])
    if isinstance(f, dict):
        f = Figure(**f)
    f.run()


@ray.remote
def _ray_construct_and_run(f):
    """Ray remote version of the construct and run function"""
    return _construct_and_run(f)


class PlotJob(LazyDraconModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]

    def run(self):
        # Initialize Ray if it hasn't been started
        if not ray.is_initialized():
            ray.init()

        # Create remote tasks for each figure
        remote_tasks = [_ray_construct_and_run.remote(fig) for fig in self.figures]

        # Wait for all tasks to complete
        ray.get(remote_tasks)


##────────────────────────────────────────────────────────────────────────────}}}


class profile:
    def __init__(self, output_file='profile_output.prof'):
        self.output_file = output_file

    def __enter__(self):
        import cProfile

        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()
        self.pr.dump_stats(self.output_file)


def main():
    prog = make_program(
        PlotJob,
        name='biocomp-plot',
        description='Make plots.',
        context=DEFAULT_CONTEXT,
    )
    pj, args = prog.parse_args(
        sys.argv[1:],
        deferred_paths=[
            '/figures.*',
        ],
        context={'DBSource': DBSource},
    )

    pj.run()


if __name__ == '__main__':
    setup_logging(default_level=logging.INFO)
    main()
