## {{{                          --     imports     --

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from dracon.utils import ser_debug
import dracon as dr
from pympler.asizeof import asizeof

import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
from typing import List, Annotated
from dracon.lazy import LazyDraconModel
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode

from biocomp.utils import PartialFunction
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout
from biocomptools.toollib.datasources import DataSource, DBSource

from biocomptools.toollib.networkprediction import NetworkPrediction

from biocomptools.toollib.common import config
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
from biocomptools.toollib.figuremakers.uorfmatrixfigure import uORFMatrixFigure, bundle_uorf_data
from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure, InnerNodesFigureSpec
import gc

from biocomptools.toollib.networkselector import (
    NetworkSelector,
    Regex,
    NetworkSet,
    NetworkDataId,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    NetworkFilter,
    CleanupFilter,
    CustomFilter,
    UorfFilter,
)

import numpy as numpy
from dracon.utils import dict_like

import pathos.multiprocessing as mp
import pickle

from numpy import (
    ndarray,
    linspace,
    array,
    arange,
    meshgrid,
    zeros,
    ones,
    full,
    empty,
    empty_like,
    full_like,
    zeros_like,
    ones_like,
    eye,
    diag,
    diagflat,
    triu,
    tril,
    vander,
    histogram,
    histogram2d,
    digitize,
)


import warnings

# suppress the fork warning from jax
warnings.filterwarnings("ignore", message="os.fork()", module="subprocess")

logger = get_logger(__name__)

DEFAULT_TYPES = [
    Figure,
    PlotConfig,
    PlotTask,
    DataSource,
    DBSource,
    FigureSpec,
    FigAx,
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
    InnerNodesFigure,
    InnerNodesFigureSpec,
    bundle_uorf_data,
    NetworkFilter,
    CleanupFilter,
    CustomFilter,
    UorfFilter,
    ndarray,
    linspace,
    array,
    arange,
    meshgrid,
    zeros,
    ones,
    full,
    empty,
    empty_like,
    full_like,
    zeros_like,
    ones_like,
    eye,
    diag,
    diagflat,
    triu,
    tril,
    vander,
    histogram,
    histogram2d,
    digitize,
]


def _init_worker():
    """Initialize worker process with appropriate environment variables."""
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    setup_logging()


def make_context_from_types(types):
    return {t.__name__: t for t in types}


##────────────────────────────────────────────────────────────────────────────}}}


def get_pretty_axis_label(i: int, d: DataSource) -> str:
    if "pretty_inputs" in d.metadata:
        return f'$\\mathbf{{X_{i+1} ({d.input_names[i]}}})$\n{d.metadata["pretty_inputs"][i]}'
    return f'$\\mathbf{{X_{i+1}}}$ ({d.input_names[i]})'


def _make_figure(figure_data):
    figure, i, total, kw = figure_data
    t0 = time.time()

    try:
        logger.debug(f"Making figure {i}/{total}")
        t_copy_start = time.time()

        f = figure.construct()

        if dict_like(f):
            logger.debug(f"Figure {i}/{total} is a dict")
            f = Figure(**f)  # type: ignore

        t_copy_end = time.time()
        logger.debug(f"Figure {i} construction took {t_copy_end - t_copy_start:.2f}s")
        assert isinstance(f, Figure), f"Expected Figure, got {type(f)}"

        f.run(**kw)
    except Exception as e:
        logger.error(f"Error making figure: {e}")
        logger.exception(e)
        return

    plt.close('all')

    t1 = time.time()
    opath = f.figure_spec.output_path
    logger.debug(f"[{i}/{total}] Figure {opath} completed in {t1 - t0:.2f}s")
    return i


class PlotJob(BaseModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworkers: Annotated[int, Arg(help='Number of workers (processes) to use')] = 8
    skip_existing: Annotated[bool, Arg(help='Overwrite existing figures')] = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self):
        self._overwrite = not self.skip_existing
        total_figures = len(self.figures)
        logger.debug(f"Going to create {total_figures} figures")

        t0 = time.time()
        from biocomptools.run_training import TrainingProgram

        for fig in self.figures:
            ser_debug(fig, 'dill')
            ser_debug(fig, 'deepcopy')
            ser_debug(fig, 'sizeof', max_size_mb=200)
            nr = dr.utils.node_repr(
                fig,
                enable_colors=True,
                context_filter=lambda k, v: isinstance(v, TrainingProgram),
                show_biggest_context=3,
            )
            print(nr)

        if self.nworkers <= 1:
            for i, fig in enumerate(self.figures):
                f = fig.copy(reroot=True)
                _make_figure((f, i + 1, total_figures, {"overwrite": self._overwrite}))

        # using pathos.multprocessing:
        else:
            logger.debug(f"Using {self.nworkers} workers")
            # unordered parallel processing
            with mp.ProcessingPool(self.nworkers, initializer=_init_worker) as pool:
                results = pool.map(
                    _make_figure,
                    [
                        (f, i + 1, total_figures, {"overwrite": self._overwrite})
                        for i, f in enumerate(self.figures)
                    ],
                )

        t1 = time.time()
        logger.info(f"Plot job of {total_figures} figures completed in {t1 - t0:.2f}s")

        return


plot_extra_context = {
    **make_context_from_types(DEFAULT_TYPES),
    'get_pretty_axis_label': get_pretty_axis_label,
    'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
}


def main():
    setup_logging()

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
        context=plot_extra_context,
    )

    pj.run()


if __name__ == '__main__':
    main()
