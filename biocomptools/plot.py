## {{{                          --     imports     --

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from tqdm import tqdm
from biocomptools.toollib.common import maybetqdm

import dracon as dr
from dracon.utils import ser_debug
from pympler.asizeof import asizeof
import os

import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
from typing import List, Annotated, Literal
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


def make_context_from_types(types):
    return {t.__name__: t for t in types}


##────────────────────────────────────────────────────────────────────────────}}}


def debug_figures(figures):
    for fig in figures:
        ser_debug(fig, 'dill')
        ser_debug(fig, 'deepcopy')
        # ser_debug(fig, 'sizeof', max_size_mb=200)
        nr = dr.utils.node_repr(
            fig,
            enable_colors=True,
            show_biggest_context=5,
        )
        print(nr)


def get_pretty_axis_label(i: int, d: DataSource) -> str:
    if "pretty_inputs" in d.metadata:
        return f'$\\mathbf{{X_{i+1} ({d.input_names[i]}}})$\n{d.metadata["pretty_inputs"][i]}'
    return f'$\\mathbf{{X_{i+1}}}$ ({d.input_names[i]})'


def _make_figure(figure_data):
    figure, i, total, kw = figure_data
    t0 = time.time()

    try:
        t_copy_start = time.time()

        f = figure.construct()

        if dict_like(f):
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
    logger.info(f"[{i}/{total}] Figure {opath} completed in {t1 - t0:.2f}s")
    return i, opath


class PlotJob(BaseModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworkers: Annotated[int, Arg(help='Number of workers (processes) to use')] = 8
    skip_existing: Annotated[bool, Arg(help='Overwrite existing figures')] = False
    parallel_mode: Annotated[
        Literal['multiprocess', 'ray', 'none'],
        Arg(help='Parallel mode to use (multiprocess, ray, none)'),
    ] = 'ray'

    clear_figure_context_keys: Annotated[
        List[str], Arg(help='Clear these keys from the figure context')
    ] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self):
        overwrite = not self.skip_existing
        total_figures = len(self.figures)
        logger.debug(f"Going to create {total_figures} figures")

        for fig in self.figures:
            for k in self.clear_figure_context_keys:
                if k in fig.context:
                    del fig.context[k]

        t0 = time.time()
        fig_copies = [
            f.copy(reroot=True)
            for f in maybetqdm(self.figures, min_len=20, desc='Copying figure tasks')
        ]
        tcopy = time.time()
        logger.debug(f"Figure copies took {tcopy - t0:.2f}s")

        if self.nworkers <= 1 or self.parallel_mode == 'none' or total_figures <= 1:
            for i, f in enumerate(fig_copies):
                _make_figure((f, i + 1, total_figures, {"overwrite": overwrite}))

        elif self.parallel_mode == 'multiprocess':
            logger.debug(f"Using multiprocess with {self.nworkers} workers")
            import multiprocess as mp
            import os

            def _init_worker():
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                os.environ["JAX_PLATFORMS"] = "cpu"
                setup_logging()

            mp.set_start_method('spawn', force=True)  # type: ignore
            with mp.Pool(self.nworkers, initializer=_init_worker) as pool:
                import numpy

                random_order = numpy.random.permutation(total_figures)

                results = pool.map(
                    _make_figure,
                    [
                        (fig_copies[i], i + 1, total_figures, {"overwrite": overwrite})
                        for i in random_order
                    ],
                )

        elif self.parallel_mode == 'ray':
            logger.debug(f"Using ray with {self.nworkers} workers")
            import ray
            import memray

            time_ray_start = time.time()

            if not ray.is_initialized():
                context = ray.init(
                    num_cpus=self.nworkers,
                    runtime_env={
                        'env_vars': {
                            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                            'JAX_PLATFORMS': 'cpu',
                        }
                    },
                )
                logger.info(f"Ray context: {context}")
                logger.info(f"Ray dashboard at: {context.dashboard_url}")

            # much faster to put all copies in a single ref even if each process will have to copy it
            all_copies_ref = ray.put(fig_copies)

            @ray.remote
            def _make_figure_ray(i):
                # with memray.Tracker(f"/tmp/ray/session_latest/logs/{i}_mem_profile.bin"):
                make_figure_args = (
                    ray.get(all_copies_ref)[i],
                    i + 1,
                    total_figures,
                    {"overwrite": overwrite},
                )
                return _make_figure(make_figure_args)

            task_refs = [_make_figure_ray.remote(i) for i in range(total_figures)]

            time_ray_work_submit = time.time()
            logger.debug(f"Ray work submitted in {time_ray_work_submit - time_ray_start:.2f}s")

            def wait_iterator(task_refs):
                while task_refs:
                    done, task_refs = ray.wait(task_refs, num_returns=1)
                    yield ray.get(done[0])

            results = [
                r
                for r in tqdm(
                    wait_iterator(task_refs),
                    total=total_figures,
                    desc='Generating figures',
                )
            ]
            fpaths = '\n  - ' + '\n  - '.join([str(p) for _, p in results])
            logger.debug(f"Generated {len(results)} figures:{fpaths}")

        t1 = time.time()
        logger.info(f"{total_figures} figures completed in {t1 - t0:.2f}s")

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
