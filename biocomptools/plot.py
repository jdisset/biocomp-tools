## {{{                          --     imports     --

import logging

import dracon as dr
from dracon.lazy import LazyDraconModel
from dracon.draconstructor import resolve_all_lazy
from typing import List, Annotated, Dict, Any, Optional
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode
import sys
from biocomp.utils import PartialFunction, ArbitraryModel
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec
from pydantic import BaseModel, Field
from biocomp import utils as ut
import matplotlib as mpl
from biocomptools.toollib.datasources import DataSource, DBSource

logging.basicConfig(level=logging.WARNING)
baselog = ut.setup_logger('biocomptools.plot', logging.WARNING)
utlog = ut.setup_logger('biocomptools.plot.utils', logging.WARNING)
inhlog = ut.setup_logger('biocomptools.plot.utils.inheritance', logging.WARNING)
reslog = ut.setup_logger('biocomptools.plot.utils.resolvable', logging.WARNING)
figlog = ut.setup_logger('biocomptools.plot.figure', logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
datalog = ut.setup_logger('biocomptools.plot.data', logging.WARNING)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     plot config     --


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}  # rc_params for matplotlib
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    rescaler: DataRescaler = Field(default_factory=DataRescaler)

    def prepare_func(self, plot_method: PartialFunction, auto_callstack_bind: bool = True):
        callstack_conf = {}
        if auto_callstack_bind:
            callstack_conf = ut.generate_full_nested_config(
                self.callstack_params, namespace='biocomp.plotting'
            ).get(f'{plot_method.get_name()}_params', {})

        def prepared_func(
            *args, rc=self.rc_context, cs=callstack_conf, rescaler=self.rescaler, **kwargs
        ):
            full_kwargs = {'rescaler': rescaler, **cs, **kwargs}

            with mpl.rc_context(rc=rc):
                return plot_method(*args, **full_kwargs)

        return prepared_func


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                         --     plot task     --


class PlotTask(ArbitraryModel):
    context: Dict = {}
    plot_config: PlotConfig = Field(default_factory=PlotConfig)
    plot_method: Optional[PartialFunction] = None
    raw_method: Optional[PartialFunction] = None
    auto_callstack_bind: bool = True  # whether to automatically bind callstack params
    plot_data: Any = None

    def model_post_init(self, *a):
        super().model_post_init(*a)

    def run(self):
        if self.plot_method:
            self.plot_config.prepare_func(self.plot_method, self.auto_callstack_bind)()
        if self.raw_method:
            self.raw_method()


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     figure     --


class Figure(ArbitraryModel):
    figure_spec: FigureSpec
    plot_tasks: List[DeferredNode[PlotTask]] = []
    after_plot_tasks: List[PartialFunction] = []

    def run(self):
        figax = self.figure_spec.make_figure()  # type: ignore
        for t in self.plot_tasks:
            pt = t.construct(context={"FIG": figax})
            if isinstance(pt, dict):
                pt = PlotTask(**pt)
            resolve_all_lazy(pt)
            pt.run()

        self.figure_spec.finalize(figax)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     PlotJob     --


class PlotJob(LazyDraconModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]

    def run(self):
        self._context = {}
        figures = []
        for f in self.figures:
            f = f.construct(deferred_paths=['/figures.*.plot_tasks.*'])
            if isinstance(f, dict):
                f = Figure(**f)
            figures.append(f)

        results = [f.run() for f in figures]


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
        context={'DBSource': DBSource},
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
    with profile():
        main()
