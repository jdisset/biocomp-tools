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
    general: Dict[str, Any] = {}  # general purpose parameters
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
        print(f'Running {len(self.plot_tasks)} plot tasks, figure {figax.flat_ax[0]}')
        plot_tasks = [t.construct(context={"FIG": figax}) for t in self.plot_tasks]
        for task in plot_tasks:
            resolve_all_lazy(task)
            task.run()

        self.figure_spec.finalize(figax)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     PlotJob     --


class PlotJob(LazyDraconModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]

    def run(self):
        self._context = {}
        print([type(f) for f in self.figures])
        figures = [f.construct(deferred_paths=['/figures.*.plot_tasks.*']) for f in self.figures]
        results = [f.run() for f in figures]
        print(results)


##────────────────────────────────────────────────────────────────────────────}}}


def main():
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
    )
    pj.run()


if __name__ == '__main__':
    main()
