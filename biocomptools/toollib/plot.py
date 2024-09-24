## {{{                          --     imports     --
from typing import Any, List, Dict, Optional
from dracon.resolvable import Resolvable
from dracon import resolvable_maker as resfactory
from dracon import KeyPath as KP
from dracon.keypath import ROOTPATH
import biocomp.datautils as du
import matplotlib as mpl

from biocomp.utils import PartialFunction, generate_full_nested_config, flatten
from biocomptools.toollib.common import ArbitraryTargetModel
from biocomp.plotutils import FigureSpec, PlotData, FigAx
from biocomp.datautils import DataRescaler

from pydantic import BaseModel, Field, Annotated

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     PlotConfig     --


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}  # rc_params for matplotlib
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    general: Dict[str, Any] = {}  # general purpose parameters
    rescaler: du.DataRescaler = Field(default_factory=du.DataRescaler)

    def prepare_func(self, plot_method: PartialFunction, auto_callstack_bind: bool = True):
        callstack_conf = {}
        if auto_callstack_bind:
            callstack_conf = generate_full_nested_config(
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

## {{{                         --     PlotTask     --


class PlotTask(ArbitraryTargetModel):
    context: Dict = {}  # inherited from parent FigureTask
    plot_config: Resolvable[PlotConfig] = Field(default_factory=resfactory(PlotConfig))
    plot_method: Resolvable[PartialFunction] = Field(default_factory=resfactory(PartialFunction))
    raw_method: Resolvable[PartialFunction] = Field(default_factory=resfactory(PartialFunction))

    auto_callstack_bind: bool = True  # whether to automatically bind callstack params

    def model_post_init(self, *a):
        super().model_post_init(*a)

    def run(self):
        ctx = {"$PTASK": self}
        if not self.raw_method:
            pc = self.plot_config.resolve(ctx)
            pc.prepare_func(self.plot_method.resolve(ctx), self.auto_callstack_bind)()
        if self.raw_method:
            self.raw_method.resolve(ctx)()


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                         --     FigureTask     --


class FigureTask(BaseModel):
    context: Dict[str, Any] = Field(default_factory=dict)
    figure_spec: Resolvable[FigureSpec] = Field(default_factory=resfactory(FigureSpec))
    plot_tasks: List[Resolvable[PlotTask]] = Field(default_factory=list)

    def make_plot_tasks(self, figax: FigAx) -> List[PlotTask]:
        ntasks = self.context.get('n_plot_tasks', self.n_plot_tasks)
        plot_tasks = self.get_n_plot_tasks(ntasks)
        return [
            task.resolve(context={"$FTASK": self, "$FIGURE": figax, "$INDEX": i})
            for i, task in enumerate(plot_tasks)
        ]

    @property
    def flat_data(self):
        return flatten(self.data)

    def run(self):
        ctx = {"$FTASK": self}
        fig_spec = self.figure_spec.resolve(context=ctx)
        assert isinstance(fig_spec, FigureSpec)
        plot_config = self.plot_config.resolve(context=ctx)
        with mpl.rc_context(rc=plot_config.rc_context):
            figax = fig_spec.make_figure()
            ntasks = self.n_plot_tasks or figax.n_axes
            self.context['n_plot_tasks'] = ntasks
            results = [t.run() for t in self.make_plot_tasks(figax)]
            for task_func in self.after_plot_tasks:
                task_func.resolve(context=ctx)(self, figax)
            fig_spec.finalize(figax)
        return results

    def get_n_plot_tasks(self, n):
        num_tasks = len(self.plot_tasks)
        if num_tasks == 1:
            return self.plot_tasks * n
        elif num_tasks > n:
            return self.plot_tasks[:n]
        elif num_tasks < n:
            multiplier = n // num_tasks
            remainder = n % num_tasks
            return self.plot_tasks * multiplier + self.plot_tasks[:remainder]
        return self.plot_tasks


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                          --     PlotJob     --


class PlotJob(InheritableAttrsModel):
    plot_config: Resolvable[PlotConfig] = Field(default_factory=resfactory(PlotConfig))
    figure_spec: Resolvable[FigureSpec] = Field(default_factory=resfactory(FigureSpec))

    figure_maker: Resolvable[FigureMaker] = {}
    data_source: Resolvable[DataSource] = {}

    metadata: Resolvable[dict] = {}
    context: Resolvable[dict] = {}
    extra: dict[str, Any] = {}


##────────────────────────────────────────────────────────────────────────────}}}
