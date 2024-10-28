## {{{                          --     imports     --
from typing import Any, List, Dict, Optional, Annotated
import matplotlib as mpl
from dracon.draconstructor import resolve_all_lazy
from biocomp.datautils import DataRescaler
from biocomp.utils import PartialFunction, ArbitraryModel
from dracon.deferred import DeferredNode
from biocomptools.toollib.datasources import DataSource, DBSource, NetworkPrediction
from biocomp import utils as ut
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout
import dracon as dr
from pydantic import BaseModel, Field, BeforeValidator

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     plot config     --


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}  # rc_params for matplotlib
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    rescaler: DataRescaler = Field(default_factory=DataRescaler)

    def prepare_func(
        self,
        plot_method: PartialFunction,
        auto_callstack_bind: bool = True,
        overwrite_kwargs: Optional[dict] = None,
    ):
        callstack_conf = {}
        if auto_callstack_bind:
            callstack_conf = ut.generate_full_nested_config(
                self.callstack_params, namespace='biocomp.plotting'
            ).get(f'{plot_method.get_name()}_params', {})

        if overwrite_kwargs:
            plot_method.set_missing_kwargs(overwrite_kwargs)

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

    # used as default if plot_method has an "ax" parameter that is not bound:
    ax: Optional[mpl.axes.Axes] = None

    def model_post_init(self, *a):
        super().model_post_init(*a)

    def run(self):
        if self.plot_method:
            kw = {'ax': self.ax} if self.ax else {}
            self.plot_config.prepare_func(
                plot_method=self.plot_method,
                auto_callstack_bind=self.auto_callstack_bind,
                overwrite_kwargs=kw,
            )()
        if self.raw_method:
            self.raw_method()


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     figure     --


def resolve(obj):
    resolve_all_lazy(obj)
    return obj


def is_dict_like(obj):
    return hasattr(obj, 'items') and hasattr(obj, 'keys')


class Figure(ArbitraryModel):
    figure_spec: Annotated[FigureSpec, BeforeValidator(resolve)]
    plot_tasks: List[DeferredNode[PlotTask]] = []

    def run(self):
        figax = self.figure_spec.make_figure()  # type: ignore
        for i, t in enumerate(self.plot_tasks):
            pt = t.construct(context={"FIG": figax})
            if is_dict_like(pt):
                pt = PlotTask(**pt)
            pt.ax = figax.flat_ax[i]  # default ax, can be overridden in the plot_method
            resolve_all_lazy(pt)
            pt.run()

        self.figure_spec.finalize(figax)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}
