## {{{                          --     imports     --
from typing import Any, List, Dict, Optional, Annotated
import matplotlib as mpl
from dracon.draconstructor import resolve_all_lazy
from biocomp.datautils import DataRescaler
from biocomp.utils import PartialFunction, ArbitraryModel
from dracon.deferred import DeferredNode
from biocomp import utils as ut
from biocomp.plotutils import FigureSpec
from dracon.utils import dict_like
from pydantic import BaseModel, Field, BeforeValidator
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)
##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     plot config     --


def load_default_plotconf():
    import dracon as dr
    from dracon.lazy import resolve_all_lazy

    plcontent = dr.load(
        'pkg:biocomptools:configs/plot_config/default_plotconf_v2',
        enable_interpolation=True,
        raw_dict=True,
    )
    resolve_all_lazy(plcontent)
    pc = PlotConfig(**plcontent)
    return pc


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    rescaler: Optional[DataRescaler] = None

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

            # logger.debug(f'Plotting with kwargs: {full_kwargs} + {plot_method.kwargs}')

            with mpl.rc_context(rc=rc):
                return plot_method(*args, **full_kwargs)

        return prepared_func

    def inherit_from(
        self, other: 'PlotConfig', keep_rescaler: bool = True, key: str = '<<{+<}[~<]'
    ):
        from dracon.merge import MergeKey, merged

        k = MergeKey(raw=key)

        if not keep_rescaler or not self.rescaler:
            self.rescaler = other.rescaler

        self.callstack_params = merged(other.callstack_params, self.callstack_params, k)  # type: ignore
        self.rc_context = merged(other.rc_context, self.rc_context, k)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                         --     plot task     --


class PlotTask(ArbitraryModel):
    plot_config: PlotConfig = Field(default_factory=PlotConfig)
    plot_method: Optional[PartialFunction] = None
    post_plot_callbacks: List[PartialFunction] = []
    auto_callstack_bind: bool = True  # whether to automatically bind callstack params

    # used as default if plot_method has an "ax" parameter that is not bound:
    ax: Optional[mpl.axes.Axes] = None

    def run(self):
        if self.plot_method:
            kw = {'ax': self.ax} if self.ax else {}
            self.plot_config.prepare_func(
                plot_method=self.plot_method,
                auto_callstack_bind=self.auto_callstack_bind,
                overwrite_kwargs=kw,
            )()

        for cb in self.post_plot_callbacks:
            resolve_all_lazy(cb, context={'ax': self.ax})
            logger.debug(f"Running post-plot callback {cb}")
            kw = {'ax': self.ax} if self.ax else {}
            cb.set_missing_kwargs(kw)
            cb()


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     figure     --


def resolve(obj):
    resolve_all_lazy(obj)
    return obj


class Figure(ArbitraryModel):
    figure_spec: Annotated[FigureSpec, BeforeValidator(resolve)]
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    plot_tasks: List[DeferredNode[PlotTask]] = []

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.figure_spec = resolve(self.figure_spec)
        self.plot_config = resolve(self.plot_config)

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        with mpl.rc_context(rc=self.plot_config.rc_context):
            try:
                logger.debug("Making FigAx")
                figax = self.figure_spec.make_figure()
                logger.debug(f"FigAx for {self.figure_spec.output_path} made")
            except Exception as e:
                logger.error(f"Error making figure: {e}")
                return

            for i, t in enumerate(self.plot_tasks):
                try:
                    logger.debug(f"Constructing plot task {i}")
                    pt = t.construct(context={"FIG": figax})
                    if dict_like(pt):
                        pt = PlotTask(**pt)  # type: ignore
                    pt.plot_config.inherit_from(self.plot_config)
                    pt.ax = figax.flat_ax[i]  # default ax, can be overridden in the plot_method
                    logger.debug(f"Task {i} constructed")

                except Exception as e:
                    logger.error(f"Error constructing plot task {i}: {e}")
                    continue

                try:
                    logger.debug(f"Resolving plot task {i}")
                    resolve_all_lazy(pt)
                    pt.run()
                    logger.debug(f"Plot task {i} done")
                except Exception as e:
                    import traceback

                    traceback_msg = traceback.format_exc()
                    logger.error(f"Error running plot task {i}: {e}")
                    logger.error(traceback_msg)
                    continue

            self.figure_spec.finalize(figax)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}
