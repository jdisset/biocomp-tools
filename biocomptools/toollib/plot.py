## {{{                          --     imports     --
from typing import Any, List, Dict, Optional, Annotated
import matplotlib as mpl
from dracon.draconstructor import resolve_all_lazy
from biocomp.datautils import DataRescaler
from biocomp.utils import PartialFunction
from dracon.deferred import DeferredNode
from biocomp import utils as ut
from biocomp.plotutils import FigureSpec
from dracon.utils import dict_like
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from biocomptools.logging_config import get_logger
from biocomptools.trainutils import make_json_ready

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

    def _auto_extract_metadata(
        self,
        callstack_conf: dict,
        plot_method: PartialFunction,
        size_limit: int = 2000,
        *args,
        **kwargs,
    ):
        """
        Returns some possibly interesting metadata from the callstack configuration and the plot method.
        """
        import json

        def can_dump(obj):
            try:
                return len(json.dumps(make_json_ready(obj))) < size_limit
            except Exception as e:
                return False

        metadata = {}
        metadata['plot_method'] = plot_method.get_name()

        if can_dump(callstack_conf):
            metadata['callstack_conf'] = callstack_conf

        extra_args = list(args)

        for pval in plot_method.args + extra_args:
            if hasattr(pval, 'metadata'):
                for k, v in pval.metadata.items():
                    if can_dump(v):
                        metadata[k] = v

        extra_kwargs = list(kwargs.items())
        for _, pval in list(plot_method.kwargs.items()) + extra_kwargs:
            if hasattr(pval, 'metadata'):
                for k, v in pval.metadata.items():
                    if can_dump(v):
                        metadata[k] = v

        return metadata

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

        def wrapped_plot_method(*args, **kwargs):
            # collect metadata after calling the plot method
            # (better than before, since some args may be modified e.g. lazy-loaded plot data)
            res = plot_method(*args, **kwargs)
            metadata = self._auto_extract_metadata(
                callstack_conf,
                plot_method,
                *args,
                size_limit=2000,
                **kwargs,
            )
            return res, metadata

        def prepared_func(
            *args, rc=self.rc_context, cs=callstack_conf, rescaler=self.rescaler, **kwargs
        ):
            full_kwargs = {'rescaler': rescaler, **cs, **kwargs}

            with mpl.rc_context(rc=rc):
                return wrapped_plot_method(*args, **full_kwargs)

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


class PlotTask(BaseModel):
    plot_config: PlotConfig = Field(default_factory=PlotConfig)
    plot_method: Optional[PartialFunction] = None
    auto_callstack_bind: bool = True  # whether to automatically bind callstack params

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _ax: Optional[mpl.axes.Axes] = None

    def run(self):
        # generates some metadata and returns it
        metadata = {}
        if self.plot_method:
            kw = {'ax': self._ax} if self._ax else {}
            f = self.plot_config.prepare_func(
                plot_method=self.plot_method,
                auto_callstack_bind=self.auto_callstack_bind,
                overwrite_kwargs=kw,
            )

            result, metadata = f()

        return metadata


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     figure     --


def resolve(obj):
    resolve_all_lazy(obj)
    return obj


class Figure(BaseModel):
    figure_spec: Annotated[FigureSpec, BeforeValidator(resolve)]
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    plot_tasks: List[DeferredNode[PlotTask]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ptasks: Optional[List[PlotTask]] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.plot_tasks = [task.copy(reroot=True) for task in self.plot_tasks]

    def prepare(self):
        with mpl.rc_context(rc=self.plot_config.rc_context):
            try:
                self._figax = self.figure_spec.make_figure()
            except Exception as e:
                logger.error(f"Error making figure axes: {e}")
                logger.exception(e)
                return

            self._ptasks = []
            for i, tc in enumerate(self.plot_tasks):
                try:
                    pt = tc.construct(context={"FIG": self._figax})
                    if dict_like(pt):
                        pt = PlotTask(**pt)  # type: ignore
                    pt.plot_config.inherit_from(self.plot_config)

                    # default ax, can be overridden in the plot_method:
                    pt._ax = self._figax.flat_ax[i]
                    self._ptasks.append(pt)
                except Exception as e:
                    logger.error(f"Error constructing plot task {i}: {e}")
                    logger.exception(e)
                    continue

    def run(self, overwrite: bool = True):
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        if self._ptasks is None:
            self.prepare()
        assert isinstance(self._ptasks, list)

        with mpl.rc_context(rc=self.plot_config.rc_context):
            metadata = {}
            metadata['plot_tasks'] = []
            for i, pt in enumerate(self._ptasks):
                try:
                    resolve_all_lazy(pt)
                    pt_metadata = pt.run()
                    metadata['plot_tasks'].append(pt_metadata)
                except Exception as e:
                    logger.error(f"Error running plot task {i}: {e}")
                    logger.exception(e)
                    continue

            metadata['figure_spec'] = self.figure_spec

            self.figure_spec.finalize(self._figax)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}

