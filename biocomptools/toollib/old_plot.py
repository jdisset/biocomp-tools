## {{{                          --     imports     -
from dataclasses import fields
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
from pydantic.functional_validators import BeforeValidator
from io import StringIO
import sys
from contextlib import contextmanager
from functools import partial
from dataclasses import is_dataclass
from tqdm import tqdm

from biocomp import utils as ut
from biocomp.utils import PartialFunction, flatten, as_list
from biocomp import datautils as du
from biocomp import plotutils as pu
from biocomp.plotutils import FigureSpec, PlotData, FigAx
from biocomp.datautils import DataRescaler

import matplotlib as mpl

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import ArbitraryTargetModel
from biocomptools.toollib.old_inheritable import (
    merged_into,
    merged_into_container,
    InheritableAttrsModel,
)
from biocomptools.toollib.old_resolvable import (
    resolved,
    resolvable,
    make_resolvable_validator,
    make_resolvable,
    ResolvableOr,
    Resolvable,
    short_conf,
    get_explicit_target_type,
    target_instantiate,
)

from typing import (
    Annotated,
    Optional,
    Union,
    List,
    Dict,
    Any,
    Callable,
    TypeVar,
)

from typing import Type, Union
from omegaconf import OmegaConf

from pydantic import BaseModel, Field

import logging
baselog = ut.setup_logger('biocomptools.plot', logging.WARNING)
utlog = ut.setup_logger('biocomptools.plot.utils', logging.WARNING)
inhlog = ut.setup_logger('biocomptools.plot.utils.inheritance', logging.WARNING)
reslog = ut.setup_logger('biocomptools.plot.utils.resolvable', logging.WARNING)
figlog = ut.setup_logger('biocomptools.plot.figure', logging.WARNING)
datalog = ut.setup_logger('biocomptools.plot.data', logging.WARNING)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     types     --

T = TypeVar('T')
ListOrSingle = Union[List[T], T]
DictLike = Union[Dict, DictConfig]
OmConfig = Union[DictConfig, ListConfig]


def annotated_resolvable_type(t: Type[T]):
    return Annotated[
        ResolvableOr[t],
        BeforeValidator(make_resolvable_validator(t, to_omegaconf=True)),
        Field(validate_default=True),
    ]


resolvable = partial(resolvable, to_omegaconf=True)
ResolvableDict = Annotated[ResolvableOr[Dict[Any, Any]], *resolvable(dict)]
ResolvableFigureSpec = Annotated[ResolvableOr[FigureSpec], *resolvable(FigureSpec)]
THIS_MODULE_NAME = __name__.split('.')[0]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     misc utils     --


def obj_get(obj: Any, attr: str):
    """
    Get an attribute from an object, handling various types of objects.
    """
    if ut.list_like(obj):
        return obj[int(attr)]
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else:
        try:  # check if we can access it with __getitem__
            return obj[attr]
        except (TypeError, KeyError):
            raise AttributeError(f'Could not find attribute {attr} in {obj}')


def get_public_attrs(obj):
    if is_dataclass(obj):
        return [f.name for f in fields(obj)]

    if isinstance(obj, BaseModel):
        return obj.model_fields.keys()

    # if it's a raw type, try to instantiate it
    if isinstance(obj, type):
        obj = obj()
    return [a for a in dir(obj) if not a.startswith('__')]


def with_str_keys(d: DictLike) -> Dict:
    return {str(k): v for k, v in d.items()}


@contextmanager
def indent_output(indent_level):
    old_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    try:
        yield
    finally:
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        indent = ' ' * indent_level
        tabulated = indent + output.replace('\n', '\n' + indent)
        print(tabulated, end='')


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                     --     instantiate utils     --


def obj_resolver(obj: Any, attr_path: str):
    res = obj
    for attr in attr_path.split('.'):
        try:
            res = obj_get(res, attr)
        except (AttributeError, KeyError, IndexError) as e:
            raise AttributeError(f'Could not resolve {attr_path} in {type(obj)} instance: {e}')
    return res


def make_resolvers(resolver_context: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    resolvers = {}
    if resolver_context is not None:
        for k, v in resolver_context.items():
            resolvers[k] = partial(obj_resolver, v)
    return resolvers


def func_resolver(funcname, *args, **kwargs):
    f = ut.PartialFunction(func=funcname, modules=[THIS_MODULE_NAME])
    res = f(*args, **kwargs)
    return res


class resolvers:
    # a context manager to temporarily register resolvers
    def __init__(self, resolvers: Dict[str, Callable]):
        self.resolvers = resolvers

    def __enter__(self):
        for key, value in self.resolvers.items():
            OmegaConf.clear_resolver(key)
            OmegaConf.register_new_resolver(key, value)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for key in self.resolvers.keys():
            OmegaConf.clear_resolver(key)


BASE_RESOLVERS = {'func': func_resolver}

MERGE_EXTEND_LISTS = {
    'merge_mode': {
        'list': 'extend',
        'tuple': 'extend',
        'set': 'extend',
        ListConfig: 'extend',
    },
}


def with_context(obj: T, context_dict: Dict) -> T:
    return merged_into(obj, {'context': context_dict}, 'context', deep=False, **MERGE_EXTEND_LISTS)


def resolved_with_context(obj: Resolvable[T], context_dict: Dict) -> T:
    return with_context(resolved(obj), context_dict)


def spawn(
    parent,
    unresolved_obj: Resolvable[T],
    inherit_attr: Optional[ListOrSingle[str]] = None,
) -> T:
    """
    Spawn a new object from a parent object and an unresolved object.
    """

    assert isinstance(unresolved_obj, Resolvable)

    utlog.debug(f'spawning from unresolved: {unresolved_obj}')

    if inherit_attr is not None:  # merge parent.attr into unresolved_obj
        unresolved_obj = merged_into(unresolved_obj, parent, inherit_attr)

    utlog.debug(f'after merging, unresolved: {unresolved_obj}')

    return resolved(unresolved_obj)


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                 --     inherited attrs per class     --

INHERIT_ATTRS = {
    'PlotJob': {
        'DataSource': ['metadata', 'plot_config', 'figure_spec', 'figure_maker', 'context']
    },
    'DataSource': {
        # NOTE: figure_maker itself is NOT inherited! (would duplicate for each subsource):
        'DataSource': ['metadata', 'plot_config', 'figure_spec', 'context'],
        'FigureMaker': ['plot_config', 'figure_spec', 'context'],
    },
    'FigureMaker': {
        'FigureMaker': ['plot_config', 'figure_spec', 'context'],
        'FigureTask': ['plot_config', 'figure_spec', 'context'],
    },
    'FigureTask': {'PlotTask': ['plot_config', 'context']},
}


##────────────────────────────────────────────────────────────────────────────}}}

# ╭──────────────────────╮
# │     Base Classes     │
# ╰──────────────────────╯
# + -> resolvable + inherited/merged from upstream
# = -> resolved (turned from a conf wrapper to an object)
# - -> not available
#              plot_config  figure_spec figure_maker  context
# PlotJob          +            +           +          =
# DataSource       +            +           +          =
# FigureMaker      +            +           +=         =
# FigureTask       +            +=          -          =
# PlotTask         +=           -           -          =
# PlotData         -            -           -          -

## {{{                        --     PlotConfig     --

ValidatedRescaler = Annotated[
    DataRescaler,
    BeforeValidator(
        partial(
            ut.build_if_has_target,
            available_module_names=['biocomp.datautils', '__main__'],
        )
    ),
]


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}  # rc_params for matplotlib
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    general: Dict[str, Any] = {}  # general purpose parameters
    rescaler: ValidatedRescaler = Field(default_factory=du.DataRescaler)

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


ResolvablePlotConfig = Annotated[ResolvableOr[PlotConfig], *resolvable(PlotConfig)]


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     Tasks     --


ResolvableFunction = Annotated[ResolvableOr[PartialFunction], *resolvable(PartialFunction)]


class PlotTask(ArbitraryTargetModel):
    context: Dict = {}  # inherited from parent FigureTask
    plot_config: ResolvablePlotConfig = {}  # the plot config

    # TODO: switch to ResolvableOr, not an automated resolvabler

    plot_method: ResolvableFunction = {}  # the function to call
    raw_method: ResolvableFunction = {}  # the raw function to call

    auto_callstack_bind: bool = True  # whether to automatically bind callstack params

    def model_post_init(self, *a):
        super().model_post_init(*a)

    def run(self):
        with resolvers({**make_resolvers({'ptask': self, 'this': self}), **BASE_RESOLVERS}):
            if self.plot_method:
                pc = resolved(self.plot_config)
                pc.prepare_func(resolved(self.plot_method), self.auto_callstack_bind)()
            if self.raw_method:
                resolved(self.raw_method)()


ResolvablePlotTask = Annotated[ResolvableOr[PlotTask], *resolvable(PlotTask)]

ValidatedPartialFunction = Annotated[
    ResolvableOr[PartialFunction],
    BeforeValidator(
        partial(
            ut.build_if_has_target,
            available_module_names=['biocomp.plotutils', '__main__', 'biocomptools.toollib.old_plot'],
        )
    ),
]


class FigureTask(ut.ArbitraryModel):
    # FigureTasks can be executed in parallel so need to be self-contained

    data: Annotated[List[PlotData], BeforeValidator(as_list)] = []  # the data to be plotted
    context: ResolvableDict = {}
    figure_spec: ResolvableFigureSpec = {}
    plot_config: ResolvablePlotConfig = {}
    plot_tasks: List[ResolvablePlotTask] = []
    after_plot_tasks: List[ValidatedPartialFunction] = []
    n_plot_tasks: int = 0  # modes: 0: auto (n_axis), >0: fixed number (tile to it)

    def make_plot_tasks(self, figax: FigAx) -> List[PlotTask]:
        ntasks = self.context.get('n_plot_tasks', self.n_plot_tasks)
        return [
            resolved_with_context(
                task,
                {
                    'figure_task': self,
                    'plot_task_index': i,
                    'figure': figax,
                },
            )
            for i, task in enumerate(self.get_n_plot_tasks(ntasks))
        ]

    @property
    def flat_data(self):
        return ut.flatten(self.data)

    def run(self):
        with resolvers(make_resolvers({'ftask': self, 'this': self})):
            self.figure_spec = resolved(self.figure_spec)
            assert isinstance(self.figure_spec, FigureSpec)
            with mpl.rc_context(rc=resolved(self.plot_config).rc_context):
                figax = self.figure_spec.make_figure()  # type: ignore
                ntasks = self.n_plot_tasks or figax.n_axes
                self.context['n_plot_tasks'] = ntasks
                results = [t.run() for t in self.make_plot_tasks(figax)]
                for task in self.after_plot_tasks:
                    resolved(task)(self, figax)
                self.figure_spec.finalize(figax)  # type: ignore

        return results

    def model_post_init(self, *a):
        super().model_post_init(*a)
        self.plot_tasks = merged_into_container(
            self.plot_tasks,
            self,
            INHERIT_ATTRS['FigureTask']['PlotTask'],
        )

    def get_n_plot_tasks(self, n):
        if len(self.plot_tasks) == 1:
            figlog.info('FigureTask has only one plot task. Repeating.')
            return self.plot_tasks * n
        elif len(self.plot_tasks) > n:
            figlog.warning(
                f'FigureTask has {len(self.plot_tasks)} plot tasks, but {n=}. Truncating.'
            )
            return self.plot_tasks[:n]
        elif len(self.plot_tasks) < n:
            figlog.warning(
                f'FigureTask has {len(self.plot_tasks)} plot tasks, but {n=}. Extending.'
            )
            closest_mult = n // len(self.plot_tasks)
            return self.plot_tasks * closest_mult + self.plot_tasks[: n % len(self.plot_tasks)]
        return self.plot_tasks


ResolvableFigureTask = Annotated[ResolvableOr[FigureTask], *resolvable(FigureTask)]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Figure Maker     --

FMaker = TypeVar("FMaker", bound="FigureMaker")


class FigureMaker(ArbitraryTargetModel):
    figure_spec: ResolvableFigureSpec = {}
    plot_config: ResolvablePlotConfig = {}
    context: ResolvableDict = {}
    plot_tasks: List[ResolvablePlotTask] = []
    n_plot_tasks: int = 0  # modes: 0: auto (n_axis), >0: fixed number

    def spawn_figure_task(
        self, data, context=None, resolver_dict: Optional[dict[str, Any]] = None
    ) -> FigureTask:
        resolver_dict = resolver_dict or {}
        with resolvers(make_resolvers({'this': self, 'data': data, **resolver_dict})):
            task = with_context(
                spawn(
                    self,
                    make_resolvable(
                        value=FigureTask(
                            plot_tasks=self.plot_tasks,
                            data=data,
                            n_plot_tasks=self.n_plot_tasks,
                        )
                    ),
                    inherit_attr=INHERIT_ATTRS['FigureMaker']['FigureTask'],
                ),
                context or {},
            )
            return task


ResolvableFigureMaker = Annotated[ResolvableOr[FigureMaker], *resolvable(FigureMaker)]


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     Data Sources     --


class DataSource(InheritableAttrsModel):
    metadata: ResolvableDict = {}
    figure_maker: ResolvableFigureMaker = {}
    context: ResolvableDict = {}
    plot_config: ResolvablePlotConfig = {}
    figure_spec: ResolvableFigureSpec = {}

    _inherit = {'figure_maker': INHERIT_ATTRS['DataSource']['FigureMaker']}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')

    @classmethod
    def from_config(cls, ds_cfg: DictConfig) -> 'DataSource':
        return build_datasource_from_config(ds_cfg)

    def __repr__(self, indent=0):
        return f'{" "*indent}{self.__class__.__name__}'

    def make_figure_tasks(self) -> List[FigureTask]:
        self.figure_maker = merged_into(
            self.figure_maker,
            self,
            INHERIT_ATTRS['DataSource']['FigureMaker'],
        )

        """Instantiate the FigureMaker associated with this data source
        and call its make_tasks method to get the list of figure tasks to run."""

        with resolvers(make_resolvers({'this': self})):
            figmaker = resolved(self.figure_maker)
            if hasattr(figmaker, 'make_tasks') and callable(figmaker.make_tasks):
                return figmaker.make_tasks(self.get_data())
            else:
                figlog.debug('FigureMaker %s can\'t make tasks', figmaker)

        return []


ResolvableDataSource = Annotated[ResolvableOr[DataSource], *resolvable(DataSource)]


class SpecializedDataSource(DataSource):
    @classmethod
    def from_config(cls, ds_cfg: DictConfig):
        return cls(**with_str_keys(ds_cfg))


class DataSourceGroup(DataSource):
    data_source: List[ResolvableDataSource] = []

    def model_post_init(self, *a):
        super().model_post_init(*a)
        datalog.debug('Initializing DataSourceGroup with %s data sources', len(self.data_source))

        self.data_source = [
            merged_into(
                ds,
                self,
                INHERIT_ATTRS['DataSource']['DataSource'],
            )
            for ds in self.data_source
        ]

    def get_data(self) -> List[PlotData]:
        """Get the data recursively from all the data sources in the group"""
        with resolvers(make_resolvers({'this': self})):
            return flatten([resolved(src).get_data() for src in self.data_source])

    def make_figure_tasks(self) -> List[FigureTask]:
        tasks = super().make_figure_tasks()  # make tasks with the FigureMaker for this source

        with resolvers(make_resolvers({'this': self})):  # add the tasks from each subsource
            tasks += [resolved(src).make_figure_tasks() for src in self.data_source]

        return flatten(tasks)

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        res = f'{indentstr}DataSourceGroup\n'
        for src in self.data_source:
            res += f'{indentstr} - ' + f'{src.__repr__(indent=indent+2)}\n'[indent + 2 :]
        return res


def as_datasourcegroup(ds_cfg: OmConfig) -> DictConfig:
    DATASOURCEGROUP_ATTRS = get_public_attrs(DataSourceGroup)
    datalog.debug('Converting %s to DataSourceGroup', ds_cfg)

    # operate on a copy of the config
    ds_cfg = OmegaConf.create(ds_cfg)
    new_cfg = OmegaConf.create()
    new_cfg['_target_'] = 'biocomptools.toollib.old_plot.DataSourceGroup'
    data_source = []
    if isinstance(ds_cfg, DictConfig):
        if 'data_source' in ds_cfg:
            data_source = ds_cfg['data_source']
            del ds_cfg['data_source']
        for k, v in ds_cfg.items():
            if not k in DATASOURCEGROUP_ATTRS:
                data_source.append(v)
            else:
                new_cfg[k] = v
    elif isinstance(ds_cfg, ListConfig):
        data_source = [*ds_cfg]
    else:
        raise ValueError(f'Invalid DataSourceGroup config {ds_cfg}')
    new_cfg['data_source'] = data_source

    datalog.debug(
        'Converted:\n%s\nto DataSourceGroup:\n%s', short_conf(ds_cfg), short_conf(new_cfg)
    )
    return new_cfg


def build_datasource_from_config(ds_cfg: OmConfig) -> DataSource:
    # we need to deal with a few cases here:
    # - we have a single data source (has a _target_ key)
    # - we have a dict of data sources: it's a DataSourceGroup (with names)
    # - we have a list of data sources: it's a *also* a DataSourceGroup

    # TODO: could be handled with a pydantic discriminator maybe?

    target_class = None
    if isinstance(ds_cfg, DictConfig):
        target_class = get_explicit_target_type(ds_cfg)
    if target_class is not None:
        return target_instantiate(ds_cfg)

    elif isinstance(ds_cfg, DictConfig) or isinstance(ds_cfg, ListConfig):
        ds_cfg = as_datasourcegroup(ds_cfg)
        target_class = get_explicit_target_type(ds_cfg)
        assert target_class == DataSourceGroup

    else:
        raise ValueError(f'Invalid DataSourceGroup config {ds_cfg}')

    return target_instantiate(ds_cfg)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     PlotJob     --

import ray


class PlotJob(InheritableAttrsModel):
    figure_spec: ResolvableFigureSpec = {}
    figure_maker: ResolvableFigureMaker = {}
    plot_config: ResolvablePlotConfig = {}
    data_source: ResolvableDataSource = {}
    metadata: ResolvableDict = {}
    context: ResolvableDict = {}
    extra: dict[str, Any] = {}

    _inherit = {'data_source': INHERIT_ATTRS['PlotJob']['DataSource']}

    def generate_figure_tasks(self) -> List[FigureTask]:
        with resolvers({**make_resolvers({'job': self, 'this': self}), **BASE_RESOLVERS}):
            return resolved(self.data_source).make_figure_tasks()

    def run_tasks(self):
        @ray.remote
        def worker(task, progress):
            task.run()
            progress.update(1)

        with resolvers({**make_resolvers({'job': self, 'this': self}), **BASE_RESOLVERS}):
            ray.init(ignore_reinit_error=True)
            tasks = self.generate_figure_tasks()
            progress = tqdm_ray.tqdm(total=len(tasks))
            # run the tasks in parallel and update the progress bar as they finish
            futures = [worker.remote(task, progress) for task in tasks]
            ray.get(futures)
            progress.close()

    def run_tasks_sequential(self):
        tasks = self.generate_figure_tasks()
        for task in tasks:
            task.run()


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     misc utils     --


def write_text(
    canvas: Union[mpl.figure.Figure, mpl.axes.Axes],
    content: dict,
    coords: tuple[float, float] = (1.1, 1.0),
    fontsize: int = 7,
    verticalalignment: str = 'top',
    horizontalalignment: str = 'left',
    **kw,
):
    kwargs = dict(
        fontsize=fontsize,
        verticalalignment=verticalalignment,
        horizontalalignment=horizontalalignment,
        **kw,
    )

    try:
        txt = ut.yaml_dump(content)
    except Exception as e:
        txt = str(content)

    canvas.text(
        *coords,
        txt,
        **kwargs,
    )


##────────────────────────────────────────────────────────────────────────────}}}
