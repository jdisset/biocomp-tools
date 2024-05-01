## {{{                          --     imports     -
from dataclasses import dataclass, field, fields
from omegaconf import OmegaConf, open_dict
from omegaconf import DictConfig, ListConfig
from io import StringIO
import sys
from contextlib import contextmanager
from biocomptools.toollib import common as cm
from functools import partial
from typing import get_origin
from dataclasses import is_dataclass
from copy import deepcopy
import logging
from biocomptools.toollib import plot as pl
from rich import print as rprint
import pandas as pd
from pathlib import Path
import numpy as np

from biocomp import utils as ut
import biocomp.plotting.plotting_3d as p3d
import biocomp.plotting.plotting_core as pc
from biocomp import datautils as du
from biocomp import plotutils as pu
import biocomp as bc

from typing import (
    Annotated,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Sequence,
    Any,
    Callable,
    TypeVar,
    Generic,
    Iterable,
)

from typing import Type, Union
import hydra
from hydra import compose, initialize, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.global_hydra import GlobalHydra
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl
import matplotlib.pyplot as plt

from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator

from rich.logging import RichHandler

LOGFORMAT = "in %(funcName)s: %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level="NOTSET", format=LOGFORMAT, datefmt="[%X]", handlers=[RichHandler()])

baselog = logging.getLogger('biocomptools.biocomplot')
utlog = logging.getLogger('biocomptools.biocomplot.utils')
inhlog = logging.getLogger('biocomptools.biocomplot.utils.inheritance')
reslog = logging.getLogger('biocomptools.biocomplot.utils.resolvable')
figlog = logging.getLogger('biocomptools.biocomplot.figure')
datalog = logging.getLogger('biocomptools.biocomplot.data')


baselog.setLevel(logging.INFO)
utlog.setLevel(logging.INFO)
inhlog.setLevel(logging.INFO)
reslog.setLevel(logging.INFO)
figlog.setLevel(logging.INFO)
datalog.setLevel(logging.INFO)

# baselog.setLevel(logging.DEBUG)
# utlog.setLevel(logging.DEBUG)
# inhlog.setLevel(logging.DEBUG)
reslog.setLevel(logging.DEBUG)
figlog.setLevel(logging.DEBUG)
# datalog.setLevel(logging.DEBUG)

from biocomp.plotutils import FigureSpec, PlotData

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --

from typing import TypeVar, Generic, Protocol, Self

T = TypeVar('T')
U = TypeVar('U')
ListOrSingle = Union[List[T], T]
Pair = Tuple[T, T]
DictOrList = Union[Dict[U, T], List[T]]
DictLike = Union[Dict, DictConfig]
AnyConfig = Union[DictConfig, ListConfig]

D = TypeVar('D', Dict, DictConfig)


class ConfigHolder(Protocol):
    config: Optional[DictLike]


Mergeable = Union[DictLike, ConfigHolder]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     misc utils     --

def obj_resolver(obj: Any, attr_path: str):
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def get_public_attrs(obj):
    if is_dataclass(obj):
        return [f.name for f in fields(obj)]
    # if it's a type, try to instantiate it
    if isinstance(obj, type):
        obj = obj()
    return [a for a in dir(obj) if not a.startswith('__')]


def with_str_keys(d: DictLike) -> Dict:
    return {str(k): v for k, v in d.items()}


def list_like(obj):
    return isinstance(obj, (list, tuple, ListConfig))


def make_flat_list(l) -> List:
    """
    if l is a list of lists, flatten it. If it's a single element, return it as a list
    """
    if not list_like(l):
        return [l]
    else:
        # only unpack an item if it's a list. should be recursive.
        res = []
        for item in l:
            if list_like(item):
                res += make_flat_list(item)
            else:
                res.append(item)
        return res


def truncated_path(path: str, max_len=50) -> str:
    if len(path) > max_len:
        return '...' + path[-max_len:]
    return path


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
from importlib import import_module


def has_target(obj):
    return '_target_' in obj


def get_target_class(obj: DictLike, default_module='biocomptools.toollib.plot') -> Type:
    """get the class of the target object"""

    assert '_target_' in obj, f'Invalid data source object: {obj=}'

    # target should be a string module.path.ClassName
    # let's check that the class exists and it is a subclass of DataSource
    target = obj['_target_']
    target_parts = target.split('.')
    target_module = '.'.join(target_parts[:-1])
    target_class = target_parts[-1]

    if target_module == '':
        target_module = import_module(default_module)
    else:
        target_module = import_module(target_module)

    target_class = getattr(target_module, target_class)
    return target_class


def target_instantiate(obj: DictLike, default_module='biocomptools.toollib.plot'):
    """simply create an instance of _target_ with the rest of the dict as kwargs"""
    target_class = get_target_class(obj, default_module)
    kwargs = {k: v for k, v in obj.items() if k != '_target_'}
    utlog.debug('Instantiating %s with %s', target_class, kwargs)
    return target_class(**kwargs)


def generic_instanciation_ctor(cfg: DictLike) -> Any:
    assert has_target(cfg), f'instanciation ctor called on non-target object {cfg}'
    return target_instantiate(cfg)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     inherit utils     --

# ╭──────────────────────╮
# │     Base Classes     │
# ╰──────────────────────╯
# + -> resolvable + inherited/merged from upstream
# = -> resolved (turned from a conf wrapper to an object)
# - -> not available
#               metadata  plot_config  figure_spec figure_maker  context
# PlotJob          +           +            +           +          =
# DataSource       +           +            +           +          =
# FigureMaker      +           +            +           +=         =
# FigureTask       +           +            +=          -          =
# PlotTask         +=          +=           -           -          =
# PlotData         -           -            -           -          -


INHERITED_ATTRIBUTES = {
    'PlotJob': {
        'DataSource': ['metadata', 'plot_config', 'figure_spec', 'figure_maker', 'context']
    },
    'DataSource': {
        # NOTE: figure_maker itself is NOT inherited! (would duplicate for each subsource):
        'DataSource': ['metadata', 'plot_config', 'figure_spec', 'context'],
        'FigureMaker': ['metadata', 'plot_config', 'figure_spec', 'context'],
    },
    'FigureMaker': {
        'FigureMaker': ['metadata', 'plot_config', 'figure_spec', 'context'],
        'FigureTask': ['metadata', 'plot_config', 'figure_spec', 'context'],
    },
    'FigureTask': {'PlotTask': ['metadata', 'plot_config']},
}


# a set of utils to transparently treat dict-like objects and resolvables as mergeable objects


def config_holder(obj) -> bool:
    return hasattr(obj, 'config') and (dict_like(obj.config) or obj.config is None)


def as_dict(obj) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, DictConfig):
        d = OmegaConf.to_container(obj, resolve=False)
        assert isinstance(d, dict), f'Invalid dict-like object {d}'
        return d
    if config_holder(obj):
        return as_dict(obj.config)
    return obj


def merged(parent: DictLike, child: D, **kw) -> D:
    """Merge parent DictLike structure into child,
    return the result without modifying child,
    and in the same type as child."""

    merged = ut.updated_dict(as_dict(parent), as_dict(child), **kw)
    assert isinstance(merged, dict), f'Invalid merged object {merged}'

    if isinstance(child, DictConfig):
        return OmegaConf.create(merged)

    return merged


def get_dict_attr(obj: DictLike, attr_name: str) -> Any:
    try:
        return obj[attr_name]
    except TypeError:
        return getattr(obj, attr_name)


def set_dict_attr(obj: DictLike, attr_name: str, value: Any):
    try:
        obj[attr_name] = value
    except TypeError:
        setattr(obj, attr_name, value)


@contextmanager
def open_dictlike(obj: Any):
    if isinstance(obj, DictConfig):
        with open_dict(obj):
            yield
    else:
        yield


def inplace_merge_into(target: Any, parent: Any, attr_names: ListOrSingle[str], **kw):
    """merge parent[attr_name] into target[attr_name]"""

    if isinstance(attr_names, str):
        attr_names = [attr_names]

    for attr_name in attr_names:
        try:
            parent_attr = get_dict_attr(parent, attr_name)
        except KeyError:
            inhlog.debug('No parent attribute %s found in %s', attr_name, parent)
            continue

        try:
            target_attr = get_dict_attr(target, attr_name)
        except KeyError:
            target_attr = {}

        inhlog.debug(
            'Merging %s from %s into %s of %s', attr_name, parent_attr, target_attr, target
        )
        merged_attr = merged(parent_attr, target_attr, **kw)
        inhlog.debug('Merged %s -> %s', attr_name, merged_attr)
        with open_dictlike(target):
            set_dict_attr(target, attr_name, merged_attr)


def merged_into(target: Any, parent: Any, attr_names: ListOrSingle[str], **kw):
    target_copy = deepcopy(target)
    inplace_merge_into(target_copy, parent, attr_names, **kw)
    return target_copy


def wrap_resolvable_constructor(obj: Resolvable[T], fn: Callable) -> Resolvable[T]:

    assert isinstance(obj, Resolvable)

    def wrapped_constructor(config: DictConfig, fn=fn, base_constructor=obj.constructor):
        return base_constructor(fn(config))

    return Resolvable(
        constructor=wrapped_constructor,
        config=obj.config,
        name=obj.name,
        target_type=obj.target_type,
    )


class InheritanceSpec(BaseModel):
    inherited_attrs: Iterable[str]
    child_attrs: Iterable[str]


def make_inheritable(
    target: Resolvable[T], parent: Any, attr_names: ListOrSingle[str]
) -> Resolvable[T]:

    # we wrap the target's constructor so that, just before resolving it,
    # we merge the inherited parent attributes into it
    return wrap_resolvable_constructor(
        target,
        partial(
            merged_into,
            parent=parent,
            attr_names=attr_names,
        ),
    )


# ╭───────────────────────────╮
# │      @inherit_attrs       │
# ╰───────────────────────────╯
def inherit_attrs(*inheritance_specs: InheritanceSpec):
    """
    A decorator to make some attributes to a class inheritable.

    Only inheritance on Resolvable attributes is supported.

    Classes can declare some attributes as being inheritable by some designated
    children objects. The idea is that you want things like metadata, context,
    or plot_config to be resolved at the last minute, and to able to be
    modified at any level of the hierarchy (for example PlotJob -> DataSource
    -> FigureTask -> PlotTask).

    So let's say we create a PlotJob from a config file, and in it we have a
    little bit of metadata information (like the name of the job, the author,
    etc.). We also have a PlotConfig object that contains the default rc_params
    for matplotlib, and a figure_maker that uses the default FigureMaker. We
    then have data_sources that are created and that can override all these
    attributes.

    First we initialize PlotJob with all the attributes as Resolvable objects
    (they keep their DictConfig representation until resolved).

    When accessing an attribute, let's say job.data_source, we are going to
    trigger its resolution. Before calling resolve() and therefore losing the
    DictConfig representation of the data_source attribute, we want to *check if
    it's been flagged as child that should inherit some attributes from the
    parent*.

    If yes, we want - prior to resolution - to merge the parent's attribute
    (metadata, context, etc.) with the config of the child attribute. Then we
    can resolve it.

    So this decorator needs to signal to a potential resolve() call that it
    should merge the parent's attribute with the child's attribute before.

    One way to do that is to inject the merge stuff in the constructor of the
    Resolvable object, which is what we are going to do here.

    There are 2 cases to consider:
    - the child attribute is being resolved first:
        we inject a function that will merge every inherited attributes into it

    - the parent attribute is being resolved first
        we inject a function that will merge this attribute with every child

    Here we only consider the first case, because the second case should not
    happen in the current setup.


    """

    # turn the inheritance specs into 2 dictionaries:
    # child -> inherited attributes and inherited attribute -> children

    child_to_attributes = {}

    for spec in inheritance_specs:
        for child_attr in spec.child_attrs:
            if child_attr not in child_to_attributes:
                child_to_attributes[child_attr] = []
            child_to_attributes[child_attr] += spec.inherited_attrs

    for k, v in child_to_attributes.items():
        child_to_attributes[k] = list(set(v))

    def decorator(cls):

        inhlog.debug('Setting up %s with inheritable attrs', cls.__name__)

        original_init = cls.__init__

        def inherit_init(self, *args, **kwargs):

            original_init(self, *args, **kwargs)

            inhlog.debug('Initializing %s with inheritable attrs', cls.__name__)

            for child_name, attrs in child_to_attributes.items():
                wrapped_child = make_inheritable(getattr(self, child_name), self, attrs)
                setattr(self, child_name, wrapped_child)

        cls.__init__ = inherit_init

        return cls

    return decorator


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     spawn utils     --
def spawn(
    parent,
    unresolved_obj: Resolvable[T],
    inherit_attr: Optional[ListOrSingle[str]] = None,
    inherit_extra: Optional[Dict[str, Any]] = None,
    resolver_context: Optional[Dict[str, Any]] = None,
    inherit_extra_args: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Spawn a new object from a parent object and an unresolved object.
    The unresolved object is resolved with the parent's context, and then
    merged with the parent's attributes.
    """

    resolver_context = resolver_context or {}
    resolver_context['this'] = resolver_context.get('this', parent)

    assert isinstance(unresolved_obj, Resolvable)

    figlog.debug(f'spawning from unresolved: {unresolved_obj}')

    if inherit_attr is not None:
        # merge parent.attr into unresolved_obj
        unresolved_obj = merged_into(unresolved_obj, parent, inherit_attr)

    if inherit_extra is not None:
        if inherit_extra_args is None:
            inherit_extra_args = {}

        unresolved_obj = merged_into(
            unresolved_obj,
            inherit_extra,
            list(inherit_extra.keys()),
            **inherit_extra_args,
        )

    figlog.debug(f'after merging, unresolved: {unresolved_obj}')

    # problem is we have a resolved PlotConfig there when coming
    # from a FigureMaker.spawn(... (figureTask) )

    obj = resolve(unresolved_obj, make_resolvers(resolver_context))

    return obj


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                        --     PlotConfig     --
class PlotConfig(BaseModel):
    # rc_params for matplotlib
    rc_context: Dict[str, Any] = field(default_factory=dict)
    # nested parameters for the plotting function
    callstack_params: Dict[str, Any] = field(default_factory=dict)
    # for general purpose storage of parameters
    general: Dict[str, Any] = field(default_factory=dict)


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                           --     Tasks     --


# all plot_methods (called from PlotTask) should have the following signature:
# def plot_method(ax: Axes, data: PlotData, plot_config: PlotConfig, **kwargs)
# NOTE : rescalers should probably be renamed to DataTransform? Or maybe not
# rescalers could be either specified in data or in the plotconfig
# advantage of having it in the PlotData instance:
# ...
# if it was not in the PlotData obj then we can use the same data with different rescaler
# later on (for example on different slices.
# Arguably this could also be done by loading the same source several times wiht a different
# rescaler everytime, and any extra loading cost can be solved with caching
# yeah I think that's the easiest for now,
#


class PlotTask(BaseModel):

    context: Dict[str, Any] = MISSING  # inherited from parent FigureTask

    ax: ListOrSingle[Axes] = MISSING  # the axes to plot on
    data: ListOrSingle[PlotData] = MISSING  # the data to be plotted

    plot_config: PlotConfig = MISSING
    plot_method: Callable = pu.auto_plot

    class Config:
        arbitrary_types_allowed = True


class FigureTask(BaseModel):

    # FigureTasks can be executed in parallel, and as such need to be self-contained
    # units of work.

    metadata: MadeResolvable[dict] = MISSING
    context: MadeResolvable[dict] = MISSING
    figure_spec: MadeResolvable[FigureSpec] = MISSING
    plot_config: MadeResolvable[PlotConfig] = MISSING
    plot_tasks: Optional[List[PlotTask]] = None

    def run(self):
        print(f'running figure task.\n{self=}')

    class Config:
        arbitrary_types_allowed = True


##────────────────────────────────────────────────────────────────────────────}}}


## {{{                       --     Figure Makers     --

FMaker = TypeVar("FMaker", bound="FigureMaker")


class FigureMaker(BaseModel):

    figure_spec: Optional[MadeResolvable[FigureSpec]] = None
    plot_config: Optional[MadeResolvable[PlotConfig]] = None
    metadata: Optional[MadeResolvable[Dict[str, Any]]] = None
    context: Optional[MadeResolvable[Dict[str, Any]]] = None

    def __repr__(self):
        return f'FigureMaker({self.figure_spec=}, {self.plot_config=}, {self.metadata=}, {self.context=})'

    def spawn_figure_task(self, **kw) -> FigureTask:
        figlog.debug(f'Plotconfig is {type(self.plot_config)}')
        return spawn(
            self,
            make_resolvable(FigureTask),
            inherit_attr=INHERITED_ATTRIBUTES['FigureMaker']['FigureTask'],
            **kw,
        )

    def can_make_tasks(self) -> bool:
        return hasattr(self, 'make_tasks')

    class Config:
        arbitrary_types_allowed = True


class SingleFigure(FigureMaker):
    """a figure maker that will create a single figure for each data"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:

        if not isinstance(data, (list, tuple)):
            data = [data]

        ftasks = []

        for pdata in data:
            ftask = self.spawn_figure_task(
                inherit_extra={
                    'context': {
                        'figure_makers': ['single'],
                    }
                },
                inherit_extra_args=MERGE_EXTEND_LISTS,
            )

            # we only plot on one ax
            ptask = PlotTask(data=pdata)

            ftasks.append(ftask)

        return ftasks


def as_list(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else obj


MERGE_EXTEND_LISTS = {
    'merge_mode': {
        'list': 'extend',
        'tuple': 'extend',
        'set': 'extend',
        ListConfig: 'extend',
    },
}


class ForEachData(FigureMaker):

    figure_maker: MadeResolvable[FigureMaker]

    # self.figure_maker = make_inheritable(
    # make_resolvable(FigureMaker, figure_maker),
    # self,
    # INHERITED_ATTRIBUTES['FigureMaker']['FigureMaker'],
    # )

    def model_post_init(self, *_):
        self.figure_maker = make_inheritable(
            self.figure_maker, self, INHERITED_ATTRIBUTES['FigureMaker']['FigureMaker']
        )

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []

        print(f'type of figure_maker={type(self.figure_maker)}')

        for d in as_list(data):
            fmaker = spawn(
                self,
                self.figure_maker,
                INHERITED_ATTRIBUTES['FigureMaker']['FigureMaker'],
                {
                    'context': {
                        'figure_makers': ['foreach'],
                    }
                },
                {'data': d},
                inherit_extra_args=MERGE_EXTEND_LISTS,
            )
            assert isinstance(fmaker, FigureMaker)
            assert fmaker.can_make_tasks(), f'FigureMaker {fmaker} cannot make tasks'
            tasks += fmaker.make_tasks(d)  # type: ignore

        return make_flat_list(tasks)


class SwitchFigure(FigureMaker):
    # a figure maker that will switch between different figure makers based on some condition

    condition: Any
    cases: Dict[Any, MadeResolvable[FigureMaker]]

    def model_post_init(self, *_):
        self.cases = {
            k: make_inheritable(v, self, INHERITED_ATTRIBUTES['FigureMaker']['FigureMaker'])
            for k, v in self.cases.items()
        }

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:

        cond = self.condition
        figlog.debug('SwitchFigure: condition is %s', cond)
        if callable(cond):
            cond = cond()

        if cond not in self.cases:
            if 'default' in self.cases:
                cond = 'default'
            else:
                raise ValueError(f'No case for condition {cond} or default for SwitchFigure')

        figmaker = spawn(
            self,
            self.cases[cond],
            None,
            {
                'context': {
                    'figure_makers': ['switch'],
                }
            },
            {'data': data},
            inherit_extra_args=MERGE_EXTEND_LISTS,
        )

        if not figmaker.can_make_tasks():
            # with %s instead of f
            figlog.debug('Case %s in SwitchFigure (%s) can\'t make tasks', cond, figmaker)
            return []

        return figmaker.make_tasks(data)  # type: ignore


class MultiFigure(FigureMaker):
    """A figure maker that will create multiple figures from the same data. Either by repeating
    the same figure maker n times, or by using different figure makers for each figure."""

    figure_makers: ListOrSingle[MadeResolvable[FigureMaker]] = []
    n_repeats: int = 1

    def model_post_init(self, *_):
        self.figure_makers = make_flat_list(self.figure_makers) * self.n_repeats

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []

        for i, figmaker in enumerate(self.figure_makers):

            fmaker = spawn(
                self,
                deepcopy(figmaker),
                INHERITED_ATTRIBUTES['FigureMaker']['FigureMaker'],
                {
                    'context': {
                        'multifigure_index': i,
                        'figure_makers': ['multi'],
                    }
                },
                {'data': data},
                inherit_extra_args=MERGE_EXTEND_LISTS,
            )

            if fmaker.can_make_tasks():
                tasks += fmaker.make_tasks(data)  # type: ignore

        return tasks


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     Data Sources     --


@inherit_attrs(
    InheritanceSpec(
        inherited_attrs=INHERITED_ATTRIBUTES['DataSource']['FigureMaker'],
        child_attrs=['figure_maker'],
    )
)
class DataSource(BaseModel):

    figure_spec: MadeResolvable[FigureSpec] = {}
    metadata: MadeResolvable[Dict[str, Any]] = {}
    context: MadeResolvable[Dict[str, Any]] = {}
    figure_maker: MadeResolvable[FigureMaker] = {}
    plot_config: MadeResolvable[PlotConfig] = {}

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')

    @classmethod
    def from_config(cls, ds_cfg: DictConfig) -> 'DataSource':
        # with %s instead of f
        datalog.debug('DataSource from_config:\n%s', short_conf(ds_cfg))
        return build_datasource_from_config(ds_cfg)

    def __repr__(self, indent=0):
        return f'{" "*indent}{self.__class__.__name__}'

    def make_figure_tasks(self) -> List[FigureTask]:
        """Instantiate the FigureMaker associated with this data source
        and call its make_tasks method to get the list of figure tasks to run."""

        figmaker = resolve(self.figure_maker, make_resolvers({'this': self}))
        # FIX: figmaker is still TypeVar(T)?:
        # -> it's the chaining of MadeResolvable?

        if figmaker.can_make_tasks():
            return figmaker.make_tasks(self.get_data())

        return []

    class Config:
        arbitrary_types_allowed = True


## {{{                           --     Group     --

OmConfig = Union[DictConfig, ListConfig]


class DataSourceGroup(DataSource):

    data_source: List[MadeResolvable[DataSource]] = []

    def model_post_init(self, *_):
        datalog.debug('Initializing DataSourceGroup with %s data sources', len(self.data_source))
        self.data_source = [
            make_inheritable(ds, self, INHERITED_ATTRIBUTES['DataSource']['DataSource'])
            for ds in self.data_source
        ]

    def get_resolved_sources(self) -> List[DataSource]:
        resolvers = make_resolvers({'this': self})
        return [resolve(src, resolvers) for src in self.data_source]

    def get_data(self) -> List[PlotData]:
        # recursively get the data from each sub data source
        return make_flat_list([src.get_data() for src in self.get_resolved_sources()])

    def make_figure_tasks(self) -> List[FigureTask]:
        # make tasks with the FigureMaker defined for this data source
        tasks = super().make_figure_tasks()
        # we also need to make figure tasks for each sub data source
        tasks += [src.make_figure_tasks() for src in self.get_resolved_sources()]
        return make_flat_list(tasks)

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        res = f'{indentstr}DataSourceGroup\n'
        for src in self.data_source:
            res += f'{indentstr} - ' + f'{src.__repr__(indent=indent+2)}\n'[indent + 2 :]
        return res


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                          --     Recipe     --


class RecipeDataSource(DataSource):

    def __init__(
        self,
        data_path: str,
        recipe_path: str,
        cache_dir: str,
        color_aliases: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.recipe_path = recipe_path
        self.cache_dir = cache_dir
        self.color_aliases = color_aliases

    def resolve(self) -> List[pu.PlotData]:
        lib = ut.load_lib()
        recipe_file = Path(self.recipe_path).expanduser().resolve()
        data_file = Path(self.data_path).expanduser().resolve()
        candidate_networks = bc.recipe.network_from_recipe(
            recipe_file, lib, inverse='shortest', use_cache=self.cache_dir
        )
        if len(candidate_networks) == 0:
            raise ValueError(f'No networks built for recipe {self.recipe_path}')
        assert len(candidate_networks) == 1
        X, Y = bc.recipe.get_network_XY(
            candidate_networks[0], data_file, color_aliases=self.color_aliases
        )
        # rescaler = hydra.utils.instantiate(self.rescaler)
        pdata = pu.extract_plot_data_from_network(
            candidate_networks[0], X, Y, rescaler=rescaler, protein_aliases=self.color_aliases
        )
        return [pdata]

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}RecipeDataSource({truncated_path(self.recipe_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                            --     Raw     --


class RawDataSource(DataSource):

    def __init__(
        self,
        data_path: Optional[str],
        output_column: Optional[str] = None,
        input_columns: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        output_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.output_column = output_column
        self.input_columns = input_columns
        self.input_names = input_names
        self.output_name = output_name

    def get_data(self) -> List[PlotData]:
        SUPPORTED_EXTENSIONS = ['.csv']

        data_file = Path(self.data_path).expanduser().resolve()

        if not data_file.exists():
            raise ValueError(f'Data path {data_file} does not exist')
        extension = data_file.suffix
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'''Unsupported extension {extension} for {data_file}.
                    Supported extensions: {SUPPORTED_EXTENSIONS}'''
            )
        else:
            if extension == '.csv':
                df = pd.read_csv(data_file, engine="pyarrow")
                assert isinstance(df, pd.DataFrame)
                assert self.input_columns is not None
                for col in self.input_columns:
                    if col not in df.columns:
                        raise ValueError(
                            f'Column {col} not found in {data_file}. Available: {df.columns}'
                        )
                assert self.output_column is not None
                if self.output_column not in df.columns:
                    raise ValueError(
                        f'''Column {self.output_column} not found in {data_file}.
                    Available: {df.columns}'''
                    )

                input_names = self.input_columns
                output_name = self.output_column

                if self.input_names is not None:
                    assert len(input_names) == len(self.input_columns)
                    input_names = self.input_names
                if self.output_name is not None:
                    assert isinstance(self.output_column, str)
                    output_name = self.output_name

                x = df[self.input_columns].to_numpy()
                y = df[self.output_column].to_numpy()

                return [
                    pu.PlotData(
                        x=x,
                        y=y,
                        input_names=input_names,
                        output_name=output_name,
                    )
                ]

            else:
                raise NotImplementedError(f'Extension {extension} not implemented')

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}RawDataSource({truncated_path(self.data_path)})'


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                            --     XP     --


class XPDataSource(DataSource):
    xp_path: str
    recipe_names: Optional[List[str]] = None
    source_type: str = 'xp'

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}XPDataSource({truncated_path(self.xp_path)})'


##────────────────────────────────────────────────────────────────────────────}}}


def as_datasourcegroup(ds_cfg: OmConfig) -> DictConfig:
    DATASOURCEGROUP_ATTRS = get_public_attrs(DataSourceGroup)
    datalog.debug('Converting %s to DataSourceGroup', ds_cfg)

    # operate on a copy of the config
    ds_cfg = OmegaConf.create(ds_cfg)
    new_cfg = OmegaConf.create()
    new_cfg['_target_'] = 'biocomptools.toollib.plot.DataSourceGroup'
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
        data_source = ds_cfg
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
    if isinstance(ds_cfg, DictConfig) and has_target(ds_cfg):
        target_class = get_target_class(ds_cfg)

    elif isinstance(ds_cfg, DictConfig) or isinstance(ds_cfg, ListConfig):
        ds_cfg = as_datasourcegroup(ds_cfg)
        target_class = get_target_class(ds_cfg)
        assert target_class == DataSourceGroup

    else:
        raise ValueError(f'Invalid DataSourceGroup config {ds_cfg}')

    return target_instantiate(ds_cfg)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     PlotJob     --


# TODO: we need to carry over the resolvers so that spawning/resolving
# objects inside a resolved object overrides/keep the parents resolvers
# explicitly using a context manager might be enough?


@inherit_attrs(
    InheritanceSpec(
        inherited_attrs=INHERITED_ATTRIBUTES['PlotJob']['DataSource'],
        child_attrs=['data_source'],
    )
)
@resolvable_attrs(
    ('figure_spec', FigureSpec),
    ('figure_maker', FigureMaker),
    ('plot_config', PlotConfig),
    ('data_source', DataSource),
    ('metadata', dict),
    ('context', dict),
)
@dataclass(kw_only=True)
class PlotJob:
    figure_spec: ResolvableOr[FigureSpec] = MISSING
    figure_maker: ResolvableOr[FigureMaker] = MISSING
    plot_config: ResolvableOr[PlotConfig] = MISSING
    data_source: ResolvableOr[DataSource] = MISSING
    metadata: ResolvableOr[Dict[str, Any]] = MISSING
    context: ResolvableOr[Dict[str, Any]] = MISSING
    extra: Dict[str, Any] = MISSING

    def generate_figure_tasks(self) -> List[FigureTask]:
        """resolve the data source and ask it to generate the corresponding figure tasks"""
        with omegaconf_resolvers(make_resolvers({'job': self, 'this': self})):
            return resolve(self.data_source).make_figure_tasks()

    @classmethod
    def from_config(cls, job_cfg: DictConfig) -> 'PlotJob':
        args = remove_non_ctor_args(cls, with_str_keys(job_cfg))
        return cls(**args)


##────────────────────────────────────────────────────────────────────────────}}}

