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

log = logging.getLogger('biocomptools.biocomplot')
log.setLevel(logging.DEBUG)

from biocomp.plotutils import FigureSpec
from biocomp.plotting.plotting_core import PlotData

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                         --     ramblings     --

# a Job declares a list of DataSources (can be nested through DataSourceGroup)
# each DataSource declares a FigureMaker, which will produce PlotTasks from the data
# A PlotTask contains some PlotData, an Axes, a PlotConfig to be used by the plotting function,
# and an entry_point, aka which plotting function to use (by default, it's auto_plot, or maybe
# I should switch to directly defining something like "smooth" or "histogram" or "scatter" etc.)


# Everything relies on successive task overrides. A Task being a self-contained unit of work,
# that has everything it needs to be executed.

# One big pain point with hydra is that the interpolation of variables happens on the final
# config object, which means we can't use variable paths that are relative to the current file or node.
#
# one way around that is to use the relative paths (...path.var), but it gets messy quickly.
# After pondering if I should roll my own config system on top of OmegaConf that treats modules as *true* modules
# and not this weird default-list + assembly of a config object that has no knowledge of the modules composed in it,
# I decided that there might be a simple way to augment hydras interpolation system with a few custom resolvers
# ${this: path.to.var} -> resolves to whatever the instanciated wrapper object is
# ${include: path/to/module, resolve=false} ->
#   load path/to/module with hydra and generate the config object, then append it to the current config


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --
T = TypeVar('T')
U = TypeVar('U')
ListOrSingle = Union[List[T], T]
Pair = Tuple[T, T]
DictOrList = Union[Dict[U, T], List[T]]
DictLike = Union[Dict, DictConfig]
AnyConfig = Union[DictConfig, ListConfig]
##────────────────────────────────────────────────────────────────────────────}}}


## {{{                        --     misc utils     --


def get_public_attrs(obj):
    if is_dataclass(obj):
        return [f.name for f in fields(obj)]

    # if it's a type, try to instantiate it
    if isinstance(obj, type):
        obj = obj()

    return [a for a in dir(obj) if not a.startswith('__')]


def as_dict(cfg: Optional[DictConfig]) -> Dict:
    if cfg is None:
        return {}
    return OmegaConf.to_container(cfg, resolve=False)


def with_str_keys(d: DictLike) -> Dict:
    return {str(k): v for k, v in d.items()}


def make_flat_list(l):
    if not isinstance(l, (list, tuple)):
        return [l]
    return [item for sublist in l for item in sublist]


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
    return target_class(**with_str_keys(kwargs))

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     resolve utils     --

CONTEXT_AWARE_RESOLVERS = ['np', 'this', 'plot_task', 'data', 'figure_task']

# A Resolvable[T] is a wrapper around a T that can be resolved by OmegaConf at the last minute
# by providing the right context (this, plot_task, data, figure_task, etc.)
# When not resolved, it carries its DictConfig representation
# and a constructor to build the object from it.


def build_from_config(cls: Type[T], cfg: DictConfig) -> T:
    # check if it has a _target_ key
    if has_target(cfg):
        subclass = get_target_class(cfg)
        assert issubclass(subclass, cls), f'Invalid target class {subclass} for {cls}'

    ctor_args = filter_out_non_ctor_args(cls, with_str_keys(cfg))
    return cls(**ctor_args)


def noop_interpolation(resolver_name, *args, **kwargs):
    # allows to defer resolve of context-dependent interpolations until we have the right context
    if args:
        args = [str(a) for a in args]
        res = f'${{{resolver_name}: {",".join(args)}}}'
        return res
    return '${' + resolver_name + '}'


class omegaconf_resolvers:
    # a context manager to temporarily register resolvers
    def __init__(self, resolvers: Dict[str, Callable]):
        self.resolvers = resolvers

    def __enter__(self):
        for key, value in self.resolvers.items():
            OmegaConf.clear_resolver(key)
            OmegaConf.register_resolver(key, value)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for key in self.resolvers.keys():
            OmegaConf.clear_resolver(key)


def short_conf(conf: DictConfig) -> str:
    long = OmegaConf.to_yaml(conf, resolve=False)
    return long[:100] + ' [...] \n' if len(long) > 100 else long


@dataclass
class Resolvable(Generic[T]):
    constructor: Callable[..., T] = MISSING
    config: Optional[DictConfig] = None
    # debug info:
    name: Optional[str] = None
    target_type: Optional[Type] = None

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        if self.config is None:
            return f'{indentstr}Resolvable({self.name})[->{self.target_type}] (empty)'

        confstr = short_conf(self.config)
        indented_conf = indentstr + confstr.replace('\n', '\n ' + indentstr)
        return f'{indentstr}Resolvable({self.name})->{self.target_type.__name__}\n{indentstr}config:\n{indented_conf}'


ResolvableOr = Union[Resolvable[T], T, AnyConfig]

def resolve(resolvable: ResolvableOr[T], resolvers=None) -> T:
    """
    Resolve a Resolvable object with the given resolvers,
    i.e. returns the object constructed by calling the constructor with the config
    using the provided resolvers.
    """

    if not isinstance(resolvable, Resolvable):
        return resolvable

    if resolvers is None:
        resolvers = {}

    log.debug(f'Resolving {resolvable}')
    with omegaconf_resolvers(resolvers):
        if resolvable.config is None:
            log.debug(f'No config found for {resolvable}, returning default')
            return resolvable.constructor(OmegaConf.create())
        else:
            return resolvable.constructor(resolvable.config)


def make_resolvable(
    target_type: Type[T],
    value: Union[DictConfig, T],
    name=None,
    clsname=None,
    **kw,
) -> ResolvableOr[T]:

    typename = target_type.__name__

    log.debug(f'Making {typename} {name if name else "resolvable"} in {clsname}. {value=}')

    # we need to set the value of the config in the resolvable object
    # OR, if it is already of the right type, there's no need to wrap it
    if isinstance(value, target_type):
        log.info(f'Attribute {name} already resolved to {target_type}, skipping')
        return value

    if value is MISSING:
        raise ValueError(f'Missing value for {name} in {clsname}')

    if not isinstance(value, AnyConfig) and (value is not None):
        raise ValueError(f'Invalid init value for {name}: {type(value)}, {value=}')


    # if type has a from_config constructor, use it
    if hasattr(target_type, 'from_config'):
        constructor = getattr(target_type, 'from_config')
    else:
        constructor = partial(build_from_config, target_type)

    wrapped_value = Resolvable(
        constructor=constructor, config=value, name=name, target_type=target_type, **kw
    )

    return wrapped_value


# ╭─────────────────────────────╮
# │      @resolvable_attrs      │
# ╰─────────────────────────────╯
def resolvable_attrs(*attrs):
    """

    A decorator to make some attributes to a dataclass resolvable.

    Args:
        attrs (List[str]): a list of attribute names to make resolvable

    A resolvable attribute keeps its DictConfig representation until it is
    resolved, at which point it calls the constructor with the resolved config.
    Therefore a resolved attribute needs to store 2 things:
    - the unresolved config (i.e. with ${OmegaConf:interpolation variable} type
      strings in it)
    - the constructor to build the object from the config

    Why resolvable attributes?
    --------------------------

    This declarative plotting system relies on modular, nested configurations.
    In order to customize figures and plots, users will often need to access
    the context of the plot task, the figure task, the data, etc. (e.g. to set
    the title of a plot, ...). The desired piece of context is oftentime not
    available until instanciation of many things down the pipeline.

    For example, if you need to know the index of the figure task in the list
    of all figure tasks to set the title, you can use the ${figure_task:index}
    interpolation variable. Or if you want to access some metadata information
    that will only be added after the data is loaded, you can use ${this:
    metadata.some_key}. But you can't resolve any of that at the time the
    configuration is parsed, because the figure task doesn't exist yet.

    How this decorator works
    ------------------------

    1. Handles constructions
    It wraps the decorated class' __init__ method with a new method (distinct
    from __post_init__ if it exists) whose role is to turn "normal" attributes
    into Resolvable objects.

    We first need to know the desired resolved type of the attribute. We need
    to explicitely declare the type of the attribute in the decorator using a
    tuple (attr_name, type)

    So we loop over the attributes that were declared as resolvable, detect
    and wrap them into a Resolvable class.

     We assume any preexisting value of the attribute is either MISSING, or
     a ConfigLike object (DictConfig, ListConfig, etc.)

    MAYBE:
    2. Handles resolution on access with the __getattr__ method


    """

    def decorator(cls):
        log.debug(f'Setting up {cls.__name__} with resolvable attrs')

        for attr_specs in attrs:
            if not isinstance(attr_specs, (tuple, list)):
                raise ValueError(
                    f'In {cls.__name__}, invalid resolvable attribute spec {attr_specs}. Need a tuple'
                )

        # get_resolvers = cls.get_resolvers if hasattr(cls, 'get_resolvers') else None
        # # lazily resolve the attribute when accessed
        # def resolvable_getattribute(self, name: str, get_resolvers=get_resolvers):
        # # attr = getattr(self, name) -> can't do that, it would recurse infinitely
        # attr = object.__getattribute__(self, name)
        # msg = f'Getting attribute {name} from {cls.__name__} with type {type(attr)}: {attr}.'
        # if isinstance(attr, Resolvable):
        # resolvers = get_resolvers() if get_resolvers is not None else {}
        # attr = resolve(attr, resolvers=resolvers)
        # setattr(self, name, attr)
        # log.debug(f'{msg} Attribute {name} is resolvable, resolved to {attr}')
        # return attr
        # log.debug(f'{msg} Attribute {name} is not resolvable, returning as is')
        # return attr

        original_init = cls.__init__

        def resolvable_init(self, *args, **kwargs):

            original_init(self, *args, **kwargs)

            log.debug(f'Initializing {cls.__name__} with resolvable attrs')

            for attr_name, target_type in attrs:

                assert isinstance(attr_name, str)
                if not hasattr(self, attr_name):
                    raise ValueError(f'Attribute {attr_name} not found in {cls}')

                attr = object.__getattribute__(self, attr_name)  # current value of the attribute

                if target_type is None:
                    target_type = type(attr)
                    if get_origin(target_type) is not None:
                        target_type = get_origin(target_type)

                assert (
                    target_type is not None
                ), f'Could not determine type for {attr_name} in {cls.__name__}'

                if attr is MISSING:
                    attr = None

                wrapped_value = make_resolvable(
                    target_type=target_type,
                    value=attr,
                    name=attr_name,
                    clsname=cls.__name__,
                )
                setattr(self, attr_name, wrapped_value)

        cls.__init__ = resolvable_init

        return cls

    return decorator


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     inherit utils     --


def dict_like(obj) -> bool:
    return isinstance(obj, dict) or isinstance(obj, DictConfig)


def conf_merge(parent: DictConfig, child: DictConfig) -> DictConfig:
    parent_dict = OmegaConf.to_container(parent, resolve=False)
    child_dict = OmegaConf.to_container(child, resolve=False)
    merged = ut.updated_dict(parent_dict, child_dict)
    assert isinstance(merged, dict)
    return OmegaConf.create(merged)


@dataclass
class InheritanceSpec:
    inherited_attrs: Iterable[str]
    child_attrs: Iterable[str]


# ╭───────────────────────────╮
# │      @inherit_attrs       │
# ╰───────────────────────────╯
def inherit_attrs(*inheritance_specs: InheritanceSpec):
    """
    A decorator to make some attributes to a dataclass inheritable.

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

    """

    # turn the inheritance specs into 2 dictionaries:
    # child -> inherited attributes and inherited attribute -> children

    attribute_to_children = {}
    child_to_attributes = {}

    for spec in inheritance_specs:
        for child_attr in spec.child_attrs:
            if child_attr not in child_to_attributes:
                child_to_attributes[child_attr] = []
            child_to_attributes[child_attr] += spec.inherited_attrs

        for inherited_attr in spec.inherited_attrs:
            if inherited_attr not in attribute_to_children:
                attribute_to_children[inherited_attr] = []
            attribute_to_children[inherited_attr] += spec.child_attrs

    for k, v in attribute_to_children.items():
        attribute_to_children[k] = list(set(v))

    for k, v in child_to_attributes.items():
        child_to_attributes[k] = list(set(v))

    def decorator(cls):

        log.debug(f'Setting up {cls.__name__} with inheritable attrs')

        def merge_attribute_into_children(
            parent_config: DictConfig, cls_instance, children_names: List[str], parent_name: str
        ):
            # for child_name in children_names:
            # child_attr = getattr(cls_instance, child_name)
            # if isinstance(child_attr, Resolvable):
            # child_attr.config = child_attr.config if child_attr.config else OmegaConf.create()
            # attr_in_child_config = child_attr.config.get(parent_name, None)
            # merged = conf_merge(parent_config, attr_in_child_config)
            # log.debug(f'Merging {parent_config=} into {child_name=} in {cls.__name__}: {merged}')
            # with open_dict(child_attr.config):
            # child_attr.config[parent_name] = merged
            # else:
            # log.debug(f'{child_name=} in {cls.__name__} is not Resolvable. Skipping')
            # return parent_config
            raise NotImplementedError('Not implemented yet')

        def merge_attributes_into_child(
            child_config: DictConfig, cls_instance, parent_names: List[str]
        ):
            # about to resolve a child attribute, merge all parent attributes into it
            for parent_name in parent_names:
                parent_attr = getattr(cls_instance, parent_name)
                if isinstance(parent_attr, Resolvable):
                    log.debug(f'In {cls.__name__}: merging {parent_name=} config into child')
                    parent_config = parent_attr.config if parent_attr.config else OmegaConf.create()
                    attr_in_child_config = child_config.get(parent_name, OmegaConf.create())
                    merged = conf_merge(parent_config, attr_in_child_config)
                    with open_dict(child_config):
                        child_config[parent_name] = merged
                    log.debug(f'child_config[{parent_name}]=\n{short_conf(merged)}')
                else:
                    log.debug(f'{parent_name=} in {cls.__name__} is not Resolvable. Skipping')
            return child_config

        def wrap_resolvable_constructor(obj: Resolvable[T], fn: Callable):

            base_constructor = obj.constructor

            def wrapped_constructor(
                config: DictConfig, obj=obj, fn=fn, base_constructor=base_constructor
            ):
                log.debug(f'Wrapped constructor of {obj.name}, from {cls.__name__}')
                merged_config = fn(config)
                return base_constructor(merged_config)

            obj.constructor = wrapped_constructor

        original_init = cls.__init__

        def inherit_init(self, *args, **kwargs):

            original_init(self, *args, **kwargs)

            log.debug(f'Initializing {cls.__name__} with inheritable attrs')

            for attr_name, children in attribute_to_children.items():
                attr = getattr(self, attr_name)
                assert isinstance(
                    attr, Resolvable
                ), f'{attr_name=} in {cls.__name__} is not Resolvable'
                log.debug(
                    f'Wrapping {attr_name=} constructor in {cls.__name__} with children {children}'
                )
                wrap_resolvable_constructor(
                    attr,
                    partial(
                        merge_attribute_into_children, cls_instance=self, children_names=children
                    ),
                )

            for child_name, parents in child_to_attributes.items():
                attr = getattr(self, child_name)
                assert isinstance(attr, Resolvable)
                log.debug(f'Wrapping {child_name=} from {cls.__name__}. {parents=}')
                wrap_resolvable_constructor(
                    attr,
                    partial(merge_attributes_into_child, cls_instance=self, parent_names=parents),
                )

        cls.__init__ = inherit_init

        return cls

    return decorator


##────────────────────────────────────────────────────────────────────────────}}}

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


## {{{                        --     PlotConfig     --
@dataclass
class PlotConfig:
    # rc_params for matplotlib
    rc_context: Dict[str, Any] = field(default_factory=dict)
    # nested parameters for the plotting function
    callstack_params: Dict[str, Any] = field(default_factory=dict)
    # for general purpose storage of parameters
    general: Dict[str, Any] = field(default_factory=dict)
    plot_method: Callable = pu.auto_plot
    data_rescaler: Optional[Any] = None


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     Tasks     --
@dataclass
class PlotTask:
    metadata: Dict[str, Any] = MISSING
    context: Dict[str, Any] = MISSING
    data: ListOrSingle[PlotData] = MISSING  # the data to be plotted
    ax: ListOrSingle[Axes] = MISSING  # the axes to plot on
    plot_config: PlotConfig = MISSING


@resolvable_attrs(('metadata', dict), ('context', dict))
@dataclass(kw_only=True)
class FigureTask:
    metadata: ResolvableOr[Dict[str, Any]] = MISSING
    context: ResolvableOr[Dict[str, Any]] = MISSING
    figure_spec: FigureSpec = MISSING
    plot_config: ResolvableOr[PlotConfig] = MISSING
    plot_tasks: Optional[List[PlotTask]] = None


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Figure Makers     --


@resolvable_attrs(
    ('figure_spec', FigureSpec), ('plot_config', PlotConfig), ('metadata', dict), ('context', dict)
)
class FigureMaker:

    def __init__(
        self,
        figure_spec: Optional[ResolvableOr[FigureSpec]] = None,
        plot_config: Optional[ResolvableOr[PlotConfig]] = None,
        metadata: Optional[ResolvableOr[Dict[str, Any]]] = None,
        context: Optional[ResolvableOr[Dict[str, Any]]] = None,
    ):
        self.figure_spec = figure_spec
        self.plot_config = plot_config
        self.metadata = metadata
        self.context = context


    def make_figure_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        raise NotImplementedError('Subclasses must implement make_plot_tasks')


class ForEachData(FigureMaker):

    def __init__(
        self,
        figure_maker: ResolvableOr[FigureMaker],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.figure_maker = figure_maker

    def make_figure_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        if not isinstance(data, (list, tuple)):
            data = [data]

        tasks = []

        # TODO
        ...

        return tasks


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     Data Sources     --


@inherit_attrs(
    InheritanceSpec(
        inherited_attrs=['metadata', 'context', 'plot_config', 'figure_spec'],
        child_attrs=['figure_maker'],
    )
)
@resolvable_attrs(
    ('figure_spec', FigureSpec),
    ('figure_maker', FigureMaker),
    ('plot_config', PlotConfig),
    ('metadata', dict),
    ('context', dict),
)
class DataSource:

    def __init__(
        self,
        figure_spec: Optional[ResolvableOr[FigureSpec]] = None,
        figure_maker: Optional[ResolvableOr[FigureMaker]] = None,
        plot_config: Optional[ResolvableOr[PlotConfig]] = None,
        metadata: Optional[ResolvableOr[Dict[str, Any]]] = None,
        context: Optional[ResolvableOr[Dict[str, Any]]] = None,
    ):
        self.figure_spec = figure_spec
        self.figure_maker = figure_maker
        self.plot_config = plot_config
        self.metadata = metadata
        self.context = context

    def get_data(self) -> List[PlotData]:
        raise NotImplementedError('Subclasses must implement get_data')

    @classmethod
    def from_config(cls, ds_cfg: DictConfig) -> 'DataSource':
        log.debug(f'DataSource from_config:\n{short_conf(ds_cfg)}')
        return build_datasource_from_config(ds_cfg)

    def __repr__(self, indent=0):
        return f'{" "*indent}{self.__class__.__name__}'


## {{{                           --     Group     --

OmConfig = Union[DictConfig, ListConfig]


class DataSourceGroup(DataSource):

    def __init__(self, data_source: Optional[List[DataSource]] = None, **kwargs):
        super().__init__(**kwargs)
        data_source = data_source or []
        log.debug(f'Initializing DataSourceGroup with {len(data_source)} data sources')
        log.debug(f'{data_source=}')
        self.data_source = [
            make_resolvable(DataSource, ds, name=f'group.data_source[{i}]')
            for i, ds in enumerate(data_source)
        ]

    def get_data(self) -> List[PlotData]:
        return make_flat_list([resolve(src).get_data() for src in self.data_source])

    def __repr__(self, indent=0):
        # uses indentations to show the hierarchy of data sources
        indentstr = ' ' * indent
        res = f'{indentstr}DataSourceGroup\n'
        for src in self.data_source:
            res += f'{indentstr} - ' + f'{src.__repr__(indent=indent+2)}\n'[indent + 2 :]
        return res


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                          --     Recipe     --
@dataclass(kw_only=True)
class RecipeDataSource(DataSource):
    data_path: str
    recipe_path: str
    source_type: str = 'recipe'
    cache_dir = cm.config.paths.cache.networks
    color_aliases = cm.config.protein_aliases

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


@dataclass(kw_only=True)
class RawDataSource(DataSource):
    data_path: str
    output_column: List[str]
    input_columns: List[str]
    input_names: Optional[List[str]] = None
    output_name: Optional[str] = None

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


@dataclass(kw_only=True)
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
    log.debug(f'DATASOURCEGROUP_ATTRS: {DATASOURCEGROUP_ATTRS}')

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

    log.debug(f'Converted:\n{short_conf(ds_cfg)}\nto DataSourceGroup:\n{short_conf(new_cfg)}')
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

    log.debug(f'Target class for datasource: {target_class}')
    ctor_args = filter_out_non_ctor_args(target_class, with_str_keys(ds_cfg))
    log.debug(f'Building {target_class} with args {ctor_args}')
    return target_class(**ctor_args)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     PlotJob     --

def filter_out_non_ctor_args(cls, kwargs):
    ctor_args = {}
    for k, v in kwargs.items():
        if k in get_public_attrs(cls):
            ctor_args[k] = v
    return ctor_args


@inherit_attrs(
    InheritanceSpec(
        inherited_attrs=['metadata', 'context', 'plot_config', 'figure_spec', 'figure_maker'],
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

    def generate_figure_tasks(self) -> List[FigureTask]:
        return generate_figure_tasks(self.data_source)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     task generation     --

def empty_config(r: Optional[pl.ResolvableOr[pl.T]]):
    if r is None:
        return True
    if not isinstance(r, Resolvable):
        return False
    return r.config is None or r.config == {} or r.config == MISSING


def generate_figure_tasks(src: ResolvableOr[DataSource]):
    if not isinstance(src, DataSource):
        src = resolve(src)
    assert isinstance(src, DataSource), f'Expected DataSource, got {type(src)}, {src=}'
    tasks = []
    log.debug(f'Generating figure tasks for {src}. Figure maker: {src.figure_maker}')
    if not empty_config(src.figure_maker):
        src.figure_maker = pl.resolve(src.figure_maker)
        data = src.get_data()
        log.debug(f'Got data: {data} and figure maker: {src.figure_maker}')
        tasks += src.figure_maker.make_figure_tasks(data)
    else:
        log.debug('No figure maker')
    if hasattr(src, 'data_source'):
        log.debug('Going down the hierarchy')
        next_src = getattr(src, 'data_source')
        if not isinstance(next_src, list):
            next_src = [next_src]
        for s in next_src:
            tasks += generate_figure_tasks(s)
    return tasks


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     archive     --
## {{{                   --     data source resolvers     --


def resolve_xp_data_source(data_source: XPDataSource) -> List[pu.PlotData]:
    assert data_source.source_type == 'xp'
    assert data_source.xp_path is not None
    if data_source.xp_path.startswith('db:'):
        return get_plot_data_from_xp_in_db(data_source.xp_path[5:])
    else:
        raise NotImplementedError(
            'Only supports plotting xp that are in the database (xp_path starts with "db:")'
        )


def get_plot_data_from_xp_in_db(
    xpname: str,
    input_order: Optional[Sequence[int]] = None,
    protein_aliases: Optional[Dict[str, str]] = None,
) -> List[pu.PlotData]:

    lib = ut.load_lib()
    netdf = cm.table_to_df('network')
    assert isinstance(netdf, pd.DataFrame)
    if xpname not in netdf['xp'].values:
        raise ValueError(f'No networks found for xp {xpname}')

    netdf = netdf[netdf['xp'] == xpname]
    nets_with_data = netdf[netdf['data_file'] != 'None']
    log.debug(f'Found {len(netdf)} networks for xp {xpname}, {len(nets_with_data)} with data')
    netdf = nets_with_data

    load_errors = {}

    def error_handler(net_name, e):
        load_errors[net_name] = f'{e.__class__.__name__}: {e}'
        return None, None, None

    netdf['network_obj'], netdf['X'], netdf['Y'] = cm.load_networks_and_data(
        netdf, lib, error_handler=error_handler
    )

    rescaler = hydra.utils.instantiate(cfg.data_config.rescaler)
    xlist = netdf['X'].tolist()
    ylist = netdf['Y'].tolist()
    netlist = netdf['network_obj'].tolist()

    return [
        pc.extract_plot_data_from_network(
            network=n,
            x=x,
            y=y,
            rescaler=rescaler,
            input_order=input_order,
            protein_aliases=protein_aliases,
        )
        for n, x, y in zip(netlist, xlist, ylist)
    ]


def instantiate_data_source(data_source) -> DataSource:
    if data_source.source_type == 'xp':
        return XPDataSource(**data_source)
    elif data_source.source_type == 'recipe':
        return RecipeDataSource(**data_source)
    elif data_source.source_type == 'raw':
        return RawDataSource(**data_source)
    else:
        raise ValueError(f'Unsupported data source type {data_source.source_type}')


def build_data_source(data_source: DataSource) -> List[pu.PlotData]:
    if data_source.source_type == 'xp':
        assert isinstance(data_source, XPDataSource)
        return resolve_xp_data_source(data_source)
    elif data_source.source_type == 'recipe':
        assert isinstance(data_source, RecipeDataSource)
        return resolve_recipe_data_source(data_source)
    elif data_source.source_type == 'raw':
        assert isinstance(
            data_source, RawDataSource
        ), f'Expected RawDataSource, got {type(data_source)}'
        return resolve_raw_data_source(data_source)
    else:
        raise ValueError(f'Unsupported data source type {data_source.source_type}')


##────────────────────────────────────────────────────────────────────────────}}}
# no! we can't do that.
# we want to resolve things at the last minute.
# plot_config, for example, might want to refer to its general parameters
# that could be modified by some plottask or figuretask

# the problem is that we might need the whole context to resolve things
# inside the main configuration but if we wait, it's going to get more specialized
# and narrow.

# we could sort of "carry" the whole file?
# -> ugly
# or we can make sure to use mostly context-aware resolvers (i.e. this)
# and the occasional "normal" interpolation is some risky gambit
# with the only guarantee being that we try to resolve it at the last minute.

# OmegaConf.resolve(job_cfg)

## {{{                --     plot task making functions     --


def cleanup_private_vars(d, prefix='__hydra_hack_'):
    for k in list(d.keys()):
        if k.startswith(prefix):
            del d[k]


def make_plot_tasks(plot_job: PlotJob) -> List[PlotTask]:
    data_sources = [instantiate_data_source(ds) for ds in plot_job.data_sources]
    for i in range(len(data_sources)):
        if 'rescaler' not in plot_job.data_sources[i]:
            data_sources[i].rescaler = plot_job.rescaler

    print(f'Instantiated {len(data_sources)} data sources')
    # print their rescalers type:
    for ds in data_sources:
        print(f'data source {ds.source_type} has rescaler {type(ds.rescaler)}')

    plot_data = []
    for i, d in enumerate(data_sources):
        # add task_index to metadata
        if d.metadata is None:
            d.metadata = {}
        d.metadata['task_index'] = i
        plot_data += build_data_source(d)

    plot_job_dict = OmegaConf.to_container(plot_job, resolve=False)
    assert isinstance(plot_job_dict, dict)
    pre_tasks = [
        OmegaConf.create(
            {
                'figure': plot_job_dict['figure'],
                'plot_config': plot_job_dict['plot_config'],
                'output_path': plot_job_dict['output_path'],
                'metadata': d.metadata,
            }
        )
        for d in data_sources
    ]

    # we want to apply any data_source overrides to each task
    # we need to do that on the dict version of the plot_task
    overriden_tasks = []
    for task, ds in zip(pre_tasks, data_sources):
        if ds.overrides is not None:
            for override in ds.overrides:
                override = OmegaConf.create(override)
                print(f'Applying override {OmegaConf.to_yaml(override)}')
                task = OmegaConf.merge(task, override)
        # now we need to resolve the plot_config
        task.plot_config = OmegaConf.create(task.plot_config)
        overriden_tasks.append(task)

    tasks = [
        PlotTask(data=d, figure=t.figure, plot_config=t.plot_config, output_path=t.output_path)
        for d, t in zip(plot_data, overriden_tasks)
    ]

    return tasks


##────────────────────────────────────────────────────────────────────────────}}}
##────────────────────────────────────────────────────────────────────────────}}}
