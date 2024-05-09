## {{{                          --     imports     -
from dataclasses import fields
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
from io import StringIO
import sys
from contextlib import contextmanager
from functools import partial
from dataclasses import is_dataclass
from copy import deepcopy
import logging
from rich import print as rprint
import pandas as pd
from pathlib import Path
import numpy as np

import biocomp as bc
from biocomp import utils as ut
from biocomp.utils import PartialFunction
from biocomp import datautils as du
from biocomp import plotutils as pu
from biocomp.plotutils import FigureSpec, PlotData

from biocomptools.toollib.common import ArbitraryTargetModel
from biocomptools.toollib.inheritable import (
    merged_into,
    merged_into_container,
    InheritableAttrsModel,
)
from biocomptools.toollib.resolvable import (
    resolved,
    resolvable,
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
from matplotlib.axes import Axes
import matplotlib as mpl
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field

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


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                           --     types     --

T = TypeVar('T')
ListOrSingle = Union[List[T], T]
DictLike = Union[Dict, DictConfig]

ResolvableDict = Annotated[ResolvableOr[dict], *resolvable(dict)]
ResolvableFigureSpec = Annotated[ResolvableOr[FigureSpec], *resolvable(FigureSpec)]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     misc utils     --


def as_list(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else obj


def obj_resolver(obj: Any, attr_path: str):
    attrs = attr_path.split('.')
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            obj = obj[attr]
    return obj


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


def make_resolvers(resolver_context: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    resolvers = {}
    if resolver_context is not None:
        for k, v in resolver_context.items():
            resolvers[k] = partial(obj_resolver, v)
    return resolvers


class resolvers:
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


MERGE_EXTEND_LISTS = {
    'merge_mode': {
        'list': 'extend',
        'tuple': 'extend',
        'set': 'extend',
        ListConfig: 'extend',
    },
}


def with_context(obj: T, context_dict: Dict[str, Any]) -> T:
    return merged_into(obj, {'context': context_dict}, 'context', **MERGE_EXTEND_LISTS)


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

    utlog.debug(f'spawning from unresolved: {unresolved_obj}')

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

    utlog.debug(f'after merging, unresolved: {unresolved_obj}')

    obj = resolved(unresolved_obj, make_resolvers(resolver_context))

    return obj


##────────────────────────────────────────────────────────────────────────────}}}
## {{{                 --     inherited attrs per class     --

INHERIT_ATTRS = {
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
    'FigureTask': {'PlotTask': ['metadata', 'plot_config', 'context']},
}


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


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}  # rc_params for matplotlib
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    general: Dict[str, Any] = {}  # general purpose parameters


ResolvablePlotConfig = Annotated[ResolvableOr[PlotConfig], *resolvable(PlotConfig)]

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     Tasks     --


ResolvablePartialFunction = Annotated[ResolvableOr[PartialFunction], *resolvable(PartialFunction)]

class PlotTask(ArbitraryTargetModel):

    ax: ListOrSingle[Axes] = []  # the axes to plot on
    context: ResolvableDict = {}  # inherited from parent FigureTask
    plot_config: ResolvablePlotConfig = Field(default_factory=PlotConfig)  # the plot config
    plot_method: ResolvablePartialFunction = {} # the function to call

    def run(self):
        if self.plot_method is not None:
            resolved(self.plot_method)()


ResolvablePlotTask = Annotated[ResolvableOr[PlotTask], *resolvable(PlotTask)]


class FigureTask(ArbitraryTargetModel):

    # FigureTasks can be executed in parallel so need to be self-contained

    metadata: ResolvableDict = {}
    context: ResolvableDict = {}
    figure_spec: ResolvableFigureSpec = {}
    plot_config: ResolvablePlotConfig = {}
    data: ListOrSingle[PlotData] = []  # the data to be plotted
    plot_tasks: List[ResolvablePlotTask] = []

    def run(self):
        for task in self.plot_tasks:
            print(f'Running task {task}...')
            resolved(task).run()

    def model_post_init(self, *_):
        self.plot_tasks = merged_into_container(
            self.plot_tasks,
            self,
            INHERIT_ATTRS['FigureTask']['PlotTask'],
        )


ResolvableFigureTask = Annotated[ResolvableOr[FigureTask], *resolvable(FigureTask)]


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                       --     Figure Makers     --

FMaker = TypeVar("FMaker", bound="FigureMaker")

class FigureMaker(ArbitraryTargetModel):

    figure_spec: ResolvableFigureSpec = {}
    plot_config: ResolvablePlotConfig = {}
    metadata: ResolvableDict = {}
    context: ResolvableDict = {}
    plot_tasks: List[ResolvablePlotTask] = []

    def spawn_figure_task(
        self, data, context=None, resolver_dict: Optional[dict[str, Any]] = None
    ) -> FigureTask:
        resolver_dict = resolver_dict or {}
        with resolvers(make_resolvers({'this': self, 'data': data, **resolver_dict})):
            task = with_context(
                spawn(
                    self,
                    make_resolvable(value=FigureTask(plot_tasks=self.plot_tasks, data=data)),
                    inherit_attr=INHERIT_ATTRS['FigureMaker']['FigureTask'],
                ),
                context or {},
            )
            return task

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        raise NotImplementedError('Cannot make tasks from a generic FigureMaker')


ResolvableFigureMaker = Annotated[ResolvableOr[FigureMaker], *resolvable(FigureMaker)]


class SingleFigure(FigureMaker):
    """A figure maker that will spawn a figure task for each data item"""

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        return [self.spawn_figure_task(d, {'figure_makers': ['single']}) for d in as_list(data)]


class ForEachData(FigureMaker, InheritableAttrsModel):
    """A figure maker that will call another figure maker for each data item"""

    figure_maker: ResolvableFigureMaker = {}

    _inherit = {'figure_maker': INHERIT_ATTRS['FigureMaker']['FigureMaker']}

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []
        for d in as_list(data):
            with resolvers(make_resolvers({'this': self, 'data': d})):
                tasks += with_context(
                    resolved(self.figure_maker), {'figure_makers': ['foreach']}
                ).make_tasks(d)
        return make_flat_list(tasks)


class SwitchFigure(FigureMaker):
    """A figure maker that will select between different figure makers based on condition"""

    condition: Any = 'default'
    cases: Dict[Any, ResolvableFigureMaker] = {}

    def model_post_init(self, *_):
        self.cases = merged_into_container(
            self.cases,
            self,
            INHERIT_ATTRS['FigureMaker']['FigureMaker'],
        )

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:

        cond = self.condition
        figlog.debug('SwitchFigure: condition is %s', cond)
        if callable(cond):
            cond = cond()

        if cond not in self.cases:
            raise ValueError(f'No case for condition {cond} or default for SwitchFigure')

        with resolvers(make_resolvers({'this': self, 'data': data})):
            figmaker = with_context(resolved(self.cases[cond]), {'figure_makers': ['switch']})

        try:
            return figmaker.make_tasks(data)
        except NotImplementedError:
            figlog.debug('Case %s in SwitchFigure can\'t make tasks', cond)
            return []


class MultiFigure(FigureMaker):
    """A figure maker that will call multiple figure makers from the same data. Either by repeating
    the same figure maker n times, or by using different figure makers for each figure."""

    figure_makers: ListOrSingle[ResolvableFigureMaker] = []
    n_repeats: int = 1

    def model_post_init(self, *_):
        self.figure_makers = (
            merged_into_container(
                as_list(self.figure_makers), self, INHERIT_ATTRS['FigureMaker']['FigureMaker']
            )
            * self.n_repeats
        )

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []

        with resolvers(make_resolvers({'this': self, 'data': data})):
            for i, figmaker in enumerate(self.figure_makers):
                fmaker = with_context(
                    resolved(figmaker), {'multifigure_index': i, 'figure_makers': ['multi']}
                )
                try:
                    tasks += fmaker.make_tasks(data)
                except NotImplementedError:
                    figlog.debug('Figure maker %s in MultiFigure can\'t make tasks', i)

        return tasks


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     Data Sources     --


class DataSource(InheritableAttrsModel):

    figure_maker: ResolvableFigureMaker = {}
    metadata: ResolvableDict = {}
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
        """Instantiate the FigureMaker associated with this data source
        and call its make_tasks method to get the list of figure tasks to run."""

        with resolvers(make_resolvers({'this': self})):
            figmaker = resolved(self.figure_maker)
            try:
                return figmaker.make_tasks(self.get_data())
            except NotImplementedError:
                figlog.debug('FigureMaker %s can\'t make tasks', figmaker)

        return []


class SpecializedDataSource(DataSource):

    @classmethod
    def from_config(cls, ds_cfg: DictConfig):
        return cls(**with_str_keys(ds_cfg))


ResolvableDataSource = Annotated[ResolvableOr[DataSource], *resolvable(DataSource)]

## {{{                           --     Group     --


class DataSourceGroup(DataSource):

    data_source: List[ResolvableDataSource] = []

    def model_post_init(self, *_):
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
            return make_flat_list([resolved(src).get_data() for src in self.data_source])

    def make_figure_tasks(self) -> List[FigureTask]:
        tasks = super().make_figure_tasks()  # make tasks with the FigureMaker for this source

        with resolvers(make_resolvers({'this': self})):  # add the tasks from each subsource
            tasks += [resolved(src).make_figure_tasks() for src in self.data_source]

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
    # TODO

    def __init__(
        self,
        data_path: str,
        recipe_path: str,
        cache_dir: str,
        color_aliases: Optional[Dict[str, str]] = None,
        input_order: Optional[
            List[str]
        ] = None,  # TODO make it work with protein names instead of column
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


class RawDataSource(SpecializedDataSource):

    data_path: Union[Path, str]
    output_column: Optional[str] = None
    input_columns: Optional[List[str]] = None
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


class XPDataSource(DataSource):
    # TODO

    xp_path: str
    recipe_names: Optional[List[str]] = None
    source_type: str = 'xp'

    def __repr__(self, indent=0):
        indentstr = ' ' * indent
        return f'{indentstr}XPDataSource({truncated_path(self.xp_path)})'


##────────────────────────────────────────────────────────────────────────────}}}


OmConfig = Union[DictConfig, ListConfig]


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
    # TODO: could be handled with a pydantic discriminator

    # we need to deal with a few cases here:
    # - we have a single data source (has a _target_ key)
    # - we have a dict of data sources: it's a DataSourceGroup (with names)
    # - we have a list of data sources: it's a *also* a DataSourceGroup

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


class PlotJob(InheritableAttrsModel):

    figure_spec: ResolvableFigureSpec = {}
    figure_maker: ResolvableFigureMaker = {}
    plot_config: ResolvablePlotConfig = {}
    data_source: ResolvableDataSource = {}
    metadata: ResolvableDict = {}
    context: ResolvableDict = {}
    extra: Dict[str, Any] = {}

    _inherit = {'data_source': INHERIT_ATTRS['PlotJob']['DataSource']}

    def generate_figure_tasks(self) -> List[FigureTask]:
        with resolvers(make_resolvers({'job': self, 'this': self})):
            return resolved(self.data_source).make_figure_tasks()


##────────────────────────────────────────────────────────────────────────────}}}
