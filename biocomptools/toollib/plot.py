## {{{                          --     imports     -
import glob
from dataclasses import fields
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
from pydantic.functional_validators import BeforeValidator
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
from biocomp.utils import PartialFunction, flatten, as_list
from biocomp import datautils as du
from biocomp import plotutils as pu
from biocomp.plotutils import FigureSpec, PlotData, FigAx
from biocomp.datautils import DataRescaler

import biocomptools.toollib.common as cm
from biocomptools.toollib.common import ArbitraryTargetModel
from biocomptools.toollib.inheritable import (
    merged_into,
    merged_into_container,
    InheritableAttrsModel,
)
from biocomptools.toollib.resolvable import (
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
# reslog.setLevel(logging.DEBUG)
# figlog.setLevel(logging.DEBUG)
# datalog.setLevel(logging.DEBUG)

# disable matplotlib logging:
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
# same for pillow
pillow_logger = logging.getLogger("PIL")
pillow_logger.setLevel(logging.WARNING)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     types     --

T = TypeVar('T')
ListOrSingle = Union[List[T], T]
DictLike = Union[Dict, DictConfig]


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
    print(f'func_resolver: {funcname}, {args=}, {kwargs=}')
    f = ut.PartialFunction(func=funcname, modules=[THIS_MODULE_NAME])
    res = f(*args, **kwargs)
    print(f'func_resolver: {res}')
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
    rescaler: ValidatedRescaler = Field(default_factory=du.NoOpRescaler)

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

    auto_callstack_bind: bool = True  # whether to automatically bind callstack params

    def model_post_init(self, *a):
        super().model_post_init(*a)

    def run(self):
        with resolvers({**make_resolvers({'ptask': self, 'this': self}), **BASE_RESOLVERS}):
            pc = resolved(self.plot_config)
            return pc.prepare_func(resolved(self.plot_method), self.auto_callstack_bind)()


ResolvablePlotTask = Annotated[ResolvableOr[PlotTask], *resolvable(PlotTask)]


def make_video(input_file_pattern, output_file, fps=30, crf=17, vcodec='libx264'):
    import os

    cmd = f'ffmpeg -y -r {fps} -i "{input_file_pattern}" -crf {crf} -vcodec {vcodec} -vf "scale=iw:ih,format=yuv420p,crop=trunc(iw/2)*2:trunc(ih/2)*2" "{output_file}"'
    print(f'Running command: {cmd}')
    os.system(cmd)
    print(f'Video created at {output_file}')


from tqdm import tqdm


class FigureTask(ut.ArbitraryModel):

    # FigureTasks can be executed in parallel so need to be self-contained

    data: Annotated[List[PlotData], BeforeValidator(as_list)] = []  # the data to be plotted
    context: ResolvableDict = {}
    figure_spec: ResolvableFigureSpec = {}
    plot_config: ResolvablePlotConfig = {}
    plot_tasks: List[ResolvablePlotTask] = []
    extra_tasks: List[ResolvableOr[PartialFunction]] = []
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
        fd = ut.flatten(self.data)
        return fd

    def run(self):
        with resolvers(make_resolvers({'ftask': self, 'this': self})):
            self.figure_spec = resolved(self.figure_spec)
            assert isinstance(self.figure_spec, FigureSpec)
            with mpl.rc_context(rc=resolved(self.plot_config).rc_context):
                figax = self.figure_spec.make_figure()  # type: ignore
                ntasks = self.n_plot_tasks or figax.n_axes
                self.context['n_plot_tasks'] = ntasks
                results = [t.run() for t in tqdm(self.make_plot_tasks(figax), desc='Plotting')]
                for task in self.extra_tasks:
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

## {{{                       --     Figure Makers     --

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
        return flatten(tasks)


class SwitchFigure(FigureMaker):
    """A figure maker that will select between different figure makers based on condition"""

    condition: Any = 'default'
    cases: Dict[Any, ResolvableFigureMaker] = {}

    def model_post_init(self, *a):
        super().model_post_init(*a)
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

    def model_post_init(self, *a):
        super().model_post_init(*a)
        self.figure_makers = merged_into_container(
            as_list(self.figure_makers), self, INHERIT_ATTRS['FigureMaker']['FigureMaker']
        )
        assert isinstance(self.figure_makers, list)
        base_len = len(self.figure_makers)
        self.figure_makers = self.figure_makers * self.n_repeats
        assert len(self.figure_makers) == base_len * self.n_repeats

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []
        with resolvers(make_resolvers({'this': self, 'data': data})):
            for i, figmaker in enumerate(self.figure_makers):
                fmaker = with_context(
                    resolved(figmaker),
                    {
                        'multifigure_index': i,
                        'figure_makers': ['multi'],
                        'multifigure_n_repeats': self.n_repeats,
                    },
                )
                try:
                    tasks += fmaker.make_tasks(data)
                except NotImplementedError:
                    figlog.debug('Figure maker %s in MultiFigure can\'t make tasks', i)

        return tasks


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
            assert len(figmaker.plot_config['callstack_params']) >= 7
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


## {{{                           --     Group     --


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
            return flatten(
                [resolved(src).get_data() for src in tqdm(self.data_source, desc='Loading data')]
            )

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


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     Recipe     --


def to_str(data: Any) -> Any:
    if not isinstance(data, str) and data is not None:
        return str(data)
    return data


ForcedStr = Annotated[str, BeforeValidator(to_str)]
ForcedOptionalStr = Annotated[Optional[str], BeforeValidator(to_str)]


class RecipeDataSource(SpecializedDataSource):

    recipe_path: ForcedStr
    data_path: ForcedStr
    cache_dir: ForcedOptionalStr = None
    color_aliases: Optional[Dict[str, str]] = cm.config.protein_aliases
    input_order: Optional[List[int]] = None

    def get_data(self) -> List[pu.PlotData]:
        lib = ut.load_lib()
        recipe_file = Path(self.recipe_path).expanduser().resolve()
        data_file = Path(self.data_path).expanduser().resolve()
        candidate_networks = bc.recipe.network_from_recipe(
            recipe_file, lib, inverse='shortest', use_cache=self.cache_dir
        )
        assert isinstance(candidate_networks, list)
        if len(candidate_networks) == 0:
            raise ValueError(f'No networks built for recipe {self.recipe_path}')
        assert len(candidate_networks) == 1
        X, Y = bc.recipe.get_network_XY(
            candidate_networks[0], data_file, color_aliases=self.color_aliases
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)

        metadata = resolved(self.metadata)
        metadata['filename'] = data_file.name
        metadata['file_path'] = data_file.as_posix()
        metadata['file_stem'] = data_file.stem
        metadata['recipe_path'] = recipe_file.as_posix()
        metadata['recipe_stem'] = recipe_file.stem
        metadata['network_info'] = bc.network.generate_network_info(candidate_networks[0])

        pdata = pu.extract_plot_data_from_network(
            candidate_networks[0],
            X,
            Y,
            input_order=self.input_order,
            protein_aliases=self.color_aliases,
            metadata=metadata,
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
    input_names: Optional[List[str]] = None  # alias to use for input_columns
    output_name: Optional[str] = None  # alias to use for output_column

    file_order: Optional[PartialFunction | Callable] = None

    def model_post_init(self, *a):
        if self.file_order is None:
            self.file_order = partial(sorted, key=lambda x: x.stem)

    def check_file(self, data_file):
        SUPPORTED_EXTENSIONS = ['.csv']
        if not data_file.exists():
            raise ValueError(f'Data path {data_file} does not exist')
        extension = data_file.suffix
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'''Unsupported extension {extension} for {data_file}.
                    Supported extensions: {SUPPORTED_EXTENSIONS}'''
            )

    def load_file(self, data_file) -> List[PlotData]:

        extension = data_file.suffix

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
                assert len(self.input_names) == len(self.input_columns)
                input_names = self.input_names
            if self.output_name is not None:
                assert isinstance(self.output_column, str)
                output_name = self.output_name

            x = df[self.input_columns].to_numpy()
            y = df[self.output_column].to_numpy()

            metadata = resolved(self.metadata)
            metadata['filename'] = data_file.name
            metadata['file_path'] = data_file.as_posix()
            metadata['file_stem'] = data_file.stem

            return [
                pu.PlotData(
                    x=x,
                    y=y,
                    input_names=input_names,
                    output_name=output_name,
                    metadata=metadata,
                )
            ]

        else:
            raise NotImplementedError(f'Extension {extension} not implemented')

    def get_data(self) -> List[PlotData]:
        if isinstance(self.data_path, str):
            # data_path can contain wildcards in the filename so we glob all into a list
            datapath = Path(self.data_path).expanduser().resolve().absolute().as_posix()
            all_data_files = [Path(f) for f in glob.glob(datapath)]
        else:
            all_data_files = [self.data_path]

        all_data_files = sorted(all_data_files, key=lambda x: x.stem)
        print(f'Found {len(all_data_files)} data files for {self.data_path}')
        all_data = []
        for data_file in all_data_files:
            self.check_file(data_file)
            all_data += self.load_file(data_file)
        print(f'Loaded {len(all_data)} data files for {self.data_path}')
        return all_data

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
        data_source = [*ds_cfg]
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

import ray
from ray.experimental import tqdm_ray


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
