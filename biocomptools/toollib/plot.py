## {{{                          --     imports     --
from dataclasses import dataclass, field
from biocomptools.toollib import common as cm
from functools import partial
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
from typing import Optional, Union, Tuple, List, Dict, Sequence, Any, Callable, TypeVar
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
log.setLevel(logging.INFO)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{              --     structured config declarations     --


# a Job declares a list of DataSources (can be nested through DataSourceGroup)
# each DataSource declares a FigureMaker, which will produce PlotTasks from the data
# A PlotTask contains some PlotData, an Axes, a PlotConfig to be used by the plotting function,
# and an entry_point, aka which plotting function to use (by default, it's auto_plot, or maybe
# I should switch to directly defining something like "smooth" or "histogram" or "scatter" etc.)


# Everything relies on successive task overrides. A Task being a self-contained unit of work,
# that has everything it needs to be executed.

T = TypeVar('T')
U = TypeVar('U')
ListOrSingle = Union[List[T], T]
Pair = Tuple[T, T]
PlotData = pu.PlotData
DictOrList = Union[Dict[U,T], List[T]]


# ╭──────────────────────╮ 
# │     Base Classes     │ 
# ╰──────────────────────╯ 

#               metadata  plot_config  figure_spec figure_maker  context
# PlotJob          +           +            +           +          +
# DataSource       +           +            +           +          +
# FigureMaker      +           +            +           -          +
# FigureTask       +           +            +           -          +
# PlotTask         +           +            -           -          +
# PlotData         -           -            -           -          -


@dataclass(kw_only=True)
class BaseConfig:
    metadata: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    locals: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class FigureLayout:
    pass

@dataclass(kw_only=True)
class SimpleFigureLayout(FigureLayout):
    cols: int = 1
    rows: int = 1
    axes_size: Pair[int] = (5, 5)
    dpi: int = 300
    def make_figure(self) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(
            self.rows,
            self.cols,
            figsize=(self.cols * self.axes_size[0], self.rows * self.axes_size[1]),
            dpi=self.dpi,
        )
        return fig, ax

@dataclass(kw_only=True)
class FigureSpec:
    title: Optional[str] = None
    output_dir: str = './'
    output_file: str = 'unnamed_figure_${metadata.task_index}.png'
    extra_info: Optional[Dict[str, Any]] = None
    layout: FigureLayout = field(default_factory=SimpleFigureLayout)


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


@dataclass
class PlotTask(BaseConfig):
    """A PlotTask is a self-contained unit of work that can be executed by a plotting function"""
    data: ListOrSingle[PlotData] = MISSING # the data to be plotted
    ax: ListOrSingle[Axes] = MISSING # the axes to plot on
    plot_config: PlotConfig = MISSING


@dataclass(kw_only=True)
class FigureMaker(BaseConfig):
    figure_spec: Optional[FigureSpec] = None
    plot_config: Optional[PlotConfig] = None

    def make_plot_tasks(self, data) -> Tuple[List[FigureSpec], List[PlotTask]]:
        raise NotImplementedError('Subclasses must implement make_plot_tasks')



@dataclass(kw_only=True)
class FigureTask(BaseConfig):
    figure_spec: FigureSpec
    plot_tasks: List[PlotTask]
    plot_config: PlotConfig = MISSING


@dataclass(kw_only=True)
class DataSource(BaseConfig):

    name: Optional[str] = None
    figure_spec: Optional[FigureSpec] = None
    figure_maker: Optional[FigureMaker] = None
    plot_config: Optional[PlotConfig] = None


@dataclass(kw_only=True)
class DataSourceGroup(DataSource):
    data_sources: DictOrList[str, DataSource] = MISSING


@dataclass
class PlotJob(BaseConfig):
    defaults: List[Any] = field(
        default_factory=lambda: [
            {'data_config/rescaler@rescaler': 'EBFP2_compressed'},
            {'plot_config': 'default_plotconf'},
        ]
    )

    figure_spec: Optional[FigureSpec] = None
    figure_maker: Optional[FigureMaker] = None
    plot_config: Optional[PlotConfig] = None
    data_sources: Any = MISSING



def make_flat_list(l):
    if not isinstance(l, (list, tuple)):
        return [l]
    return [item for sublist in l for item in sublist]


@dataclass(kw_only=True)
class RecipeDataSource(DataSource):
    data_path: str
    recipe_path: str
    source_type: str = 'recipe'
    cache_dir = cm.config.paths.cache.networks
    color_aliases = cm.config.protein_aliases

    # def resolve(self) -> List[pu.PlotData]:
        # lib = ut.load_lib()
        # recipe_file = Path(self.recipe_path).expanduser().resolve()
        # data_file = Path(self.data_path).expanduser().resolve()
        # candidate_networks = bc.recipe.network_from_recipe(
            # recipe_file, lib, inverse='shortest', use_cache=self.cache_dir
        # )
        # if len(candidate_networks) == 0:
            # raise ValueError(f'No networks built for recipe {self.recipe_path}')
        # assert len(candidate_networks) == 1
        # X, Y = bc.recipe.get_network_XY(
            # candidate_networks[0], data_file, color_aliases=self.color_aliases
        # )
        # # rescaler = hydra.utils.instantiate(self.rescaler)
        # pdata = pu.extract_plot_data_from_network(
            # candidate_networks[0], X, Y, rescaler=rescaler, protein_aliases=self.color_aliases
        # )
        # return [pdata]


@dataclass(kw_only=True)
class RawDataSource(DataSource):
    data_path: str
    output_column: List[str]
    input_columns: List[str]
    input_names: Optional[List[str]] = None
    output_name: Optional[str] = None

    def resolve(self) -> List[pu.PlotData]:
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


@dataclass(kw_only=True)
class XPDataSource(DataSource):
    xp_path: str
    recipe_names: Optional[List[str]] = None
    source_type: str = 'xp'




##────────────────────────────────────────────────────────────────────────────}}}

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
