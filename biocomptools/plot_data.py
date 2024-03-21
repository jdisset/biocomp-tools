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
from typing import Optional, Union, Tuple, List, Dict, Sequence, Any, Callable
import hydra
from hydra import compose, initialize, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.global_hydra import GlobalHydra

log = logging.getLogger('biocomptools.biocomplot')
log.setLevel(logging.INFO)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     plugin for searchpath     --


class BiocompSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="biocomptools", path="pkg://biocomptools/configs")


Plugins.instance().register(BiocompSearchPathPlugin)


def reset_hydra(config_dir=None):
    GlobalHydra.instance().clear()
    if config_dir is not None:
        config_dir_path = Path(config_dir).expanduser().resolve().absolute()
        print(f'Initializing hydra with config dir {config_dir_path}')
        # make absolute:
        assert config_dir_path.exists()
        assert config_dir_path.is_dir()
        initialize_config_dir(config_dir=config_dir, version_base="1.3")
    else:
        initialize(version_base="1.3")


"""
Utils for plotting data in various ways, from the network representation of an experiment.
It can build this network from scratch given a recipe, library and data file, or it can
use a database file to load the network and plot it.
Uses plotjob descriptions to define the plot to be made, and the data to be used.
"""

# ut.generate_full_nested_config(namespace='biocomp.plotutils')
# print(ut.dump_default_config('biocomp.plotutils'))

# rprint(OmegaConf.to_yaml(cfg))
# rprint(OmegaConf.to_yaml(cfg.data_location))

# 2024-02-18_BPv4_BPv5
"""
desired usage:
    biocomplot +job=./local_plotjob.yaml

"""

##────────────────────────────────────────────────────────────────────────────}}}
## {{{              --     structured config declarations     --


@dataclass(kw_only=True)
class DataSource:
    source_type: str
    rescaler: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    overrides: Optional[Dict] = None


@dataclass(kw_only=True)
class RecipeDataSource(DataSource):
    data_path: str
    recipe_path: str
    source_type: str = 'recipe'


@dataclass(kw_only=True)
class RawDataSource(DataSource):
    data_path: str
    output_column: List[str]
    input_columns: List[str]
    input_names: Optional[List[str]] = None
    output_name: Optional[str] = None
    source_type: str = 'raw'


@dataclass(kw_only=True)
class XPDataSource(DataSource):
    xp_path: str
    recipe_names: Optional[List[str]] = None
    source_type: str = 'xp'


@dataclass
class PlotTask:
    data: pu.PlotData
    figure: pu.FigureConfig
    plot_config: Any = MISSING
    output_path: str = MISSING


@dataclass
class PlotJob:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {'data_config/rescaler@rescaler': 'EBFP2_compressed'},
            {'plot_config': 'default_plotconf'},
        ]
    )
    figure: pu.FigureConfig = field(default_factory=pu.FigureConfig)
    output_path: str = './output_plot/$[task_index].png'
    plot_config: Any = MISSING
    data_sources: List[Any] = MISSING
    rescaler: Optional[Any] = MISSING
    plot_job_file: Optional[str] = '../../playground/local_job.yaml'


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


def resolve_recipe_data_source(
    data_source: RecipeDataSource,
    cache_dir=cm.config.paths.cache.networks,
    color_aliases=cm.config.protein_aliases,
) -> List[pu.PlotData]:

    recipe_file = Path(data_source.recipe_path).expanduser().resolve()
    data_file = Path(data_source.data_path).expanduser().resolve()
    candidate_networks = bc.recipe.network_from_recipe(
        recipe_file, lib, inverse='shortest', use_cache=cache_dir
    )
    if len(candidate_networks) == 0:
        raise ValueError(f'No networks built for recipe {data_source.recipe_path}')
    assert len(candidate_networks) == 1
    X, Y = bc.recipe.get_network_XY(candidate_networks[0], data_file, color_aliases=color_aliases)
    rescaler = hydra.utils.instantiate(data_source.rescaler)
    pdata = pu.extract_plot_data_from_network(
        candidate_networks[0], X, Y, rescaler=rescaler, protein_aliases=color_aliases
    )
    return [pdata]


def resolve_raw_data_source(data_source: RawDataSource) -> List[pu.PlotData]:

    SUPPORTED_EXTENSIONS = ['.csv']

    data_file = Path(data_source.data_path).expanduser().resolve()

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
            assert data_source.input_columns is not None
            for col in data_source.input_columns:
                if col not in df.columns:
                    raise ValueError(
                        f'Column {col} not found in {data_file}. Available: {df.columns}'
                    )
            assert data_source.output_column is not None
            if data_source.output_column not in df.columns:
                raise ValueError(
                    f'''Column {data_source.output_column} not found in {data_file}.
                Available: {df.columns}'''
                )

            input_names = data_source.input_columns
            output_name = data_source.output_column

            if data_source.input_names is not None:
                assert len(input_names) == len(data_source.input_columns)
                input_names = data_source.input_names
            if data_source.output_name is not None:
                assert isinstance(data_source.output_column, str)
                output_name = data_source.output_name

            x = df[data_source.input_columns].to_numpy()
            y = df[data_source.output_column].to_numpy()
            rescaler = hydra.utils.instantiate(data_source.rescaler)

            return [
                pu.PlotData(
                    x=x,
                    y=y,
                    input_names=input_names,
                    output_name=output_name,
                    rescaler=rescaler,
                )
            ]
        else:
            raise NotImplementedError(f'Extension {extension} not implemented')


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

reset_hydra()

cs = ConfigStore.instance()
cs.store(group="figure", name="default_figure", node=pu.FigureConfig)
cs.store(name="base_plotjob", node=PlotJob)
# base_cfg = compose(config_name="base_plotjob", overrides=["data_config/rescaler@rescaler=EBFP2_compressed"])
base_cfg = compose(config_name="base_plotjob")

if base_cfg.plot_job_file is not None:
    file_path = Path(base_cfg.plot_job_file).expanduser().resolve()
    if not file_path.exists():
        raise ValueError(f'Plot job file {file_path} does not exist')
    file_dir = Path(base_cfg.plot_job_file).parent.resolve().absolute().as_posix()
    file_ext = file_path.suffix
    reset_hydra(config_dir=file_dir)
    job_cfg = compose(config_name=file_path.stem)
else:
    job_cfg = base_cfg


tasks = make_plot_tasks(job_cfg)

log.info(f'Generated {len(tasks)} plot tasks')
task = tasks[0]
task.figure
task.output_path
task.data.y
task.data.x
task.plot_config.default_values
task.data.rescaler.fwd(task.data.x).max()
task.data.rescaler.fwd(task.data.x).min()

# do rescaling:


def get_plot_config(plot_config):
    resolved_plot_config = OmegaConf.create(plot_config)
    OmegaConf.resolve(resolved_plot_config)
    conf_as_dict = OmegaConf.to_container(resolved_plot_config)
    assert isinstance(conf_as_dict, dict)
    callstack_params = ut.generate_full_nested_config(
        conf_as_dict['callstack_params'], namespace='biocomp.plotutils'
    )
    resoved_callstack = OmegaConf.create(callstack_params)
    resolved_plot_config.callstack_params = resoved_callstack
    return resolved_plot_config


# rprint(OmegaConf.to_yaml(callstack_params.auto_plot_params.smooth_params.smooth_3d_params))

import matplotlib.pyplot as plt


def plot_task(task: PlotTask):
    plot_config = get_plot_config(task.plot_config)
    task.data.y = task.data.rescaler.fwd(task.data.y)
    task.data.x = task.data.rescaler.fwd(task.data.x)

    pu.auto_plot(
        task.data,
        figure_config=task.figure,
        **plot_config.callstack_params.auto_plot_params,
        rc_context=plot_config.rc_context,
    )
    plt.show()


plot_task(tasks[0])

# sorted(plt.rcParams.keys())
