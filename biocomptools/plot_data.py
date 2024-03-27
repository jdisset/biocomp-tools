## {{{                          --     imports     --
from dataclasses import dataclass, field
from biocomptools.toollib import common as cm
from functools import partial
from dataclasses import dataclass, asdict
from copy import deepcopy
import logging
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

from biocomptools.toollib import plot as pl
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


##────────────────────────────────────────────────────────────────────────────}}}

from biocomptools.toollib.plot import PlotTask, PlotConfig, FigureMaker, DataSource


reset_hydra()


cs = ConfigStore.instance()
cs.store(group="figure", name="default_figure", node=pu.FigureConfig)
cs.store(name="base_plotjob", node=pl.PlotJob)
base_cfg = compose(config_name="base_plotjob")

plot_job_file = '~/Code/Weiss/playground/local_job.yaml'
plot_file_path = Path(plot_job_file).expanduser().resolve().absolute()
if not plot_file_path.exists():
    raise ValueError(f'Plot job file {plot_file_path} does not exist')
file_dir = Path(plot_file_path).parent.resolve().absolute().as_posix()
file_ext = plot_file_path.suffix

reset_hydra(config_dir=file_dir)
job_cfg = compose(config_name=plot_file_path.stem)






## {{{                          --     archive     --

# tasks = make_plot_tasks(job_cfg)

# log.info(f'Generated {len(tasks)} plot tasks')
# task = tasks[0]

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

##

yaml_test = """
A:
    - 1
    - 2

nested:
    B:
        - 3
        - ${A.0}
        - "${eval: ${A.0} + 10}"
        # using local variable (B[1]) but with relative path:
        - ${..B.1}
"""

oc = OmegaConf.create(yaml_test)

oc.nested.B[3]


##────────────────────────────────────────────────────────────────────────────}}}
