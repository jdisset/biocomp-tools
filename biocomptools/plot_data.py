## {{{                        --     imports     --

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
from dataclasses import fields
from typing import Type
import importlib
from omegaconf import OmegaConf
from toollib.plot import DataSource, DataSourceGroup, Resolvable
from omegaconf import DictConfig, ListConfig

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
## {{{                         --     load cfg     --
log = logging.getLogger('biocomptools.biocomplot')
log.setLevel(logging.DEBUG)
from biocomptools.toollib.plot import PlotTask, PlotConfig, FigureMaker, DataSource


reset_hydra()

# cs = ConfigStore.instance()
# cs.store(group="figure", name="default_figure", node=pu.FigureSpec)
# cs.store(name="base_plotjob", node=pl.PlotJob)
# base_cfg = compose(config_name="base_plotjob")

plot_job_file = '~/Code/Weiss/playground/local_job.yaml'
plot_file_path = Path(plot_job_file).expanduser().resolve().absolute()
if not plot_file_path.exists():
    raise ValueError(f'Plot job file {plot_file_path} does not exist')
file_dir = Path(plot_file_path).parent.resolve().absolute().as_posix()
file_ext = plot_file_path.suffix

reset_hydra(config_dir=file_dir)
# job_cfg = compose(config_name=plot_file_path.stem, return_hydra_config=True)
job_cfg = compose(config_name=plot_file_path.stem)
job_cfg.extra.base_figure_maker
job = pl.PlotJob.from_config(job_cfg)

##────────────────────────────────────────────────────────────────────────────}}}##

from time import time

time0 = time()
tasks = job.generate_figure_tasks()
time1 = time()

for task in tasks:
    print()
    print(f'{task=}')
    print()

log.info(f'Generated {len(tasks)} plot tasks in {time1 - time0:.2f} seconds')

# TODO: make Resolvable and inheritable work for dictlike objects, not just omegaconf

# FIX: it seems spawn_figure_task, when spawning its task, is using an already resolved plot_config, which is why l:479 fails
# is it normal? Maybe. maybe it should let that pass 
# it seems spawn is tring go resolve an already resolved PlotConfig obj

# the constructor of the figure_task has some inheritable attrs, that are calling resolvable_init

# - FigureMaker spawns figure a FigureTask by first make_resolvable(FigureTask) [plot.py:1001]
# - spawn merges thi
# -


# I think since PlotConfig is a pydantic BaseModel, it gets perfectly initialized - as a PlotConfig - first from the dictlike thing that is passed to the parent figuretask. That means my wrapped constructor, which would usually expect PlotConfig to be a DictConfig and then would wrap it into a Resolvable, is trying to wrap an actual PlotConfig into the resolvable, which fails.
# I think if I declare it as being a ResolvableOr, the problem is that pydantic choses to construct a PlotConfig instead of either - doing nothing, or making it into a Resolvable (which should be possible since I declare the type as being ResolvableOr[PlotConfig]. 
# Forcing it to be a Resolvable in all case is annoying becase then we lose the "normal" way of initializing things (which would be passing an actual obj) 
# I should change the init wrapper approach. Using pydantic, there's prob a way to have Resolvable be the priority, and then I can ask make_resolvable, or directly the wrapper to tolerate fully constructed target_type. Actually no if I can ask pydantic to handle the priority construction of a resolvable object I don;t even need the decorator anymore, I can just use ResolvableOr, and let pydantic decide to make either a resolvable or the type, and that's it!
# so, pydantic should: try: make_resolvable. if it fails, it should build the correct obj.
# I guess I still need the decorator. I just need to execute it before pydantic does its shit



## {{{                          --     archive     --

# tasks = make_plot_tasks(job_cfg)

# log.info(f'Generated {len(tasks)} plot tasks')
# task = tasks[0]

# do rescaling:

OmegaConf.clear_resolver('np')
OmegaConf.clear_resolver('numpy')


def numpy_resolver(key, fname, *args, **kwargs):
    import numpy

    f = getattr(numpy, fname)
    return f(*args, **kwargs)


OmegaConf.register_new_resolver('np', numpy_resolver)
OmegaConf.register_new_resolver('numpy', numpy_resolver)


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

## {{{                           --     tests     --

from toollib.plot import resolvable_attrs, inherit_attrs

# test cases


@resolvable_attrs(('attr_0', dict))
class B:
    attr_0 = MISSING


@resolvable_attrs((attr_0, attr_1))
@dataclass
class A:
    attr_0: int
    attr_1: str


##

ds = pl.DataSource()
dir(pl.DataSourceGroup())
dir(pl.DataSourceGroup)

pl.DataSourceGroup.__init__()

##

def printer(thing_to_say):
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            print(thing_to_say)

        cls.__init__ = new_init
        return cls

    return decorator


@printer('hello from decorator')
class Base:
    def __init__(self, **kwargs):
        print('Base init')


b = Base()


@dataclass
class Derived(Base):
    pass


d = Derived()


##────────────────────────────────────────────────────────────────────────────}}}

from pydantic import BaseModel, ValidationError, validator, Field, field_validator

class A(BaseModel):
    a: int

class B(BaseModel):
    b: int
    a: pl.ResolvableOr[A]

    class Config:
        arbitrary_types_allowed = True


ra = pl.make_resolvable(A, {'a': 3})
# ra.resolve()

rb = pl.make_resolvable(B, {'b': 1, 'a': {'a': 42}})
rb.resolve()

rb = pl.make_resolvable(B, {'b': 1, 'a': ra})
rb.resolve()


##

from typing import Any, List, Callable

from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)
from pydantic.functional_validators import WrapValidator
from pydantic.functional_validators import AfterValidator, BeforeValidator

def make_validator(label: str) -> Callable[[str, ValidationInfo], str]:
    def validator(v: Any, info: ValidationInfo) -> Any:
        print('VALIDATOOOOOOOR')
        print(f'{info=}')
        return v

    return validator


class A(BaseModel):
    x: Annotated[ str, BeforeValidator(make_validator('before-1'))]

A(x='hi')




