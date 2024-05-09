# {{{                        --     imports     -

from functools import partial
from biocomptools.toollib.resolvable import make_resolvable, resolved
from biocomp.utils import PartialFunction
from copy import deepcopy
import logging
import pandas as pd
from pathlib import Path

import hydra
from hydra import compose, initialize, initialize_config_dir
from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.global_hydra import GlobalHydra

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic.functional_validators import AfterValidator, BeforeValidator
from biocomptools.toollib.resolvable import open_dictlike, short_conf
from biocomptools.toollib import common as cm
from biocomptools.toollib.resolvable import build_from_config, resolved
from biocomptools.toollib import plot as pl
from biocomptools.toollib import resolvable as br

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

job = pl.PlotJob.model_validate(job_cfg)

##────────────────────────────────────────────────────────────────────────────}}}##

from time import time

time0 = time()
ftasks = job.generate_figure_tasks()
time1 = time()

for task in ftasks:
    print(f'{task.context=}')
    print()

log.info(f'Generated {len(ftasks)} plot tasks in {time1 - time0:.2f} seconds')


## {{{                   --     directly from python     --
pcfg = pl.PlotConfig(rc_context={'hey': 120000})

pf = PartialFunction(func=lambda: print('hello world'))
pfr = make_resolvable(value=pf)
rpf = resolved(pfr) # -> ok
rpf()


pt = pl.PlotTask(plot_method=pf)
pt.plot_method
rpm = resolved(pt.plot_method)
rpm() # -> ok

ptr = make_resolvable(pl.PlotTask, value=pt) # this adds some bullshit...
rptr = resolved(ptr)
rptr.plot_method
resolved(rptr.plot_method) # -> fails


##


ftask = pl.FigureTask(plot_tasks=[pt])#, plot_config=pcfg)

pt0 = resolved(ftask.plot_tasks[0])
pt0.plot_method # -> ok

resolved(pt0.plot_config) # -> ok


rpm0 = resolved(pt0.plot_method) # fails


# fmaker = pl.SingleFigure(plot_tasks=[pl.PlotTask(plot_method=pf)])
# pt0 = resolved(fmaker.plot_tasks[0])
# resolved(pt0.plot_method) # -> not ok


##


dsource = pl.RawDataSource(
    data_path='~/Dropbox (MIT)/Biocomp/Experiments/2023-10-01_Cascades_CCv4/data/calibrated_data_v3/8xCsy4R_CasE.2023-10-01_Cascades_CCv4.csv',
    input_columns=["EBFP2", "MKO2", "MNEONGREEN"],
    output_column="1XIRFP720",
)

job = pl.PlotJob(data_source=dsource, figure_maker=fmaker, plot_config=pcfg)
ds = resolved(job.data_source)

rds = resolved(ds)
rds.plot_config # -> ok

t0 = job.generate_figure_tasks()[0]
rt0 = resolved(t0)
rt0.plot_config # -> ok

p0 = rt0.plot_tasks[0]
rp0 = resolved(p0)
rp0.plot_config # -> ok!
rp0.plot_method


ftasks = job.generate_figure_tasks()
for task in ftasks:
    task.run()



##────────────────────────────────────────────────────────────────────────────}}}
