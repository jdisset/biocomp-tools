# {{{                        --     imports     -
from time import time
import matplotlib.pyplot as plt
import argparse
from functools import partial
from typing import Annotated
from biocomptools.toollib.resolvable import (
    make_resolvable,
    resolved,
    Resolvable,
    resolvable,
    ResolvableOr,
)

from biocomptools.toollib.inheritable import merged_into

# from biocomptools.toollib.configutils import load_config_file
# from biocomptools.toollib.configutils import load_config_file

from dracon import load_config_file


import biocomp.utils as ut
import biocomp.datautils as du
from biocomp.utils import PartialFunction, ArbitraryModel
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


##────────────────────────────────────────────────────────────────────────────}}}

log = logging.getLogger('biocomptools.biocomplot')
log.setLevel(logging.WARNING)
raylog = logging.getLogger('ray')
raylog.setLevel(logging.WARN)
jaxlog = logging.getLogger('jax')
jaxlog.setLevel(logging.WARN)

JOB_FILE = '~/Dropbox (MIT)/Biocomp/Plots/manual_plots/jobs/2A_case.yaml'


job_file = Path(JOB_FILE).expanduser().resolve()
job = pl.PlotJob.model_validate(load_config_file(job_file))

time0 = time()
job.run_tasks_sequential()
time1 = time()
print(f'All plot tasks done in {time1 - time0:.2f} seconds')


