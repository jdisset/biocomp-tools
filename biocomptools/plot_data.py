# {{{                        --     imports     -

from time import time
import matplotlib.pyplot as plt
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
from biocomptools.toollib.configutils import load_hydra_config_file
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

log = logging.getLogger('biocomptools.biocomplot')
log.setLevel(logging.INFO)

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                         --     load cfg     --

# JOB_FILE = '~/Code/Weiss/playground/plot_jobs/uorf_matrices.yaml'
JOB_FILE = '~/Code/Weiss/playground/local_job.yaml'
JOB_FILE = '~/Code/Weiss/playground/local_job.yaml'
JOB_FILE = '~/Code/Weiss/playground/georg_job.yaml'

job = pl.PlotJob.model_validate(load_hydra_config_file(JOB_FILE))

# ray log to warn:
raylog = logging.getLogger('ray')
raylog.setLevel(logging.WARN)
jaxlog = logging.getLogger('jax')
jaxlog.setLevel(logging.WARN)

time0 = time()
job.run_tasks_sequential()
time1 = time()
print(f'Time to run tasks: {time1 - time0:.2f} seconds')


# # find all dirs with *3Dframes at the end:
# frame_dirs = list(Path('./output_plot').glob('*3Dframes'))
# for frame_dir in frame_dirs:
# print(f'Processing {frame_dir}')
# pl.make_video(input_file_pattern=f'{frame_dir}/frame_%d.png',
# output_file=f'{frame_dir}.mp4',
# fps=15)

##────────────────────────────────────────────────────────────────────────────}}}##
georg_xpdir = Path("~/Dropbox (MIT)/Biocomp/Experiments/miR_bandpass").expanduser()
df = pd.read_csv(georg_xpdir / "csv_jean.csv", index_col=0)
##
# mkdir for calib raw:
calib_dir = georg_xpdir / "calibration/controls/gated"
calib_dir.mkdir(exist_ok=True, parents=True)

tube_dir = georg_xpdir / "tubes"
tube_dir.mkdir(exist_ok=True, parents=True)

# list all tube names:
tube_names = df['TUBE_NAME'].unique()

cdf = df[(df['Cells']) & (~df['singlets'])]

color_controls = {
    'mNeongreen': cdf[cdf['TUBE_NAME'] == 'A1'],
    'eBFP2': cdf[cdf['TUBE_NAME'] == 'A2'],
    'mKO2': cdf[cdf['TUBE_NAME'] == 'A3'],
    'mMaroon1': cdf[cdf['TUBE_NAME'] == 'A4'],
    'all': cdf[cdf['TUBE_NAME'] == 'A5'],
    'empty': cdf[cdf['TUBE_NAME'] == 'A6'],
}

tubes = {tbname: df[df['TUBE_NAME'] == tbname] for tbname in tube_names if not tbname.startswith('A')}

exclude_columns = ['TUBE_NAME', 'Cells', 'singlets']
columns = [col for col in df.columns if col not in exclude_columns]

# save
for cname, c in color_controls.items():
    c[columns].to_csv(calib_dir / f'{cname}.csv', index=False)

for tbname, t in tubes.items():
    t[columns].to_csv(tube_dir / f'{tbname}.csv', index=False)

print('Done')
