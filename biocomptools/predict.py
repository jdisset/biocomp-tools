### {{{                          --     imports     --
import ray

from typing import List, Tuple

import shutil
from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
import biocomp.train as train
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np

# pretty print from rich
import json
from pathlib import Path

from biocomptools.toollib import common as cm

import logging

from omegaconf import OmegaConf

logger = logging.getLogger('biocomp_tools_train')
logger.setLevel(logging.INFO)
config = cm.load_config()


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     command line arguments     --

# TODO: switch both this and plot_data to a better config system (cf jeanplot?)
# or maybe hydra?
# OMEGACONF IS THE WAY TO GO!!!

prog = cm.CLIProgram()

prog.add_argument(
    '--model_path',
    help="""name of the model to load from the database (db://<model_name>)
    or path to a directory that contains a compute_config.json and [best_]model.pkl""",
)

prog.add_argument(
    '--output_dir',
    help='Directory to save the results and the model. Can use any of the following placeholders: <model_name>, <timestamp>',
    default=(f'{cte.BIOCOMP_ROOT}/predictions/<model_name>'),
)

prog.add_argument(
    '--network_collection',
    help='List of collection_names to predict on (from db)',
    nargs='+',
    default=[],
)

# test args, we use training_set and prediction_set as single_uorfs and case_matrix_4_corners:
prog.parse_args([])


##────────────────────────────────────────────────────────────────────────────}}}

### {{{       --     load plot config from ./configs/default.yaml     --
this_file_path = Path(__file__).resolve()

plot_config = OmegaConf.load(f'{this_file_path.parent}/configs/default.yaml')

##────────────────────────────────────────────────────────────────────────────}}}


