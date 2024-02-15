### {{{                          --     imports     --
import sys
import ray

from pathos.multiprocessing import ProcessPool
import urllib.parse
import openpyxl
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp.plotutils as pu
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json5

# pretty print from rich
from rich import print as rprint
import argparse
import json
from pathlib import Path
from rich import print as rprint

import common as cm
import constants as cte

import logging
from jax.tree_util import tree_flatten, tree_unflatten
from biocomp.parameters import ParameterTree
from biocomp.compute import ComputeStack
from biocomp.train import get_optimizer, huber_quantile_loss, value_and_grad, console_log
import optax

logger = logging.getLogger('biocomp_tools_train')
logger.setLevel(logging.INFO)

##────────────────────────────────────────────────────────────────────────────}}}


"""
    Train a model, save it, log everything, and run predictions on the test set.
    Meant to be used in a docker container.
"""


# Once the training is done we need to collect the results from W&B, pick the best model,
# download the artifact and put them on dropbox and log everything to the database.
# Dropbox is used to store the model, the data plots and the prediction plots. The webserver simply
# has a read-only sync with the dropbox folder.

# Finally, we run the predictions on the validation set, store the plots in dropbox
# and log everything to the database, i.e. for each prediction, we store an entry with the error,
# the prediction plot's path, and the model's id (technically, the training_run entry's id).

# Pick a list of collection_ids to identify the networks to train on (same for the prediction set)

### {{{                  --     command line arguments     --

prog = cm.CLIProgram()

prog.add_argument(
    '--compute_config_path',
    help='Path of the compute config to use, use "db:<name>" to load from db',
    default='db:default_compute_v0',
)

prog.add_argument(
    '--data_config_path',
    help='Path of the data config to use, use "db:<name>" to load from db',
    default='db:default_data_v0',
)

prog.add_argument(
    '--training_config_path',
    help='Path of the training config to use, use "db:<name>" to load from db',
    default='db:default_training_v0',
)

prog.add_argument(
    'trainprog_args',
    help='Arguments to pass to the training program',
    nargs='*',
)

# required
prog.add_argument(
    '--training_set', help='List of collection_names to use for training', nargs='+', default=[]
)
prog.add_argument(
    '--prediction_set', help='List of collection_names to use for prediction', nargs='+', default=[]
)

# prog.parse_args()
# test args, we use training_set and prediction_set as single_uorfs and case_matrix_4_corners:
prog.parse_args(
    [
        '--training_set',
        'single_uorfs',
        'case_matrix_4_corners',
        '--prediction_set',
        'single_uorfs',
        'case_matrix_4_corners',
        '--',
        '--seed',
        '42',
    ]
)

prog.args.training_set


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     args validation     --

assert prog.args.training_set, 'Training set must be non-empty'
if prog.args.prediction_set:
    logger.warning('Prediction set is empty')

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     load all configurations     --


def load_config(path):
    if path.startswith('db:'):
        conf = cm.get_row_by_id('configurations', 'name', path[3:])
        print(conf)
        assert conf, f'Configuration with name {path[3:]} not found'
        return conf['config']  # directly loaded as a dictionary
    else:
        return json5.load(open(path))


compute_config = load_config(prog.args.compute_config_path)
data_config = load_config(prog.args.data_config_path)
training_config = load_config(prog.args.training_config_path)

compute_config = bc.compute.ComputeConfigManager.from_dict(compute_config)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     Load networks from db     --

lib = ut.load_lib()
training_networks = cm.get_networks_in_collections(prog.args.training_set)
networks, raw_Xs, raw_Ys = cm.load_networks_and_data(training_networks, lib)

# [x] construct data manager

BIOCOMP_DATA_CACHE_DIR = Path(cte.BIOCOMP_CACHE_DIR) / 'data'

dman = du.DataManager(
    raw_Xs,
    raw_Ys,
    networks,
    data_cfg=data_config,
    cache_location=BIOCOMP_DATA_CACHE_DIR,
)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{           --     Pass everything to a training program     --
import wandb as wb
from functools import partial

trainprog = train.TrainingProgram()
trainprog.parse_args(prog.args.trainprog_args)

# overwrite configurations
trainprog.training_config = training_config
trainprog.data_config = data_config
trainprog.compute_config = compute_config

trainprog.update_config_from_args()

assert isinstance(trainprog.compute_config, bc.compute.ComputeConfigManager)
assert isinstance(trainprog.data_config, dict)
assert isinstance(trainprog.training_config, dict)

full_config = {
    **trainprog.training_config,
    **trainprog.data_config,
    **trainprog.compute_config.config,
}


# [x] start wandb logging
# [x] fetch wb name and id

WANDB_ENTITY = cte.get_env_or_local('WANDB_ENTITY', 'jdisset')
assert wb.api.api_key, 'W&B API key not found'
wb_run = wb.init(config=full_config, project=trainprog.wandb_project, entity=WANDB_ENTITY)
assert wb_run, 'Failed to start W&B run'

loggers = [
    (1, bc.train.console_log),
    (1, bc.train.wandb_log_epoch),
    (trainprog.wandb_plot_period, partial(trainprog.wandb_plot_pred, dman=dman)),
]

# TODO:
# [x] --n_replicates:
#   [x] vmap over replicates
#   [x] write "get_best_loss_id" function that returns the lowest *smoothed* loss
#   [x] wandb_plot_pred should log only the best replicate
#   [x] wandb_log_epoch should log each replicate's loss separately
# [x] get best model from W&B

# params, loss_history = bc.train.start(
# dman, trainprog.training_config, trainprog.compute_config, loggers, seed=trainprog.seed
# )


def best_loss_id(loss_history):
    pass


# [x] use the wb name and id to log everything to the database
# [ ] log networks being used into a network_training_run table (more reliable than the collection name)
# [ ] but also log the arguments used to load the networks, including the collection names as its own column


##────────────────────────────────────────────────────────────────────────────}}}

from scipy.ndimage import gaussian_filter1d
assert isinstance(training_config, dict)

N_EPOCHS = 2
N_REPLICATES = 20

training_config['epochs'] = N_EPOCHS
training_config['n_replicates'] = N_REPLICATES

params, loss_history = train.start(
    dman,
    training_config,
    compute_config,
    # loggers,
)

##

def get_best_smoothed_loss_id(loss_history, sigma=10):
    all_losses = np.hstack(loss_history)
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma)
    best_loss_id = np.argmin(smoothed_losses)
    return best_loss_id, smoothed_losses

best_loss_id, smoothed_losses = get_best_smoothed_loss_id(loss_history)

all_losses = np.hstack(loss_history)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(all_losses.T, alpha=0.5, color='gray', linewidth=0.5)
ax.plot(smoothed_losses.T)

##
best_params = ut.tree_get(params, best_loss_id)





