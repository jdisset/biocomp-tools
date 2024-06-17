### {{{                          --     imports     --
import sys
import ray

from typing import List, Tuple
import argparse
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
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
import json5

# pretty print from rich
import json
from pathlib import Path

from toollib import common as cm

import logging

logger = logging.getLogger('biocomp_tools_train')
logger.setLevel(logging.INFO)

##────────────────────────────────────────────────────────────────────────────}}}

"""""
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


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     args validation     --

logger = logging.getLogger('biocomp_tools_train')

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
compute_config = bc.compute.ComputeConfigManager.from_dict(compute_config)

data_config = load_config(prog.args.data_config_path)
training_config = load_config(prog.args.training_config_path)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                   --     Load networks from db     --

lib = ut.load_lib()
training_networks = cm.get_networks_in_collections(prog.args.training_set)
networks, raw_Xs, raw_Ys = cm.load_networks_and_data(training_networks, lib)

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
prog.args.trainprog_args
trainprog.parse_args(prog.args.trainprog_args)

assert isinstance(training_config, dict)
training_network_name_list = training_networks['name'].to_list()
training_config['training_set_networks'] = training_network_name_list
training_config['training_set_collections'] = prog.args.training_set

# overwrite configurations
trainprog.training_config = training_config
trainprog.data_config = data_config
trainprog.compute_config = compute_config

trainprog.update_config_from_args()

assert isinstance(trainprog.compute_config, bc.compute.ComputeConfigManager)
assert isinstance(trainprog.data_config, dict)

full_config = {
    **trainprog.training_config,
    **trainprog.data_config,
    **trainprog.compute_config.config,
}


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     start training     --

ndArray = bc.trainutils.ndArray
from scipy.ndimage import gaussian_filter1d

WANDB_ENTITY = cte.get_env_or_local('WANDB_ENTITY', 'jdisset')
WANDB_PROJECT = cte.get_env_or_local('WANDB_project', 'biocomp_paper')

assert wb.api.api_key, 'W&B API key not found'
wb_run = wb.init(config=full_config, project=trainprog.wandb_project, entity=WANDB_ENTITY)
assert wb_run, 'Failed to start W&B run'

today = time.strftime('%Y-%m-%d', time.localtime())

training_run_name = f'{today}_{wb_run.project}_{wb_run.id}_{wb_run.name}'
save_dir = Path(wb_run.dir) / training_run_name
save_dir.mkdir(exist_ok=True, parents=True)


loggers = [
    (1, bc.trainutils.console_log),
    (
        trainprog.wandb_save_period,
        partial(
            bc.trainutils.local_save,
            compute_config=trainprog.compute_config,
            training_config=trainprog.training_config,
            data_config=trainprog.data_config,
            save_dir=save_dir,
        ),
    ),
    (1, bc.trainutils.wandb_log_epoch),
]


params, loss_history, epoch_history = bc.train.start(
    dman,
    training_config,
    compute_config,
    loggers,
)


##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                      --     get best model     --

# use normal matplotlib backend (not agg) to plot the losses
# plt.switch_backend('module://matplotlib_inline.backend_inline')

from labellines import labelLines, labelLine
def get_best_smoothed_loss_id(
    loss_history: List[ndArray], sigma: float = 10.0
) -> Tuple[int, np.ndarray]:
    all_losses = np.hstack(loss_history)
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma)
    best_loss_id = int(np.argmin(smoothed_losses[:,-1]))
    return best_loss_id, smoothed_losses


best_loss_id, smoothed_losses = get_best_smoothed_loss_id(loss_history)
all_losses = np.hstack(loss_history)
# fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig, ax = pu.mkfig(size=(8, 5), dpi=300)
ax.plot(all_losses.T, alpha=0.5, color='gray', linewidth=0.5)
# ax.plot(smoothed_losses.T)
for i in range(smoothed_losses.shape[0]):
    ax.plot(smoothed_losses[i], label=f'Replicate {i}')
labelLines(ax.get_lines(), zorder=2.5)
ax.set_yscale('log')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss')
# write id of all losses
# write id of smoothed losses

# save the plot in the save_dir
plt.savefig(save_dir / 'losses.png')
plt.close(fig)
wb_run.log({'All losses': wb.Image((save_dir / 'losses.png').as_posix())})
best_params = ut.tree_get(params, best_loss_id)


# for now, let's just not plot the predictions, but focus on logging the losses, finding the best model and 
# dumping everything into a folder with the following structure:
# training_id/
# - metadata.json
# - data_config.json
# - training_config.json
# - compute_config.json
# - best_model.pkl
# - loss_history.npy
# - predictions/
#   - <recipe_hash>/
#     - fullgrid_<pred_hash>.png # if we predict on a grid
#     - original_<pred_hash>.png # if we predict on the original data
#     - maskedgrid_<pred_hash>.png
#   ...

# we will make sure wandb uploads the whole folder, and we will also upload it to dropbox

# save everything

training_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wb_run.start_time))
metadata = {
    'training_id': training_run_name,
    'wb_run_id': wb_run.id,
    'wb_run_name': wb_run.name,
    'wb_project': wb_run.project,
    'wb_entity': wb_run.entity,
    'training_start_date': training_start_time,
    'training_end_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    'best_loss_id': best_loss_id,
}

json.dump(metadata, open(save_dir / 'metadata.json', 'w'), indent=2)
ut.save(best_params, save_dir / 'best_model.pkl')
np.save(save_dir / 'loss_history.npy', all_losses)

##────────────────────────────────────────────────────────────────────────────}}}##

### {{{                      --     copy to export dir --

export_dir = Path(prog.args.export_dir)
export_dir.mkdir(exist_ok=True, parents=True)
shutil.copytree(save_dir, export_dir / training_run_name, dirs_exist_ok=True)

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                      --     log to database     --

biocomp_version = ut.get_biocomp_version()
biocomp_git_hash = ut.get_git_commit_hash()


model_path = (export_dir / training_run_name / 'best_model.pkl')
# remove BIOCOMP_ROOT from the path
if model_path.is_absolute():
    # check if BIOCOMP_ROOT is in the path
    if cte.BIOCOMP_ROOT in model_path.as_posix():
        model_path = model_path.relative_to(cte.BIOCOMP_ROOT)
    else:
        model_path = model_path.as_posix()

entry = {
    'name': training_run_name,
    'date_started': training_start_time,
    'duration': int(time.time() - wb_run.start_time),
    'wb_run_id': wb_run.id,
    'wb_run_name': wb_run.name,
    'wb_project': wb_run.project,
    'best_replicate': best_loss_id,
    'export_dir': (export_dir / training_run_name).as_posix(),
    'training_config': json.dumps(trainprog.training_config, indent=2),
    'data_config': json.dumps(trainprog.data_config, indent=2),
    'base_compute_config_name': prog.args.compute_config_path,
    'compute_config': json.dumps(trainprog.compute_config.config, indent=2),
    'biocomp_version': biocomp_version,
    'biocomp_git_hash': biocomp_git_hash,
    'end_loss': all_losses[best_loss_id, -1],
    'model_path': (export_dir / training_run_name / 'best_model.pkl').as_posix(),
}

# log everything to the database
cm.insert_row('training_run', entry)
# cm.update_row('training_run', entry, key_column='name')

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                        --     predictions     --
# TODO:
# [x] W&B loss logger that works with replicates
# [-] get best model id, plot predictions on training set to W&B
# [x] use the wb name and id to log everything to the database, including the best model id
#   [-] log networks being used into a network_training_run table (more reliable than the collection name)
#   [-] but also log the arguments used to load the networks, including the collection names as its own column
# [ ] try this file as a standalone script
# [ ] write down what the docker image should contain and do:
#   [ ] copy of biocomp core
#   [ ] copy of biocomp_tools
# [ ] write terraform that attaches the EBS volume to the instance and provisions the instance
# [ ] ansible that sends the right config files, and runs the docker image
# [ ] test ansible on a local docker image (docker in docker) or on a VM
# [ ] visualize training results
# optional:
# [ ] W&B pred plot logger that works with replicates

wb_run.finish()

##────────────────────────────────────────────────────────────────────────────}}}


import biocomp as bc
from biocomp import compute as cmp
from biocomp import utils as ut
from biocomp import datautils as du

# after training, we call the predict tool with the best model to generate the predictions in a local folder
# then we upload the predictions to wandb AND to dropbox 
# (if we ever run out of space on wandb, we can just link to the served dropbox folder)
# one important aspect of predictions is the mode: do we want to predict at the same location as the training set, or on a grid + mask?
# we log everything to the database

# print(OmegaConf.to_yaml(du.DEFAULT_DATA_CONFIG))
# print(OmegaConf.to_yaml(cmp.DEFAULT_COMPUTE_CONFIG))

##

def example_function(posarg_1, posarg_2, kwarg_1=1, kwarg_2=2):
    print(f'{posarg_1=}, {posarg_2=}, {kwarg_1=}, {kwarg_2=}')

partial1 = ut.partial(example_function, posarg_2='p2', kwarg_2=4)

# ut.serialize_function(example_function)
# ut.deserialize_function(ut.serialize_function(partial1))(2)

ut.encode_function(partial1)
ut.decode_function(ut.encode_function(partial1))(2)



