### {{{                          --     imports     --
import sys
import ray

from typing import List, Tuple
import argparse
import json
from pathlib import Path

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

### {{{                  --     training program helper     --

class UpdateConfigAction(argparse.Action):
    def __init__(self, config_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_name = config_name

    def __call__(self, parser, namespace, values, option_string=None):
        updates = getattr(namespace, f"{self.config_name}_updates", None)
        if updates is None:
            updates = []
        updates.append(values)
        setattr(namespace, f"{self.config_name}_updates", updates)


class TrainingProgram:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_base_arguments()

    def _add_base_arguments(self):
        self.parser.add_argument(
            '--wandb_project', type=str, default=None, help='name of wandb project'
        )
        self.parser.add_argument(
            '--compute_config_file', type=str, default=None, help='path to compute config'
        )
        self.parser.add_argument(
            '--training_config_file', type=str, default=None, help='path to training config'
        )
        self.parser.add_argument(
            '--data_config_file', type=str, default=None, help='path to data config'
        )
        self.parser.add_argument(
            '--local_save_dir', type=str, default='./results', help='path to save results'
        )
        self.parser.add_argument(
            '--seed', type=int, default=None, help='random seed (default: random)'
        )
        self.parser.add_argument(
            '--enable_checks',
            action='store_true',
            help='enable checks (default: False)',
        )
        self.parser.add_argument(
            '--loglevel', type=str, default='info', help='log level (default: debug)'
        )
        # self.parser.add_argument(
        # '--device', type=str, default='cpu', help='jax device (default: cpu)'
        # )
        self.parser.add_argument(
            '--data_path',
            type=str,
            default='./data/calibrated_data',
            help='path to xp data directory',
        )
        self.parser.add_argument(
            '--wandb_plot_period',
            type=int,
            default=-1,  # only at the end
            help='wandb plot period, None = no plots, -1 = only at the end',
        )

        self.parser.add_argument(
            '--wandb_eval_period',
            type=int,
            default=-1,
            help='wandb eval plot period, None = no plots, -1 = only at the end',
        )
        self.parser.add_argument(
            '--wandb_save_period',
            type=int,
            default=-1,
            help='wandb params save period, None = no save, -1 = only at the end',
        )

        self.parser.add_argument(
            '--config',
            type=str,
            action=partial(UpdateConfigAction, 'config'),
            help='update training_config with format: <parameter>=<value>',
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, default_args=None):

        import sys

        is_notebook = 'ipykernel' in sys.modules

        ut.logger.info(f'is_notebook: {is_notebook}')

        extra_args = default_args if default_args is not None else []

        # combine parsed args and extra_args. parsed args have priority over extra_args.
        # if we're in a notebook, only use extra_args. Otherwise we can combine them.
        if is_notebook:
            self.args = self.parser.parse_args(extra_args)
        else:
            self.args = self.parser.parse_args(extra_args + sys.argv[1:])
            ut.logger.info(f'args: {self.args}')

        # load the 3 config files (training, compute, data)
        self.training_config = DEFAULT_TRAINING_CONFIG
        if self.args.training_config_file is not None:
            if not Path(self.args.training_config_file).is_file():
                raise ValueError(f'{self.args.training_config_file} is not a file')
            self.training_config = json.load(open(self.args.training_config_file))

        self.compute_config = cmp.DEFAULT_COMPUTE_CONFIG
        if self.args.compute_config_file is not None:
            if not Path(self.args.compute_config_file).is_file():
                raise ValueError(f'{self.args.compute_config_file} is not a file')
            self.compute_config = cmp.ComputeConfigManager.from_file(self.args.compute_config_file)

        self.data_config = du.DEFAULT_DATA_CONFIG
        if self.args.data_config_file is not None:
            if not Path(self.args.data_config_file).is_file():
                raise ValueError(f'{self.args.data_config_file} is not a file')
            self.data_config = json.load(open(self.args.data_config_file))

        if self.args.enable_checks:
            ut.set_enable_checks(True)

        self.local_save_dir = Path(self.args.local_save_dir)
        ut.logger.info(f"Saving results to {self.local_save_dir}")

        # loglevel
        ut.set_loglevel(self.args.loglevel)

        if self.args.seed is not None:
            self.seed = self.args.seed
        else:
            self.seed = np.random.randint(0, 2**32)

    def update_config_from_args(self):
        # Apply updates to the training_config dict
        print(f'config_updates: {self.args.config_updates}')
        updates = getattr(self.args, f"config_updates", [])
        for update in updates:
            ut.logger.info(f"Updating training_config with {update}")
            parameter, value = update.split('=')
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Keep value as a string if it's not JSON-parseable
            self.training_config[parameter] = value

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self, 'args') and hasattr(self.args, attr):
            return getattr(self.args, attr)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{attr}'")


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     command line arguments     --

prog = cm.CLIProgram()

prog.add_argument(
    '--wandb_project', type=str, default=None, help='name of wandb project'
)

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
    '--export_dir',
    help='Directory to save the results and the model',
    default=(Path(cte.BIOCOMP_ROOT) / 'training_runs').as_posix(),
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
        '--config',
        'n_replicates=5',
        '--config',
        'n_epochs=2',
    ]
)


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

wb_run.finish()

##────────────────────────────────────────────────────────────────────────────}}}


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




# after training, we call the predict tool with the best model to generate the predictions in a local folder
# then we upload the predictions to wandb AND to dropbox 
# (if we ever run out of space on wandb, we can just link to the served dropbox folder)
# one important aspect of predictions is the mode: do we want to predict at the same location as the training set, or on a grid + mask?
# we log everything to the database


# ↓ ⇩ ☟ ⟱
# ╰───⬍──🡇─🠻──☟─🡃────🡻───ᐁ─ˬ───̬⇩──ᐯ────⍖─────ˇ─────⤈───────ᗐ──╯


### {{{              --     model retrieval and loss plots     --
def get_best_run_id(losses, smooth_window=20, return_smooth_losses=False):
    from scipy.ndimage import gaussian_filter1d

    smoothed_losses = [gaussian_filter1d(loss, smooth_window) for loss in losses]
    best_loss = np.argmin([loss[-1] for loss in smoothed_losses])
    if return_smooth_losses:
        return best_loss, smoothed_losses
    return best_loss


def retrieve_wandb_results(project_name, entity='jdisset', with_losses=True, **kw):
    import wandb
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    wandb.login()
    api = wandb.Api()
    project_path = f"{entity}/{project_name}" if entity else project_name
    runs = api.runs(project_path, **kw)

    if with_losses:

        def get_loss_history(run):
            if 'loss' in run.summary and run.summary['loss'] is not None:
                history = run.scan_history(keys=['loss'], page_size=25000)
                losses = [row["loss"] for row in history]
                return np.array(losses)
            else:
                return np.array([np.inf])

        with ThreadPoolExecutor() as executor:
            full_losses = list(tqdm(executor.map(get_loss_history, runs), total=len(runs)))

        return runs, full_losses

    return runs


def get_wandb_trained_params(run, save_to=None):
    if save_to is None:
        save_to = Path(f'/tmp/biocomp_runs/{run.name}')
    save_to.mkdir(parents=True, exist_ok=True)
    param_file = run.file('latest_params.pkl').download(replace=True, root=save_to)
    trained_params = ut.load(param_file.name)
    shared_trained_params, local = trained_params.filter_by_tag('shared')
    compute_config_file = run.file('compute_config.json').download(replace=True, root=save_to)
    training_config_file = run.file('training_config.json').download(replace=True, root=save_to)
    compute_config = cmp.ComputeConfigManager.from_file(compute_config_file.name)
    with open(training_config_file.name, 'r') as f:
        training_config = json.load(f)
    shared_trained_params.set_read_only(True)
    return shared_trained_params, compute_config, training_config, local


def get_wandb_archive(run, save_path=None, filename=None):
    (
        shared_trained_params,
        compute_config,
        training_config,
        local,
    ) = get_wandb_trained_params(run, save_to=None)

    archive = {
        'shared_parameters': shared_trained_params,
        'local_parameters': local,
        'compute_config': compute_config,
        'training_config': training_config,
        'metadata': run.metadata,
    }

    archive_path = None
    if save_path is not None:
        if filename is None:
            date_started = run.metadata['startedAt'].split('T')[0]
            filename = f'{date_started}_{run.name}.pkl'
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        archive_path = save_path / filename
        with open(archive_path, 'wb') as f:
            pickle.dump(archive, f)
            ut.logger.info(f'Saved training archive to {archive_path}')

    return archive, archive_path


##────────────────────────────────────────────────────────────────────────────}}}

    # @classmethod
    # def from_xps(
        # cls, xplist, config=cmp.DEFAULT_COMPUTE_CONFIG, cache_location=DEFAULT_DATA_CACHE_DIR
    # ):

        # # build all networks and get all sample names, for each xp
        # # networks, samples = zip(*[xp.build_networks(**kw) for xp in xplist])
        # net_sample_pairs = []
        # for xp in xplist:
            # net_sample_pairs.append(
                # ut.get_cache(lambda: xp.build_networks(**kw), f'{str(xp)}_net', cache_location)
            # )

        # networks, samples = zip(*net_sample_pairs)

        # # get all X (independent vars) and Y (dependent vars) for each xp
        # # X, Y = zip(*[xp.get_XY(n, s) for xp, n, s in zip(xplist, networks, samples)])
        # XY_pairs = []
        # for xp, n, s in zip(xplist, networks, samples):
            # XY_pairs.append(
                # ut.get_cache(lambda: xp.get_XY(n, s, **kw), f'{str(xp)}_XY', cache_location)
            # )

        # X, Y = zip(*XY_pairs)
        # # get everything as a long concatenated list
        # X, Y, networks = (
            # list(itertools.chain(*X)),
            # list(itertools.chain(*Y)),
            # list(itertools.chain(*networks)),
        # )
        # return cls(X, Y, networks, config, cache_location=cache_location)



