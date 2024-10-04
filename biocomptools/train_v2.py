## {{{                          --     imports     --
from biocomp import utils as ut
import biocomp as bc
import optax
from labellines import labelLines, labelLine
from biocomp.recipe import get_network_XY
import jax
import biocomp.compute as cmp
from biocomp.train import DEFAULT_TRAINING_CONFIG as training_config
from pathlib import Path
from functools import partial
import numpy as np
import biocomp.datautils as du
from biocomp import nodes
from pydantic import Field, BaseModel, BeforeValidator

import dracon as dr
from sqlmodel import Session, select
import time
import pandas as pd
from typing import Optional, List, Tuple, TypeVar, Dict, Any, Callable, Annotated
from tqdm import tqdm
from biocomp.utils import EncodedPartialFunction

import logging
from biocomptools.toollib.common import config
import biocomptools.toollib.models as md
from rich.console import Console

logger = logging.getLogger('build_xp_table')
logger.setLevel(logging.DEBUG)
logging.getLogger('biocomp').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     compute config     --

ROOT = Path(config.paths.root).expanduser().resolve()
lib = ut.load_lib()
engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, True)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                   --     get data and networks     --

# get all xps and recipes
with Session(engine) as session:
    experiments = session.exec(select(md.Experiment)).all()
    recipes = [recipe for xp in experiments for recipe in xp.recipes]
    datafiles = [recipe.get_best_datafile() for recipe in recipes]
    assert len(datafiles) == len(recipes), f"{len(datafiles)=} != {len(recipes)=}"
    assert all(datafiles), f"{datafiles=}"


data, networks = [], []
for recipe, datafile in tqdm(list(zip(recipes, datafiles))):
    for n in recipe.build_networks(lib=lib):
        data.append(
            get_network_XY(n._network, ROOT / datafile.file)  # type: ignore
        )
        networks.append(n._network)

X, Y = zip(*data)

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                           --     train     --
datamanager = du.DataManager(X, Y, networks)

# params, loss_history, epoch_history = bc.train.start(
    # datamanager,
    # training_config,
    # compute_config,
    # [(1, bc.trainutils.console_log)],
# )


##────────────────────────────────────────────────────────────────────────────}}}##

## {{{                           --     dump     --
# save params, loss history and epoch history in a pickle:
import pickle

fname = Path('./tmp/train_results.pkl')
fname.parent.mkdir(exist_ok=True, parents=True)
with open(fname, 'wb') as f:
    pickle.dump(
        {
            'params': params,
            'loss_history': loss_history,
            'epoch_history': epoch_history,
        },
        f,
    )


##────────────────────────────────────────────────────────────────────────────}}}##

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import wandb as wb


def get_best_smoothed_loss_id(
    loss_history: List[ndArray], sigma: float = 50.0
) -> Tuple[int, np.ndarray]:
    all_losses = np.hstack(loss_history)
    smoothed_losses = gaussian_filter1d(all_losses, sigma=sigma)
    best_loss_id = int(np.argmin(smoothed_losses[:, -1]))
    return best_loss_id, smoothed_losses


best_loss_id, smoothed_losses = get_best_smoothed_loss_id(loss_history)
all_losses = np.hstack(loss_history)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(all_losses.T, alpha=0.5, color='gray', linewidth=0.5)
# ax.plot(smoothed_losses.T)
for i in range(smoothed_losses.shape[0]):
    ax.plot(smoothed_losses[i], label=f'Replicate {i}')
labelLines(ax.get_lines(), zorder=2.5)
ax.set_yscale('log')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss')

##
# save the plot in the save_dir
# plt.savefig(save_dir / 'losses.png')
# plt.close(fig)
best_params = ut.tree_get(params, best_loss_id)

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
