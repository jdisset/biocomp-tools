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

from biocomp.datautils import DEFAULT_DATA_CONFIG
from constants import (
    DEFAULT_DATA_CONFIG_PATH,
    BIOCOMP_ROOT,
)

##────────────────────────────────────────────────────────────────────────────}}}

"""
Utils for plotting data in various ways, from the network representation of an experiment.
It can build this network from scratch given a recipe, library and data file, or it can
use a database file to load the network and plot it.
"""

### {{{                   --     constants and config     --
lib = ut.load_lib()
DEFAULT_OUTPUT_DIR = Path(BIOCOMP_ROOT) / 'biocomp-static/dataplots'
##────────────────────────────────────────────────────────────────────────────}}}

### {{{                --     arg declaration and parsing     --
prog = cm.CLIProgram()

# and a network id (or list of ids, or 'all')
prog.add_argument(
    '--network_id', help='network id to plot: int, list of network ids, or "all"', default='all'
)

prog.add_argument('--data_config', type=str, default=None)
prog.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
prog.add_argument('--plot_root', type=str, default='dataplots', help='Path to prepend to plot URLs')

prog.parse_args()

netdf = cm.table_to_df('network')

prog.database_mode = True

if prog.args.network_id == 'all':
    net_names = netdf['name'].tolist()
else:
    raise NotImplementedError('Only "all" is supported for network_id right now')

prog.args.output_dir = Path(prog.args.output_dir).resolve()
prog.args.output_dir.mkdir(parents=True, exist_ok=True)

if prog.data_config is None:
    prog.data_config = DEFAULT_DATA_CONFIG
else:
    import json5

    assert Path(prog.data_config).exists()
    prog.data_config = json5.load(open(prog.data_config, 'r'))


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                --     load networks and data     --

load_errors = {}


def error_handler(net_name, e):
    load_errors[net_name] = f'{e.__class__.__name__}: {e}'
    return None, None, None


netdf['network_obj'], netdf['X'], netdf['Y'] = cm.load_networks_and_data(
    netdf, lib, error_handler=error_handler
)

netdf['plot_error'] = netdf['name'].map(load_errors)


# cm.update_table(netdf, 'network', key_column='name', columns=['plot_error'], update_only=True)

plotdf = netdf[netdf['plot_error'].isna()]
plotdf = plotdf.reset_index(drop=True)

dman = du.DataManager(
    plotdf['X'].tolist(),
    plotdf['Y'].tolist(),
    plotdf['network_obj'].tolist(),
    data_cfg=prog.data_config,
)

# drop new columns so they don't get transmitted to the workers
plotdf = plotdf.drop(columns=['network_obj', 'X', 'Y'])
netdf = netdf.drop(columns=['network_obj', 'X', 'Y'])


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                       --     plot function     --


def generate_config(n_inputs, extra_args=None):
    cm.tlog.debug(f'Generating config for {n_inputs} inputs, extra_args: {extra_args}')
    extra_args = extra_args or {}
    plot_config = pu.BASE_DEFAULT_CONFIG
    plot_config['kde'] = False
    if n_inputs == 1:
        plot_config = ut.updated_dict(plot_config, pu.DEFAULT_1D_CONFIG)
    elif n_inputs == 2:
        plot_config = ut.updated_dict(plot_config, pu.DEFAULT_2D_CONFIG)
    elif n_inputs == 3:
        plot_config = ut.updated_dict(plot_config, pu.DEFAULT_3D_CONFIG)
    else:
        raise NotImplementedError(f'Plotting {n_inputs} inputs is not implemented')
    plot_config = ut.updated_dict(plot_config, extra_args)
    return plot_config


def do_plot(network, x, y, rescaler, plot_config):
    ax, axes = None, None
    fig = None
    n_inputs = network.get_nb_inputs()
    input_order = plot_config.get('input_order', None)
    if 'input_order' in plot_config:
        del plot_config['input_order']
    if n_inputs <= 2:
        fig, ax = pu.mkfig(1, 1, size=plot_config['size'])
        if input_order is None:
            input_order = list(range(n_inputs))
        pu.direct_network_plot(
            network, x, y, rescaler, ax=ax, input_order=input_order, **plot_config
        )
    else:
        if 'slices' not in plot_config:
            raise ValueError('You must specify slices for 3D plots')
        if plot_config['method'] == 'smooth':
            nslices = len(plot_config['slices'])
            if input_order is None:
                # we plot every ordering
                fig, axes = pu.mkfig(n_inputs, nslices, size=plot_config['size'])
                for i in range(n_inputs):
                    iorder = list(range(n_inputs))
                    iorder = iorder[i:] + iorder[:i]
                    pu.direct_network_plot(
                        network, x, y, rescaler, axes=axes[i, :], input_order=iorder, **plot_config
                    )
            else:
                fig, axes = pu.mkfig(1, nslices, size=plot_config['size'])
                pu.direct_network_plot(
                    network, x, y, rescaler, axes=axes, input_order=input_order, **plot_config
                )
    return fig


def make_network_title(net_row):
    title = r"\fontsize{12}{12}\selectfont " + net_row.recipe_name + '\n'
    title += r"\fontsize{8}{8}\selectfont from " + net_row.xp + '\n'
    return title


def save_plot(net_row, fig, fpath):
    title = make_network_title(net_row)
    fig.suptitle(title, fontsize=12)
    fig.savefig(
        fpath,
        bbox_inches='tight',
        pad_inches=0.05,
        dpi=300,
    )
    plt.close(fig)
    cm.tlog.debug(f'Saved plot for {net_row.name} to {fpath}')


##────────────────────────────────────────────────────────────────────────────}}}


encoded_names = {net_name: cm.get_name_hash(net_name) for net_name in net_names}
plot_errors = {}

ray.init(ignore_reinit_error=True, num_cpus=12)
dman_ref = ray.put(dman)

output_dir = prog.args.output_dir


@ray.remote
def submit_plot_job(net_row, overwrite=False, **kw):
    dmanager = ray.get(dman_ref)
    network_id = net_row.Index
    net_name = net_row.name
    fpath = output_dir / f'{encoded_names[net_name]}.png'
    if fpath.exists() and not overwrite:
        cm.tlog.debug(f'File {fpath} exists, skipping')
        return net_name, None
    else:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        cm.tlog.debug(f'Plotting network {net_name} to {fpath}')
        network = dmanager.get_networks()[network_id]
        x, y = dmanager.get_X()[network_id], dmanager.get_Y()[network_id]
        rescaler = pu.DataManagerRescaler(dmanager)
        n_inputs = network.get_nb_inputs()
        plot_config = generate_config(n_inputs, extra_args=kw)
        try:
            fig = do_plot(network, x, y, rescaler, plot_config)
            save_plot(net_row, fig, fpath)
            return net_name, None
        except Exception as e:
            return net_name, f'{e.__class__.__name__}: {e}'


t0 = time.time()
results = [submit_plot_job.remote(row, overwrite=False) for row in plotdf.itertuples()]
results = ray.get(results)
t1 = time.time()
cm.log.info(f'Elapsed time: {t1 - t0:.2f} s')

##
base_dir_url = prog.args.plot_root
netdf['data_plot'] = None
for net_name, err in results:
    if err is not None:
        netdf.loc[netdf['name'] == net_name, 'plot_error'] = err
        netdf.loc[netdf['name'] == net_name, 'data_plot'] = None
    else:
        fname = f'{base_dir_url}/{encoded_names[net_name]}.png'
        netdf.loc[netdf['name'] == net_name, 'data_plot'] = fname
        netdf.loc[netdf['name'] == net_name, 'plot_error'] = None

cm.log.info(f'Total plots: {netdf["data_plot"].notna().sum()}')

##

cm.update_table(
    netdf, 'network', key_column='name', columns=['data_plot', 'plot_error'], update_only=True
)
