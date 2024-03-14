### {{{                          --     imports     --

from dataclasses import dataclass, field

from biocomptools.toollib import common as cm
from biocomptools.toollib import plot as pl

from biocomp import utils as ut
from biocomp import datautils as du
from biocomp import plotutils as pu
from typing import Optional, Union, Tuple, List, Dict, Sequence, Any

##────────────────────────────────────────────────────────────────────────────}}}


## {{{                    --     search path plugin     --

import hydra


from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


hydra.core.global_hydra.GlobalHydra.instance().clear()
# initialize(config_path="conf")
class BiocompSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # search_path.append(provider="biocomp", path="pkg://biocomp/config")
        search_path.append(provider="biocomptools", path="pkg://biocomptools/configs")

Plugins.instance().register(BiocompSearchPathPlugin)


##────────────────────────────────────────────────────────────────────────────}}}

"""
Utils for plotting data in various ways, from the network representation of an experiment.
It can build this network from scratch given a recipe, library and data file, or it can
use a database file to load the network and plot it.
Uses plotjob descriptions to define the plot to be made, and the data to be used.
"""

ut.generate_full_nested_config(namespace='biocomp.plotutils')
print(ut.dump_default_config('biocomp.plotutils'))


##

@dataclass
class FigureConfig:
    title: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    dpi: Optional[int] = 300


defaults = [
    "plotting/plotting_config@plot_config",
    "default_data_config@data_config",
    "_self_",
]


@dataclass
class PlotJob:
    data_location: str
    recipe_location: Optional[str] = None
    defaults: List[Any] = field(default_factory=lambda: defaults)
    data_config: du.DataConfig = MISSING
    plot_config: Any = MISSING
    figure: FigureConfig = field(default_factory=FigureConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=PlotJob)

# cfg = compose(config_name="plotting/plotting_config")
cfg = compose(config_name="config")

plot_data(cfg)

##


### {{{                --     load networks and data     --

lib = ut.load_lib()
netdf = cm.table_to_df('network')

# keep only the networks in xp 2023-11-17_PguConstraints1_BP_DR
# netdf = netdf[netdf['xp'] == '2024-02-18_BPv4_BPv5']
netdf = netdf[netdf['xp'] == '2023-03-26_MatrixCsy4']

# select only the rows that have a data_file not None
netdf = netdf[netdf['data_file'] != 'None']

netdf['data_file']


load_errors = {}
def error_handler(net_name, e):
    load_errors[net_name] = f'{e.__class__.__name__}: {e}'
    return None, None, None


netdf['network_obj'], netdf['X'], netdf['Y'] = cm.load_networks_and_data(
    netdf, lib, error_handler=error_handler
)

netdf['plot_error'] = netdf['name'].map(load_errors)

netdf['n_markers'] = netdf['markers'].apply(lambda x: len(x.split(',')))

plotdf = netdf.reset_index(drop=True)


data_cfg = hydra.utils.instantiate(cfg.data_config)
dman = du.DataManager(
    plotdf['X'].tolist(),
    plotdf['Y'].tolist(),
    plotdf['network_obj'].tolist(),
    data_cfg=data_cfg,
)

# drop new columns so they don't get transmitted to the workers
# plotdf = plotdf.drop(columns=['network_obj', 'X', 'Y'])
# netdf = netdf.drop(columns=['network_obj', 'X', 'Y'])


##────────────────────────────────────────────────────────────────────────────}}}


def network_plot(network, x, y, rescaler, mkfig_conf=None, plot_config=None):
    ax, axes = None, None
    fig = None
    n_inputs = network.get_nb_inputs()
    if plot_config is None:
        plot_config = {}
    if mkfig_conf is None:
        mkfig_conf = {}
    input_order = plot_config.get('input_order', None)
    if 'input_order' in plot_config:
        del plot_config['input_order']
    if n_inputs <= 2:
        fig, ax = pu.mkfig(1, 1, **mkfig_conf)
        if input_order is None:
            input_order = list(range(n_inputs))
        pu.direct_network_plot_new(
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


def get_network_and_data_from_row(net_row, dman):
    network_id = net_row.Index
    dmanager = dman
    network = dmanager.get_networks()[network_id]
    x, y = dmanager.get_X()[network_id], dmanager.get_Y()[network_id]
    rescaler = pu.DataRescaler.from_data_manager(dmanager)
    return network, x, y, rescaler

df = plotdf
ptuples = list(df.itertuples(index=True))
IDX = 4

# plotconf = OmegaConf.create(pu.generate_full_nested_config(local_conf))
plotconf = cfg.plot_config

print()
# print(pu.dump_default_config())

# well I guess I should try switching to the new 3D plot function

network, x, y, rescaler = get_network_and_data_from_row(ptuples[IDX], dman)
fig = pu.network_figure(network, x, y, rescaler, **plotconf.plot_callstack_params.network_figure_params)



