import os
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).parent.resolve() if '__file__' in locals() else Path.cwd()
TRAINING_OUTPUT_PATH = CURRENT_DIR / '__training_output'
os.environ['BIOCOMP_TRAINING_OUTPUT_PATH'] = str(TRAINING_OUTPUT_PATH)
os.environ['BIOCOMP_CACHE_DIR'] = str(CURRENT_DIR / '__temp_cache')

import matplotlib.pyplot as plt

# print current backend for matplotlib:
print("Current matplotlib backend:", plt.get_backend())

import time
import json
from biocomptools.toollib.networkselector import CleanupFilter, build_data_manager
from biocomp.library import load_lib
import biocomptools.toollib.models as md
from biocomptools.toollib.common import config
from biocomptools.run_training import TrainingProgram
from biocomptools.run_training import make_context_from_types, DEFAULT_TYPES
from sqlmodel import Session
from biocomp.train import generate_batches, start
from biocomp.train import TrainingConfig, init_stack
from biocomp.datautils import DataManager
import biocomp.datautils as du
import biocomp as bc
import jax
import dracon as dr


TEST_TRAINING_FILE = CURRENT_DIR / 'jobs/test_training.yaml'

CTX = {
    **make_context_from_types(DEFAULT_TYPES),
}


def load_config(**param_overrides):
    if param_overrides is None:
        param_overrides = {}
    loader = dr.DraconLoader(context={**CTX, **param_overrides})
    conf = dict(**loader.load(TEST_TRAINING_FILE))
    return conf


conf = load_config()
tp = TrainingProgram(**conf)


## {{{                     --     expected content     --
BASE_UORF_PATH = 'Experiments/2022-11-10_uORFs_and_company/data/calibrated/final-UPVAKUSZDBKLU'
UORFS_CONTENT = [
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_1w_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/1w_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_1w_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/1w_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_1x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/1x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_1x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/1x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_2x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/2x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_2x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/2x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_3x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/3x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_3x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/3x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_4x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/4x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_4x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/4x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_5x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/5x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_5x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/5x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_6x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/6x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_6x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/6x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_8x_Y+B_eBFP2',
        datafile_path=BASE_UORF_PATH + '/8x_Y+B.parquet',
    ),
    md.NetworkDataPair(
        network_name='2022-11-10_uORFs_and_company_8x_Y+B_eYFP',
        datafile_path=BASE_UORF_PATH + '/8x_Y+B.parquet',
    ),
]
ERN_PGU3CORNER_PATH = 'Experiments/2023-11-26_MatrixPgu/data/calibrated/FINAL-KNEJE6TDNLNB2'
ERN_PGU3CORNER_CONTENT = [
    md.NetworkDataPair(
        network_name='2023-11-26_MatrixPgu_(Pgu)+(PguR_Y+B)_eBFP2-mMaroon1',
        datafile_path=ERN_PGU3CORNER_PATH + '/(Pgu)+(PguR_Y+B).parquet',
    ),
    md.NetworkDataPair(
        network_name='2023-11-26_MatrixPgu_(8xPgu)+(PguR_Y+B)_eBFP2-mMaroon1',
        datafile_path=ERN_PGU3CORNER_PATH + '/(8xPgu)+(PguR_Y+B).parquet',
    ),
    md.NetworkDataPair(
        network_name='2023-11-26_MatrixPgu_(Pgu)+(PguR8x_Y+B)_eBFP2-mMaroon1',
        datafile_path=ERN_PGU3CORNER_PATH + '/(Pgu)+(PguR8x_Y+B).parquet',
    ),
]
##────────────────────────────────────────────────────────────────────────────}}}


assert isinstance(tp.training_set, CleanupFilter)
assert len(tp.training_set.content) == len(UORFS_CONTENT) + len(ERN_PGU3CORNER_CONTENT)
N_NETS = len(tp.training_set.content)
assert all(isinstance(x, md.NetworkDataPair) for x in tp.training_set.content)
for pair in UORFS_CONTENT + ERN_PGU3CORNER_CONTENT:
    assert pair in tp.training_set.content, f"Network {pair.network_name} not found in training set"

# now we manually unroll the run() method and test as much as we can

tp._build_dman()
dman = tp._training_dman
assert isinstance(dman, DataManager)

assert len(dman._networks) == N_NETS
assert len(dman._X) == N_NETS
assert len(dman._Y) == N_NETS
xshapes = [x.shape for x in dman._X]
yshapes = [y.shape for y in dman._Y]
xdims_total = sum(x.shape[1] for x in dman._X)
ydims_total = sum(y.shape[1] for y in dman._Y)
assert ydims_total == xdims_total + N_NETS

all_x_concat = np.concatenate([x.flatten() for x in dman._X])
all_y_concat = np.concatenate([y.flatten() for y in dman._Y])
assert all_y_concat.min() >= -0.1
assert all_y_concat.max() <= 1

fig, ax = plt.subplots()
histx = np.histogram(all_x_concat, bins=100, density=True)
ax.bar(histx[1][:-1], histx[0], width=np.diff(histx[1]), align='edge', edgecolor='black', alpha=0.7)
ax.set_title('Distribution of all_x_concat')
plt.show()

##
training_config = tp.training_conf
compute_config = tp.compute_conf
assert isinstance(training_config, TrainingConfig)
assert isinstance(compute_config, bc.compute.ComputeConfig)
rng_key = jax.random.PRNGKey(0)

# stack, params = init_stack(compute_config, dman, training_config.n_replicates, rng_key)

xbatches, ybatches = generate_batches(
    dman,
    training_config.n_replicates,
    training_config.n_batches,
    training_config.batch_size,
    rng_key,
)

# assert xbatches.shape == (
#     training_config.n_replicates,
#     training_config.n_batches,
#     training_config.batch_size,
#     stack.total_nb_of_inputs,
# )
# assert ybatches.shape == (
#     training_config.n_replicates,
#     training_config.n_batches,
#     training_config.batch_size,
#     stack.total_nb_of_outputs,
# )

ybatches.shape

#

xbatches_flat = np.concatenate([x.flatten() for x in xbatches])

fig, ax = plt.subplots()
hist_batches = np.histogram(xbatches_flat, bins=100, density=True)
ax.bar(
    hist_batches[1][:-1],
    hist_batches[0],
    width=np.diff(hist_batches[1]),
    align='edge',
    edgecolor='black',
    alpha=0.7,
)
ax.set_title('Distribution of xbatches_flat')
plt.show()


def flatness_metrics(counts):
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    N = counts.sum()
    mean = N / k
    var = counts.var(ddof=0)
    cv = np.sqrt(var) / mean  # 0 = perfect coefficient of variation (i.e. uniform distribution)
    p = counts / N
    entropy_norm = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(k)  # 1 = perfect
    return cv, entropy_norm


cv, entropy = flatness_metrics(hist_batches[0])  # should be close to 0, 1
assert cv < 0.7, f"CV is too high: {cv}"
assert entropy > 0.9, f"Entropy is too low: {entropy}"  # should be close to 1


##
n_batches = training_config.n_batches
batch_size = training_config.batch_size

direct_xbatches, direct_ybatches = dman.get_batches(
    training_config.n_batches, training_config.batch_size, rng_key
)
cv, entropy = flatness_metrics(np.histogram(direct_xbatches.flatten(), bins=100, density=True)[0])
entropy
cv

xb, yb = dman._get_batches_numpy(n_batches, batch_size, rng_key, True)
xb_hist = np.histogram(xb.flatten(), bins=100, density=True)
fig, ax = plt.subplots()
ax.bar(
    xb_hist[1][:-1],
    xb_hist[0],
    width=np.diff(xb_hist[1]),
    align='edge',
    edgecolor='black',
    alpha=0.7,
)
ax.set_title('Distribution of xbatches from _get_batches_numpy')
plt.show()


xj, yj = dman._get_batches_jax(n_batches, batch_size, rng_key, True)
xj_hist = np.histogram(xj.flatten(), bins=100, density=True)
fig, ax = plt.subplots()
ax.bar(
    xj_hist[1][:-1],
    xj_hist[0],
    width=np.diff(xj_hist[1]),
    align='edge',
    edgecolor='black',
    alpha=0.7,
)
ax.set_title('Distribution of xbatches from _get_batches_jax')
plt.show()

##

I = 12
xt = dman._X[I]
yt = dman._Y[I]
density = dman._densities[I]
dthreshold = 0.025
# dthreshold = 1

xtj, ytj = du.sample_batches((yt, yt, batch_size, n_batches, density, dthreshold, rng_key))
xtj_hist = np.histogram(ytj.flatten(), bins=100, density=True)
fig, ax = plt.subplots()
ax.bar(
    xtj_hist[1][:-1],
    xtj_hist[0],
    width=np.diff(xtj_hist[1]),
    align='edge',
    edgecolor='black',
    alpha=0.7,
)
ax.set_title('Distribution of xtj from sample_batches_jax')

# CAN'T LEARN THE DISTRIBUTION IF INPUTS ARE THERE!!!!!!!!!!!!
# always 1 -> 1
##
# TODO: write dman.get_dependent_output_mask()

ni = dman._networks[I]

ni.name
ni.get_inverted_input_positions()
ni.get_output_proteins()
ni.get_dependent_output_proteins()
ni.get_dependent_output_positions()
ni.get_dependent_output_mask()

all_dependent_outputs = dman.get_dependent_output_mask()
all_dependent_outputs.mean()
