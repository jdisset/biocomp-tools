from biocomptools.trainutils import plot_loss, get_best_smoothed_loss_id
import pickle
import numpy as np
from biocomptools.modelmodel import BiocompModel, get_shared_params
import biocomp.utils as ut
from pathlib import Path
import dracon as dr


trainingdir = (
    Path('~/Dropbox (MIT)/Biocomp_v2/Training/Runs/joyful-sashay/training/').expanduser().resolve()
)

with open(trainingdir / 'all_models.pickle', 'rb') as f:
    all_models = pickle.load(f)


loss_history = np.load(trainingdir / 'loss_history.npy')
plot_loss(loss_history)

all_losses = np.hstack(loss_history)
best_model_id, _ = get_best_smoothed_loss_id(all_losses)
best_params = get_shared_params(ut.tree_get(all_models, best_model_id))

training_program_file = trainingdir / 'training_program_dump'
compute_conf = dr.load(f'file:{training_program_file}@training_conf.compute_config')

model = BiocompModel(compute_config=compute_conf, shared_params=ut.tree_to_np(best_params))
# model.save(trainingdir / 'best_model.pickle')

