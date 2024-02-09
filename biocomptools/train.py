### {{{                          --     imports     --
from . import common as cm
##────────────────────────────────────────────────────────────────────────────}}}


"""
    Train a model, save it, log everything, and run predictions on the test set.
    Meant to be used in a docker container.
"""


# First step is to load the data locally. For that, we need to know the list of networks to use.

# We will assume that all the necessary data is available (in the path specified by ENV variables)
# to handle that for different networks subsets and play nice with containerized deployment, 
# (and avoid deploying unnecessary data), I need a script that will create a local directory with:
# 1. symlinks to the stricly necessary data (need to query the database for that)
# 2. cloning the latest version of both biocomp and biocomp-tools
# Then we can simply build the container from the local folder (it will copy what's necessary).

# Then we start the training process, forwarding all parameters. Everything is logged to W&B.

# Once the training is done we need to collect the results from W&B, pick the best model,
# download the artifact and put them on dropbox and log everything to the database. 
# Dropbox is used to store the model, the data plots and the prediction plots. The webserver simply
# has a read-only sync with the dropbox folder.

# Finally, we run the predictions on the validation set, store the plots in dropbox
# and log everything to the database, i.e. for each prediction, we store an entry with the error, 
# the prediction plot's path, and the model's id (technically, the training_run entry's id).


# Pick a list of collection_ids to identify the networks to train on (same for the prediction set)



### {{{              --     find required networks from xp     --

##────────────────────────────────────────────────────────────────────────────}}}
