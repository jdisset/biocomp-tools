# biocomp-tools
The biocomp-tools repo is a collection of tools, modules, functions that are used in the biocomp project, which is a machine learning research project that trains a foundation model of synthetic biology. It also defines some pydantic models of several important classes.
2 major tools are:
- the plotting module (biocomp-plot): it leverages biocomp plotting functions as well as dracon in order to plot varied biological data (mostly flow cytometry data). It revolves around the concept of Figure and Tasks.
- the training tool (biocomp-training): it deploys training runs of the core biocomp model

Another central concept is that of DataSources, which are classes that provide data either to plot, predict on, or train on.
Biocomp-tools also provide useful classes and helpers to define set-based selection of data (e.g. for training or plotting, we can define union, intersection, difference of experimental data, apply filters, etc.)
It also provides helpers to wrap around a model (which needs to be saved with it's hyperparameters, aka ComputeConfig) and make predictions with it.

