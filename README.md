# biocomp-tools

The Python side of the biocompiler. Training, plotting, dataset selection,
design, replay, hyperopt. Companion to the core `biocomp` framework.

If you don't know what a BNN or a circuit is, read `biocomp/README.md` first.
Then come back.

## what's in here

- **`biocomp-plot`** - plot stuff. Mostly flow cytometry. YAML-driven via dracon. (Legacy; new work uses `jeanplot` — see `jeanplot_panels/` below.)
- **`biocomptools.jeanplot_panels`** - jeanplot Component shells over biocomp-domain draw functions (CircuitPanel, NetworkDiagramPanel, MVPNetworkPanel, BlurbPanel, smooth-voxel / benchmark / quantile-coverage panels, the `build_per_network_row` Container composer, plus biocomp-aware data holders). Used by `paper-jobs/plot/` to render via `jeanplot +foo.yaml` while keeping all biocomp data loading on this side. `!import biocomptools.jeanplot_panels` in any `jeanplot` YAML brings every panel + helper into tag namespace.
- **`biocomp-train`** - train models.
- **`biocomp-design`** - design circuits (inverse mode).
- **`biocomp-replay`** - re-run loggers on a saved run without re-doing the work.
- **`biocomp-hyperopt`** / **`biocomp-design-hyperopt`** - optuna sweeps.
- **`biocomp-eval`** / **`biocomp-check`** / **`biocomp-updatedb`** / **`biocomp-circuitplot`** / **`biocomp-model-prepare`** - smaller tools you'll find as you need them.

The package also defines the pydantic models used everywhere
(`BiocompModel`, `NetworkPrediction`, `NetworkSet`, ...) and a few core ideas:

- **DataSources** - anything that gives you data to plot, predict on, or train on.
- **NetworkSet / Union / Difference / Filters** - set-algebra over experimental data.
- **Figures + Tasks** - composable plot units, defined in YAML.

## install

editable install from a checkout:

```bash
pip install -e .
```

you'll also need `biocomp` (sibling repo) and a few env vars - look at
`biocomptools/configs/default.yaml` for the list (root path, db path, mlflow
server, parts-db, etc.).

## quick start

```bash
biocomp-train +biocomp-jobs/train/start
biocomp-plot +biocomp-jobs/plot/dataset_prediction.yaml
biocomp-design +biocomp-jobs/design/start ++architecture=two_and_one
```

real examples live in `biocomp-jobs/` (separate repo). docs live in
`biocomp-doc/` (also separate). dracon syntax: see the dracon repo.

## license

MIT. see `LICENSE`.
