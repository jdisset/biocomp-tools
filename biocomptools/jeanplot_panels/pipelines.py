# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Paper-figure data pipeline helpers, exposed as dracon `!fn` templates.

These wrap existing biocomp helpers (``datasetsummary``, ``DBSource``,
``DataRescaler``) into named stages that paper-jobs YAML files can compose
declaratively. The new file is purely interface — implementations remain in
``datasetsummary``/``datasources``.
"""

from typing import Any

from dracon import register_template

from jeanplot.data.plot_data import PlotData as JeanplotPlotData
from biocomptools.jeanplot_panels.data import _biocomp_to_jeanplot


def load_paper_dataset(dataset_file: str) -> list[Any]:
    """Read a NetworkSet/CleanupFilter YAML, return ``filter_compatible`` output."""
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.figuremakers.datasetsummary import filter_compatible
    from biocomptools.toollib import networkselector as _ns
    import dracon as dr

    loader = dr.DraconLoader(
        enable_interpolation=True,
        capture_globals=True,
        context={
            "NetworkSet": _ns.NetworkSet,
            "NetworkSetUnion": _ns.NetworkSetUnion,
            "NetworkSetIntersection": _ns.NetworkSetIntersection,
            "NetworkSetDifference": _ns.NetworkSetDifference,
            "NetworkSelector": _ns.NetworkSelector,
            "NetworkFilter": _ns.NetworkFilter,
            "CleanupFilter": _ns.CleanupFilter,
            "Regex": _ns.Regex,
            "iRegex": _ns.iRegex,
        },
    )
    content = loader.load(dataset_file)
    src = DBSource(content=content)
    return filter_compatible(src.get_data())


def network_plot_data(D: list[Any], index: int = 0, rescaler: Any = None) -> JeanplotPlotData:
    """Build a jeanplot ``PlotData`` for one network, optionally rescaling x/y."""
    return _biocomp_to_jeanplot(D[index], rescaler=rescaler)


def paper_per_network_pds(
    D: list[Any],
    rescaler: Any = None,
) -> list[JeanplotPlotData]:
    """One jeanplot ``PlotData`` per network, all with the same rescaler applied."""
    return [network_plot_data(D, index=i, rescaler=rescaler) for i in range(len(D))]


def opt_list(item: Any) -> list:
    """Wrap a single item in a one-element list; ``None`` becomes ``[]``.

    Lets paper-panel templates conditionally include overlay specs without
    multi-line ``${...}`` blocks or ``!if`` keys nested in a sequence (which
    YAML cannot parse).
    """
    return [item] if item is not None else []


def paper_data(
    xp_name: str,
    rcp_name: str,
    calibration_regex: str = ".*[Ff][Ii][Nn][Aa][Ll].*",
):
    """Load experimental PlotData for one (xp, recipe) without running a model."""
    from biocomp.network import recipe_to_networks  # noqa: F401
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.networkselector import iRegex

    src = DBSource(content=[{
        "experiment_name": xp_name,
        "recipe_name": rcp_name,
        "calibration_name": iRegex(calibration_regex),
    }])
    return src.get_data()[0]


def paper_predict(
    xp_name: str,
    rcp_name: str,
    model_name: str | None = None,
    model_path: str | None = None,
    calibration_regex: str = ".*[Ff][Ii][Nn][Aa][Ll].*",
    input_order: list[list[int]] = [[0, 1, 2]],  # noqa: B006
    z_value: str = "uniform",
    max_evals: int = 300000,
    mode: str = "prediction",
):
    """Load model+experiment, run a NetworkPrediction, return the first PlotData.

    When ``mode='data'`` (or both model_name/model_path are unset), skip the
    model entirely and return the raw experimental PlotData. This lets a single
    panel YAML render both fig4 c/g (predictions) and d/h (experiment) rows.
    """
    if mode == "data" or (model_name is None and model_path is None):
        return paper_data(xp_name, rcp_name, calibration_regex)
    from biocomp.network import recipe_to_networks  # noqa: F401  (ensure builders registered)
    from biocomptools.modelmodel import BiocompModel, NetworkModel
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.networkprediction import NetworkPrediction
    from biocomptools.toollib.networkselector import iRegex

    model = BiocompModel.resolve(name=model_name, path=model_path)
    src = DBSource(content=[{
        "experiment_name": xp_name,
        "recipe_name": rcp_name,
        "calibration_name": iRegex(calibration_regex),
    }])
    d_train = [src.get_data()[0]]
    pred = NetworkPrediction(
        predict_at=[d.x for d in d_train],
        ground_truth=[d.y for d in d_train],
        per_prediction_info=[d.metadata for d in d_train],
        input_order=input_order,
        z_value=z_value,
        max_evals=max_evals,
        network_model=NetworkModel(
            network=[d.metadata["built_network"] for d in d_train],
            model=model,
        ),
    )
    return pred.get_data()[0]


def matrix_predict(
    xp: str,
    recipe: str,
    calib: str = ".*FINAL",
    model_name: str | None = None,
    model_path: str | None = None,
    input_order: list[list[int]] = [[1, 0]],  # noqa: B006
    z_value: str = "uniform",
    max_evals: int = 300000,
    mode: str = "prediction",
):
    """Load a uORF-bundled matrix experiment, return ``(model, D, uorf_info)``.

    With ``mode='prediction'`` (default), ``D`` is the lazy NetworkPrediction
    output and ``uorf_info`` is the per-network uORF annotation list pulled
    from the model's training dataset.

    With ``mode='data'`` (or both model_name/model_path unset), skip the
    model entirely and return the raw experimental PlotData list.
    """
    from biocomptools.modelmodel import BiocompModel, NetworkModel
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.figuremakers.uorfmatrixfigure import (
        bundle_uorf_data,
        extract_uorf_info,
    )
    from biocomptools.toollib.networkprediction import NetworkPrediction
    from biocomptools.toollib.networkselector import NetworkSet, Regex

    data = DBSource(content=[{
        "experiment_name": Regex(xp),
        "recipe_name": Regex(recipe),
        "calibration_name": Regex(calib),
    }])
    matrix_pd = bundle_uorf_data(data.get_data())[0]

    if mode == "data" or (model_name is None and model_path is None):
        return {"model": None, "D": matrix_pd, "uorf_info": None}

    model = BiocompModel.resolve(name=model_name, path=model_path)
    pred = NetworkPrediction(
        predict_at=[d.x for d in matrix_pd],
        ground_truth=[d.y for d in matrix_pd],
        per_prediction_info=[d.metadata for d in matrix_pd],
        input_order=input_order,
        z_value=z_value,
        max_evals=max_evals,
        network_model=NetworkModel(
            network=[d.metadata["built_network"] for d in matrix_pd],
            model=model,
        ),
    )

    training = getattr(model, "training_dataset", None)
    if training is not None:
        training_set = NetworkSet(content=training.network_data_pairs)
        uorf_info = [extract_uorf_info(n) for n, _ in training_set.get_networks_and_data()]
    else:
        uorf_info = None

    return {"model": model, "D": pred.get_data_lazy(), "uorf_info": uorf_info}


register_template(load_paper_dataset)
register_template(network_plot_data)
register_template(paper_per_network_pds)
register_template(opt_list)
register_template(paper_data)
register_template(paper_predict)
register_template(matrix_predict)


PAPER_PIPELINE_HELPERS: dict[str, Any] = {
    "load_paper_dataset": load_paper_dataset,
    "network_plot_data": network_plot_data,
    "paper_per_network_pds": paper_per_network_pds,
    "opt_list": opt_list,
    "paper_data": paper_data,
    "paper_predict": paper_predict,
    "matrix_predict": matrix_predict,
}


__all__ = [
    "load_paper_dataset",
    "matrix_predict",
    "network_plot_data",
    "opt_list",
    "paper_data",
    "paper_per_network_pds",
    "paper_predict",
    "PAPER_PIPELINE_HELPERS",
]
