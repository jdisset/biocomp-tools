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


def load_recipe_dataset(experiment: str, recipe: str, calibration: str = "latest") -> list[Any]:
    """Like ``load_paper_dataset`` but addressed by (experiment, recipe) instead of
    a NetworkSet file. ``calibration='latest'`` (default) auto-picks the highest-
    version calibration; any other value is an iRegex (e.g. ``v3_satfix``)."""
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.figuremakers.datasetsummary import filter_compatible
    from biocomptools.toollib.networkselector import NetworkSelector, iRegex

    sel = NetworkSelector(
        experiment_name=experiment,
        recipe_name=recipe,
        latest_calibration=calibration == "latest",
        calibration_name=None if calibration == "latest" else iRegex(calibration),
    )
    return filter_compatible(DBSource(content=[sel]).get_data())


def load_figure_dataset(
    dataset_file: str | None = None,
    experiment: str | None = None,
    recipe: str | None = None,
    calibration: str = "latest",
) -> list[Any]:
    """Unified figure data source: a NetworkSet file, or an (experiment, recipe) pair."""
    if dataset_file:
        return load_paper_dataset(dataset_file)
    if experiment and recipe:
        return load_recipe_dataset(experiment, recipe, calibration)
    raise ValueError("load_figure_dataset: provide either dataset_file or (experiment, recipe)")


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

    src = DBSource(
        content=[
            {
                "experiment_name": xp_name,
                "recipe_name": rcp_name,
                "calibration_name": iRegex(calibration_regex),
            }
        ]
    )
    return src.get_data()[0]


def family_members(
    xp_name: str,
    tokens: list[str],
    recipe_tmpl: str,
    output_name: str | None = None,
    rescaler: Any = None,
    calibration_regex: str = ".*[Ff][Ii][Nn][Aa][Ll].*",
) -> list[JeanplotPlotData]:
    """Ordered 1-input jeanplot ``PlotData`` per token for an overlaid curve family.

    ``recipe_tmpl`` is a ``{tok}`` ``str.format`` template (e.g. ``".*uORF_{tok}$"``).
    Each recipe yields both transfection directions; ``output_name`` picks one.
    """
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.networkselector import iRegex

    out = []
    for t in tokens:
        src = DBSource(
            content=[
                {
                    "experiment_name": xp_name,
                    "recipe_name": iRegex(recipe_tmpl.format(tok=t)),
                    "calibration_name": iRegex(calibration_regex),
                }
            ]
        )
        ds = src.get_data()
        if output_name is not None:
            ds = [d for d in ds if d.output_name == output_name]
        assert ds, f"no dataset: xp={xp_name} recipe~{recipe_tmpl.format(tok=t)} out={output_name}"
        out.append(_biocomp_to_jeanplot(ds[0], rescaler=rescaler))
    return out


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
    src = DBSource(
        content=[
            {
                "experiment_name": xp_name,
                "recipe_name": rcp_name,
                "calibration_name": iRegex(calibration_regex),
            }
        ]
    )
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

    data = DBSource(
        content=[
            {
                "experiment_name": Regex(xp),
                "recipe_name": Regex(recipe),
                "calibration_name": Regex(calib),
            }
        ]
    )
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


# --- replicate strip + metrics (per-topology row/table) ---------------------

# Canonical token order for model names (e.g. "SNMLcLbrLbl").
_MODEL_CANON = ["S", "N", "M", "Lc", "Lbr", "Lbl"]
_MODEL_GREEDY = ["Lbl", "Lbr", "Lc", "S", "M", "N"]  # longest-first for parsing


def normalize_model_name(slug: str | None) -> str | None:
    """Reorder a model architecture code into canonical ``S,N,M,Lc,Lbr,Lbl`` order."""
    if not slug:
        return slug
    found: set[str] = set()
    i = 0
    while i < len(slug):
        for tok in _MODEL_GREEDY:
            if slug[i : i + len(tok)] == tok:
                found.add(tok)
                i += len(tok)
                break
        else:
            i += 1
    return "".join(t for t in _MODEL_CANON if t in found) or slug


def replicate_info_for(groups: dict, network_name: str | None) -> dict | None:
    """Find the biological-replicate group a network belongs to.

    ``groups`` is ``data/replicates/groups_metadata.yaml@groups`` (passed in by the
    YAML job - the package stays path-free). Matches by ``recipe_name`` prefix of the
    network's ``network_name``. Returns ``{group_id, runs, short_name, n}`` or ``None``.
    """
    if not groups or not network_name:
        return None
    for gid, g in groups.items():
        for run in g.get("runs", []) or []:
            rn = run.get("recipe_name")
            if rn and network_name.startswith(rn):
                runs = list(g.get("runs", []))
                return {
                    "group_id": gid,
                    "runs": runs,
                    "short_name": g.get("short_name"),
                    "n": len(runs),
                }
    return None


def load_replicate_runs(runs: list[dict]) -> list[Any]:
    """One ground-truth ``PlotData`` per replicate run (selector-free; built from the
    group's ``runs`` entries). Mirrors ``plot/replicates/replicate_surfaces.yaml``."""
    import re

    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.networkselector import iRegex

    content = [
        {
            "experiment_name": r["xp"],
            "recipe_name": iRegex("^" + re.escape(r["recipe_name"]) + "$"),
            "calibration_name": iRegex("^" + re.escape(r["calibration"]) + "$"),
        }
        for r in runs
    ]
    return DBSource(content=content).get_data()


def rep_noise_for(
    group_id: str | None,
    runs: list[Any] | None = None,
    short_name: str | None = None,
) -> float | None:
    """Replicate noise = biological ``ermse_pooled`` (bioRMSE) for a group.

    Reads the replicate study's precomputed
    ``$BIOCOMP_ROOT/Plots/replicates/<gid>/sigma_repeat.yaml`` when present; else,
    if the group's loaded ``runs`` (ground-truth PlotData, e.g. from
    ``load_replicate_runs``) are given, computes it on the fly via ``compute_group``
    and caches the result back to that yaml. ``None`` when there's no precomputed
    file and fewer than two biological replicates to compute from."""
    import os

    import yaml

    root = os.environ.get("BIOCOMP_ROOT")
    if not group_id:
        return None
    path = (
        os.path.join(root, "Plots", "replicates", group_id, "sigma_repeat.yaml") if root else None
    )
    if path and os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        v = ((data.get("biological") or {}).get("aggregate") or {}).get("ermse_pooled")
        if v is not None:
            return float(v)

    if not runs:
        return None
    from biocomptools.toollib.replicate_metrics import (
        compute_group,
        runs_from_plotdata,
        write_yaml,
    )

    rr = runs_from_plotdata(runs)
    if len(rr) < 2:
        return None
    gm = compute_group(group_id, short_name or group_id, group_id, rr)
    v = gm.aggregate("biological").get("ermse_pooled")
    if v is None:
        return None
    if path:
        write_yaml(gm, path)
    return float(v)


def build_topology_table(
    rows: list[Any],
    headers: list[str] | None = None,
    column_widths: list[Any] | None = None,
    *,
    table_class: str = "topology-table",
    header_height: float = 0.22,
) -> Any:
    """Assemble per-network row bodies into one aligned jeanplot ``Table`` — **structure
    only**. All chrome (frame, grid, header band, row/column separators, dashing, rounded
    corners) is a **jstyle cascade** keyed on the Table's ``style_class`` (``table_class``,
    default ``topology-table``); the SSOT for those defaults is the
    ``Table[style_class=topology-table]`` rule hierarchy in ``paper-jobs/common/theme.yaml``
    (which deep-merges onto jeanplot's base ``Table`` rules in
    ``jeanplot/resources/themes/default.yaml``). Override by layering a more-specific rule
    or a job's ``_theme_overrides`` — never a wall of style kwargs here.

    ``rows`` are the per-network row ``Container``s (each one's ``children`` is its column
    cells, in canonical order); each cell drops straight in, so any component composes.
    The ``Table`` owns column alignment (every row's cell *c* shares column *c*'s width);
    ``column_widths`` entries are a number (fixed inches) or ``"auto"`` (natural max across
    rows). The cascade top-aligns cells so the GT / prediction surfaces line up across
    columns regardless of how many badges/captions sit under each.
    """
    from jeanplot.core.models import Size
    from jeanplot.core.table import ColumnStyle, Table, TableCell

    from biocomptools.jeanplot_panels.empty import ConstantTextPanel

    body = [list(getattr(r, "children", r)) for r in rows]
    ncols = max((len(r) for r in body), default=len(headers or []))
    if ncols == 0:
        return Table(data=[])

    widths = list(column_widths or [])
    widths += ["auto"] * (ncols - len(widths))
    col_styles = [ColumnStyle(width=widths[c]) for c in range(ncols)]

    data: list[list[Any]] = []
    n_head = 0
    if headers:
        labels = list(headers[:ncols]) + [""] * (ncols - len(headers))
        # bare header cells: just the label text; the cascade (table-header-cell /
        # table-header-row rules) paints weight/color/band/underline.
        data.append(
            [
                TableCell(children=[ConstantTextPanel(text=str(h), axes_size=Size(0.5, header_height))])
                for h in labels
            ]
        )
        n_head = 1
    data.extend([[TableCell(children=[c]) for c in row] for row in body])

    table = Table(data=data, column_styles=col_styles, header_rows=n_head)
    table.style_class = list(table.style_class) + [table_class]
    return table


register_template(load_paper_dataset)
register_template(load_recipe_dataset)
register_template(load_figure_dataset)
register_template(network_plot_data)
register_template(paper_per_network_pds)
register_template(opt_list)
register_template(paper_data)
register_template(family_members)
register_template(paper_predict)
register_template(matrix_predict)
register_template(normalize_model_name)
register_template(replicate_info_for)
register_template(load_replicate_runs)
register_template(rep_noise_for)
register_template(build_topology_table)


PAPER_PIPELINE_HELPERS: dict[str, Any] = {
    "load_paper_dataset": load_paper_dataset,
    "load_recipe_dataset": load_recipe_dataset,
    "load_figure_dataset": load_figure_dataset,
    "network_plot_data": network_plot_data,
    "paper_per_network_pds": paper_per_network_pds,
    "opt_list": opt_list,
    "paper_data": paper_data,
    "family_members": family_members,
    "paper_predict": paper_predict,
    "matrix_predict": matrix_predict,
    "normalize_model_name": normalize_model_name,
    "replicate_info_for": replicate_info_for,
    "load_replicate_runs": load_replicate_runs,
    "rep_noise_for": rep_noise_for,
    "build_topology_table": build_topology_table,
}


__all__ = [
    "load_paper_dataset",
    "load_recipe_dataset",
    "load_figure_dataset",
    "matrix_predict",
    "network_plot_data",
    "opt_list",
    "paper_data",
    "family_members",
    "paper_per_network_pds",
    "paper_predict",
    "normalize_model_name",
    "replicate_info_for",
    "load_replicate_runs",
    "rep_noise_for",
    "build_topology_table",
    "PAPER_PIPELINE_HELPERS",
]
