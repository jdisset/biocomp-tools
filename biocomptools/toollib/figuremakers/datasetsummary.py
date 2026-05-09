"""Panel-level Python helpers for dataset-summary rows.

``expand_panel_atomics`` is a pure-data helper called at compose-time by
``autofig_dataset_row.yaml``. It maps each panel descriptor to the list
of atomic plot tasks that should render it (one task for 1D / 2D / mvp /
diagram / circuit kinds; one cube task + ``R*C`` slice tasks for 3D data
kinds), each carrying the sub-axis spec consumed by ``FigAx.subdivide``.

The atomic flow keeps all rendering policy in YAML: dim dispatch, slice
count, z-range, per-slice styling, vlim policy. Python only computes the
list of (task, axnum, sub-axis role, slice z-value) tuples and does
matplotlib bbox math via ``FigAx.subdivide``.

Figure-metadata composition lives here too: ``build_figure_metadata`` is
the canonical entry point used by every plotting job to populate the
PDF/PNG ``Subject`` field with a consistent shape.
"""

from typing import Optional, Sequence

import numpy as np

from biocomp.datautils import DataRescaler  # noqa: F401  (re-exported via context)
from biocomp.plotutils import PlotData  # noqa: F401  (re-exported via context)


def _data_subax_spec_3d(
    cube_frac_w: float,
    gap_frac: float,
    slice_grid: Sequence[int],
    slice_vgap_frac: float,
    slice_hgap_frac: float,
) -> dict:
    """Build the FigAx.subdivide spec for a 3D data panel cell."""
    cube_w = float(cube_frac_w)
    gap_w = float(gap_frac)
    slice_w = 1.0 - cube_w - gap_w
    assert slice_w > 0, (
        f"cube_frac_w + gap_frac too large for cell ({cube_w} + {gap_w} >= 1)"
    )
    return {
        "regions": {
            "cube": {"x": 0.0, "y": 0.0, "w": cube_w, "h": 1.0},
            "slices": {
                "x": cube_w + gap_w,
                "y": 0.0,
                "w": slice_w,
                "h": 1.0,
                "grid": [int(slice_grid[0]), int(slice_grid[1])],
                "vgap_frac": float(slice_vgap_frac),
                "hgap_frac": float(slice_hgap_frac),
            },
        }
    }


def expand_panel_atomics(
    panel: dict,
    axnum: int,
    *,
    slice_grid: Sequence[int] = (3, 3),
    slice_zrange: Sequence[float] = (0.05, 0.5),
    slice_zvalues: Optional[Sequence[float]] = None,
    cube_frac_w: float = 0.57,
    cube_slice_gap_frac: float = 0.04,
    slice_vgap_frac: float = 0.04,
    slice_hgap_frac: float = 0.04,
) -> list[dict]:
    """Expand one panel descriptor into the list of atomic plot tasks for it.

    Each atomic descriptor has the shape::

        {
          "task_file": "data/atomic_2d" | "data/atomic_cube" | ...,
          "panel": <original panel dict>,
          "axnum": int,                 # parent cell index in FIG.flat_ax
          "subax_role": str | None,     # key into FigAx.subdivide(...) result
          "subax_spec": dict | None,    # spec passed to FigAx.subdivide(axnum, ...)
          "r": int, "c": int,           # slice grid coords (3D-slice atomics only)
          "z": float,                   # slice z value (3D-slice atomics only)
        }

    Non-data panels (``mvp``, ``schematic``, future kinds) expand to a
    single atomic with ``task_file = '<kind>_panel'`` and no sub-axis spec.
    Data panels expand based on ``panel['plot_data'].dimensions.input``:
    1D / 2D produce one full-cell atomic; 3D produces one cube atomic +
    ``slice_grid[0] * slice_grid[1]`` slice atomics, all sharing the same
    sub-axis spec (so ``FigAx.subdivide`` is called once per cell).
    """
    kind = panel["kind"]
    if kind != "data":
        return [
            {
                "task_file": f"{kind}_panel",
                "panel": panel,
                "axnum": axnum,
                "subax_role": None,
                "subax_spec": None,
            }
        ]

    plot_data = panel["plot_data"]
    dim = plot_data.dimensions.input

    if dim == 1:
        return [
            {
                "task_file": "data/atomic_1d",
                "panel": panel,
                "axnum": axnum,
                "subax_role": None,
                "subax_spec": None,
            }
        ]

    if dim == 2:
        return [
            {
                "task_file": "data/atomic_2d",
                "panel": panel,
                "axnum": axnum,
                "subax_role": None,
                "subax_spec": None,
            }
        ]

    if dim == 3:
        R, C = int(slice_grid[0]), int(slice_grid[1])
        if slice_zvalues is None:
            zs = np.linspace(slice_zrange[0], slice_zrange[1], R * C)
        else:
            zs = np.asarray(slice_zvalues, dtype=float)
            assert zs.size == R * C, (
                f"slice_zvalues has {zs.size} entries, expected R*C={R * C}"
            )
        spec = _data_subax_spec_3d(
            cube_frac_w=cube_frac_w,
            gap_frac=cube_slice_gap_frac,
            slice_grid=(R, C),
            slice_vgap_frac=slice_vgap_frac,
            slice_hgap_frac=slice_hgap_frac,
        )
        atomics: list[dict] = [
            {
                "task_file": "data/atomic_cube",
                "panel": panel,
                "axnum": axnum,
                "subax_role": "cube",
                "subax_spec": spec,
            }
        ]
        for r in range(R):
            for c in range(C):
                atomics.append(
                    {
                        "task_file": "data/atomic_slice",
                        "panel": panel,
                        "axnum": axnum,
                        "subax_role": "slices",
                        "subax_spec": spec,
                        "r": int(r),
                        "c": int(c),
                        "z": float(zs[r * C + c]),
                    }
                )
        return atomics

    raise ValueError(f"expand_panel_atomics: unsupported data dim={dim}")


def compose_rows(groups: Sequence[Sequence[Optional[dict]]], layout: str = "row") -> list[list[dict]]:
    """Compose row-groups into a rows-of-panels list for ``autofig_dataset_row``.

    Each group is a list of panel dicts with ``None`` entries representing
    toggled-off panels (filtered out). For ``layout='row'``, all groups
    concatenate into a single row. For ``layout='stacked'``, every non-empty
    group becomes its own row.
    """
    if layout == "row":
        flat = [p for g in groups for p in g if p is not None]
        return [flat] if flat else []
    if layout == "stacked":
        out = []
        for g in groups:
            nz = [p for p in g if p is not None]
            if nz:
                out.append(nz)
        return out
    raise ValueError(f"compose_rows: unknown layout {layout!r} (expected 'row' or 'stacked')")


def _panel_width(panel: dict, kind_widths: dict) -> float:
    """Resolve a panel's width in inches.

    Resolution order:
      1. ``panel['width']`` if explicitly set on the panel.
      2. ``kind: 'gap'`` requires explicit width (raises if missing).
      3. ``kind_widths[kind]`` — either a scalar (one width for the kind)
         or a mapping ``{input_dim: width}`` for kinds whose natural size
         depends on data dimensionality (1D / 2D / 3D ``data`` panels).
         For mapping form, the panel must carry a ``plot_data`` so
         ``dimensions.input`` can be read.
    """
    if "width" in panel:
        return float(panel["width"])
    if panel["kind"] == "gap":
        raise ValueError("gap panels require an explicit `width` (inches)")

    kw = kind_widths[panel["kind"]]
    if isinstance(kw, dict):
        plot_data = panel.get("plot_data")
        assert plot_data is not None, (
            f"per-dimension kind_widths[{panel['kind']!r}] requires the panel "
            "to carry a `plot_data` so input_dim can be resolved"
        )
        dim = int(plot_data.dimensions.input)
        if dim not in kw:
            raise KeyError(
                f"kind_widths[{panel['kind']!r}] has no entry for input_dim={dim}; "
                f"available dims: {sorted(kw)}"
            )
        return float(kw[dim])
    return float(kw)


def layout_dimensions(
    rows: list[list[dict]],
    kind_widths: dict[str, float],
    row_height: float,
    row_heights: Optional[Sequence[float]] = None,
    figure_scale: float = 1.0,
) -> dict:
    """Derive ``MultiRowGridLayout`` inputs from a rows-of-panels list.

    Resolves per-panel widths via ``panel.get('width', kind_widths[kind])``,
    normalises within each row, and computes the figure size as
    ``(max row width inches, total height inches)``. ``figure_scale``
    uniformly scales the resulting absolute figure size (relative widths
    and heights are unchanged, so layout ratios are preserved).

    ``kind: 'gap'`` panels are emitted as empty-axis spacer columns
    (mask returned in ``gap_mask`` so callers can hide them after render).
    """
    assert figure_scale > 0, f"figure_scale must be positive, got {figure_scale}"
    inches = [[_panel_width(p, kind_widths) for p in row] for row in rows]
    relative = [[w / sum(row_w) for w in row_w] for row_w in inches]
    heights = list(row_heights) if row_heights is not None else [float(row_height)] * len(rows)
    assert len(heights) == len(rows), (
        f"row_heights length {len(heights)} must equal len(rows) {len(rows)}"
    )
    h_total = sum(heights)
    return {
        "row_widths_relative": relative,
        "row_heights_relative": [h / h_total for h in heights],
        "figure_size": (
            figure_scale * max(sum(row_w) for row_w in inches),
            figure_scale * h_total,
        ),
        "gap_mask": [[p["kind"] == "gap" for p in row] for row in rows],
    }


def compose_atomics(
    rows: list[list[dict]],
    *,
    slice_grid: Sequence[int] = (3, 3),
    slice_zrange: Sequence[float] = (0.05, 0.5),
    slice_zvalues: Optional[Sequence[float]] = None,
    cube_frac_w: float = 0.57,
    cube_slice_gap_frac: float = 0.04,
    slice_vgap_frac: float = 0.04,
    slice_hgap_frac: float = 0.04,
) -> list[dict]:
    """Flatten rows-of-panels into atomic plot tasks with row-major axnums.

    Calls ``expand_panel_atomics`` for each panel in row-major order; ``axnum``
    starts at 0 for the first panel of row 0 and increments by 1 per panel
    (regardless of how many atomics the panel produces — atomics inside one
    panel share an axnum and target sub-axes via ``FigAx.subdivide``).
    """
    out: list[dict] = []
    axnum = 0
    for row in rows:
        for panel in row:
            # `gap` panels are layout-only spacers — they own a column in
            # the gridspec (and thus an axnum / `flat_ax` slot) but produce
            # no atomic. The layout hides their axes via `gap_mask`.
            if panel["kind"] == "gap":
                axnum += 1
                continue
            out.extend(
                expand_panel_atomics(
                    panel,
                    axnum,
                    slice_grid=slice_grid,
                    slice_zrange=slice_zrange,
                    slice_zvalues=slice_zvalues,
                    cube_frac_w=cube_frac_w,
                    cube_slice_gap_frac=cube_slice_gap_frac,
                    slice_vgap_frac=slice_vgap_frac,
                    slice_hgap_frac=slice_hgap_frac,
                )
            )
            axnum += 1
    return out


def extract_plot_data_metadata(plot_data) -> dict:
    """Serializable summary of a PlotData for figure-metadata embedding.

    Includes per-channel x/y stats, network identity, datafile provenance,
    and the upstream `network_info` block. Cell type is pulled from the
    built network when available.
    """
    md = dict(plot_data.metadata or {})
    bn = md.get("built_network")
    cell_type = None
    if bn is not None:
        cell_type = (getattr(bn, "metadata", None) or {}).get("cell_type")
    x = plot_data.x
    y = plot_data.y
    return {
        "network_name": md.get("network_name"),
        "file_stem": md.get("file_stem"),
        "cell_type": cell_type,
        "datasource_type": md.get("datasource_type"),
        "datafile": md.get("datafile"),
        "n_points": int(x.shape[0]),
        "n_inputs": int(x.shape[1]),
        "input_names": md.get("input_names"),
        "output_name": md.get("output_name"),
        "ordered_input_names": md.get("ordered_input_names"),
        "input_order": md.get("input_order"),
        "x_stats": {
            "min": [float(v) for v in x.min(axis=0)],
            "max": [float(v) for v in x.max(axis=0)],
            "mean": [float(v) for v in x.mean(axis=0)],
            "std": [float(v) for v in x.std(axis=0)],
        },
        "y_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
        },
        "network_info": md.get("network_info"),
        "recipe_path": md.get("recipe_path"),
    }


def training_set_count(model) -> tuple[int, bool]:
    """``(n_trained, weights_recorded)`` for the model.

    ``n_trained`` counts entries with weight > 0 when per-network weights
    are recorded (``data_manager_info.network_weights``); otherwise it
    falls back to ``len(network_names)`` and ``weights_recorded=False``.
    """
    if model is None:
        return 0, False
    dmi = model.metadata.get('data_manager_info') or {}
    names = dmi.get('network_names', [])
    weights = dmi.get('network_weights', [])
    if weights and len(weights) == len(names):
        return sum(1 for w in weights if float(w) > 0), True
    return len(names), False


def trained_on_status(model, network_name: str | None) -> str:
    """Markdown fragment describing whether ``network_name`` was actually
    trained on by ``model``. Empty string when not in the training set.

    Distinguishes three regimes:
      - weights recorded, weight > 0  → "seen during training (w=…)"
      - weights recorded, weight == 0 → "in dataset YAML but excluded (w=0)"
      - weights NOT recorded          → "name in training YAML, weight unknown"
    """
    if model is None or not network_name:
        return ''
    dmi = model.metadata.get('data_manager_info') or {}
    names = dmi.get('network_names', [])
    if network_name not in names:
        return ''
    weights = dmi.get('network_weights', [])
    if not weights or len(weights) != len(names):
        return '[purple]*name in training set YAML, weight unknown*[/purple]'
    w = float(weights[names.index(network_name)])
    if w > 0:
        return f'[purple]*seen during training (w={w:.3g})*[/purple]'
    return '[grey]*in training YAML but excluded (w=0)*[/grey]'


def extract_model_metadata(model) -> dict:
    """Serializable summary of a BiocompModel for figure-metadata embedding."""
    mm = dict(model.metadata or {})
    return {
        "signature": model.signature,
        "experiment_name": mm.get("experiment_name"),
        "run_name": mm.get("run_name"),
        "host": mm.get("host"),
        "start_time": mm.get("start_time"),
        "end_time": mm.get("end_time"),
        "biocomp_hash": mm.get("biocomp_hash"),
        "biocomptools_hash": mm.get("biocomptools_hash"),
        "dracon_hash": mm.get("dracon_hash"),
        "base_config": mm.get("base_config"),
        "replicate": mm.get("replicate_number"),
    }


def extract_generation_metadata() -> dict:
    """When/where this figure is being rendered."""
    import datetime
    import os
    import socket

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": f"{socket.gethostname()}:{os.environ.get('USER', '?')}",
    }


def smart_title(s: str) -> str:
    """Title-case a string while preserving any word that already contains
    uppercase letters (so protein names like ``CasE`` / ``PgU`` / ``Csy4``
    survive). Pure-lowercase words go through ``str.title()`` so embedded
    digits/hyphens still get the right capitalization (``1-input`` →
    ``1-Input``).

    Examples:
        smart_title("triple region Csy4+CasE+PgU") -> "Triple Region Csy4+CasE+PgU"
        smart_title("single CasE")                  -> "Single CasE"
        smart_title("constitutive 1-input")         -> "Constitutive 1-Input"
    """
    if not s:
        return s
    return " ".join(w.title() if w.islower() else w for w in s.split(" "))


def format_z_label(z_latent: float, rescaler=None, fmt: str | None = None, prefix: str = "z=") -> str:
    """Format a slice-z label for a heatmap title.

    With ``rescaler=None`` returns the latent value formatted ``"z=0.25"``.
    With a rescaler, converts via ``rescaler.inv`` (latent → raw). When
    ``fmt`` is ``None`` (default) the same compact ``format_powers``
    formatter the x/y tick labels use is applied — yields ``"z=$3e4$"``
    rendered as `z=3e4` in matplotlib (mathtext). Pass an explicit Python
    format spec (e.g. ``".0e"``) to bypass mathtext.
    """
    if rescaler is None:
        return f"{prefix}{float(z_latent):.2f}"
    raw = float(rescaler.inv(float(z_latent)))
    if fmt is None:
        from biocomp.plotting.plotting_core import format_powers
        # n_decimals=0 → "4e3" instead of "4.4e3"; matches the visual
        # density of x/y tick labels which are mostly integer-mantissa.
        return f"{prefix}{format_powers(raw, n_decimals=0)}"
    return f"{prefix}{raw:{fmt}}"


def extract_prediction_config(pred) -> dict:
    """Config-level snapshot of a NetworkPrediction (not the per-network results)."""
    return {
        "n_networks": len(pred.network_model.network),
        "max_evals": int(pred.max_evals) if pred.max_evals is not None else None,
        "seed": int(pred.seed) if pred.seed is not None else None,
        "device": str(pred.device),
        "already_latent": bool(pred.already_latent),
        "disable_variational": bool(pred.disable_variational),
    }


def filter_compatible(D):
    """Drop NetworkDataPairs whose X-column count doesn't match the network's
    ``nb_inputs``.

    These are typically 1-input/N-output networks measured inside multi-color
    experiments: ``PlotData.force_single_output`` pre-pads the extra Y
    columns into X, so X.shape[1] > nb_inputs. The result is malformed for
    per-network plotting; skipping is simpler than reconstructing the
    mapping post-hoc.
    """
    import sys
    out = []
    for i, d in enumerate(D):
        bn = d.metadata.get('built_network')
        if bn is None:
            out.append(d)
            continue
        x_cols = d.x.shape[1] if d.x.ndim > 1 else 1
        if x_cols != bn.nb_inputs:
            name = d.metadata.get('network_name') or d.metadata.get('file_stem') or f'#{i}'
            print(
                f'[filter_compatible] skip {name}: X has {x_cols} columns '
                f'but network expects {bn.nb_inputs} inputs',
                file=sys.stderr,
            )
            continue
        out.append(d)
    return out


def build_prediction_pipeline(
    model_name: Optional[str],
    model_path: Optional[str],
    D,
    **pred_kwargs,
):
    """SSOT prediction-pipeline builder.

    Returns ``(model, pred, prediction_data)``. When both ``model_name`` and
    ``model_path`` are ``None`` (data-only mode), returns
    ``(None, None, [None]*len(D))`` without touching ``BiocompModel.resolve``
    or constructing a ``NetworkPrediction`` — letting the caller leave the
    model identifier null when no prediction is needed.

    Wrapped as a single helper so dracon's eager identifier resolution (which
    short-circuits at eval time but resolves all referenced symbols upfront)
    sees a safe ``None``-returning value rather than a chain of lazy
    references that fail one after the other.
    """
    if model_name is None and model_path is None:
        return None, None, [None] * len(D)
    from biocomptools.modelmodel import BiocompModel, NetworkModel
    from biocomptools.toollib.networkprediction import NetworkPrediction

    model = BiocompModel.resolve(name=model_name, path=model_path)
    pred = NetworkPrediction(
        predict_at=[d.x for d in D],
        ground_truth=[d.y for d in D],
        per_prediction_info=[d.metadata for d in D],
        network_model=NetworkModel(
            model=model,
            network=[d.metadata["built_network"] for d in D],
        ),
        **pred_kwargs,
    )
    return model, pred, pred.get_data_lazy()


def predicted_stats(predicted_data) -> dict:
    """Force-evaluate `predicted_data.y` to populate the stats cache, then
    return `prediction_stats`. Returns ``{}`` when ``predicted_data`` is
    ``None`` (data-only mode where no prediction was computed).
    """
    if predicted_data is None:
        return {}
    _ = predicted_data.y  # populate prediction_stats cache via lazy eval
    return dict(predicted_data.metadata.get("prediction_stats", {}))


def build_figure_metadata(
    *,
    dataset_file: Optional[str] = None,
    plot_data=None,
    plot_data_list: Optional[Sequence] = None,
    model=None,
    model_meta: Optional[dict] = None,
    pred=None,
    pred_meta: Optional[dict] = None,
    metrics: Optional[dict] = None,
    **extra,
) -> dict:
    """Single entry point for figure-metadata composition.

    Pass `plot_data` for per-network figures, `plot_data_list` for aggregate
    figures across multiple networks, or neither for figures with no
    underlying network data.

    Pass `model`/`pred` to extract metadata directly, OR pass `model_meta`/
    `pred_meta` (pre-extracted dicts) when the call site lives inside a
    deferred whose context has cleared `pred`/`model`. The pre-extracted
    form wins when both are provided.

    Any additional kwargs are merged into the result under their own keys
    (escape hatch for one-off context).
    """
    out: dict = {"generated": extract_generation_metadata()}
    if dataset_file is not None:
        out["dataset_file"] = dataset_file
    if model_meta is not None:
        out["model"] = model_meta
    elif model is not None:
        out["model"] = extract_model_metadata(model)
    if pred_meta is not None:
        out["prediction"] = pred_meta
    elif pred is not None:
        out["prediction"] = extract_prediction_config(pred)
    if plot_data is not None:
        out["data"] = extract_plot_data_metadata(plot_data)
    elif plot_data_list is not None:
        out["data"] = {
            "n_networks": len(plot_data_list),
            "networks": [extract_plot_data_metadata(pd) for pd in plot_data_list],
        }
    if metrics is not None:
        out["metrics"] = metrics
    out.update(extra)
    return out
