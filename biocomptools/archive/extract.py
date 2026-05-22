# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Build the master index.json for the paper-archive static site.

Walks the paper-jobs tree (models, topology_classes, metrics.csv, design runs,
tour manifests) and emits one self-contained JSON consumed by the static site.
No JAX, no DB — pure metadata aggregation."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import yaml
from dracon import Arg, dracon_program
from pydantic import BaseModel

from biocomptools.toollib.analysis.generalization.metrics_extract import METRIC_COLUMNS

FINE_CLASSES = ["N", "S", "M", "Lc", "Lbl", "Lbr"]

CLASS_LABELS = {
    "N": "no sequestron",
    "S": "single ERN",
    "M": "multi-ERN, 1 layer",
    "Lc": "multi-layer cascade",
    "Lbl": "multi-layer bandpass (TRBL)",
    "Lbr": "multi-layer bandpass (TLBR)",
}

CLASS_COLORS = {
    "N": "#888888",
    "S": "#3b82f6",
    "M": "#10b981",
    "Lc": "#f59e0b",
    "Lbl": "#ef4444",
    "Lbr": "#a855f7",
}

DESIGN_KEYS = ["mit_i", "mit_t", "mit_m"]


def _slugify(s: str) -> str:
    """Stable URL-safe slug. Preserves case but replaces non-alphanum-_."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def network_id(experiment: str, recipe_name: str) -> str:
    return f"{_slugify(experiment)}__{_slugify(recipe_name)}"


# ---- topology_classes parsing ----------------------------------------------


class _TagAgnosticLoader(yaml.SafeLoader):
    pass


def _ignore_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TagAgnosticLoader.add_multi_constructor("!", _ignore_tag)
_TagAgnosticLoader.add_multi_constructor("tag:yaml.org,2002:", _ignore_tag)


def _load_yaml_lax(path: Path) -> dict:
    """Parse a Dracon-tagged YAML ignoring all tags. Returns a plain dict."""
    text = path.read_text()
    # Strip `<<(<):` and `<<:` merge keys to bare keys — pyyaml chokes otherwise.
    text = re.sub(r"^<<\([^)]*\):", "_merge_in_:", text, flags=re.MULTILINE)
    text = re.sub(r"^<<:", "_merge_:", text, flags=re.MULTILINE)
    return yaml.load(text, Loader=_TagAgnosticLoader) or {}


def parse_topology_classes(root: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Walk root/<CLASS>/<subset>.yaml; return (networks, class_subsets).

    networks: list of {id, experiment, recipe_name, classes, display_name}
    class_subsets: {class -> [subset_name, ...]} (e.g. "S" -> ["S_1", ..., "S_19"])
    """
    by_id: dict[str, dict] = {}
    class_subsets: dict[str, list[str]] = defaultdict(list)

    for cls in FINE_CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.glob("*.yaml")):
            data = _load_yaml_lax(f)
            subset_name = f.stem
            class_subsets[cls].append(subset_name)
            merged = data.get("_merge_") or {}
            by_experiment = merged.get("by_experiment") or {}
            for xp, recipes in by_experiment.items():
                if not isinstance(recipes, list):
                    continue
                for r in recipes:
                    if r is None:
                        continue
                    nid = network_id(xp, r)
                    if nid not in by_id:
                        by_id[nid] = {
                            "id": nid,
                            "experiment": xp,
                            "recipe_name": r,
                            "classes": [],
                            "subsets": [],
                            "display_name": r,
                        }
                    if cls not in by_id[nid]["classes"]:
                        by_id[nid]["classes"].append(cls)
                    if subset_name not in by_id[nid]["subsets"]:
                        by_id[nid]["subsets"].append(subset_name)

    networks = sorted(
        by_id.values(), key=lambda n: (n["classes"][0], n["experiment"], n["recipe_name"])
    )
    return networks, dict(class_subsets)


# ---- models ----------------------------------------------------------------

# Class order for parsing model slugs (longest first to avoid prefix collisions).
_MODEL_CLASS_ORDER = sorted(FINE_CLASSES, key=len, reverse=True)


def _parse_scope(slug: str) -> list[str]:
    remaining = slug
    out: list[str] = []
    while remaining:
        for c in _MODEL_CLASS_ORDER:
            if remaining.startswith(c):
                out.append(c)
                remaining = remaining[len(c) :]
                break
        else:
            remaining = remaining[1:]
    # Canonical order
    return [c for c in FINE_CLASSES if c in out]


def enumerate_models(models_dir: Path) -> list[dict]:
    """Light-weight model enumeration: filename slug → scope, no pickle load."""
    out: list[dict] = []
    for f in sorted(models_dir.glob("*.bestmodel.pickle")):
        slug = f.name[: -len(".bestmodel.pickle")]
        out.append(
            {
                "slug": slug,
                "scope": _parse_scope(slug),
                "asset": str(f.relative_to(models_dir.parent)),
            }
        )
    return out


# ---- predictions + heatmap from metrics.csv -------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def load_metrics_rows(metrics_csv: Path) -> list[dict]:
    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in METRIC_COLUMNS:
            r[k] = _safe_float(r.get(k))
    return rows


def build_predictions(
    rows: list[dict],
    network_lookup: dict[tuple[str, str], str],
    asset_subdir: str = "assets/predictions",
) -> list[dict]:
    """Predictions joined to network ids. `condition` field = model slug."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        xp = r.get("experiment") or ""
        stem = r.get("file_stem") or ""
        nid = network_lookup.get((xp, stem))
        if nid is None:
            continue  # network not in topology_classes corpus — skip
        model = r.get("condition") or ""
        key = (model, nid)
        if key in seen:
            continue
        seen.add(key)
        metrics = {k: r[k] for k in METRIC_COLUMNS if r[k] is not None}
        out.append(
            {
                "model": model,
                "network": nid,
                "model_signature": r.get("model_signature") or "",
                "loss_type": r.get("loss_type") or "",
                "metrics": metrics,
                "asset": f"{asset_subdir}/{model}/{nid}.png",
                "pdf_source": r.get("pdf_path") or "",
            }
        )
    return out


def build_heatmap(
    rows: list[dict],
    models: list[dict],
    metric: str = "grid_nrmse",
    loss_filter: str = "regression",
) -> dict:
    """Pivot metrics.csv into (model condition × test fine_class) grid.

    Aggregation: median over networks within each (condition, fine_class) cell.
    Cell value also tracks the n (network count) feeding the median.
    """
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if loss_filter and not (r.get("loss_type") or "").startswith(loss_filter):
            continue
        cond = r.get("condition") or ""
        cls = r.get("fine_topo_class") or ""
        val = r.get(metric)
        if val is None or not cond or cls not in FINE_CLASSES:
            continue
        cells[(cond, cls)].append(float(val))

    rows_out = sorted({m["slug"] for m in models}, key=lambda c: (len(c), c))
    cols = FINE_CLASSES
    cell_list = []
    for cond in rows_out:
        for cls in cols:
            vals = cells.get((cond, cls), [])
            if not vals:
                continue
            srt = sorted(vals)
            median = (
                srt[len(srt) // 2]
                if len(srt) % 2
                else 0.5 * (srt[len(srt) // 2 - 1] + srt[len(srt) // 2])
            )
            cell_list.append(
                {
                    "row": cond,
                    "col": cls,
                    "value": median,
                    "n": len(vals),
                }
            )

    return {
        "rows": rows_out,
        "cols": cols,
        "metric": metric,
        "cells": cell_list,
    }


# ---- design + tours --------------------------------------------------------


def build_design(selected_runs_root: Path) -> dict:
    """Enumerate the three MIT-logo designs and their on-disk artifacts."""
    out: dict[str, Any] = {}
    val_root = selected_runs_root / "validation_plots"
    val_files: dict[str, Path] = {}
    if val_root.is_dir():
        for sub in val_root.iterdir():
            if sub.is_dir():
                for pdf in sub.glob("*.pdf"):
                    name = pdf.name
                    if "_I_" in name:
                        val_files["mit_i"] = pdf
                    elif "_T_" in name:
                        val_files["mit_t"] = pdf
                    elif "_M_" in name or "_m_" in name:
                        val_files["mit_m"] = pdf

    for key in DESIGN_KEYS:
        run_dir = selected_runs_root / key
        if not run_dir.is_dir():
            continue
        summary_pdf = next(iter(run_dir.glob("*_summary.pdf")), None)
        recipe_yaml = next(iter(run_dir.glob("*.yaml")), None)
        assets: dict[str, str] = {}
        for png_name in ("circuit", "network", "prediction"):
            p = run_dir / f"{png_name}.png"
            if p.exists():
                assets[png_name] = f"assets/design/{key}/{png_name}.png"
        val = val_files.get(key)
        out[key] = {
            "key": key,
            "label": {"mit_i": "MIT I", "mit_t": "MIT T", "mit_m": "MIT M"}[key],
            "source": {
                "summary_pdf": str(summary_pdf) if summary_pdf else None,
                "recipe_yaml": str(recipe_yaml) if recipe_yaml else None,
                "validation_pdf": str(val) if val else None,
                "run_dir": str(run_dir),
            },
            "assets": assets,
        }
    return out


def load_tours(tours_dir: Path) -> dict:
    out: dict[str, Any] = {}
    if not tours_dir.is_dir():
        return out
    for f in sorted(tours_dir.glob("*.yaml")):
        data = _load_yaml_lax(f)
        if data:
            out[f.stem] = data
    return out


# ---- assembly --------------------------------------------------------------


def build_index(
    *,
    paper_jobs_root: Path,
    metrics_csv: Path,
    archive_build_root: Path,
) -> dict:
    topo_root = paper_jobs_root / "data" / "topology_classes"
    models_dir = paper_jobs_root / "models" / "generalization"
    selected_runs = paper_jobs_root / "design" / "selected_runs"
    tours_dir = paper_jobs_root / "archive" / "tours"

    networks, class_subsets = parse_topology_classes(topo_root)
    models = enumerate_models(models_dir)
    # Mark networks with their asset paths (filled in by render fanout).
    for n in networks:
        nid = n["id"]
        n["assets"] = {
            "circuit": f"assets/networks/{nid}/circuit.png",
            "schematic": f"assets/networks/{nid}/schematic.png",
            "data": f"assets/networks/{nid}/data.png",
        }

    # Lookup for predictions join: (experiment, file_stem) -> network id.
    lookup = {(n["experiment"], n["recipe_name"]): n["id"] for n in networks}

    rows = load_metrics_rows(metrics_csv) if metrics_csv.exists() else []
    predictions = build_predictions(rows, lookup)
    heatmap = build_heatmap(rows, models)

    design = build_design(selected_runs)
    tours = load_tours(tours_dir)

    return {
        "meta": {
            "paper_title": "Biomorphic Neural Networks for Neuromorphic Synthetic Biology",
            "build_date": date.today().isoformat(),
            "n_networks": len(networks),
            "n_models": len(models),
            "n_predictions": len(predictions),
        },
        "topology_classes": {
            c: {
                "label": CLASS_LABELS[c],
                "color": CLASS_COLORS[c],
                "subsets": class_subsets.get(c, []),
            }
            for c in FINE_CLASSES
        },
        "networks": networks,
        "models": models,
        "predictions": predictions,
        "heatmap": heatmap,
        "design": design,
        "tours": tours,
    }


@dracon_program(
    name="biocomp-archive-extract",
    version="0.1",
    description="Build paper-archive index.json from paper-jobs source tree.",
)
class ArchiveExtractJob(BaseModel):
    paper_jobs_root: Annotated[
        str,
        Arg(help="Root of paper-jobs/ (auto-detected from this file's location if empty)"),
    ] = ""
    metrics_csv: Annotated[
        str,
        Arg(help="Path to analysis/generalization/data/metrics.csv"),
    ] = ""
    output: Annotated[
        str,
        Arg(short="o", help="Output index.json path"),
    ] = ""

    model_config = {"arbitrary_types_allowed": True}

    def run(self) -> dict:
        # Auto-detect paper-jobs root if not given: walk up from this file.
        if self.paper_jobs_root:
            pj = Path(self.paper_jobs_root).expanduser().resolve()
        else:
            here = Path(__file__).resolve()
            # biocomp-tools/biocomptools/archive/extract.py -> repo root -> paper-jobs/
            pj = here.parents[3] / "paper-jobs"
        assert pj.is_dir(), f"paper-jobs root not a directory: {pj}"

        mcsv = (
            Path(self.metrics_csv).expanduser().resolve()
            if self.metrics_csv
            else pj / "analysis" / "generalization" / "data" / "metrics.csv"
        )

        if self.output:
            out = Path(self.output).expanduser().resolve()
        else:
            import os

            root_env = os.environ.get("BIOCOMP_ROOT")
            base = Path(root_env) if root_env else pj.parent
            out = base / "archive_build" / "data" / "index.json"

        idx = build_index(
            paper_jobs_root=pj, metrics_csv=mcsv, archive_build_root=out.parent.parent
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(idx, indent=2, sort_keys=False))

        summary = {
            "networks": len(idx["networks"]),
            "models": len(idx["models"]),
            "predictions": len(idx["predictions"]),
            "heatmap_cells": len(idx["heatmap"]["cells"]),
            "design_keys": list(idx["design"].keys()),
            "tours": list(idx["tours"].keys()),
            "output": str(out),
        }
        print(json.dumps(summary, indent=2))
        return summary


def main():
    ArchiveExtractJob.cli()


if __name__ == "__main__":
    main()
