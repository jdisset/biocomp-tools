# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Annotated, Iterator

import pikepdf
from dracon import Arg, dracon_program
from pydantic import BaseModel

METRIC_COLUMNS: list[str] = [
    "grid_nrmse",
    "grid_rmse",
    "grid_mse",
    "grid_r_squared",
    "grid_snr",
    "grid_kl",
    "grid_kl_similarity",
    "kratio",
    "ratio_rmse",
    "ratio_r_squared",
    "model_rmse_latent",
    "kernel_rmse_latent",
    "excess_rmse_latent",
    "bias_mag_latent",
    "model_r_squared_latent",
    "kernel_r_squared_latent",
    "model_nrmse_local",
    "kernel_nrmse_local",
    "mse",
    "rmse",
    "samples",
    "eval_npoints",
]

ID_COLUMNS: list[str] = [
    "basic_set",
    "fine_topo_class",
    "condition",
    "network_name",
    "file_stem",
    "cell_type",
    "experiment",
    "model_signature",
    "replicate",
    "base_config",
    "loss_type",
    "pdf_path",
]


_FINE_CLASS_RE = re.compile(r"^(Lbl|Lbr|Lc|N|S|M)")


def _parse_basic_set(name: str) -> str:
    m = _FINE_CLASS_RE.match(name)
    return m.group(1) if m else name.split("_", 1)[0]


def _experiment_from_datafile(datafile: dict | None) -> str | None:
    if not datafile:
        return None
    fpath = datafile.get("file") or ""
    parts = fpath.split("/")
    if "Experiments" in parts:
        idx = parts.index("Experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _read_pdf_metadata(pdf_path: Path) -> dict | None:
    try:
        with pikepdf.open(str(pdf_path)) as p:
            subj = p.docinfo.get("/Subject", "")
            raw = str(subj)
            if not raw:
                return None
            return json.loads(raw)
    except (pikepdf.PdfError, ValueError, json.JSONDecodeError):
        return None


def _row_from_pdf(pdf_path: Path, predictions_root: Path) -> dict | None:
    """Decode one prediction PDF into a flat dict, or None on failure."""
    raw = _read_pdf_metadata(pdf_path)
    if raw is None:
        return None
    fm = raw.get("FigureMetadata", raw)

    rel = pdf_path.relative_to(predictions_root)
    parts = rel.parts
    if len(parts) < 6:
        return None
    basic_set = parts[0]
    condition = parts[1]

    model = fm.get("model") or {}
    data = fm.get("data") or {}
    metrics = fm.get("metrics") or {}
    base_config = model.get("base_config") or ""
    loss_type = base_config.split("-", 1)[0] if base_config else ""

    row = {
        "basic_set": basic_set,
        "fine_topo_class": _parse_basic_set(basic_set),
        "condition": condition,
        "network_name": data.get("network_name", ""),
        "file_stem": data.get("file_stem", ""),
        "cell_type": data.get("cell_type", ""),
        "experiment": _experiment_from_datafile(data.get("datafile")) or "",
        "model_signature": model.get("signature", ""),
        "replicate": model.get("replicate", ""),
        "base_config": base_config,
        "loss_type": loss_type,
        "pdf_path": str(pdf_path),
    }
    for k in METRIC_COLUMNS:
        row[k] = metrics.get(k)
    _backfill_excess(row)
    return row


def _backfill_excess(row: dict) -> None:
    """Derive excess_rmse_latent / bias_mag_latent for PDFs that predate them."""
    m, k = row.get("model_rmse_latent"), row.get("kernel_rmse_latent")
    if row.get("excess_rmse_latent") is None and m is not None and k is not None:
        row["excess_rmse_latent"] = m - k
    if row.get("bias_mag_latent") is None and m is not None and k is not None:
        diff = m * m - k * k
        row["bias_mag_latent"] = math.sqrt(diff) if diff > 0 else 0.0


def iter_prediction_pdfs(predictions_root: Path) -> Iterator[Path]:
    yield from sorted(predictions_root.rglob("*.pdf"))


def extract_metrics(predictions_root: Path) -> list[dict]:
    return [
        r for pdf in iter_prediction_pdfs(predictions_root)
        if (r := _row_from_pdf(pdf, predictions_root)) is not None
    ]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ID_COLUMNS + METRIC_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@dracon_program(
    name="extract-generalization-metrics",
    version="0.1",
    description="Walk a prediction-PDF tree and extract embedded metrics into CSV.",
)
class MetricsExtractJob(BaseModel):
    predictions_root: Annotated[
        str,
        Arg(short="i", help="Root of the prediction PDF tree (`.../predictions/`)"),
    ] = ""
    output: Annotated[
        str,
        Arg(short="o", help="Output CSV path"),
    ] = "./metrics.csv"
    quiet: Annotated[bool, Arg(help="Suppress per-file progress")] = False

    model_config = {"arbitrary_types_allowed": True}

    def run(self) -> dict:
        root = Path(self.predictions_root).expanduser().resolve()
        assert root.is_dir(), f"predictions_root not a directory: {root}"
        out = Path(self.output).expanduser().resolve()

        rows: list[dict] = []
        n_pdf = 0
        n_ok = 0
        for pdf in iter_prediction_pdfs(root):
            n_pdf += 1
            row = _row_from_pdf(pdf, root)
            if row is None:
                if not self.quiet:
                    print(f"  skip (no metadata): {pdf.name}")
                continue
            rows.append(row)
            n_ok += 1
            if not self.quiet and n_ok % 50 == 0:
                print(f"  [{n_ok}/{n_pdf}]")

        write_csv(rows, out)
        summary = {"pdfs_seen": n_pdf, "rows_written": n_ok, "csv_path": str(out)}
        print(f"Wrote {n_ok}/{n_pdf} rows -> {out}")
        return summary


def main():
    MetricsExtractJob.cli()


if __name__ == "__main__":
    main()
