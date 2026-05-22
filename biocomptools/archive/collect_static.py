# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Collect pre-existing static assets into the archive build tree.

Handles three asset categories that don't require model inference:
  - design/<key>/: copy existing PNGs (mit_m has them) + pdf->png for summary
  - design/<key>/validation.png: from validation_plots/*.pdf (pdftoppm)
  - analysis/: copy SVG files from analysis/generalization/figures_out/

Idempotent: skip if target exists and is newer than source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _stale(src: Path, dst: Path) -> bool:
    return not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def copy_if_stale(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if not _stale(src, dst):
        return False
    _ensure_parent(dst)
    shutil.copy2(src, dst)
    return True


def pdf_to_png(src: Path, dst: Path, dpi: int = 150) -> bool:
    """Convert first page of PDF to PNG via pdftoppm."""
    if not src.exists():
        return False
    if not _stale(src, dst):
        return False
    _ensure_parent(dst)
    # pdftoppm writes <prefix>-1.png ... we ask for a single page (first only)
    prefix = dst.parent / dst.stem
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", str(src), str(prefix)],
        check=True,
        capture_output=True,
    )
    # pdftoppm appends -1 (or -01) suffix depending on version; find and rename
    for candidate in (
        prefix.with_name(prefix.name + "-1.png"),
        prefix.with_name(prefix.name + "-01.png"),
    ):
        if candidate.exists():
            candidate.rename(dst)
            return True
    return False


# ---- design ----------------------------------------------------------------

DESIGN_VALIDATION_PATTERNS = {
    "mit_i": "_I_",
    "mit_t": "_T_",
    "mit_m": ["_M_", "_m_"],
}


def collect_design(paper_jobs: Path, build: Path) -> dict:
    out_dir_root = build / "assets" / "design"
    selected_runs = paper_jobs / "design" / "selected_runs"
    val_root = selected_runs / "validation_plots"
    n_copied = 0
    n_converted = 0
    for key in ("mit_i", "mit_t", "mit_m"):
        run_dir = selected_runs / key
        if not run_dir.is_dir():
            continue
        out_dir = out_dir_root / key

        # Existing PNGs (only mit_m has them)
        for name in ("circuit", "network", "prediction"):
            src = run_dir / f"{name}.png"
            if copy_if_stale(src, out_dir / f"{name}.png"):
                n_copied += 1

        # Summary PDF -> single composite PNG (for mit_i / mit_t which lack per-panel PNGs)
        summary = next(iter(run_dir.glob("*_summary.pdf")), None)
        if summary:
            if pdf_to_png(summary, out_dir / "summary.png"):
                n_converted += 1

        # Validation PDF -> PNG
        patterns = DESIGN_VALIDATION_PATTERNS[key]
        patterns = patterns if isinstance(patterns, list) else [patterns]
        if val_root.is_dir():
            for sub in val_root.iterdir():
                if not sub.is_dir():
                    continue
                for pdf in sub.glob("*.pdf"):
                    if any(p in pdf.name for p in patterns):
                        if pdf_to_png(pdf, out_dir / "validation.png", dpi=150):
                            n_converted += 1
                        break
    return {"design_copied": n_copied, "design_converted": n_converted}


# ---- analysis --------------------------------------------------------------


def collect_analysis(paper_jobs: Path, build: Path) -> dict:
    """Copy SVG/PNG figures + convert any PDF lacking a sibling SVG to PNG."""
    src_dir = paper_jobs / "analysis" / "generalization" / "figures_out"
    dst_dir = build / "assets" / "analysis"
    if not src_dir.is_dir():
        return {"analysis_copied": 0, "analysis_converted": 0}
    n_copy = 0
    n_conv = 0
    have_vector: set[str] = set()
    # First pass: copy SVG/PNG, remember stems.
    for f in src_dir.iterdir():
        if f.suffix.lower() in (".svg", ".png"):
            if copy_if_stale(f, dst_dir / f.name):
                n_copy += 1
            have_vector.add(f.stem)
    # Second pass: for PDFs whose stem has no vector sibling, convert.
    for f in src_dir.iterdir():
        if f.suffix.lower() != ".pdf":
            continue
        if f.stem in have_vector:
            continue
        if pdf_to_png(f, dst_dir / f"{f.stem}.png", dpi=150):
            n_conv += 1
    return {"analysis_copied": n_copy, "analysis_converted": n_conv}


# ---- main ------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paper-jobs", type=Path, default=None, help="paper-jobs/ root (auto-detected if omitted)"
    )
    p.add_argument(
        "--build",
        type=Path,
        default=None,
        help="archive build dir (default: $BIOCOMP_ROOT/archive_build)",
    )
    p.add_argument("--only", choices=["design", "analysis"], help="run only one collector")
    args = p.parse_args()

    if args.paper_jobs:
        pj = args.paper_jobs.expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        pj = here.parents[3] / "paper-jobs"

    if args.build:
        build = args.build.expanduser().resolve()
    else:
        root = os.environ.get("BIOCOMP_ROOT")
        if not root:
            raise SystemExit("BIOCOMP_ROOT not set; pass --build explicitly.")
        build = Path(root) / "archive_build"

    results: dict = {}
    if args.only in (None, "design"):
        results.update(collect_design(pj, build))
    if args.only in (None, "analysis"):
        results.update(collect_analysis(pj, build))
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
