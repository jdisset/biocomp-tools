# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Convert source prediction PDFs to PNG tiles for the archive.

Strategy: index.json carries `pdf_source` for each prediction (taken from
metrics.csv). We just need to walk that list and pdftoppm each one. This is
1000x faster than re-running the plot jobs from scratch and produces
identical output (since the same PDFs were used to feed metrics.csv).

Same approach also handles per-network ground-truth plots: each prediction
PDF contains a `[circuit | data | prediction | ...]` row, so we crop / use it
as the per-(model,network) tile directly. For per-network only (no model)
plots we'd need to re-run dataset_plot.yaml — deferred to v2.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _stale(src: Path, dst: Path) -> bool:
    return not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


def convert_one(src: Path, dst: Path, dpi: int) -> tuple[Path, bool, str | None]:
    """Convert src PDF to dst PNG via pdftoppm. Returns (dst, did_convert, err)."""
    if not src.exists():
        return (dst, False, f"missing source: {src}")
    if not _stale(src, dst):
        return (dst, False, None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    prefix = dst.parent / dst.stem
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", str(src), str(prefix)],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        return (dst, False, e.stderr.decode("utf-8", errors="replace")[:200])
    except subprocess.TimeoutExpired:
        return (dst, False, "timeout")
    for cand in (
        prefix.with_name(prefix.name + "-1.png"),
        prefix.with_name(prefix.name + "-01.png"),
    ):
        if cand.exists():
            cand.rename(dst)
            return (dst, True, None)
    return (dst, False, "pdftoppm produced no output")


def iter_jobs(
    index: dict, build: Path, only_model: str | None = None, only_network: str | None = None
) -> Iterable[tuple[Path, Path]]:
    for p in index.get("predictions", []):
        if only_model and p.get("model") != only_model:
            continue
        if only_network and p.get("network") != only_network:
            continue
        pdf = p.get("pdf_source")
        if not pdf:
            continue
        yield (Path(pdf), build / p["asset"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--build",
        type=Path,
        default=None,
        help="archive build dir (default: $BIOCOMP_ROOT/archive_build)",
    )
    ap.add_argument("--dpi", type=int, default=80, help="output PNG DPI (80 gives ~80KB/tile)")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4, help="parallel workers")
    ap.add_argument("--model", default=None, help="only this model slug")
    ap.add_argument("--network", default=None, help="only this network id")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N successful conversions (for smoke tests)",
    )
    args = ap.parse_args()

    build = args.build or Path(os.environ.get("BIOCOMP_ROOT", ".")) / "archive_build"
    assert build.is_dir(), f"build dir not a directory: {build}"
    index_path = build / "data" / "index.json"
    assert index_path.exists(), f"missing {index_path} — run biocomp-archive-extract first"
    index = json.loads(index_path.read_text())

    jobs = list(iter_jobs(index, build, args.model, args.network))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"converting {len(jobs)} PDFs at {args.dpi} DPI with {args.jobs} workers")

    n_ok = n_skip = n_err = 0
    errs: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(convert_one, s, d, args.dpi): (s, d) for s, d in jobs}
        for i, fut in enumerate(cf.as_completed(futs)):
            dst, did, err = fut.result()
            if err:
                n_err += 1
                if len(errs) < 5:
                    errs.append(f"{dst.name}: {err}")
            elif did:
                n_ok += 1
            else:
                n_skip += 1
            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{len(jobs)}] ok={n_ok} skip={n_skip} err={n_err}")

    print(f"done: converted={n_ok} skipped={n_skip} errors={n_err}")
    for e in errs:
        print(f"  err: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
