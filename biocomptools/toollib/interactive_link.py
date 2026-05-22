# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Brush-interactive linked-pixel multi-panel renderer.

Architecture
------------
This is a thin orchestrator on top of the canonical biocomp-plot pipeline.
It does not invent a parallel rendering path - every panel is a regular
`PlotTask` (with its own `plot_method` and any `overlays`). The canonical
`Figure` runs them through `mpl.rc_context` from `default_plotconf_v2`
and dispatches `plot_method` calls with full `callstack_params` styling
inheritance.

The interactive layer's only job is post-render aggregation:

  1. Load the dataset, concat raws across networks into a single PlotData.
  2. Subsample.
  3. Construct the `Figure` (declared in YAML), injecting `plot_data` via
     dracon's deferred-construct context.
  4. Run the figure normally - it writes an SVG.
  5. Walk `figure._ptasks` and harvest each task's `_overlay_results`. The
     selection-grid overlays exposed `pixel_for_raw` arrays + per-pixel
     metadata; we bundle them into a JSON payload.
  6. Wrap the SVG in an HTML shell containing the JSON payload and the
     hover-link JS engine.

Single-source-of-truth: every overlay type lives in `overlays.py` and is
generic (any `PlotTask` can declare overlays). The HTML/JS shell only
needs the panel kind + forward maps; it does not care which `plot_method`
drew the base panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import numpy as np
from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode
from dracon.diagnostics import DraconError, handle_dracon_error
from pydantic import BaseModel, ConfigDict

from biocomp.plotutils import PlotData
from jeanplot.panels import Figure
from biocomptools.logging_config import get_logger, setup_logging
from biocomptools.toollib.datasources import DBSource
from biocomptools.toollib.networkselector import (
    CleanupFilter,
    NetworkDataPair,
    NetworkFilter,
    NetworkSelector,
    NetworkSet,
    NetworkSetDifference,
    NetworkSetIntersection,
    NetworkSetUnion,
    Regex,
    iRegex,
)
from biocomptools.toollib.overlays import OVERLAY_TYPES
from biocomptools.toollib.plot import PlotConfig, PlotTask

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# HTML / JS template
# ----------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body { font-family: -apple-system, sans-serif; background: #fafafa; padding: 1em; margin: 0; }
h1 { font-size: 0.95em; color: #555; margin: 0 0 0.4em 0; font-weight: 500; }
.hint { font-size: 0.78em; color: #888; margin: 0 0 0.8em 0; }
.hint kbd { background: #eee; border: 1px solid #ccc; border-bottom-width: 2px;
            border-radius: 3px; padding: 0 0.4em; font-family: ui-monospace, monospace;
            font-size: 0.95em; color: #333; }
#stats { display: flex; flex-wrap: wrap; gap: 0 1.4em; align-items: baseline;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         font-size: 0.82em; color: #333; padding: 0.5em 0.8em;
         background: #fff; border: 1px solid #e5e5e5; border-radius: 4px;
         margin-bottom: 0.6em; }
#stats .grp { display: flex; gap: 0.4em; align-items: baseline; }
#stats .lab { color: #888; text-transform: uppercase; font-size: 0.72em;
              letter-spacing: 0.04em; }
#stats .val { color: #111; min-width: 2.5em; display: inline-block; }
#stats .src { color: #ff2d2d; font-weight: 600; }
[id^="ov-"] > path { fill-opacity: 0; pointer-events: none; }
[id^="hit-"] { pointer-events: all; cursor: crosshair; }
[id^="hit-"] > path { fill-opacity: 0 !important; stroke-opacity: 0 !important; }
svg { max-width: 100%; height: auto; }
</style>
</head>
<body>
<h1>__TITLE__ — hover any panel: brush selects raws via panel pixels; targets paint by selected fraction.</h1>
<p class="hint">Brush: <kbd>+</kbd> grow, <kbd>-</kbd> shrink, <kbd>0</kbd> reset</p>
<div id="stats">
  <div class="grp"><span class="lab">brush</span><span class="val" id="s-brush">—</span><span class="lab">px</span></div>
  <div class="grp"><span class="lab">source</span><span class="val src" id="s-src">—</span></div>
  <div class="grp"><span class="lab">raws</span><span class="val" id="s-sel">0</span><span class="lab">/</span><span class="val" id="s-total">—</span><span class="lab">(</span><span class="val" id="s-pct">0.0</span><span class="lab">%)</span></div>
  <div class="grp"><span class="lab">selected output (MEF)</span>
    <span class="lab">mean</span><span class="val" id="s-mean">—</span>
    <span class="lab">p10</span><span class="val" id="s-p10">—</span>
    <span class="lab">p50</span><span class="val" id="s-p50">—</span>
    <span class="lab">p90</span><span class="val" id="s-p90">—</span>
  </div>
</div>
<div>__SVG__</div>
<script id="link-data" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("link-data").textContent);
  const panels = data.panels;
  const yRaw = Float32Array.from(data.raw.y_raw);
  const totalN = yRaw.length;
  const initialBrushPx = data.brush_pixels;
  let brushPx = initialBrushPx;
  let brushPx2 = brushPx * brushPx;
  const BRUSH_MIN = 3, BRUSH_MAX = 400, BRUSH_STEP = 1.15;

  const pathCache = {};
  function getPath(name, idx) {
    const k = name + ":" + idx;
    if (pathCache[k] === undefined) {
      const g = document.getElementById("ov-" + name + "-" + idx);
      pathCache[k] = g ? (g.querySelector("path") || g) : null;
    }
    return pathCache[k];
  }

  function setIntensity(name, idx, val) {
    const el = getPath(name, idx);
    if (!el) return;
    el.style.fillOpacity = val > 0 ? String(val) : "0";
  }

  const lit = {};
  for (const k in panels) lit[k] = new Set();

  const distState = {};
  for (const k in panels) {
    if (panels[k].kind !== "output_distribution_1d") continue;
    const hitG = document.getElementById("hit-" + k);
    const hitPath = hitG ? hitG.querySelector("path") : null;
    const curveG = document.getElementById("ov-" + k + "-curve");
    const curvePath = curveG ? curveG.querySelector("path") : null;
    if (!hitPath || !curvePath) continue;
    const nums = (hitPath.getAttribute("d").match(/-?\d+(?:\.\d+)?/g) || []).map(parseFloat);
    if (nums.length < 8) continue;
    const xs = [nums[0], nums[2], nums[4], nums[6]];
    const ys = [nums[1], nums[3], nums[5], nums[7]];
    distState[k] = {
      curvePath,
      vbXmin: Math.min(...xs), vbXmax: Math.max(...xs),
      vbYmin: Math.min(...ys), vbYmax: Math.max(...ys),
    };
  }

  function gaussianSmooth(arr, sigma) {
    const n = arr.length;
    const out = new Float32Array(n);
    const half = Math.max(1, Math.ceil(sigma * 3));
    const ker = new Float32Array(2 * half + 1);
    let kSum = 0;
    for (let i = -half; i <= half; i++) {
      const w = Math.exp(-i * i / (2 * sigma * sigma));
      ker[i + half] = w; kSum += w;
    }
    for (let i = 0; i < ker.length; i++) ker[i] /= kSum;
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let j = -half; j <= half; j++) {
        const idx = i + j;
        if (idx >= 0 && idx < n) v += arr[idx] * ker[j + half];
      }
      out[i] = v;
    }
    return out;
  }

  function paintDistribution(k, counts) {
    const panel = panels[k];
    const st = distState[k];
    if (!st) return;
    const sigma = panel.bandwidth_bins || 1.5;
    const smoothed = gaussianSmooth(counts, sigma);
    let peak = 0;
    for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > peak) peak = smoothed[i];
    if (peak <= 0) {
      st.curvePath.style.fillOpacity = "0";
      return;
    }
    const sxVB = (st.vbXmax - st.vbXmin) / (panel.xmax - panel.xmin);
    const vbHeight = st.vbYmax - st.vbYmin;
    let d = "M " + (st.vbXmin + (panel.pixel_x[0] - panel.xmin) * sxVB).toFixed(2)
          + "," + st.vbYmax.toFixed(2);
    for (let i = 0; i < smoothed.length; i++) {
      const xVB = st.vbXmin + (panel.pixel_x[i] - panel.xmin) * sxVB;
      const yVB = st.vbYmax - (smoothed[i] / peak) * vbHeight;
      d += " L " + xVB.toFixed(2) + "," + yVB.toFixed(2);
    }
    const xLast = panel.pixel_x[panel.pixel_x.length - 1];
    d += " L " + (st.vbXmin + (xLast - panel.xmin) * sxVB).toFixed(2)
       + "," + st.vbYmax.toFixed(2) + " Z";
    st.curvePath.setAttribute("d", d);
    st.curvePath.style.fillOpacity = "0.85";
  }

  function clearDistribution(k) {
    const st = distState[k];
    if (!st) return;
    st.curvePath.style.fillOpacity = "0";
  }

  function clearAll() {
    for (const k in lit) {
      if (panels[k].kind === "output_distribution_1d") {
        clearDistribution(k);
        continue;
      }
      for (const idx of lit[k]) setIntensity(k, idx, 0);
      lit[k].clear();
    }
    resetSelStats();
  }

  const elBrush = document.getElementById("s-brush");
  const elSrc = document.getElementById("s-src");
  const elSel = document.getElementById("s-sel");
  const elTotal = document.getElementById("s-total");
  const elPct = document.getElementById("s-pct");
  const elMean = document.getElementById("s-mean");
  const elP10 = document.getElementById("s-p10");
  const elP50 = document.getElementById("s-p50");
  const elP90 = document.getElementById("s-p90");

  function fmtMEF(v) {
    if (!isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e4 || a < 1e-2) return v.toExponential(2);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  function updateBrushDisplay() { elBrush.textContent = brushPx.toFixed(0); }

  function resetSelStats() {
    elSrc.textContent = "—"; elSel.textContent = "0"; elPct.textContent = "0.0";
    elMean.textContent = "—"; elP10.textContent = "—";
    elP50.textContent = "—"; elP90.textContent = "—";
  }

  function updateSelStats(srcName, selectedY, n) {
    elSrc.textContent = srcName;
    elSel.textContent = String(n);
    elPct.textContent = totalN ? (100 * n / totalN).toFixed(1) : "0.0";
    if (n === 0) {
      elMean.textContent = "—"; elP10.textContent = "—";
      elP50.textContent = "—"; elP90.textContent = "—";
      return;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) sum += selectedY[i];
    const mean = sum / n;
    const sorted = Array.from(selectedY.subarray(0, n)).sort((a, b) => a - b);
    const q = (p) => sorted[Math.min(n - 1, Math.max(0, Math.floor(p * n)))];
    elMean.textContent = fmtMEF(mean);
    elP10.textContent = fmtMEF(q(0.10));
    elP50.textContent = fmtMEF(q(0.50));
    elP90.textContent = fmtMEF(q(0.90));
  }

  const yBuf = new Float32Array(totalN);
  let lastHover = null;

  updateBrushDisplay();
  elTotal.textContent = String(totalN);

  function brushedPixels(srcName, mx, my) {
    const src = panels[srcName];
    const hit = document.getElementById("hit-" + srcName).getBoundingClientRect();
    const sx = hit.width / (src.xmax - src.xmin);
    const sy = hit.height / (src.ymax - src.ymin);
    const xs = src.pixel_x, ys = src.pixel_y, cnts = src.pixel_count;
    const out = new Uint8Array(src.n_pixels);
    const isDist = src.kind === "output_distribution_1d";
    for (let p = 0; p < src.n_pixels; p++) {
      if (cnts[p] === 0) continue;
      const screenX = hit.left + (xs[p] - src.xmin) * sx;
      const dx = screenX - mx;
      let d2;
      if (isDist) {
        d2 = dx * dx;
      } else {
        const screenY = hit.top + (src.ymax - ys[p]) * sy;
        const dy = screenY - my;
        d2 = dx * dx + dy * dy;
      }
      if (d2 <= brushPx2) out[p] = 1;
    }
    return out;
  }

  function onHover(srcName, clientX, clientY) {
    lastHover = { srcName, clientX, clientY };
    const brushed = brushedPixels(srcName, clientX, clientY);
    const counts = {};
    for (const k in panels) counts[k] = new Int32Array(panels[k].n_pixels);
    const pxSrc = data.raw["px_" + srcName];
    const N = pxSrc.length;
    let nSel = 0;
    for (let i = 0; i < N; i++) {
      const p = pxSrc[i];
      if (p < 0 || !brushed[p]) continue;
      yBuf[nSel++] = yRaw[i];
      for (const k in panels) {
        const q = data.raw["px_" + k][i];
        if (q >= 0) counts[k][q]++;
      }
    }
    for (const k in panels) {
      if (panels[k].kind === "output_distribution_1d") {
        paintDistribution(k, counts[k]);
        continue;
      }
      const newLit = new Set();
      const pc = panels[k].pixel_count;
      const c = counts[k];
      for (let p = 0; p < c.length; p++) {
        if (c[p] > 0 && pc[p] > 0) newLit.add(p);
      }
      for (const idx of lit[k]) if (!newLit.has(idx)) setIntensity(k, idx, 0);
      for (const idx of newLit) setIntensity(k, idx, Math.min(1, c[idx] / pc[idx]));
      lit[k] = newLit;
    }
    updateSelStats(srcName, yBuf, nSel);
  }

  function setBrush(newPx) {
    brushPx = Math.max(BRUSH_MIN, Math.min(BRUSH_MAX, newPx));
    brushPx2 = brushPx * brushPx;
    updateBrushDisplay();
    if (lastHover) onHover(lastHover.srcName, lastHover.clientX, lastHover.clientY);
  }

  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (e.key === "+" || e.key === "=") { setBrush(brushPx * BRUSH_STEP); e.preventDefault(); }
    else if (e.key === "-" || e.key === "_") { setBrush(brushPx / BRUSH_STEP); e.preventDefault(); }
    else if (e.key === "0") { setBrush(initialBrushPx); e.preventDefault(); }
  });

  for (const k in panels) {
    const hit = document.getElementById("hit-" + k);
    if (!hit) continue;
    hit.addEventListener("mousemove", e => onHover(k, e.clientX, e.clientY));
    hit.addEventListener("mouseleave", () => { lastHover = null; clearAll(); });
  }
})();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# Job
# ----------------------------------------------------------------------------


_DATASET_TYPES = [
    NetworkSelector, NetworkSet, NetworkDataPair,
    NetworkSetUnion, NetworkSetIntersection, NetworkSetDifference,
    NetworkFilter, CleanupFilter,
    DBSource,
    Regex, iRegex,
]


def _pool_plot_data(plot_data_list, n_subsample: int | None, seed: int) -> PlotData:
    """Concatenate raws across networks, optionally subsample, single PlotData."""
    xs, ys = [], []
    for pd in plot_data_list:
        x = np.asarray(pd.x, dtype=np.float32)
        y = np.asarray(pd.y, dtype=np.float32)
        assert y.shape[1] == 1, f"expected single output column, got {y.shape[1]}"
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    if n_subsample is not None and x_all.shape[0] > n_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_all.shape[0], size=n_subsample, replace=False)
        x_all = x_all[idx]
        y_all = y_all[idx]
    base = plot_data_list[0]
    return PlotData(
        xval=x_all, yval=y_all,
        input_names=list(base.input_names) if base.input_names else [],
        output_name=base.output_name,
    )


@dracon_program(
    name="biocomp-interactive-link",
    description="Brush-interactive linked-pixel multi-panel renderer.",
    context_types=[
        Figure, PlotTask, PlotConfig,
        *OVERLAY_TYPES, *_DATASET_TYPES,
    ],
)
class InteractiveLinkJob(BaseModel):
    """Loads a dataset, renders a multi-panel `Figure` (declared in YAML)
    where each panel may carry overlays, then bundles overlay metadata
    into an interactive HTML wrapper around the canonical SVG output."""

    dataset: Annotated[
        NetworkSet,
        Arg(help="dataset selector (compose via `!include file:...` in YAML)"),
    ]
    figure: Annotated[
        DeferredNode[Figure],
        Arg(help="canonical biocomp-plot Figure with overlays declared in YAML"),
    ]
    output_dir: Annotated[str, Arg(help="output directory")]
    output_stem: Annotated[
        str | None, Arg(help="output filename stem (defaults to dataset.name)"),
    ] = None
    title: Annotated[str, Arg(help="title shown above the figure")] = ""

    n_subsample: Annotated[
        int, Arg(help="cap raw points (random subsample, reproducible). 0 = use all"),
    ] = 50000
    seed: Annotated[int, Arg(help="random seed for subsampling")] = 0
    brush_pixels: Annotated[
        float, Arg(help="brush radius in screen pixels (initial value)"),
    ] = 20.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self):
        plot_data_list = DBSource(content=[self.dataset]).get_data()
        assert plot_data_list, f"empty dataset: {self.dataset.name!r}"
        pd = _pool_plot_data(
            plot_data_list,
            n_subsample=self.n_subsample if self.n_subsample > 0 else None,
            seed=self.seed,
        )
        logger.info(f"pooled raw points: {pd.x.shape[0]} (from {len(plot_data_list)} networks)")

        out_dir = Path(self.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_stem or self.dataset.name
        assert stem, "no output_stem given and dataset has no .name — cannot derive filename"
        title = self.title or stem

        figure: Figure = self.figure.construct(context={"plot_data": pd})
        figure.figure_spec.output_dir = str(out_dir)
        figure.figure_spec.output_file = f"{stem}_interactive.svg"
        figure.run()

        svg_path = Path(figure.figure_spec.output_dir) / figure.figure_spec.output_file

        panels: dict[str, dict] = {}
        for pt in (figure._ptasks or []):
            for ov_meta in pt._overlay_results:
                if ov_meta.get("kind") == "hit_rect":
                    continue
                panels[ov_meta["name"]] = ov_meta

        for name, meta in panels.items():
            n_emit = meta.get("n_emitted_rects", 0)
            logger.info(
                f"panel {name} ({meta['kind']}): {meta['n_pixels']} pixels, "
                f"{n_emit} emitted rects"
            )

        # Build JSON payload — strip pixel_for_raw arrays out into `raw.px_{name}`
        # so the JS never has to walk panel metadata to find them.
        raw_payload: dict[str, list] = {"y_raw": pd.y[:, 0].astype(float).tolist()}
        panels_payload: dict[str, dict] = {}
        for name, meta in panels.items():
            raw_payload[f"px_{name}"] = meta["pixel_for_raw"]
            panels_payload[name] = {k: v for k, v in meta.items() if k != "pixel_for_raw"}

        payload = {
            "raw": raw_payload,
            "panels": panels_payload,
            "brush_pixels": self.brush_pixels,
        }

        svg_text = svg_path.read_text()
        idx = svg_text.find("<svg")
        assert idx >= 0, f"no <svg in {svg_path}"
        svg_inline = svg_text[idx:]

        html = (
            HTML_TEMPLATE
            .replace("__SVG__", svg_inline)
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__TITLE__", title)
        )
        html_path = out_dir / f"{stem}_interactive.html"
        html_path.write_text(html)

        print(f"raws: {pd.x.shape[0]}")
        for name, meta in panels.items():
            print(f"  {name} ({meta['kind']}): "
                  f"{int(sum(c > 0 for c in meta['pixel_count']))} non-empty pixels / {meta['n_pixels']}")
        print(f"SVG:  {svg_path}")
        print(f"HTML: {html_path}")


def main():
    setup_logging()
    try:
        InteractiveLinkJob.cli()
    except DraconError as e:
        handle_dracon_error(e, exit_code=1)
    except Exception as e:
        root = e
        while root.__cause__ is not None:
            root = root.__cause__
        if isinstance(root, DraconError):
            handle_dracon_error(root, exit_code=1)
        raise


if __name__ == "__main__":
    main()
