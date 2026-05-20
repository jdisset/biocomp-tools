# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
import numpy as np

from biocomp.plotutils import PlotData
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def _strip_latex(s: str) -> str:
    return s.replace("$", "")


def _is_latent(vals: np.ndarray, tol: float = 0.05) -> bool:
    return float(vals.min()) >= -tol and float(vals.max()) <= 1.0 + tol


def _nice_ticks(data_vals: np.ndarray, rescaler=None) -> tuple[list[float], list[str]]:
    from biocomp.plotting.plotting_core import format_powers, powers_of_ten

    vmin, vmax = float(data_vals.min()), float(data_vals.max())

    if rescaler is not None and _is_latent(data_vals):
        from biocomp.plotting.plotting_core import get_transformed_ticks_and_labels

        _, labels = get_transformed_ticks_and_labels([vmin, vmax], rescaler)
        return [float(p) for p, _ in labels], [_strip_latex(lbl) for _, lbl in labels]

    raw_min = max(vmin, 1.0)
    p10 = powers_of_ten(xmin=raw_min, xmax=vmax)
    return [float(v) for v in p10], [_strip_latex(format_powers(v)) for v in p10]


def _to_b64(arr: np.ndarray, dtype: np.dtype | type = np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _resolve_lims(spec: tuple | list | None, col: np.ndarray) -> tuple[float, float]:
    lo = float(col.min()) if spec is None or spec[0] is None else float(spec[0])
    hi = float(col.max()) if spec is None or spec[1] is None else float(spec[1])
    return lo, hi


def render_scatter3d_html(
    data: PlotData,
    output_path: str | Path,
    ax=None,
    *,
    title: bool | str | None = None,
    marker_size: int = 3,
    colorscale: str = "YlGnBu",
    width: int = 1000,
    height: int = 800,
    rescaler=None,
    xlims=None,
    ylims=None,
    zlims=None,
    vlims=None,
    **_kw,
):
    x_raw, y_raw = data.x, data.y
    input_names = data.input_names
    output_name = data.output_name if isinstance(data.output_name, str) else data.output_name[0]

    assert x_raw.shape[1] >= 3, f"Need >= 3 input dims for 3D scatter, got {x_raw.shape[1]}"

    if rescaler is not None:
        x = np.asarray(rescaler.fwd(x_raw), dtype=np.float32)
        y = np.asarray(rescaler.fwd(y_raw), dtype=np.float32)
    else:
        x = np.asarray(x_raw, dtype=np.float32)
        y = np.asarray(y_raw, dtype=np.float32)

    yflat = y[:, 0] if y.ndim > 1 else y

    xlo, xhi = _resolve_lims(xlims, x[:, 0])
    ylo, yhi = _resolve_lims(ylims, x[:, 1])
    zlo, zhi = _resolve_lims(zlims, x[:, 2])
    vlo, vhi = _resolve_lims(vlims, yflat)

    mask = (
        (x[:, 0] >= xlo)
        & (x[:, 0] <= xhi)
        & (x[:, 1] >= ylo)
        & (x[:, 1] <= yhi)
        & (x[:, 2] >= zlo)
        & (x[:, 2] <= zhi)
    )
    x, yflat = x[mask], yflat[mask]

    order = np.random.permutation(len(x))
    x, yflat = x[order], yflat[order]

    lo_arr = np.array([xlo, ylo, zlo], dtype=np.float32)
    ranges = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=np.float32)
    ranges[ranges == 0] = 1.0
    pos_norm = 2.0 * (x[:, :3] - lo_arr) / ranges - 1.0
    pos_i16 = np.round(pos_norm * 32767).clip(-32768, 32767).astype(np.int16)

    cmap = matplotlib.colormaps[colorscale]
    vrange = vhi - vlo if vhi > vlo else 1.0
    ynorm = np.clip((yflat - vlo) / vrange, 0, 1)
    colors_u8 = (cmap(ynorm)[:, :3] * 255).clip(0, 255).astype(np.uint8)

    cb_rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).clip(0, 255).astype(np.uint8)

    cb_tv, cb_tt = _nice_ticks(np.array([vlo, vhi]), rescaler)
    cb_ticks = [
        {"frac": float(np.clip((v - vlo) / vrange, 0, 1)), "label": lbl}
        for v, lbl in zip(cb_tv, cb_tt)
        if vlo <= v <= vhi
    ]

    axis_names = input_names[:3] if len(input_names) >= 3 else [f"x{i}" for i in range(3)]
    all_lims = [(xlo, xhi, ranges[0]), (ylo, yhi, ranges[1]), (zlo, zhi, ranges[2])]
    axes = []
    for i, (lo, hi, rng) in enumerate(all_lims):
        tv, tt = _nice_ticks(np.array([lo, hi]), rescaler)
        ticks = [
            {"pos": float(np.clip(2.0 * (v - lo) / float(rng) - 1.0, -1.0, 1.0)), "label": lbl}
            for v, lbl in zip(tv, tt)
            if lo <= v <= hi
        ]
        axes.append({"name": _strip_latex(axis_names[i]), "ticks": ticks})

    title_text = ""
    if title is True:
        network_name = data.metadata.get("network_name", "")
        title_text = f"{network_name} \u2014 {output_name} smoothed mean"
    elif isinstance(title, str):
        title_text = title

    config = json.dumps(
        {
            "positions_b64": _to_b64(pos_i16, np.int16),
            "colors_b64": _to_b64(colors_u8, np.uint8),
            "cb_colors_b64": _to_b64(cb_rgb, np.uint8),
            "point_size": marker_size,
            "axes": axes,
            "title": title_text,
            "output_name": _strip_latex(output_name),
            "cb_ticks": cb_ticks,
        }
    ).replace("</", "<\\/")

    html = _HTML_TEMPLATE.replace('"__DATA__"', config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info(f"Interactive 3D scatter saved to {out}")

    if ax is not None:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"Interactive 3D plot saved to:\n{out.name}",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
            wrap=True,
        )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* { margin: 0; padding: 0; box-sizing: border-box }
body { overflow: hidden; background: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif }
#container { width: 100vw; height: 100vh; position: relative }
#title { position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  font-size: 15px; color: #222; z-index: 10; pointer-events: none }
#colorbar { position: absolute; right: 24px; top: 50%; transform: translateY(-50%);
  display: flex; align-items: stretch; z-index: 10; pointer-events: none }
.cb-title { writing-mode: vertical-rl; transform: rotate(180deg);
  font-size: 12px; color: #333; display: flex; align-items: center;
  justify-content: center; margin-right: 6px }
.cb-track { position: relative }
.cb-track canvas { display: block; border: 1px solid #bbb }
.cb-tick { position: absolute; right: -4px; transform: translate(100%, -50%);
  font-size: 11px; color: #333; white-space: nowrap; padding-left: 6px }
.axis-label { color: #444; font-size: 11px; pointer-events: none; white-space: nowrap }
.axis-name { color: #222; font-size: 13px; font-weight: 600;
  pointer-events: none; white-space: nowrap }
#hint { position: absolute; bottom: 10px; left: 12px; font-size: 11px;
  color: #999; z-index: 10; pointer-events: none }
#crosshair { display: none; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); color: #666; font-size: 20px; font-weight: 300;
  z-index: 10; pointer-events: none; line-height: 1 }
</style>
<script type="importmap">
{"imports":{
  "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
}}
</script>
</head>
<body>
<div id="container"></div>
<div id="title"></div>
<div id="colorbar">
  <div class="cb-title"></div>
  <div class="cb-track"><canvas id="cb-canvas" width="20" height="200"></canvas></div>
</div>
<div id="crosshair">+</div>
<div id="hint">Orbit: drag &middot; Zoom: scroll &middot; F: FPS mode</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const D = "__DATA__";

function decodeB64(b64) {
  const bin = atob(b64), buf = new ArrayBuffer(bin.length), v = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) v[i] = bin.charCodeAt(i);
  return buf;
}

const positions = new Int16Array(decodeB64(D.positions_b64));
const colors = new Uint8Array(decodeB64(D.colors_b64));
const cbColors = new Uint8Array(decodeB64(D.cb_colors_b64));

if (D.title) document.getElementById('title').textContent = D.title;

/* ---- scene ---- */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
camera.position.set(2.8, 2.2, 2.8);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
const ctr = document.getElementById('container');
ctr.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
Object.assign(labelRenderer.domElement.style,
  { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
ctr.appendChild(labelRenderer.domElement);

/* ---- controls: orbit (default) + FPS (press F) ---- */
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;

const fps = new PointerLockControls(camera, document.body);
const hintEl = document.getElementById('hint');
const crossEl = document.getElementById('crosshair');

fps.addEventListener('lock', () => {
  orbit.enabled = false;
  crossEl.style.display = 'block';
  hintEl.textContent = 'WASD: move \u00b7 Q/E: down/up \u00b7 Mouse: look \u00b7 ESC: orbit mode';
});
fps.addEventListener('unlock', () => {
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  orbit.target.copy(camera.position).add(dir.multiplyScalar(2));
  orbit.enabled = true;
  crossEl.style.display = 'none';
  hintEl.textContent = 'Orbit: drag \u00b7 Zoom: scroll \u00b7 F: FPS mode';
});

const mv = { f: 0, b: 0, l: 0, r: 0, u: 0, d: 0 };
addEventListener('keydown', e => {
  if (e.code === 'KeyF' && !fps.isLocked) { fps.lock(); return; }
  if (!fps.isLocked) return;
  switch (e.code) {
    case 'KeyW': mv.f = 1; break; case 'KeyS': mv.b = 1; break;
    case 'KeyA': mv.l = 1; break; case 'KeyD': mv.r = 1; break;
    case 'KeyE': mv.u = 1; break; case 'KeyQ': mv.d = 1; break;
  }
});
addEventListener('keyup', e => {
  switch (e.code) {
    case 'KeyW': mv.f = 0; break; case 'KeyS': mv.b = 0; break;
    case 'KeyA': mv.l = 0; break; case 'KeyD': mv.r = 0; break;
    case 'KeyE': mv.u = 0; break; case 'KeyQ': mv.d = 0; break;
  }
});

/* ---- circle sprite for round points ---- */
const _c = document.createElement('canvas'); _c.width = _c.height = 64;
const _x = _c.getContext('2d');
_x.beginPath(); _x.arc(32, 32, 31, 0, Math.PI * 2); _x.fillStyle = '#fff'; _x.fill();
const circleTex = new THREE.CanvasTexture(_c);

/* ---- point cloud ---- */
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(positions, 3, true));
geom.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
const mat = new THREE.PointsMaterial({
  size: D.point_size, sizeAttenuation: false, vertexColors: true,
  map: circleTex, alphaTest: 0.5
});
scene.add(new THREE.Points(geom, mat));

/* ---- wireframe cube ---- */
scene.add(new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
  new THREE.LineBasicMaterial({ color: 0xdddddd })
));

/* ---- axes ---- */
const axMat = new THREE.LineBasicMaterial({ color: 0x333333 });
/* x-axis: horizontal right-growing, y-axis: vertical up-growing, z-axis: depth */
const origins = [[[-1,-1,-1],[1,-1,-1]], [[-1,-1,-1],[-1,1,-1]], [[-1,-1,-1],[-1,-1,1]]];
const tickDir = [[0, -0.06, 0], [0, 0, -0.06], [0, -0.06, 0]];
const lblOff  = [[0, -0.14, 0], [0, 0, -0.14], [0, -0.14, 0]];
const nameOff = [[0, -0.3,  0], [0, 0, -0.3 ], [0, -0.3,  0]];

function mkLbl(text, cls, pos) {
  const d = document.createElement('div'); d.className = cls; d.textContent = text;
  const o = new CSS2DObject(d); o.position.set(...pos); return o;
}

for (let ai = 0; ai < 3; ai++) {
  const [s, e] = origins[ai];
  scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(...s), new THREE.Vector3(...e)]),
    axMat));

  for (const tk of D.axes[ai].ticks) {
    const p = [...s]; p[ai] = tk.pos;
    const p2 = [p[0]+tickDir[ai][0], p[1]+tickDir[ai][1], p[2]+tickDir[ai][2]];
    scene.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(...p), new THREE.Vector3(...p2)]),
      axMat));
    scene.add(mkLbl(tk.label, 'axis-label',
      [p[0]+lblOff[ai][0], p[1]+lblOff[ai][1], p[2]+lblOff[ai][2]]));
  }

  const mid = [(s[0]+e[0])/2, (s[1]+e[1])/2, (s[2]+e[2])/2];
  scene.add(mkLbl(D.axes[ai].name, 'axis-name',
    [mid[0]+nameOff[ai][0], mid[1]+nameOff[ai][1], mid[2]+nameOff[ai][2]]));
}

/* ---- colorbar ---- */
document.querySelector('.cb-title').textContent = D.output_name;
const cbCanvas = document.getElementById('cb-canvas');
const ctx = cbCanvas.getContext('2d');
const cbH = cbCanvas.height;
for (let i = 0; i < cbH; i++) {
  const ci = Math.floor((1 - i / cbH) * (cbColors.length / 3 - 1)) * 3;
  ctx.fillStyle = `rgb(${cbColors[ci]},${cbColors[ci+1]},${cbColors[ci+2]})`;
  ctx.fillRect(0, i, 20, 1);
}
const track = document.querySelector('.cb-track');
for (const tk of D.cb_ticks) {
  const d = document.createElement('div'); d.className = 'cb-tick'; d.textContent = tk.label;
  d.style.top = ((1 - tk.frac) * cbH) + 'px';
  track.appendChild(d);
}

/* ---- animate ---- */
const camDir = new THREE.Vector3();
const camRight = new THREE.Vector3();
const worldUp = new THREE.Vector3(0, 1, 0);
const flySpeed = 0.04;

(function animate() {
  requestAnimationFrame(animate);
  if (fps.isLocked) {
    const fwd = mv.f - mv.b, lr = mv.r - mv.l, ud = mv.u - mv.d;
    if (fwd || lr || ud) {
      camera.getWorldDirection(camDir);
      camRight.crossVectors(camDir, worldUp).normalize();
      camera.position.addScaledVector(camDir, fwd * flySpeed);
      camera.position.addScaledVector(camRight, lr * flySpeed);
      camera.position.y += ud * flySpeed;
    }
  } else {
    orbit.update();
  }
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
})();

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""
