# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class _FigaxStub:
    def __init__(self, fig):
        self.figure = fig
        self.flat_ax: list = []


def find_default_pictogram_dir(subdir: str = "") -> str:
    if env := os.environ.get("BIOCOMP_PICTOGRAM_DIR"):
        return str(Path(env) / subdir) if subdir else env
    for parent in Path(__file__).resolve().parents:
        cand = parent / "biocomp-jobs" / "analysis" / "generalization" / "v2"
        if cand.is_dir():
            return str(cand / subdir) if subdir else str(cand)
    return ""


def truncate_cmap(cmap_name: str, frac: float):
    base = plt.get_cmap(cmap_name)
    lo = (1 - frac) / 2
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc", base(np.linspace(lo, 1 - lo, 256))
    )


def maybe_truncate(cmap_name: str, frac: float):
    return truncate_cmap(cmap_name, frac) if frac < 1.0 else plt.get_cmap(cmap_name)


def load_rgba(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = plt.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    elif img.shape[2] == 3:
        img = np.concatenate([img, np.ones((*img.shape[:2], 1))], axis=-1)
    return img.astype(np.float32)


def load_pictograms(picto_dir: Path, players: list[str]):
    pictos: dict[str, np.ndarray] = {}
    for p in players:
        img = load_rgba(picto_dir / f"{p}.png")
        if img is not None:
            pictos[p] = img
    return pictos, load_rgba(picto_dir / "background.png")


def composite_icon(
    members: list[str],
    pictograms: dict[str, np.ndarray],
    picto_bg: np.ndarray | None,
    invert: bool = False,
) -> np.ndarray | None:
    layers: list[np.ndarray] = []
    if picto_bg is not None:
        layers.append(picto_bg)
    for p in members:
        if p in pictograms:
            layers.append(pictograms[p])
    if not layers:
        return None
    h, w = layers[0].shape[:2]
    result = np.zeros((h, w, 4), dtype=np.float32)
    for layer in layers:
        ly = layer
        if ly.shape[:2] != (h, w):
            from PIL import Image

            ly = (
                np.array(
                    Image.fromarray((ly * 255).astype(np.uint8)).resize(
                        (w, h), Image.LANCZOS
                    )
                )
                / 255.0
            )
        a = ly[:, :, 3:4]
        result[:, :, :3] = result[:, :, :3] * (1 - a) + ly[:, :, :3] * a
        result[:, :, 3:4] = result[:, :, 3:4] * (1 - a) + a
    if invert:
        result = result.copy()
        result[:, :, :3] = 1.0 - result[:, :, :3]
    return result


def fit_zoom(
    img: np.ndarray,
    *,
    max_w_inch: float | None,
    max_h_inch: float | None,
    dpi: float,
    margin: float = 0.9,
) -> float:
    h_px, w_px = img.shape[:2]
    candidates = []
    if max_w_inch is not None:
        candidates.append(max_w_inch * dpi / w_px)
    if max_h_inch is not None:
        candidates.append(max_h_inch * dpi / h_px)
    return min(candidates) * margin if candidates else margin


def draw_heatmap_cells(ax, matrix: np.ndarray, cmap, norm, vector: bool = False):
    if not vector:
        return ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cmap_obj(norm(v)), edgecolor="none",
            ))
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array(matrix)
    return sm
