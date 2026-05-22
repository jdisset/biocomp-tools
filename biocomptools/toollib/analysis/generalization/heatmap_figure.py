# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
from pydantic import BaseModel, Field

from ._render import (
    composite_icon,
    draw_heatmap_cells,
    find_default_pictogram_dir,
    fit_zoom,
    load_pictograms,
    maybe_truncate,
)
from .heatmap_math import col_stats, deviation, log_fold_dev
from .views import GenViewConfig, parse_condition

DataMode = Literal["abs", "dev_mean", "dev_best"]
ClassDataMode = Literal["abs", "devmean", "devfull"]


class HeatmapConfig(BaseModel):
    figsize: tuple[float, float] = (15.0, 5.5)
    class_summary_figsize: tuple[float, float] = (7.0, 5.0)
    class_summary_row_inch: float = 0.55
    class_summary_strip_ratio: float = 0.45
    class_summary_bracket_ratio: float = 0.12
    class_summary_heatmap_ratio: float = 1.0
    dpi: int = 300
    height_ratios: list[float] = Field(default_factory=lambda: [0.25, 20.0, 0.25, 0.01])
    width_ratios: list[float] = Field(default_factory=lambda: [20.0, 0.7])
    gridspec_hspace: float = 0.02
    gridspec_wspace: float = 0.07

    topo_label_fontsize: float = 7.0
    topo_label_y: float = 1.5
    ytick_fontsize: float = 7.5
    ytick_fontfamily: str = "monospace"
    row_stat_fontsize: float = 7.0
    row_stat_color: str = "gray"
    row_stat_x_offset: float = 1.0
    title_fontsize: float = 11.0
    title_y: float = 0.98

    hsep_linewidth: float = 0.8
    hsep_alpha: float = 0.7
    vsep_linewidth: float = 0.5
    vsep_alpha: float = 0.5

    xp_colormap: str = "bc_multi"
    xp_color_offset: float = 0.0
    xp_color_step: float = 1.0 / 7.0

    dev_mode: Literal["absolute", "relative"] = "relative"
    norm_type: Literal["symlog", "linear"] = "symlog"
    symlog_linthresh: float = 0.2
    percentile_low: float = 2.0
    percentile_high: float = 95.0
    dev_percentile: float = 95.0
    abs_cmap: str = "bc_blues"
    dev_mean_cmap: str = "bc_blrd"
    dev_best_cmap: str = "bc_blues"
    cmap_truncate: float = 0.95

    colorbar_height_frac: float = 0.3
    colorbar_width_frac: float = 0.2
    colorbar_margin: float = 0.05

    class_summary_cell_sep: float = 0.5
    class_summary_cell_sep_color: str = "white"
    class_summary_linthresh: float = 0.1
    class_summary_group_by_size: bool = False
    group_bracket_fontsize: float = 7.0
    group_bracket_color: str = "0.4"

    fold_cmap: str = "bc_blrd_r"
    vector_cells: bool = False
    weighted: bool = True

    pictogram_dir: str = find_default_pictogram_dir()
    pictogram_h_dir: str = find_default_pictogram_dir("pictograms_h")
    pictogram_v_dir: str = find_default_pictogram_dir("pictograms_v")
    use_pictograms: bool = True
    show_values_threshold: int = 4

    metric_label: str = "Grid nRMSE"
    metric_short: str = "nRMSE"


def _norm_for(mat: np.ndarray, kind: str, cfg: HeatmapConfig):
    finite = mat[np.isfinite(mat)]
    if kind == "abs":
        vmin, vmax = np.nanpercentile(finite, [cfg.percentile_low, cfg.percentile_high])
        if cfg.norm_type == "symlog":
            return SymLogNorm(linthresh=cfg.symlog_linthresh, vmin=vmin, vmax=vmax), cfg.abs_cmap
        return TwoSlopeNorm(vcenter=(vmin + vmax) / 2, vmin=vmin, vmax=vmax), cfg.abs_cmap
    if kind == "best":
        mat = np.clip(mat, 0, None)
        vmax = float(np.nanpercentile(np.abs(finite), cfg.percentile_high))
        if cfg.norm_type == "symlog":
            return SymLogNorm(linthresh=cfg.symlog_linthresh, vmin=0, vmax=vmax), cfg.dev_best_cmap
        return Normalize(vmin=0, vmax=vmax), cfg.dev_best_cmap
    vmax = float(np.nanpercentile(np.abs(finite), cfg.dev_percentile))
    if cfg.norm_type == "symlog":
        return SymLogNorm(linthresh=cfg.symlog_linthresh, vmin=-vmax, vmax=vmax), cfg.dev_mean_cmap
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), cfg.dev_mean_cmap


def _set_horizontal_cb_ticks(cb, norm, cfg: HeatmapConfig):
    if cfg.norm_type != "symlog" or not isinstance(norm, SymLogNorm):
        return
    ticks = [t for t in (0.1, 0.5, 1, 2, 5, 10) if norm.vmin <= t <= norm.vmax]
    if ticks:
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])


def _set_fold_cb_ticks(cb, norm):
    fold_ticks = (1.5, 2, 3, 5, 10)
    log_ticks = [0.0]
    labels = ["±1×"]
    vmax = getattr(norm, "vmax", 1.0)
    for f in fold_ticks:
        lf = math.log(f)
        if lf <= vmax * 1.01:
            log_ticks.extend([-lf, lf])
            labels.extend([f"−{f:g}×", f"+{f:g}×"])
    order = sorted(range(len(log_ticks)), key=lambda i: log_ticks[i])
    cb.set_ticks([log_ticks[i] for i in order])
    cb.set_ticklabels([labels[i] for i in order])


def _resolve_horizontal_heatmap(
    data: np.ndarray, net_meta: pd.DataFrame, row_order: list[str],
    *, view: GenViewConfig, view_name: str, data_mode: DataMode,
    loss_filter: str | None, cfg: HeatmapConfig,
):
    loss = loss_filter or ""
    _ref_label, data = _matrix_for_mode(data, data_mode, cfg.dev_mode)
    kind = {"abs": "abs", "dev_mean": "dev", "dev_best": "best"}[data_mode]
    norm, cmap_name = _norm_for(data, kind, cfg)
    cmap = maybe_truncate(cmap_name, cfg.cmap_truncate)
    stats = col_stats(data, net_meta, view.players, weighted=cfg.weighted)
    order = list(np.argsort(stats))
    data = data[:, order]
    row_order = [row_order[i] for i in order]
    stats = stats[order]
    title, cb_label = _titles_for(data_mode, cfg, view, view_name, loss)
    return data, row_order, stats, cmap, norm, title, cb_label


def _resolve_class_summary(
    matrix: np.ndarray, cond_order: list[str],
    *, view: GenViewConfig, view_name: str, data_mode: ClassDataMode,
    loss_filter: str | None, cfg: HeatmapConfig,
):
    loss = (loss_filter or "").title()
    if data_mode == "abs":
        finite = matrix[np.isfinite(matrix)]
        vmin, vmax = np.nanpercentile(finite, [cfg.percentile_low, cfg.percentile_high])
        norm = (SymLogNorm(linthresh=cfg.symlog_linthresh, vmin=vmin, vmax=vmax)
                if cfg.norm_type == "symlog" else Normalize(vmin=vmin, vmax=vmax))
        cmap = maybe_truncate(cfg.abs_cmap, cfg.cmap_truncate)
        data = matrix
        title = f"Class-Level Prediction Quality ({loss} Loss, {view_name})"
        cb_label = cfg.metric_label
        fold = False
    else:
        if data_mode == "devmean":
            ref = np.nanmean(matrix, axis=1, keepdims=True)
            title = f"Fold Change from Mean ({loss} Loss, {view_name})"
        else:
            full_cond = "".join(sorted(view.players))
            if full_cond not in cond_order:
                raise ValueError(f"devfull requires '{full_cond}' in cond_order")
            ref = matrix[:, cond_order.index(full_cond):cond_order.index(full_cond) + 1]
            title = f"Fold Change from Full Training ({loss} Loss, {view_name})"
        data = log_fold_dev(matrix, ref)
        finite = data[np.isfinite(data)]
        vmax = float(np.nanpercentile(np.abs(finite), cfg.dev_percentile)) or 1.0
        norm = SymLogNorm(linthresh=cfg.class_summary_linthresh, vmin=-vmax, vmax=vmax, base=10)
        cmap = maybe_truncate(cfg.fold_cmap, cfg.cmap_truncate)
        cb_label = f"{cfg.metric_short} fold change"
        fold = True
    return data, cmap, norm, title, cb_label, fold


def _matrix_for_mode(matrix: np.ndarray, mode: DataMode, dev_mode: str):
    if mode == "abs":
        return None, matrix
    if mode == "dev_mean":
        return "mean", deviation(matrix, np.nanmean(matrix, axis=1, keepdims=True), dev_mode)
    return "best", deviation(matrix, np.nanmin(matrix, axis=1, keepdims=True), dev_mode)


def _titles_for(mode: DataMode, cfg: HeatmapConfig, view: GenViewConfig, view_name: str, loss: str):
    loss_t = loss.title()
    rel = cfg.dev_mode == "relative"
    if mode == "abs":
        return (
            f"Per-Network Prediction Quality ({loss_t} Loss, {view_name})",
            cfg.metric_label,
        )
    if mode == "dev_mean":
        return (
            f"{'Relative ' if rel else ''}Deviation from Mean ({loss_t} Loss, {view_name})",
            "Relative deviation from mean (%)" if rel else f"{cfg.metric_short} − mean {cfg.metric_short}",
        )
    return (
        f"{'Relative ' if rel else ''}Deviation from Best ({loss_t} Loss, {view_name})",
        "Relative deviation from best (%)" if rel else f"{cfg.metric_short} − best {cfg.metric_short}",
    )


def draw_horizontal_heatmap_to_ax(
    ax: plt.Axes,
    data: np.ndarray, net_meta: pd.DataFrame, row_order: list[str],
    stats: np.ndarray, cmap, norm, title: str, cb_label: str,
    view: GenViewConfig, cfg: HeatmapConfig,
) -> None:
    mat_t = data.T
    n_conds, n_nets = mat_t.shape
    topo_arr = net_meta["topo_class"].values
    xp_arr = net_meta["experiment"].values
    topo_colors = view.get_colors()

    pictos, picto_bg = load_pictograms(Path(cfg.pictogram_h_dir), view.players)
    use_picto = cfg.use_pictograms and bool(pictos) and len(view.players) > cfg.show_values_threshold

    fig = ax.get_figure()
    pos = ax.get_position()
    ax.set_axis_off()
    gs = gridspec.GridSpec(
        4, 2, height_ratios=cfg.height_ratios, width_ratios=cfg.width_ratios,
        hspace=cfg.gridspec_hspace, wspace=cfg.gridspec_wspace,
        figure=fig, left=pos.x0, right=pos.x1, top=pos.y1, bottom=pos.y0,
    )

    ax_topo = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(topo_arr):
        ax_topo.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                                        facecolor=topo_colors.get(t, "gray"), edgecolor="none"))
    ax_topo.set_xlim(-0.5, n_nets - 0.5)
    ax_topo.set_ylim(0, 1)
    ax_topo.set_xticks([])
    ax_topo.set_yticks([])
    for t in view.players:
        idxs = np.where(topo_arr == t)[0]
        if len(idxs) > 0:
            mid = (idxs[0] + idxs[-1]) / 2
            ax_topo.text(mid, cfg.topo_label_y, view.labels.get(t, t),
                         ha="center", va="bottom", fontsize=cfg.topo_label_fontsize,
                         fontweight="bold", color=topo_colors.get(t, "gray"))

    ax_heat = fig.add_subplot(gs[1, 0])
    im = draw_heatmap_cells(ax_heat, mat_t, cmap, norm, vector=cfg.vector_cells)
    ax_heat.set_xticks([])

    fine_keys = view.fine_keys()
    if use_picto:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        ax_heat.set_yticks([])
        fig.canvas.draw()
        pos = ax_heat.get_position()
        fw, fh = fig.get_size_inches()
        row_inch = (pos.height * fh) / n_conds
        margin_inch = pos.x0 * fw - 0.05
        ref_icon = picto_bg if picto_bg is not None else next(iter(pictos.values()))
        icon_zoom = fit_zoom(
            ref_icon,
            max_w_inch=margin_inch if margin_inch > 0 else None,
            max_h_inch=row_inch,
            dpi=cfg.dpi,
            margin=0.85,
        )
        for i, cond in enumerate(row_order):
            members = parse_condition(cond, fine_keys)
            icon = composite_icon(members, pictos, picto_bg, invert=False)
            if icon is not None:
                ax_heat.add_artist(AnnotationBbox(
                    OffsetImage(icon, zoom=icon_zoom, interpolation="bilinear"),
                    (0, i),
                    xycoords=("axes fraction", "data"),
                    box_alignment=(1.0, 0.5), frameon=False,
                    pad=0.0, xybox=(-3, 0), boxcoords="offset points",
                    annotation_clip=False,
                ))
    else:
        ax_heat.set_yticks(range(n_conds))
        ax_heat.set_yticklabels(row_order, fontsize=cfg.ytick_fontsize, fontfamily=cfg.ytick_fontfamily)

    for i in range(1, n_conds):
        ax_heat.axhline(i - 0.5, color="white", linewidth=cfg.hsep_linewidth, alpha=cfg.hsep_alpha)
    prev_t = topo_arr[0]
    for i in range(1, n_nets):
        if topo_arr[i] != prev_t:
            ax_heat.axvline(i - 0.5, color="white", linewidth=cfg.vsep_linewidth, alpha=cfg.vsep_alpha)
            prev_t = topo_arr[i]

    for i, val in enumerate(stats):
        if np.isfinite(val):
            fmt = f"{val:+.2f}" if cb_label.startswith(cfg.metric_short) else f"{val:.2f}"
            ax_heat.text(n_nets + cfg.row_stat_x_offset, i, fmt,
                         ha="left", va="center", fontsize=cfg.row_stat_fontsize,
                         color=cfg.row_stat_color, fontfamily=cfg.ytick_fontfamily)

    ax_xp = fig.add_subplot(gs[2, 0], sharex=ax_heat)
    xp_cmap = plt.get_cmap(cfg.xp_colormap)
    unique_xps = list(dict.fromkeys(xp_arr))
    xp_colors = {xp: xp_cmap(cfg.xp_color_offset + k * cfg.xp_color_step)
                 for k, xp in enumerate(unique_xps)}
    for i, xp in enumerate(xp_arr):
        ax_xp.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                                      facecolor=xp_colors[xp], edgecolor="none"))
    ax_xp.set_xlim(-0.5, n_nets - 0.5)
    ax_xp.set_ylim(0, 1)
    ax_xp.set_xticks([])
    ax_xp.set_yticks([])

    fig.canvas.draw()
    pos = ax_heat.get_position()
    cb_w = (cfg.width_ratios[1] / sum(cfg.width_ratios)) * pos.width * cfg.colorbar_width_frac
    cb_h = pos.height * cfg.colorbar_height_frac
    ax_cb = fig.add_axes([
        pos.x1 + cfg.colorbar_margin,
        pos.y0 + (pos.height - cb_h) / 2,
        cb_w, cb_h,
    ])
    cb = fig.colorbar(im, cax=ax_cb, label=cb_label)
    _set_horizontal_cb_ticks(cb, norm, cfg)
    fig.text(
        (pos.x0 + pos.x1) / 2, pos.y1,
        title, fontsize=cfg.title_fontsize, ha="center", va="bottom",
    )


def draw_class_summary_to_ax(
    ax: plt.Axes,
    matrix: np.ndarray, class_order: list[str], cond_order: list[str],
    cmap, norm, cb_label: str, title: str,
    view: GenViewConfig, cfg: HeatmapConfig, *, fold_colorbar: bool = False,
) -> None:
    n_classes = len(class_order)
    n_conds = len(cond_order)
    view_keys = sorted(view.players, key=len, reverse=True)

    cond_members = [set(parse_condition(c, view_keys)) for c in cond_order]
    sizes = [len(m) for m in cond_members]
    groups: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, n_conds):
        if sizes[i] != sizes[start]:
            groups.append((start, i - 1, sizes[start]))
            start = i
    groups.append((start, n_conds - 1, sizes[start]))

    picto_v_dir = Path(cfg.pictogram_v_dir)
    picto_classes = list({*view.topo_mapping.values()} | set(view.players))
    pictos_v, picto_bg_v = load_pictograms(picto_v_dir, picto_classes)

    fig = ax.get_figure()
    pos = ax.get_position()
    ax.set_axis_off()
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[cfg.class_summary_strip_ratio, cfg.class_summary_bracket_ratio, cfg.class_summary_heatmap_ratio],
        width_ratios=[20, 0.8],
        hspace=0.04, wspace=0.05,
        figure=fig, left=pos.x0, right=pos.x1, top=pos.y1, bottom=pos.y0,
    )

    ax_comp = fig.add_subplot(gs[0, 0])
    ax_comp.set_xlim(-0.5, n_conds - 0.5)
    ax_comp.set_ylim(0, 1)
    ax_comp.set_xticks([])
    ax_comp.set_yticks([])
    for sp in ax_comp.spines.values():
        sp.set_visible(False)

    if pictos_v:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        fig.canvas.draw()
        comp_pos = ax_comp.get_position()
        fw, fh = fig.get_size_inches()
        ref_v = picto_bg_v if picto_bg_v is not None else next(iter(pictos_v.values()))
        icon_zoom_v = fit_zoom(
            ref_v,
            max_w_inch=(comp_pos.width * fw) / n_conds,
            max_h_inch=comp_pos.height * fh,
            dpi=cfg.dpi,
            margin=0.9,
        )
        for ci, members in enumerate(cond_members):
            member_list = [p for p in view.players if p in members]
            icon = composite_icon(member_list, pictos_v, picto_bg_v, invert=False)
            if icon is not None:
                ax_comp.add_artist(AnnotationBbox(
                    OffsetImage(icon, zoom=icon_zoom_v, interpolation="bilinear"),
                    (ci, 0.5), xycoords="data", box_alignment=(0.5, 0.5),
                    frameon=False, pad=0.0, annotation_clip=False,
                ))

    for g_start, _, _ in groups:
        if g_start > 0:
            ax_comp.axvline(g_start - 0.5, color="white", linewidth=1.0, alpha=0.8)

    ax_bracket = fig.add_subplot(gs[1, 0], sharex=ax_comp)
    if cfg.class_summary_group_by_size:
        max_size = max(s for _, _, s in groups)
        for g_start, g_end, g_size in groups:
            mid = (g_start + g_end + 1) / 2
            label = "all" if g_size == max_size else str(g_size)
            ax_bracket.text(mid, 0.5, label, ha="center", va="center",
                            fontsize=cfg.group_bracket_fontsize, fontweight="bold",
                            color=cfg.group_bracket_color)
            if g_start > 0:
                ax_bracket.axvline(g_start - 0.5, color=cfg.group_bracket_color,
                                   linewidth=0.5, alpha=0.4)
    ax_bracket.set_xlim(-0.5, n_conds - 0.5)
    ax_bracket.set_ylim(0, 1)
    ax_bracket.set_xticks([])
    ax_bracket.set_yticks([])
    for sp in ax_bracket.spines.values():
        sp.set_visible(False)

    ax_heat = fig.add_subplot(gs[2, 0], sharex=ax_comp)
    im = draw_heatmap_cells(ax_heat, matrix, cmap, norm, vector=cfg.vector_cells)
    ax_heat.set_xticks([])
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    if cfg.class_summary_group_by_size:
        for g_start, _, _ in groups:
            if g_start > 0:
                ax_heat.axvline(g_start - 0.5, color="white", linewidth=1.2, alpha=0.9)

    ax_heat.set_yticks(range(n_classes))
    ax_heat.set_yticklabels(
        [view.labels.get(p, p) for p in class_order],
        fontsize=cfg.ytick_fontsize, fontweight="bold", fontfamily=cfg.ytick_fontfamily,
    )

    sep_w = cfg.class_summary_cell_sep
    sep_c = cfg.class_summary_cell_sep_color
    full_cond = "".join(sorted(view.players))
    full_col_idx = cond_order.index(full_cond) if full_cond in cond_order else -1
    if sep_w > 0:
        for i in range(1, n_classes):
            ax_heat.axhline(i - 0.5, color=sep_c, linewidth=sep_w)
        for j in range(1, n_conds):
            lw = sep_w * 3 if j == full_col_idx else sep_w
            ax_heat.axvline(j - 0.5, color=sep_c, linewidth=lw)
        if full_col_idx > 0:
            ax_comp.axvline(full_col_idx - 0.5, color="0.5", linewidth=1.0)

    ax_cb = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(im, cax=ax_cb, label=cb_label)
    if fold_colorbar:
        _set_fold_cb_ticks(cb, norm)
    else:
        _set_horizontal_cb_ticks(cb, norm, cfg)

    fig.text(
        (pos.x0 + pos.x1) / 2, min(1.0, pos.y1 + 0.02),
        title, fontsize=cfg.title_fontsize, ha="center", va="bottom",
    )
