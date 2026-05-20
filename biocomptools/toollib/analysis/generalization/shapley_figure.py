# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Annotated, Literal

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biocomp.plotutils import *  # noqa: F401, F403
from dracon import Arg
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from pydantic import BaseModel, ConfigDict, Field, model_validator

from biocomptools.toollib.plot import Figure

from ._render import (
    _FigaxStub,
    composite_icon,
    find_default_pictogram_dir,
    load_pictograms,
    maybe_truncate,
)
from .pivot_build import build_pivot, load_metrics_csv
from .shapley_math import compute_shapley, fmt_value
from .views import ViewConfig, parse_condition

_DEFAULT_PICTOGRAM_DIR = find_default_pictogram_dir()


def _draw_grouped_marginal(
    ax, mat, players, topo_colors, cmap, norm, order_mode,
    axis: Literal["col", "row"],
    bar_width=0.8, label_fontsize=6.0, use_heatmap_colors=False,
):
    n = len(players)
    if n == 0:
        return
    step = bar_width / n
    offsets = np.linspace(-bar_width / 2 + step / 2, bar_width / 2 - step / 2, n)
    sort_key = {"by_value": lambda v: -v, "by_value_r": lambda v: v}.get(order_mode)
    for k in range(n):
        vals = mat[:, k] if axis == "col" else mat[k, :]
        indices = (
            sorted(range(n), key=lambda i: sort_key(vals[i])) if sort_key else list(range(n))
        )
        for slot, idx in enumerate(indices):
            v = vals[idx]
            color = cmap(norm(v)) if use_heatmap_colors else topo_colors[players[idx]]
            if axis == "col":
                ax.bar(k + offsets[slot], v, width=step * 0.9, color=color,
                       edgecolor="white", linewidth=0.3)
                va = "bottom" if v >= 0 else "top"
                ax.text(k + offsets[slot], v + v * 0.02, players[idx],
                        ha="center", va=va, fontsize=label_fontsize, fontweight="bold")
            else:
                ax.barh(k + offsets[slot], v, height=step * 0.9, color=color,
                        edgecolor="white", linewidth=0.3)
                ha = "left" if v >= 0 else "right"
                ax.text(v + v * 0.02, k + offsets[slot], players[idx],
                        ha=ha, va="center", fontsize=label_fontsize, fontweight="bold")
    if axis == "col":
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks([])
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    else:
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)


class ShapleyDetailConfig(BaseModel):
    figsize_per_player: Annotated[float, Arg(help="Figure dim per player")] = 2.75
    min_figsize: Annotated[float, Arg(help="Min figure dim")] = 9.0
    dpi: Annotated[int, Arg(help="Output DPI")] = 300
    weighted: Annotated[bool, Arg(help="Weight by experiment")] = True
    aggregation: Annotated[Literal["mean", "median"], Arg(help="Aggregation")] = "median"
    marginal_mode: Annotated[
        Literal["absolute", "relative", "fold", "percent"],
        Arg(help="Marginal mode (display unit)"),
    ] = "percent"
    style: Annotated[Literal["detailed", "simple"], Arg(help="Render style")] = "detailed"
    exclude_players: Annotated[list[str], Arg(help="Hide these players")] = Field(default_factory=list)
    exclude_target: Annotated[bool, Arg(help="Drop coalitions containing the target")] = False

    # Layout
    top: float = 0.88
    bottom: float = 0.14
    left: float = 0.08
    right: float = 0.92
    height_ratios: list[float] = Field(default_factory=lambda: [0.3, 5.0])
    width_ratios: list[float] = Field(default_factory=lambda: [5.0, 0.3])
    gridspec_hspace: float = 0.03
    gridspec_wspace: float = 0.03
    bar_width: float = 0.75

    # Colormap
    norm_type: Literal["symlog", "linear"] = "symlog"
    linthresh: float = 5.0
    symlog_linthresh: float = 5.0
    vmax_scale: float = 1.5
    vmax_percentile: float = 95.0
    heatmap_cmap: str = "bc_blrd_r"
    cmap_truncate: float = 0.95

    # Sub-cell styling
    cell_margin: float = 0.06
    subcell_fontsize: float = 6.5
    avg_fontsize: float = 7.5
    subcell_pad: float = 0.0
    subcell_edgecolor: str = "#000"
    subcell_edgewidth: float = 0.8
    cell_border_color: str = "white"
    cell_border_width: float = 2.5
    contrast_threshold: float = 0.55
    missing_color: str = "#f0f0f0"
    missing_text_color: str = "#ccc"
    value_format: str = "+.2g"
    subcell_order: Literal["by_size", "by_value", "by_value_r"] = "by_value_r"
    extra_agg: Literal["none", "trim", "median"] = "median"
    show_values_threshold: int = 4
    show_diagonal: bool = False

    # Pictograms
    use_pictograms: bool = True
    pictogram_dir: str = _DEFAULT_PICTOGRAM_DIR
    pictogram_h_dir: str = ""
    pictogram_pad: float = 0.08
    pictogram_ylabel_zoom: float = 0.25
    pictogram_ylabel_x: float = 0.0
    pictogram_ylabel_offset_x: float = -4.0
    pictogram_ylabel_sep: float = 2.0

    # Marginals
    show_col_marginal: bool = True
    show_row_marginal: bool = True
    col_marginal_mode: str = "rank_by_value"
    row_marginal_mode: str = "detail_by_value"
    marginal_use_heatmap_colors: bool = True

    # Rank badges
    show_rank: bool = False
    rank_fontsize: float = 6.0
    rank_position: str = "tl"

    # Tension barplot
    show_tension: bool = True
    tension_height_ratio: float = 0.4
    tension_hspace: float = 0.3
    tension_width_frac: float = 0.95
    tension_bar_width: float = 0.5
    tension_fontsize: float = 7.0
    tension_title: str = "Pairwise tension (−) or agreement (+)"
    tension_title_x: float | None = None
    tension_title_y: float | None = 0.5
    tension_picto_offset_y: float = -10.0
    tension_picto_pad_frac: float = 0.5

    # Typography
    xtick_fontsize: float = 9.5
    ytick_fontsize: float = 12.0
    ytick_fontfamily: str = "monospace"
    xlabel_fontsize: float = 10.0
    xlabel_pad: float = 20.0
    ylabel_fontsize: float = 10.0
    ylabel_pad: float = 130.0
    bar_edgecolor: str = "white"
    bar_linewidth: float = 0.8
    bar_alpha: float = 0.9
    bar_label_fontsize: float = 9.0
    bar_label_offset: float = 0.03
    row_bar_label_fontsize: float = 9.5
    row_bar_label_offset: float = 0.05
    col_bar_ylabel_fontsize: float = 8.5
    row_bar_title_fontsize: float = 8.5
    row_bar_title_pad: float = 6.0
    axis_tick_fontsize: float = 7.0
    row_bar_left_pad: float = 0.005

    # Colorbar
    colorbar_y: float = 0.065
    colorbar_height: float = 0.015
    colorbar_tick_fontsize: float = 7.0
    colorbar_annotation_fontsize: float = 7.5
    colorbar_annotation_y: float = -2.5
    hurts_color: str = "#b2182b"
    helps_color: str = "#2166ac"

    # Title / footnote
    title_fontsize: float = 14.0
    title_y: float = 0.98
    subtitle_fontsize: float = 9.5
    subtitle_y: float = 0.94
    subtitle_color: str = "0.4"
    footnote_fontsize: float = 7.0
    footnote_y: float = -0.005
    footnote_color: str = "0.5"

    title_text: str = "Training Data Transfer Between Topology Classes (Detail)"
    subtitle_template: str = (
        "Each sub-cell = % nRMSE change from adding row type to a baseline "
        "coalition ({loss_type} loss, {view} view)"
    )
    footnote_text: str = "Bordered diagonal = self-prediction.  Blue = helps, red = hurts."
    xlabel_text: str = "Prediction target (topology class)"
    ylabel_text: str = "Topology class added to training"
    col_bar_ylabel_text: str = "Needs training\nfrom"
    row_bar_title_text: str = "Good at\npredicting"
    colorbar_label: str = "nRMSE improvement (symmetric %)"
    hurts_label: str = "← hurts prediction"
    helps_label: str = "helps prediction ->"

    @model_validator(mode="after")
    def _link_linthresh(self):
        # Back-compat: paper_plots_v2 uses `symlog_linthresh`; figuremaker
        # uses `linthresh`. Sync if user only set one.
        if self.symlog_linthresh != 5.0 and self.linthresh == 5.0:
            object.__setattr__(self, "linthresh", self.symlog_linthresh)
        return self


class ShapleyDetailFigure(Figure):
    view: ViewConfig
    view_name: str = ""
    shapley_conf: ShapleyDetailConfig = Field(default_factory=ShapleyDetailConfig)
    dataframe_path: str | None = None
    df: pd.DataFrame | None = None
    loss_filter: str | None = "regression"
    metric: str = "grid_nrmse"
    loss_label: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self, overwrite: bool = True) -> None:
        if (
            not overwrite
            and self.figure_spec.output_path
            and self.figure_spec.output_path.exists()
        ):
            return

        df = self.df if self.df is not None else load_metrics_csv(self.dataframe_path or "")
        pivot, net_meta, row_order = build_pivot(
            df, self.view, metric=self.metric, loss_filter=self.loss_filter
        )

        sc = self.shapley_conf
        shapley_mat, players, detailed = compute_shapley(
            pivot, net_meta, row_order, self.view,
            exclude_target=sc.exclude_target,
            weighted=sc.weighted,
            aggregation=sc.aggregation,
            marginal_mode=sc.marginal_mode,
        )

        with mpl.rc_context(rc=self.plot_config.rc_context):
            fig = self._draw(shapley_mat, players, detailed)
            metadata = dict(self.figure_spec.metadata) if self.figure_spec.metadata else {}
            metadata["analysis"] = {
                "view": self.view_name or "unnamed",
                "players": list(players),
                "loss_filter": self.loss_filter,
                "metric": self.metric,
                "marginal_mode": sc.marginal_mode,
                "aggregation": sc.aggregation,
                "shapley_matrix": shapley_mat.tolist(),
            }
            self.figure_spec.metadata = metadata
            self.figure_spec.finalize(_FigaxStub(fig))

    def _draw(
        self,
        shapley: np.ndarray,
        players: list[str],
        detailed: list[list[list[tuple[str, float, float]]]],
    ) -> plt.Figure:
        sc = self.shapley_conf
        view = self.view
        loss_label = self.loss_label or (self.loss_filter or "")
        view_name = self.view_name or "view"
        topo_colors = view.get_colors()

        if sc.exclude_players:
            keep = [k for k, p in enumerate(players) if p not in sc.exclude_players]
            players = [players[k] for k in keep]
            shapley = shapley[np.ix_(keep, keep)]
            detailed = [[detailed[i][j] for j in keep] for i in keep]

        n = len(players)
        is_simple = sc.style == "simple"

        if is_simple:
            med = np.zeros_like(shapley)
            for i in range(n):
                for j in range(n):
                    contribs = [v for _, v, _ in detailed[i][j]]
                    med[i, j] = float(np.median(contribs)) if contribs else shapley[i, j]
            if not sc.show_diagonal:
                np.fill_diagonal(med, 0.0)
            shapley = med

        n_extra = 2 if sc.extra_agg != "none" else 1
        if is_simple:
            n_coalitions = 0
            n_total = 1
        elif sc.exclude_target:
            n_coalitions = 2 ** max(n - 2, 0) - 1
            n_total = n_coalitions + n_extra
        else:
            n_coalitions = 2 ** (n - 1) - 1
            n_total = n_coalitions + n_extra

        grid_cols = int(np.ceil(np.sqrt(n_total)))
        grid_rows = int(np.ceil(n_total / grid_cols))

        fig_w = max(sc.min_figsize, n * sc.figsize_per_player)
        h_ratios_all = list(sc.height_ratios) + (
            [sc.tension_hspace, sc.tension_height_ratio] if sc.show_tension else []
        )
        w_frac = sc.width_ratios[0] / sum(sc.width_ratios)
        h_frac = h_ratios_all[1] / sum(h_ratios_all)
        cell_w = fig_w * (sc.right - sc.left) * w_frac
        fig_h = cell_w / ((sc.top - sc.bottom) * h_frac)
        figsize = (fig_w, fig_h)

        contrib_map: list[list[dict[str, float]]] = [[{} for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for label, val, _w in detailed[i][j]:
                    contrib_map[i][j][label] = val

        def _build_coalition_labels(others_list: list[str]) -> list[str]:
            labels: list[str] = []
            for size in range(1, len(others_list)):
                for combo in itertools.combinations(others_list, size):
                    labels.append("".join(combo))
            labels.append("".join(others_list))
            return labels

        if sc.exclude_target:
            all_coalitions_2d: list[list[list[str]]] = []
            for p in players:
                others = sorted(q for q in players if q != p)
                row = []
                for target in players:
                    excl = sorted(q for q in others if q != target)
                    row.append(_build_coalition_labels(excl) if excl else [])
                all_coalitions_2d.append(row)
            all_coalitions = None
        else:
            all_coalitions_2d = None
            all_coalitions: list[list[str]] = []
            for p in players:
                others = sorted(q for q in players if q != p)
                all_coalitions.append(_build_coalition_labels(others))

        if is_simple:
            all_vals = shapley.flatten().tolist()
        else:
            all_vals = [v for row in detailed for cell in row for _, v, _ in cell]
            all_vals.extend(shapley.flatten().tolist())
        if all_vals:
            abs_vals = np.abs(all_vals)
            vmax = float(np.percentile(abs_vals, sc.vmax_percentile)) * sc.vmax_scale
        else:
            vmax = 1.0
        if vmax == 0:
            vmax = 1.0
        cmap_obj = maybe_truncate(sc.heatmap_cmap, sc.cmap_truncate)
        if sc.norm_type == "symlog":
            color_norm = SymLogNorm(linthresh=sc.linthresh, vmin=-vmax, vmax=vmax, base=10)
        else:
            color_norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

        pictograms, picto_bg = load_pictograms(Path(sc.pictogram_dir), players)
        use_picto = False

        def _make_coalition_icon(label: str, invert: bool):
            return composite_icon(parse_condition(label, players), pictograms, picto_bg, invert=invert)

        n_gs_rows = 4 if sc.show_tension else 2
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=sc.top, bottom=sc.bottom, left=sc.left, right=sc.right)
        gs = gridspec.GridSpec(
            n_gs_rows, 2,
            height_ratios=h_ratios_all, width_ratios=sc.width_ratios,
            hspace=sc.gridspec_hspace, wspace=sc.gridspec_wspace,
        )

        ax = fig.add_subplot(gs[1, 0])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")

        margin = sc.cell_margin
        pad = sc.subcell_pad
        vfmt = sc.value_format
        is_fold = sc.marginal_mode == "fold"
        is_percent = sc.marginal_mode == "percent"

        def _fv(val: float) -> str:
            return fmt_value(val, sc.marginal_mode, vfmt)

        inner = 1.0 - 2 * margin
        sub_w = inner / grid_cols
        sub_h = inner / grid_rows

        def _text_color(val: float) -> str:
            rgba = cmap_obj(color_norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            return "white" if lum < sc.contrast_threshold else "black"

        def _draw_subcell(rx, ry, rw, rh, val, txt, bold=False, coalition_label=None):
            if val is not None:
                color = cmap_obj(color_norm(val))
                tc = _text_color(val)
                is_dark = tc == "white"
            else:
                color = sc.missing_color
                tc = sc.missing_text_color
                is_dark = False
            ax.add_patch(plt.Rectangle(
                (rx, ry), rw, rh,
                facecolor=color, edgecolor=sc.subcell_edgecolor,
                linewidth=sc.subcell_edgewidth,
            ))
            if use_picto and coalition_label is not None and not bold:
                icon = _make_coalition_icon(coalition_label, invert=is_dark)
                if icon is not None:
                    pp = sc.pictogram_pad
                    avail_w = rw * (1 - 2 * pp)
                    avail_h = rh * (1 - 2 * pp)
                    img_h, img_w = icon.shape[:2]
                    img_aspect = img_w / img_h
                    cell_aspect = avail_w / avail_h
                    if img_aspect > cell_aspect:
                        draw_w = avail_w
                        draw_h = avail_w / img_aspect
                    else:
                        draw_h = avail_h
                        draw_w = avail_h * img_aspect
                    cx_img = rx + rw / 2
                    cy_img = ry + rh / 2
                    ax.imshow(
                        icon,
                        extent=[
                            cx_img - draw_w / 2, cx_img + draw_w / 2,
                            cy_img + draw_h / 2, cy_img - draw_h / 2,
                        ],
                        aspect="auto", interpolation="bilinear", zorder=2,
                    )
                    return
            fs = sc.avg_fontsize if bold else sc.subcell_fontsize
            ax.text(rx + rw / 2, ry + rh / 2, txt,
                    ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", color=tc)

        grid_positions = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
        show_vals = sc.show_values_threshold == 0 or n <= sc.show_values_threshold
        use_picto = sc.use_pictograms and not show_vals and bool(pictograms)
        n_coalition_slots = n_coalitions

        for i in range(n):
            for j in range(n):
                if not sc.show_diagonal and i == j:
                    continue
                if is_simple:
                    inner_w = 1.0 - 2 * margin
                    inner_h = 1.0 - 2 * margin
                    _draw_subcell(
                        j - 0.5 + margin, i - 0.5 + margin,
                        inner_w, inner_h, shapley[i, j],
                        _fv(shapley[i, j]), bold=True,
                    )
                    continue

                coalition_labels = (
                    all_coalitions_2d[i][j] if sc.exclude_target else all_coalitions[i]
                )
                cx = j - 0.5 + margin
                cy = i - 0.5 + margin
                cm = contrib_map[i][j]

                if sc.subcell_order in ("by_value", "by_value_r"):
                    desc = sc.subcell_order == "by_value"
                    ordered = sorted(
                        coalition_labels,
                        key=lambda lb: (cm.get(lb) is None, (-1 if desc else 1) * (cm.get(lb) or 0)),
                    )
                else:
                    ordered = list(coalition_labels)

                for idx, label in enumerate(ordered[:n_coalition_slots]):
                    sr, scol = grid_positions[idx]
                    rx = cx + scol * sub_w + pad
                    ry = cy + sr * sub_h + pad
                    rw = sub_w - 2 * pad
                    rh = sub_h - 2 * pad
                    val = cm.get(label)
                    txt = (
                        (f"{label}\n{_fv(val)}" if val is not None else f"{label}\n-")
                        if show_vals else label
                    )
                    _draw_subcell(rx, ry, rw, rh, val, txt, coalition_label=label)

                avg_val = shapley[i, j]
                last_coalition_pos = (
                    grid_positions[n_coalition_slots - 1] if n_coalition_slots > 0 else (0, 0)
                )
                last_row = (
                    grid_positions[n_coalition_slots][0]
                    if n_coalition_slots < len(grid_positions) else last_coalition_pos[0]
                )
                agg_start_col = (
                    grid_positions[n_coalition_slots][1]
                    if n_coalition_slots < len(grid_positions) else 0
                )
                remaining_cols = grid_cols - agg_start_col

                if sc.extra_agg != "none" and remaining_cols >= 2:
                    half_w = remaining_cols / 2.0
                    avg_rx = cx + agg_start_col * sub_w + pad
                    avg_ry = cy + last_row * sub_h + pad
                    avg_rw = half_w * sub_w - 2 * pad
                    avg_rh = sub_h - 2 * pad
                    _draw_subcell(avg_rx, avg_ry, avg_rw, avg_rh,
                                  avg_val, f"avg\n{_fv(avg_val)}", bold=True)

                    contribs = [(v, w) for _, v, w in detailed[i][j]]
                    if sc.extra_agg == "median":
                        extra_val = (
                            float(np.median([v for v, _ in contribs])) if contribs else avg_val
                        )
                        extra_label = "med"
                    else:
                        if len(contribs) >= 3:
                            contribs.sort(key=lambda x: x[0])
                            trimmed = contribs[1:-1]
                            tw = sum(w for _, w in trimmed)
                            extra_val = (
                                sum(v * w for v, w in trimmed) / tw if tw > 0 else avg_val
                            )
                        else:
                            extra_val = avg_val
                        extra_label = "trim"
                    extra_rx = cx + (agg_start_col + half_w) * sub_w + pad
                    extra_rw = half_w * sub_w - 2 * pad
                    _draw_subcell(extra_rx, avg_ry, extra_rw, avg_rh,
                                  extra_val, f"{extra_label}\n{_fv(extra_val)}", bold=True)
                else:
                    avg_rx = cx + agg_start_col * sub_w + pad
                    avg_ry = cy + last_row * sub_h + pad
                    avg_rw = remaining_cols * sub_w - 2 * pad
                    avg_rh = sub_h - 2 * pad
                    _draw_subcell(avg_rx, avg_ry, avg_rw, avg_rh,
                                  avg_val, f"avg\n{_fv(avg_val)}", bold=True)

        for edge in np.arange(-0.5, n, 1):
            ax.axhline(edge, color=sc.cell_border_color, linewidth=sc.cell_border_width)
            ax.axvline(edge, color=sc.cell_border_color, linewidth=sc.cell_border_width)

        x_labels = [view.labels.get(p, p) for p in players]
        y_labels = list(players)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(x_labels, fontsize=sc.xtick_fontsize, fontweight="bold")
        ax.set_yticklabels(y_labels, fontsize=sc.ytick_fontsize, fontweight="bold",
                           fontfamily=sc.ytick_fontfamily)
        for k, p in enumerate(players):
            ax.get_xticklabels()[k].set_color(topo_colors.get(p, "gray"))
            ax.get_yticklabels()[k].set_color(topo_colors.get(p, "gray"))
        ax.set_xlabel(sc.xlabel_text, fontsize=sc.xlabel_fontsize, labelpad=sc.xlabel_pad)
        ax.set_ylabel(sc.ylabel_text, fontsize=sc.ylabel_fontsize, labelpad=sc.ylabel_pad)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if use_picto:
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

            ax.set_yticklabels([""] * n)
            for k, p in enumerate(players):
                icon = composite_icon([p], pictograms, picto_bg, invert=False)
                if icon is None:
                    continue
                oimg = OffsetImage(icon, zoom=sc.pictogram_ylabel_zoom, interpolation="bilinear")
                txt = TextArea(
                    view.labels.get(p, p),
                    textprops=dict(
                        fontsize=sc.ytick_fontsize, fontweight="bold",
                        fontfamily=sc.ytick_fontfamily,
                        color=topo_colors.get(p, "gray"), ha="center",
                    ),
                )
                pack = VPacker(children=[oimg, txt], align="center", pad=0,
                               sep=sc.pictogram_ylabel_sep)
                ab = AnnotationBbox(
                    pack, (sc.pictogram_ylabel_x, k),
                    xycoords=("axes fraction", "data"),
                    box_alignment=(1.0, 0.5), frameon=False, pad=0.0,
                    xybox=(sc.pictogram_ylabel_offset_x, 0),
                    boxcoords="offset points", annotation_clip=False,
                )
                ax.add_artist(ab)

        # Marginals
        row_sums = shapley.sum(axis=1)
        col_sums = shapley.sum(axis=0)
        bw = sc.bar_width

        ax_col = fig.add_subplot(gs[0, 0])
        if not sc.show_col_marginal:
            ax_col.set_visible(False)
        elif sc.col_marginal_mode.startswith("rank"):
            rank_order = sc.col_marginal_mode.replace("rank_", "").replace("rank", "fixed")
            _draw_grouped_marginal(ax_col, shapley, players, topo_colors, cmap_obj, color_norm,
                                order_mode=rank_order, axis="col",
                                label_fontsize=sc.rank_fontsize,
                                use_heatmap_colors=sc.marginal_use_heatmap_colors)
            ax_col.set_ylabel(sc.col_bar_ylabel_text, fontsize=sc.col_bar_ylabel_fontsize,
                              fontweight="bold")
        else:
            colors = (
                [cmap_obj(color_norm(v)) for v in col_sums]
                if sc.marginal_use_heatmap_colors
                else [topo_colors.get(p, "gray") for p in players]
            )
            ax_col.bar(range(n), col_sums, width=bw, color=colors,
                       edgecolor=sc.bar_edgecolor, linewidth=sc.bar_linewidth, alpha=sc.bar_alpha)
            ax_col.set_xlim(-0.5, n - 0.5)
            ax_col.set_xticks([])
            ax_col.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
            ax_col.set_ylabel(sc.col_bar_ylabel_text, fontsize=sc.col_bar_ylabel_fontsize,
                              fontweight="bold")
            for k, val in enumerate(col_sums):
                va = "bottom" if val >= 0 else "top"
                offset = sc.bar_label_offset if val >= 0 else -sc.bar_label_offset
                ax_col.text(k, val + offset, _fv(val), ha="center", va=va,
                            fontsize=sc.bar_label_fontsize)
            for s in ("top", "right", "left"):
                ax_col.spines[s].set_visible(False)
            ax_col.tick_params(axis="x", length=0, labelbottom=False)
            ax_col.tick_params(axis="y", length=0, labelleft=False)

        ax_row = fig.add_subplot(gs[1, 1])
        if not sc.show_row_marginal:
            ax_row.set_visible(False)
        elif sc.row_marginal_mode.startswith("detail"):
            detail_order = sc.row_marginal_mode.replace("detail_", "").replace("detail", "fixed")
            _draw_grouped_marginal(ax_row, shapley, players, topo_colors, cmap_obj, color_norm,
                                      order_mode=detail_order, axis="row", bar_width=bw,
                                      label_fontsize=sc.rank_fontsize,
                                      use_heatmap_colors=sc.marginal_use_heatmap_colors)
            ax_row.set_xlabel(sc.row_bar_title_text, fontsize=sc.row_bar_title_fontsize,
                              fontweight="bold", labelpad=sc.row_bar_title_pad)
        else:
            colors = (
                [cmap_obj(color_norm(v)) for v in row_sums]
                if sc.marginal_use_heatmap_colors
                else [topo_colors.get(p, "gray") for p in players]
            )
            ax_row.barh(range(n), row_sums, height=bw, color=colors,
                        edgecolor=sc.bar_edgecolor, linewidth=sc.bar_linewidth, alpha=sc.bar_alpha)
            ax_row.set_ylim(n - 0.5, -0.5)
            ax_row.set_yticks([])
            ax_row.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
            ax_row.set_xlabel(sc.row_bar_title_text, fontsize=sc.row_bar_title_fontsize,
                              fontweight="bold", labelpad=sc.row_bar_title_pad)
            for k, val in enumerate(row_sums):
                ax_row.text(val + sc.row_bar_label_offset, k, _fv(val),
                            va="center", ha="left", fontsize=sc.row_bar_label_fontsize)
            for s in ("top", "right", "bottom"):
                ax_row.spines[s].set_visible(False)
            ax_row.tick_params(axis="y", length=0, labelleft=False)
            ax_row.tick_params(axis="x", length=0, labelbottom=False)

        fig.canvas.draw()
        heat_pos = ax.get_position()
        if sc.show_col_marginal:
            col_pos = ax_col.get_position()
            ax_col.set_position([heat_pos.x0, col_pos.y0, heat_pos.width, col_pos.height])
        if sc.show_row_marginal:
            row_pos = ax_row.get_position()
            ax_row.set_position([
                heat_pos.x1 + sc.row_bar_left_pad, heat_pos.y0, row_pos.width, heat_pos.height
            ])

        if sc.show_tension:
            ax_tension = fig.add_subplot(gs[3, 0])
            if sc.tension_width_frac < 1.0:
                pos = ax_tension.get_position()
                new_w = pos.width * sc.tension_width_frac
                ax_tension.set_position([
                    pos.x0 + (pos.width - new_w) / 2, pos.y0, new_w, pos.height
                ])
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    agreement = (shapley[i, j] + shapley[j, i]) / 2
                    pairs.append((players[i], players[j], agreement))
            pairs.sort(key=lambda x: x[2])
            x_pos = np.arange(len(pairs))
            vals = [p[2] for p in pairs]
            colors = [cmap_obj(color_norm(v)) for v in vals]
            ax_tension.bar(x_pos, vals, width=sc.tension_bar_width, color=colors,
                           edgecolor="white", linewidth=0.5)
            ax_tension.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax_tension.set_xlim(-0.5, len(pairs) - 0.5)
            title_kwargs = dict(fontsize=sc.tension_fontsize + 1, fontweight="bold")
            if sc.tension_title_x is not None:
                title_kwargs["x"] = sc.tension_title_x
            if sc.tension_title_y is not None:
                title_kwargs["y"] = sc.tension_title_y
            ax_tension.set_title(sc.tension_title, **title_kwargs)
            for spine in ("top", "right", "left", "bottom"):
                ax_tension.spines[spine].set_visible(False)
            ax_tension.tick_params(axis="y", length=0, labelleft=False)
            ax_tension.tick_params(axis="x", length=0)
            label_offset = max(abs(max(vals, key=abs)) * 0.03, 0.3) if vals else 0.3
            lowest_label_y = 0.0
            for k, val in enumerate(vals):
                va = "bottom" if val >= 0 else "top"
                label_y = val + (label_offset if val >= 0 else -label_offset)
                ax_tension.text(k, label_y, _fv(val), ha="center", va=va,
                                fontsize=sc.tension_fontsize)
                if val < 0:
                    lowest_label_y = min(lowest_label_y, label_y)
            if use_picto and pictograms:
                from matplotlib.offsetbox import AnnotationBbox, OffsetImage

                ax_tension.set_xticks(x_pos)
                ax_tension.set_xticklabels([""] * len(pairs))
                ylo, yhi = ax_tension.get_ylim()
                data_range = yhi - ylo
                text_height = data_range * 0.12
                picto_anchor = (
                    lowest_label_y - text_height + sc.tension_picto_offset_y * (data_range / 100)
                )
                pad = data_range * sc.tension_picto_pad_frac
                ax_tension.set_ylim(picto_anchor - pad, yhi)
                picto_y = picto_anchor - pad * 0.5
                for k, (p1, p2, _) in enumerate(pairs):
                    icon = composite_icon([p1, p2], pictograms, picto_bg, invert=False)
                    if icon is not None:
                        oimg = OffsetImage(icon, zoom=0.18, interpolation="bilinear")
                        ab = AnnotationBbox(oimg, (k, picto_y), xycoords="data",
                                            box_alignment=(0.5, 0.5), frameon=False,
                                            pad=0.0, annotation_clip=False)
                        ax_tension.add_artist(ab)
            else:
                ax_tension.set_xticks(x_pos)
                ax_tension.set_xticklabels(
                    [f"{p[0]}↔{p[1]}" for p in pairs],
                    fontsize=sc.tension_fontsize, fontfamily="monospace",
                    rotation=45, ha="right",
                )

        heat_pos = ax.get_position()
        ax_cb = fig.add_axes([heat_pos.x0, sc.colorbar_y, heat_pos.width, sc.colorbar_height])
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=color_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        cb.set_label(sc.colorbar_label, fontsize=sc.colorbar_annotation_fontsize)
        if is_fold:
            fold_ticks = [1, 1.5, 2, 3, 5, 10, 20, 50]
            log_ticks: list[float] = []
            labels: list[str] = []
            for f in fold_ticks:
                lf = math.log(f)
                if lf <= vmax * 1.01:
                    if f > 1:
                        log_ticks.extend([-lf, lf])
                        labels.extend([f"−{f:g}×", f"+{f:g}×"])
                    else:
                        log_ticks.append(0.0)
                        labels.append("±1×")
            order = sorted(range(len(log_ticks)), key=lambda i: log_ticks[i])
            cb.set_ticks([log_ticks[i] for i in order])
            cb.set_ticklabels([labels[i] for i in order])
            ax_cb.minorticks_off()
        elif is_percent:
            percent_ticks = [5, 10, 20, 50, 100, 200, 500]
            log_ticks = [0.0]
            labels = ["0%"]
            for pct in percent_ticks:
                lf = math.log(1 + pct / 100)
                if lf <= vmax * 1.01:
                    log_ticks.extend([-lf, lf])
                    labels.extend([f"−{pct:g}%", f"+{pct:g}%"])
            order = sorted(range(len(log_ticks)), key=lambda i: log_ticks[i])
            cb.set_ticks([log_ticks[i] for i in order])
            cb.set_ticklabels([labels[i] for i in order])
            ax_cb.minorticks_off()
        elif sc.norm_type == "symlog":
            if vmax > 50:
                ticks = [t for t in [-100, -50, -10, 0, 10, 50, 100] if -vmax <= t <= vmax]
            elif vmax > 5:
                ticks = [t for t in [-50, -10, -5, 0, 5, 10, 50] if -vmax <= t <= vmax]
            else:
                ticks = [t for t in [-1, -0.1, 0, 0.1, 1] if -vmax <= t <= vmax]
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:g}" for t in ticks])
            ax_cb.minorticks_off()
        ax_cb.tick_params(labelsize=sc.colorbar_tick_fontsize)
        ax_cb.text(0.0, sc.colorbar_annotation_y, sc.hurts_label,
                   transform=ax_cb.transAxes, ha="left", va="top",
                   fontsize=sc.colorbar_annotation_fontsize, color=sc.hurts_color,
                   fontweight="bold")
        ax_cb.text(1.0, sc.colorbar_annotation_y, sc.helps_label,
                   transform=ax_cb.transAxes, ha="right", va="top",
                   fontsize=sc.colorbar_annotation_fontsize, color=sc.helps_color,
                   fontweight="bold")

        fig.suptitle(sc.title_text, fontsize=sc.title_fontsize, fontweight="bold", y=sc.title_y)
        fig.text(0.5, sc.subtitle_y,
                 sc.subtitle_template.format(loss_type=loss_label or "n/a", view=view_name),
                 ha="center", fontsize=sc.subtitle_fontsize, color=sc.subtitle_color)
        fig.text(0.5, sc.footnote_y, sc.footnote_text,
                 ha="center", fontsize=sc.footnote_fontsize, color=sc.footnote_color,
                 style="italic")
        return fig
