# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import matplotlib as mpl
import matplotlib.pyplot as plt
from dracon.draconstructor import resolve_all_lazy
from typing import Literal, Union, List, Optional, Dict, Tuple
from collections import defaultdict
from biocomp.plotutils import PlotData, smooth, slice_panel_args, IDENTITY_RESCALER
from jeanplot.plots.smooth_1d import smooth_1d
from biocomp.datautils import DataRescaler
from biocomptools.toollib.plot import PlotConfig, load_default_plotconf
from biocomptools.toollib._bbox_subfig import carve_bbox
from biocomptools.logging_config import get_logger
from matplotlib.lines import Line2D
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import numpy as np

logger = get_logger(__name__)


class GridPlotConfig(BaseModel):
    row: Optional[int] = None
    col: Optional[int] = None
    plot_config: Optional[PlotConfig] = None

    def matches(self, r: int, c: int, grid_size: tuple[int, int]) -> bool:
        if self.row is not None and self.row < 0:
            row = grid_size[0] + self.row
        else:
            row = self.row
        if self.col is not None and self.col < 0:
            col = grid_size[1] + self.col
        else:
            col = self.col
        return (row is None or row == r) and (col is None or col == c)


DEFAULT_RESOLUTION = 30

DEFAULT_GRID_PLOTCONFIGS = [
    GridPlotConfig(
        plot_config=PlotConfig(
            rc_context={
                "xtick.bottom": False,
                "xtick.top": False,
                "ytick.left": False,
                "ytick.right": False,
                "xtick.labelbottom": False,
                "xtick.labeltop": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
                "xtick.labelsize": 40,
                "ytick.labelsize": 40,
                "ytick.major.size": 10,
                "ytick.minor.size": 5,
                "xtick.major.size": 10,
                "xtick.minor.size": 5,
                "axes.spines.left": False,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "axes.spines.bottom": False,
            },
            callstack_params={
                "smooth_2d_params": {
                    "knn_grid_params": {"grid_resolution": DEFAULT_RESOLUTION},
                    "draw_colorbar": False,
                    "xtitle": False,
                    "ytitle": False,
                    "title": None,
                    "heatmap_params": {"contours": None},
                },
            },
        )
    ),
    GridPlotConfig(
        col=0,
        plot_config=PlotConfig(
            rc_context={
                "ytick.left": True,
                "axes.spines.left": True,
                "xtick.labelbottom": False,
                "xtick.labeltop": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
            }
        ),
    ),
    GridPlotConfig(
        row=-1,
        plot_config=PlotConfig(
            rc_context={
                "xtick.bottom": True,
                "axes.spines.bottom": True,
                "xtick.labelbottom": False,
                "xtick.labeltop": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
            }
        ),
    ),
    GridPlotConfig(
        col=-1,
        row=0,
        plot_config=PlotConfig(
            callstack_params={
                "smooth_2d_params": {
                    "draw_colorbar": True,
                    "colorbar_params": {
                        "size": [0.3, 3],
                        "position": [1.75, -2],
                        "tick_props": {"labelsize": 15, "pad": 8, "length": 10, "width": 1},
                        "label_position": "left",
                        "label_props": {"size": 15},
                    },
                }
            }
        ),
    ),
]

DEFAULT_UORF_VALUE_TO_STR = {
    0: "0x",
    5: "1w",
    10: "1x",
    20: "2x",
    30: "3x",
    40: "4x",
    50: "5x",
    60: "6x",
    70: "7x",
    80: "8x",
}

ANNOTATION_STYLE = {
    "color": "#F76665",
    "linewidth": 2.5,
    "clip_on": False,
    "zorder": 200,
    "linestyle": "--",
}


def _merge_lims(user, auto):
    if user is None:
        return list(auto)
    return [auto[i] if v is None else v for i, v in enumerate(user)]


class UORFCell(BaseModel):
    row: int
    col: int
    uorf_values: Tuple[float, float]
    data: PlotData

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UORFMatrixFigure(BaseModel):
    plot_data: List[PlotData]
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    rescaler: DataRescaler = Field(default_factory=lambda: IDENTITY_RESCALER)

    uorf_axis_order: tuple[int, int] = (1, 0)
    grid_plotconfigs: List[GridPlotConfig] = []

    reverse_rows: bool = True
    reverse_cols: bool = False

    uorf_value_to_str: Optional[Dict[int, str]] = DEFAULT_UORF_VALUE_TO_STR

    label_fontsize: int = 20
    rmse_fontsize: int = 8
    xlabel_padding: float = 0.3
    ylabel_padding: float = 0.3
    axis_title_padding: float = 0.6
    label_fontweight: int = 500
    legend_row: int = -1
    legend_col: int = 0

    grid_line_color: str = "#444444"
    grid_line_width: float = 0.5
    grid_line_margin: float = 0.0
    grid_line_style: str | tuple = (0, (0, 3))

    grid_cross_color: str = "#aaaaaa"
    grid_cross_width: float = 1.0
    grid_cross_length: float = 0.0
    grid_cross_style: str | tuple = (0, (2, 8))

    annotate: List[tuple[int, int]] = []
    annotation_style: dict[str, Union[str, float]] = ANNOTATION_STYLE
    annotation_margin: float = 0.08
    show_individual_rmse: bool = True
    show_overall_rmse: bool = True
    overall_rmse_fontsize: int = 8
    rmse_prefix: str = "RMSE: "

    # heatmap (default) or 1D slice overlay; held values are in plot_data.x space
    cell_kind: Literal["heatmap", "slices"] = "heatmap"
    slice_x_held_raw: List[float] = []
    slice_y_held_raw: List[float] = []
    slice_x_cmap: str = "bc_reds"
    slice_y_cmap: str = "bc_greens"
    slice_cmap_range: tuple[float, float] = (0.35, 0.9)
    # None = auto-scale to union range across cells; per-bound null falls back to auto
    slice_xlims: Optional[List[Optional[float]]] = None
    slice_vlims: Optional[List[Optional[float]]] = None
    slice_knn_stats_params: dict = {}
    slice_lineplot_props: dict = {"lw": 1.2, "marker": ""}
    slice_res: int = 200

    wspace: float = 0.25
    hspace: float = 0.25

    _global_lims: dict = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        resolve_all_lazy(self.plot_config)
        for gc in DEFAULT_GRID_PLOTCONFIGS + self.grid_plotconfigs:
            resolve_all_lazy(gc.plot_config)

    def get_plot_config_for_cell(
        self, row: int, col: int, grid_size: tuple[int, int]
    ) -> PlotConfig:
        config = self.plot_config.model_copy()
        for gc in DEFAULT_GRID_PLOTCONFIGS + self.grid_plotconfigs:
            if gc.matches(row, col, grid_size) and gc.plot_config:
                config.inherit_from(gc.plot_config, key="<<{+>}[~>]")
        return config

    def create_grid_cells(self) -> List[UORFCell]:
        all_pairs = {get_uorf_values(d) for d in self.plot_data}
        ax0, ax1 = self.uorf_axis_order
        unique_vals_0 = sorted({p[ax0] for p in all_pairs}, reverse=self.reverse_rows)
        unique_vals_1 = sorted({p[ax1] for p in all_pairs}, reverse=self.reverse_cols)
        cells = []
        for d in self.plot_data:
            p = get_uorf_values(d)
            cells.append(
                UORFCell(
                    row=unique_vals_0.index(p[ax0]),
                    col=unique_vals_1.index(p[ax1]),
                    uorf_values=p,
                    data=d,
                )
            )
        return cells

    def _draw_cell(self, ax, cell: UORFCell, grid_size: tuple[int, int]):
        if self.cell_kind == "slices":
            return self._draw_cell_slices(ax, cell, grid_size)
        cfg = self.get_plot_config_for_cell(cell.row, cell.col, grid_size)
        rescaler = cfg.rescaler or self.rescaler
        kw = dict(cfg.callstack_params.get("smooth_2d_params", {}))
        smooth(cell.data, ax=ax, rescaler=rescaler, force_dim=2, smooth_2d_params=kw)

    def _resolve_slice_lims(self, rescaler):
        """Auto-scale xlims/vlims to the union range across cells; cached per draw."""
        if not self._global_lims:
            xs = np.vstack([rescaler.fwd(d.x) for d in self.plot_data])
            ys = np.vstack([rescaler.fwd(d.y) for d in self.plot_data])
            self._global_lims.update({
                "x": [float(np.nanmin(xs)), float(np.nanmax(xs))],
                "y": [float(np.nanmin(ys)), float(np.nanmax(ys))],
            })
        return (
            _merge_lims(self.slice_xlims, self._global_lims["x"]),
            _merge_lims(self.slice_vlims, self._global_lims["y"]),
        )

    def _draw_cell_slices(self, ax, cell: UORFCell, grid_size: tuple[int, int]):
        d = cell.data
        # prefer plot_config's rescaler (else slice values read as raw fluo)
        rescaler = self.plot_config.rescaler or self.rescaler
        xlims, vlims = self._resolve_slice_lims(rescaler)
        X_lat = rescaler.fwd(d.x)
        Y_lat = rescaler.fwd(d.y)
        names = list(d.input_names) if d.input_names else ["x", "y"]
        # only border cells get ticks/labels; first smooth_1d owns label drawing
        is_bottom = cell.row == grid_size[0] - 1
        is_left = cell.col == 0
        common = dict(
            Y=Y_lat,
            output_name=d.output_name,
            rescaler=rescaler,
            ax=ax,
            xlims=xlims,
            vlims=vlims,
            res=self.slice_res,
            show_std=False,
            show_legend=False,
            knn_stats_params=dict(self.slice_knn_stats_params),
            lineplot_props=dict(self.slice_lineplot_props),
            show_theta=False,
            show_slopes=False,
        )
        for i, (axis, held_raw, cmap_name) in enumerate((
            ("x", self.slice_x_held_raw, self.slice_x_cmap),
            ("y", self.slice_y_held_raw, self.slice_y_cmap),
        )):
            if not held_raw:
                continue
            args = slice_panel_args(axis, X_lat, rescaler, held_raw, input_names=names)
            colors = plt.get_cmap(cmap_name)(
                np.linspace(self.slice_cmap_range[0], self.slice_cmap_range[1], len(held_raw))
            )
            smooth_1d(
                X=args["X"],
                input_names=args["input_names"],
                slices=args["slices_latent"],
                colors=list(colors),
                draw_xlabel=(i == 0) and is_bottom,
                draw_ylabel=(i == 0) and is_left,
                **common,
            )
        if not is_bottom:
            ax.set_xticklabels([])
        if not is_left:
            ax.set_yticklabels([])

    def _final_touches(self, fig, axes, grid_size, cells):
        legend_row = self.legend_row if self.legend_row >= 0 else grid_size[0] - 1
        legend_col = self.legend_col if self.legend_col >= 0 else grid_size[1] - 1

        def legend(cell: UORFCell, axis_idx: int) -> tuple[str, str]:
            value = cell.uorf_values[axis_idx]
            value_str = (
                self.uorf_value_to_str.get(value, str(value))
                if self.uorf_value_to_str
                else str(value)
            )
            side = "ERN" if axis_idx == 0 else "target"
            return side, value_str

        for r in range(grid_size[0]):
            ax = axes[r][legend_col]
            matches = [c for c in cells if c.row == r and c.col == legend_col]
            if matches:
                side, val = legend(matches[0], self.uorf_axis_order[0])
                ax.text(
                    -self.ylabel_padding,
                    0.5,
                    val,
                    fontsize=self.label_fontsize,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                )
                if r == grid_size[0] // 2:
                    ax.text(
                        -self.axis_title_padding - self.ylabel_padding,
                        0.5,
                        f"uORFs on {side} side",
                        fontsize=self.label_fontsize * 1.0,
                        ha="center",
                        va="bottom",
                        fontweight=self.label_fontweight,
                        color="black",
                        rotation=90,
                        rotation_mode="anchor",
                        transform=ax.transAxes,
                    )

        for c in range(grid_size[1]):
            ax = axes[legend_row][c]
            matches = [x for x in cells if x.row == legend_row and x.col == c]
            if matches:
                side, val = legend(matches[0], self.uorf_axis_order[1])
                ax.text(
                    0.5,
                    -self.xlabel_padding,
                    val,
                    fontsize=self.label_fontsize,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                )
                if c == grid_size[1] // 2:
                    ax.text(
                        0.5,
                        -self.axis_title_padding - self.xlabel_padding,
                        f"uORFs on {side} side",
                        fontsize=self.label_fontsize * 1.0,
                        ha="center",
                        va="top",
                        fontweight=self.label_fontweight,
                        color="black",
                        transform=ax.transAxes,
                    )

        for uorf_values in self.annotate:
            for cell in cells:
                if cell.uorf_values == uorf_values:
                    ax = axes[cell.row][cell.col]
                    am = self.annotation_margin
                    ax.add_artist(
                        Line2D(
                            [-am, 1 + am, 1 + am, -am, -am],
                            [-am, -am, 1 + am, 1 + am, -am],
                            transform=ax.transAxes,
                            **self.annotation_style,
                        )
                    )

        rmses = []
        for cell in cells:
            stats = cell.data.metadata.get("prediction_stats", {})
            if "rmse" in stats or "grid_rmse" in stats:
                rmse = stats.get("grid_rmse", stats.get("rmse"))
                rmses.append(rmse)
                if self.show_individual_rmse:
                    ax = axes[cell.row][cell.col]
                    ax.text(
                        0.5,
                        0.5,
                        f"{self.rmse_prefix}{rmse:.3f}",
                        fontsize=self.rmse_fontsize,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="black",
                    )

        if rmses and self.show_overall_rmse:
            fig.text(
                0.5,
                0.97,
                f"Overall RMSE: {np.mean(rmses):.3f}",
                fontsize=self.overall_rmse_fontsize,
                ha="center",
                va="center",
                color="black",
            )

    def _grid_lines(self, fig, axes, grid_size):
        margin = self.grid_line_margin / grid_size[0]
        first_pos = axes[0][0].get_position()
        last_pos = axes[grid_size[0] - 1][grid_size[1] - 1].get_position()
        x_min, x_max = first_pos.x0 - margin, last_pos.x1 + margin
        y_min, y_max = last_pos.y0 - margin, first_pos.y1 + margin

        x_intersections = []
        for col in range(1, grid_size[1]):
            x_pos = (axes[0][col - 1].get_position().x1 + axes[0][col].get_position().x0) / 2
            if self.grid_line_width > 0:
                fig.add_artist(
                    Line2D(
                        [x_pos, x_pos],
                        [y_min, y_max],
                        color=self.grid_line_color,
                        linewidth=self.grid_line_width,
                        clip_on=False,
                        zorder=0,
                        dash_capstyle="round",
                        linestyle=self.grid_line_style,
                    )
                )
            x_intersections.append(x_pos)

        for row in range(1, grid_size[0]):
            y_pos = (axes[row - 1][0].get_position().y0 + axes[row][0].get_position().y1) / 2
            if self.grid_line_width > 0:
                fig.add_artist(
                    Line2D(
                        [x_min, x_max],
                        [y_pos, y_pos],
                        color=self.grid_line_color,
                        linewidth=self.grid_line_width,
                        clip_on=False,
                        zorder=0,
                        dash_capstyle="round",
                        linestyle=self.grid_line_style,
                    )
                )
            for x_pos in x_intersections:
                half = self.grid_cross_length / 2
                fig.add_artist(
                    Line2D(
                        [x_pos - half, x_pos + half],
                        [y_pos, y_pos],
                        color=self.grid_cross_color,
                        linewidth=self.grid_cross_width,
                        clip_on=False,
                        zorder=1,
                        dash_capstyle="round",
                        linestyle=self.grid_cross_style,
                    )
                )
                fig.add_artist(
                    Line2D(
                        [x_pos, x_pos],
                        [y_pos - half, y_pos + half],
                        color=self.grid_cross_color,
                        linewidth=self.grid_cross_width,
                        clip_on=False,
                        zorder=1,
                        dash_capstyle="round",
                        linestyle=self.grid_cross_style,
                    )
                )

    def draw_into(self, ax) -> None:
        with mpl.rc_context(rc=self.plot_config.rc_context):
            cells = sorted(self.create_grid_cells(), key=lambda c: (c.row, c.col))
            if not cells:
                ax.text(
                    0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes
                )
                return
            grid_size = (max(c.row for c in cells) + 1, max(c.col for c in cells) + 1)
            sub = carve_bbox(ax)
            gs = sub.gridspec(grid_size[0], grid_size[1], wspace=self.wspace, hspace=self.hspace)
            fig = sub.fig
            axes = [
                [fig.add_subplot(gs[r, c]) for c in range(grid_size[1])]
                for r in range(grid_size[0])
            ]
            for cell in cells:
                with mpl.rc_context(
                    rc=self.get_plot_config_for_cell(cell.row, cell.col, grid_size).rc_context
                ):
                    try:
                        self._draw_cell(axes[cell.row][cell.col], cell, grid_size)
                    except Exception as e:
                        logger.error(f"Error drawing cell ({cell.row}, {cell.col}): {e}")
                        logger.exception(e)
            self._grid_lines(fig, axes, grid_size)
            self._final_touches(fig, axes, grid_size, cells)


def bundle_uorf_data(
    plot_data: List[PlotData],
    uorf_values: List[float] | None = None,
    same_xp: bool = False,
) -> List[List[PlotData]]:
    if uorf_values is None:
        uorf_values = [0, 5, 10, 20, 30, 40, 50, 60, 80]
    required = set(uorf_values)
    if same_xp:
        xp_groups: dict[str, list[PlotData]] = defaultdict(list)
        for d in plot_data:
            if "network" not in d.metadata:
                continue
            xp = d.metadata["network"].recipe.experiment.name
            if xp:
                xp_groups[xp].append(d)
        groups = list(xp_groups.values())
    else:
        groups = [plot_data]

    bundles: list[list[PlotData]] = []
    for group in groups:
        ern_groups: dict[str, list[PlotData]] = defaultdict(list)
        for d in group:
            ern_names = d.metadata.get("network_info", {}).get("ern_names", [])
            if ern_names:
                ern_groups[ern_names[0]].append(d)
        for ern_name, ern_data in ern_groups.items():
            seen = {
                tuple(v)
                for d in ern_data
                for v in d.metadata["network_info"].get("uorf_values", [])
            }
            missing = {(a, b) for a in required for b in required} - seen
            if missing:
                logger.debug(f"ERN {ern_name} missing uORF combinations: {missing}")
                continue
            bundles.append(ern_data)
    logger.info(f"Found {len(bundles)} complete uORF matrices")
    return bundles


def get_ern_info(bundle: List[PlotData]) -> Dict[str, str]:
    if not bundle:
        return {}
    info = bundle[0].metadata.get("network_info", {})
    ern_names = info.get("ern_names", [])
    if not ern_names:
        return {}
    return {
        "ern_name": ern_names[0],
        "experiment": bundle[0].metadata.get("experiment", {}).get("name", "unknown"),
    }


def get_uorf_values(data: PlotData) -> Tuple[float, float]:
    return tuple(data.metadata["network_info"]["uorf_values"][0])


def extract_uorf_info(network):
    uvals = network.network_info.get("uorf_values", None)
    if not uvals:
        return None
    return {"uorf_values": uvals[0], "ern_name": network.network_info.get("ern_names", None)[0]}
