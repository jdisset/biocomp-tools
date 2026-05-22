# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import matplotlib as mpl
from dracon.draconstructor import resolve_all_lazy
from typing import TypeVar, Union
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from biocomp.utils import PartialFunction
from biocomp.plotutils import GridLayout
from biocomptools.toollib.plot import PlotTask, PlotConfig, BiocompPlotFigure, load_default_plotconf
from biocomp.plotutils import PlotData
from biocomptools.logging_config import get_logger
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field
import numpy as np

logger = get_logger(__name__)


def dummy_plot(*args, **kwargs):
    print("Dummy plot called with", args, kwargs)


dummy_partial_func = PartialFunction(func=dummy_plot, kwargs={})

# TODO: same tick_proms and label_props system as the colorbar but for the setup_transformed_axis function


class GridPlotConfig(BaseModel):
    """Configuration for specific grid positions"""

    row: Optional[int] = None  # None means all rows
    col: Optional[int] = None  # None means all columns
    plot_config: Optional[PlotConfig] = None
    # post_plot_callbacks: List[PartialFunction] = []

    def matches(self, r: int, c: int, grid_size: tuple[int, int]) -> bool:
        """Check if this config applies to the given grid position"""
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
                # no spines:
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
                    "heatmap_params": {
                        "contours": None,
                    },
                },
            },
        ),
    ),
    GridPlotConfig(
        col=0,
        plot_config=PlotConfig(
            rc_context={
                "ytick.left": True,
                "axes.spines.left": True,
                # hide tick labels:
                "xtick.labelbottom": False,
                "xtick.labeltop": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
            },
        ),
    ),
    GridPlotConfig(
        row=-1,
        plot_config=PlotConfig(
            rc_context={
                "xtick.bottom": True,
                "axes.spines.bottom": True,
                # hide tick labels:
                "xtick.labelbottom": False,
                "xtick.labeltop": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
            },
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
                        "tick_props": {
                            "labelsize": 15,
                            "pad": 8,
                            "length": 10,
                            "width": 1,
                        },
                        "label_position": "left",
                        "label_props": {
                            "size": 15,
                        },
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


class UORFCell:
    """Represents a cell in the uORF matrix with its data and configuration"""

    def __init__(self, row: int, col: int, uorf_values: Tuple[float, float], data: PlotData):
        self.row = row
        self.col = col
        self.uorf_values = uorf_values
        self.data = data

    def __repr__(self):
        return f"UORFCell(row={self.row}, col={self.col}, uorf_values={self.uorf_values})"


T = TypeVar("T")

ANNOTATION_STYLE = {
    "color": "#F76665",
    "linewidth": 2.5,
    "clip_on": False,
    "zorder": 200,
    "linestyle": "--",
}


class UORFMatrixFigure(BiocompPlotFigure):
    """A figure that automatically distributes data across a matrix based on uORF values"""

    # Input data as list of PlotData objects
    plot_data: List[PlotData]

    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)

    uorf_axis_order: tuple[int, int] = (1, 0)
    plot_only: int = -1
    grid_plotconfigs: List[GridPlotConfig] = []

    reverse_rows: bool = True
    reverse_cols: bool = False

    uorf_value_to_str: Optional[Dict[int, str]] = DEFAULT_UORF_VALUE_TO_STR

    draw_colorbar: bool = True
    colorbar_col: int = -1

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

    wspace: float = 0.25
    hspace: float = 0.25

    xtitle: Optional[str] = None
    ytitle: Optional[str] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        resolve_all_lazy(self.plot_config)
        for grid_conf in DEFAULT_GRID_PLOTCONFIGS + self.grid_plotconfigs:
            resolve_all_lazy(grid_conf.plot_config)

    def get_plot_config_for_cell(
        self, row: int, col: int, grid_size: tuple[int, int]
    ) -> PlotConfig:
        """Get the appropriate plot configuration for a specific grid position"""

        # start with base config
        config = self.plot_config.model_copy()

        for grid_conf in DEFAULT_GRID_PLOTCONFIGS + self.grid_plotconfigs:
            if grid_conf.matches(row, col, grid_size):
                if grid_conf.plot_config:
                    config.inherit_from(grid_conf.plot_config, key="<<{+>}[~>]")

        return config

    def create_grid_cells(self) -> List[UORFCell]:
        """Create matrix cells with their corresponding data"""
        all_pairs = {get_uorf_values(data) for data in self.plot_data}

        ax0 = self.uorf_axis_order[0]
        ax1 = self.uorf_axis_order[1]
        # create sorted lists of unique values for rows and columns
        unique_vals_0 = sorted(set(pair[ax0] for pair in all_pairs), reverse=self.reverse_rows)
        unique_vals_1 = sorted(set(pair[ax1] for pair in all_pairs), reverse=self.reverse_cols)

        cells = []
        for data in self.plot_data:
            pair = get_uorf_values(data)
            row = unique_vals_0.index(pair[ax0])
            col = unique_vals_1.index(pair[ax1])
            cells.append(UORFCell(row, col, pair, data))

        return cells

    def create_plot_task_for_cell(self, cell: UORFCell, ax, grid_size: tuple[int, int]) -> PlotTask:
        """Create a plot task for a specific cell"""
        task = PlotTask(
            plot_config=self.get_plot_config_for_cell(cell.row, cell.col, grid_size),
            plot_method=PartialFunction(func="biocomp.plotutils.smooth", kwargs={"force_dim": 2}),
        )

        task._ax = ax
        task.plot_method.kwargs["plot_data"] = cell.data

        # for config in DEFAULT_GRID_PLOTCONFIGS + self.grid_plotconfigs:
        #     if config.matches(cell.row, cell.col, grid_size):
        #         task.post_plot_callbacks += config.post_plot_callbacks

        return task

    def final_touches(self, figax, grid_size: tuple[int, int], cells: List[UORFCell]):
        """Add final cosmetic touches to the figure including labels and legends"""

        # convert negative indices to positive
        legend_row = self.legend_row if self.legend_row >= 0 else grid_size[0] - 1
        legend_col = self.legend_col if self.legend_col >= 0 else grid_size[1] - 1

        def get_uorf_legend(cell: UORFCell, axis_idx: int) -> tuple[str, str]:
            """Extract side and value information from uORF values"""
            value = cell.uorf_values[axis_idx]
            value_str = self.uorf_value_to_str.get(value, str(value))
            side = "ERN" if axis_idx == 0 else "target"
            return side, value_str

        # add column legends
        for r in range(grid_size[0]):
            ax = figax.ax[r][legend_col]
            matching_cells = [c for c in cells if c.row == r and c.col == legend_col]

            if matching_cells:
                cell = matching_cells[0]
                side, u_value = get_uorf_legend(cell, self.uorf_axis_order[0])

                # add value label
                ax.text(
                    -self.ylabel_padding,
                    0.5,
                    u_value,
                    fontsize=self.label_fontsize,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                )

                # add side label for the last row
                if r == grid_size[0] // 2:
                    ax.text(
                        -self.axis_title_padding - self.ylabel_padding,
                        0.5,
                        # f'uORFs on {side} side\n$\\rightarrow$',
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

        # add row legends
        for col in range(grid_size[1]):
            ax = figax.ax[legend_row][col]
            matching_cells = [c for c in cells if c.row == legend_row and c.col == col]

            if matching_cells:
                cell = matching_cells[0]
                side, u_value = get_uorf_legend(cell, self.uorf_axis_order[1])

                # add value label
                ax.text(
                    0.5,
                    -self.xlabel_padding,
                    u_value,
                    fontsize=self.label_fontsize,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                )

                # add side label for the first column
                if col == grid_size[1] // 2:
                    ax.text(
                        0.5,
                        -self.axis_title_padding - self.xlabel_padding,
                        # f'$\\rightarrow$\nuORFs on {side} side',
                        f"uORFs on {side} side",
                        fontsize=self.label_fontsize * 1.0,
                        ha="center",
                        va="top",
                        fontweight=self.label_fontweight,
                        color="black",
                        transform=ax.transAxes,
                    )

        # dashed line around the annotated cells
        for uorf_values in self.annotate:
            for cell in cells:
                if cell.uorf_values == uorf_values:
                    ax = figax.ax[cell.row][cell.col]
                    ax.add_artist(
                        Line2D(
                            # [0, 1, 1, 0, 0],
                            # [0, 0, 1, 1, 0],
                            [
                                -self.annotation_margin,
                                1 + self.annotation_margin,
                                1 + self.annotation_margin,
                                -self.annotation_margin,
                                -self.annotation_margin,
                            ],
                            [
                                -self.annotation_margin,
                                -self.annotation_margin,
                                1 + self.annotation_margin,
                                1 + self.annotation_margin,
                                -self.annotation_margin,
                            ],
                            transform=ax.transAxes,
                            **self.annotation_style,
                        )
                    )

        # add rmse annotations
        rmses = []
        for cell in cells:
            if (
                "prediction_stats" in cell.data.metadata
                and "rmse" in cell.data.metadata["prediction_stats"]
            ):
                rmse = cell.data.metadata["prediction_stats"]["grid_rmse"]
                rmses.append(rmse)
                if self.show_individual_rmse:
                    ax = figax.ax[cell.row][cell.col]
                    ax.text(
                        0.5,
                        0.5,
                        f"RMSE: {rmse:.3f}",
                        fontsize=self.rmse_fontsize,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="black",
                    )

        if rmses and self.show_overall_rmse:
            overall_rmse = np.mean(rmses)
            fig = figax.figure
            fig.text(
                0.5,
                0.97,
                f"Overall RMSE: {overall_rmse:.3f}",
                fontsize=self.rmse_fontsize,
                ha="center",
                va="center",
                color="black",
            )

    def add_grid_lines(self, figax, grid_size: tuple[int, int]):
        """Add thin lines between subplots to show the grid structure"""
        fig = figax.figure
        # as a proportion of a single subplot
        margin_dist = self.grid_line_margin / grid_size[0]
        # get the position of the first and last subplot to determine boundaries
        first_ax = figax.ax[0][0]
        last_ax = figax.ax[grid_size[0] - 1][grid_size[1] - 1]
        # get the bounds of the entire subplot area
        first_pos = first_ax.get_position()
        last_pos = last_ax.get_position()
        x_min = first_pos.x0 - margin_dist
        x_max = last_pos.x1 + margin_dist
        y_min = last_pos.y0 - margin_dist
        y_max = first_pos.y1 + margin_dist

        intersection_points = []

        # add vertical lines between columns
        for col in range(1, grid_size[1]):
            ax_left = figax.ax[0][col - 1]
            ax_right = figax.ax[0][col]
            x_pos = (ax_left.get_position().x1 + ax_right.get_position().x0) / 2
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
            intersection_points.append((x_pos, None))

        # and horizontal lines between rows
        for row in range(1, grid_size[0]):
            ax_top = figax.ax[row - 1][0]
            ax_bottom = figax.ax[row][0]
            y_pos = (ax_top.get_position().y0 + ax_bottom.get_position().y1) / 2
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

            # Add crosses at each intersection
            for x_pos, _ in intersection_points:
                fig.add_artist(
                    Line2D(
                        [x_pos - self.grid_cross_length / 2, x_pos + self.grid_cross_length / 2],
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
                        [y_pos - self.grid_cross_length / 2, y_pos + self.grid_cross_length / 2],
                        color=self.grid_cross_color,
                        linewidth=self.grid_cross_width,
                        clip_on=False,
                        zorder=1,
                        dash_capstyle="round",
                        linestyle=self.grid_cross_style,
                    )
                )

    def run(self, overwrite=True):
        logger.info(f"Running uORF matrix figure with {len(self.plot_data)} data sets")
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        with mpl.rc_context(rc=self.plot_config.rc_context):
            cells = self.create_grid_cells()
            cells = sorted(cells, key=lambda c: (c.row, c.col))

            rows = max(cell.row for cell in cells) + 1
            cols = max(cell.col for cell in cells) + 1
            grid_size = (rows, cols)

            self.figure_spec.layout = GridLayout(
                rows=rows,
                cols=cols,
                axes_size=(1, 1),
                wspace=self.wspace,
                hspace=self.hspace,
            )

            figax = self.figure_spec.make_figure()

            metadata = dict(self.figure_spec.metadata) if self.figure_spec.metadata else {}
            metadata["plot_tasks"] = []
            for i, cell in enumerate(cells):
                ax = figax.ax[cell.row][cell.col]
                task = self.create_plot_task_for_cell(cell, ax, grid_size)

                if self.plot_only > 0 and i >= self.plot_only:
                    task.plot_method = dummy_partial_func

                try:
                    pt_metadata = task.run()
                    metadata["plot_tasks"].append(pt_metadata)
                    logger.info(f"Executed plot task for cell {cell} ({i + 1}/{len(cells)})")
                except Exception as e:
                    logger.error(f"Error executing plot task {task} for cell {cell}: {e}")
                    logger.exception(e)
                    continue

            # collect and serialize grid data from all tasks into one blob
            all_grid_data = []
            for pt_meta in metadata.get("plot_tasks", []):
                gd = pt_meta.pop("grid_data", None)
                if gd:
                    all_grid_data.extend(gd)
            if all_grid_data:
                from jeanplot.data.grid import grid_data_to_b64

                metadata["grid_data"] = grid_data_to_b64(all_grid_data)

            self.add_grid_lines(figax, grid_size)
            self.final_touches(figax, grid_size, cells)
            self.figure_spec.metadata = metadata
            self.figure_spec.finalize(figax)


## {{{                    --     uorf bundle helper     --


def bundle_uorf_data(
    plot_data: List[PlotData],
    uorf_values: List[float] | None = None,
    same_xp: bool = False,
) -> List[List[PlotData]]:
    """
    Analyze a list of PlotData and create bundles for uORF matrix figures.
    Each bundle contains data for a complete matrix of uORF combinations.

    Args:
        plot_data: List of PlotData objects to analyze
        uorf_values: Required uORF values to consider a matrix complete
        same_xp: If True, only bundle data from the same experiment

    Returns:
        List of PlotData bundles, where each bundle can be used to create a uORF matrix figure
    """
    if uorf_values is None:
        uorf_values = [0, 5, 10, 20, 30, 40, 50, 60, 80]
    required_values = set(uorf_values)

    # Group data by experiment if needed
    if same_xp:
        xp_groups = defaultdict(list)
        for data in plot_data:
            if "network" not in data.metadata:
                continue
            xp_name = data.metadata["network"].recipe.experiment.name
            if xp_name:
                xp_groups[xp_name].append(data)

        data_groups = list(xp_groups.values())
    else:
        data_groups = [plot_data]

    bundles = []

    for group in data_groups:
        # find ERNs and their associated data
        ern_groups = defaultdict(list)

        for data in group:
            network_info = data.metadata.get("network_info", {})
            ern_names = network_info.get("ern_names", [])

            if not ern_names:
                continue

            # group by first ERN name for now
            ern_name = ern_names[0]
            ern_groups[ern_name].append(data)

        # process each ERN group
        for ern_name, ern_data in ern_groups.items():
            value_sets = set()
            for data in ern_data:
                uorf_vals = data.metadata["network_info"].get("uorf_values", [])
                for vals in uorf_vals:
                    value_sets.add(tuple(vals))

            # check if we have all combinations of required values
            required_combinations = {(v1, v2) for v1 in required_values for v2 in required_values}

            missing = required_combinations - value_sets
            if missing:
                logger.debug(
                    f"ERN {ern_name} missing uORF combinations: {missing}"
                    f" (from {'same experiment' if same_xp else 'any experiment'})"
                )
                logger.debug(f"Found uORF combinations: {value_sets}")
                continue

            # This is a complete set - let's add it to our bundles
            bundles.append(ern_data)

    logger.info(
        f"Found {len(bundles)} complete uORF matrices"
        f" (from {'same experiment' if same_xp else 'any experiment'})"
    )

    return bundles


def get_ern_info(bundle: List[PlotData]) -> Dict[str, str]:
    """Helper to get ERN information from a bundle"""
    if not bundle:
        return {}

    network_info = bundle[0].metadata.get("network_info", {})
    ern_names = network_info.get("ern_names", [])
    if not ern_names:
        return {}

    return {
        "ern_name": ern_names[0],
        "experiment": bundle[0].metadata.get("experiment", {}).get("name", "unknown"),
    }


def get_uorf_values(data: PlotData) -> Tuple[float, float]:
    # (ERN, target)
    return tuple(data.metadata["network_info"]["uorf_values"][0])


def extract_uorf_info(network):
    uvals = network.network_info.get("uorf_values", None)
    if not uvals:
        return None
    return {
        "uorf_values": uvals[0],
        "ern_name": network.network_info.get("ern_names", None)[0],
    }


##────────────────────────────────────────────────────────────────────────────}}}
