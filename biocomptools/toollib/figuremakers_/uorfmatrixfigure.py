import matplotlib as mpl
from dracon.draconstructor import resolve_all_lazy
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from biocomp.utils import PartialFunction, ArbitraryModel
from biocomp.plotutils import FigureSpec, SimpleLayout
from biocomptools.toollib.plot import PlotTask, PlotConfig, Figure, load_default_plotconf
from biocomp.plotutils import PlotData
from biocomptools.logging_config import get_logger
from biocomptools.toollib.datasources import DataSource
from pydantic import BaseModel, Field, BeforeValidator
import numpy as np

logger = get_logger(__name__)


class GridPlotConfig(BaseModel):
    """Configuration for specific grid positions"""

    row: Optional[int] = None  # None means all rows
    col: Optional[int] = None  # None means all columns
    plot_config: PlotConfig

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


DEFAULT_GRID_PLOTCONFIGS = [
    GridPlotConfig(
        plot_config=PlotConfig(
            callstack_params={
                "smooth_2d_params": {
                    "draw_colorbar": False,
                }
            }
        ),
    ),
    GridPlotConfig(
        col=-1,
        row=-1,
        plot_config=PlotConfig(
            callstack_params={
                "smooth_2d_params": {
                    "draw_colorbar": True,
                    "colorbar_params": {
                        "size": [0.05, 0.85],
                        "position": [1.05, 0.075],
                    },
                }
            }
        ),
    ),
]


class UORFCell:
    """Represents a cell in the uORF matrix with its data and configuration"""

    def __init__(self, row: int, col: int, uorf_values: Tuple[float, float], data: PlotData):
        self.row = row
        self.col = col
        self.uorf_values = uorf_values
        self.data = data

    def __repr__(self):
        return f"UORFCell(row={self.row}, col={self.col}, uorf_values={self.uorf_values})"


class uORFMatrixFigure(Figure):
    """A figure that automatically distributes data across a matrix based on uORF values"""

    # Input data as list of PlotData objects
    plot_data: List[PlotData]

    # Configuration
    uorf_axis_order: tuple[int, int] = (0, 1)
    plot_only: int = -1
    plot_template: Optional[PlotTask] = None
    grid_plotconfigs: List[GridPlotConfig] = DEFAULT_GRID_PLOTCONFIGS

    # Layout configuration
    draw_colorbar: bool = True
    colorbar_col: int = -1

    def get_plot_config_for_cell(
        self, row: int, col: int, grid_size: tuple[int, int]
    ) -> PlotConfig:
        """Get the appropriate plot configuration for a specific grid position"""
        # Start with base config

        resolve_all_lazy(self.plot_config)
        config = self.plot_config.model_copy()

        # Apply grid-specific configs in order
        for grid_conf in self.grid_plotconfigs:
            if grid_conf.matches(row, col, grid_size):
                config.inherit_from(grid_conf.plot_config, key="<<{+>}")

        return config

    def partition_data(self, data: PlotData) -> Dict[Tuple[float, float], List[int]]:
        """Group data points by their uORF values"""
        network_info = data.metadata['network_info']
        uorf_values = np.array(network_info['uorf_values'])

        if len(uorf_values) == 0:
            raise ValueError("No uORF values found in network info")

        value_pairs = [
            (v[self.uorf_axis_order[0]], v[self.uorf_axis_order[1]]) for v in uorf_values
        ]
        unique_pairs = list(set(value_pairs))

        grouped_indices = {pair: [] for pair in unique_pairs}
        for idx, pair in enumerate(value_pairs):
            grouped_indices[pair].extend(range(len(data.x)))

        return grouped_indices

    def create_cell_data(
        self, base_data: PlotData, indices: List[int], uorf_values: Tuple[float, float]
    ) -> PlotData:
        """Create a new PlotData object for a specific cell"""
        x = base_data.x[indices] if indices else base_data.x[[]]
        y = base_data.y[indices] if indices else base_data.y[[]]

        metadata = base_data.metadata.copy()
        metadata['uorf_values'] = uorf_values

        return PlotData(
            xval=x,
            yval=y,
            input_names=base_data.input_names,
            output_name=base_data.output_name,
            metadata=metadata,
        )

    def create_grid_cells(self) -> List[UORFCell]:
        """Create matrix cells with their corresponding data"""

        all_pairs = set()
        for data in self.plot_data:
            pairs = self.partition_data(data).keys()
            all_pairs.update(pairs)

        unique_vals_0 = sorted(set(pair[0] for pair in all_pairs))
        unique_vals_1 = sorted(set(pair[1] for pair in all_pairs))

        cells = []
        for pair in all_pairs:
            row = unique_vals_0.index(pair[0])
            col = unique_vals_1.index(pair[1])

            combined_data = []
            for data in self.plot_data:
                grouped = self.partition_data(data)
                if pair in grouped:
                    cell_data = self.create_cell_data(data, grouped[pair], pair)
                    combined_data.append(cell_data)

            if combined_data:
                cells.append(UORFCell(row, col, pair, combined_data[0]))

        return cells

    def create_plot_task_for_cell(self, cell: UORFCell, ax, grid_size: tuple[int, int]) -> PlotTask:
        """Create a plot task for a specific cell"""
        if self.plot_template:
            task = self.plot_template.model_copy(deep=True)
        else:
            task = PlotTask(
                plot_config=self.get_plot_config_for_cell(cell.row, cell.col, grid_size),
                plot_method=PartialFunction(
                    func='biocomp.plotutils.smooth', kwargs={'force_dim': 2}
                ),
            )

        task.plot_config.inherit_from(
            self.get_plot_config_for_cell(cell.row, cell.col, grid_size), key="<<{+>}"
        )

        task.ax = ax
        task.plot_method.kwargs['plot_data'] = cell.data

        return task

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        print(f"uORFMatrixFigure initialized with {len(self.plot_data)} data sets")

    def run(self, overwrite=True):
        """Execute the figure creation"""
        print(f"Running uORF matrix figure with {len(self.plot_data)} data sets")
        if not overwrite and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        with mpl.rc_context(rc=self.plot_config.rc_context):
            # Take first dataset as example
            d = self.plot_data[0]
            print(f"Shape of data: x={d.x.shape}, y={d.y.shape}")
            print(f"Data range: x=[{d.x.min()}, {d.x.max()}], y=[{d.y.min()}, {d.y.max()}]")
            print(f"uORF values: {d.metadata['network_info']['uorf_values']}")

            cells = self.create_grid_cells()

            rows = max(cell.row for cell in cells) + 1
            cols = max(cell.col for cell in cells) + 1
            grid_size = (rows, cols)

            print(f"Grid size: {grid_size}")

            self.figure_spec.layout = SimpleLayout(rows=rows, cols=cols)

            figax = self.figure_spec.make_figure()

            for i, cell in enumerate(cells):
                if self.plot_only > 0 and i >= self.plot_only:
                    break

                ax = figax.ax[cell.row][cell.col]
                task = self.create_plot_task_for_cell(cell, ax, grid_size)

                try:
                    task.run()
                    print(f"Executed plot task for cell {cell} ({i + 1}/{len(cells)})")
                except Exception as e:
                    logger.error(f"Error executing plot task {task} for cell {cell}: {e}")
                    logger.exception(e)
                    continue

        self.figure_spec.finalize(figax)


## {{{                    --     uorf bundle helper     --


def bundle_uorf_data(
    plot_data: List[PlotData],
    uorf_values: List[float] = [0, 5, 10, 20, 30, 40, 50, 60, 80],
    same_xp: bool = True,
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
    # Convert uorf_values to set for faster lookup
    required_values = set(uorf_values)

    # Group data by experiment if needed
    if same_xp:
        xp_groups = defaultdict(list)
        for data in plot_data:
            xp_name = data.metadata.get('experiment', {}).get('name')
            if xp_name:
                xp_groups[xp_name].append(data)
        data_groups = list(xp_groups.values())
    else:
        data_groups = [plot_data]

    bundles = []

    for group in data_groups:
        # Find ERNs and their associated data
        ern_groups = defaultdict(list)

        for data in group:
            network_info = data.metadata.get('network_info', {})
            ern_names = network_info.get('ern_names', [])

            if not ern_names:
                continue

            # Group by first ERN name for now
            ern_name = ern_names[0]
            ern_groups[ern_name].append(data)

        # Process each ERN group
        for ern_name, ern_data in ern_groups.items():
            # Check the uORF values for this group
            value_sets = set()
            for data in ern_data:
                uorf_vals = data.metadata['network_info'].get('uorf_values', [])
                for vals in uorf_vals:
                    value_sets.add(tuple(sorted(vals)))

            # Check if we have all combinations of required values
            required_combinations = {(v1, v2) for v1 in required_values for v2 in required_values}

            # Identify missing combinations
            missing = required_combinations - value_sets
            if missing:
                logger.debug(
                    f"ERN {ern_name} missing uORF combinations: {missing}"
                    f" (from {'same experiment' if same_xp else 'any experiment'})"
                )
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

    network_info = bundle[0].metadata.get('network_info', {})
    ern_names = network_info.get('ern_names', [])
    if not ern_names:
        return {}

    return {
        'ern_name': ern_names[0],
        'experiment': bundle[0].metadata.get('experiment', {}).get('name', 'unknown'),
    }


##────────────────────────────────────────────────────────────────────────────}}}
