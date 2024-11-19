from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, Union, Annotated, Tuple, Optional, List

import biocomp.utils as ut
from biocomp.plotutils import FigureSpec, PlotData, SimpleLayout

import biocomptools.toollib.plot as pl
from biocomptools.toollib.resolvable import Resolvable, make_resolvable
from biocomptools.toollib.inheritable import merged
from biocomptools.toollib.plot import (
    FigureTask,
    ListOrSingle,
    FigureMaker,
    ArbitraryTargetModel,
    PlotTask,
    ResolvablePlotConfig,
    ResolvableOr,
    PlotConfig
)

# todo, maybe it should be a plot, not a figure maker?
# not sure..
# - Make parallel with multifigure?
# - Use a after_task + ray.bind to have parallel things + able to compose tasks
# - Turn into plot file output


class GridPlotConfig(ArbitraryTargetModel):
    """simply a plot config with row and col attributes"""

    row: Optional[int] = None  # None means all rows
    col: Optional[int] = None  # None means all columns

    plot_config: ResolvablePlotConfig = {}

    def matches(self, r: int, c: int, grid_size: tuple[int, int]):
        if self.row is not None and self.row < 0:
            row = grid_size[0] + self.row
        else:
            row = self.row
        if self.col is not None and self.col < 0:
            col = grid_size[1] + self.col
        else:
            col = self.col
        return (row is None or row == r) and (col is None or col == c)


ResolvableGridPlotConfig = Annotated[ResolvableOr[GridPlotConfig], *pl.resolvable(PlotConfig)]


class uORFMatrixFigure(FigureMaker):

    uorf_axis_order: tuple[int, int] = (0, 1)
    plot_only: int = -1  # -1 means all plots. For debugging.

    grid_plotconfigs: List[ResolvableGridPlotConfig] = []

    legend_row: int = -1
    legend_col: int = 0

    draw_colorbar: bool = False
    colorbar_row: int = 0
    colorbar_col: int = -1

    uorf_label_fontsize: int = 16
    uorf_label_padding: float = 0.325
    label_fontweight: int = 1000

    plot_func: str = 'biocomp.plotutils.smooth'

    def model_post_init(self, *_):
        super().model_post_init()

        LEGEND_ROW_CONF = GridPlotConfig(
            row=self.legend_row,
            plot_config={
                'callstack_params': {
                    'smooth_2d_params': {
                        'draw_xlabel': True,
                    }
                },
                'rc_context': {
                    'xtick.labelbottom': True,
                },
            },
        )

        LEGEND_COL_CONF = GridPlotConfig(
            col=self.legend_col,
            plot_config={
                'callstack_params': {
                    'smooth_2d_params': {
                        'draw_ylabel': True,
                    }
                },
                'rc_context': {
                    'ytick.labelleft': True,
                },
            },
        )

        UORF_COLORBAR_SPECIALIZATION = GridPlotConfig(
            _target_=None,
            row=self.colorbar_row,
            col=self.colorbar_col,
            plot_config={
                'callstack_params': {
                    'smooth_2d_params': {
                        'draw_colorbar': True,
                        'colorbar_params': {
                            'position': (1.5, -0.5),
                            'size': (0.1, 1.0),
                            'label_props': {
                                'fontsize': 12,
                                'labelpad': 3,
                            },
                            'orientation': 'vertical',
                            'label_position': 'left',
                        },
                    }
                },
            },
        )

        self.grid_plotconfigs = [
            make_resolvable(value=LEGEND_COL_CONF),
            make_resolvable(value=LEGEND_ROW_CONF),
            make_resolvable(value=UORF_COLORBAR_SPECIALIZATION),
            *self.grid_plotconfigs,
        ]

    def make_grid(self, data: ListOrSingle[PlotData]):
        dlist = as_list(data)
        for d in dlist:
            assert 'network_info' in d.metadata, f"Missing network_info in {d.metadata['name']}"

        grid_coords = get_uorf_grid(
            [d.metadata['network_info'] for d in dlist], *self.uorf_axis_order
        )
        grid_size = np.max(grid_coords, axis=0) + 1

        layout = make_resolvable(
            value=SimpleLayout(
                rows=grid_size[0],
                cols=grid_size[1],
                axes_size=(2.5, 2.5),
                kwargs={'sharex': 'all', 'sharey': 'all'},
            )
        )

        self.figure_spec = merged(
            self.figure_spec,
            {'layout': {'_target_': 'biocomp.plotutils.SimpleLayout', **layout.config}},
        )

        grid_coords_to_net_info = {
            (r, c): d.metadata['network_info'] for (r, c), d in zip(grid_coords, dlist)
        }

        return grid_coords, grid_size, grid_coords_to_net_info

    def make_tasks(self, data: ListOrSingle[PlotData]) -> list[FigureTask]:

        grid_coords, grid_size, grid_coords_to_net_info = self.make_grid(data)

        self.plot_tasks = []
        for i, (r, c) in enumerate(grid_coords):
            # create a plotconfig for these coordinates by
            # merging all the specialized plot configs
            task_plotconf = self.plot_config
            for gpc in self.grid_plotconfigs:
                resolved_gpc = rs.resolved(gpc)
                if resolved_gpc.matches(r, c, grid_size):
                    # task_plotconf = merged(task_plotconf, resolved_gpc.plot_config)
                    task_plotconf = merged(resolved_gpc.plot_config, task_plotconf)

            t = PlotTask(
                plot_config=task_plotconf,
                plot_method=ut.PartialFunction(
                    func=self.plot_func,
                    kwargs={
                        'ax': '${ptask: context.figure.ax.' + f'{r}.{c}' + '}',
                        'plot_data': '${ftask: flat_data.' + str(i) + '}',
                    },
                ),
            )

            self.plot_tasks.append(t)

        if self.plot_only > 0:
            self.plot_tasks = self.plot_tasks[: self.plot_only]

        self.n_plot_tasks = len(self.plot_tasks)

        def uorf_legend(s):
            value = s.split(': ')[1].split(' ')[0]
            side = s.split(':')[0].split(' ')[1]
            if side == 'REC':
                side = 'target'
            if value == 'No':
                value = 'none'
            return side, value

        legend_row = self.legend_row if self.legend_row >= 0 else grid_size[0] - 1
        legend_col = self.legend_col if self.legend_col >= 0 else grid_size[1] - 1

        # now we add an extra_task to write legend on the sides
        def final_touches(
            figure_task,
            figax,
            grid_size=grid_size,
            grid_coords_to_net_info=grid_coords_to_net_info,
            legend_row=legend_row,
            legend_col=legend_col,
        ):
            fig = figax.figure
            # column legend:
            for r in range(grid_size[0]):
                ax = figax.ax[r][legend_col]
                net_info = grid_coords_to_net_info.get((r, legend_col), None)
                side, u_value = uorf_legend(net_info['uorf_names'][self.uorf_axis_order[0]])
                if net_info is not None:
                    ax.text(
                        -self.uorf_label_padding,
                        0.4,
                        u_value,
                        fontsize=self.uorf_label_fontsize,
                        ha='right',
                        va='center',
                    )

                if r == grid_size[0] - 1:
                    ax.text(
                        -self.uorf_label_padding,
                        0.025,
                        f'x uORFs\n{side} side\n→',  # ↑
                        fontsize=self.uorf_label_fontsize * 1.0,
                        ha='right',
                        va='center',
                        fontweight=self.label_fontweight,
                        color='black',
                        rotation=90,
                    )

            for c in range(grid_size[1]):
                ax = figax.ax[legend_row][c]
                net_info = grid_coords_to_net_info.get((legend_row, c), None)
                side, u_value = uorf_legend(net_info['uorf_names'][self.uorf_axis_order[1]])
                if net_info is not None:
                    ax.text(
                        0.4,
                        -self.uorf_label_padding,
                        u_value,
                        fontsize=self.uorf_label_fontsize,
                        ha='center',
                        va='center',
                    )

                if c == 0:
                    ax.text(
                        0.1,
                        -self.uorf_label_padding,
                        f'→\nx uORFs\n{side} side',
                        fontsize=self.uorf_label_fontsize * 1.0,
                        ha='right',
                        va='center',
                        fontweight=self.label_fontweight,
                        color='black',
                    )

        ftask = self.spawn_figure_task(data, {'figure_makers': ['uorf_matrix']})
        ftask.extra_tasks.append(ut.PartialFunction(func=final_touches))
        return [ftask]
