## {{{                          --     imports     --
from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, Union, Annotated, Tuple, Optional, List

import biocomp.utils as ut
from biocomp.plotutils import FigureSpec, PlotData, SimpleLayout

from biocomp.utils import PartialFunction, flatten, as_list
import biocomptools.toollib.old_plot as pl
from biocomptools.toollib.old_resolvable import Resolvable, make_resolvable, resolved
from biocomptools.toollib.old_inheritable import (
    merged,
    InheritableAttrsModel,
    merged_into,
    merged_into_container,
)
from biocomptools.toollib.old_plot import (
    FigureTask,
    ListOrSingle,
    FigureMaker,
    ArbitraryTargetModel,
    PlotTask,
    ResolvablePlotConfig,
    ResolvableFigureMaker,
    ResolvableOr,
    PlotConfig,
    resolvers,
    make_resolvers,
    with_context,
    INHERIT_ATTRS,
)

import numpy as np
import logging

figlog = logging.getLogger('biocomptools.biocomplot.figure')
figlog.setLevel(logging.INFO)
##────────────────────────────────────────────────────────────────────────────}}}


## {{{                       --     SingleFigure     --
class SingleFigure(FigureMaker):
    """A figure maker that will spawn a figure task for each data group"""

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        # return [self.spawn_figure_task(d, {'figure_makers': ['single']}) for d in as_list(data)]
        return [self.spawn_figure_task(data, {'figure_makers': ['single']})]
##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     ForEachData     --
class ForEachData(FigureMaker, InheritableAttrsModel):
    """A figure maker that will call another figure maker for each data item"""

    figure_maker: ResolvableFigureMaker = {}

    _inherit = {'figure_maker': INHERIT_ATTRS['FigureMaker']['FigureMaker']}

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []
        for d in as_list(data):
            with resolvers(make_resolvers({'this': self, 'data': d})):
                tasks += with_context(
                    resolved(self.figure_maker), {'figure_makers': ['foreach']}
                ).make_tasks(d)
        return flatten(tasks)
##────────────────────────────────────────────────────────────────────────────}}}
## {{{                       --     SwitchFigure     --
class SwitchFigure(FigureMaker):
    """A figure maker that will select between different figure makers based on condition"""

    condition: Any = 'default'
    cases: Dict[Any, ResolvableFigureMaker] = {}

    def model_post_init(self, *a):
        super().model_post_init(*a)
        self.cases = merged_into_container(
            self.cases,
            self,
            INHERIT_ATTRS['FigureMaker']['FigureMaker'],
        )

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:

        cond = self.condition
        figlog.debug('SwitchFigure: condition is %s', cond)
        if callable(cond):
            cond = cond()

        if cond not in self.cases:
            raise ValueError(f'No case for condition {cond} or default for SwitchFigure')

        with resolvers(make_resolvers({'this': self, 'data': data})):
            figmaker = with_context(resolved(self.cases[cond]), {'figure_makers': ['switch']})

        try:
            return figmaker.make_tasks(data)
        except NotImplementedError:
            figlog.debug('Case %s in SwitchFigure can\'t make tasks', cond)
            return []

##────────────────────────────────────────────────────────────────────────────}}}
## {{{                        --     MultiFigure     --
class MultiFigure(FigureMaker):
    """A figure maker that will call multiple figure makers from the same data. Either by repeating
    the same figure maker n times, or by using different figure makers for each figure."""

    figure_makers: ListOrSingle[ResolvableFigureMaker] = []
    n_repeats: int = 1

    def model_post_init(self, *a):
        super().model_post_init(*a)
        self.figure_makers = merged_into_container(
            as_list(self.figure_makers), self, INHERIT_ATTRS['FigureMaker']['FigureMaker']
        )
        assert isinstance(self.figure_makers, list)
        base_len = len(self.figure_makers)
        self.figure_makers = self.figure_makers * self.n_repeats
        assert len(self.figure_makers) == base_len * self.n_repeats

    def make_tasks(self, data: ListOrSingle[PlotData]) -> List[FigureTask]:
        tasks = []
        with resolvers(make_resolvers({'this': self, 'data': data})):
            for i, figmaker in enumerate(self.figure_makers):
                fmaker = with_context(
                    resolved(figmaker),
                    {
                        'multifigure_index': i,
                        'figure_makers': ['multi'],
                        'multifigure_n_repeats': self.n_repeats,
                    },
                )
                try:
                    tasks += fmaker.make_tasks(data)
                except NotImplementedError:
                    figlog.debug('Figure maker %s in MultiFigure can\'t make tasks', i)

        return tasks

##────────────────────────────────────────────────────────────────────────────}}}

## {{{                      --     GridPlotConfig     --
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
##────────────────────────────────────────────────────────────────────────────}}}
## {{{                     --     UORFMatrixFigure     --

def get_single_seq_uorf_vals(network_info: dict) -> tuple[int, int]:
    uorf_vals = network_info.get('uorf_values', None)
    assert uorf_vals is not None, f"Missing uorf_values"
    return uorf_vals[0]


def get_uorf_grid(network_infos, x_axis=0, y_axis=1, invert_cols=False, invert_rows=True):
    uorfs = np.array([get_single_seq_uorf_vals(n) for n in network_infos])
    uorfs_row, uorfs_col = uorfs[:, x_axis], uorfs[:, y_axis]
    coords_row = {row: i for i, row in enumerate(np.sort(np.unique(uorfs_row)))}
    coords_col = {col: i for i, col in enumerate(np.sort(np.unique(uorfs_col)))}
    if invert_cols:
        coords_col = {k: len(coords_col) - 1 - v for k, v in coords_col.items()}
    if invert_rows:
        coords_row = {k: len(coords_row) - 1 - v for k, v in coords_row.items()}
    network_coords = np.array(
        [(coords_row[r], coords_col[c]) for r, c in zip(uorfs_row, uorfs_col)]
    )
    return network_coords


class UORFMatrixFigure(FigureMaker):

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

    def model_post_init(self, *a):
        super().model_post_init(*a)

        LEGEND_ROW_CONF = GridPlotConfig(  # type: ignore
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

        LEGEND_COL_CONF = GridPlotConfig(  # type: ignore
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

        UORF_COLORBAR_SPECIALIZATION = GridPlotConfig(  # type: ignore
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
                resolved_gpc = resolved(gpc)
                if resolved_gpc.matches(r, c, grid_size):
                    task_plotconf = merged(task_plotconf, resolved_gpc.plot_config)

            t = PlotTask(  # type: ignore
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
        ftask.after_plot_tasks.append(ut.PartialFunction(func=final_touches))
        return [ftask]

##────────────────────────────────────────────────────────────────────────────}}}

