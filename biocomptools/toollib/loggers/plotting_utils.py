## {{{                          --     imports     --

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from biocomptools.trainutils import ffill
from biocomptools.toollib.loggers.paramgradlogger import get_plot_rows_and_columns
from biocomptools.toollib.loggers.metrics_models import LoggerMetricsHistory
from biocomptools.logging_config import get_logger
from typing import List, Union
from biocomptools.toollib.loggers.metrics_models import (
    StepMetrics,
    ReplicateMetrics,
    NetworkDataPairMetrics,
)
from biocomp.plotutils import FigureSpec

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}

# Network name truncation constants
TRUNCATE_START_LEN = 10
TRUNCATE_END_LEN = 25
TRUNCATE_MIN_LEN = TRUNCATE_START_LEN + TRUNCATE_END_LEN + 3
MAX_NETWORK_FOR_LEGEND = 60


def truncate_network_name(name: str) -> str:
    """Truncate network name if longer than TRUNCATE_MIN_LEN characters."""
    if len(name) <= TRUNCATE_MIN_LEN:
        return name
    return f"{name[:TRUNCATE_START_LEN]}...{name[-TRUNCATE_END_LEN:]}"


class MetricsPlotter:
    """Shared plotting functionality for metrics loggers"""

    @staticmethod
    def plot_metrics_history(
        metrics_history: List[StepMetrics],
        title_prefix: str,
        output_path: Path,
        logger_name: str,
        vmin: float = 0.005,
        vmax: float = 0.5,
        training_id: Optional[str] = None,
    ):
        """Plot metrics history using StepMetrics format (for DetailedTrainingStatsLogger)."""
        MetricsPlotter._plot_metrics_internal(
            metrics_history, title_prefix, output_path, logger_name, vmin, vmax, training_id
        )

    @staticmethod
    def plot_validation_history(
        validation_history: List[Dict],
        title_prefix: str,
        output_path: Path,
        logger_name: str,
        vmin: float = 0.005,
        vmax: float = 0.5,
        training_id: Optional[str] = None,
    ):
        """Plot validation history using ValidationLossLogger's dict format."""
        # Convert validation history to StepMetrics format
        metrics_history = MetricsPlotter._convert_validation_to_metrics(validation_history)
        MetricsPlotter._plot_metrics_internal(
            metrics_history, title_prefix, output_path, logger_name, vmin, vmax, training_id
        )

    @staticmethod
    def _convert_validation_to_metrics(validation_history: List[Dict]) -> List[StepMetrics]:
        """Convert ValidationLossLogger history format to StepMetrics format."""
        step_metrics_list = []

        for hist_entry in validation_history:
            step = hist_entry['step']
            metrics_list = hist_entry['metrics']  # List of dicts, one per replicate
            training_loss = hist_entry.get('training_loss')

            replicate_metrics = []
            for rep_idx, rep_dict in enumerate(metrics_list):
                # Convert per_network list to NetworkDataPairMetrics
                per_network_list = []
                for net_dict in rep_dict.get('per_network', []):
                    net_metric = NetworkDataPairMetrics(
                        network_name=net_dict['network_name'],
                        networkdatapair=net_dict.get(
                            'networkdatapair', {'network_name': net_dict['network_name']}
                        ),
                        RMSE=net_dict['rmse'],
                        MSE=net_dict['rmse'] ** 2,  # Approximate MSE from RMSE
                        n_samples=rep_dict.get('n_evaluated', 0),
                    )
                    per_network_list.append(net_metric)

                # Create ReplicateMetrics
                rep_metrics = ReplicateMetrics(
                    replicate=rep_idx,
                    overall_RMSE=rep_dict['avg_rmse'],
                    overall_MSE=rep_dict['avg_rmse'] ** 2,  # Approximate MSE from RMSE
                    n_samples=rep_dict.get('n_evaluated', 0),
                    per_networkdatapair=per_network_list,
                )
                replicate_metrics.append(rep_metrics)

            step_metrics = StepMetrics(
                step=step, metrics=replicate_metrics, training_loss=training_loss
            )
            step_metrics_list.append(step_metrics)

        return step_metrics_list

    @staticmethod
    def _plot_metrics_internal(
        metrics_history: List[StepMetrics],
        title_prefix: str,
        output_path: Path,
        logger_name: str,
        vmin: float = 0.001,
        vmax: float = 0.5,
        training_id: Optional[str] = None,
    ):
        """
        Plot metrics history with support for both overall and per-network metrics.

        If there's only one timepoint, shows only a bar plot.
        If there are multiple timepoints, shows bar plot at top + line plots below.
        """
        if not metrics_history:
            return

        single_timepoint = len(metrics_history) == 1
        n_replicates = len(metrics_history[0].metrics)

        all_networks = set()
        for metric in metrics_history:
            for rep in metric.metrics:
                for net_metric in rep.per_networkdatapair:
                    all_networks.add(net_metric.network_name)
        all_networks = sorted(list(all_networks))
        n_networks = len(all_networks)

        # Check if sublosses are available
        has_sublosses = (metrics_history and 
                        metrics_history[0].metrics and 
                        metrics_history[0].metrics[0].sublosses is not None)
        
        # Adjust layout based on sublosses availability
        if has_sublosses and not single_timepoint:
            n_rows = 3 + n_replicates  # barplot + overall + sublosses + per-replicate panels
            height_ratios = [1.2, 2, 2] + [3] * n_replicates
        else:
            n_rows = 2 + n_replicates  # barplot + overall + per-replicate panels
            height_ratios = [1.2, 2] + [3] * n_replicates
            
        fig_height = min(40, 15 + 3 * (n_replicates + (1 if has_sublosses else 0) + 1))
        fig = plt.figure(figsize=(30, fig_height))
        gs = fig.add_gridspec(
            n_rows, 2, height_ratios=height_ratios, hspace=0.3, wspace=0.3
        )
        ax_bar = fig.add_subplot(gs[0, :])
        _plot_rmse_bars(ax_bar, metrics_history[-1], n_replicates, all_networks, vmin, vmax)

        if not single_timepoint:
            ax_overall = fig.add_subplot(gs[1, :])
            _plot_overall_rmse_over_time(ax_overall, metrics_history, n_replicates, vmin, vmax)
            
            # Add sublosses plot if available
            if has_sublosses:
                ax_sublosses = fig.add_subplot(gs[2, :])
                _plot_sublosses_over_time(ax_sublosses, metrics_history, n_replicates)
                per_rep_start_idx = 3
            else:
                per_rep_start_idx = 2
            
            for rep_idx in range(n_replicates):
                row_idx = per_rep_start_idx + rep_idx
                ax_rep = fig.add_subplot(gs[row_idx, :])
                _plot_networks_rmse_single_replicate(
                    ax_rep, metrics_history, all_networks, rep_idx, vmin, vmax
                )

        plt.suptitle(f'{title_prefix} Metrics History - Step {metrics_history[-1].step}')
        # plt.tight_layout()

        metadata = {
            'logger_name': logger_name,
            'title_prefix': title_prefix,
            'final_step': metrics_history[-1].step,
            'n_replicates': n_replicates,
            'n_networks': n_networks,
            'network_names': all_networks,
        }

        # Add training_id to metadata if provided
        if training_id is not None:
            metadata['training_id'] = training_id

        fig_spec = FigureSpec(
            output_dir=str(output_path.parent),
            output_file=output_path.name,
            metadata={'plot_method': 'plot_metrics_history', 'metadata': metadata},
        )

        from dataclasses import dataclass

        @dataclass
        class FigAx:
            figure: object
            axes: object = None

        figax = FigAx(figure=fig)
        fig_spec.save_figure(figax)
        plt.close(fig)


def _plot_overall_rmse_over_time(
    ax,
    metrics_history: List[StepMetrics],
    n_replicates: int,
    vmin=None,
    vmax=None,
):
    """Plot overall RMSE over time for all replicates."""
    steps = [m.step for m in metrics_history]

    max_colors = 12
    colors = plt.cm.get_cmap('Set3', max_colors)

    for rep_idx in range(n_replicates):
        overall_rmses = [m.metrics[rep_idx].overall_RMSE for m in metrics_history]
        color = colors(rep_idx % max_colors)
        ax.plot(
            steps,
            overall_rmses,
            marker='o',
            linewidth=2,
            markersize=4,
            color=color,
            label=f'Replicate {rep_idx}',
        )

    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('RMSE (log scale)', fontsize=11)
    ax.set_title('Average RMSE over Time', fontsize=12, y=0.8)

    if n_replicates > 1:
        ax.legend(fontsize=9)

    ax.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5, which='major')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, which='minor')
    ax.set_yscale('log')

    # Apply fixed y-axis limits if specified
    if vmin is not None or vmax is not None:
        current_ylim = ax.get_ylim()
        new_vmin = vmin if vmin is not None else current_ylim[0]
        new_vmax = vmax if vmax is not None else current_ylim[1]
        ax.set_ylim(new_vmin, new_vmax)

    # Improve axis formatting
    ax.tick_params(labelsize=9)


def _plot_networks_rmse_single_replicate(
    ax,
    metrics_history: List[StepMetrics],
    all_networks: List[str],
    replicate_idx: int,
    vmin=None,
    vmax=None,
):
    """Plot per-network RMSE over time for a single replicate."""
    if not all_networks:
        ax.text(
            0.5,
            0.5,
            'No per-network data available',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f'Replicate {replicate_idx} - Per-Network RMSE Over Time', fontsize=12, pad=10)
        return

    steps = [m.step for m in metrics_history]

    import matplotlib.cm as cm

    # get network order from last timestep RMSE values for legend ordering
    last_step_metrics = metrics_history[-1].metrics[replicate_idx].per_networkdatapair
    network_last_rmse = {}
    for net_metric in last_step_metrics:
        network_last_rmse[net_metric.network_name] = net_metric.RMSE

    ordered_networks = sorted(
        all_networks, key=lambda name: network_last_rmse.get(name, float('inf'))
    )

    cmap_name = 'jet'
    network_colors = cm.get_cmap(cmap_name, len(all_networks))

    for net_idx, network_name in enumerate(
        ordered_networks
    ):  # Changed from all_networks to ordered_networks
        # Find the original index to get the correct color
        orig_idx = all_networks.index(network_name)
        color = network_colors(net_idx % len(ordered_networks))

        network_rmses = []
        for metric in metrics_history:
            rep_metrics = metric.metrics[replicate_idx].per_networkdatapair
            net_metric = next((m for m in rep_metrics if m.network_name == network_name), None)
            if net_metric:
                network_rmses.append(net_metric.RMSE)
            else:
                network_rmses.append(np.nan)

        if any(~np.isnan(network_rmses)):
            ax.plot(
                steps,
                network_rmses,
                linewidth=2,
                color=color,
                label=network_name,
            )

    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('RMSE (log scale)', fontsize=11)
    ax.set_title(f'Replicate {replicate_idx} - Per-Network RMSE Over Time', fontsize=12, pad=10)

    # Create legend with truncated names, ordered by last step performance
    if len(ordered_networks) > 0:
        # Create legend handles with truncated names in order
        legend_handles = []
        legend_labels = []

        for idx, network_name in enumerate(ordered_networks):
            color = network_colors(idx % len(ordered_networks))

            truncated_name = truncate_network_name(network_name)

            import matplotlib.lines as mlines

            handle = mlines.Line2D(
                [], [], color=color, marker='o', linestyle='-', markersize=4, linewidth=2
            )
            legend_handles.append(handle)
            legend_labels.append(truncated_name)

        if len(legend_handles) <= MAX_NETWORK_FOR_LEGEND:
            ncol = 2 if len(legend_handles) > 13 else 1
            ax.legend(
                legend_handles,
                legend_labels,
                fontsize=7,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                ncol=ncol,
            )
        else:
            top_handles = legend_handles[:MAX_NETWORK_FOR_LEGEND]
            top_labels = legend_labels[:MAX_NETWORK_FOR_LEGEND]
            ax.legend(
                top_handles,
                top_labels,
                fontsize=7,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                ncol=2,
                title=f'Worst {MAX_NETWORK_FOR_LEGEND} of {len(ordered_networks)} networks',
            )

    ax.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5, which='major')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, which='minor')
    ax.set_yscale('log')

    if vmin is not None or vmax is not None:
        current_ylim = ax.get_ylim()
        new_vmin = vmin if vmin is not None else current_ylim[0]
        new_vmax = vmax if vmax is not None else current_ylim[1]
        new_vmin = min(new_vmin, current_ylim[0])
        new_vmax = max(new_vmax, current_ylim[1])
        ax.set_ylim(new_vmin, new_vmax)

    ax.tick_params(labelsize=9)


def _plot_rmse_bars(
    ax,
    metric: StepMetrics,
    n_replicates: int,
    all_networks: List[str],
    vmin=None,
    vmax=None,
):
    """
    Helper to plot improved RMSE bar chart with grouped bars per network + average.

    Creates grouped bar plots where:
    - Each group represents one network (or aggregate)
    - Within each group, bars for different replicates are placed side by side
    - Network names are displayed below each group
    - Clear visual separation between network groups and aggregate
    """

    # prepare data for each replicate
    rep_data = []
    for rep_idx in range(n_replicates):
        rep_metrics = metric.metrics[rep_idx]

        # Get RMSE for each network
        network_rmses = {}
        for net_metric in rep_metrics.per_networkdatapair:
            network_rmses[net_metric.network_name] = net_metric.RMSE
        network_rmses['Average'] = rep_metrics.overall_RMSE

        rep_data.append(network_rmses)

    group_averages = {}
    for group_name in all_networks + ['Average']:
        rmse_values = []
        for rep_idx in range(n_replicates):
            value = rep_data[rep_idx].get(group_name, np.nan)
            if not np.isnan(value):
                rmse_values.append(value)
        if rmse_values:
            group_averages[group_name] = np.mean(rmse_values)
        else:
            group_averages[group_name] = float('inf')

    # DEBUG: Print group averages and ordering

    # Set up groups: individual networks + aggregate average, ordered by average RMSE
    if all_networks:
        sorted_networks = sorted(all_networks, key=lambda name: group_averages[name])
        groups_to_plot = sorted_networks + ['Average']
    else:
        groups_to_plot = ['Average']

    # Group positioning: networks together, then gap, then average
    group_width = 0.8  # Total width allocated to each group
    rep_width = group_width / n_replicates  # Width of each replicate bar within group

    # Create positions with spacing: networks clustered, then gap before average
    if len(all_networks) > 0:
        network_positions = np.arange(len(all_networks))  # Networks clustered together
        avg_position = len(all_networks) + 0.5  # Gap before average
        group_positions = np.concatenate([network_positions, [avg_position]])
    else:
        group_positions = np.array([0])  # Just average if no networks

    import matplotlib.cm as cm

    max_colors = 12
    replicate_colors = cm.get_cmap('Set3', max_colors)

    for rep_idx in range(n_replicates):
        rep_offset = (rep_idx - (n_replicates - 1) / 2) * rep_width
        values = [rep_data[rep_idx].get(group_name, np.nan) for group_name in groups_to_plot]
        bars = ax.bar(
            group_positions + rep_offset,
            values,
            rep_width,
            color=replicate_colors(rep_idx % max_colors),
            label=f'Replicate {rep_idx}',
            edgecolor='#00000077',
            linewidth=1,
        )

    ax.set_ylabel('RMSE (log scale)', fontsize=11)

    ax.set_xticks([])
    ax.set_xticklabels([])

    from matplotlib.transforms import blended_transform_factory

    N_STAGGER_LEVELS = 7
    STAGGER_OFFSET = 0.07

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for idx, (pos, group_name) in enumerate(zip(group_positions, groups_to_plot)):
        stag_level = idx % N_STAGGER_LEVELS
        y_offset_ax = -STAGGER_OFFSET * (stag_level + 1)
        TSIZE = 6
        ax.text(
            pos,
            y_offset_ax,
            truncate_network_name(group_name),
            ha="center",
            va="center",
            fontsize=TSIZE,
            fontweight='bold',
            color='white',
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        ax.text(
            pos,
            y_offset_ax,
            truncate_network_name(group_name),
            ha="center",
            va="center",
            fontsize=TSIZE,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

        label_top_ax = -STAGGER_OFFSET * (stag_level + 0.5)
        ax.plot(
            [pos, pos],
            [0.0, label_top_ax],
            transform=trans,
            color="0.6",
            linewidth=0.7,
            alpha=0.6,
            zorder=0,
            clip_on=False,
        )

    if n_replicates > 1:
        ax.legend(loc='upper right', fontsize=9)

    # Add grid for better readability
    ax.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5, which='major')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, which='minor')
    ax.set_yscale('log')

    # Add visual separation between networks and average
    if len(all_networks) > 0:
        separator_x = len(all_networks) - 0.5
        ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    ax.set_title('RMSE by Network (Latest Step)', fontsize=12, pad=15)

    if vmin is not None or vmax is not None:
        current_ylim = ax.get_ylim()
        new_vmin = vmin if vmin is not None else current_ylim[0]
        new_vmax = vmax if vmax is not None else current_ylim[1]
        new_vmin = min(new_vmin, current_ylim[0])
        new_vmax = max(new_vmax, current_ylim[1])
        ax.set_ylim(new_vmin, new_vmax)

    elif len(values) > 0 and not all(np.isnan(values)):
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            y_min, y_max = min(valid_values), max(valid_values)
            ax.set_ylim(y_min * 0.7, y_max * 1.3)


def _plot_sublosses_over_time(ax, metrics_history: List[StepMetrics], n_replicates: int):
    """Plot average sublosses over time with Set3 colormap and different markers per replicate."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not metrics_history or not metrics_history[0].metrics[0].sublosses:
        return
    
    steps = [metric.step for metric in metrics_history]
    
    # Get all subloss names from first replicate
    loss_names = list(metrics_history[0].metrics[0].sublosses.keys())
    
    # Use Set3 colormap
    colors = plt.cm.Set3(np.linspace(0, 1, len(loss_names)))
    
    # Different markers per replicate
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for loss_idx, loss_name in enumerate(loss_names):
        color = colors[loss_idx]
        
        for rep_idx in range(n_replicates):
            # Extract values for this loss and replicate over time
            values = []
            for metric in metrics_history:
                if (rep_idx < len(metric.metrics) and 
                    metric.metrics[rep_idx].sublosses and 
                    loss_name in metric.metrics[rep_idx].sublosses):
                    values.append(metric.metrics[rep_idx].sublosses[loss_name])
                else:
                    values.append(np.nan)
            
            marker = markers[rep_idx % len(markers)]
            label = loss_name if rep_idx == 0 else None
            
            ax.plot(steps, values, 
                   color=color, marker=marker, markersize=6, 
                   linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Value')
    ax.set_title('Average Sublosses over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set log scale if values vary widely
    all_values = []
    for metric in metrics_history:
        for rep in metric.metrics:
            if rep.sublosses:
                all_values.extend([v for v in rep.sublosses.values() if not np.isnan(v)])
    
    if all_values:
        val_range = max(all_values) / min([v for v in all_values if v > 0]) if min([v for v in all_values if v > 0]) > 0 else 1
        if val_range > 100:  # Use log scale if range is > 100x
            ax.set_yscale('log')
