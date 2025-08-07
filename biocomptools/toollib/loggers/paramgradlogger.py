## {{{                          --     imports     --
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues in threads
import matplotlib.pyplot as plt
import numpy as np
import jax
from math import ceil, sqrt
from pathlib import Path
from pydantic import Field
from typing import Union, Optional, List, Tuple, Callable, Literal


from biocomp.parameters import ParameterTree, PTree, flatten_PTree, get_path_components
from biocomp.jaxutils import tree_get
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger
from biocomptools.modelmodel import get_shared_params, get_nonshared_params

logger = get_logger(__name__)
##────────────────────────────────────────────────────────────────────────────}}}

IDEAL_RATIO = 0.65


def get_plot_rows_and_columns(num_plots: int, ideal_ratio: float = IDEAL_RATIO):
    if num_plots == 0:
        return 1, 1
    n_cols = int(ceil(sqrt(num_plots / ideal_ratio)))
    n_rows = int(ceil(num_plots / n_cols))
    if n_rows == 0:
        n_rows = 1
    if n_cols == 0:
        n_cols = 1
    return n_rows, n_cols


def _count_filtered_parameters(params_section):
    """Count the number of non-ArrayRef parameters in a section."""
    if params_section is None:
        return 0

    params_data = (
        params_section.data if isinstance(params_section, ParameterTree) else params_section
    )
    pvalues, (pnames, _) = flatten_PTree(params_data)

    count = 0
    for name in pnames:
        # Skip ArrayRef parameters
        if hasattr(name, "actual_path"):
            continue
        # Skip parameters whose name ends with "variable_id"
        name_components = get_path_components(name)
        if name_components and name_components[-1].endswith("variable_id"):
            continue
        count += 1

    return count


def _calculate_figure_dimensions(
    shared_params,
    nonshared_params,
    plot_local_params,
    plot_shared_params=True,
    base_width=4.0,
    base_height=3,
):
    """Calculate figure dimensions based on parameter counts and subplot layouts."""
    shared_count = _count_filtered_parameters(shared_params) if plot_shared_params else 0
    nonshared_count = _count_filtered_parameters(nonshared_params) if plot_local_params else 0

    # Calculate rows and columns for each section
    shared_rows = 0
    shared_cols = 0
    if shared_count > 0:
        shared_rows, shared_cols = get_plot_rows_and_columns(shared_count, ideal_ratio=IDEAL_RATIO)

    nonshared_rows = 0
    nonshared_cols = 0
    if nonshared_count > 0:
        nonshared_rows, nonshared_cols = get_plot_rows_and_columns(
            nonshared_count, ideal_ratio=IDEAL_RATIO
        )

    # Determine overall dimensions
    max_cols = max(shared_cols, nonshared_cols)
    total_rows = shared_rows + nonshared_rows

    if max_cols == 0:
        max_cols = 1
    if total_rows == 0:
        total_rows = 1

    # Calculate figure size with consistent subplot sizing
    fwidth = max_cols * base_width + 1.0  # add margin
    fheight = total_rows * base_height + 2.0  # add margin for titles

    # Apply reasonable bounds
    fwidth = max(7, min(fwidth, 55))
    fheight = max(7, min(fheight, 40))

    return fwidth, fheight, shared_rows, nonshared_rows


def format_stat(value: float, threshold: float = 1e-3) -> str:
    if abs(value) < threshold and value != 0:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _compute_statistics(pvalues_flat, gvalues_flat, pnames, learning_rate, shapes):
    stats_list = []
    has_grads = len(gvalues_flat) > 0
    has_lr = learning_rate is not None

    for i, w_flat in enumerate(pvalues_flat):
        if w_flat.size == 0:
            stats_list.append(None)
            continue

        w_l2 = np.linalg.norm(w_flat)
        w_mean = np.mean(w_flat)
        # Handle single element case for std calculation
        w_std = np.std(w_flat) if w_flat.size > 1 else 0.0

        stats = {
            'w': {
                'l2': w_l2,
                'mean': w_mean,
                'std': w_std,
                'size': w_flat.size,
                'shape': shapes[i] if i < len(shapes) else w_flat.size,
            },
            'g': {},
            'ratio': np.nan,
        }

        if has_grads and i < len(gvalues_flat):
            g_flat = gvalues_flat[i]
            # Handle None gradients
            if g_flat is not None:
                # double-check shape consistency
                if g_flat.size != w_flat.size:
                    logger.warning(
                        f"Gradient size {g_flat.size} does not match parameter size "
                        f"{w_flat.size} for parameter {pnames[i]}"
                    )
                else:
                    g_l2 = np.linalg.norm(g_flat)
                    stats['g'] = {
                        'l2': g_l2,
                        'mean': np.mean(g_flat),
                        # Handle single element case for std calculation
                        'std': np.std(g_flat) if g_flat.size > 1 else 0.0,
                    }
                    if has_lr and w_l2 > 1e-9:
                        stats['ratio'] = (learning_rate * g_l2) / w_l2

        stats_list.append(stats)

    is_normal = [
        not (
            get_path_components(name)[-1].endswith('rate')
            or get_path_components(name)[-1].endswith('affinities')
        )
        for name in pnames
    ]

    stat_ranges = _compute_stat_ranges_fast(stats_list, is_normal, has_grads, has_lr)

    return stats_list, stat_ranges


def _compute_stat_ranges_fast(stats_list, is_normal, has_grads, has_lr):
    ranges = {}

    w_l2_vals = []
    w_mean_vals = []
    w_std_vals = []
    g_l2_vals = []
    g_mean_vals = []
    g_std_vals = []
    ratio_vals = []

    for i, stats in enumerate(stats_list):
        if stats is None or not is_normal[i]:
            continue

        w_l2_vals.append(np.abs(stats['w']['l2']))
        w_mean_vals.append(np.abs(stats['w']['mean']))
        w_std_vals.append(np.abs(stats['w']['std']))

        if has_grads and 'l2' in stats['g']:
            g_l2_vals.append(np.abs(stats['g']['l2']))
            g_mean_vals.append(np.abs(stats['g']['mean']))
            g_std_vals.append(np.abs(stats['g']['std']))

        if has_grads and has_lr and not np.isnan(stats['ratio']):
            ratio_vals.append(stats['ratio'])

    for key, vals in [
        ('l2', w_l2_vals + g_l2_vals),
        ('mean', w_mean_vals + g_mean_vals),
        ('std', w_std_vals + g_std_vals),
    ]:
        if vals:
            ranges[key] = (min(vals), max(vals))
        else:
            ranges[key] = (0, 1)

    if ratio_vals:
        ranges['ratio'] = (min(ratio_vals), max(ratio_vals))

    return ranges


def _create_shared_bins(w_flat, g_flat, bins):
    if g_flat is None or g_flat.size == 0:
        if w_flat.size == 0:
            return np.linspace(0, 1, bins + 1)
        min_val, max_val = np.min(w_flat), np.max(w_flat)
    else:
        if w_flat.size == 0:
            min_val, max_val = np.min(g_flat), np.max(g_flat)
        else:
            min_val = min(np.min(w_flat), np.min(g_flat))
            max_val = max(np.max(w_flat), np.max(g_flat))

    # Handle single value case - create a small range around the value
    if min_val >= max_val:
        if min_val == 0:
            return np.linspace(-1e-9, 1e-9, bins + 1)
        else:
            # Create a small range (±1% or ±1e-9, whichever is larger)
            range_size = max(abs(min_val) * 0.01, 1e-9)
            return np.linspace(min_val - range_size, min_val + range_size, bins + 1)
    else:
        return np.linspace(min_val, max_val, bins + 1)


def _get_title_color(last_name_part, colors):
    if last_name_part == 'w':
        return colors['title_w']
    elif last_name_part == 'b':
        return colors['title_b']
    elif last_name_part.endswith('rate') or last_name_part.endswith('affinities'):
        return colors['title_special']
    else:
        return colors['title_default']


def _format_name_tuple_for_title(name_tuple, skip_first_name=True):
    """
    Format name_tuple for display, handling ArrayRefPath properly.

    For ArrayRefPath, show the actual_path clearly marked as a reference.
    For regular ParamPath, show the path components.
    """
    if hasattr(name_tuple, "actual_path"):  # ArrayRefPath
        actual_components = get_path_components(name_tuple.actual_path)
        display_path = '/'.join(
            actual_components[1:]
            if skip_first_name and len(actual_components) > 1
            else actual_components
        )
        return f"{display_path} [ArrayRef]"
    else:  # Regular ParamPath or tuple
        components = get_path_components(name_tuple)
        return '/'.join(components[1:] if skip_first_name and len(components) > 1 else components)


def _build_stat_matrix(stats, stat_ranges, stat_keys, num_rows):
    matrix = np.zeros((num_rows, len(stat_keys)))

    norm_factors = {}
    for key in stat_keys:
        if key in stat_ranges:
            vmin, vmax = stat_ranges[key]
            norm_factors[key] = (vmin, vmax - vmin) if vmax > vmin else (vmin, 1)

    for r_idx, source in enumerate(['w', 'g']):
        if r_idx == 1 and num_rows == 1:
            break

        for c_idx, key in enumerate(stat_keys):
            if key == 'ratio':
                val = stats['ratio'] if r_idx == 1 else np.nan
            else:
                val = stats[source].get(key, np.nan) if source in stats else np.nan

            if not np.isnan(val) and key in norm_factors:
                vmin, vrange = norm_factors[key]
                matrix[r_idx, c_idx] = (np.abs(val) - vmin) / vrange

    return matrix


def _annotate_heatmap(ax, matrix, stats, stat_keys, col_labels, num_rows):
    row_labels = ['W', 'G']

    formatted_values = {}
    for source in ['w', 'g']:
        if source in stats:
            for key in stat_keys:
                if key != 'ratio':
                    formatted_values[(source, key)] = format_stat(stats[source].get(key, np.nan))

    if 'ratio' in stat_keys:
        formatted_values[('ratio',)] = format_stat(stats.get('ratio', np.nan))

    ratio_opt = 1e-3
    log_fold_range = 3.0

    for r_idx in range(num_rows):
        for c_idx, key in enumerate(stat_keys):
            if key == 'ratio' and r_idx == 1:
                ratio_val = stats.get('ratio', np.nan)
                if np.isfinite(ratio_val) and ratio_val > 1e-9:
                    log_dist = np.log10(ratio_val) - np.log10(ratio_opt)
                    norm_val = 0.5 * (log_dist / log_fold_range + 1.0)
                    norm_val = np.clip(norm_val, 0, 1)
                    text_color = 'white' if norm_val < 0.2 or norm_val > 0.8 else 'black'
                else:
                    text_color = 'black'
            else:
                text_color = 'white' if matrix[r_idx, c_idx] > 0.6 else 'black'

            if key == 'ratio':
                text_val = formatted_values.get(('ratio',), " ") if r_idx == 1 else " "
            else:
                source = row_labels[r_idx].lower()
                text_val = formatted_values.get((source, key), " ")

            ax.text(c_idx, r_idx, text_val, ha='center', va='center', color=text_color, fontsize=7)

            if r_idx == 0:
                ax.text(c_idx, -1.2, col_labels[c_idx], ha='center', va='center', fontsize=9)

        ax.text(-0.7, r_idx, row_labels[r_idx], ha='center', va='center', fontsize=9)


def _add_stats_heatmap(ax, stats, stat_ranges, has_grads, colors):
    # Adjust height based on whether we have gradients
    height = 0.25 if has_grads else 0.15
    inset_ax = ax.inset_axes([0, 1.02, 1, height], transform=ax.transAxes)
    inset_ax.set_axis_off()

    col_labels = ['L2', 'µ', 'σ']
    stat_keys = ['l2', 'mean', 'std']
    if has_grads and 'ratio' in stat_ranges:
        col_labels.append('Upd/W')
        stat_keys.append('ratio')

    num_rows = 2 if has_grads else 1
    num_cols = len(col_labels)
    stat_matrix = _build_stat_matrix(stats, stat_ranges, stat_keys, num_rows)
    rgba_img = np.zeros((num_rows, num_cols, 4))
    cmap_main = plt.get_cmap(colors['cmap'])
    cmap_diverging = plt.get_cmap('RdBu_r')

    ratio_opt = 1e-3
    log_fold_range = 3.0

    for r_idx in range(num_rows):
        for c_idx, key in enumerate(stat_keys):
            if key == 'ratio' and r_idx == 1:
                ratio_val = stats.get('ratio', np.nan)
                if np.isfinite(ratio_val) and ratio_val > 1e-9:
                    log_dist = np.log10(ratio_val) - np.log10(ratio_opt)
                    norm_val = 0.5 * (log_dist / log_fold_range + 1.0)
                    norm_val = np.clip(norm_val, 0, 1)
                    rgba_img[r_idx, c_idx, :] = cmap_diverging(norm_val)
                else:
                    rgba_img[r_idx, c_idx, :] = [0.9, 0.9, 0.9, 1.0]
            else:
                norm_val = stat_matrix[r_idx, c_idx]
                rgba_img[r_idx, c_idx, :] = cmap_main(norm_val)

    inset_ax.imshow(rgba_img, aspect='auto', interpolation='nearest')

    _annotate_heatmap(inset_ax, stat_matrix, stats, stat_keys, col_labels, num_rows)


def _plot_layer(
    ax,
    idx,
    w_flat,
    g_flat,
    name_tuple,
    stats,
    stat_ranges,
    bins,
    skip_first_name,
    colors,
    grid_alpha,
):
    shared_bins = _create_shared_bins(w_flat, g_flat, bins)

    w_counts, _ = np.histogram(w_flat, bins=shared_bins)
    bin_centers = (shared_bins[:-1] + shared_bins[1:]) / 2
    bin_width = shared_bins[1] - shared_bins[0]

    ax.bar(
        bin_centers,
        w_counts,
        width=bin_width * 0.8,
        color=colors['weights_hist'],
        edgecolor='#000000ff',
        linewidth=0.5,
    )

    ax.set_ylabel("Weights Count", color=colors['weights_label'], fontsize=9)
    ax.grid(True, alpha=grid_alpha, color=colors['grid_color'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(colors['grid_color'])
    ax.spines['bottom'].set_color(colors['grid_color'])
    ax.tick_params(colors=colors['text_color'])
    ax.tick_params(axis='y', labelcolor=colors['weights_label'])

    if g_flat is not None and g_flat.size > 0:
        ax_grad = ax.twinx()
        g_counts, _ = np.histogram(g_flat, bins=shared_bins)
        ax_grad.bar(
            bin_centers,
            g_counts,
            width=bin_width * 0.8,
            color=colors['gradients_hist'],
            edgecolor='#000000ff',
            linewidth=0.5,
        )
        ax_grad.set_ylabel("Gradients Count", color=colors['gradients_label'], fontsize=9)
        ax_grad.tick_params(axis='y', labelcolor=colors['gradients_label'])
        ax_grad.spines['top'].set_visible(False)
        ax_grad.spines['left'].set_visible(False)
        ax_grad.spines['right'].set_color(colors['grid_color'])
    elif g_flat is None:
        # Show "[no gradients]" label when gradients are missing
        ax_grad = ax.twinx()
        ax_grad.set_ylabel("[no gradients]", color=colors['gradients_label'], fontsize=9)
        ax_grad.set_ylim(0, 1)
        ax_grad.set_yticks([])
        ax_grad.spines['top'].set_visible(False)
        ax_grad.spines['left'].set_visible(False)
        ax_grad.spines['right'].set_color(colors['grid_color'])

    last_name = get_path_components(name_tuple)[-1]
    title_color = _get_title_color(last_name, colors)
    layer_name = _format_name_tuple_for_title(name_tuple, skip_first_name)

    # Create title with parameter shape in brackets
    param_shape = stats['w'].get('shape', stats['w']['size'])
    if isinstance(param_shape, tuple):
        shape_str = str(param_shape)
    else:
        shape_str = f"({param_shape:,},)"  # fallback to size if shape not available
    full_title = f"{layer_name} {shape_str}"
    ax.set_title(
        full_title,
        fontsize=11,
        color=title_color,
        y=1.4,
    )

    _add_stats_heatmap(ax, stats, stat_ranges, g_flat is not None and g_flat.size > 0, colors)


def _plot_parameter_section(
    subfig,
    params_section,
    gradients_section,
    section_name: str,
    learning_rate: Optional[float],
    bins: int,
    skip_first_name: bool,
    colors,
    grid_alpha: float,
    bg_color: str = '#ffffff',
):
    """Plot a section (shared or non-shared) of parameters in a subfigure."""
    if params_section is None:
        return

    # Filter out ArrayRef parameters
    params_data = (
        params_section.data if isinstance(params_section, ParameterTree) else params_section
    )
    pvalues, (pnames, _) = flatten_PTree(params_data)

    filtered_indices = []
    for i, name in enumerate(pnames):
        # Skip ArrayRef parameters
        if hasattr(name, "actual_path"):
            continue
        # Skip parameters whose name ends with "variable_id"
        name_components = get_path_components(name)
        if name_components and name_components[-1].endswith("variable_id"):
            continue
        filtered_indices.append(i)

    if not filtered_indices:
        return

    # Filter the data
    pvalues_filtered = [pvalues[i] for i in filtered_indices]
    pnames_filtered = [pnames[i] for i in filtered_indices]

    pvalues_flat = []
    pvalues_shapes = []
    for p in pvalues_filtered:
        p_arr = np.array(p, dtype=np.float32)
        pvalues_flat.append(p_arr.flatten())
        pvalues_shapes.append(p_arr.shape)

    # Handle gradients
    gvalues_flat = []
    if gradients_section is not None:
        grads_data = (
            gradients_section.data
            if isinstance(gradients_section, ParameterTree)
            else gradients_section
        )
        gvalues, (gnames, _) = flatten_PTree(grads_data)

        # Create a mapping of gradient names to values for efficient lookup
        gname_to_value = {}
        for gname, gvalue in zip(gnames, gvalues):
            # Convert to comparable format (both as strings of the path)
            gname_str = str(gname.actual_path if hasattr(gname, "actual_path") else gname)
            gname_to_value[gname_str] = gvalue

        # Match gradients to filtered parameters by name
        gvalues_filtered = []
        for pname in pnames_filtered:
            # Convert parameter name to comparable format
            pname_str = str(pname.actual_path if hasattr(pname, "actual_path") else pname)

            # Look up the corresponding gradient by name
            if pname_str in gname_to_value:
                gvalues_filtered.append(gname_to_value[pname_str])
            else:
                # If gradient doesn't exist for this parameter, mark as None
                gvalues_filtered.append(None)

        gvalues_flat = []
        for i, g in enumerate(gvalues_filtered):
            if g is not None:
                g_arr = np.array(g, dtype=np.float32)

                # validate gradient shape matches parameter shape (ignoring trailing dims of size 1)
                g_shape_squeezed = tuple(d for d in g_arr.shape if d != 1)
                p_shape_squeezed = tuple(d for d in pvalues_shapes[i] if d != 1)

                if g_shape_squeezed != p_shape_squeezed:
                    logger.warning(
                        f"Gradient shape {g_arr.shape} does not match parameter shape "
                        f"{pvalues_shapes[i]} for parameter {pnames_filtered[i]} "
                        f"(after squeezing: {g_shape_squeezed} vs {p_shape_squeezed})"
                    )
                    gvalues_flat.append(None)
                else:
                    gvalues_flat.append(g_arr.flatten())
            else:
                gvalues_flat.append(None)

    if not pvalues_flat:
        return

    stats_list, stat_ranges = _compute_statistics(
        pvalues_flat, gvalues_flat, pnames_filtered, learning_rate, pvalues_shapes
    )

    num_layers = len(pvalues_flat)
    n_rows, n_cols = get_plot_rows_and_columns(num_layers, ideal_ratio=IDEAL_RATIO)

    # Create subplots in the subfigure
    axes = subfig.subplots(
        n_rows,
        n_cols,
        gridspec_kw={
            'hspace': 1.2,
            'wspace': 0.8,
        },
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # Set background color for all axes
    for ax in axes_flat:
        ax.set_facecolor(bg_color)

    # Plot each parameter
    for i, ax in enumerate(axes_flat):
        if i >= num_layers or stats_list[i] is None:
            ax.set_visible(False)
            continue

        _plot_layer(
            ax,
            i,
            pvalues_flat[i],
            gvalues_flat[i] if gvalues_flat and i < len(gvalues_flat) else None,
            pnames_filtered[i],
            stats_list[i],
            stat_ranges,
            bins,
            skip_first_name,
            colors,
            grid_alpha,
        )

    # Add section title
    subfig.suptitle(f"{section_name} Parameters", fontsize=12, color=colors['text_color'], y=0.95)


def plot_parameter_diagnostics(
    params: Union[PTree, ParameterTree],
    param_gradients: Optional[Union[PTree, ParameterTree]] = None,
    learning_rate: Optional[float] = None,
    title: str = "Parameter & Gradient Diagnostics",
    bins: int = 30,
    show_plot: bool = False,
    skip_first_name: bool = True,
    colors=None,
    grid_alpha: float = 0.15,
    plot_local_params: bool = True,
    plot_shared_params: bool = True,
):
    if colors is None:
        colors = {
            'weights_label': '#0db3d0',
            'gradients_label': '#F5BD70',
            'weights_hist': '#0db3d0b0',
            'gradients_hist': '#fff00080',
            'title_w': 'black',
            'title_b': 'black',
            'title_special': 'black',
            'title_default': '#333333',
            'cmap': 'Greys',
            'bg_color': 'white',
            'text_color': 'black',
            'grid_color': 'gray',
            'shared_bg': '#EEFDF7aa',
            'nonshared_bg': '#FDF6EEaa',
        }

    # Separate shared and non-shared parameters using the proper functions
    shared_params = get_shared_params(params) if plot_shared_params else None
    nonshared_params = get_nonshared_params(params) if plot_local_params else None

    shared_gradients = None
    nonshared_gradients = None
    if param_gradients is not None:
        shared_gradients = get_shared_params(param_gradients) if plot_shared_params else None
        nonshared_gradients = get_nonshared_params(param_gradients) if plot_local_params else None

    # Determine height ratios based on which sections have data
    has_shared = shared_params is not None and plot_shared_params
    has_nonshared = nonshared_params is not None and plot_local_params

    # Calculate optimal figure dimensions based on parameter counts
    fwidth, fheight, shared_rows, nonshared_rows = _calculate_figure_dimensions(
        shared_params, nonshared_params, plot_local_params, plot_shared_params
    )

    # Create figure with subfigures
    fig = plt.figure(figsize=(fwidth, fheight))
    fig.suptitle(title, fontsize=16, color=colors['text_color'], y=0.98)
    fig.patch.set_facecolor(colors['bg_color'])

    if has_shared and has_nonshared:
        # Use proportional height ratios based on actual row counts
        height_ratios = (
            [shared_rows, nonshared_rows] if shared_rows > 0 and nonshared_rows > 0 else [1, 1]
        )
        subfigs = fig.subfigures(2, 1, height_ratios=height_ratios, hspace=0.0)
        shared_subfig = subfigs[0]
        nonshared_subfig = subfigs[1]
        shared_subfig.patch.set_facecolor(colors['shared_bg'])
        nonshared_subfig.patch.set_facecolor(colors['nonshared_bg'])
    elif has_shared:
        shared_subfig = (
            fig.subfigures(1, 1)[0]
            if isinstance(fig.subfigures(1, 1), list)
            else fig.subfigures(1, 1)
        )
        shared_subfig.patch.set_facecolor(colors['shared_bg'])
        nonshared_subfig = None
    elif has_nonshared:
        shared_subfig = None
        nonshared_subfig = (
            fig.subfigures(1, 1)[0]
            if isinstance(fig.subfigures(1, 1), list)
            else fig.subfigures(1, 1)
        )
        nonshared_subfig.patch.set_facecolor(colors['nonshared_bg'])
    else:
        # No parameters to plot
        return fig

    # Plot shared parameters with light blue background
    if has_shared and shared_subfig is not None:
        _plot_parameter_section(
            shared_subfig,
            shared_params,
            shared_gradients,
            "SHARED",
            learning_rate,
            bins,
            skip_first_name,
            colors,
            grid_alpha,
        )

    # Plot non-shared parameters with light green background
    if has_nonshared and nonshared_subfig is not None:
        _plot_parameter_section(
            nonshared_subfig,
            nonshared_params,
            nonshared_gradients,
            "LOCAL",
            learning_rate,
            bins,
            skip_first_name,
            colors,
            grid_alpha,
        )

    if show_plot:
        plt.show()
    return fig


class ParamGradLogger(Logger):
    """
    Logs and visualizes parameter and gradient diagnostics during training.

    For each logging step, this logger generates a plot for each replicate,
    showing distributions and statistics for model parameters and their gradients.
    Gradients are averaged over the batches within the step.
    """

    output_dir: str = "plots/params_diagnostics"
    plot_local_params: bool = False
    plot_shared_params: bool = True
    learning_rate: Optional[float] = None  # if None, will try to extract from step_history
    replicate: Optional[int] = None  # if None, will plot all replicates
    grad_aggregation: Literal['mean', 'first'] = 'mean'
    file_format: Literal['png', 'pdf'] = 'png'
    dpi: int = 200

    _save_dir: Optional[Path] = None

    def initialize(self, training_program):
        """Initializes the logger by setting up the output directory."""
        self._save_dir = training_program._save_dir / self.output_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"ParamGradLogger saving plots to {self._save_dir}")

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Returns the callback for logging parameter and gradient diagnostics."""

        def log_param_grads(
            step: int,
            training_config,
            step_history: Optional[dict] = None,
            **kwargs,
        ):
            if step == 0:  # Skip initial state
                return

            if self._save_dir is None:
                logger.warning("ParamGradLogger not initialized, cannot save plots.")
                return

            if step_history is None or 'latest_params' not in step_history:
                logger.warning("No 'latest_params' in step_history for ParamGradLogger.")
                return

            logger.info(f"ParamGradLogger: Generating diagnostic plots for step {step}...")

            params = step_history['latest_params']
            grads = step_history.get('grad')  # Gradients are optional

            # Extract learning rate from step_history if available
            effective_learning_rate = self.learning_rate
            lr_display_text = ""

            if 'learning_rate' in step_history and self.learning_rate is None:
                # learning_rate contains history across all batches in the step
                lr_history = step_history['learning_rate']

                # Extract learning rates for first replicate to check pattern
                first_replicate_lr = tree_get(lr_history, 0)
                if first_replicate_lr is not None:
                    lr_array = np.array(first_replicate_lr)
                    if lr_array.ndim > 0:  # array of learning rates across batches
                        start_lr = float(lr_array[0])
                        end_lr = float(lr_array[-1])
                        mean_lr = float(lr_array.mean())

                        if np.allclose(lr_array, start_lr):
                            lr_display_text = f"learning rate: {start_lr:.2e}"
                        else:
                            lr_display_text = f"learning rate: {start_lr:.2e} → {end_lr:.2e}"

                        effective_learning_rate = mean_lr
                    else:  # scalar learning rate
                        effective_learning_rate = float(lr_array)
                        lr_display_text = f"learning rate: {effective_learning_rate:.2e}"

            n_replicates = training_config.n_replicates
            import time

            times = []
            for i in range(n_replicates):
                t0 = time.time()
                replicate_params = tree_get(params, i)
                replicate_gradients = None
                if grads is not None:
                    # grads contains history across all batches in the step
                    # we need to average them to get the mean gradient for the step
                    replicate_grads_history = tree_get(grads, i)

                    # average gradients across batch history
                    if replicate_grads_history is not None:
                        # the gradient history is structured as a tree where each leaf
                        # contains an array with shape (batches_per_step, *param_shape)
                        if self.grad_aggregation == 'mean':
                            replicate_gradients = jax.tree.map(
                                lambda g: g.mean(axis=0) if g is not None else None,
                                replicate_grads_history,
                            )
                        elif self.grad_aggregation == 'first':
                            replicate_gradients = jax.tree.map(
                                lambda g: g[0] if g is not None else None,
                                replicate_grads_history,
                            )
                        else:
                            logger.warning(
                                f"Unknown grad_aggregation method '{self.grad_aggregation}', "
                                "defaulting to 'mean'."
                            )
                            replicate_gradients = jax.tree.map(
                                lambda g: g.mean(axis=0) if g is not None else None,
                                replicate_grads_history,
                            )

                t1 = time.time()

                # Create title with learning rate info
                main_title = f"Parameter Diagnostics - Step {step}"
                if lr_display_text:
                    main_title += f"\n{lr_display_text}"

                fig = plot_parameter_diagnostics(
                    params=replicate_params,
                    param_gradients=replicate_gradients,
                    learning_rate=effective_learning_rate,
                    title=main_title,
                    show_plot=False,
                    plot_local_params=self.plot_local_params,
                    plot_shared_params=self.plot_shared_params,
                )
                t2 = time.time()

                try:
                    save_path = (
                        self._save_dir / f"replicate{i}/{step:04d}_params.{self.file_format}"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=self.dpi)
                except Exception as e:
                    logger.error(f"Failed to save parameter diagnostic plot: {e}")
                finally:
                    plt.close(fig)
                    t3 = time.time()
                    times.append((t1 - t0, t2 - t1, t3 - t2))

            for i, (t0, t1, t2) in enumerate(times):
                logger.debug(
                    f"Replicate {i}: "
                    f"Param extraction: {t0:.2f}s, "
                    f"Plot generation: {t1:.2f}s, "
                    f"Saving: {t2:.2f}s"
                )

        return [(self.periods, log_param_grads)]

    def finalize(self):
        """Create videos from parameter diagnostic plots using ffmpeg."""
        if self._save_dir is None:
            return

        from biocomptools.toollib.video_utils import create_videos_from_subdirs

        logger.info("ParamGradLogger: Creating videos from diagnostic plots...")

        videos_created = create_videos_from_subdirs(
            base_dir=self._save_dir,
            subdir_pattern="replicate*",
            plot_pattern=f"*.{self.file_format}",
            video_name="param_diagnostics_video.mp4",
        )

        if videos_created > 0:
            logger.info(f"ParamGradLogger: Created {videos_created} diagnostic videos")
        else:
            logger.debug("ParamGradLogger: No videos created (insufficient plots or errors)")
