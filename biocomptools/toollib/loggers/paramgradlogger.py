## {{{                          --     imports     --
import matplotlib.pyplot as plt
import numpy as np
import jax
from math import ceil, sqrt
from pathlib import Path
from pydantic import Field
from typing import Union, Optional, List, Tuple, Callable

from biocomp.parameters import ParameterTree, PTree, flatten_PTree, get_path_components
from biocomp.jaxutils import tree_get
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)
##────────────────────────────────────────────────────────────────────────────}}}


def get_plot_rows_and_columns(num_plots: int, ideal_ratio: float = 1.5):
    n_cols = int(ceil(sqrt(num_plots / ideal_ratio)))
    n_rows = int(ceil(num_plots / n_cols))
    if n_rows == 0:
        n_rows = 1
    if n_cols == 0:
        n_cols = 1
    return n_rows, n_cols


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
            },
            'g': {},
            'ratio': np.nan,
        }

        if has_grads and i < len(gvalues_flat):
            g_flat = gvalues_flat[i]
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

    if g_flat is not None:
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

    last_name = get_path_components(name_tuple)[-1]
    title_color = _get_title_color(last_name, colors)
    layer_name = '/'.join(get_path_components(name_tuple)[1:] if skip_first_name else name_tuple)
    ax.set_title(
        f"{layer_name} ({stats['w']['size']:,})",
        fontsize=11,
        color=title_color,
        y=1.4,
    )

    _add_stats_heatmap(ax, stats, stat_ranges, g_flat is not None, colors)


def plot_parameter_diagnostics(
    params: Union[PTree, ParameterTree],
    param_gradients: Optional[Union[PTree, ParameterTree]] = None,
    learning_rate: Optional[float] = None,
    title: str = "Parameter & Gradient Diagnostics",
    bins: int = 30,
    show_plot: bool = True,
    skip_first_name: bool = True,
    colors=None,
    grid_alpha: float = 0.15,
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
        }

    params_data = params.data if isinstance(params, ParameterTree) else params
    pvalues, (pnames, _) = flatten_PTree(params_data)

    pvalues_flat = []
    pvalues_shapes = []
    for p in pvalues:
        p_arr = np.array(p, dtype=np.float32)
        pvalues_flat.append(p_arr.flatten())
        pvalues_shapes.append(p_arr.shape)

    has_grads = param_gradients is not None
    has_lr = learning_rate is not None

    gvalues_flat = []
    if has_grads:
        grads_data = (
            param_gradients.data if isinstance(param_gradients, ParameterTree) else param_gradients
        )
        gvalues, _ = flatten_PTree(grads_data)
        gvalues_flat = [np.array(g, dtype=np.float32).flatten() for g in gvalues]

    stats_list, stat_ranges = _compute_statistics(
        pvalues_flat, gvalues_flat, pnames, learning_rate, pvalues_shapes
    )

    num_layers = len(pvalues_flat)
    n_rows, n_cols = get_plot_rows_and_columns(num_layers)

    hspace = 1.2
    wspace = 0.7
    top = 0.92
    bottom = 0.02
    left = 0.03
    right = 0.97

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.5),
        gridspec_kw={
            'hspace': hspace,
            'wspace': wspace,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
        },
    )
    fig.suptitle(title, fontsize=16, color=colors['text_color'], y=0.96)
    fig.patch.set_facecolor(colors['bg_color'])

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_facecolor(colors['bg_color'])

    for i, ax in enumerate(axes_flat):
        if i >= num_layers or stats_list[i] is None:
            ax.set_visible(False)
            continue

        _plot_layer(
            ax,
            i,
            pvalues_flat[i],
            gvalues_flat[i] if has_grads else None,
            pnames[i],
            stats_list[i],
            stat_ranges,
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

    output_dir: str = Field(
        default="param_grad_plots",
        description="Subdirectory within the run's save directory to store plots.",
    )
    learning_rate: Optional[float] = Field(
        default=None,
        description="Optional learning rate to compute update/weight ratio. If not provided, this ratio is not computed.",
    )
    dpi: int = Field(default=200, description="DPI for the saved plot images.")

    # Private state
    _save_dir: Optional[Path] = None

    def initialize(self, training_program):
        """Initializes the logger by setting up the output directory."""
        self._save_dir = training_program._save_dir / self.output_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"ParamGradLogger saving plots to {self._save_dir}")

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        """Returns the callback for logging parameter and gradient diagnostics."""

        def log_param_grads(
            step: int, training_config, step_history: Optional[dict] = None, **kwargs
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

            n_replicates = training_config.n_replicates

            for i in range(n_replicates):
                replicate_params = tree_get(params, i)

                replicate_gradients = None
                if grads is not None:
                    grads_for_rep = tree_get(grads, i)
                    if grads_for_rep is not None:
                        # Grads have a leading dimension for batches_per_step, so we average over it.
                        replicate_gradients = jax.tree_util.tree_map(
                            lambda x: x.mean(axis=0), grads_for_rep
                        )

                fig = plot_parameter_diagnostics(
                    params=replicate_params,
                    param_gradients=replicate_gradients,
                    learning_rate=self.learning_rate,
                    title=f"Parameter Diagnostics - Step {step}",
                    show_plot=False,
                )

                try:
                    save_path = (
                        self._save_dir / f"plots/rep{i}/paramsdiag/{step:04d}_paramsdiag.png"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_path, dpi=self.dpi)
                except Exception as e:
                    logger.error(f"Failed to save parameter diagnostic plot: {e}")
                finally:
                    plt.close(fig)

        return [(self.periods, log_param_grads)]
