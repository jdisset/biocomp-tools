from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Annotated, Optional
import numpy as np
from pydantic import BaseModel

from dracon.commandline import make_program, Arg

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    optuna = None

from biocomp.plotting.plotting_core import CUSTOM_CMAPS
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BC_BLUES = CUSTOM_CMAPS['bc_blues']
BC_REDS = CUSTOM_CMAPS['bc_reds']


def _cmap_diverging():
    colors = list(BC_REDS(np.linspace(1, 0, 256))) + list(BC_BLUES(np.linspace(0, 1, 256)))
    return LinearSegmentedColormap.from_list('bc_diverging', colors, N=512)


def _cmap_objective():
    colors = list(BC_BLUES(np.linspace(1, 0, 256))) + list(BC_REDS(np.linspace(0, 1, 256)))
    return LinearSegmentedColormap.from_list('bc_objective', colors, N=512)


def _hide_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


class HyperoptAnalysisConfig(BaseModel):
    db_path: Annotated[str, Arg(positional=True, help='Path to Optuna SQLite database')]
    study: Annotated[
        Optional[str], Arg(short='s', help='Study name (default: infer from filename)')
    ] = None
    output: Annotated[Optional[str], Arg(short='o', help='Output directory for report')] = None
    top_n: Annotated[int, Arg(help='Number of params in bar chart')] = 25
    export: Annotated[Optional[str], Arg(help='Export trials to CSV/JSON file')] = None
    text_only: Annotated[bool, Arg(help='Print text summary only, no PDF')] = False


def load_study(db_path: str | Path, study_name: str | None = None) -> optuna.Study:
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    storage = f"sqlite:///{db_path}"
    if study_name is None:
        study_name = db_path.stem
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        summaries = optuna.get_all_study_summaries(storage=storage)
        available = [s.study_name for s in summaries]
        if len(available) == 1:
            return optuna.load_study(study_name=available[0], storage=storage)
        raise ValueError(f"Study '{study_name}' not found. Available: {available}") from None


def get_trials_data(study: optuna.Study) -> tuple[np.ndarray, np.ndarray, list[str]]:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError("No completed trials found")
    param_names = list(study.best_params.keys())
    values = np.array([t.value for t in completed])
    params = np.array([[t.params[p] for p in param_names] for t in completed])
    return params, values, param_names


def compute_correlations(
    params: np.ndarray, values: np.ndarray, param_names: list[str], log_scale: bool = True
) -> dict[str, float]:
    correlations = {}
    for i, name in enumerate(param_names):
        p = params[:, i]
        if log_scale and np.all(p > 0):
            p = np.log10(p)
        correlations[name] = 0.0 if np.std(p) < 1e-10 else float(np.corrcoef(p, values)[0, 1])
    return correlations


def analyze_study(study: optuna.Study) -> dict:
    params, values, param_names = get_trials_data(study)
    importance = get_param_importances(study)
    correlations = compute_correlations(params, values, param_names)

    beneficial, harmful, neutral = [], [], []
    for name in param_names:
        corr, best, imp = correlations[name], study.best_params[name], importance.get(name, 0)
        entry = (name, corr, best, imp)
        if corr < -0.15 and best > 0.3:
            beneficial.append(entry)
        elif corr > 0.15 and best < 0.1:
            harmful.append(entry)
        else:
            neutral.append(entry)

    beneficial.sort(key=lambda x: x[1])
    harmful.sort(key=lambda x: -x[1])

    return {
        'importance': importance,
        'correlations': correlations,
        'best_params': study.best_params,
        'params': params,
        'values': values,
        'param_names': param_names,
        'beneficial': beneficial,
        'harmful': harmful,
        'neutral': neutral,
        'stats': {
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'n_complete': len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            'median_value': float(np.median(values)),
            'worst_value': float(np.max(values)),
            'p10_value': float(np.percentile(values, 10)),
        },
    }


def plot_comprehensive_report(
    study: optuna.Study, output_path: str | Path, top_n: int = 25, n_scatter: int = 16
):
    from matplotlib.backends.backend_pdf import PdfPages

    analysis = analyze_study(study)
    importance, correlations, best_params = (
        analysis['importance'],
        analysis['correlations'],
        analysis['best_params'],
    )
    params, values, param_names, stats = (
        analysis['params'],
        analysis['values'],
        analysis['param_names'],
        analysis['stats'],
    )

    cmap_div = _cmap_diverging()
    all_params_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    sorted_params = all_params_sorted[:top_n]
    if not sorted_params:
        print("No parameters to plot")
        return

    # page 1: summary
    fig1 = plt.figure(figsize=(22, 14), constrained_layout=False)
    subfigs = fig1.subfigures(2, 1, height_ratios=[2.5, 1], hspace=0.1)
    subfigs_top = subfigs[0].subfigures(1, 2, width_ratios=[1, 1.2], wspace=0.05)

    # panel A: importance bars
    ax_bars = subfigs_top[0].add_subplot(111)
    names, imps = [p[0] for p in sorted_params], [p[1] for p in sorted_params]
    corrs, bests = [correlations[n] for n in names], [best_params[n] for n in names]

    color_low, color_mid, color_high = BC_REDS(0.7), '#888888', BC_BLUES(0.7)

    def get_cat(b):
        return (
            ('low', color_low)
            if b < 0.1
            else (('high', color_high) if b > 1.0 else ('mid', color_mid))
        )

    categories = [get_cat(b) for b in bests]
    colors = [c[1] for c in categories]

    y_pos = np.arange(len(names))
    ax_bars.barh(y_pos, imps, color=colors, edgecolor='#2c3e50', linewidth=0.5)

    for i, (_, imp, _, best, cat) in enumerate(
        zip(names, imps, corrs, bests, [c[0] for c in categories], strict=True)
    ):
        best_str = f'{best:.1e}' if best < 0.01 else (f'{best:.2f}' if best < 1 else f'{best:.1f}')
        ax_bars.annotate(
            f'opt@{cat}: {best_str}',
            xy=(imp, i),
            xytext=(4, 0),
            textcoords='offset points',
            va='center',
            fontsize=7,
            fontfamily='monospace',
            color='#2c3e50',
        )

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels([n.replace('dataweight_', '') for n in names], fontsize=8)
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel('Importance (fANOVA)', fontsize=10)
    ax_bars.set_xlim(0, max(imps) * 2.0)
    ax_bars.set_title(
        'A. Parameter Importance (fANOVA)\n(red=opt@low, gray=opt@mid, blue=opt@high)',
        fontsize=11,
        fontweight='bold',
    )
    _hide_spines(ax_bars)

    # panel B: history
    gs_right = subfigs_top[1].add_gridspec(2, 1, height_ratios=[1, 0.3], hspace=0.3)
    ax_history = subfigs_top[1].add_subplot(gs_right[0])
    trial_numbers = np.arange(len(values))
    ax_history.scatter(trial_numbers, values, s=8, alpha=0.5, c='#3498db', edgecolors='none')
    ax_history.plot(
        trial_numbers, np.minimum.accumulate(values), 'r-', linewidth=2, label='Best so far'
    )
    ax_history.set_xlabel('Trial', fontsize=10)
    ax_history.set_ylabel('Objective', fontsize=10)
    ax_history.set_yscale('log')
    ax_history.set_title('B. Optimization History', fontsize=11, fontweight='bold')
    ax_history.legend(loc='upper right', fontsize=9)
    _hide_spines(ax_history)

    ax_text = subfigs_top[1].add_subplot(gs_right[1])
    ax_text.axis('off')
    text_lines = [
        f"SUMMARY: Best={stats['best_value']:.4f} (trial #{stats['best_trial']}) | "
        f"Trials={stats['n_complete']} | Median={stats['median_value']:.4f} | Top10%={stats['p10_value']:.4f}",
        f"Total parameters: {len(param_names)} | See page 2 for all param vs objective scatter plots",
    ]
    ax_text.text(
        0.02,
        0.8,
        '\n'.join(text_lines),
        transform=ax_text.transAxes,
        fontsize=10,
        fontfamily='monospace',
        va='top',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.8),
    )

    # panel C: best params
    ax_best = subfigs[1].add_subplot(111)
    sorted_best = sorted(best_params.items(), key=lambda x: x[0])
    bp_names, bp_values = (
        [p[0].replace('dataweight_', '') for p in sorted_best],
        [p[1] for p in sorted_best],
    )
    bp_imps = [importance.get(p[0], 0) for p in sorted_best]
    max_imp = max(bp_imps) if bp_imps else 1
    bp_colors = [BC_BLUES(imp / max_imp) for imp in bp_imps]

    ax_best.bar(
        np.arange(len(bp_names)), bp_values, color=bp_colors, edgecolor='#2c3e50', linewidth=0.5
    )
    ax_best.set_yscale('log')
    ax_best.set_xticks(np.arange(len(bp_names)))
    ax_best.set_xticklabels(bp_names, rotation=45, fontsize=6, ha='right')
    ax_best.set_ylabel('Best weight (log)', fontsize=10)
    ax_best.set_title(
        'C. Best Model Parameters (alphabetical, color=importance: darker=more important)',
        fontsize=11,
        fontweight='bold',
    )
    ax_best.axhline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_best.set_xlim(-0.5, len(bp_names) - 0.5)
    _hide_spines(ax_best)
    subfigs[1].subplots_adjust(bottom=0.35)

    fig1.suptitle(
        f'Hyperparameter Optimization Report: {study.study_name}',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    # page 2: scatter plots
    n_params, n_cols = len(all_params_sorted), 8
    n_rows = (n_params + n_cols - 1) // n_cols

    fig2 = plt.figure(figsize=(24, 3 * n_rows + 1), constrained_layout=False)
    gs = fig2.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.45,
        wspace=0.3,
        top=0.94,
        bottom=0.04,
    )

    vmin, vmax = np.percentile(values, 5), np.percentile(values, 95)
    scatter_ref, cmap_scatter = None, _cmap_objective()

    for idx, (pname, imp) in enumerate(all_params_sorted):
        row, col = idx // n_cols, idx % n_cols
        ax = fig2.add_subplot(gs[row, col])

        pidx = param_names.index(pname)
        pvals, corr = params[:, pidx], correlations[pname]

        if np.all(pvals > 0):
            ax.set_xscale('log')

        scatter = ax.scatter(
            pvals,
            values,
            c=values,
            cmap=cmap_scatter,
            s=6,
            alpha=1.0,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax,
        )
        if scatter_ref is None:
            scatter_ref = scatter

        best_idx = np.argmin(values)
        ax.scatter(
            [pvals[best_idx]],
            [values[best_idx]],
            c='lime',
            s=30,
            marker='*',
            zorder=10,
            edgecolors='black',
            linewidths=0.5,
        )

        try:
            from scipy.ndimage import uniform_filter1d

            sort_idx = np.argsort(pvals)
            smoothed = uniform_filter1d(values[sort_idx], size=max(10, len(values) // 20))
            ax.plot(pvals[sort_idx], smoothed, 'k-', linewidth=1, alpha=0.7)
        except Exception:
            pass

        short_name = pname.replace('dataweight_', '')
        if len(short_name) > 34:
            short_name = short_name[:32] + '..'
        direction = '↑' if corr < -0.1 else ('↓' if corr > 0.1 else '·')
        ax.set_title(f'{short_name}\n{direction} r={corr:+.2f} imp={imp:.3f}', fontsize=6)
        ax.tick_params(labelsize=5)
        _hide_spines(ax)

    cbar_ax = fig2.add_subplot(gs[:, n_cols])
    cbar = fig2.colorbar(scatter_ref, cax=cbar_ax)
    cbar.set_label('objective', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig2.suptitle(
        'Parameter vs Objective (all params, ordered by fANOVA importance, model-free)',
        fontsize=13,
        fontweight='bold',
        y=0.995,
    )

    output_path = Path(output_path)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig1, dpi=150, bbox_inches='tight')
        pdf.savefig(fig2, dpi=150, bbox_inches='tight')
        shap_fig = _plot_shap_page(study, analysis, cmap_div)
        if shap_fig is not None:
            pdf.savefig(shap_fig, dpi=150)
            plt.close(shap_fig)

    plt.close(fig1)
    plt.close(fig2)
    print(f"Report saved to {output_path}")


def _select_best_surrogate(X, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    candidates = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, oob_score=True
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1
        ),
    }

    try:
        from xgboost import XGBRegressor

        candidates['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1, n_jobs=-1
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        candidates['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1, n_jobs=-1, verbose=-1
        )
    except ImportError:
        pass

    results = {}
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        results[name] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'model': model}

    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_info = results[best_name]
    best_info['model'].fit(X, y)
    oob_r2 = getattr(best_info['model'], 'oob_score_', None)

    return best_name, best_info['model'], best_info['cv_mean'], best_info['cv_std'], oob_r2, results


def _plot_shap_page(study: optuna.Study, analysis: dict, cmap_div, top_n: int = 25):
    try:
        import shap
        from scipy.stats import spearmanr
    except ImportError:
        return None

    params, values, param_names, importance = (
        analysis['params'],
        analysis['values'],
        analysis['param_names'],
        analysis['importance'],
    )
    log_params = np.where(params > 0, np.log10(params), params)

    model_name, model, cv_r2_mean, cv_r2_std, oob_r2, all_results = _select_best_surrogate(
        log_params, values
    )

    print("\nSurrogate model comparison (5-fold CV R²):")
    for name, info in sorted(all_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True):
        marker = " <-- selected" if name == model_name else ""
        print(f"  {name}: {info['cv_mean']:.3f} ± {info['cv_std']:.3f}{marker}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(log_params)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    shap_ranking = {param_names[i]: rank for rank, i in enumerate(top_indices)}
    fanova_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    fanova_ranking = {name: rank for rank, (name, _) in enumerate(fanova_sorted[:top_n])}
    common_params = set(shap_ranking.keys()) & set(fanova_ranking.keys())
    if len(common_params) > 5:
        rank_corr, _ = spearmanr(
            [shap_ranking[p] for p in common_params], [fanova_ranking[p] for p in common_params]
        )
    else:
        rank_corr = float('nan')

    shap_top, params_top = shap_values[:, top_indices], log_params[:, top_indices]
    names_top = [param_names[i].replace('dataweight_', '') for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 14))

    for i in range(len(names_top)):
        sv, pv = shap_top[:, i], params_top[:, i]
        sort_idx = np.argsort(pv)
        sv_sorted, pv_sorted = sv[sort_idx], pv[sort_idx]
        y_offsets = np.linspace(-0.3, 0.3, len(pv_sorted))
        scatter = ax.scatter(
            sv_sorted,
            i * 1.2 + y_offsets,
            c=pv_sorted,
            cmap=cmap_div,
            s=8,
            alpha=0.6,
            vmin=np.percentile(params_top, 5),
            vmax=np.percentile(params_top, 95),
        )

    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yticks(np.arange(len(names_top)) * 1.2)
    ax.set_yticklabels(names_top, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(
        'SHAP value (impact on objective)\n← reduces objective | increases objective →', fontsize=11
    )

    oob_str = f"OOB R²={oob_r2:.3f} | " if oob_r2 is not None else ""
    diagnostics = f"{oob_str}5-fold CV R²={cv_r2_mean:.3f}±{cv_r2_std:.3f} | SHAP-fANOVA rank corr={rank_corr:.2f}"
    ax.set_title(
        f'SHAP Feature Importance (Surrogate: {model_name})\n{diagnostics}',
        fontsize=12,
        fontweight='bold',
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('log10(param value)', fontsize=10)
    _hide_spines(ax)

    quality = (
        "excellent"
        if cv_r2_mean > 0.8
        else ("good" if cv_r2_mean > 0.6 else ("moderate" if cv_r2_mean > 0.4 else "poor"))
    )
    note = f"Surrogate quality: {quality}. " + (
        "SHAP values are reliable." if cv_r2_mean > 0.6 else "Interpret SHAP with caution."
    )
    fig.text(0.5, 0.01, note, ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def export_trials(
    study: optuna.Study, output_path: str | Path, format: Literal['csv', 'json'] = 'csv'
):
    import json

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if format == 'csv':
        param_names = list(study.best_params.keys())
        with open(output_path, 'w') as f:
            f.write(','.join(['trial', 'value'] + param_names) + '\n')
            for trial in completed:
                row = [str(trial.number), str(trial.value)] + [
                    str(trial.params.get(p, '')) for p in param_names
                ]
                f.write(','.join(row) + '\n')
    else:
        data = [{'trial': t.number, 'value': t.value, 'params': t.params} for t in completed]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"Exported {len(completed)} trials to {output_path}")


def print_summary(study: optuna.Study):
    analysis = analyze_study(study)
    stats = analysis['stats']

    print('=' * 70)
    print('HYPERPARAMETER OPTIMIZATION ANALYSIS')
    print('=' * 70)
    print(f"\nStudy: {study.study_name}")
    print(f"Best value: {stats['best_value']:.6f} (trial #{stats['best_trial']})")
    print(f"Trials: {stats['n_complete']} complete / {stats['n_trials']} total")
    print(f"Median: {stats['median_value']:.6f} | Top 10%: {stats['p10_value']:.6f}")

    print('\n' + '-' * 70)
    print('INCREASE WEIGHT (r < -0.15, optimizer favored higher values):')
    print('-' * 70)
    for name, corr, best, imp in analysis['beneficial'][:12]:
        print(f"  {name.replace('dataweight_', '')}: best={best:.3f}, r={corr:+.3f}, imp={imp:.4f}")

    print('\n' + '-' * 70)
    print('REDUCE WEIGHT (r > 0.15, optimizer favored lower values):')
    print('(Note: low weight ≠ disabled, just lower relative contribution)')
    print('-' * 70)
    for name, corr, best, imp in analysis['harmful'][:12]:
        print(f"  {name.replace('dataweight_', '')}: best={best:.4f}, r={corr:+.3f}, imp={imp:.4f}")

    print('\n' + '-' * 70)
    print('TOP 10 BY IMPORTANCE (fANOVA - variance explained):')
    print('-' * 70)
    sorted_imp = sorted(analysis['importance'].items(), key=lambda x: -x[1])[:10]
    for name, imp in sorted_imp:
        corr, best = analysis['correlations'][name], analysis['best_params'][name]
        direction = '↑HIGH' if corr < -0.1 else ('↓LOW' if corr > 0.1 else 'neutral')
        print(
            f"  {name.replace('dataweight_', '')}: imp={imp:.4f}, {direction} better (best={best:.4f}, r={corr:+.2f})"
        )


def run_analysis(config: HyperoptAnalysisConfig):
    if optuna is None:
        print("Please install optuna: pip install optuna")
        sys.exit(1)

    study = load_study(config.db_path, config.study)
    print(f"Loaded study '{study.study_name}' with {len(study.trials)} trials")

    print_summary(study)

    if not config.text_only:
        output_dir = Path(config.output) if config.output else Path(config.db_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comprehensive_report(
            study, output_dir / f'{study.study_name}_report.pdf', top_n=config.top_n
        )

    if config.export:
        fmt = 'json' if config.export.endswith('.json') else 'csv'
        export_trials(study, config.export, format=fmt)


def main():
    program = make_program(
        HyperoptAnalysisConfig,
        name='biocomp-hyperopt-analysis',
        description='Analyze Optuna hyperparameter optimization results',
    )
    config, _ = program.parse_args(sys.argv[1:])
    run_analysis(config)


if __name__ == '__main__':
    main()
