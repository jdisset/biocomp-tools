import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Any
from pydantic import ConfigDict
import csv

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.design_results import (
    DesignResultsManager,
    compute_design_metrics,
    extract_recipe_summary,
    compute_nre_for_network,
    NREMetrics,
)
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


def make_lattice_grid(res: Tuple[int, int], xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0) -> np.ndarray:
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, res[0]), np.linspace(ymin, ymax, res[1]))
    return np.column_stack([xx.ravel(), yy.ravel()])


class DesignSummaryLogger(Logger):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_period: int = 500
    log_at_end: bool = True
    topk_per_target: int = 3
    generate_circuit_diagrams: bool = True
    generate_histograms: bool = True
    output_formats: List[str] = ["png"]
    dpi: int = 150
    figsize: Tuple[float, float] = (18, 14)
    grid_resolution: Tuple[int, int] = (48, 48)
    max_evals: int = 100000
    eval_seed: int = 42

    output_dir: Optional[str] = None
    model: Optional[Any] = None
    targets: Optional[List[Any]] = None
    dmanager: Optional[Any] = None

    _results_manager: Optional[DesignResultsManager] = None
    _loss_history: List[float] = []
    _step_count: int = 0
    _all_metrics: List[dict] = []
    _cached_stack: Optional[Any] = None

    def initialize(self, training_program=None):
        if self.output_dir:
            design_dir = Path(self.output_dir) / 'design'
            self._results_manager = DesignResultsManager(design_dir)
            self._all_metrics = []
            logger.info(f"DesignSummaryLogger initialized: {design_dir}")

    def _get_top_candidates(
        self, all_losses: np.ndarray, target_id: int, n: int
    ) -> List[Tuple[int, int, float]]:
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 2:
            all_losses = all_losses[None, :, :]
        elif all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)

        n_networks = all_losses.shape[-1]
        target_losses = all_losses[:, target_id, :].ravel()
        top_indices = np.argsort(target_losses)[:n]

        return [
            (int(i // n_networks), int(i % n_networks), float(target_losses[i]))
            for i in top_indices
        ]

    def _render_heatmap_panel(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'viridis',
        diverging: bool = False,
    ):
        x, y = np.asarray(x), np.asarray(y).ravel()
        if x.ndim == 1 or x.shape[1] == 1:
            ax.scatter(range(len(y)), y, c=y, cmap=cmap, s=3, alpha=0.7, vmin=vmin, vmax=vmax)
            ax.set_xlabel('Sample')
            ax.set_title(title)
            return

        n_points = len(y)
        res = int(np.sqrt(n_points))
        if res * res == n_points:
            xs, ys = np.unique(x[:, 0]), np.unique(x[:, 1])
            if len(xs) * len(ys) == n_points:
                z_grid = y.reshape(len(ys), len(xs))
                extent = [xs.min(), xs.max(), ys.min(), ys.max()]
                if diverging:
                    vabs = max(abs(vmin or z_grid.min()), abs(vmax or z_grid.max()))
                    vmin, vmax = -vabs, vabs
                im = ax.imshow(
                    z_grid,
                    extent=extent,
                    origin='lower',
                    aspect='equal',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='bilinear',
                )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel('Input 1')
                ax.set_ylabel('Input 2')
                ax.set_title(title)
                return

        if diverging:
            vabs = max(abs(vmin or y.min()), abs(vmax or y.max()))
            vmin, vmax = -vabs, vabs
        sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, s=3, alpha=0.7, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('Input 1')
        ax.set_ylabel('Input 2')
        ax.set_title(title)
        ax.set_aspect('equal')

    def _render_target_panel(self, ax: plt.Axes, target: Any, title: str = "Target"):
        from biocomp.design import Target

        if isinstance(target, Target) and hasattr(target, 'path'):
            try:
                import cairosvg
                from PIL import Image
                import io

                png_data = cairosvg.svg2png(
                    url=str(target.path), output_width=400, output_height=400
                )
                img = Image.open(io.BytesIO(png_data))
                ax.imshow(img, extent=[0, 1, 0, 1], origin='lower', aspect='equal')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Input 1')
                ax.set_ylabel('Input 2')
                ax.set_title(title)
                return
            except Exception as e:
                logger.debug(f"SVG rendering failed: {e}, falling back to sampling")

        try:
            if hasattr(target, 'get_lattice'):
                X, Y = target.get_lattice(self.grid_resolution)
            elif hasattr(target, 'get_samples'):
                n = self.grid_resolution[0] * self.grid_resolution[1]
                X, Y = target.get_samples(n=n, grid=self.grid_resolution)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f'Unknown target type:\n{type(target).__name__}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                return
            self._render_heatmap_panel(ax, np.asarray(X), np.asarray(Y).squeeze(), title)
        except Exception as e:
            logger.warning(f"Target rendering failed: {e}")
            ax.text(
                0.5,
                0.5,
                f'Target failed:\n{str(e)[:50]}',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            ax.set_title(title)

    def _render_metrics_panel(self, ax: plt.Axes, metrics):
        ax.axis('off')
        lines = ["Design Metrics", "─" * 30, "", f"Loss: {metrics.loss.total:.6f}"]
        if metrics.loss.sinkhorn is not None:
            lines.append(f"  Sinkhorn: {metrics.loss.sinkhorn:.6f}")
        if metrics.loss.lncc is not None:
            lines.append(f"  LNCC: {metrics.loss.lncc:.6f}")

        if metrics.nre is not None:
            lines.extend(["", "NRE (Noise-Relative Error):"])
            if metrics.nre.design_nre is not None:
                status = ""
                if metrics.nre.baseline_nre is not None:
                    status = (
                        " ✓"
                        if metrics.nre.design_nre <= metrics.nre.baseline_nre * 1.5
                        else (" ✗" if metrics.nre.design_nre > 10.0 else "")
                    )
                lines.append(f"  Design: {metrics.nre.design_nre:.2f}{status}")
            if metrics.nre.baseline_nre is not None:
                lines.append(f"  Baseline: {metrics.nre.baseline_nre:.2f} (target)")
            if metrics.nre.data_nrmse is not None:
                lines.append(f"  Data noise: {metrics.nre.data_nrmse:.3f}")

        r = metrics.regression
        d = metrics.distribution
        lines.extend(
            [
                "",
                "Regression:",
                f"  RMSE: {r.rmse:.4f}",
                f"  MAE: {r.mae:.4f}",
                f"  R²: {r.r2:.4f}",
                f"  Pearson r: {r.pearson_r:.4f}",
                f"  Max Error: {r.max_error:.4f}",
                "",
                "Distribution:",
                f"  Target: {d.target_mean:.3f} ± {d.target_std:.3f}",
                f"  Pred: {d.prediction_mean:.3f} ± {d.prediction_std:.3f}",
                "",
                "Design Info:",
                f"  Rank: {metrics.rank}",
                f"  Replicate: {metrics.replicate_id}",
                f"  Step: {metrics.step}",
            ]
        )
        ax.text(
            0.05,
            0.95,
            '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    def _render_loss_history_panel(
        self, ax: plt.Axes, loss_history: List[float], current_loss: float
    ):
        if not loss_history:
            ax.text(0.5, 0.5, 'No loss history', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Loss History")
            return
        ax.semilogy(range(len(loss_history)), loss_history, 'b-', alpha=0.7, linewidth=1)
        ax.axhline(
            y=current_loss,
            color='r',
            linestyle='--',
            alpha=0.7,
            label=f'Current: {current_loss:.4f}',
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title("Loss History")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _render_histogram_panel(self, ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray):
        yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        bins = np.linspace(min(yt.min(), yp.min()), max(yt.max(), yp.max()), 30)
        ax.hist(yt, bins=bins, alpha=0.5, label='Target', density=True)
        ax.hist(yp, bins=bins, alpha=0.5, label='Prediction', density=True)
        ax.set_xlabel('Output')
        ax.set_ylabel('Density')
        ax.set_title("Distribution")
        ax.legend(fontsize=8)

    def _render_recipe_panel(self, ax: plt.Axes, recipe_summary: dict):
        ax.axis('off')
        lines = [
            "Recipe Summary",
            "─" * 30,
            "",
            f"Network: {recipe_summary.get('network_name', 'N/A')[:40]}",
            "",
        ]
        if uorfs := recipe_summary.get('uorfs', []):
            lines.append("uORFs:")
            for u in uorfs[:6]:
                lines.append(
                    f"  {u.get('node_id', '?')}: {u.get('value', '?')}"
                    if isinstance(u, dict)
                    else f"  {u}"
                )
        if ratios := recipe_summary.get('ratios', {}):
            lines.extend(["", "Ratios:"])
            for cotx, vals in list(ratios.items())[:3]:
                if isinstance(vals, (list, tuple)):
                    vs = ', '.join(f'{v:.2f}' for v in vals[:4]) + ('...' if len(vals) > 4 else '')
                    lines.append(f"  {cotx}: [{vs}]")
        ax.text(
            0.05,
            0.95,
            '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        )

    def _render_comprehensive_figure(
        self,
        target: Any,
        network: Any,
        x_data: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics,
        loss_history: List[float],
        recipe_summary: dict,
        output_path: Path,
    ):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        ax_target, ax_pred, ax_diff = (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
        )
        self._render_target_panel(ax_target, target, "Target Data")
        vmin, vmax = (
            min(np.nanmin(y_true), np.nanmin(y_pred)),
            max(np.nanmax(y_true), np.nanmax(y_pred)),
        )
        self._render_heatmap_panel(
            ax_pred,
            x_data,
            y_pred,
            f"Prediction (loss={metrics.loss.total:.4f})",
            vmin=vmin,
            vmax=vmax,
        )
        self._render_heatmap_panel(
            ax_diff,
            x_data,
            np.asarray(y_pred).ravel() - np.asarray(y_true).ravel(),
            "Difference (Pred - Target)",
            diverging=True,
            cmap='RdBu_r',
        )

        ax_diagram, ax_circuit, ax_metrics = (
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        )
        if self.generate_circuit_diagrams and network is not None:
            try:
                from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

                render_diagram_to_ax(network, ax_diagram, simplified=True, title="Network Diagram")
            except Exception as e:
                logger.warning(f"Failed to render network diagram: {e}")
                ax_diagram.text(
                    0.5,
                    0.5,
                    f'Diagram failed:\n{str(e)[:50]}',
                    ha='center',
                    va='center',
                    transform=ax_diagram.transAxes,
                    fontsize=8,
                )
                ax_diagram.set_title("Network Diagram")
            try:
                from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

                render_circuit_to_ax(
                    network, ax_circuit, hide_marker_tus=True, title="Genetic Circuit"
                )
            except Exception as e:
                logger.warning(f"Failed to render circuit: {e}")
                ax_circuit.text(
                    0.5,
                    0.5,
                    f'Circuit failed:\n{str(e)[:50]}',
                    ha='center',
                    va='center',
                    transform=ax_circuit.transAxes,
                    fontsize=8,
                )
                ax_circuit.set_title("Genetic Circuit")
        else:
            for a, t in [(ax_diagram, "Network Diagram"), (ax_circuit, "Genetic Circuit")]:
                a.text(
                    0.5,
                    0.5,
                    f'{t.split()[0]} disabled',
                    ha='center',
                    va='center',
                    transform=a.transAxes,
                )
                a.set_title(t)
        self._render_metrics_panel(ax_metrics, metrics)

        ax_loss, ax_hist, ax_recipe = (
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
        )
        self._render_loss_history_panel(ax_loss, loss_history, metrics.loss.total)
        if self.generate_histograms:
            self._render_histogram_panel(ax_hist, y_true, y_pred)
        else:
            ax_hist.text(
                0.5,
                0.5,
                'Histogram disabled',
                ha='center',
                va='center',
                transform=ax_hist.transAxes,
            )
            ax_hist.set_title("Distribution")
        self._render_recipe_panel(ax_recipe, recipe_summary)

        target_name = getattr(target, 'name', 'Unknown Target')
        network_name = getattr(network, 'name', 'Unknown') if network else 'N/A'
        fig.suptitle(
            f"Target: {target_name[:60]}\nRank {metrics.rank}: {network_name[:40]} | Rep {metrics.replicate_id} | "
            f"Loss: {metrics.loss.total:.4f} | RMSE: {metrics.regression.rmse:.4f} | R²: {metrics.regression.r2:.3f}",
            fontsize=11,
            fontweight='bold',
        )
        for fmt in self.output_formats:
            plt.savefig(
                output_path.with_suffix(f'.{fmt}'),
                dpi=self.dpi,
                bbox_inches='tight',
                facecolor='white',
            )
        plt.close(fig)

    def _get_or_build_stack(self, stack: Any = None) -> Any:
        if stack is not None:
            self._cached_stack = stack
            return stack
        if self._cached_stack is not None:
            return self._cached_stack
        if self.dmanager is not None and self.model is not None:
            try:
                self._cached_stack = self.dmanager.build_stack(self.model)
                logger.info("Built stack from dmanager and model")
                return self._cached_stack
            except Exception as e:
                logger.warning(f"Failed to build stack: {e}")
        return None

    def _generate_summaries(
        self, step: int, params: Any, stack: Any, all_losses: np.ndarray, is_final: bool = False
    ):
        stack = self._get_or_build_stack(stack)
        if self._results_manager is None:
            logger.warning("Results manager not initialized")
            return
        if stack is None:
            logger.warning("No stack available for summary generation")
            return

        targets = self.targets or (self.dmanager.targets if self.dmanager else [])
        all_losses = np.asarray(all_losses)
        if all_losses.ndim == 4:
            all_losses = np.mean(all_losses, axis=1)
        n_targets = all_losses.shape[-2] if all_losses.ndim >= 2 else 1

        for target_id in range(min(n_targets, len(targets))):
            target = targets[target_id]
            target_name = getattr(target, 'name', f'target_{target_id}')
            candidates = self._get_top_candidates(all_losses, target_id, self.topk_per_target)
            self._results_manager.save_rankings(
                target_name, candidates, step=None if is_final else step
            )

            for rank, (rep_id, net_id, loss) in enumerate(candidates, 1):
                try:
                    self._process_single_design(
                        params, stack, target, target_id, rep_id, net_id, loss, rank, step, is_final
                    )
                except Exception as e:
                    logger.error(f"Failed rank {rank} for {target_name}: {e}")
                    logger.exception(e)

        if is_final:
            self._generate_comparison_outputs()

    def _process_single_design(
        self,
        params: Any,
        stack: Any,
        target: Any,
        target_id: int,
        rep_id: int,
        net_id: int,
        loss: float,
        rank: int,
        step: int,
        is_final: bool,
    ):
        target_name = getattr(target, 'name', f'target_{target_id}')
        rank_dir = self._results_manager.get_rank_dir(
            target_name, rank, step=None if is_final else step
        )

        try:
            import jax

            specific_params = jax.tree.map(lambda x: x[rep_id, target_id], params)
            committed = stack.commit(specific_params)
            network = committed[net_id] if net_id < len(committed) else committed[0]
        except Exception as e:
            logger.warning(f"Failed to commit network: {e}")
            logger.exception(e)
            network = stack.networks[net_id] if net_id < len(stack.networks) else stack.networks[0]

        try:
            x_data, y_true, y_pred = self._get_evaluation_data(target, network, target_id)
        except Exception as e:
            logger.warning(f"Failed to get evaluation data: {e}")
            n = self.grid_resolution[0] * self.grid_resolution[1]
            x_data, y_true, y_pred = (
                make_lattice_grid(self.grid_resolution),
                np.random.rand(n),
                np.random.rand(n),
            )

        network_name = getattr(network, 'name', f'network_{net_id}')
        nre_metrics = None
        from biocomp.design import DataTarget

        if isinstance(target, DataTarget):
            design_nre, design_nrmse, data_nrmse = compute_nre_for_network(
                target, network, self.model, max_evals=self.max_evals
            )
            baseline_nre, baseline_nrmse = (
                (None, None)
                if target.original_network is None
                else compute_nre_for_network(
                    target, target.original_network, self.model, max_evals=self.max_evals
                )[:2]
            )
            nre_metrics = NREMetrics(
                design_nre=design_nre,
                baseline_nre=baseline_nre,
                design_nrmse=design_nrmse,
                baseline_nrmse=baseline_nrmse,
                data_nrmse=data_nrmse,
            )
            if design_nre is not None:
                logger.info(
                    f"  [{target_name} rank {rank}] Design NRE: {design_nre:.2f}"
                    + (f" (baseline: {baseline_nre:.2f})" if baseline_nre else "")
                )

        metrics = compute_design_metrics(
            y_true,
            y_pred,
            loss,
            target_name,
            network_name,
            rep_id,
            net_id,
            rank,
            step,
            nre_metrics=nre_metrics,
        )
        metrics.to_json(rank_dir / 'metrics.json')
        self._all_metrics.append(metrics.to_dict())
        self._save_recipe(network, rank_dir / 'recipe.yaml')
        self._save_evaluation_data(
            rank_dir, x_data, y_true, y_pred, target_name, network_name, step
        )
        self._render_comprehensive_figure(
            target,
            network,
            x_data,
            y_true,
            y_pred,
            metrics,
            self._loss_history.copy(),
            extract_recipe_summary(network, params),
            rank_dir / 'design_summary',
        )

    def _save_recipe(self, network: Any, output_path: Path):
        try:
            import dracon

            with open(output_path, 'w') as f:
                f.write(dracon.dump(network.to_recipe()))
        except Exception as e:
            logger.warning(f"Failed to save recipe: {e}")
            output_path.write_text(
                f"# Recipe extraction failed: {e}\nnetwork_name: {getattr(network, 'name', 'unknown')}\n"
            )

    def _save_evaluation_data(
        self,
        rank_dir: Path,
        x_data: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
        network_name: str,
        step: int,
    ):
        import json

        (rank_dir / 'evaluation_data.json').write_text(
            json.dumps(
                {
                    'x': x_data.tolist(),
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'target_name': target_name,
                    'network_name': network_name,
                    'step': step,
                    'grid_resolution': list(self.grid_resolution),
                    'max_evals': self.max_evals,
                },
                indent=2,
            )
        )
        np.savez_compressed(
            rank_dir / 'evaluation_data.npz', x=x_data, y_true=y_true, y_pred=y_pred
        )

    def _get_evaluation_data(
        self, target: Any, network: Any, target_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if hasattr(target, 'get_lattice'):
            X, Y_true = target.get_lattice(self.grid_resolution)
        elif hasattr(target, 'get_samples'):
            X, Y_true = target.get_samples(
                n=self.grid_resolution[0] * self.grid_resolution[1], grid=self.grid_resolution
            )
        else:
            X, Y_true = (
                make_lattice_grid(self.grid_resolution),
                np.zeros(self.grid_resolution[0] * self.grid_resolution[1]),
            )
        X, Y_true = np.asarray(X), np.asarray(Y_true).squeeze()

        if self.model is not None:
            try:
                from biocomptools.modelmodel import NetworkModel
                from biocomptools.toollib.networkprediction import NetworkPrediction

                predictor = NetworkPrediction(
                    predict_at=[X],
                    network_model=NetworkModel(network=[network], model=self.model),
                    max_evals=self.max_evals,
                    z_value='uniform',
                    verbose=False,
                    enable_gridstats=False,
                    already_latent=True,
                )
                Y_pred = np.asarray(predictor.get_data()[0].y).squeeze()
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                Y_pred = np.zeros_like(Y_true)
        else:
            Y_pred = np.zeros_like(Y_true)
        return X, Y_true, Y_pred

    def _generate_comparison_outputs(self):
        if self._results_manager is None or not self._all_metrics:
            return
        comp_dir = self._results_manager.get_comparison_dir()
        csv_path = comp_dir / 'metrics_table.csv'
        if self._all_metrics:
            flat = [
                {
                    'target': m['target_name'],
                    'network': m['network_name'],
                    'rank': m['rank'],
                    'replicate': m['replicate_id'],
                    'loss': m['loss']['total'],
                    'rmse': m['regression']['rmse'],
                    'mae': m['regression']['mae'],
                    'r2': m['regression']['r2'],
                    'pearson_r': m['regression']['pearson_r'],
                }
                for m in self._all_metrics
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat[0].keys())
                writer.writeheader()
                writer.writerows(flat)
            logger.info(f"Saved metrics table to {csv_path}")
        self._plot_loss_comparison(comp_dir / 'loss_comparison.png')
        self._plot_all_targets_summary(comp_dir / 'all_targets_summary.png')

    def _plot_loss_comparison(self, output_path: Path):
        if not self._loss_history:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(range(len(self._loss_history)), self._loss_history, 'b-', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log)')
        ax.set_title('Design Optimization Progress')
        ax.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved loss comparison to {output_path}")

    def _plot_all_targets_summary(self, output_path: Path):
        rank1 = [m for m in self._all_metrics if m['rank'] == 1]
        if not rank1:
            return
        n_targets, n_cols = len(rank1), min(4, len(rank1))
        n_rows = (n_targets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        for idx, m in enumerate(rank1):
            ax = axes[idx // n_cols, idx % n_cols]
            loss, rmse, r2 = m['loss']['total'], m['regression']['rmse'], m['regression']['r2']
            ax.bar(
                ['Loss', 'RMSE', 'R²'], [loss, rmse, r2], color=['steelblue', 'coral', 'seagreen']
            )
            ax.set_title(f"{m['target_name'][:30]}\nLoss={loss:.4f}")
            ax.set_ylim(0, max(1, loss * 1.2))
        for idx in range(n_targets, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        plt.suptitle('Best Design per Target (Rank 1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved all targets summary to {output_path}")

    def get_callbacks(self, training_program=None) -> List[Tuple[int, Callable]]:
        def periodic_callback(step, training_config, step_history=None, stack=None, **kwargs):
            self._step_count = step
            if step_history and 'loss' in step_history:
                lv = step_history['loss']
                self._loss_history.append(
                    lv.item()
                    if hasattr(lv, 'item')
                    else float(np.mean(lv))
                    if isinstance(lv, np.ndarray)
                    else lv
                )
            if step_history is None:
                return
            all_losses, params = step_history.get('all_losses'), step_history.get('latest_params')
            if all_losses is None or params is None:
                return
            try:
                self._generate_summaries(step, params, stack, all_losses, is_final=False)
            except Exception as e:
                logger.error(f"Summary generation failed at step {step}: {e}")
                logger.exception(e)

        callbacks = [(self.log_period, periodic_callback)]
        if self.log_at_end:

            def final_callback(step, training_config, step_history=None, stack=None, **kwargs):
                if step_history is None:
                    return
                all_losses, params = (
                    step_history.get('all_losses'),
                    step_history.get('latest_params'),
                )
                if all_losses is None or params is None:
                    logger.warning(
                        f"Missing data for final summary: all_losses={all_losses is not None}, params={params is not None}"
                    )
                    return
                try:
                    self._generate_summaries(step, params, stack, all_losses, is_final=True)
                except Exception as e:
                    logger.error(f"Final summary generation failed: {e}")
                    logger.exception(e)

            callbacks.append((-1, final_callback))
        return callbacks

    def get_metrics(self, replicate: Optional[int] = None) -> Optional[dict]:
        return {
            'summaries_generated': self._step_count,
            'loss_history_length': len(self._loss_history),
        }

    def finalize(self):
        logger.info(f"DesignSummaryLogger finalized at step {self._step_count}")
