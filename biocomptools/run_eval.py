# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""biocomp-eval: Fast batch evaluation of trained models against datasets.

Builds one compute stack, then iterates over models with fast param swaps.
Saves predictions as Parquet + optionally renders plots.
"""

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Annotated
from dracon.commandline import Arg, dracon_program
import glob as globmod
import numpy as np

from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.toollib.networkprediction import NetworkPrediction
from biocomptools.toollib.datasources import DBSource
from biocomp.plotutils import PlotData

setup_logging()
logger = get_logger(__name__)


def _load_ground_truth(eval_datasets_dir: str) -> list[PlotData]:
    """Load ground truth from all basic set YAMLs in a directory."""
    from biocomptools.toollib.networkselector import NetworkSet, CleanupFilter, iRegex, NetworkSetUnion, NetworkSetDifference
    from dracon import DraconLoader

    loader = DraconLoader(context={
        "CleanupFilter": CleanupFilter, "NetworkSet": NetworkSet,
        "NetworkSetUnion": NetworkSetUnion, "NetworkSetDifference": NetworkSetDifference,
        "iRegex": iRegex, "DBSource": DBSource,
    })

    all_gt = []
    for yaml_path in sorted(Path(eval_datasets_dir).glob("*.yaml")):
        try:
            dataset = loader.load(str(yaml_path))
            gt = DBSource(content=dataset).get_data()
            # Filter out networks that don't have built_network metadata
            gt = [d for d in gt if d.metadata.get('built_network') is not None]
            all_gt.extend(gt)
        except Exception as e:
            logger.warning(f"Skipping {yaml_path.name}: {e}")
    logger.info(f"Loaded {len(all_gt)} networks from {eval_datasets_dir}")
    return all_gt


def _save_stats_json(output_dir: Path, results: list[PlotData], model_signature: str):
    """Save lightweight stats JSON - one file per model with all network stats."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for r in results:
        stats = r.metadata.get("prediction_stats", {})
        all_stats.append({
            "network_name": stats.get("network_name", r.metadata.get("network_name", "")),
            **{k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))},
        })

    fpath = output_dir / "stats.json"
    with open(fpath, "w") as f:
        json.dump({"model_signature": model_signature, "networks": all_stats}, f, indent=1, default=str)
    logger.info(f"Saved stats for {len(all_stats)} networks to {fpath}")


def _render_plots(
    results: list[PlotData],
    ground_truth: list[PlotData],
    output_dir: Path,
    rescaler,
    nworkers: int,
):
    """Render prediction plots using the smooth auto-dispatch system."""
    from biocomp.plotutils import smooth
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir.mkdir(parents=True, exist_ok=True)

    def render_one(i: int) -> str | None:
        gt, pred = ground_truth[i], results[i]
        name = gt.metadata.get('network_name', f'network_{i}')
        safe_name = name.replace('/', '_').replace(' ', '_')

        try:
            ndim = gt.dimensions.input
            stats = pred.metadata.get('prediction_stats', {})
            rmse = stats.get('rmse', stats.get('grid_rmse', None))
            rmse_str = f" (rmse: {rmse:.4f})" if rmse is not None else ""

            fast_grid = {'knn_grid_params': {'grid_resolution': 50}}
            if ndim == 3:
                kw = {
                    'zslices': [np.linspace(0, 0.6, 15)],
                    'smooth_3d_params': {'smooth_2d_params': fast_grid},
                }
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            elif ndim == 2:
                kw = {'smooth_2d_params': fast_grid}
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            else:
                kw = {}
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            short_name = safe_name.split('_')[-1] if '_' in safe_name else safe_name
            fig.suptitle(short_name)

            smooth(gt, ax1, rescaler, title="Ground Truth", **kw)
            smooth(pred, ax2, rescaler, title=f"Predictions{rmse_str}", **kw)

            fpath = output_dir / f"{safe_name}_combined_pred.pdf"
            fig.savefig(fpath, bbox_inches='tight')
            plt.close(fig)
            return None
        except Exception as e:
            plt.close('all')
            return f"{name}: {e}"

    errors = []
    if nworkers <= 1:
        for i in range(len(results)):
            err = render_one(i)
            if err:
                errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            futs = {pool.submit(render_one, i): i for i in range(len(results))}
            for fut in as_completed(futs):
                err = fut.result()
                if err:
                    errors.append(err)

    if errors:
        logger.warning(f"{len(errors)} plot errors: {errors[:5]}")
    logger.info(f"Rendered {len(results) - len(errors)}/{len(results)} plots to {output_dir}")


@dracon_program(
    name='biocomp-eval',
    description='Fast batch evaluation: predict + stats + optional plots for multiple models.',
    auto_context=True,
)
class EvalProgram(BaseModel):
    model_paths: Annotated[list[str], Arg(help="Model pickle paths (glob patterns supported). Use @file.txt to read paths from file.")]
    eval_datasets: Annotated[str, Arg(help="Directory of basic set YAMLs")]
    output_dir: Annotated[str, Arg(short='o', help="Output directory for results")]
    skip_plots: Annotated[bool, Arg(help="Skip rendering plots (predict + save only)")] = False
    stats_only: Annotated[bool, Arg(help="Save only stats JSON (no raw Parquet data)")] = False
    plot_only: Annotated[bool, Arg(help="Only render plots from existing Parquet results")] = False
    nworkers: Annotated[int, Arg(short='j', help="Parallel plot workers")] = 4
    device: Annotated[str, Arg(help="Prediction device")] = 'gpu'

    def run(self):
        output_dir = Path(self.output_dir)

        # Expand globs and @file references
        model_files = []
        for pattern in self.model_paths:
            if pattern.startswith('@'):
                with open(pattern[1:]) as f:
                    model_files.extend(l.strip() for l in f if l.strip())
            else:
                model_files.extend(sorted(globmod.glob(pattern)))
        if not model_files:
            logger.error(f"No model files found for patterns: {self.model_paths}")
            return

        if self.plot_only:
            self._plot_from_saved(output_dir, model_files)
            return

        # Load all ground truth ONCE
        all_gt = _load_ground_truth(self.eval_datasets)
        all_networks = [d.metadata['built_network'] for d in all_gt]

        # Group networks by actual input count (can't mix 1D/2D/3D in one prediction)
        from collections import defaultdict
        dim_groups: dict[int, list[int]] = defaultdict(list)
        for i, d in enumerate(all_gt):
            net = all_networks[i]
            ndim = net.nb_inputs
            dim_groups[ndim].append(i)
        logger.info(f"Network groups by input dim: {dict((k, len(v)) for k, v in dim_groups.items())}")

        # Build one NetworkModel per dimension group (stack + JIT compiled once each)
        first_model = BiocompModel.load(model_files[0])
        group_models: dict[int, NetworkModel] = {}
        for ndim, indices in dim_groups.items():
            networks = [all_networks[i] for i in indices]
            logger.info(f"Building stack for {ndim}D networks ({len(networks)} networks)")
            group_models[ndim] = NetworkModel(model=first_model, network=networks)

        logger.info(f"Processing {len(model_files)} models.")

        for model_path in model_files:
            model = BiocompModel.load(model_path)
            model_output = output_dir / model.signature
            logger.info(f"Predicting with {model.signature} ({model_path})")

            all_results: list[PlotData] = [None] * len(all_gt)  # type: ignore
            all_dim_results: list[PlotData] = []  # accumulated across dim groups

            for ndim, indices in dim_groups.items():
                nm = group_models[ndim]
                nm.swap_params(model)

                gt_group = [all_gt[i] for i in indices]
                pred = NetworkPrediction(
                    predict_at=[d.x[:, :ndim] for d in gt_group],
                    ground_truth=[d.y for d in gt_group],
                    network_model=nm,
                    device=self.device,
                )
                results = pred.get_data()
                all_dim_results.extend(results)

                if not self.stats_only:
                    pred.save_results(model_output, prediction_data=results, ground_truth=gt_group)

                for j, idx in enumerate(indices):
                    all_results[idx] = results[j]

            if self.stats_only:
                _save_stats_json(model_output, all_dim_results, model.signature)

            if not self.skip_plots:
                valid = [(all_gt[i], all_results[i]) for i in range(len(all_gt)) if all_results[i] is not None]
                _render_plots(
                    [r for _, r in valid],
                    [g for g, _ in valid],
                    model_output,
                    model.rescaler,
                    self.nworkers,
                )

    def _plot_from_saved(self, output_dir: Path, model_files: list[str]):
        """Re-render plots from existing Parquet files."""
        for model_path in model_files:
            model = BiocompModel.load(model_path)
            model_dir = output_dir / model.signature
            if not model_dir.exists():
                logger.warning(f"No results for {model.signature}, skipping")
                continue

            pairs = NetworkPrediction.load_results(model_dir)
            gt_list = [gt for gt, _ in pairs]
            pred_list = [pred for _, pred in pairs]
            _render_plots(pred_list, gt_list, model_dir, self.nworkers)


def main():
    EvalProgram.cli()
