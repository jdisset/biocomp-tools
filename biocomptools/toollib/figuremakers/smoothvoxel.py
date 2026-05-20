# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Figuremaker utilities for smooth voxel-conditioned violin plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from biocomp.plotting.plotting_smooth import smooth_voxel_conditioned_violin
from biocomptools.logging_config import get_logger
from biocomptools.toollib.figuremakers.quantilecoverage import _load_dataset, _load_model

logger = get_logger(__name__)


def _pick_default_model_path() -> str:
    root = Path("biocomp_traces/quantile_11_l1_uorfs/quantile_11_L1_uORFs_distribution_default")
    models = sorted(root.glob("*/training/*.bestmodel.pickle"))
    if not models:
        raise RuntimeError(f"No model found under {root} to recover rescaler.")
    return str(models[0])


def _to_single_output(y: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] > 1:
        arr = arr[:, :1]
    return arr


def _pick_network_with_min_points(pdata_list: list[Any], min_points: int):
    candidates = [pd for pd in pdata_list if np.asarray(pd.x).shape[0] >= int(min_points)]
    return candidates[0] if candidates else pdata_list[0]


def _pick_two_networks_with_min_points(pdata_list: list[Any], min_points: int):
    candidates = [pd for pd in pdata_list if np.asarray(pd.x).shape[0] >= int(min_points)]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    if len(pdata_list) >= 2:
        return pdata_list[0], pdata_list[1]
    raise ValueError("Need at least two networks to render split smooth voxel figure.")


def _network_labels_from_pdata(pd: Any, x: np.ndarray):
    net = pd.metadata.get("built_network")
    if net is None:
        return [f"x{i}" for i in range(x.shape[1])], "y"
    input_names = net.get_inverted_input_proteins()
    out_names = net.get_output_proteins(only_dependent_outputs=False)
    output_name = out_names[0] if out_names else "y"
    return input_names, output_name


def render_smooth_voxel_example(
    ax: Axes,
    *,
    dataset_file: str,
    mode: str = "single",
    model: Any = None,
    model_path: str | None = None,
    model_name: str | None = None,
    min_points_single: int = 60,
    min_points_split: int = 80,
    xlims: tuple[float, float] = (0.0, 0.7),
    ylims: tuple[float, float] = (0.0, 0.7),
    title: str | None = None,
    **_kwargs,
):
    """Render a smooth voxel-conditioned violin figure for one dataset.

    Args:
        ax: Matplotlib axis.
        dataset_file: Dataset yaml path.
        mode: ``single`` or ``split``.
        model/model_path/model_name: model loader inputs; if none provided, uses
            the first model under the default quantile trace path.
    """
    if model is None and model_path is None and not model_name:
        model_path = _pick_default_model_path()

    loaded_model = _load_model(model=model, model_path=model_path, model_name=model_name)
    rescaler = loaded_model.rescaler
    pdata_list = _load_dataset(dataset_file)
    if not pdata_list:
        raise ValueError(f"No data loaded from dataset_file={dataset_file!r}")

    mode = str(mode).strip().lower()
    if mode == "single":
        pdata = _pick_network_with_min_points(pdata_list, min_points=min_points_single)
        x = np.asarray(pdata.x, dtype=float)
        y = _to_single_output(pdata.y)
        x_lat = np.asarray(rescaler.fwd(x), dtype=float)
        y_lat = np.asarray(rescaler.fwd(y), dtype=float)
        input_names, output_name = _network_labels_from_pdata(pdata, x)
        return smooth_voxel_conditioned_violin(
            X=x_lat,
            Y=y_lat,
            input_names=input_names,
            output_name=output_name,
            rescaler=rescaler,
            ax=ax,
            mode="single",
            title=title,
            xlims=xlims,
            ylims=ylims,
        )

    if mode == "split":
        pdata_l, pdata_r = _pick_two_networks_with_min_points(
            pdata_list, min_points=min_points_split
        )
        xa = np.asarray(pdata_l.x, dtype=float)
        xb = np.asarray(pdata_r.x, dtype=float)
        ya = _to_single_output(pdata_l.y)
        yb = _to_single_output(pdata_r.y)

        xa_lat = np.asarray(rescaler.fwd(xa), dtype=float)
        xb_lat = np.asarray(rescaler.fwd(xb), dtype=float)
        ya_lat = np.asarray(rescaler.fwd(ya), dtype=float)
        yb_lat = np.asarray(rescaler.fwd(yb), dtype=float)

        input_names, output_name = _network_labels_from_pdata(pdata_l, xa)
        return smooth_voxel_conditioned_violin(
            X=(xa_lat, xb_lat),
            Y=(ya_lat, yb_lat),
            input_names=input_names,
            output_name=output_name,
            rescaler=rescaler,
            ax=ax,
            mode="split",
            title=title,
            xlims=xlims,
            ylims=ylims,
        )

    raise ValueError(f"Unknown mode={mode!r}; expected 'single' or 'split'.")


def render_benchmark_distribution(
    ax: Axes,
    *,
    item: Any,
    bench: Any,
    xlims: tuple[float, float] = (0.0, 0.7),
    ylims: tuple[float, float] = (0.0, 0.7),
    show_marginal_kde: bool = False,
    tick_count: int = 5,
    grid_resolution: int = 32,
    draw_xlabel: bool = False,
    draw_ylabel: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Render a split smooth voxel violin comparing measured vs predicted for a benchmark item.

    Left violin = ground truth, right violin = model prediction.
    Data is converted from raw to latent space via the model's rescaler.
    """
    rescaler = bench.loaded_model.rescaler

    x_gt = np.asarray(item.gt_data.x, dtype=float)
    y_gt = _to_single_output(item.gt_data.y)
    x_pred = np.asarray(item.pred_data.x, dtype=float)
    y_pred = _to_single_output(item.pred_data.y)

    x_gt_lat = np.asarray(rescaler.fwd(x_gt), dtype=float)
    y_gt_lat = np.asarray(rescaler.fwd(y_gt), dtype=float)
    x_pred_lat = np.asarray(rescaler.fwd(x_pred), dtype=float)
    y_pred_lat = np.asarray(rescaler.fwd(y_pred), dtype=float)

    input_names, output_name = _network_labels_from_pdata(item.gt_data, x_gt)

    return smooth_voxel_conditioned_violin(
        X=(x_gt_lat, x_pred_lat),
        Y=(y_gt_lat, y_pred_lat),
        input_names=input_names,
        output_name=output_name,
        rescaler=rescaler,
        ax=ax,
        mode="split",
        xlims=xlims,
        ylims=ylims,
        show_marginal_kde=show_marginal_kde,
        tick_count=tick_count,
        grid_resolution=grid_resolution,
        draw_xlabel=draw_xlabel,
        draw_ylabel=draw_ylabel,
        **kwargs,
    )

