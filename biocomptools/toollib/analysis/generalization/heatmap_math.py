# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .pivot_build import remap_topo
from .views import GenViewConfig, parse_condition

DevMode = Literal["absolute", "relative"]


def build_network_pivot(
    df: pd.DataFrame,
    view: GenViewConfig,
    *,
    metric: str = "grid_nrmse",
    loss_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sub = df if loss_filter is None else df[df["loss_type"].str.startswith(loss_filter)]
    sub = remap_topo(sub, view)
    sub["view_condition"] = sub["condition"].map(
        lambda c: _remap_condition(c, view)
    )

    agg = (
        sub.groupby(["view_condition", "network_name", "view_topo", "experiment"])[metric]
        .median()
        .reset_index()
    )
    pivot = agg.pivot_table(index="network_name", columns="view_condition", values=metric)
    conds = sorted(pivot.columns, key=lambda c: (len(c), c))
    pivot = pivot.reindex(columns=conds)

    net_meta = (
        agg.groupby("network_name")
        .first()[["view_topo", "experiment"]]
        .rename(columns={"view_topo": "topo_class"})
        .reindex(pivot.index)
    )

    full_col = "".join(sorted(view.players))
    ref = pivot[full_col] if full_col in pivot.columns else pivot.median(axis=1)
    xp_avg = ref.groupby(net_meta["experiment"]).median()
    topo_rank = {t: i for i, t in enumerate(view.players)}
    sort_key = (
        net_meta["topo_class"].map(topo_rank).fillna(999) * 1e8
        + net_meta["experiment"].map(xp_avg.rank()).fillna(999) * 1e4
        + ref.rank()
    )
    order = sort_key.sort_values().index
    return pivot.loc[order], net_meta.loc[order], conds


def _remap_condition(cond: str, view: GenViewConfig) -> str:
    if not view.topo_mapping:
        return cond
    fine_keys = list(view.topo_mapping.keys())
    view_classes = {view.topo_mapping[f] for f in parse_condition(cond, fine_keys) if f in view.topo_mapping}
    return "".join(sorted(view_classes))


def build_class_pivot(
    pivot: pd.DataFrame, net_meta: pd.DataFrame, view: GenViewConfig
) -> tuple[pd.DataFrame, list[str], list[str]]:
    class_pivot = pivot.groupby(net_meta["topo_class"]).median()
    class_order = [p for p in view.players if p in class_pivot.index]

    view_keys = sorted(view.players, key=len, reverse=True)
    players_set = set(view.players)
    conds = [
        c for c in class_pivot.columns
        if set(parse_condition(c, view_keys)) <= players_set
    ]

    full_cond = "".join(sorted(view.players))
    col_means = class_pivot[conds].mean(axis=0)
    conds = sorted(conds, key=lambda c: (c == full_cond, -col_means.get(c, 0)))

    return class_pivot.loc[class_order, conds], class_order, conds


def deviation(matrix: np.ndarray, ref: np.ndarray, mode: DevMode) -> np.ndarray:
    if mode == "relative":
        safe = np.where(np.abs(ref) > 1e-10, ref, 1.0)
        return (matrix - ref) / safe * 100
    return matrix - ref


def log_fold_dev(matrix: np.ndarray, ref: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.log(np.maximum(np.abs(ref), eps) / np.maximum(np.abs(matrix), eps))


def col_stats(
    matrix: np.ndarray, net_meta: pd.DataFrame, players: list[str], weighted: bool
) -> np.ndarray:
    if not weighted:
        return np.nanmean(matrix, axis=0)
    topo = net_meta["topo_class"].values
    out = np.full(matrix.shape[1], np.nan)
    for i in range(matrix.shape[1]):
        topo_means = [
            float(np.nanmean(matrix[topo == t, i]))
            for t in players
            if np.any(np.isfinite(matrix[topo == t, i]))
        ]
        out[i] = float(np.mean(topo_means)) if topo_means else np.nan
    return out
