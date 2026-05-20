# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import pandas as pd

from .views import ViewConfig

_NUMERIC_METRICS = ("grid_nrmse", "grid_rmse", "grid_mse", "grid_r_squared", "kratio")


def remap_topo(df: pd.DataFrame, view: ViewConfig) -> pd.DataFrame:
    df = df.copy()
    if view.topo_mapping:
        df["view_topo"] = df["fine_topo_class"].map(view.topo_mapping)
        df = df[df["view_topo"].notna()]
    else:
        df["view_topo"] = df["fine_topo_class"]
    return df


def build_pivot(
    df: pd.DataFrame,
    view: ViewConfig,
    *,
    metric: str = "grid_nrmse",
    loss_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sub = df if loss_filter is None else df[df["loss_type"].str.startswith(loss_filter)]
    sub = remap_topo(sub, view)

    agg = (
        sub.groupby(["condition", "network_name", "view_topo", "experiment"])[metric]
        .median()
        .reset_index()
    )
    pivot = agg.pivot_table(index="network_name", columns="condition", values=metric)
    row_order = sorted(pivot.columns, key=lambda c: (len(c), c))
    pivot = pivot.reindex(columns=row_order)
    net_meta = (
        agg.groupby("network_name")
        .first()[["view_topo", "experiment"]]
        .rename(columns={"view_topo": "topo_class"})
        .reindex(pivot.index)
    )
    return pivot, net_meta, row_order


def load_metrics_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in _NUMERIC_METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
