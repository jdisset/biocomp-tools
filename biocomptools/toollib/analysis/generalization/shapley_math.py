# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import itertools
import math
from math import factorial
from typing import Literal

import numpy as np
import pandas as pd

from .views import ViewConfig, parse_condition

MarginalMode = Literal["absolute", "relative", "fold", "percent"]
Aggregation = Literal["mean", "median"]

FOLD_APPROX_THRESH = 0.05


def log_fold(v_before: float, v_after: float, eps: float = 1e-10) -> float:
    return math.log(max(abs(v_before), eps) / max(abs(v_after), eps))


def log_to_fold(x: float) -> float:
    return math.exp(x) if x >= 0 else -math.exp(-x)


def log_to_percent(x: float) -> float:
    return (math.exp(x) - 1) * 100 if x >= 0 else -(math.exp(-x) - 1) * 100


def fmt_value(x: float, mode: MarginalMode, vfmt: str) -> str:
    if mode == "fold":
        return "~" if abs(x) < FOLD_APPROX_THRESH else f"{log_to_fold(x):{vfmt}}×"
    if mode == "percent":
        return f"{log_to_percent(x):{vfmt}}%"
    return f"{x:{vfmt}}{'%' if mode == 'relative' else ''}"


def compute_shapley(
    pivot: pd.DataFrame,
    net_meta: pd.DataFrame,
    row_order: list[str],
    view: ViewConfig,
    *,
    exclude_target: bool = False,
    weighted: bool = True,
    aggregation: Aggregation = "median",
    marginal_mode: MarginalMode = "percent",
) -> tuple[np.ndarray, list[str], list[list[list[tuple[str, float, float]]]]]:
    players = list(view.players)
    n = len(players)
    topo_arr = net_meta["topo_class"].values
    xp_arr = net_meta["experiment"].values
    matrix = pivot.values

    use_median = aggregation == "median"
    fine_classes = view.fine_keys()
    cond_lookup: dict[frozenset[str], str] = {}
    for c in row_order:
        cond_lookup[frozenset(parse_condition(c, fine_classes))] = c

    def _agg(arr: np.ndarray) -> float:
        return float(np.median(arr)) if use_median else float(np.mean(arr))

    def v(coalition: frozenset[str], test_cls: str) -> float | None:
        fine_coalition = view.expand_coalition(coalition)
        cond = cond_lookup.get(fine_coalition)
        if cond is None:
            return None
        col_idx = row_order.index(cond)
        mask = topo_arr == test_cls
        if not weighted:
            vals = matrix[mask, col_idx]
            finite = vals[np.isfinite(vals)]
            return _agg(finite) if len(finite) > 0 else None
        xp_means: list[float] = []
        for xp in np.unique(xp_arr[mask]):
            vals = matrix[mask & (xp_arr == xp), col_idx]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                xp_means.append(_agg(finite))
        return _agg(np.array(xp_means)) if xp_means else None

    shapley_mat = np.zeros((n, n))
    detailed: list[list[list[tuple[str, float, float]]]] = [
        [[] for _ in range(n)] for _ in range(n)
    ]

    for i_idx, i in enumerate(players):
        others = [p for p in players if p != i]
        for j_idx, j in enumerate(players):
            total = 0.0
            w_total = 0.0
            for s_size in range(1, n):  # skip empty coalition
                w = factorial(s_size) * factorial(n - 1 - s_size) / factorial(n)
                for combo in itertools.combinations(others, s_size):
                    S = frozenset(combo)
                    if exclude_target and j in S:
                        continue
                    v_S = v(S, j)
                    v_Si = v(S | {i}, j)
                    if v_S is None or v_Si is None:
                        continue
                    if marginal_mode in ("fold", "percent"):
                        marginal = log_fold(v_S, v_Si)
                    elif marginal_mode == "relative" and abs(v_S) > 1e-10:
                        marginal = (v_S - v_Si) / v_S * 100
                    else:
                        marginal = v_S - v_Si
                    total += w * marginal
                    w_total += w
                    detailed[i_idx][j_idx].append(("".join(sorted(S)), marginal, w))
            shapley_mat[i_idx, j_idx] = total / w_total if w_total > 0 else 0.0

    return shapley_mat, players, detailed
