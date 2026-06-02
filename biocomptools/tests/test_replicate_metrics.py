# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""bioRMSE / per-pair eRMSE in replicate_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from biocomp.metric_utils import ermse, DEFAULT_GRIDSTATS_PARAMS
from biocomptools.toollib.replicate_metrics import (
    GroupMetrics,
    KernelParams,
    PairMetrics,
    Run,
    adaptive_knn_floor,
    compute_group,
    lattice_floor,
)


def _pair(cross: float, floor: float, kind: str = "biological") -> PairMetrics:
    return PairMetrics(
        a="A",
        b="B",
        kind=kind,
        sigma_rmse=float(np.sqrt(cross)),
        sigma_rre=float(np.sqrt(cross / floor)),
        mse_cross=cross,
        mse_self=floor,
        ermse=ermse(cross, floor),
        n_a=10,
        n_b=10,
    )


def _group(pairs: list[PairMetrics]) -> GroupMetrics:
    return GroupMetrics(
        group_id="g",
        short_name="g",
        fingerprint="f",
        pairs=pairs,
    )


def test_ermse_pooled_averages_in_mse_space():
    crosses, floors = [0.05, 0.08, 0.02], [0.02, 0.03, 0.02]
    agg = _group([_pair(c, f) for c, f in zip(crosses, floors, strict=True)]).aggregate(
        "biological"
    )
    assert agg["n_pairs"] == 3
    assert agg["ermse_pooled"] == pytest.approx(ermse(np.mean(crosses), np.mean(floors)))


def test_ermse_pooled_clamped_when_cross_below_floor():
    agg = _group([_pair(0.01, 0.02), _pair(0.015, 0.02)]).aggregate("biological")
    assert agg["ermse_pooled"] == 0.0


def test_aggregate_exposes_cell_noise_as_pooled_floor():
    floors = [0.02, 0.03, 0.02]
    agg = _group([_pair(0.05, f) for f in floors]).aggregate("biological")
    assert agg["cell_noise"] == pytest.approx(np.sqrt(np.mean(floors)))


def test_compute_group_emits_finite_per_pair_ermse():
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 100.0, (400, 2))
    y_a = x[:, 0:1] + 2.0 * x[:, 1:2]
    runs = [
        Run(label="xp1/r", xp="xp1", basename="r", x=x.copy(), y=y_a.copy()),
        Run(label="xp2/r", xp="xp2", basename="r", x=x.copy(), y=(y_a * 1.5).copy()),
    ]
    gm = compute_group("g", "g", "f", runs, adaptive_knn_floor(KernelParams(k=10, min_points=5)))
    assert gm.pairs
    assert gm.estimator == "adaptive_knn"
    for p in gm.pairs:
        assert p.ermse >= 0.0
        assert np.isfinite(p.mse_cross) and np.isfinite(p.mse_self)
    assert gm.aggregate("biological")["ermse_pooled"] >= 0.0


def test_lattice_floor_shares_model_eRMSE_settings():
    """rep-noise on the model ruler: lattice_floor defaults to the model's
    DEFAULT_GRIDSTATS_PARAMS and stays in-range (no corruption guard trip)."""
    est = lattice_floor()
    assert est.name == "lattice"
    assert est.params["k"] == DEFAULT_GRIDSTATS_PARAMS["k"]
    assert est.params["radius"] == DEFAULT_GRIDSTATS_PARAMS["radius"]

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 0.7, (600, 2)).astype(np.float32)
    y = (0.4 + 0.3 * np.sin(5 * x[:, 0:1])).astype(np.float32)
    pred = est.predict(x, y, x)
    finite = np.isfinite(pred)
    assert finite.any()
    assert pred[finite].min() >= float(y.min()) - 1e-3
    assert pred[finite].max() <= float(y.max()) + 1e-3
