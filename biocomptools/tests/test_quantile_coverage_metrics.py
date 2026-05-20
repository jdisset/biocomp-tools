# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import numpy as np

from biocomptools.toollib.figuremakers.quantilecoverage import (
    compute_quantile_metrics_from_samples,
)


def test_quantile_metrics_returns_expected_fields():
    rng = np.random.RandomState(0)
    n_points = 32
    n_samples = 64
    targets = np.full((n_points, 1), 0.5, dtype=float)
    samples = rng.uniform(0.0, 1.0, size=(n_points, n_samples, 1))

    metrics = compute_quantile_metrics_from_samples(
        samples=samples,
        targets=targets,
        quantiles=(0.1, 0.5, 0.9),
        coverage_interval=(0.1, 0.9),
    )

    assert metrics["n_points"] == n_points
    assert metrics["n_samples_per_point"] == n_samples
    assert set(metrics["pinball"].keys()) == {"0.1", "0.5", "0.9"}
    assert np.isfinite(metrics["mean_pinball"])
    assert np.isfinite(metrics["mc_crps"])
    assert np.isfinite(metrics["ks_marginal"])


def test_quantile_metrics_coverage_tracks_interval_miss():
    n_points = 20
    n_samples = 32
    # Samples are high, targets are low -> almost no coverage in [0.1, 0.9] interval.
    samples = np.full((n_points, n_samples, 1), 0.95, dtype=float)
    targets = np.zeros((n_points, 1), dtype=float)

    metrics = compute_quantile_metrics_from_samples(
        samples=samples,
        targets=targets,
        quantiles=(0.1, 0.5, 0.9),
        coverage_interval=(0.1, 0.9),
    )

    assert metrics["coverage"] <= 0.05
    assert metrics["coverage_error"] >= 0.75
