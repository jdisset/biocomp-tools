# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Unified target data preparation for design evaluation.

Single source of truth for extracting and preparing data from design targets.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any

from biocomp.design_targets import DataTarget
from biocomp.plotutils import PlotData


@dataclass(frozen=True)
class TargetData:
    """Prepared target data ready for prediction or plotting."""

    X: np.ndarray  # (n_samples, n_inputs) in latent space
    Y: np.ndarray | None  # (n_samples,) ground truth if available
    is_data_target: bool
    n_samples: int

    def __post_init__(self):
        assert self.X.ndim == 2, f"X must be 2D, got {self.X.ndim}D"
        assert self.X.shape[0] == self.n_samples
        if self.Y is not None:
            assert self.Y.shape[0] == self.n_samples, (
                f"Y length {self.Y.shape[0]} != n_samples {self.n_samples}"
            )

    def to_plot_data(self, model: Any = None, input_names: list[str] | None = None) -> PlotData:
        """Convert to PlotData, optionally rescaling to raw space."""
        X, Y = self.X, self.Y if self.Y is not None else np.zeros(self.n_samples)
        if model is not None:
            X = model.rescaler.inv(X)
            Y = model.rescaler.inv(Y.reshape(-1, 1)).ravel()
        names = input_names or [f'X{i + 1}' for i in range(X.shape[1])]
        return PlotData(xval=X, yval=Y, input_names=names, output_name='Y')

    def reshape_Y_for_prediction(self) -> np.ndarray | None:
        """Return Y reshaped for NetworkPrediction ground_truth parameter."""
        if self.Y is None:
            return None
        return self.Y.reshape(-1, 1) if self.Y.ndim == 1 else self.Y


def prepare_target_data(
    target: Any,
    max_samples: int = 20000,
    seed: int = 42,
    grid_resolution: tuple[int, int] = (300, 300),
) -> TargetData:
    """Extract and prepare data from a design target.

    Handles DataTarget (experimental data) and SVGTarget (synthetic targets).
    Subsamples if data exceeds max_samples.
    """
    is_data_target = isinstance(target, DataTarget)

    if is_data_target:
        X = np.asarray(target.X)
        Y = np.atleast_1d(np.asarray(target.Y).squeeze())
        assert X.ndim == 2, f"DataTarget.X must be 2D, got {X.ndim}D"

        if len(X) > max_samples:
            idx = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
            X, Y = X[idx], Y[idx]
    else:
        assert hasattr(target, 'get_lattice'), "target must have get_lattice method"
        X, Y_grid = target.get_lattice(grid_resolution, seed=seed)
        X = np.asarray(X)
        Y = np.asarray(Y_grid).ravel()

    return TargetData(X=X, Y=Y, is_data_target=is_data_target, n_samples=len(X))


def prepare_target_data_for_nre(
    target: Any,
    max_samples: int = 50000,
    seed: int = 42,
) -> TargetData:
    """Prepare target data specifically for NRE computation.

    NRE requires ground truth, so only works with DataTarget.
    Uses higher default max_samples for statistical accuracy.
    """
    if not isinstance(target, DataTarget):
        return TargetData(
            X=np.zeros((0, 2)),
            Y=None,
            is_data_target=False,
            n_samples=0,
        )
    return prepare_target_data(target, max_samples=max_samples, seed=seed)
