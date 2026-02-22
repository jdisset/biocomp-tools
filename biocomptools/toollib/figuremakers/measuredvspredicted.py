"""Data holder for measured-vs-predicted scatter plots.

Extracts (measured, predicted) pairs from NetworkPrediction objects,
projecting to dependent outputs and optionally subsampling or
aggregating via lattice queries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray as NdArray
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from biocomp.plotutils import get_reordered_protein_names
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class MeasuredVsPredictedData(BaseModel):
    """Extracts (measured, predicted) 1D arrays from NetworkPrediction objects.

    Two modes:
    - Raw points (``lattice_res=None``, default): one scatter point per data point.
      If a prediction has more points than ``resample_per_experiment``, subsamples.
    - Lattice query (``lattice_res=N``): aggregates via KNN on an N-point 1D lattice
      over measured values, returning smoothed (measured_mean, predicted_mean).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predictions: list[Any]  # list of NetworkPrediction objects
    resample_per_experiment: int = 50_000
    lattice_res: int | None = None
    knn_stats_params: dict = {}
    dependent_output_only: bool = True

    _measured: NdArray = PrivateAttr()
    _predicted: NdArray = PrivateAttr()
    _rescaler: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _initialize(self):
        all_measured: list[NdArray] = []
        all_predicted: list[NdArray] = []

        for pred in self.predictions:
            if self._rescaler is None and hasattr(pred, "network_model"):
                self._rescaler = pred.network_model.model.rescaler
            if pred._yhats is None:
                pred.compute_all_network_predictions()

            networks = pred.network_model.network
            for i, network in enumerate(networks):
                gt = pred._gtruths[i]
                yhat = pred._yhats[i]
                if gt is None:
                    logger.warning(f"Skipping network {i}: no ground truth")
                    continue

                gt = np.asarray(gt, dtype=np.float32)
                yhat = np.asarray(yhat, dtype=np.float32)

                if gt.ndim == 1:
                    gt = gt[:, None]
                if yhat.ndim == 1:
                    yhat = yhat[:, None]

                # project to dependent outputs
                if self.dependent_output_only:
                    _, dep_pos, _, _ = get_reordered_protein_names(network)
                    if isinstance(dep_pos, int):
                        dep_pos = [dep_pos]
                    if dep_pos and yhat.shape[1] > max(dep_pos):
                        yhat = yhat[:, dep_pos]
                    if dep_pos and gt.shape[1] > max(dep_pos):
                        gt = gt[:, dep_pos]

                # ensure same number of output columns
                n_cols = min(gt.shape[1], yhat.shape[1])
                gt = gt[:, :n_cols]
                yhat = yhat[:, :n_cols]

                # flatten to 1D (all output columns concatenated)
                m_flat = gt.ravel()
                p_flat = yhat.ravel()

                # remove non-finite
                mask = np.isfinite(m_flat) & np.isfinite(p_flat)
                m_flat = m_flat[mask]
                p_flat = p_flat[mask]

                if self.lattice_res is not None:
                    m_flat, p_flat = self._lattice_query(m_flat, p_flat)
                elif len(m_flat) > self.resample_per_experiment:
                    rng = np.random.default_rng(42 + i)
                    idx = rng.choice(len(m_flat), self.resample_per_experiment, replace=False)
                    m_flat = m_flat[idx]
                    p_flat = p_flat[idx]

                all_measured.append(m_flat)
                all_predicted.append(p_flat)

        assert len(all_measured) > 0, "No valid (measured, predicted) pairs found"
        self._measured = np.concatenate(all_measured)
        self._predicted = np.concatenate(all_predicted)
        return self

    def _lattice_query(self, measured: NdArray, predicted: NdArray) -> tuple[NdArray, NdArray]:
        from biocomp.plotting.plotting_core import build_tree, knn_stats

        assert self.lattice_res is not None
        lattice = np.linspace(measured.min(), measured.max(), self.lattice_res)[:, None]
        measured_2d = measured[:, None]

        kw = dict(self.knn_stats_params)
        tree = build_tree(measured_2d)

        m_mean = knn_stats(lattice, y=measured_2d, tree=tree, stats="mean", **kw)
        p_mean = knn_stats(lattice, y=predicted[:, None], tree=tree, stats="mean", **kw)

        m_out = np.asarray(m_mean).ravel()
        p_out = np.asarray(p_mean).ravel()

        mask = np.isfinite(m_out) & np.isfinite(p_out)
        return m_out[mask], p_out[mask]

    @property
    def measured(self) -> NdArray:
        return self._measured

    @property
    def predicted(self) -> NdArray:
        return self._predicted

    @property
    def rescaler(self):
        return self._rescaler
