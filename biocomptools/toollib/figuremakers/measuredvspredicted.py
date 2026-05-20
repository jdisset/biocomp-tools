# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Data holder for measured-vs-predicted scatter plots.

Extracts (measured, predicted) pairs from NetworkPrediction objects,
projecting to dependent outputs and optionally subsampling or
aggregating via lattice queries.

Two optional cube-view overlays are also exposed (when the upstream
NetworkPrediction has ``enable_gridstats=True``, which is the default):

- **Grid means** (free, always populated when stats arrays exist):
  per-grid-cell smoothed (``gt``, ``yhat``) means from the same Gaussian
  KNN kernel that drives ``grid_nrmse``. Useful as a "yellow cross"
  overlay on top of the raw-point density to show the systematic
  bias structure.

- **Noise floor** (opt-in via ``compute_noise_floor=True``): an MVP-shaped
  data-only panel built from the *same* Gaussian-KNN kernel that feeds
  the grid stats. Implementation: take the upstream ``grid_gt_mean_latent``
  (kernel-smoothed gt mean at the hypercube lattice - already computed for
  ``grid_nrmse``), wrap it in a ``RegularGridInterpolator`` (linear), and
  evaluate at every actual data-point ``x_j``. Cloud =
  ``(y_j, μ̂_kernel(x_j))`` per data point - same N as the model MVP, same
  axes, same density semantics, just predicted by the kernel smoother
  instead of the model. ``μ̂_kernel`` is the optimal nonparametric
  predictor at the cube-view bandwidth, so cloud spread around ``y = x``
  is the irreducible noise floor a smooth model with comparable
  resolution cannot cross. The matching ``mvp_panel`` cloud's spread vs
  this one's is exactly the model-vs-kernel comparison.

Output spaces follow the same convention as ``measured`` /
``predicted``: raw fluorescence (post-rescaler.inv) when the upstream
``NetworkPrediction.already_latent`` is False (default), latent
otherwise.
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

    Optional cube-view overlays (raw-points mode only):
    - Grid means: auto-populated from upstream grid stats if available.
    - Noise floor: opt-in via ``compute_noise_floor=True``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predictions: list[Any]  # list of NetworkPrediction objects
    resample_per_experiment: int = 50_000
    lattice_res: int | None = None
    knn_stats_params: dict = {}
    dependent_output_only: bool = True
    # If set, only include networks whose index (within each NetworkPrediction)
    # is in this list. Lets per-network MVP panels reuse a single shared
    # NetworkPrediction without recomputing predictions per panel.
    network_indices: list[int] | None = None

    # Cube-view overlay controls. Grid means are auto-pulled when present
    # in the upstream stats (free); the noise-floor cloud is built from
    # the same lattice means via a `RegularGridInterpolator` evaluated at
    # every data point - no extra KNN, no Gumbel sampling.
    compute_noise_floor: bool = False

    _measured: NdArray = PrivateAttr()
    _predicted: NdArray = PrivateAttr()
    _rescaler: Any = PrivateAttr(default=None)
    _grid_measured: NdArray | None = PrivateAttr(default=None)
    _grid_predicted: NdArray | None = PrivateAttr(default=None)
    _grid_weights: NdArray | None = PrivateAttr(default=None)
    _noise_floor_measured: NdArray | None = PrivateAttr(default=None)
    _noise_floor_predicted: NdArray | None = PrivateAttr(default=None)
    _noise_floor_model_predicted: NdArray | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _initialize(self):
        all_measured: list[NdArray] = []
        all_predicted: list[NdArray] = []
        all_grid_m: list[NdArray] = []
        all_grid_p: list[NdArray] = []
        all_grid_w: list[NdArray] = []
        all_floor_m: list[NdArray] = []
        all_floor_p: list[NdArray] = []
        all_floor_mp: list[NdArray] = []

        for pred in self.predictions:
            if self._rescaler is None and hasattr(pred, "network_model"):
                self._rescaler = pred.network_model.model.rescaler
            if pred._yhats is None:
                pred.compute_all_network_predictions()

            networks = pred.network_model.network
            for i, network in enumerate(networks):
                if self.network_indices is not None and i not in self.network_indices:
                    continue
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

                _, dep_pos, _, _ = get_reordered_protein_names(network)
                if isinstance(dep_pos, int):
                    dep_pos = [dep_pos]
                if self.dependent_output_only:
                    if dep_pos and yhat.shape[1] > max(dep_pos):
                        yhat = yhat[:, dep_pos]
                    if dep_pos and gt.shape[1] > max(dep_pos):
                        gt = gt[:, dep_pos]

                n_cols = min(gt.shape[1], yhat.shape[1])
                gt = gt[:, :n_cols]
                yhat = yhat[:, :n_cols]

                stats_dict = pred.get_network_stats(network_idx=i)
                sub_rows = None if stats_dict is None else stats_dict.get('subsample_indices')

                if self.lattice_res is not None:
                    m_flat = gt.ravel()
                    p_flat = yhat.ravel()
                    mask = np.isfinite(m_flat) & np.isfinite(p_flat)
                    m_flat, p_flat = self._lattice_query(m_flat[mask], p_flat[mask])
                elif sub_rows is not None and len(sub_rows) > 0:
                    gt_sub = gt[sub_rows]
                    yhat_sub = yhat[sub_rows]
                    m_flat = gt_sub.ravel()
                    p_flat = yhat_sub.ravel()
                    ok = np.isfinite(m_flat) & np.isfinite(p_flat)
                    m_flat = m_flat[ok]
                    p_flat = p_flat[ok]
                else:
                    m_flat = gt.ravel()
                    p_flat = yhat.ravel()
                    ok = np.isfinite(m_flat) & np.isfinite(p_flat)
                    m_flat, p_flat = m_flat[ok], p_flat[ok]
                    if self.resample_per_experiment and len(m_flat) > self.resample_per_experiment:
                        rng = np.random.default_rng(42 + i)
                        idx = rng.choice(len(m_flat), self.resample_per_experiment, replace=False)
                        m_flat = m_flat[idx]
                        p_flat = p_flat[idx]

                all_measured.append(m_flat)
                all_predicted.append(p_flat)

                if self.compute_noise_floor and self.lattice_res is None:
                    floor = self._compute_noise_floor(pred, i, gt, yhat, stats_dict, sub_rows)
                    if floor is not None:
                        fm, fp, fmp = floor
                        ok = np.isfinite(fm) & np.isfinite(fp) & np.isfinite(fmp)
                        all_floor_m.append(fm[ok])
                        all_floor_p.append(fp[ok])
                        all_floor_mp.append(fmp[ok])

                if stats_dict is not None:
                    grid_gt = stats_dict.get('grid_gt_mean_latent')
                    grid_yh = stats_dict.get('grid_yhat_mean_latent')
                    grid_w = stats_dict.get('grid_n_eff')
                    if grid_gt is not None and grid_yh is not None and grid_w is not None:
                        gm, gp, gw = self._project_grid_means(pred, grid_gt, grid_yh, grid_w)
                        all_grid_m.append(gm)
                        all_grid_p.append(gp)
                        all_grid_w.append(gw)

        assert len(all_measured) > 0, "No valid (measured, predicted) pairs found"
        self._measured = np.concatenate(all_measured)
        self._predicted = np.concatenate(all_predicted)
        if all_grid_m:
            self._grid_measured = np.concatenate(all_grid_m)
            self._grid_predicted = np.concatenate(all_grid_p)
            self._grid_weights = np.concatenate(all_grid_w)
        if all_floor_m:
            self._noise_floor_measured = np.concatenate(all_floor_m)
            self._noise_floor_predicted = np.concatenate(all_floor_p)
            self._noise_floor_model_predicted = np.concatenate(all_floor_mp)
        return self

    def _project_grid_means(
        self,
        pred,
        grid_gt_latent: NdArray,
        grid_yhat_latent: NdArray,
        n_eff: NdArray,
    ) -> tuple[NdArray, NdArray, NdArray]:
        """Convert per-grid-cell latent means to raw and ravel to flat scatter."""
        rescaler = pred.network_model.model.rescaler if not pred.already_latent else None
        if rescaler is not None:
            gm = np.asarray(rescaler.inv(grid_gt_latent))
            gp = np.asarray(rescaler.inv(grid_yhat_latent))
        else:
            gm, gp = np.asarray(grid_gt_latent), np.asarray(grid_yhat_latent)
        # Each row is one grid point; ravel concatenates output columns.
        # `n_eff` is one value per grid point - broadcast across columns.
        n_outs = gm.shape[1] if gm.ndim == 2 else 1
        weights = np.broadcast_to(n_eff[:, None], (n_eff.shape[0], n_outs)).ravel()
        return gm.ravel(), gp.ravel(), weights

    def _compute_noise_floor(
        self,
        pred,
        network_idx: int,
        gt_proj: NdArray,
        yhat_proj: NdArray,
        stats: dict | None,
        row_indices: NdArray | None,
    ) -> tuple[NdArray, NdArray, NdArray] | None:
        """Kernel-smoother MVP cloud, evaluated at ``row_indices`` if given.

        Returns ``(measured, kernel_pred_raw, model_pred_raw)`` index-aligned.
        Uses ``kernel_lattice_interp`` over ``grid_gt_mean_latent`` (already
        computed by ``_calculate_grid_stats``).
        """
        from biocomp.datautils import IdentityRescaler
        from biocomptools.toollib.networkprediction import kernel_lattice_interp

        if stats is None:
            return None
        grid_gt_mean_lat = stats.get('grid_gt_mean_latent')
        if grid_gt_mean_lat is None:
            return None

        rescaler = pred.network_model.model.rescaler if not pred.already_latent else IdentityRescaler()
        latent_x = np.asarray(rescaler.fwd(pred._x[network_idx]), dtype=np.float32)
        if row_indices is not None and len(row_indices) > 0:
            sel_x = latent_x[row_indices]
            gt_v = np.asarray(gt_proj[row_indices])
            yh_v = np.asarray(yhat_proj[row_indices])
        else:
            valid = np.all(np.isfinite(latent_x), axis=1)
            sel_x = latent_x[valid]
            gt_v = np.asarray(gt_proj[valid])
            yh_v = np.asarray(yhat_proj[valid])
        if sel_x.shape[0] == 0:
            return None
        if gt_v.ndim == 1:
            gt_v = gt_v[:, None]
        if yh_v.ndim == 1:
            yh_v = yh_v[:, None]

        interp = kernel_lattice_interp(grid_gt_mean_lat, pred.get_gridstats_params(), sel_x.shape[1])
        kernel_pred_raw = np.asarray(rescaler.inv(np.asarray(interp(sel_x))))
        return gt_v.ravel(), kernel_pred_raw.ravel(), yh_v.ravel()

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

    @property
    def grid_measured(self) -> NdArray | None:
        """Per-grid-cell smoothed gt-mean (raw space). None if no upstream gridstats."""
        return self._grid_measured

    @property
    def grid_predicted(self) -> NdArray | None:
        """Per-grid-cell smoothed yhat-mean (raw space). None if no upstream gridstats."""
        return self._grid_predicted

    @property
    def grid_weights(self) -> NdArray | None:
        """Per-grid-cell effective neighbour count (n_eff). None if unavailable."""
        return self._grid_weights

    @property
    def noise_floor_measured(self) -> NdArray | None:
        """Measured (y_j) axis of the data-only kernel-smoother MVP cloud.

        One entry per actual data point - the literal ground-truth
        observation. Paired index-wise with ``noise_floor_predicted``.
        Requires ``compute_noise_floor=True``.
        """
        return self._noise_floor_measured

    @property
    def noise_floor_predicted(self) -> NdArray | None:
        """Predicted axis (μ̂_kernel(x_j)) of the kernel-smoother MVP cloud.

        Bilinear interpolation of the upstream lattice ``grid_gt_mean_latent``
        evaluated at each data-point input ``x_j`` - the kernel-smoother's
        best nonparametric prediction at the cube-view bandwidth.
        """
        return self._noise_floor_predicted

    @property
    def noise_floor_model_predicted(self) -> NdArray | None:
        """Model prediction at the *same* indices as ``noise_floor_measured``.

        Aligned with ``noise_floor_predicted`` (kernel) so a delta plot
        ``model_pred − kernel_pred`` per data point can be rendered as
        a marginal strip below the model MVP. None unless
        ``compute_noise_floor=True``.
        """
        return self._noise_floor_model_predicted
