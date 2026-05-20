# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def normalize_losses_for_ranking(all_losses: Any) -> np.ndarray | None:
    """Normalize step-history all_losses to (n_replicates, n_targets, n_networks)."""
    if all_losses is None:
        return None
    arr = np.asarray(all_losses)
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, 1, arr.shape[0])
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # (n_replicates, n_batches, n_targets, n_networks) -> last batch
        return arr[:, -1, :, :]
    return None


def get_selection_losses_from_step_history(
    *,
    step_history: Mapping[str, Any] | None,
    n_replicates: int,
    n_targets: int,
    n_networks: int,
) -> np.ndarray:
    """Get SSOT loss tensor used for top-k design selection.

    Source of truth:
    - Final-step `step_history['all_losses']` (same source as heatmap ranking).

    Raises:
    - ValueError when step_history/all_losses is missing or shape is incompatible.
    """
    if step_history is None:
        raise ValueError(
            "Selection SSOT missing: step_history is None; expected final-step all_losses."
        )

    expected_shape = (n_replicates, n_targets, n_networks)
    step_losses = normalize_losses_for_ranking(step_history.get("all_losses"))
    if step_losses is None:
        raise ValueError(
            "Selection SSOT missing: step_history['all_losses'] is absent or unsupported shape."
        )
    if tuple(step_losses.shape) != expected_shape:
        raise ValueError(
            "Selection SSOT shape mismatch: "
            f"expected {expected_shape}, got {tuple(step_losses.shape)}."
        )

    return np.asarray(step_losses, dtype=np.float32)
