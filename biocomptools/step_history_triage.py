# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Routes step_history keys to DB tables."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ARRAY_KEYS: frozenset[str] = frozenset({"yhatdep", "X", "Y", "all_losses"})
BLOB_KEYS: frozenset[str] = frozenset({"params", "latest_params", "grad", "apply_aux", "opt_state"})
PARAMS_KEYS: frozenset[str] = frozenset({"params", "latest_params", "grad", "opt_state"})


@dataclass(frozen=True)
class TriagedStepData:
    loss: float | None = None
    scalars: dict[str, float] = field(default_factory=dict)
    dicts: dict[str, dict] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    blobs: dict[str, Any] = field(default_factory=dict)


def _extract_loss(raw: Any) -> float | None:
    if raw is None:
        return None
    if hasattr(raw, "shape"):
        arr = np.asarray(raw)
        return float(arr.flat[0]) if arr.size == 1 else float(arr.mean())
    if hasattr(raw, "__float__"):
        return float(raw)
    return None


def partition_required_keys(required: list[str]) -> tuple[list[str], list[str]]:
    req = set(required)
    return sorted(req & ARRAY_KEYS), sorted(req & BLOB_KEYS)


def triage_step_history(step_history: dict[str, Any]) -> TriagedStepData:
    loss = _extract_loss(step_history.get("loss"))
    scalars: dict[str, float] = {}
    dicts: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    blobs: dict[str, Any] = {}

    for k, v in step_history.items():
        if k == "loss" or v is None:
            continue
        if k in BLOB_KEYS:
            blobs[k] = v
        elif k in ARRAY_KEYS:
            arrays[k] = v if isinstance(v, np.ndarray) else np.asarray(v)
        elif isinstance(v, dict):
            dicts[k] = v
        elif hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.size <= 1:
                scalars[k] = float(arr.flat[0]) if arr.size == 1 else float("nan")
            elif arr.size <= 100:
                dicts[k] = {"_list": arr.tolist()}
            else:
                arrays[k] = arr
        elif hasattr(v, "__float__") or isinstance(v, (int, float)):
            scalars[k] = float(v)
        else:
            try:
                scalars[k] = float(v)
            except (TypeError, ValueError):
                dicts[k] = {"_value": str(v)}

    return TriagedStepData(loss=loss, scalars=scalars, dicts=dicts, arrays=arrays, blobs=blobs)
