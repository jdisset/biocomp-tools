"""Single source of truth for routing step_history keys to DB tables.

Replaces two divergent triage implementations:
- BatchData.from_step_history() in logger_history.py
- _split_step_history() in history_db.py

Each step_history key is routed to exactly one destination:
- step.loss (scalar, extracted separately)
- step_scalar (flat numeric values)
- step_dict (JSON-serializable dicts)
- step_array (flat numpy arrays, stored as .npy bytes)
- step_blob (nested structures: pytrees, dicts-of-arrays, pickled)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Flat numpy arrays -> step_array (stored as .npy bytes)
ARRAY_KEYS: frozenset[str] = frozenset({"yhatdep", "X", "Y", "all_losses"})

# Nested structures (pytrees, dicts of arrays) -> step_blob (pickled)
BLOB_KEYS: frozenset[str] = frozenset(
    {
        "params",
        "latest_params",
        "grad",
        "apply_aux",
        "opt_state",
    }
)

# Params subset of BLOB_KEYS (for WritePolicy interval logic)
PARAMS_KEYS: frozenset[str] = frozenset({"params", "latest_params", "grad", "opt_state"})


@dataclass(frozen=True)
class TriagedStepData:
    """Step history split into DB-table-aligned categories."""

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
        if arr.size == 1:
            return float(arr.flat[0])
        return float(arr.mean())
    if hasattr(raw, "__float__"):
        return float(raw)
    return None


def triage_step_history(step_history: dict[str, Any]) -> TriagedStepData:
    """Route step_history keys to the correct DB table.

    Single implementation used by both StepWriter (DB writes)
    and BatchData.from_step_history() (in-memory construction).
    """
    loss = _extract_loss(step_history.get("loss"))
    scalars: dict[str, float] = {}
    dicts: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    blobs: dict[str, Any] = {}

    for k, v in step_history.items():
        if k == "loss":
            continue

        if k in BLOB_KEYS:
            if v is not None:
                blobs[k] = v
            continue

        if k in ARRAY_KEYS:
            if v is not None:
                arrays[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v
            continue

        # Auto-triage unknown keys by type
        if isinstance(v, dict):
            dicts[k] = v
        elif hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.size <= 1:
                scalars[k] = float(arr.flat[0]) if arr.size == 1 else float("nan")
            elif arr.size <= 100:
                # Small arrays go to dicts as lists (JSON-safe)
                dicts[k] = {"_list": arr.tolist()}
            else:
                arrays[k] = arr
        elif hasattr(v, "__float__"):
            scalars[k] = float(v)
        elif isinstance(v, (int, float)):
            scalars[k] = float(v)
        else:
            # Last resort: try to store as scalar, else skip
            try:
                scalars[k] = float(v)
            except (TypeError, ValueError):
                # Non-numeric, non-dict, non-array — store as dict with string value
                dicts[k] = {"_value": str(v)}

    return TriagedStepData(
        loss=loss,
        scalars=scalars,
        dicts=dicts,
        arrays=arrays,
        blobs=blobs,
    )
