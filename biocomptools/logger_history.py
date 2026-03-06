"""Centralized history management for loggers.

Provides batch-level data storage and windowed views for logger callbacks.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import dill

# numpy compatibility shim for older pickle files created with np.bool, np.int, etc.
# In NumPy 2.0, np.bool_, np.int_, np.float_ were removed - use Python/NumPy equivalents
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = np.intp  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]


@dataclass
class BatchData:
    """Data from a single batch (forward/backward pass)."""

    batch_index: int
    step_index: int
    timestamp: float = 0.0
    loss: float = float("nan")
    metrics: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _safe_array(v):
            if hasattr(v, "data") and hasattr(v.data, "iter_leaves"):
                return v
            return np.asarray(v)

        return {
            "batch_index": self.batch_index,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "loss": self.loss,
            "metrics": self.metrics,
            "arrays": {k: _safe_array(v) for k, v in self.arrays.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchData":
        return cls(
            batch_index=d.get("batch_index", d.get("step_index", 0)),
            step_index=d["step_index"],
            timestamp=d.get("timestamp", 0.0),
            loss=d.get("loss", float("nan")),
            metrics=d.get("metrics", {}),
            arrays=d.get("arrays", {}),
        )

    @classmethod
    def from_step_history(
        cls,
        step: int,
        step_history: dict,
        timestamp: float = 0.0,
    ) -> "BatchData":
        """Convert legacy step_history dict to BatchData.

        Delegates to triage_step_history() for consistent routing.
        ParameterTree objects in blob keys are preserved as-is for in-memory use.
        """
        from biocomptools.step_history_triage import triage_step_history

        triaged = triage_step_history(step_history)

        loss = triaged.loss if triaged.loss is not None else float("nan")

        metrics: dict[str, Any] = {}
        metrics.update(triaged.scalars)
        # Unwrap {"_list": [...]} wrappers — those are a DB serialization
        # detail, in-memory callers should see original numpy arrays.
        for k, v in triaged.dicts.items():
            if isinstance(v, dict) and "_list" in v and len(v) == 1:
                metrics[k] = np.asarray(v["_list"])
            else:
                metrics[k] = v

        arrays: dict[str, Any] = {}
        arrays.update(triaged.arrays)

        for k, v in triaged.blobs.items():
            arrays[k] = v

        return cls(
            batch_index=step,
            step_index=step,
            timestamp=timestamp,
            loss=loss,
            metrics=metrics,
            arrays=arrays,
        )


@dataclass
class LoggerContext:
    """Context passed to logger callbacks."""

    training_config: Any = None
    stack: Any = None
    output_dir: Path | None = None
    current_step: int = 0
    is_replay: bool = False
    is_final: bool = False
    dmanager: Any = None
    model: Any = None
    training_program: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        step: int,
        training_program: object | None = None,
        output_dir: Path | None = None,
        stack: object | None = None,
        model: object | None = None,
        dmanager: object | None = None,
        training_config: object | None = None,
        is_replay: bool = False,
        is_final: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> "LoggerContext":
        """Factory that extracts fields from training_program with consistent defaults."""
        tp = training_program
        return cls(
            training_config=training_config or (getattr(tp, "_last_config", None) if tp else None),
            stack=stack or (getattr(tp, "_stack", None) if tp else None),
            output_dir=output_dir or (getattr(tp, "_save_dir", None) if tp else None),
            current_step=step,
            is_replay=is_replay,
            is_final=is_final,
            dmanager=dmanager or (getattr(tp, "_dmanager", None) if tp else None),
            model=model or (getattr(tp, "_model", None) if tp else None),
            training_program=tp,
            extra=extra or {},
        )


class HistoryView:
    """Read-only windowed view of batch history."""

    def __init__(
        self,
        batches: list[BatchData],
        requested_metrics: list[str] | None = None,
        requested_arrays: list[str] | None = None,
    ):
        self._batches = batches
        self._requested_metrics = requested_metrics
        self._requested_arrays = requested_arrays
        self._cached_arrays: dict[str, np.ndarray] = {}

    @property
    def n_batches(self) -> int:
        return len(self._batches)

    @property
    def n_steps(self) -> int:
        if not self._batches:
            return 0
        return len(set(b.step_index for b in self._batches))

    @property
    def batch_indices(self) -> np.ndarray:
        return np.array([b.batch_index for b in self._batches])

    @property
    def step_indices(self) -> np.ndarray:
        return np.array([b.step_index for b in self._batches])

    @property
    def losses(self) -> np.ndarray:
        return np.array([b.loss for b in self._batches])

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([b.timestamp for b in self._batches])

    def latest(self) -> BatchData | None:
        return self._batches[-1] if self._batches else None

    def get_metric(self, key: str) -> np.ndarray:
        """Get metric values across all batches (returns array of values or dicts)."""
        return np.array([b.metrics.get(key) for b in self._batches], dtype=object)

    def get_array(self, key: str) -> np.ndarray | None:
        """Get array data from most recent batch (arrays are too large to stack)."""
        if not self._batches:
            return None
        return self._batches[-1].arrays.get(key)

    def get_array_history(self, key: str, max_batches: int = 10) -> list[np.ndarray]:
        """Get list of arrays from last N batches."""
        result = []
        for b in self._batches[-max_batches:]:
            arr = b.arrays.get(key)
            if arr is not None:
                result.append(arr)
        return result

    def iter_batches(self):
        """Iterate over all batches in order."""
        yield from self._batches

    def to_step_history(self) -> dict[str, Any]:
        """Convert latest batch to flat step_history dict for backward compat.

        Unwraps internal serialization wrappers ({"_list": [...]}) back to
        numpy arrays so callers see the original types.
        """
        if not self._batches:
            return {}
        b = self._batches[-1]
        result: dict[str, Any] = {"loss": b.loss}
        for k, v in b.metrics.items():
            if isinstance(v, dict) and "_list" in v and len(v) == 1:
                result[k] = np.asarray(v["_list"])
            else:
                result[k] = v
        result.update(b.arrays)
        return result


class HistoryManager:
    """Centralized history storage with windowing."""

    def __init__(self, max_batches: int = 10000):
        self._max_batches = max_batches
        self._batches: deque[BatchData] = deque(maxlen=max_batches)
        self._batch_index = 0

    def append_batch(self, batch: BatchData):
        self._batches.append(batch)
        self._batch_index = max(self._batch_index, batch.batch_index + 1)

    def append_from_step(
        self,
        step: int,
        step_history: dict,
        timestamp: float = 0.0,
    ):
        batch = BatchData.from_step_history(
            step=step,
            step_history=step_history,
            timestamp=timestamp,
        )
        batch.batch_index = self._batch_index
        self._batches.append(batch)
        self._batch_index += 1

    def append_loss_only(
        self,
        step: int,
        loss: np.ndarray | float,
        timestamp: float = 0.0,
        all_losses: np.ndarray | None = None,
    ):
        """Lightweight append capturing only loss values.

        Used for every-step accumulation without full step_history serialization overhead.
        """
        if hasattr(loss, "shape"):
            loss_arr = np.asarray(loss)
            loss_scalar = float(np.nanmean(loss_arr))
        else:
            loss_scalar = float(loss)
            loss_arr = np.array(loss)

        arrays: dict[str, np.ndarray] = {"loss": loss_arr}
        if all_losses is not None:
            arrays["all_losses"] = np.asarray(all_losses)

        batch = BatchData(
            batch_index=self._batch_index,
            step_index=step,
            timestamp=timestamp,
            loss=loss_scalar,
            metrics={},
            arrays=arrays,
        )
        self._batches.append(batch)
        self._batch_index += 1

    def get_view(
        self,
        window: int | None = None,
        since_batch: int | None = None,
        metrics: list[str] | None = None,
        arrays: list[str] | None = None,
    ) -> HistoryView:
        """Get a windowed view of history.

        Args:
            window: Number of most recent batches to include
            since_batch: Only include batches with index >= since_batch
            metrics: Filter to these metric keys (None = all)
            arrays: Filter to these array keys (None = all)
        """
        batches = list(self._batches)

        if since_batch is not None:
            batches = [b for b in batches if b.batch_index >= since_batch]

        if window is not None and len(batches) > window:
            batches = batches[-window:]

        return HistoryView(batches, metrics, arrays)

    def clear(self):
        self._batches.clear()

    @classmethod
    def load_from_step_files(
        cls,
        history_dir: Path,
        step_filter: "Callable[[int], bool] | None" = None,
        show_progress: bool = True,
    ) -> list[BatchData]:
        """Load batches from legacy step_*.pkl files.

        Args:
            history_dir: Directory containing step_*.pkl files
            step_filter: Optional filter to only load matching steps (avoids loading all files)
            show_progress: Show tqdm progress bar
        """
        from tqdm import tqdm

        # First pass: get all step files and their step numbers (fast - no loading)
        step_files: list[tuple[int, Path]] = []
        for path in history_dir.glob("step_*.pkl"):
            parts = path.stem.split("_")
            if len(parts) < 2:
                continue
            if len(parts) > 2 and parts[2] in ("start", "end"):
                continue
            try:
                step = int(parts[1])
                if step_filter is None or step_filter(step):
                    step_files.append((step, path))
            except ValueError:
                continue

        step_files.sort(key=lambda x: x[0])

        if not step_files:
            return []

        # Second pass: load only the files we need
        batches = []
        iterator = tqdm(step_files, desc="Loading steps", disable=not show_progress)
        for step, path in iterator:
            try:
                with open(path, "rb") as f:
                    data = dill.load(f)
                step_history = data.get("step_history", {})
                timestamp = data.get("timestamp", 0.0)
                batch = BatchData.from_step_history(
                    step=step,
                    step_history=step_history,
                    timestamp=timestamp,
                )
                batch.batch_index = step
                batches.append(batch)
            except Exception:
                continue
        return sorted(batches, key=lambda b: b.batch_index)


# history mode type
HistoryMode = Literal["window", "since_last", "all"]
