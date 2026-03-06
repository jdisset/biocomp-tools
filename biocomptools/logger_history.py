"""Centralized history management for loggers.

Provides batch-level data storage and windowed views for logger callbacks.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import dill

# numpy compatibility shim for older pickle files created with np.bool, np.int, etc.
# In NumPy 2.0, np.bool_, np.int_, np.float_ were removed - use Python/NumPy equivalents
if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, 'int'):
    np.int = np.intp  # type: ignore[attr-defined]
if not hasattr(np, 'float'):
    np.float = np.float64  # type: ignore[attr-defined]


@dataclass
class BatchData:
    """Data from a single batch (forward/backward pass)."""

    batch_index: int
    step_index: int
    batch_in_step: int  # index within step (0 to batches_per_step-1)
    timestamp: float = 0.0
    loss: float = float('nan')
    metrics: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _safe_array(v):
            # Preserve ParameterTree objects
            if hasattr(v, 'data') and hasattr(v.data, 'iter_leaves'):
                return v
            return np.asarray(v)

        return {
            'batch_index': self.batch_index,
            'step_index': self.step_index,
            'batch_in_step': self.batch_in_step,
            'timestamp': self.timestamp,
            'loss': self.loss,
            'metrics': self.metrics,
            'arrays': {k: _safe_array(v) for k, v in self.arrays.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> BatchData:
        return cls(
            batch_index=d['batch_index'],
            step_index=d['step_index'],
            batch_in_step=d['batch_in_step'],
            timestamp=d.get('timestamp', 0.0),
            loss=d.get('loss', float('nan')),
            metrics=d.get('metrics', {}),
            arrays=d.get('arrays', {}),
        )

    @classmethod
    def from_step_history(
        cls,
        step: int,
        step_history: dict,
        batch_in_step: int = 0,
        batches_per_step: int = 1,
        timestamp: float = 0.0,
    ) -> BatchData:
        """Convert legacy step_history dict to BatchData."""
        batch_index = step * batches_per_step + batch_in_step
        raw_loss = step_history.get('loss')
        if raw_loss is None:
            loss = float('nan')
        elif hasattr(raw_loss, 'shape'):
            # Handle arrays (including JAX arrays) - check shape before __float__
            arr = np.asarray(raw_loss)
            loss = float(arr.item()) if arr.size == 1 else float(np.nanmean(arr))
        elif hasattr(raw_loss, '__float__'):
            loss = float(raw_loss)
        else:
            loss = float('nan')

        # separate metrics (scalars/small) from arrays (large)
        metrics = {}
        arrays = {}
        array_keys = {'yhatdep', 'X', 'Y', 'params', 'latest_params', 'grad', 'all_losses'}
        # Keys that should be preserved as-is (not converted to numpy)
        preserve_keys = {'params', 'latest_params', 'grad'}

        for k, v in step_history.items():
            if k in array_keys:
                if v is not None:
                    # Preserve ParameterTree and similar objects as-is
                    if k in preserve_keys and hasattr(v, 'data') and hasattr(v.data, 'iter_leaves'):
                        arrays[k] = v  # Store ParameterTree directly
                    else:
                        arrays[k] = np.asarray(v)
            elif k == 'loss':
                continue  # handled above
            elif isinstance(v, dict):
                metrics[k] = v
            elif hasattr(v, 'shape'):
                # Handle arrays - check shape before __float__
                arr = np.asarray(v)
                if arr.size == 1:
                    metrics[k] = float(arr.item())
                elif arr.size <= 100:
                    metrics[k] = arr.tolist()  # small arrays as lists
                else:
                    arrays[k] = arr  # large arrays go to arrays dict
            elif hasattr(v, '__float__'):
                metrics[k] = float(v)
            else:
                metrics[k] = v

        return cls(
            batch_index=batch_index,
            step_index=step,
            batch_in_step=batch_in_step,
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
    current_batch: int = 0
    current_step: int = 0
    total_batches: int | None = None
    total_steps: int | None = None
    batches_per_step: int = 1
    is_replay: bool = False
    is_final: bool = False
    dmanager: Any = None  # design manager if available
    model: Any = None  # BiocompModel for prediction
    training_program: Any = None  # full program reference
    extra: dict[str, Any] = field(default_factory=dict)


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
        """Convert latest batch to legacy step_history format for backward compat."""
        if not self._batches:
            return {}
        b = self._batches[-1]
        result: dict[str, Any] = {'loss': b.loss}
        result.update(b.metrics)
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
        batches_per_step: int = 1,
        timestamp: float = 0.0,
    ):
        """Append step data as one or more batches."""
        # for now, treat each step as one batch (could expand later)
        batch = BatchData.from_step_history(
            step=step,
            step_history=step_history,
            batch_in_step=0,
            batches_per_step=batches_per_step,
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
        if hasattr(loss, 'shape'):
            loss_arr = np.asarray(loss)
            loss_scalar = float(np.nanmean(loss_arr))
        else:
            loss_scalar = float(loss)
            loss_arr = np.array(loss)

        arrays: dict[str, np.ndarray] = {'loss': loss_arr}
        if all_losses is not None:
            arrays['all_losses'] = np.asarray(all_losses)

        batch = BatchData(
            batch_index=self._batch_index,
            step_index=step,
            batch_in_step=0,
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

    @property
    def current_batch_index(self) -> int:
        return self._batch_index

    def save_batch(self, batch: BatchData, output_dir: Path):
        """Save a batch to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"batch_{batch.batch_index:08d}.pkl"
        with open(path, 'wb') as f:
            dill.dump(batch.to_dict(), f)

    @classmethod
    def load_batches(cls, history_dir: Path) -> list[BatchData]:
        """Load all batches from a directory."""
        batches = []
        for path in sorted(history_dir.glob("batch_*.pkl")):
            try:
                with open(path, 'rb') as f:
                    d = dill.load(f)
                batches.append(BatchData.from_dict(d))
            except Exception:
                continue
        return batches

    @classmethod
    def load_from_step_files(
        cls,
        history_dir: Path,
        step_filter: 'Callable[[int], bool] | None' = None,
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
            parts = path.stem.split('_')
            if len(parts) < 2:
                continue
            if len(parts) > 2 and parts[2] in ('start', 'end'):
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
                with open(path, 'rb') as f:
                    data = dill.load(f)
                step_history = data.get('step_history', {})
                timestamp = data.get('timestamp', 0.0)
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
