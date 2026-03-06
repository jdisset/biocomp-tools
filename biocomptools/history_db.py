"""Per-run SQLite database for step history and replay artifacts.

Each optimization run gets a self-contained ``run_history.db`` storing:
- Run metadata (config, commit hashes, host)
- Pickled BiocompModel, DesignManager, DesignConfig (for full-fidelity replay)
- Step records with metrics (JSON) and arrays (pickled BLOBs)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dill
import numpy as np
from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, Session, SQLModel, create_engine, select

from biocomptools.logger_history import BatchData
from biocomptools.logging_config import get_logger

if TYPE_CHECKING:
    from biocomp.design import DesignConfig, DesignManager
    from biocomptools.modelmodel import BiocompModel

logger = get_logger(__name__)

# --- Keys mirroring BatchData.from_step_history() triage logic ---
_ARRAY_KEYS = frozenset({"yhatdep", "X", "Y", "params", "latest_params", "grad", "all_losses"})
_LARGE_DICT_KEYS = frozenset({"apply_aux"})  # dicts too large for JSON; pickle instead
_PRESERVE_KEYS = frozenset({"params", "latest_params", "grad"})
_PARAMS_COLS = frozenset({"params", "latest_params"})
_OPT_STATE_KEY = "opt_state"


class RunInfo(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_type: str  # "training" | "design"
    start_time: float
    end_time: float | None = None
    config_json: str = "{}"
    commit_hashes_json: str = "{}"
    host: str = ""
    model_signature: str | None = None
    model_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    dmanager_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    dconfig_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    extra_json: str | None = None


class StepRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    step: int = Field(index=True)
    timestamp: float = 0.0
    loss: float | None = None
    metrics_json: str = "{}"
    arrays_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    params_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    opt_state_pickle: bytes | None = Field(default=None, sa_column=Column(LargeBinary))


class RunHistoryDB:
    """Per-run SQLite database for step history and replay artifacts."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self._path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(
            self._engine,
            tables=[
                RunInfo.__table__,
                StepRecord.__table__,
            ],
        )

    @property
    def path(self) -> Path:
        return self._path

    # ---- Write API ----

    def save_run_info(
        self,
        *,
        run_type: str,
        config: dict[str, Any] | None = None,
        commit_hashes: dict[str, str | None] | None = None,
        host: str = "",
        model: BiocompModel | None = None,
        dmanager: DesignManager | None = None,
        dconfig: DesignConfig | None = None,
        model_signature: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RunInfo:
        info = RunInfo(
            run_type=run_type,
            start_time=time.time(),
            config_json=json.dumps(config or {}, default=_json_default),
            commit_hashes_json=json.dumps({k: v for k, v in (commit_hashes or {}).items()}),
            host=host,
            model_signature=model_signature,
            model_pickle=dill.dumps(model) if model is not None else None,
            dmanager_pickle=dill.dumps(dmanager) if dmanager is not None else None,
            dconfig_pickle=dill.dumps(dconfig) if dconfig is not None else None,
            extra_json=json.dumps(extra, default=_json_default) if extra else None,
        )
        with Session(self._engine) as session:
            session.add(info)
            session.commit()
            session.refresh(info)
        logger.debug(f"Saved RunInfo id={info.id} run_type={run_type}")
        return info

    def save_step(self, step: int, timestamp: float, step_history: dict[str, Any]) -> None:
        metrics, arrays, params_data, opt_state_data = _split_step_history(step_history)
        loss_val = _extract_loss(step_history)

        record = StepRecord(
            step=step,
            timestamp=timestamp,
            loss=loss_val,
            metrics_json=json.dumps(metrics, default=_json_default),
            arrays_pickle=dill.dumps(arrays) if arrays else None,
            params_pickle=dill.dumps(params_data) if params_data else None,
            opt_state_pickle=dill.dumps(opt_state_data) if opt_state_data is not None else None,
        )
        with Session(self._engine) as session:
            session.add(record)
            session.commit()

    def update_end_time(self) -> None:
        with Session(self._engine) as session:
            info = session.exec(select(RunInfo)).first()
            if info is not None:
                info.end_time = time.time()
                session.add(info)
                session.commit()

    # ---- Read API ----

    def load_run_info(self) -> RunInfo | None:
        with Session(self._engine) as session:
            info = session.exec(select(RunInfo)).first()
            if info is not None:
                session.expunge(info)
            return info

    def load_model(self) -> BiocompModel | None:
        info = self.load_run_info()
        if info is None or info.model_pickle is None:
            return None
        return dill.loads(info.model_pickle)

    def load_dmanager(self) -> DesignManager | None:
        info = self.load_run_info()
        if info is None or info.dmanager_pickle is None:
            return None
        return dill.loads(info.dmanager_pickle)

    def load_dconfig(self) -> DesignConfig | None:
        info = self.load_run_info()
        if info is None or info.dconfig_pickle is None:
            return None
        return dill.loads(info.dconfig_pickle)

    def load_steps(
        self,
        step_filter: callable | None = None,
        step_range: tuple[int, int] | None = None,
        show_progress: bool = False,
    ) -> list[BatchData]:
        query = select(StepRecord).order_by(StepRecord.step)
        if step_range is not None:
            query = query.where(StepRecord.step >= step_range[0], StepRecord.step <= step_range[1])
        with Session(self._engine) as session:
            records: list[StepRecord] = list(session.exec(query).all())

        if step_filter is not None:
            records = [r for r in records if step_filter(r.step)]

        if show_progress:
            try:
                from tqdm import tqdm

                records = list(tqdm(records, desc="Loading steps from DB"))
            except ImportError:
                pass

        batches: list[BatchData] = []
        for rec in records:
            metrics = json.loads(rec.metrics_json) if rec.metrics_json else {}
            arrays: dict[str, Any] = {}
            if rec.arrays_pickle:
                arrays.update(dill.loads(rec.arrays_pickle))
            if rec.params_pickle:
                arrays.update(dill.loads(rec.params_pickle))

            batches.append(
                BatchData(
                    batch_index=rec.step,
                    step_index=rec.step,
                    batch_in_step=0,
                    timestamp=rec.timestamp,
                    loss=rec.loss if rec.loss is not None else float("nan"),
                    metrics=metrics,
                    arrays=arrays,
                )
            )

        return batches

    def step_count(self) -> int:
        with Session(self._engine) as session:
            from sqlalchemy import func

            result = session.exec(select(func.count()).select_from(StepRecord)).one()
            return result

    def step_range(self) -> tuple[int, int]:
        with Session(self._engine) as session:
            from sqlalchemy import func

            result = session.exec(
                select(func.min(StepRecord.step), func.max(StepRecord.step))
            ).one()
            return (result[0] or 0, result[1] or 0)


# ---- Internal helpers ----


def _extract_loss(sh: dict[str, Any]) -> float | None:
    raw = sh.get("loss")
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


def _split_step_history(
    sh: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """Split step_history into (metrics, arrays, params, opt_state).

    Reuses the same triage logic as BatchData.from_step_history().
    """
    metrics: dict[str, Any] = {}
    arrays: dict[str, Any] = {}
    params_data: dict[str, Any] = {}
    opt_state_data: Any = None

    for k, v in sh.items():
        if k == "loss":
            continue

        if k == _OPT_STATE_KEY:
            opt_state_data = v
            continue

        if k in _PARAMS_COLS:
            params_data[k] = v
            continue

        if k in _ARRAY_KEYS:
            if k in _PRESERVE_KEYS:
                arrays[k] = v
            else:
                arrays[k] = np.asarray(v) if hasattr(v, "__array__") else v
            continue

        # Triage remaining keys same as BatchData.from_step_history()
        if isinstance(v, dict):
            if k in _LARGE_DICT_KEYS:
                arrays[k] = v
            else:
                metrics[k] = v
        elif hasattr(v, "shape"):
            arr = np.asarray(v)
            if arr.size <= 1:
                metrics[k] = float(arr.flat[0]) if arr.size == 1 else None
            elif arr.size <= 100:
                metrics[k] = arr.tolist()
            else:
                arrays[k] = arr
        elif hasattr(v, "__float__"):
            metrics[k] = float(v)
        else:
            metrics[k] = v

    return metrics, arrays, params_data, opt_state_data


def _json_default(obj: Any) -> Any:
    """Fallback serializer for JSON — handles numpy scalars, enums, JAX arrays, etc."""
    import enum

    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__jax_array__") or type(obj).__module__.startswith("jaxlib"):
        arr = np.asarray(obj)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
