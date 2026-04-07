"""Per-run SQLite database for step history and replay artifacts.

SSOT for all optimization data. Dual storage layer:
- SQLModel (via SQLAlchemy engine) for RunInfo/RunArtifact (typed domain objects)
- Raw sqlite3 conn for step_scalar/step_dict/step_array/step_blob (bulk I/O)

Both layers share the same underlying sqlite3 connection via WAL mode.
"""

import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

import dill
import numpy as np
from sqlalchemy import Column, LargeBinary
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select

from biocomptools.logger_history import BatchData
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQLModel domain tables (typed, loaded once)
# ---------------------------------------------------------------------------


class RunInfo(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_type: str  # "training" | "design"
    start_time: float
    end_time: float | None = None
    config_json: str = "{}"
    commit_hashes_json: str = "{}"
    host: str = ""
    model_signature: str | None = None
    extra_json: str | None = None


class RunArtifact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    artifact_pickle: bytes = Field(sa_column=Column(LargeBinary))
    size_bytes: int = 0
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# Raw sqlite3 schema for time-series step data
# ---------------------------------------------------------------------------

_STEP_TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS step (
    step INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    loss REAL
);

CREATE TABLE IF NOT EXISTS step_scalar (
    step INTEGER NOT NULL REFERENCES step(step),
    key TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (step, key)
);

CREATE TABLE IF NOT EXISTS step_dict (
    step INTEGER NOT NULL REFERENCES step(step),
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (step, key)
);

CREATE TABLE IF NOT EXISTS step_array (
    step INTEGER NOT NULL REFERENCES step(step),
    key TEXT NOT NULL,
    array_blob BLOB NOT NULL,
    dtype TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    PRIMARY KEY (step, key)
);

CREATE TABLE IF NOT EXISTS step_blob (
    step INTEGER NOT NULL REFERENCES step(step),
    key TEXT NOT NULL,
    blob_pickle BLOB NOT NULL,
    size_bytes INTEGER NOT NULL,
    PRIMARY KEY (step, key)
);

CREATE TABLE IF NOT EXISTS trace_event (
    id INTEGER PRIMARY KEY,
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    component TEXT NOT NULL,
    scope TEXT,
    event_type TEXT NOT NULL,
    message TEXT,
    data_json TEXT,
    snapshot_pickle BLOB
);

CREATE INDEX IF NOT EXISTS idx_step_scalar_key ON step_scalar(key);
CREATE INDEX IF NOT EXISTS idx_step_array_key ON step_array(key);
CREATE INDEX IF NOT EXISTS idx_step_dict_key ON step_dict(key);
CREATE INDEX IF NOT EXISTS idx_step_blob_key ON step_blob(key);
CREATE INDEX IF NOT EXISTS idx_trace_component ON trace_event(component);
"""


class RunHistoryDB:
    """Per-run SQLite database — SSOT for all optimization data.

    Dual storage layer:
    - SQLModel (via SQLAlchemy engine) for RunInfo/RunArtifact
    - Raw sqlite3 conn for step_scalar/step_dict/step_array/step_blob
    """

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only

        uri = f"file:{self._path}"
        if read_only:
            uri += "?mode=ro"
        self._conn = sqlite3.connect(
            uri if read_only else str(self._path),
            uri=read_only,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        version = self._detect_schema_version()
        if version < 2 and not read_only:
            self._conn.executescript(_STEP_TABLES_SCHEMA)

        # SQLAlchemy engine sharing the same underlying sqlite3 connection
        self._engine = create_engine(
            "sqlite://",
            creator=lambda: self._conn,
            echo=False,
            poolclass=StaticPool,
        )
        tables_to_create = [RunInfo.__table__, RunArtifact.__table__]
        if not read_only:
            SQLModel.metadata.create_all(self._engine, tables=tables_to_create)

        self._schema_version = self._detect_schema_version()

    @property
    def path(self) -> Path:
        return self._path

    def schema_version(self) -> int:
        return self._schema_version

    def _detect_schema_version(self) -> int:
        tables = {
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "step_scalar" in tables or "step" in tables:
            return 2
        return 0  # empty

    # ====================================================================
    # Domain objects (SQLModel)
    # ====================================================================

    def save_run_info(
        self,
        *,
        run_type: str,
        config: dict[str, Any] | None = None,
        commit_hashes: dict[str, str | None] | None = None,
        host: str = "",
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
            extra_json=json.dumps(extra, default=_json_default) if extra else None,
        )
        with Session(self._engine) as session:
            session.add(info)
            session.commit()
            session.refresh(info)
        logger.debug(f"Saved RunInfo id={info.id} run_type={run_type}")
        return info

    def load_run_info(self) -> RunInfo | None:
        with Session(self._engine) as session:
            info = session.exec(select(RunInfo)).first()
            if info is not None:
                session.expunge(info)
            return info

    def save_artifact(self, name: str, obj: object) -> None:
        blob = dill.dumps(obj)
        with Session(self._engine) as session:
            existing = session.exec(select(RunArtifact).where(RunArtifact.name == name)).first()
            if existing:
                existing.artifact_pickle = blob
                existing.size_bytes = len(blob)
                existing.created_at = time.time()
                session.add(existing)
            else:
                session.add(
                    RunArtifact(
                        name=name,
                        artifact_pickle=blob,
                        size_bytes=len(blob),
                        created_at=time.time(),
                    )
                )
            session.commit()
        logger.debug(f"Saved artifact '{name}' ({len(blob)} bytes)")

    def load_artifact(self, name: str) -> object | None:
        with Session(self._engine) as session:
            art = session.exec(select(RunArtifact).where(RunArtifact.name == name)).first()
            if art is None:
                return None
            return dill.loads(art.artifact_pickle)

    def mark_finished(self) -> None:
        with Session(self._engine) as session:
            info = session.exec(select(RunInfo)).first()
            if info is not None:
                info.end_time = time.time()
                session.add(info)
                session.commit()

    def is_run_finished(self) -> bool:
        row = self._conn.execute("SELECT end_time FROM runinfo LIMIT 1").fetchone()
        return row is not None and row[0] is not None

    # Compat aliases
    def update_end_time(self) -> None:
        self.mark_finished()

    # ====================================================================
    # Step data — Write API (raw sqlite3)
    # ====================================================================

    def save_step(self, step: int, timestamp: float, loss: float | None) -> None:
        sql_loss = None if (loss is not None and np.isnan(loss)) else loss
        self._conn.execute(
            "INSERT OR REPLACE INTO step VALUES (?,?,?)",
            (step, timestamp, sql_loss),
        )

    def save_scalars(self, step: int, scalars: dict[str, float]) -> None:
        if not scalars:
            return
        rows = []
        for k, v in scalars.items():
            sql_v = None if (isinstance(v, float) and np.isnan(v)) else v
            rows.append((step, k, sql_v))
        self._conn.executemany("INSERT OR REPLACE INTO step_scalar VALUES (?,?,?)", rows)

    def save_dicts(self, step: int, dicts: dict[str, dict]) -> None:
        if not dicts:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO step_dict VALUES (?,?,?)",
            [(step, k, json.dumps(v, default=_json_default)) for k, v in dicts.items()],
        )

    def save_array(self, step: int, key: str, array: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, array)
        blob = buf.getvalue()
        self._conn.execute(
            "INSERT OR REPLACE INTO step_array VALUES (?,?,?,?,?,?)",
            (
                step,
                key,
                blob,
                str(array.dtype),
                json.dumps(list(array.shape)),
                len(blob),
            ),
        )

    def save_arrays(self, step: int, arrays: dict[str, np.ndarray]) -> None:
        for k, arr in arrays.items():
            self.save_array(step, k, arr)

    _SQLITE_MAX_BLOB = 1_000_000_000  # 1 GB default

    def save_blob(self, step: int, key: str, obj: object) -> None:
        if not hasattr(self, "_oversized_blob_keys"):
            self._oversized_blob_keys: dict[str, int] = {}
        if key in self._oversized_blob_keys:
            self._oversized_blob_keys[key] += 1
            if self._oversized_blob_keys[key] % 100 == 0:
                logger.warning(f"Key '{key}': skipped {self._oversized_blob_keys[key]} times (too large)")
            return
        blob = dill.dumps(obj)
        if len(blob) >= self._SQLITE_MAX_BLOB:
            logger.warning(
                f"Step {step}, key '{key}': blob too large for SQLite "
                f"({len(blob)/1e6:.0f} MB >= {self._SQLITE_MAX_BLOB/1e6:.0f} MB limit), "
                f"skipping all future saves for this key"
            )
            self._oversized_blob_keys[key] = 1
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO step_blob VALUES (?,?,?,?)",
            (step, key, blob, len(blob)),
        )

    def save_blobs(self, step: int, blobs: dict[str, Any]) -> None:
        for k, obj in blobs.items():
            self.save_blob(step, k, obj)

    def commit(self) -> None:
        self._conn.commit()

    # ====================================================================
    # Step data — Read API (raw sqlite3)
    # ====================================================================

    def get_step_range(self) -> tuple[int, int]:
        row = self._conn.execute("SELECT MIN(step), MAX(step) FROM step").fetchone()
        if row is None or row[0] is None:
            return (0, 0)
        return (row[0], row[1])

    def get_step_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM step").fetchone()
        return row[0] if row else 0

    def get_steps_since(self, after_step: int, limit: int = 1000) -> list[int]:
        rows = self._conn.execute(
            "SELECT step FROM step WHERE step > ? ORDER BY step LIMIT ?",
            (after_step, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_steps(self) -> list[int]:
        rows = self._conn.execute("SELECT step FROM step ORDER BY step").fetchall()
        return [r[0] for r in rows]

    def load_loss_series(
        self, step_range: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if step_range:
            rows = self._conn.execute(
                "SELECT step, loss FROM step WHERE step >= ? AND step <= ? ORDER BY step",
                step_range,
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT step, loss FROM step ORDER BY step").fetchall()
        if not rows:
            return np.array([]), np.array([])
        steps = np.array([r[0] for r in rows])
        losses = np.array([r[1] if r[1] is not None else float("nan") for r in rows])
        return steps, losses

    def load_scalars(self, step: int, keys: list[str] | None = None) -> dict[str, float]:
        if keys is not None:
            if not keys:
                return {}
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, value FROM step_scalar WHERE step=? AND key IN ({placeholders})",
                (step, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value FROM step_scalar WHERE step=?", (step,)
            ).fetchall()
        return {k: (v if v is not None else float("nan")) for k, v in rows}

    def load_dict(self, step: int, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT value_json FROM step_dict WHERE step=? AND key=?",
            (step, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def load_dicts(self, step: int, keys: list[str] | None = None) -> dict[str, dict]:
        if keys is not None:
            if not keys:
                return {}
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, value_json FROM step_dict WHERE step=? AND key IN ({placeholders})",
                (step, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value_json FROM step_dict WHERE step=?", (step,)
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def load_array(self, step: int, key: str) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT array_blob FROM step_array WHERE step=? AND key=?",
            (step, key),
        ).fetchone()
        if row is None:
            return None
        return np.load(io.BytesIO(row[0]))

    def load_arrays(self, step: int, keys: list[str] | None = None) -> dict[str, np.ndarray]:
        if keys is not None:
            if not keys:
                return {}
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, array_blob FROM step_array WHERE step=? AND key IN ({placeholders})",
                (step, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, array_blob FROM step_array WHERE step=?", (step,)
            ).fetchall()
        return {k: np.load(io.BytesIO(blob)) for k, blob in rows}

    def load_blob(self, step: int, key: str) -> object | None:
        row = self._conn.execute(
            "SELECT blob_pickle FROM step_blob WHERE step=? AND key=?",
            (step, key),
        ).fetchone()
        return dill.loads(row[0]) if row else None

    def load_blobs(self, step: int, keys: list[str] | None = None) -> dict[str, Any]:
        if keys is not None:
            if not keys:
                return {}
            placeholders = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, blob_pickle FROM step_blob WHERE step=? AND key IN ({placeholders})",
                (step, *keys),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, blob_pickle FROM step_blob WHERE step=?", (step,)
            ).fetchall()
        return {k: dill.loads(blob) for k, blob in rows}

    # ---- Composite loading for HistoryView ----

    def load_step_data(
        self,
        step: int,
        *,
        scalar_keys: list[str] | None = None,
        dict_keys: list[str] | None = None,
        array_keys: list[str] | None = None,
        blob_keys: list[str] | None = None,
    ) -> BatchData | None:
        row = self._conn.execute(
            "SELECT step, timestamp, loss FROM step WHERE step=?", (step,)
        ).fetchone()
        if row is None:
            return None

        _, timestamp, loss_raw = row
        loss = loss_raw if loss_raw is not None else float("nan")

        metrics: dict[str, Any] = {}
        arrays: dict[str, Any] = {}

        metrics.update(self.load_scalars(step, scalar_keys))
        metrics.update(self.load_dicts(step, dict_keys))
        arrays.update(self.load_arrays(step, array_keys))
        arrays.update(self.load_blobs(step, blob_keys))

        return BatchData(
            batch_index=step,
            step_index=step,
            timestamp=timestamp,
            loss=loss,
            metrics=metrics,
            arrays=arrays,
        )

    def load_step_range_data(
        self,
        start: int,
        end: int,
        *,
        scalar_keys: list[str] | None = None,
        dict_keys: list[str] | None = None,
        array_keys: list[str] | None = None,
        blob_keys: list[str] | None = None,
    ) -> list[BatchData]:
        rows = self._conn.execute(
            "SELECT step FROM step WHERE step >= ? AND step <= ? ORDER BY step",
            (start, end),
        ).fetchall()
        batches = []
        for (s,) in rows:
            bd = self.load_step_data(
                s,
                scalar_keys=scalar_keys,
                dict_keys=dict_keys,
                array_keys=array_keys,
                blob_keys=blob_keys,
            )
            if bd is not None:
                batches.append(bd)
        return batches

    # ---- Introspection ----

    def get_blob_steps(self, key: str) -> list[int]:
        """Return sorted list of steps that have a blob stored under *key*."""
        rows = self._conn.execute(
            "SELECT DISTINCT step FROM step_blob WHERE key=? ORDER BY step",
            (key,),
        ).fetchall()
        return [r[0] for r in rows]

    _AVAILABLE_KEY_TABLES = frozenset({"step_scalar", "step_dict", "step_array", "step_blob"})

    def available_keys(
        self,
        table: Literal["step_scalar", "step_dict", "step_array", "step_blob"] = "step_scalar",
    ) -> list[str]:
        assert table in self._AVAILABLE_KEY_TABLES, f"Unknown table: {table}"
        rows = self._conn.execute(f"SELECT DISTINCT key FROM {table}").fetchall()
        return [r[0] for r in rows]

    def load_steps(
        self,
        step_filter: Any = None,
        step_range: tuple[int, int] | None = None,
        show_progress: bool = False,
    ) -> list[BatchData]:
        query = "SELECT step FROM step"
        params: list[Any] = []
        if step_range is not None:
            query += " WHERE step >= ? AND step <= ?"
            params.extend(step_range)
        query += " ORDER BY step"
        rows = self._conn.execute(query, params).fetchall()
        steps = [r[0] for r in rows]

        if step_filter is not None:
            steps = [s for s in steps if step_filter(s)]

        if show_progress:
            try:
                from tqdm import tqdm

                steps = list(tqdm(steps, desc="Loading steps from DB"))
            except ImportError:
                pass

        batches = []
        for s in steps:
            bd = self.load_step_data(s)
            if bd is not None:
                batches.append(bd)
        return batches

    def save_step_legacy(self, step: int, timestamp: float, step_history: dict[str, Any]) -> None:
        """Write a raw step_history dict with automatic triage into granular tables."""
        from biocomptools.step_history_triage import triage_step_history

        triaged = triage_step_history(step_history)
        self.save_step(step, timestamp, triaged.loss)
        self.save_scalars(step, triaged.scalars)
        self.save_dicts(step, triaged.dicts)
        self.save_arrays(step, triaged.arrays)
        self.save_blobs(step, triaged.blobs)
        self.commit()

    # Backward compat alias
    def step_count(self) -> int:
        return self.get_step_count()

    def step_range(self) -> tuple[int, int]:
        return self.get_step_range()

    def close(self) -> None:
        self._conn.close()
        self._engine.dispose()


# ---- Internal helpers ----


def _json_default(obj: Any) -> Any:
    """Fallback serializer for JSON."""
    import enum

    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__jax_array__") or type(obj).__module__.startswith("jaxlib"):
        arr = np.asarray(obj)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return repr(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
