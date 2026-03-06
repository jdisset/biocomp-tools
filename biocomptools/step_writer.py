"""StepWriter: thin adapter between optimization loop and RunHistoryDB.

Owns snapshot normalization and triage. Applies WritePolicy to control
what gets persisted and when. Single atomic commit per step.
"""

import time
from typing import Any

from biocomp.step_history import StepHistoryLike, ensure_step_history_snapshot
from biocomptools.history_db import RunHistoryDB
from biocomptools.logging_config import get_logger
from biocomptools.step_history_triage import triage_step_history
from biocomptools.write_policy import WritePolicy

logger = get_logger(__name__)


class StepWriter:
    """Writes optimization step data to RunHistoryDB.

    Sits between the optimization loop and the DB.
    Owns snapshot normalization and triage — callers pass raw step_history.
    Applies WritePolicy to control what gets persisted and when.
    """

    def __init__(self, db: RunHistoryDB, policy: WritePolicy | None = None) -> None:
        self._db = db
        self._policy = policy or WritePolicy()

    @property
    def db(self) -> RunHistoryDB:
        return self._db

    @property
    def policy(self) -> WritePolicy:
        return self._policy

    def write_step(self, step: int, timestamp: float, step_history: dict[str, Any]) -> None:
        """Normalize, triage, and write to DB according to policy.

        All inserts for one step happen without an intermediate commit,
        then a single db.commit() makes them visible atomically.
        """
        triaged = triage_step_history(step_history)
        policy = self._policy

        # Step row (always written)
        self._db.save_step(step, timestamp, triaged.loss)

        # Scalars: always write
        if triaged.scalars:
            self._db.save_scalars(step, triaged.scalars)

        # Dicts: always write (small)
        if triaged.dicts:
            self._db.save_dicts(step, triaged.dicts)

        # Arrays: respect policy intervals
        for key, array in triaged.arrays.items():
            if policy.should_save_key(key, step):
                self._db.save_array(step, key, array)

        for key, obj in triaged.blobs.items():
            if policy.should_save_key(key, step):
                self._db.save_blob(step, key, obj)

        self._db.commit()

    def write_step_from_raw(
        self,
        step: int,
        step_history: StepHistoryLike,
        timestamp: float | None = None,
    ) -> None:
        """Normalize a raw StepHistoryLike and write."""
        snapshot = ensure_step_history_snapshot(step_history, context=f"StepWriter at step {step}")
        self.write_step(step, timestamp or time.time(), dict(snapshot))

    def create_callback(self):
        """Returns a lightweight callback for the optimization loop.

        Matches the signature expected by LoggerDispatch.on_step():
        callback(step, training_config, step_history=None, stack=None)
        """

        def callback(
            step: int,
            training_config: object,
            step_history: StepHistoryLike | None = None,
            stack: object = None,
        ) -> None:
            if step_history is not None:
                self.write_step_from_raw(step, step_history)

        return callback
