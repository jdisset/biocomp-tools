"""WritePolicy: configurable per-data-category write intervals.

Controls what data gets persisted to the DB and at what frequency.
Loggers declare what they *need*; WritePolicy declares what gets *saved*.
"""

from pydantic import BaseModel, ConfigDict

from biocomptools.step_history_triage import BLOB_KEYS, PARAMS_KEYS


class WritePolicy(BaseModel):
    """Declares what data to persist and at what frequency."""

    model_config = ConfigDict(extra="forbid")

    always_save: frozenset[str] = frozenset(
        {
            "loss",
            "sublosses",
            "tu_stats",
            "ratio_stats",
            "pred_stats_per_network",
            "learning_rate",
            "l0_penalty",
            "entropy_penalty",
            "coupling_penalty",
            "spread_penalty",
            "commitment_penalty",
        }
    )
    never_save: frozenset[str] = frozenset()
    every_step_arrays: frozenset[str] = frozenset({"yhatdep"})
    periodic_arrays: dict[str, int] = {}
    params_interval: int = 100
    save_all: bool = False

    def get_interval(self, key: str) -> int:
        if key in self.never_save:
            return 0
        if self.save_all:
            return 1
        if key in self.always_save:
            return 1
        if key in self.every_step_arrays:
            return 1
        if key in self.periodic_arrays:
            return self.periodic_arrays[key]
        if key in BLOB_KEYS:
            return self.params_interval
        return 1

    def should_save_key(self, key: str, step: int) -> bool:
        interval = self.get_interval(key)
        if interval <= 0:
            return False
        if interval == 1:
            return True
        return step % interval == 0


DESIGN_DEFAULT = WritePolicy(
    never_save=frozenset({"z", "apply_aux"}),
    periodic_arrays={"all_losses": 10},
    params_interval=1,
)

TRAINING_DEFAULT = WritePolicy(
    always_save=frozenset({"loss", "sublosses", "learning_rate"}),
    every_step_arrays=frozenset(),
    periodic_arrays={"yhatdep": 10, "all_losses": 10},
    params_interval=500,
)

DEBUG_POLICY = WritePolicy(save_all=True)
