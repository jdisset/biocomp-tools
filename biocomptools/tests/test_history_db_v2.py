# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for RunHistoryDB v2 granular schema."""

import numpy as np
import pytest

from biocomptools.history_db import RunHistoryDB


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_run_history.db"


@pytest.fixture
def db(db_path):
    return RunHistoryDB(db_path)


def test_schema_version_is_2(db):
    assert db.schema_version() >= 2


def test_save_load_step_scalars(db):
    db.save_step(1, 100.0, 0.5)
    db.save_scalars(1, {"lr": 0.001, "penalty": 0.2})
    db.commit()

    scalars = db.load_scalars(1)
    assert scalars["lr"] == pytest.approx(0.001)
    assert scalars["penalty"] == pytest.approx(0.2)


def test_save_load_step_scalars_selective(db):
    db.save_step(1, 100.0, 0.5)
    db.save_scalars(1, {"lr": 0.001, "penalty": 0.2, "extra": 99.0})
    db.commit()

    scalars = db.load_scalars(1, keys=["lr", "penalty"])
    assert "lr" in scalars
    assert "penalty" in scalars
    assert "extra" not in scalars


def test_save_load_dicts(db):
    db.save_step(1, 100.0, 0.5)
    db.save_dicts(1, {"sublosses": {"mse": 0.1, "l0": 0.2}})
    db.commit()

    d = db.load_dict(1, "sublosses")
    assert d is not None
    assert d["mse"] == pytest.approx(0.1)


def test_save_load_array(db):
    arr = np.random.randn(10, 3).astype(np.float32)
    db.save_step(1, 100.0, 0.5)
    db.save_array(1, "yhatdep", arr)
    db.commit()

    loaded = db.load_array(1, "yhatdep")
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, arr)


def test_save_load_blob(db):
    obj = {"nested": {"tree": [1, 2, 3], "arr": np.ones(5)}}
    db.save_step(1, 100.0, 0.5)
    db.save_blob(1, "apply_aux", obj)
    db.commit()

    loaded = db.load_blob(1, "apply_aux")
    assert loaded is not None
    assert loaded["nested"]["tree"] == [1, 2, 3]
    np.testing.assert_array_equal(loaded["nested"]["arr"], np.ones(5))


def test_nan_loss_roundtrip(db):
    db.save_step(1, 100.0, float("nan"))
    db.commit()

    bd = db.load_step_data(1)
    assert bd is not None
    assert np.isnan(bd.loss)


def test_nan_scalar_roundtrip(db):
    db.save_step(1, 100.0, 0.5)
    db.save_scalars(1, {"x": float("nan")})
    db.commit()

    scalars = db.load_scalars(1)
    assert np.isnan(scalars["x"])


def test_get_steps_since(db):
    for s in range(5):
        db.save_step(s, float(s), float(s) * 0.1)
    db.commit()

    steps = db.get_steps_since(2)
    assert steps == [3, 4]


def test_get_step_range(db):
    for s in [10, 20, 30]:
        db.save_step(s, float(s), 0.0)
    db.commit()

    lo, hi = db.get_step_range()
    assert lo == 10
    assert hi == 30


def test_get_step_count(db):
    for s in range(7):
        db.save_step(s, float(s), 0.0)
    db.commit()

    assert db.get_step_count() == 7


def test_save_load_artifact(db):
    obj = {"model_data": np.ones(10), "config": {"lr": 0.01}}
    db.save_artifact("model", obj)

    loaded = db.load_artifact("model")
    assert loaded is not None
    np.testing.assert_array_equal(loaded["model_data"], np.ones(10))
    assert loaded["config"]["lr"] == 0.01


def test_load_nonexistent_artifact(db):
    assert db.load_artifact("nonexistent") is None


def test_mark_finished(db):
    db.save_run_info(run_type="test")
    assert not db.is_run_finished()

    db.mark_finished()
    assert db.is_run_finished()


def test_load_step_data_composite(db):
    db.save_step(5, 100.0, 0.42)
    db.save_scalars(5, {"lr": 0.001})
    db.save_dicts(5, {"sub": {"a": 1}})
    db.save_array(5, "yhatdep", np.zeros((3, 2)))
    db.commit()

    bd = db.load_step_data(5)
    assert bd is not None
    assert bd.step_index == 5
    assert bd.loss == pytest.approx(0.42)
    assert bd.metrics["lr"] == pytest.approx(0.001)
    assert "sub" in bd.metrics
    assert "yhatdep" in bd.arrays


def test_load_step_range_data(db):
    for s in range(10):
        db.save_step(s, float(s), float(s) * 0.1)
        db.save_scalars(s, {"lr": 0.001 * s})
    db.commit()

    batches = db.load_step_range_data(3, 7)
    assert len(batches) == 5
    assert batches[0].step_index == 3
    assert batches[-1].step_index == 7


def test_available_keys(db):
    db.save_step(1, 100.0, 0.5)
    db.save_scalars(1, {"lr": 0.001, "penalty": 0.2})
    db.save_array(1, "yhatdep", np.zeros(5))
    db.commit()

    scalar_keys = db.available_keys("step_scalar")
    assert "lr" in scalar_keys
    assert "penalty" in scalar_keys

    array_keys = db.available_keys("step_array")
    assert "yhatdep" in array_keys


def test_save_step_history(db):
    sh = {
        "loss": 0.5,
        "learning_rate": 0.001,
        "sublosses": {"mse": 0.1},
        "yhatdep": np.ones((5, 3)),
        "apply_aux": {"layer": {"w": [1, 2]}},
    }
    db.save_step_history(1, 100.0, sh)

    bd = db.load_step_data(1)
    assert bd is not None
    assert bd.loss == pytest.approx(0.5)
    assert "learning_rate" in bd.metrics
    assert "yhatdep" in bd.arrays
    assert "apply_aux" in bd.arrays  # loaded from step_blob


def test_load_steps_v2(db):
    for s in range(5):
        db.save_step(s, float(s), float(s) * 0.1)
        db.save_scalars(s, {"lr": 0.001 * s})
    db.commit()

    batches = db.load_steps()
    assert len(batches) == 5
    assert batches[0].step_index == 0
    assert batches[-1].step_index == 4


def test_read_only_mode(db_path):
    # Create DB first
    db = RunHistoryDB(db_path)
    db.save_step(1, 100.0, 0.5)
    db.save_scalars(1, {"lr": 0.001})
    db.commit()
    db.close()

    # Open read-only
    ro = RunHistoryDB(db_path, read_only=True)
    scalars = ro.load_scalars(1)
    assert scalars["lr"] == pytest.approx(0.001)
    ro.close()
