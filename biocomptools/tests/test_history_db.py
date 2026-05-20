# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Round-trip tests for RunHistoryDB (v2 granular schema)."""

import json
import time

import numpy as np
import pytest

from biocomptools.history_db import RunHistoryDB
from biocomptools.logger_history import BatchData


@pytest.fixture
def db(tmp_path):
    return RunHistoryDB(tmp_path / "test_run_history.db")


def test_db_creates_file(tmp_path):
    db_path = tmp_path / "sub" / "run_history.db"
    db = RunHistoryDB(db_path)
    assert db_path.exists()
    assert db.path == db_path


def test_save_and_load_run_info(db):
    info = db.save_run_info(
        run_type="design",
        config={"epochs": 100, "lr": 0.01},
        commit_hashes={"biocomp": "abc123", "biocomptools": "def456"},
        host="user@host",
        model_signature="test-sig-123",
    )
    assert info.id is not None

    loaded = db.load_run_info()
    assert loaded is not None
    assert loaded.run_type == "design"
    assert loaded.host == "user@host"
    assert loaded.model_signature == "test-sig-123"
    assert json.loads(loaded.config_json) == {"epochs": 100, "lr": 0.01}
    assert json.loads(loaded.commit_hashes_json)["biocomp"] == "abc123"


def test_save_and_load_steps(db):
    db.save_run_info(run_type="training")

    for step in range(5):
        sh = {
            "loss": 1.0 - step * 0.1,
            "yhatdep": np.random.randn(10, 3),
            "sublosses": {"mse": 0.5 - step * 0.05, "l0": 0.1},
            "lr": 0.001,
        }
        db.save_step_history(step, time.time(), sh)

    assert db.get_step_count() == 5
    assert db.get_step_range() == (0, 4)

    batches = db.load_steps()
    assert len(batches) == 5
    assert all(isinstance(b, BatchData) for b in batches)

    b0 = batches[0]
    assert b0.step_index == 0
    assert abs(b0.loss - 1.0) < 1e-6
    assert "sublosses" in b0.metrics
    assert b0.metrics["sublosses"]["mse"] == pytest.approx(0.5, abs=1e-6)
    assert "yhatdep" in b0.arrays
    assert b0.arrays["yhatdep"].shape == (10, 3)


def test_step_filter(db):
    db.save_run_info(run_type="training")

    for step in range(10):
        db.save_step_history(step, time.time(), {"loss": float(step)})

    evens = db.load_steps(step_filter=lambda s: s % 2 == 0)
    assert len(evens) == 5
    assert [b.step_index for b in evens] == [0, 2, 4, 6, 8]


def test_params_stored_separately(db):
    db.save_run_info(run_type="design")

    params = {"shared/NN/weights": np.ones((4, 4))}
    sh = {
        "loss": 0.5,
        "latest_params": params,
        "yhatdep": np.zeros((5, 2)),
    }
    db.save_step_history(0, time.time(), sh)

    batches = db.load_steps()
    assert len(batches) == 1
    b = batches[0]
    assert "latest_params" in b.arrays
    assert "yhatdep" in b.arrays


def test_mark_finished(db):
    db.save_run_info(run_type="training")
    info_before = db.load_run_info()
    assert info_before.end_time is None

    db.mark_finished()
    info_after = db.load_run_info()
    assert info_after.end_time is not None
    assert info_after.end_time > info_before.start_time


def test_model_pickle_roundtrip(db):
    """Test that arbitrary objects can be stored and loaded via artifacts."""

    class FakeModel:
        def __init__(self):
            self.signature = "fake-sig"
            self.weights = np.random.randn(3, 3)

    model = FakeModel()
    db.save_run_info(run_type="design", model_signature="fake-sig")
    db.save_artifact("model", model)

    restored = db.load_artifact("model")
    assert restored is not None
    assert restored.signature == "fake-sig"
    np.testing.assert_array_equal(restored.weights, model.weights)


def test_empty_step_history(db):
    db.save_run_info(run_type="training")
    db.save_step_history(0, time.time(), {})
    batches = db.load_steps()
    assert len(batches) == 1
    assert batches[0].metrics == {}
    assert batches[0].arrays == {}


def test_numpy_scalar_metrics(db):
    db.save_run_info(run_type="training")
    sh = {
        "loss": np.float32(0.5),
        "accuracy": np.float64(0.95),
        "epoch": np.int32(10),
    }
    db.save_step_history(0, time.time(), sh)
    batches = db.load_steps()
    assert batches[0].loss == pytest.approx(0.5, abs=1e-5)
    assert batches[0].metrics["accuracy"] == pytest.approx(0.95, abs=1e-10)
    assert batches[0].metrics["epoch"] == 10


def test_large_array_goes_to_arrays(db):
    db.save_run_info(run_type="training")
    big = np.random.randn(200, 50)
    sh = {"loss": 0.1, "big_data": big}
    db.save_step_history(0, time.time(), sh)
    batches = db.load_steps()
    assert "big_data" in batches[0].arrays
    np.testing.assert_array_almost_equal(batches[0].arrays["big_data"], big)


def test_small_array_goes_to_metrics(db):
    db.save_run_info(run_type="training")
    small = np.array([1.0, 2.0, 3.0])
    sh = {"loss": 0.1, "small_vec": small}
    db.save_step_history(0, time.time(), sh)
    batches = db.load_steps()
    assert "small_vec" in batches[0].metrics
    assert batches[0].metrics["small_vec"]["_list"] == [1.0, 2.0, 3.0]
