"""Tests for StepWriter + WritePolicy."""

import numpy as np
import pytest

from biocomptools.history_db import RunHistoryDB
from biocomptools.step_writer import StepWriter
from biocomptools.write_policy import WritePolicy


@pytest.fixture
def db(tmp_path):
    return RunHistoryDB(tmp_path / "test.db")


def test_write_step_basic(db):
    writer = StepWriter(db)
    sh = {
        "loss": 0.5,
        "learning_rate": 0.001,
        "sublosses": {"mse": 0.1},
        "yhatdep": np.ones((5, 3)),
    }
    writer.write_step(1, 100.0, sh)

    bd = db.load_step_data(1)
    assert bd is not None
    assert bd.loss == pytest.approx(0.5)
    assert "learning_rate" in bd.metrics
    assert "yhatdep" in bd.arrays


def test_write_policy_periodic_arrays(db):
    policy = WritePolicy(
        periodic_arrays={"all_losses": 5},
        every_step_arrays=frozenset(),
    )
    writer = StepWriter(db, policy)

    for step in range(1, 11):
        sh = {
            "loss": float(step),
            "all_losses": np.ones(10) * step,
        }
        writer.write_step(step, float(step), sh)

    # all_losses should only be saved at steps 5 and 10
    assert db.load_array(1, "all_losses") is None
    assert db.load_array(3, "all_losses") is None
    arr5 = db.load_array(5, "all_losses")
    assert arr5 is not None
    np.testing.assert_array_equal(arr5, np.ones(10) * 5)
    arr10 = db.load_array(10, "all_losses")
    assert arr10 is not None


def test_write_policy_params_interval(db):
    policy = WritePolicy(params_interval=3)
    writer = StepWriter(db, policy)

    for step in range(1, 7):
        sh = {
            "loss": float(step),
            "latest_params": {"w": np.ones(5) * step},
        }
        writer.write_step(step, float(step), sh)

    # Params saved at steps 3, 6 (every 3rd step)
    assert db.load_blob(1, "latest_params") is None
    assert db.load_blob(2, "latest_params") is None
    p3 = db.load_blob(3, "latest_params")
    assert p3 is not None
    assert db.load_blob(4, "latest_params") is None
    p6 = db.load_blob(6, "latest_params")
    assert p6 is not None


def test_write_policy_save_all(db):
    policy = WritePolicy(
        save_all=True,
        periodic_arrays={"all_losses": 100},  # would be periodic, but save_all overrides
        params_interval=100,
    )
    writer = StepWriter(db, policy)

    sh = {
        "loss": 0.5,
        "all_losses": np.ones(5),
        "latest_params": {"w": 1},
    }
    writer.write_step(1, 100.0, sh)

    assert db.load_array(1, "all_losses") is not None
    assert db.load_blob(1, "latest_params") is not None


def test_write_policy_scalars_always_saved(db):
    policy = WritePolicy()
    writer = StepWriter(db, policy)

    sh = {"loss": 0.5, "learning_rate": 0.01}
    writer.write_step(1, 100.0, sh)

    scalars = db.load_scalars(1)
    assert "learning_rate" in scalars


def test_create_callback(db):
    writer = StepWriter(db)
    cb = writer.create_callback()

    sh = {"loss": 0.42, "lr": 0.001}
    cb(1, None, step_history=sh, stack=None)

    bd = db.load_step_data(1)
    assert bd is not None
    assert bd.loss == pytest.approx(0.42)


def test_callback_skips_none_step_history(db):
    writer = StepWriter(db)
    cb = writer.create_callback()

    cb(1, None, step_history=None, stack=None)
    assert db.get_step_count() == 0
