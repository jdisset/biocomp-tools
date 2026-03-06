"""Tests for step_history_triage — SSOT data routing."""

import numpy as np
import pytest

from biocomptools.step_history_triage import (
    ARRAY_KEYS,
    BLOB_KEYS,
    triage_step_history,
)


def test_loss_extraction_float():
    t = triage_step_history({"loss": 0.5})
    assert t.loss == pytest.approx(0.5)


def test_loss_extraction_numpy():
    t = triage_step_history({"loss": np.array(0.42)})
    assert t.loss == pytest.approx(0.42)


def test_loss_extraction_none():
    t = triage_step_history({"loss": None})
    assert t.loss is None


def test_loss_extraction_missing():
    t = triage_step_history({})
    assert t.loss is None


def test_array_keys_go_to_arrays():
    sh = {
        "yhatdep": np.ones((10, 3)),
        "X": np.zeros((10, 2)),
        "Y": np.zeros((10, 1)),
        "all_losses": np.ones(10),
    }
    t = triage_step_history(sh)
    for k in ("yhatdep", "X", "Y", "all_losses"):
        assert k in t.arrays, f"{k} should be in arrays"
        assert isinstance(t.arrays[k], np.ndarray)
    assert len(t.blobs) == 0


def test_blob_keys_go_to_blobs():
    sh = {
        "params": {"a": 1},
        "latest_params": {"b": 2},
        "grad": {"c": 3},
        "apply_aux": {"nested": {"deep": [1, 2, 3]}},
        "opt_state": {"state": "data"},
    }
    t = triage_step_history(sh)
    for k in ("params", "latest_params", "grad", "apply_aux", "opt_state"):
        assert k in t.blobs, f"{k} should be in blobs"
    assert len(t.arrays) == 0


def test_apply_aux_never_in_dicts():
    """Critical: apply_aux must NOT end up in dicts (the original bug)."""
    sh = {"apply_aux": {"layer0": {"weight": np.ones(100).tolist()}}}
    t = triage_step_history(sh)
    assert "apply_aux" not in t.dicts
    assert "apply_aux" not in t.scalars
    assert "apply_aux" in t.blobs


def test_scalar_values_go_to_scalars():
    sh = {"learning_rate": 0.001, "step_time": 1.5}
    t = triage_step_history(sh)
    assert "learning_rate" in t.scalars
    assert t.scalars["learning_rate"] == pytest.approx(0.001)
    assert "step_time" in t.scalars


def test_dict_values_go_to_dicts():
    sh = {"sublosses": {"mse": 0.1, "l0": 0.2}, "tu_stats": {"count": 5}}
    t = triage_step_history(sh)
    assert "sublosses" in t.dicts
    assert "tu_stats" in t.dicts


def test_small_numpy_scalar_goes_to_scalars():
    sh = {"lr": np.float32(0.01)}
    t = triage_step_history(sh)
    assert "lr" in t.scalars
    assert t.scalars["lr"] == pytest.approx(0.01, abs=1e-5)


def test_small_array_goes_to_dicts():
    sh = {"small_arr": np.arange(50)}
    t = triage_step_history(sh)
    assert "small_arr" in t.dicts
    assert "_list" in t.dicts["small_arr"]


def test_large_unknown_array_goes_to_arrays():
    sh = {"big_unknown": np.zeros(500)}
    t = triage_step_history(sh)
    assert "big_unknown" in t.arrays


def test_none_values_skipped_in_blobs():
    sh = {"params": None, "grad": None}
    t = triage_step_history(sh)
    assert "params" not in t.blobs
    assert "grad" not in t.blobs


def test_none_values_skipped_in_arrays():
    sh = {"yhatdep": None}
    t = triage_step_history(sh)
    assert "yhatdep" not in t.arrays


def test_triaged_is_frozen():
    t = triage_step_history({"loss": 1.0})
    with pytest.raises(AttributeError):
        t.loss = 2.0  # type: ignore[misc]


def test_key_constants_disjoint():
    assert ARRAY_KEYS.isdisjoint(BLOB_KEYS), "ARRAY_KEYS and BLOB_KEYS must not overlap"
