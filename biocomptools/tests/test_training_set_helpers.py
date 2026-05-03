"""Tests for `training_set_count` / `trained_on_status` blurb helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from biocomptools.toollib.figuremakers.datasetsummary import (
    trained_on_status,
    training_set_count,
)


def _model(names=None, weights=None):
    m = MagicMock()
    dmi = {}
    if names is not None:
        dmi['network_names'] = names
    if weights is not None:
        dmi['network_weights'] = weights
    m.metadata = {'data_manager_info': dmi}
    return m


def test_count_none_model():
    assert training_set_count(None) == (0, False)


def test_count_only_names_no_weights():
    m = _model(names=['a', 'b', 'c'])
    assert training_set_count(m) == (3, False)


def test_count_with_weights_excludes_zero():
    m = _model(names=['a', 'b', 'c', 'd'], weights=[1.0, 0.0, 2.0, 0.0])
    assert training_set_count(m) == (2, True)


def test_count_weights_length_mismatch_falls_back():
    """Mismatched length → ignore weights, fall back to name count."""
    m = _model(names=['a', 'b', 'c'], weights=[1.0])
    assert training_set_count(m) == (3, False)


def test_count_empty_metadata():
    m = MagicMock()
    m.metadata = {}
    assert training_set_count(m) == (0, False)


def test_status_none_model_or_name():
    assert trained_on_status(None, 'foo') == ''
    assert trained_on_status(_model(names=['foo']), None) == ''
    assert trained_on_status(_model(names=['foo']), '') == ''


def test_status_name_not_in_training_set():
    m = _model(names=['a', 'b'], weights=[1.0, 1.0])
    assert trained_on_status(m, 'c') == ''


def test_status_weights_recorded_positive():
    m = _model(names=['a', 'b'], weights=[1.5, 0.0])
    s = trained_on_status(m, 'a')
    assert 'seen during training' in s
    assert 'w=1.5' in s
    assert '[purple]' in s


def test_status_weights_recorded_zero():
    m = _model(names=['a', 'b'], weights=[1.0, 0.0])
    s = trained_on_status(m, 'b')
    assert 'excluded' in s
    assert 'w=0' in s
    assert '[grey]' in s


def test_status_weights_not_recorded():
    m = _model(names=['a', 'b'])
    s = trained_on_status(m, 'a')
    assert 'weight unknown' in s


def test_status_weights_length_mismatch_treated_as_unrecorded():
    m = _model(names=['a', 'b'], weights=[1.0])
    s = trained_on_status(m, 'a')
    assert 'weight unknown' in s
