# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""PlotConfig.rescaler default + prepare_func explicit-kwarg precedence.

Regression test for the silent-None propagation that caused
`'NoneType' object has no attribute 'fwd'` deep inside plot functions
when paper-jobs replaced the legacy plot_config YAML.

Class of bug guarded against: any "default-then-override" injection that
clobbers an explicit kwarg with None.
"""

import pytest

from biocomp.datautils import DataRescaler
from biocomp.utils import PartialFunction
from biocomptools.toollib.plot import PlotConfig, load_default_rescaler


class _FakeRescaler(DataRescaler):
    """Distinct DataRescaler subclass so identity checks are unambiguous."""

    def fwd(self, x):
        return x

    def inv(self, x):
        return x


def _record_kwargs(rescaler=None, **kw):
    _record_kwargs.last = {"rescaler": rescaler, **kw}
    return None


def test_load_default_rescaler_returns_real_rescaler():
    r = load_default_rescaler()
    assert isinstance(r, DataRescaler)
    assert hasattr(r, "fwd") and hasattr(r, "inv")


def test_plot_config_has_rescaler_by_default():
    """PlotConfig must ship with a usable rescaler — no silent None."""
    pc = PlotConfig()
    assert pc.rescaler is not None
    assert isinstance(pc.rescaler, DataRescaler)


def test_prepare_func_injects_default_rescaler_when_not_in_kwargs():
    pc = PlotConfig()
    pm = PartialFunction(
        func=f"{__name__}._record_kwargs",
        kwargs={},
        modules=[__name__],
    )
    f = pc.prepare_func(plot_method=pm, auto_callstack_bind=False)
    f()
    assert _record_kwargs.last["rescaler"] is pc.rescaler


def test_prepare_func_respects_explicit_rescaler_in_plot_method_kwargs():
    """Explicit rescaler in plot_method.kwargs MUST win over PlotConfig.rescaler.

    This is the regression: previously prepare_func unconditionally injected
    self.rescaler (potentially None), clobbering the YAML's explicit value.
    """
    explicit = _FakeRescaler()
    config_default = _FakeRescaler()
    pc = PlotConfig(rescaler=config_default)
    pm = PartialFunction(
        func=f"{__name__}._record_kwargs",
        kwargs={"rescaler": explicit},
        modules=[__name__],
    )
    f = pc.prepare_func(plot_method=pm, auto_callstack_bind=False)
    f()
    assert _record_kwargs.last["rescaler"] is explicit
    assert _record_kwargs.last["rescaler"] is not config_default


def test_prepare_func_with_none_config_rescaler_still_passes_explicit():
    """Even if PlotConfig.rescaler is explicitly None, explicit kwargs survive."""
    explicit = _FakeRescaler()
    pc = PlotConfig(rescaler=None)
    pm = PartialFunction(
        func=f"{__name__}._record_kwargs",
        kwargs={"rescaler": explicit},
        modules=[__name__],
    )
    f = pc.prepare_func(plot_method=pm, auto_callstack_bind=False)
    f()
    assert _record_kwargs.last["rescaler"] is explicit


def test_inherit_from_preserves_explicit_rescaler():
    parent = PlotConfig(rescaler=_FakeRescaler())
    child_rescaler = _FakeRescaler()
    child = PlotConfig(rescaler=child_rescaler)
    child.inherit_from(parent)
    assert child.rescaler is child_rescaler


def test_inherit_from_fills_missing_rescaler():
    parent_rescaler = _FakeRescaler()
    parent = PlotConfig(rescaler=parent_rescaler)
    child = PlotConfig(rescaler=None)
    child.inherit_from(parent)
    assert child.rescaler is parent_rescaler
