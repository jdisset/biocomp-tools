# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Refactor-05 invariants: legacy plotting machinery emits DeprecationWarning.

The warnings name the canonical jeanplot replacement so consumers know what
to migrate to. The migrated `paper-jobs/plot/fig1_matrix_gradient.yaml` path
constructs only jeanplot-native objects and must stay warning-free.
"""

import pytest

from biocomp._legacy_deprecation import reset_seen


@pytest.fixture(autouse=True)
def _clear_warning_cache():
    reset_seen()
    yield
    reset_seen()


def test_simple_layout_warns():
    from biocomp.plotutils import SimpleLayout

    with pytest.warns(DeprecationWarning, match="jeanplot"):
        SimpleLayout()


def test_grid_layout_warns():
    from biocomp.plotutils import GridLayout

    with pytest.warns(DeprecationWarning, match="jeanplot"):
        GridLayout()


def test_multi_row_grid_layout_warns():
    from biocomp.plotutils import MultiRowGridLayout

    with pytest.warns(DeprecationWarning, match="jeanplot"):
        MultiRowGridLayout(rows=[[1.0]], row_heights=[1.0])


def test_figure_spec_warns():
    from biocomp.plotutils import FigureSpec

    with pytest.warns(DeprecationWarning, match="jeanplot"):
        FigureSpec()


def test_merge_spec_warns():
    from biocomp.plotutils import MergeSpec

    with pytest.warns(DeprecationWarning, match="jeanplot"):
        MergeSpec()


def test_generate_full_nested_config_warns():
    from biocomp.utils import generate_full_nested_config

    with pytest.warns(DeprecationWarning, match="cascade-fill"):
        generate_full_nested_config({})


def test_legacy_figure_warns():
    from biocomptools.toollib.plot import BiocompPlotFigure
    from biocomp.plotutils import FigureSpec

    reset_seen()
    with pytest.warns(DeprecationWarning) as record:
        BiocompPlotFigure(figure_spec=FigureSpec())
    messages = [str(w.message) for w in record]
    assert any("biocomptools.toollib.plot.Figure" in m for m in messages)
    assert any("jeanplot" in m for m in messages)


def test_biocomp_figure_adapter_warns():
    from biocomptools.jeanplot_panels.biocomp_figure_adapter import BiocompFigureAdapter

    class _Stub:
        figure_spec = None

    with pytest.warns(DeprecationWarning, match="native jeanplot panels"):
        BiocompFigureAdapter(biocomp_figure=_Stub())


def test_jeanplot_figure_is_warning_free():
    """The migrated fig1 path constructs jeanplot.Figure directly. No legacy warning fires."""
    import warnings

    from jeanplot.panels.figure import Figure as JpFigure

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        JpFigure()
