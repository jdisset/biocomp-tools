# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Refactor-04 invariants on the biocomp side.

- paper_figure skeleton loads cleanly under a calling scope.
- paper-theme cascade fills biocomp-domain defaults (rescaler, knn_grid).
- per-panel YAML values still win over theme defaults.
- PaperSliceColumn / PaperValueHeatmap aliases override theme defaults when
  the caller passes a kwarg.

These tests are the canary for the biocomp <-> jeanplot paper-theme handoff;
if they go red, the cascade-fill chain through the `_body` extension hook
or the alias kwargs broke.
"""

from pathlib import Path

import dracon as dr
import pytest

from jeanplot import (
    SmoothPanel1D,
    SmoothPanel2D,
    jstyle,
    make_plot_context,
)
from biocomptools.jeanplot_panels import (
    JEANPLOT_PANEL_TYPES,
    get_jeanplot_panel_helpers,
)


_THIS = Path(__file__).resolve()
_PAPER_JOBS = _THIS.parents[4] / "paper-jobs"
_PLOT_CONFIG_PAPER = _PAPER_JOBS / "common" / "plot_config_paper.yaml"


def _make_loader(**extra_ctx):
    ctx = make_plot_context(
        extra_types=JEANPLOT_PANEL_TYPES,
        extra=get_jeanplot_panel_helpers(),
    )
    ctx.update(extra_ctx)
    return dr.DraconLoader(context=ctx, enable_interpolation=True)


@pytest.fixture(autouse=True)
def reset_jstyle():
    original = jstyle._cascade
    jstyle._cascade = None
    yield
    jstyle._cascade = original


def test_paper_theme_carries_biocomp_rescaler():
    """Cascade-fill: the rescaler default reaches every PlotPanel descendant."""
    loader = _make_loader()
    cfg = loader.load(str(_PLOT_CONFIG_PAPER))
    panel = SmoothPanel2D()
    props = cfg["rules"].invoke(component=panel)
    assert "rescaler" in props
    # CompressedSymLogRescaler — not just IdentityRescaler from the jeanplot side.
    assert props["rescaler"].__class__.__name__ == "CompressedSymLogRescaler"


def test_paper_theme_carries_jeanplot_paper_defaults():
    """Cascade-fill: biocomp paper theme inherits the colormap/colorbar_pad
    defaults from jeanplot's paper.yaml via the `_body` hook."""
    loader = _make_loader()
    cfg = loader.load(str(_PLOT_CONFIG_PAPER))
    panel = SmoothPanel2D()
    props = cfg["rules"].invoke(component=panel)
    assert props["colorbar_pad"] == 0.6
    assert props["vlim_min_floor"] == 0.0
    assert props["vlim_min_range"] == 0.1
    assert props["heatmap_params"]["cmap"] == "bc_blues"


def test_paper_theme_does_not_clobber_explicit_panel_values():
    """The headline invariant: a per-panel YAML value wins over the theme."""
    loader = _make_loader()
    cfg = loader.load(str(_PLOT_CONFIG_PAPER))
    panel = SmoothPanel2D(
        vlim_min_floor=0.42,
        colorbar_pad=0.123,
        heatmap_params={"cmap": "magma"},
    )
    jstyle.update(cfg["rules"])
    jstyle.apply(panel)
    assert panel.vlim_min_floor == 0.42
    assert panel.colorbar_pad == 0.123
    assert panel.heatmap_params["cmap"] == "magma"


def test_paper_theme_carries_shared_knn_grid():
    """Biocomp delta: the shared KNN config reaches every smooth panel."""
    loader = _make_loader()
    cfg = loader.load(str(_PLOT_CONFIG_PAPER))
    panel = SmoothPanel2D()
    props = cfg["rules"].invoke(component=panel)
    knn = props["knn_grid_params"]
    assert knn["knn_stats_params"]["k"] == 4000
    assert knn["grid_resolution"] == 250


def test_paper_theme_smooth1d_knn_stats_default():
    loader = _make_loader()
    cfg = loader.load(str(_PLOT_CONFIG_PAPER))
    panel = SmoothPanel1D()
    props = cfg["rules"].invoke(component=panel)
    assert props["knn_stats_params"]["k"] == 4000


def test_paper_slice_column_alias_overrides_kwargs():
    """!fn alias kwargs round-trip: caller-provided knn_stats override default."""
    loader = _make_loader()
    yaml = """
<<(<): !include file:{path}/common/paper_panels.yaml

panel: !PaperSliceColumn
  plot_data: !PlotData {{ xval: [1.0], yval: [2.0], input_names: ['a'], output_name: y }}
  slices: [0.1, 0.2, 0.3]
  colors: [['#000000', '#111111', '#222222']]
  slice_labels: ['a', 'b', 'c']
  chord_X: [[1.0]]
  chord_Y: [2.0]
  xlims: [0.0, 1.0]
  knn_stats_params: {{ k: 99, min_points: 1, radius: 0.5 }}
""".format(path=_PAPER_JOBS)
    cfg = loader.loads(yaml)
    panel = cfg["panel"]
    assert isinstance(panel, SmoothPanel1D)
    # The kwargs the caller passed should appear on the constructed panel.
    assert panel.knn_stats_params["k"] == 99
    assert panel.xlims == [0.0, 1.0]


def test_paper_figure_skeleton_loads_with_caller_scope():
    """Skeleton-include resolves variables from the calling scope.

    Setting `output_file` at the caller-level should reach the skeleton's
    `!set_default output_file:` and not be clobbered.
    """
    loader = _make_loader()
    yaml = """
<<(<): !include file:{path}/common/paper_figure.yaml

!define output_file: my_caller_value.svg
caller_output_file: ${{output_file}}
""".format(path=_PAPER_JOBS)
    cfg = loader.loads(yaml)
    assert cfg["caller_output_file"] == "my_caller_value.svg"


def test_fig1_matrix_gradient_round_trip():
    """The canonical migrated figure constructs into a 3-panel Figure tree."""
    from jeanplot.panels.figure import Figure

    loader = _make_loader()
    cfg = loader.load(str(_PAPER_JOBS / "plot" / "fig1_matrix_gradient.yaml"))
    fig = cfg["figure"]
    assert isinstance(fig, Figure)
    assert len(fig.children) == 3
    # Panel 1: value heatmap (built by !PaperValueHeatmap alias -> SmoothPanel2D)
    assert type(fig.children[0]).__name__ == "SmoothPanel2D"
    # Panel 2: gradient magnitude heatmap with gradient field overlay child
    assert type(fig.children[1]).__name__ == "SmoothGradMagnitudePanel2D"
    assert len(fig.children[1].children) == 1
    # Panel 3: 1D slice column built by !PaperSliceColumn alias
    assert type(fig.children[2]).__name__ == "SmoothPanel1D"
    assert len(fig.children[2].children) >= 1  # chord overlay always present
