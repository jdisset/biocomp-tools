# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Smoke tests for jeanplot_panels: each panel constructs + draws without crash.

Data-dependent panels (mvp, voxel, quantile, ...) are exercised lightly: we
construct them with minimal shells and rely on draw() being a thin delegation
to the underlying render_*_to_ax function.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biocomp.library import LibraryContext, load_lib
from biocomp.network import recipe_to_networks
from biocomp.recipe import CoTransfection, Recipe, TranscriptionUnit


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture(scope="module")
def simple_network(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="jp_smoke",
            content=[
                CoTransfection(
                    name="cotx0",
                    units=[
                        TranscriptionUnit(
                            name="reporter",
                            slots=["hEF1a", "mNeonGreen", "L0.T_4560"],
                        ),
                        TranscriptionUnit(
                            name="reg",
                            slots=["hEF1a", "CasE", "L0.T_4560"],
                        ),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )
        return recipe_to_networks(recipe, invert=True)[0]


@pytest.fixture
def ax():
    fig, a = plt.subplots(figsize=(4, 4))
    yield a
    plt.close(fig)


def test_circuit_panel_draws(simple_network, ax):
    from biocomptools.jeanplot_panels import CircuitPanel

    CircuitPanel(network=simple_network).draw(ax)


def test_network_diagram_panel_draws(simple_network, ax):
    from biocomptools.jeanplot_panels import NetworkDiagramPanel

    NetworkDiagramPanel(network=simple_network).draw(ax)


def test_blurb_panel_draws(ax):
    from biocomptools.jeanplot_panels import BlurbPanel

    BlurbPanel(
        text="# Header\nBody text with **bold** and *italic*.",
        title="A blurb",
    ).draw(ax)


def test_empty_panel_draws(ax):
    from biocomptools.jeanplot_panels import EmptyPanel

    EmptyPanel(text="placeholder").draw(ax)


def test_constant_text_panel_draws(ax):
    from biocomptools.jeanplot_panels import ConstantTextPanel

    ConstantTextPanel(text="hello", x=0.5, y=0.5).draw(ax)


def test_activations_panel_draws(ax):
    from biocomptools.jeanplot_panels import ActivationsPanel

    ActivationsPanel().draw(ax)


def test_lattice_heatmap_panel_handles_missing_data(ax):
    """Panel must degrade gracefully when result has no lattice_grid."""
    from biocomptools.jeanplot_panels import LatticeHeatmapPanel
    from biocomptools.toollib.figuremakers.designutils import DesignResult
    from biocomp.plotutils import PlotData

    dummy_pd = PlotData(
        xval=np.zeros((2, 2), dtype=np.float32),
        yval=np.zeros((2, 1), dtype=np.float32),
        input_names=["a", "b"],
        output_name="y",
    )
    result = DesignResult(
        network=None,
        target=None,
        target_name="t",
        rank=0,
        replicate=0,
        scaffold_network_name="scaffold",
        loss=0.5,
        recipe_hash="abc",
        run_name="run",
        model=None,
        gt_data=dummy_pd,
        pred_data=dummy_pd,
        lattice_data=None,
        lattice_grid=None,
        lattice_extent=None,
        lattice_resolution=None,
    )
    LatticeHeatmapPanel(result=result).draw(ax)


def test_design_metrics_panel_draws(ax):
    from biocomptools.jeanplot_panels import DesignMetricsPanel
    from biocomptools.toollib.figuremakers.designutils import DesignResult
    from biocomp.plotutils import PlotData

    dummy_pd = PlotData(
        xval=np.zeros((2, 2), dtype=np.float32),
        yval=np.zeros((2, 1), dtype=np.float32),
        input_names=["a", "b"],
        output_name="y",
    )
    result = DesignResult(
        network=None,
        target=None,
        target_name="t",
        rank=0,
        replicate=0,
        scaffold_network_name="scaf",
        loss=0.4,
        recipe_hash="abc",
        run_name="run",
        model=None,
        gt_data=dummy_pd,
        pred_data=dummy_pd,
        lattice_data=None,
        lattice_grid=None,
        lattice_extent=None,
        lattice_resolution=None,
    )
    DesignMetricsPanel(result=result).draw(ax)


def test_network_plot_data_adapter_roundtrip():
    from biocomp.plotutils import PlotData

    from biocomptools.jeanplot_panels import NetworkPlotData

    src = PlotData(
        xval=np.random.rand(50, 2).astype(np.float32),
        yval=np.random.rand(50, 1).astype(np.float32),
        input_names=["x1", "x2"],
        output_name="y",
    )
    jpd = NetworkPlotData(source=src).to_jeanplot()
    assert jpd.x.shape == (50, 2)
    assert jpd.y.shape == (50, 1)
    assert jpd.input_names == ["x1", "x2"]
    assert jpd.output_name == "y"


def test_default_types_registration_count():
    from biocomptools.jeanplot_panels import JEANPLOT_PANEL_TYPES

    assert len(JEANPLOT_PANEL_TYPES) == 24


def test_helpers_registration_includes_row_composer():
    from biocomptools.jeanplot_panels import get_jeanplot_panel_helpers

    helpers = get_jeanplot_panel_helpers()
    assert "build_per_network_row" in helpers
    assert "filter_compatible" in helpers
    assert "build_figure_metadata" in helpers
