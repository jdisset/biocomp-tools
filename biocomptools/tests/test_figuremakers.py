"""Tests for figuremakers (NetworkDiagram, GeneticCircuit, themes)."""

import pytest
import matplotlib.pyplot as plt

from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
from biocomp.network import recipe_to_networks
from biocomp.library import load_lib, LibraryContext


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture(scope="module")
def simple_network(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_simple",
            content=[
                CoTransfection(
                    name="test_cotx",
                    units=[
                        TranscriptionUnit(
                            name="reporter", slots=["hEF1a", "mNeonGreen", "L0.T_4560"]
                        ),
                        TranscriptionUnit(name="ern_unit", slots=["hEF1a", "CasE", "L0.T_4560"]),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )
        networks = recipe_to_networks(recipe, invert=True)
        return networks[0]


@pytest.fixture
def cleanup():
    yield
    plt.close("all")


class TestNetworkDiagram:
    def test_import(self):
        from biocomptools.toollib.figuremakers.networkdiagram import (
            NetworkDiagram,
            render_diagram_to_ax,
        )

        assert NetworkDiagram is not None
        assert render_diagram_to_ax is not None

    def test_create_diagram(self, simple_network, cleanup):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=simple_network, simplified=True)
        assert diagram is not None
        assert len(diagram.children) > 0

    def test_render_diagram(self, simple_network, cleanup):
        from biocomptools.toollib.figuremakers.networkdiagram import render_diagram_to_ax

        fig, ax = plt.subplots(figsize=(8, 6))
        render_diagram_to_ax(simple_network, ax, simplified=True)
        assert len(ax.get_children()) > 0

    def test_non_simplified(self, simple_network, cleanup):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=simple_network, simplified=False)
        assert diagram is not None
        assert len(diagram.children) > 0


class TestGeneticCircuit:
    def test_import(self):
        from biocomptools.toollib.figuremakers.geneticcircuit import (
            render_circuit_to_ax,
            GeneticCircuitFigure,
        )

        assert render_circuit_to_ax is not None
        assert GeneticCircuitFigure is not None

    def test_render_circuit(self, simple_network, cleanup):
        from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

        fig, ax = plt.subplots(figsize=(8, 6))
        render_circuit_to_ax(simple_network, ax, hide_marker_tus=True)
        assert len(ax.get_children()) > 0

    def test_to_circuit_data(self, simple_network):
        circuit_data = simple_network.to_circuit_data(hide_markers=False)
        assert circuit_data is not None
        assert len(circuit_data.transcription_units) > 0
        assert len(circuit_data.sources) > 0

    def test_ern_interaction_rendering(self, lib, cleanup):
        """Test that ERN interaction lines render correctly."""
        from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax

        with LibraryContext.with_library(lib):
            recipe = Recipe(
                name="test_ern",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(
                                name="reporter", slots=["hEF1a", "CasE_rec", "mNeonGreen", "L0.T_4560"]
                            ),
                            TranscriptionUnit(
                                name="ern_unit", slots=["hEF1a", "CasE", "L0.T_4560"]
                            ),
                        ]
                    )
                ],
            )
            networks = recipe_to_networks(recipe, invert=True)
            network = networks[0]

        # Check circuit data has interaction
        circuit_data = network.to_circuit_data(hide_markers=True)
        assert len(circuit_data.interactions) > 0
        assert circuit_data.interactions[0].interaction_type == "inhibition"

        # Render and check patches exist
        fig, ax = plt.subplots(figsize=(8, 6))
        render_circuit_to_ax(network, ax, hide_marker_tus=True)
        assert len(ax.patches) > 0


class TestNetworkAdapter:
    def test_import(self):
        from biocomptools.toollib.figuremakers.network_adapter import get_tu_informations, TUInfo

        assert get_tu_informations is not None
        assert TUInfo is not None

    def test_get_tu_infos(self, simple_network):
        from biocomptools.toollib.figuremakers.network_adapter import get_tu_informations

        tu_infos = get_tu_informations(simple_network)
        assert len(tu_infos) > 0
        for info in tu_infos.values():
            assert info.tu_name is not None


class TestThemeLoading:
    def test_network_diagram_theme_loads(self):
        from dracon import load, resolve_all_lazy
        from jeanplot.core import (
            Size,
            BoxStyle,
            LayoutConstraints,
            Offset,
            Shadow,
            SimpleBezierCurve,
            StraightCurve,
            OrthogonalCurve,
            LineEndFlat,
            LineEndCircle,
            LineEndArrow,
        )
        import importlib.resources

        jeanplot_types = [
            Size,
            BoxStyle,
            LayoutConstraints,
            Offset,
            Shadow,
            SimpleBezierCurve,
            StraightCurve,
            OrthogonalCurve,
            LineEndFlat,
            LineEndCircle,
            LineEndArrow,
        ]
        theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
            "network_diagram.yaml"
        )
        theme = load(
            str(theme_file), context={t.__name__: t for t in jeanplot_types}, raw_dict=True
        )
        resolve_all_lazy(theme)

        assert "ComputeNode" in theme
        assert "NetworkDiagram" in theme

    def test_genetic_schematic_theme_loads(self):
        from dracon import load, resolve_all_lazy
        from jeanplot.core import Size, BoxStyle, LayoutConstraints, Offset, Shadow
        from jeanplot.core.svg import LineEndFlat
        import importlib.resources

        jeanplot_types = [Size, BoxStyle, LayoutConstraints, Offset, Shadow, LineEndFlat]
        theme_file = importlib.resources.files("biocomptools.configs.themes").joinpath(
            "genetic_schematic.yaml"
        )
        theme = load(
            str(theme_file), context={t.__name__: t for t in jeanplot_types}, raw_dict=True
        )
        resolve_all_lazy(theme)

        assert "SourceAnnotation" in theme
        assert "GeneticSchematic" in theme


class TestCircuitPlot:
    def test_import(self):
        from biocomptools.circuitplot import CircuitPlotConfig, run_circuitplot

        assert CircuitPlotConfig is not None
        assert run_circuitplot is not None

    def test_render_card(self, lib, simple_network, cleanup, tmp_path):
        from biocomptools.circuitplot import _render_card, CircuitPlotConfig
        from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit

        with LibraryContext.with_library(lib):
            recipe = Recipe(
                name="test",
                content=[
                    CoTransfection(
                        units=[
                            TranscriptionUnit(name="r", slots=["hEF1a", "mNeonGreen", "L0.T_4560"])
                        ]
                    )
                ],
            )
            config = CircuitPlotConfig(
                recipe=recipe,
                output=str(tmp_path / "test_card.pdf"),
                plot_type="card",
            )
            output_path = tmp_path / "test_card.pdf"
            _render_card(simple_network, recipe, output_path, config)
            assert output_path.exists()
