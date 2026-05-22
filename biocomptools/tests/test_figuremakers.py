# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for figuremakers (NetworkDiagram, GeneticCircuit, themes)."""

import pytest
import matplotlib.pyplot as plt
from jeanplot.core.text import Text
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

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


@pytest.fixture(scope="module")
def marker_input_network(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_marker_input",
            content=[
                CoTransfection(
                    name="c1",
                    units=[
                        TranscriptionUnit(name="x1", slots=["hEF1a", "mNeonGreen", "L0.T_4560"]),
                        TranscriptionUnit(name="reg", slots=["hEF1a", "CasE", "L0.T_4560"]),
                    ],
                    ratios=[0.5, 0.5],
                ),
                CoTransfection(
                    name="c2",
                    units=[
                        TranscriptionUnit(
                            name="y", slots=["hEF1a", "CasE_rec", "mKO2", "L0.T_4560"]
                        ),
                    ],
                    ratios=[1.0],
                ),
            ],
        )
        return recipe_to_networks(recipe, invert=True)[0]


@pytest.fixture(scope="module")
def bias_only_network(lib):
    from biocomp.recipe import FluoIntensity

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_bias_only",
            content=[
                CoTransfection(
                    name="x1",
                    units=[TranscriptionUnit(name="reg", slots=["hEF1a", "CasE", "L0.T_4560"])],
                    ratios=[1.0],
                    fluo_bias=FluoIntensity(tu_id=0, value=100.0, protein="mNeonGreen"),
                ),
                CoTransfection(
                    name="x2",
                    units=[
                        TranscriptionUnit(
                            name="rep", slots=["hEF1a", "CasE_rec", "mKO2", "L0.T_4560"]
                        ),
                    ],
                    ratios=[1.0],
                ),
            ],
        )
        return recipe_to_networks(recipe, invert=True)[0]


@pytest.fixture(scope="module")
def bias_marker_network(lib):
    from biocomp.recipe import FluoIntensity

    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_bias_marker",
            content=[
                CoTransfection(
                    name="c1",
                    units=[
                        TranscriptionUnit(name="x1", slots=["hEF1a", "mNeonGreen", "L0.T_4560"]),
                        TranscriptionUnit(name="reg", slots=["hEF1a", "CasE", "L0.T_4560"]),
                    ],
                    ratios=[0.5, 0.5],
                    fluo_bias=FluoIntensity(tu_id=0, value=100.0, protein="mNeonGreen"),
                ),
                CoTransfection(
                    name="c2",
                    units=[
                        TranscriptionUnit(
                            name="y", slots=["hEF1a", "CasE_rec", "mKO2", "L0.T_4560"]
                        ),
                    ],
                    ratios=[1.0],
                ),
            ],
        )
        return recipe_to_networks(recipe, invert=True)[0]


@pytest.fixture(scope="module")
def multi_dependent_output_network(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="test_multi_dependent_outputs",
            content=[
                CoTransfection(
                    name="c1",
                    units=[TranscriptionUnit(name="reg1", slots=["hEF1a", "CasE", "L0.T_4560"])],
                ),
                CoTransfection(
                    name="c2",
                    units=[TranscriptionUnit(name="reg2", slots=["hEF1a", "Csy4", "L0.T_4560"])],
                ),
                CoTransfection(
                    name="c3",
                    units=[
                        TranscriptionUnit(
                            name="o1", slots=["hEF1a", "CasE_rec", "mKO2", "L0.T_4560"]
                        )
                    ],
                ),
                CoTransfection(
                    name="c4",
                    units=[
                        TranscriptionUnit(
                            name="o2", slots=["hEF1a", "Csy4_rec", "eBFP2", "L0.T_4560"]
                        )
                    ],
                ),
            ],
        )
        return recipe_to_networks(recipe, invert=True)[0]


def _apply_network_theme(diagram):
    from dracon import load, resolve_all_lazy
    from jeanplot import jstyle, Container
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

    types = [
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
    theme = load(str(theme_file), context={t.__name__: t for t in types}, raw_dict=True)
    resolve_all_lazy(theme)
    jstyle.update(theme)

    root = Container(
        children=[diagram],
        layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
    )
    jstyle.apply(root)
    return root


@pytest.fixture
def cleanup():
    yield
    plt.close("all")


class TestNetworkDiagram:
    @staticmethod
    def _get_input_layer_legend_texts(diagram):
        input_layers = [
            child
            for child in diagram.children
            if hasattr(child, "style_class") and "input_layer" in child.style_class
        ]
        assert input_layers
        return [
            child
            for child in input_layers[0].children
            if isinstance(child, Text) and "legend_text" in child.style_class
        ]

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

    def test_simplified_hides_marker_only_subgraph(self, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=marker_input_network, simplified=True)
        marker_tu_ids = diagram._marker_tu_ids
        assert marker_tu_ids

        graph = marker_input_network.compute_graph
        marker_compute_nodes = set()
        for edge in graph.edges.values():
            edge_tu_ids = set(diagram._edge_tu_ids(edge))
            if not (edge_tu_ids & marker_tu_ids):
                continue
            for node_id in (edge.source_id, edge.target_id):
                node = graph.nodes[node_id]
                if node.node_type in ("transcription", "translation"):
                    marker_compute_nodes.add(node_id)

        assert marker_compute_nodes
        assert marker_compute_nodes.issubset(diagram._marker_only_nodes)
        assert marker_compute_nodes.isdisjoint(set(diagram._nodes))

    def test_simplified_hides_marker_only_subgraph_with_bias(self, bias_marker_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=bias_marker_network, simplified=True)
        marker_tu_ids = diagram._marker_tu_ids
        assert marker_tu_ids

        graph = bias_marker_network.compute_graph
        marker_compute_nodes = set()
        for edge in graph.edges.values():
            edge_tu_ids = set(diagram._edge_tu_ids(edge))
            if not (edge_tu_ids & marker_tu_ids):
                continue
            for node_id in (edge.source_id, edge.target_id):
                node = graph.nodes[node_id]
                if node.node_type in ("transcription", "translation"):
                    marker_compute_nodes.add(node_id)

        assert marker_compute_nodes
        assert marker_compute_nodes.issubset(diagram._marker_only_nodes)
        assert marker_compute_nodes.isdisjoint(set(diagram._nodes))

    def test_marker_input_nodes_get_shadow_from_theme(self, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=marker_input_network, simplified=False)
        _apply_network_theme(diagram)

        marker_names = {
            p.upper() for p in marker_input_network.get_inverted_input_proteins(include_biases=True)
        }
        input_nodes = [
            node
            for node in diagram._nodes.values()
            if node.node_type == "input" and marker_names.intersection(node.style_class)
        ]
        assert input_nodes
        assert all(node.style.shadow is not None for node in input_nodes)

    def test_simplified_shows_colored_inputs_for_collapsed_sources(self, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=marker_input_network, simplified=True)
        _apply_network_theme(diagram)

        marker_names = {
            p.upper() for p in marker_input_network.get_inverted_input_proteins(include_biases=True)
        }
        legends = self._get_input_layer_legend_texts(diagram)
        input_legends = [
            t
            for t in legends
            if t.text.startswith("(") and t.text.split(")")[0].lstrip("(").upper() in marker_names
        ]
        assert input_legends
        assert all(t.attached_to is not None for t in input_legends)
        assert all(len(t.text.split()) >= 2 for t in input_legends)

    def test_bias_collapsed_aggregation_gets_marker_class(self, bias_only_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=bias_only_network, simplified=True)
        assert not diagram._collapsed_marker_proteins
        assert diagram._cotx_marker_map.get("x1") == "mNeonGreen"

        graph = bias_only_network.compute_graph
        agg_node = next(
            n for n in graph.get_nodes_by_type("aggregation") if n.extra.get("cotx_group") == "x1"
        )
        agg_component = diagram._nodes[agg_node.node_id]
        assert "MNEONGREEN" in agg_component.style_class

    def test_bias_nodes_get_shadow_from_theme(self, bias_marker_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=bias_marker_network, simplified=False)
        _apply_network_theme(diagram)

        bias_nodes = [
            node
            for node in diagram._nodes.values()
            if node.node_type == "bias" and "MNEONGREEN" in node.style_class
        ]
        assert bias_nodes
        assert all(node.style.shadow is not None for node in bias_nodes)

    def test_simplified_shows_colored_bias_inputs(self, bias_marker_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=bias_marker_network, simplified=True)
        _apply_network_theme(diagram)

        legends = self._get_input_layer_legend_texts(diagram)
        expected_marker = diagram._cotx_marker_map.get("c1")
        assert expected_marker is not None
        bias_legends = [t for t in legends if t.text.startswith(f"({expected_marker})")]
        assert bias_legends
        assert all(t.attached_to is not None for t in bias_legends)
        assert all(len(t.text.split()) >= 2 for t in bias_legends)

    def test_simplified_orders_input_before_bias(self, bias_marker_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=bias_marker_network, simplified=True)
        _apply_network_theme(diagram)
        legends = self._get_input_layer_legend_texts(diagram)

        attached = [t for t in legends if t.attached_to is not None]
        assert attached
        if len(attached) < 2:
            return
        attached.sort(key=lambda t: t.attachment_offset.absolute[1])
        node_types = []
        for text in attached:
            if text.id is None or not text.id.startswith("legend_"):
                continue
            nid = int(text.id.split("_", 1)[1])
            node = diagram._nodes.get(nid)
            if node is not None:
                node_types.append(node.node_type)
        if "input" in node_types and "bias" in node_types:
            assert node_types.index("input") < node_types.index("bias")

    def test_simplified_legend_right_align_anchors_to_offset(self, marker_input_network, cleanup):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=marker_input_network, simplified=True)
        root = _apply_network_theme(diagram)

        legends = [
            t for t in self._get_input_layer_legend_texts(diagram) if t.attached_to is not None
        ]
        assert legends
        legend = legends[0]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.set_aspect("equal")
        ax.axis("off")
        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, root, adjust_lims=True)
        fig.canvas.draw()

        artists = [a for a in ax.texts if a.get_text() == legend.text]
        assert artists
        artist = artists[0]
        bbox = artist.get_window_extent(fig.canvas.get_renderer())
        (x0, _y0), (x1, _y1) = ax.transData.inverted().transform(
            [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]]
        )
        assert x1 >= x0

        bounds = legend.get_world_bounds()
        assert bounds is not None
        assert abs(x1 - bounds[2]) < 2.0

    def test_output_color_uses_dependent_output_order(self, multi_dependent_output_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        class ReversedInfoNetwork:
            def __init__(self, network):
                self._network = network
                self.compute_graph = network.compute_graph

            def generate_network_info(self):
                info = self._network.generate_network_info()
                info = dict(info)
                info["dependent_outputs"] = tuple(reversed(info.get("dependent_outputs", ())))
                return info

            def __getattr__(self, name):
                return getattr(self._network, name)

        wrapped = ReversedInfoNetwork(multi_dependent_output_network)
        expected = wrapped.get_dependent_output_proteins()
        assert len(expected) > 1

        info_first = wrapped.generate_network_info()["dependent_outputs"][0].upper()
        true_first = expected[0].upper()
        assert info_first != true_first

        diagram = NetworkDiagram(network=wrapped, simplified=True)
        output_nodes = [node for node in diagram._nodes.values() if node.node_type == "output"]
        assert len(output_nodes) == 1
        assert true_first in output_nodes[0].style_class


class TestLayoutSpec:
    def test_layout_spec_from_networks(self, simple_network, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import LayoutSpec

        spec = LayoutSpec.from_networks([simple_network, marker_input_network])
        assert spec.ern_slot_order is not None
        assert len(spec.ern_slot_order) > 0
        assert spec.ern_slot_order == sorted(spec.ern_slot_order)
        assert spec.max_ern_layers is not None
        assert spec.max_ern_layers >= 1
        assert spec.canvas_size is None

    def test_layout_spec_no_effect_when_none(self, simple_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram

        diagram = NetworkDiagram(network=simple_network, simplified=True, layout_spec=None)
        assert diagram is not None
        assert len(diagram.children) > 0

    def test_layout_spec_canvas_size(self, simple_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec
        from jeanplot.core.models import Size

        spec = LayoutSpec(canvas_size=Size(400, 300))
        diagram = NetworkDiagram(network=simple_network, simplified=True, layout_spec=spec)
        assert diagram.min_dimensions.width == 400
        assert diagram.min_dimensions.height == 300
        assert diagram.max_dimensions.width == 400
        assert diagram.max_dimensions.height == 300

    def test_layout_spec_ern_slot_order_adds_spacers(self, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec

        spec = LayoutSpec(ern_slot_order=["CasE", "Csy4", "PgU"])
        diagram = NetworkDiagram(network=marker_input_network, simplified=True, layout_spec=spec)
        ern_layers = [
            c
            for c in diagram.children
            if hasattr(c, "style_class") and "main_layer" in c.style_class
        ]
        assert ern_layers
        spacers = []
        for layer in ern_layers:
            for child in layer.children:
                if hasattr(child, "style_class") and "ern_spacer" in child.style_class:
                    spacers.append(child)
        assert (
            len(spacers) >= 2
        )  # Csy4 and PgU should be spacers (marker_input_network only has CasE)

    def test_layout_spec_max_ern_layers_pads_columns(self, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec

        spec = LayoutSpec(max_ern_layers=3)
        diagram = NetworkDiagram(network=marker_input_network, simplified=True, layout_spec=spec)
        ern_layers = [
            c
            for c in diagram.children
            if hasattr(c, "style_class") and "main_layer" in c.style_class
        ]
        assert len(ern_layers) >= 1  # at least the original ERN layer(s)

    def test_layout_spec_layer_min_height(self, simple_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec

        spec = LayoutSpec(layer_min_height=200.0)
        diagram = NetworkDiagram(network=simple_network, simplified=True, layout_spec=spec)
        layers = [
            c for c in diagram.children if hasattr(c, "style_class") and "layer" in c.style_class
        ]
        for layer in layers:
            assert layer.min_dimensions.height >= 200.0

    def test_layout_spec_column_widths(self, simple_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec

        spec = LayoutSpec(column_widths={"input_layer": 150.0, "output_layer": 100.0})
        diagram = NetworkDiagram(network=simple_network, simplified=True, layout_spec=spec)
        input_layers = [
            c
            for c in diagram.children
            if hasattr(c, "style_class") and "input_layer" in c.style_class
        ]
        if input_layers:
            assert input_layers[0].min_dimensions.width >= 150.0

    def test_shared_layout_spec_across_diagrams(self, simple_network, marker_input_network):
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, LayoutSpec
        from jeanplot.core.models import Size

        spec = LayoutSpec.from_networks([simple_network, marker_input_network])
        spec.canvas_size = Size(400, 300)

        diagram_a = NetworkDiagram(network=simple_network, simplified=True, layout_spec=spec)
        diagram_b = NetworkDiagram(network=marker_input_network, simplified=True, layout_spec=spec)
        assert diagram_a.min_dimensions.width == diagram_b.min_dimensions.width
        assert diagram_a.min_dimensions.height == diagram_b.min_dimensions.height

    def test_render_diagram_with_layout_spec(self, simple_network, cleanup):
        from biocomptools.toollib.figuremakers.networkdiagram import (
            render_diagram_to_ax,
            LayoutSpec,
        )
        from jeanplot.core.models import Size

        spec = LayoutSpec(
            canvas_size=Size(400, 300),
            ern_slot_order=["CasE", "Csy4"],
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        render_diagram_to_ax(simple_network, ax, simplified=True, layout_spec=spec)
        assert len(ax.get_children()) > 0

    def test_semantic_key(self, simple_network):
        from biocomptools.toollib.figuremakers.networkdiagram import semantic_key

        graph = simple_network.compute_graph
        for node in graph.nodes.values():
            key = semantic_key(node, graph)
            assert ":" in key
            prefix = key.split(":")[0]
            assert prefix, f"Empty prefix for node {node.node_id} type={node.node_type}"


class TestGeneticCircuit:
    def test_import(self):
        from biocomptools.toollib.figuremakers.geneticcircuit import render_circuit_to_ax
        from biocomptools.jeanplot_panels import CircuitPanel

        assert render_circuit_to_ax is not None
        assert CircuitPanel is not None

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
                                name="reporter",
                                slots=["hEF1a", "CasE_rec", "mNeonGreen", "L0.T_4560"],
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


class TestEmbeddingTrajectories:
    def test_resolve_trajectories_from_memory(self):
        """_resolve_trajectories uses in-memory trajectories when provided."""
        from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure

        trajectories = {
            "tl_rate_0": [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
            "tl_rate_1": [(0.7, 0.8), (0.9, 1.0)],
        }
        # Call the static-like logic directly via the method on a minimal stub
        result = InnerNodesFigure._resolve_trajectories_static(
            embedding_trajectories=trajectories,
            history_dir=None,
            emb_type="tl_rate",
            names=["uorf_a", "uorf_b"],
        )
        assert result["uorf_a"] == trajectories["tl_rate_0"]
        assert result["uorf_b"] == trajectories["tl_rate_1"]

    def test_resolve_trajectories_none_falls_back(self):
        """When embedding_trajectories is None, falls back to disk (returns empty without history_dir)."""
        from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure

        result = InnerNodesFigure._resolve_trajectories_static(
            embedding_trajectories=None,
            history_dir=None,
            emb_type="tl_rate",
            names=["uorf_a"],
        )
        assert result == {}

    def test_plotlogger_build_trajectories(self):
        """PlotLogger._build_trajectories converts snapshots to trajectory dict."""
        from biocomptools.toollib.loggers.plotlogger import PlotLogger

        snapshots = [
            (10, {"tl_rate": [[0.1, 0.2], [0.3, 0.4]]}),
            (20, {"tl_rate": [[0.5, 0.6], [0.7, 0.8]]}),
            (30, {"tl_rate": [[0.9, 1.0], [1.1, 1.2]]}),
        ]
        result = PlotLogger._build_trajectories(snapshots)
        assert "tl_rate_0" in result
        assert "tl_rate_1" in result
        assert len(result["tl_rate_0"]) == 3
        assert result["tl_rate_0"][0] == (0.1, 0.2)
        assert result["tl_rate_1"][2] == (1.1, 1.2)

    def test_plotlogger_build_trajectories_empty(self):
        """Empty snapshots produce empty trajectories."""
        from biocomptools.toollib.loggers.plotlogger import PlotLogger

        assert PlotLogger._build_trajectories([]) == {}


class TestNodeInfo:
    def test_scalar_emb(self):
        from biocomptools.toollib.figuremakers.innernodes import NodeInfo, ApplyFn

        dummy_fn = ApplyFn(single=lambda: 0.0, batch=lambda arr, **kw: arr)
        n = NodeInfo("test", "Translation", dummy_fn, 0, "tl_rate", 0.42)
        assert n.emb_dim == 1
        assert n.emb_scalar == 0.42

    def test_tuple_emb(self):
        from biocomptools.toollib.figuremakers.innernodes import NodeInfo, ApplyFn

        dummy_fn = ApplyFn(single=lambda: 0.0, batch=lambda arr, **kw: arr)
        n = NodeInfo("test", "ERN", dummy_fn, 0, "affinity", (0.1, 0.9))
        assert n.emb_dim == 2
        assert n.emb_scalar is None

    def test_none_emb(self):
        from biocomptools.toollib.figuremakers.innernodes import NodeInfo, ApplyFn

        dummy_fn = ApplyFn(single=lambda: 0.0, batch=lambda arr, **kw: arr)
        n = NodeInfo("test", "Source", dummy_fn, 0)
        assert n.emb_dim == 0
        assert n.emb_scalar is None

    def test_high_dim_emb(self):
        from biocomptools.toollib.figuremakers.innernodes import NodeInfo, ApplyFn

        dummy_fn = ApplyFn(single=lambda: 0.0, batch=lambda arr, **kw: arr)
        n = NodeInfo("test", "ERN", dummy_fn, 0, "affinity", (0.1, 0.5, 0.9))
        assert n.emb_dim == 3
        assert n.emb_scalar is None


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
