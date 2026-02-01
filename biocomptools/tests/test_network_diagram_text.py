"""Test text sizes in actual NetworkDiagram components."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from jeanplot import jstyle
from jeanplot.core.renderer.matplotlib import MatplotlibRenderer

from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
from biocomp.network import recipe_to_networks
from biocomp.library import load_lib, LibraryContext


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.fixture
def cleanup():
    yield
    plt.close("all")


def build_simple_single_reporter(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="simple_single_reporter",
            content=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        TranscriptionUnit(name="reporter", slots=["hEF1a", "mNeonGreen", "L0.T_4560"]),
                    ],
                    ratios=[1.0],
                )
            ],
        )
        networks = recipe_to_networks(recipe, invert=True)
        return networks[0] if networks else None


def build_multi_cotx_aggregation(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="multi_cotx_aggregation",
            content=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        TranscriptionUnit(name="reporter1", slots=["hEF1a", "mNeonGreen", "L0.T_4560"]),
                    ],
                    ratios=[1.0],
                ),
                CoTransfection(
                    name="cotx2",
                    units=[
                        TranscriptionUnit(name="reporter2", slots=["hEF1a", "mKO2", "L0.T_4560"]),
                    ],
                    ratios=[1.0],
                ),
            ],
        )
        networks = recipe_to_networks(recipe, invert=True)
        return networks[0] if networks else None


def build_two_reporters_with_ern(lib):
    with LibraryContext.with_library(lib):
        recipe = Recipe(
            name="two_reporters_with_ern",
            content=[
                CoTransfection(
                    name="cotx1",
                    units=[
                        TranscriptionUnit(name="reporter", slots=["hEF1a", "CasE_rec", "mNeonGreen", "L0.T_4560"]),
                        TranscriptionUnit(name="ern", slots=["hEF1a", "CasE", "L0.T_4560"]),
                    ],
                    ratios=[0.5, 0.5],
                )
            ],
        )
        networks = recipe_to_networks(recipe, invert=True)
        return networks[0] if networks else None


class TestNetworkDiagramText:
    """Test text rendering in actual network diagrams."""

    def measure_diagram_text(self, network, name: str):
        """Measure text sizes in a network diagram."""
        from biocomptools.toollib.figuremakers.networkdiagram import NetworkDiagram, TranscriptionNode
        from jeanplot import Container, LayoutConstraints, load_default_theme

        load_default_theme(force=True)

        diagram = NetworkDiagram(network=network, simplified=True)
        root = Container(
            children=[diagram],
            layout=LayoutConstraints(direction="row", justify_content="center", align_items="stretch"),
        )
        jstyle.apply(root)

        tx_node = None
        def find_tx(comp):
            nonlocal tx_node
            if isinstance(comp, TranscriptionNode):
                tx_node = comp
                return
            if hasattr(comp, 'children'):
                for c in comp.children:
                    find_tx(c)
        find_tx(root)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.set_aspect("equal")
        ax.axis("off")

        renderer = MatplotlibRenderer()
        renderer.create_context(ax=ax)
        renderer.render_component(ax, root, adjust_lims=True)

        fig.canvas.draw()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        bbox = ax.get_window_extent()
        ppu = bbox.height / data_height if data_height > 0 else 0

        node_size_data = 18
        if tx_node:
            bounds = tx_node.get_world_bounds()
            if bounds:
                node_world_height = bounds[3] - bounds[1]
                node_size_pixels = node_world_height * ppu
            else:
                node_size_pixels = node_size_data * ppu
        else:
            node_size_pixels = node_size_data * ppu

        tx_tl_artists = []
        for child in ax.get_children():
            if hasattr(child, 'get_text') and hasattr(child, 'get_fontsize'):
                text = child.get_text()
                if text in ["Tx", "Tl"]:
                    tx_tl_artists.append({
                        'text': text,
                        'fontsize': child.get_fontsize(),
                    })

        font_size_data = 7

        results = {
            'name': name,
            'data_extent': (data_width, data_height),
            'ppu': ppu,
            'node_size_data': node_size_data,
            'node_size_pixels': node_size_pixels,
            'font_size_data': font_size_data,
            'text_artists': tx_tl_artists,
        }

        if tx_tl_artists:
            avg_fontsize = np.mean([t['fontsize'] for t in tx_tl_artists])
            text_pixels = avg_fontsize * (fig.dpi / 72.0)
            results['avg_text_points'] = avg_fontsize
            results['avg_text_pixels'] = text_pixels
            results['text_to_node_ratio'] = text_pixels / node_size_pixels if node_size_pixels > 0 else 0
            results['expected_ratio'] = font_size_data / node_size_data

        plt.close(fig)
        return results

    def test_simple_single_reporter(self, lib, cleanup):
        """Test text sizes in simple_single_reporter diagram."""
        network = build_simple_single_reporter(lib)
        if network is None:
            pytest.skip("Could not build simple_single_reporter network")

        results = self.measure_diagram_text(network, "simple_single_reporter")

        if 'avg_text_points' in results:
            deviation = abs(results['text_to_node_ratio'] - results['expected_ratio']) / results['expected_ratio'] * 100
            if deviation > 20:
                pytest.fail(f"Text/node ratio off by {deviation:.1f}% (expected ~{results['expected_ratio']:.3f}, got {results['text_to_node_ratio']:.3f})")

    def test_multi_networks(self, lib, cleanup):
        """Test text sizes across multiple network diagrams."""
        builders = [
            ("simple_single_reporter", build_simple_single_reporter),
            ("multi_cotx_aggregation", build_multi_cotx_aggregation),
            ("two_reporters_with_ern", build_two_reporters_with_ern),
        ]

        results = []
        for name, builder in builders:
            network = builder(lib)
            if network is None:
                continue
            result = self.measure_diagram_text(network, name)
            results.append(result)


        ratios = [r['text_to_node_ratio'] for r in results if 'text_to_node_ratio' in r]
        if ratios:
            variance = np.var(ratios)
            mean = np.mean(ratios)
            cv = np.sqrt(variance) / mean if mean > 0 else 0

            if cv > 0.10:
                pytest.fail(f"Text/node ratios vary too much across diagrams! CV={cv:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
