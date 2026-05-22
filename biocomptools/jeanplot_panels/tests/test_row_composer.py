# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for ``build_per_network_row``.

Verify the returned Container tree's shape for the common compositions
(2D vs 3D data, layout='row' vs 'stacked', mvp-only).
"""

import numpy as np

from biocomp.plotutils import PlotData

from biocomptools.jeanplot_panels import (
    BlurbPanel,
    CircuitPanel,
    MVPNetworkPanel,
    build_per_network_row,
)


def _make_plot_data(input_dim: int, n: int = 80, network=None) -> PlotData:
    return PlotData(
        xval=np.random.rand(n, input_dim).astype(np.float32),
        yval=np.random.rand(n, 1).astype(np.float32),
        input_names=[f"x{i}" for i in range(input_dim)],
        output_name="y",
        metadata={"built_network": network} if network is not None else {},
    )


class _FakeNetwork:
    """Stand-in for biocomp.Network sufficient for the row composer dispatch."""

    def __init__(self):
        self.metadata = {}
        self.compute_graph = None
        self.nb_inputs = 2


def test_row_with_2d_data_has_three_cells():
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["diagram", "circuit", "ground_truth"],
        plot_data=pd,
        network=net,
    )
    # 3 cells, no gaps between them (none of these panels declare gaps)
    assert row.layout.direction == "row"
    kinds = [type(c).__name__ for c in row.children]
    assert "NetworkDiagramPanel" in kinds
    assert "CircuitPanel" in kinds
    assert "SmoothPanel2D" in kinds


def test_row_with_3d_data_uses_smooth3d():
    from jeanplot.panels.smooth_3d import SmoothPanel3D

    net = _FakeNetwork()
    pd = _make_plot_data(3, network=net)
    row = build_per_network_row(
        panels=["ground_truth"],
        plot_data=pd,
        network=net,
    )
    cell = next(c for c in row.children if isinstance(c, SmoothPanel3D))
    assert isinstance(cell, SmoothPanel3D)
    # SmoothPanel3D wires up cube + slice grid as children in its
    # ``model_post_init`` - confirm that happened.
    assert len(cell.children) == 2


def test_stacked_layout_produces_rows_per_cell():
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    tree = build_per_network_row(
        panels=["diagram", "circuit", "ground_truth"],
        plot_data=pd,
        network=net,
        layout="stacked",
    )
    assert tree.layout.direction == "column"
    # 3 panel rows
    assert len(tree.children) == 3
    for row_container in tree.children:
        assert row_container.layout.direction == "row"
        assert len(row_container.children) == 1


def test_mvp_global_single_panel():
    # mvp_global doesn't strictly need a real MeasuredVsPredictedData since
    # the row composer only constructs the panel; it doesn't invoke draw().
    class _FakeMVP:
        measured = predicted = None
        rescaler = None
        grid_measured = grid_predicted = grid_weights = None
        noise_floor_measured = None

    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["mvp_global"],
        plot_data=pd,
        mvp_data=_FakeMVP(),
        network=net,
    )
    mvp_cells = [c for c in row.children if isinstance(c, MVPNetworkPanel)]
    assert len(mvp_cells) == 1


def test_unknown_panel_kinds_are_skipped():
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["definitely_not_a_panel", "circuit"],
        plot_data=pd,
        network=net,
    )
    cells = [c for c in row.children if isinstance(c, CircuitPanel)]
    assert len(cells) == 1


def test_blurb_panel_carries_text():
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["blurb"],
        plot_data=pd,
        blurb_text="**model** info",
        blurb_title="Notes",
        network=net,
    )
    blurb = next(c for c in row.children if isinstance(c, BlurbPanel))
    assert blurb.text == "**model** info"
    assert blurb.title == "Notes"


def test_none_kwargs_dropped_for_3d_data():
    # Regression: YAML knobs default to None to mean "absent". Splatting
    # {"zslices": None, ...} into SmoothPanel3D(zslices: list[float]) used to
    # raise pydantic "Input should be a valid list". Composer must drop None
    # so Pydantic defaults apply.
    from jeanplot.panels.smooth_3d import SmoothPanel3D

    net = _FakeNetwork()
    pd = _make_plot_data(3, network=net)
    row = build_per_network_row(
        panels=["ground_truth"],
        plot_data=pd,
        network=net,
        slice_grid_kwargs={
            "slice_grid": [3, 3],
            "slice_zrange": [0.0, 0.5],
            "slice_zvalues": None,
            "zslices": None,
            "cube_frac_w": 0.45,
        },
    )
    cell = next(c for c in row.children if isinstance(c, SmoothPanel3D))
    # Pydantic default kicked in instead of None.
    assert cell.zslices == [0.05, 0.25, 0.4, 0.55]


def test_none_kwargs_dropped_for_slices_panel():
    # Same class of bug for the slices-only panel: `sg_kw.get("zslices", default)`
    # returns None when the key is present-but-None. Composer must drop None
    # so the fallback default applies.
    from jeanplot.panels.smooth_2d import SmoothPanel2D

    net = _FakeNetwork()
    pd = _make_plot_data(3, network=net)
    row = build_per_network_row(
        panels=["ground_truth_slices"],
        plot_data=pd,
        network=net,
        slice_grid_kwargs={"slice_grid": None, "zslices": None},
    )
    # Should build a Container of SmoothPanel2D cells; no NoneType errors.
    leaves = []

    def _walk(c):
        if isinstance(c, SmoothPanel2D):
            leaves.append(c)
        for ch in getattr(c, "children", []) or []:
            _walk(ch)

    _walk(row)
    assert len(leaves) == 9  # default slice_grid=(3,3)


def test_none_kind_widths_dropped():
    # kind_widths={"diagram": None} used to override the per-kind default with
    # None, then float(None) blew up. None means "use default" here too.
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["diagram", "ground_truth"],
        plot_data=pd,
        network=net,
        kind_widths={"diagram": None},
    )
    # Diagram cell still present with the default width.
    from biocomptools.jeanplot_panels import NetworkDiagramPanel

    diag = next(c for c in row.children if isinstance(c, NetworkDiagramPanel))
    assert diag.min_dimensions.width == 5.0  # _DEFAULT_KIND_WIDTHS["diagram"]


def test_prediction_panel_dropped_when_missing():
    net = _FakeNetwork()
    pd = _make_plot_data(2, network=net)
    row = build_per_network_row(
        panels=["ground_truth", "prediction"],
        plot_data=pd,
        predicted_data=None,
        network=net,
    )
    # prediction panel silently dropped, ground_truth kept
    from jeanplot.panels.smooth_2d import SmoothPanel2D

    data_cells = [c for c in row.children if isinstance(c, SmoothPanel2D)]
    assert len(data_cells) == 1
