# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from biocomptools.toollib.figuremakers.geneticcircuit import _load_genetic_schematic_theme
from jeanplot.core import BoxStyle, LayoutConstraints, Offset, Shadow, Size
from jeanplot.core.svg import LineEndFlat


_GENE_TYPES = [Size, BoxStyle, LayoutConstraints, Offset, Shadow, LineEndFlat]


def _fresh_ern(part_name: str):
    from jeanplot import jstyle
    from jeanplot.core.svg import SVGElement
    from jeanplot.gene.elements import ERN
    import biocomptools.toollib.figuremakers.geneticcircuit as gc

    gc._GENETIC_SCHEMATIC_THEME_CACHE = None
    jstyle.clear()
    jstyle.update(_load_genetic_schematic_theme(_GENE_TYPES))
    ern = ERN(part_name=part_name, id=f"t_{part_name}")
    jstyle.apply(ern)
    return next(c for c in ern.children if isinstance(c, SVGElement)), ern


def test_default_ern_color_remap_survives_biocomp_override():
    # the per-ERN color_remap from jeanplot's default theme must survive the stack
    svg, _ = _fresh_ern("Csy4")
    assert svg.color_remap
    blue_keys = [k for k in svg.color_remap if k.startswith("#0000ff")]
    assert blue_keys
    assert not svg.color_remap[blue_keys[0]].lower().startswith("#0000ff")


def test_biocomp_ern_layout_overrides_default():
    _, ern = _fresh_ern("Csy4")
    assert ern.layout.direction == "row"


def test_fluo_marker_color_remap_from_biocomp_theme():
    from jeanplot import jstyle
    from jeanplot.core.svg import SVGElement
    from jeanplot.gene.elements import FluoMarker
    import biocomptools.toollib.figuremakers.geneticcircuit as gc

    gc._GENETIC_SCHEMATIC_THEME_CACHE = None
    jstyle.clear()
    jstyle.update(_load_genetic_schematic_theme(_GENE_TYPES))
    fm = FluoMarker(part_name="mNeonGreen", id="t_mng")
    jstyle.apply(fm)
    svg = next(c for c in fm.children if isinstance(c, SVGElement))
    assert svg.color_remap
    assert any(k.startswith("#00ff00") for k in svg.color_remap)
