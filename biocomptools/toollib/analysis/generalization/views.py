# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

from typing import Annotated

import matplotlib.pyplot as plt
from dracon import Arg
from matplotlib.colors import to_rgba
from pydantic import BaseModel, Field

FINE_CLASSES: list[str] = ["Lbl", "Lbr", "Lc", "NSM", "M", "N", "S"]


def parse_condition(cond: str, known_classes: list[str] | None = None) -> list[str]:
    classes = sorted(known_classes or FINE_CLASSES, key=len, reverse=True)
    remaining = cond
    result: list[str] = []
    while remaining:
        for cls in classes:
            if remaining.startswith(cls):
                result.append(cls)
                remaining = remaining[len(cls):]
                break
        else:
            remaining = remaining[1:]
    return result


class GenViewConfig(BaseModel):
    enabled: Annotated[bool, Arg(help="Generate this view")] = True
    players: Annotated[list[str], Arg(help="Players for this view")] = Field(default_factory=list)
    labels: Annotated[dict[str, str], Arg(help="Display labels per player")] = Field(default_factory=dict)
    colors: Annotated[dict[str, str] | None, Arg(help="Manual color overrides")] = None
    colormap: Annotated[str, Arg(help="Colormap for auto colors")] = "bc_multi"
    color_step: Annotated[float, Arg(help="Colormap sampling step")] = 1.0 / 7.0
    topo_mapping: Annotated[dict[str, str], Arg(help="fine_topo_class -> player mapping")] = Field(
        default_factory=dict
    )
    atom_members: Annotated[dict[str, list[str]], Arg(help="Pseudo-player -> fine classes")] = Field(
        default_factory=dict
    )

    def get_colors(self) -> dict[str, tuple[float, ...]]:
        if self.colors:
            return {t: to_rgba(self.colors.get(t, "gray")) for t in self.players}
        cmap = plt.get_cmap(self.colormap)
        return {t: cmap(i * self.color_step) for i, t in enumerate(self.players)}

    def expand_coalition(self, coalition: frozenset[str]) -> frozenset[str]:
        fine: set[str] = set()
        for player in coalition:
            if player in self.atom_members:
                fine.update(self.atom_members[player])
            elif self.topo_mapping:
                fine.update(f for f, m in self.topo_mapping.items() if m == player)
            else:
                fine.add(player)
        return frozenset(fine)

    def fine_keys(self) -> list[str]:
        return sorted(list(self.topo_mapping.keys()) or self.players, key=len, reverse=True)


ViewConfig = GenViewConfig
