# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Multi-figure merge spec. Used by `biocomp-plot --merge-spec` to combine
several figure outputs into one grid or one multi-page PDF."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict


class MergeSpec(BaseModel):
    mode: Literal["grid", "pages"] = "grid"
    rows: int = 1
    cols: int = 1
    row_heights: list[float] | None = None
    col_widths: list[float] | None = None
    output_dir: str = "./"
    output_file: str = "merged.png"
    title: str | None = None
    hspace: int = 10
    vspace: int = 10
    bg_color: str = "white"
    delete_intermediates: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.output_file
