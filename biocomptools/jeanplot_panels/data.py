# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from jeanplot.data.plot_data import PlotData as JeanplotPlotData
from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData


def _biocomp_to_jeanplot(pd: Any) -> JeanplotPlotData:
    """Convert a biocomp ``PlotData`` (or ``LazyPlotData``) to ``jeanplot.PlotData``.

    Biocomp's ``PlotData`` has ``.x``, ``.y``, ``.input_names``, ``.output_name``,
    ``.metadata``; jeanplot's has ``xval``, ``yval``, ``input_names``,
    ``output_name``, ``metadata``. Field names match modulo ``x`` -> ``xval``.
    """
    x = np.asarray(pd.x, dtype=np.float32)
    y = np.asarray(pd.y, dtype=np.float32)
    output_name = pd.output_name
    if isinstance(output_name, list) and len(output_name) == 1:
        output_name = output_name[0]
    return JeanplotPlotData(
        xval=x,
        yval=y,
        input_names=list(pd.input_names),
        output_name=output_name,
        column_names=getattr(pd, "column_names", None),
        metadata=dict(pd.metadata or {}),
    )


class NetworkPlotData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any

    @property
    def jeanplot(self) -> JeanplotPlotData:
        return _biocomp_to_jeanplot(self.source)

    def to_jeanplot(self) -> JeanplotPlotData:
        return self.jeanplot


class NetworkPredictedPlotData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any

    @property
    def jeanplot(self) -> JeanplotPlotData:
        return _biocomp_to_jeanplot(self.source)

    def to_jeanplot(self) -> JeanplotPlotData:
        return self.jeanplot


# Alias for registration symmetry with the other data holders.
MVPDataHolder = MeasuredVsPredictedData


__all__ = [
    "NetworkPlotData",
    "NetworkPredictedPlotData",
    "MVPDataHolder",
]
