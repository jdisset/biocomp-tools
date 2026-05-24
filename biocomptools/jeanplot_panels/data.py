# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from jeanplot.data.plot_data import PlotData as JeanplotPlotData
from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData


def _biocomp_to_jeanplot(pd: Any, rescaler: Any = None) -> JeanplotPlotData:
    """Convert a biocomp ``PlotData`` to ``jeanplot.PlotData``.

    biocomp data lives in raw experimental units; jeanplot panels expect
    latent ([0, 1]) space and use ``rescaler`` to project axis ticks back to
    raw at render time. Pass a rescaler here to project at the boundary.
    """
    x = np.asarray(pd.x, dtype=np.float32)
    y = np.asarray(pd.y, dtype=np.float32)
    if rescaler is not None:
        x = np.asarray(rescaler.fwd(x), dtype=np.float32)
        y = np.asarray(rescaler.fwd(y), dtype=np.float32)
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
    """Wraps a biocomp PlotData; exposes `.jeanplot` for panels."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any
    rescaler: Any = None

    @property
    def jeanplot(self) -> JeanplotPlotData:
        return _biocomp_to_jeanplot(self.source, rescaler=self.rescaler)

    def to_jeanplot(self) -> JeanplotPlotData:
        return self.jeanplot


MVPDataHolder = MeasuredVsPredictedData


__all__ = [
    "NetworkPlotData",
    "MVPDataHolder",
]
