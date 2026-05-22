# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Paper-figure data pipeline helpers, exposed as dracon `!fn` templates.

These wrap existing biocomp helpers (``datasetsummary``, ``DBSource``,
``DataRescaler``) into named stages that paper-jobs YAML files can compose
declaratively. The new file is purely interface — implementations remain in
``datasetsummary``/``datasources``.
"""

from typing import Any

from dracon import register_template

from jeanplot.data.plot_data import PlotData as JeanplotPlotData
from biocomptools.jeanplot_panels.data import _biocomp_to_jeanplot


def load_paper_dataset(dataset_file: str) -> list[Any]:
    """Read a NetworkSet/CleanupFilter YAML, return ``filter_compatible`` output."""
    from biocomptools.toollib.datasources import DBSource
    from biocomptools.toollib.figuremakers.datasetsummary import filter_compatible
    import dracon as dr

    content = dr.load(dataset_file, enable_interpolation=True)
    src = DBSource(content=content)
    return filter_compatible(src.get_data())


def network_plot_data(D: list[Any], index: int = 0, rescaler: Any = None) -> JeanplotPlotData:
    """Build a jeanplot ``PlotData`` for one network, optionally rescaling x/y."""
    pd = D[index]
    jp = _biocomp_to_jeanplot(pd)
    if rescaler is not None:
        jp = jp.model_copy(
            update={
                "xval": rescaler.fwd(jp.xval),
                "yval": rescaler.fwd(jp.yval),
            }
        )
    return jp


def paper_per_network_pds(
    D: list[Any],
    rescaler: Any = None,
) -> list[JeanplotPlotData]:
    """One jeanplot ``PlotData`` per network, all with the same rescaler applied."""
    return [network_plot_data(D, index=i, rescaler=rescaler) for i in range(len(D))]


def opt_list(item: Any) -> list:
    """Wrap a single item in a one-element list; ``None`` becomes ``[]``.

    Lets paper-panel templates conditionally include overlay specs without
    multi-line ``${...}`` blocks or ``!if`` keys nested in a sequence (which
    YAML cannot parse).
    """
    return [item] if item is not None else []


register_template(load_paper_dataset)
register_template(network_plot_data)
register_template(paper_per_network_pds)
register_template(opt_list)


PAPER_PIPELINE_HELPERS: dict[str, Any] = {
    "load_paper_dataset": load_paper_dataset,
    "network_plot_data": network_plot_data,
    "paper_per_network_pds": paper_per_network_pds,
    "opt_list": opt_list,
}


__all__ = [
    "load_paper_dataset",
    "network_plot_data",
    "opt_list",
    "paper_per_network_pds",
    "PAPER_PIPELINE_HELPERS",
]
