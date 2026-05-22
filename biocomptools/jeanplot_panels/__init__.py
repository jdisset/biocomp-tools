# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from biocomptools.jeanplot_panels.activations import ActivationsPanel
from biocomptools.jeanplot_panels.biocomp_figure_adapter import BiocompFigureAdapter
from biocomptools.jeanplot_panels.benchmark import (
    BenchmarkCircuitMergedLeftPanel,
    BenchmarkHeaderPanel,
    BenchmarkMetricsMergedRightPanel,
    BenchmarkMetricsPanel,
    BenchmarkSliceGridPanel,
)
from biocomptools.jeanplot_panels.blurb import BlurbPanel
from biocomptools.jeanplot_panels.circuit import CircuitPanel
from biocomptools.jeanplot_panels.data import (
    MVPDataHolder,
    NetworkPlotData,
    NetworkPredictedPlotData,
)
from biocomptools.jeanplot_panels.design import (
    DesignMetricsPanel,
    LatticeHeatmapPanel,
    NetworkDiagramFullWidthPanel,
)
from biocomptools.jeanplot_panels.empty import ConstantTextPanel, EmptyPanel
from biocomptools.jeanplot_panels.fluo import FluoDensitiesPanel
from biocomptools.jeanplot_panels.heatmap import (
    ClassSummaryHeatmapPanel,
    HorizontalHeatmapPanel,
)
from biocomptools.jeanplot_panels.latent_density import LatentProjectionHistogramPanel
from biocomptools.jeanplot_panels.mvp_network import MVPNetworkPanel
from biocomptools.jeanplot_panels.network_diagram import NetworkDiagramPanel
from biocomptools.jeanplot_panels.pipelines import (
    PAPER_PIPELINE_HELPERS,
    load_paper_dataset,
    network_plot_data,
    paper_per_network_pds,
)
from biocomptools.jeanplot_panels.quantile import QuantileCoveragePanel
from biocomptools.jeanplot_panels.row_composer import build_per_network_row
from biocomptools.jeanplot_panels.shapley import ShapleyDetailPanel
from biocomptools.jeanplot_panels.voxel import (
    BenchmarkDistributionPanel,
    SmoothVoxelPanel,
)


JEANPLOT_PANEL_TYPES = [
    BiocompFigureAdapter,
    CircuitPanel,
    NetworkDiagramPanel,
    BlurbPanel,
    MVPNetworkPanel,
    ActivationsPanel,
    EmptyPanel,
    ConstantTextPanel,
    FluoDensitiesPanel,
    HorizontalHeatmapPanel,
    ClassSummaryHeatmapPanel,
    ShapleyDetailPanel,
    DesignMetricsPanel,
    LatticeHeatmapPanel,
    NetworkDiagramFullWidthPanel,
    QuantileCoveragePanel,
    SmoothVoxelPanel,
    BenchmarkDistributionPanel,
    BenchmarkHeaderPanel,
    BenchmarkMetricsPanel,
    BenchmarkCircuitMergedLeftPanel,
    BenchmarkMetricsMergedRightPanel,
    BenchmarkSliceGridPanel,
    LatentProjectionHistogramPanel,
    NetworkPlotData,
    NetworkPredictedPlotData,
    MVPDataHolder,
]


def _bio_helpers() -> dict:
    """Lazily fetch the bio-domain helpers from ``datasetsummary``.

    Kept lazy because importing ``datasetsummary`` pulls biocomp early;
    callers that don't need these helpers shouldn't pay for them at
    package import.
    """
    from biocomptools.toollib.figuremakers.datasetsummary import (
        build_figure_metadata,
        build_prediction_pipeline,
        extract_model_metadata,
        extract_prediction_config,
        filter_compatible,
        maybe_build_mvp,
        predicted_stats,
        smart_title,
    )

    return {
        "build_prediction_pipeline": build_prediction_pipeline,
        "filter_compatible": filter_compatible,
        "predicted_stats": predicted_stats,
        "extract_model_metadata": extract_model_metadata,
        "extract_prediction_config": extract_prediction_config,
        "maybe_build_mvp": maybe_build_mvp,
        "smart_title": smart_title,
        "build_figure_metadata": build_figure_metadata,
    }


def get_jeanplot_panel_helpers() -> dict:
    """Return the helper map for dracon ``context=`` registration."""
    return {
        "build_per_network_row": build_per_network_row,
        **PAPER_PIPELINE_HELPERS,
        **_bio_helpers(),
    }


__all__ = [
    "ActivationsPanel",
    "BiocompFigureAdapter",
    "BenchmarkCircuitMergedLeftPanel",
    "BenchmarkDistributionPanel",
    "BenchmarkHeaderPanel",
    "BenchmarkMetricsMergedRightPanel",
    "BenchmarkMetricsPanel",
    "BenchmarkSliceGridPanel",
    "BlurbPanel",
    "CircuitPanel",
    "ClassSummaryHeatmapPanel",
    "ConstantTextPanel",
    "DesignMetricsPanel",
    "EmptyPanel",
    "FluoDensitiesPanel",
    "HorizontalHeatmapPanel",
    "JEANPLOT_PANEL_TYPES",
    "LatentProjectionHistogramPanel",
    "LatticeHeatmapPanel",
    "MVPDataHolder",
    "MVPNetworkPanel",
    "NetworkDiagramFullWidthPanel",
    "NetworkDiagramPanel",
    "NetworkPlotData",
    "NetworkPredictedPlotData",
    "PAPER_PIPELINE_HELPERS",
    "QuantileCoveragePanel",
    "ShapleyDetailPanel",
    "SmoothVoxelPanel",
    "build_per_network_row",
    "get_jeanplot_panel_helpers",
    "load_paper_dataset",
    "network_plot_data",
    "paper_per_network_pds",
]
