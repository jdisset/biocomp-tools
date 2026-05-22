# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
## {{{                          --     imports     --

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, ConfigDict, Field
from biocomptools.toollib.common import maybetqdm, make_context_from_types
import numpy as np

import dracon as dr
from dracon.utils import ser_debug
from dracon.diagnostics import DraconError, handle_dracon_error

import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import List, Annotated, Literal, Optional
from dracon.commandline import Arg, dracon_program
from dracon.deferred import DeferredNode

from biocomp.utils import PartialFunction
import biocomp
from biocomp.datautils import DataRescaler
from biocomp.plotutils import (
    FigureSpec,
    FigAx,
    SimpleLayout,
    GridLayout,
    MultiRowGridLayout,
    MergeSpec,
    diagonal_xy,
    diagonal_xy_raw,
    plot_diagonal_paths,
    slice_panel_args,
    plot_slice_overlay,
    plot_slice_chords,
    plot_addition_vs_removal_overlay,
    IDENTITY_RESCALER,
)
from biocomptools.toollib.datasources import DataSource, DBSource

from biocomptools.toollib.networkprediction import NetworkPrediction, PredictionSamplingConfig
from biocomptools.toollib.typical_experimental_distribution import sample_latent

from biocomptools.toollib.common import config
from biocomptools.toollib.plot import PlotConfig, PlotTask, BiocompPlotFigure
from jeanplot.panels import Figure
from biocomptools.toollib.overlays import OVERLAY_TYPES
from biocomptools.toollib.figuremakers.uorfmatrixfigure import (
    UORFMatrixFigure,
    bundle_uorf_data,
    get_uorf_values,
    extract_uorf_info,
)
from biocomptools.toollib.figuremakers.innernodes import InnerNodesFigure, InnerNodesFigureSpec
from biocomptools.toollib.figuremakers.benchmarkutils import BenchmarkData, BenchmarkItem
from biocomptools.toollib.figuremakers.datasetsummary import (
    expand_panel_atomics,
    compose_rows,
    compose_atomics,
    build_rows,
    panel_plot_method,
    panel_plot_config,
    layout_dimensions,
    extract_plot_data_metadata,
    extract_model_metadata,
    extract_prediction_config,
    build_figure_metadata,
    predicted_stats,
    build_prediction_pipeline,
    maybe_build_mvp,
    filter_compatible,
    format_z_label,
    smart_title,
    training_set_count,
    trained_on_status,
)
from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData
from biocomptools.toollib.analysis.generalization.shapley_figure import (
    ShapleyDetailFigure,
    ShapleyDetailConfig,
)
from biocomptools.toollib.analysis.generalization.heatmap_figure import (
    HorizontalHeatmapFigure,
    ClassSummaryHeatmapFigure,
    HeatmapConfig,
)
from biocomptools.toollib.analysis.generalization.views import GenViewConfig
from biocomptools.toollib.analysis.generalization.pivot_build import load_metrics_csv
from biocomptools.modelmodel import BiocompModel, NetworkModel
from biocomptools.toollib.modelselector import ModelSelector
from biocomptools.toollib.networkselector import (
    NetworkSelector,
    Regex,
    iRegex,
    NetworkSet,
    NetworkDataPair,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    NetworkFilter,
    CleanupFilter,
    CustomFilter,
    UorfFilter,
)

import numpy as numpy
from dracon.utils import dict_like


from numpy import (
    ndarray,
    linspace,
    array,
    arange,
    meshgrid,
    zeros,
    ones,
    full,
    empty,
    empty_like,
    full_like,
    zeros_like,
    ones_like,
    eye,
    diag,
    diagflat,
    triu,
    tril,
    vander,
    histogram,
    histogram2d,
    digitize,
)


import warnings

warnings.filterwarnings("ignore", message="os.fork()", module="subprocess")

logger = get_logger(__name__)


class FigureResult(BaseModel):
    path: str
    figure_spec: FigureSpec
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return self.path


class PlotResult(BaseModel):
    figures: List[FigureResult] = Field(default_factory=list)
    merged_path: Optional[str] = None

    @property
    def paths(self) -> List[str]:
        return [f.path for f in self.figures]

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.figures)

    def __getitem__(self, idx):
        return self.figures[idx]


##────────────────────────────────────────────────────────────────────────────}}}


def debug_figures(figures):
    for fig in figures:
        ser_debug(fig, 'dill')
        ser_debug(fig, 'deepcopy')
        logger.debug(dr.utils.node_repr(fig, enable_colors=True, show_biggest_context=5))


def get_pretty_axis_label(i: int, d: DataSource) -> str:
    if "pretty_inputs" in d.metadata and len(d.metadata["pretty_inputs"]) > i:
        return f'$\\mathbf{{X_{i + 1} ({d.input_names[i]}}})$\n{d.metadata["pretty_inputs"][i]}'
    return f'$\\mathbf{{X_{i + 1}}}$ ({d.input_names[i]})'


def urlencoded(s: str) -> str:
    import urllib.parse

    return urllib.parse.quote(s, safe='')


def construct_figure(
    figure_node,
    output_path_override: str | None = None,
    text_mode: bool = False,
    stdout_txt_plot: bool = True,
):
    try:
        figure = figure_node.construct(deferred_paths=['/plot_tasks.*'])
        if dict_like(figure):
            figure = Figure(**figure)  # type: ignore
        assert isinstance(figure, Figure), f"Expected Figure, got {type(figure)}"
        if output_path_override:
            figure.figure_spec.output_dir = str(Path(output_path_override).parent)
            figure.figure_spec.output_file = Path(output_path_override).name
        figure.text_mode = text_mode
        figure.stdout_txt_plot = stdout_txt_plot
    except DraconError:
        raise
    except Exception as e:
        logger.error(f"Error constructing figure: {e}")
        logger.exception(e)
        raise
    return figure


def warm_caches_for_figure(figure):
    # NOTE: every `tc.construct(...)` mutates the DeferredNode's internal
    # composition state, so the *subsequent* construct (done by
    # `_prepare_mpl` with the real FIG) can return an empty dict if the
    # task uses a `<<: !include` merge (as `tasks/mvp_panel.yaml` does).
    # Always work on a fresh copy here.
    try:
        from biocomp.plotting.plotting_smooth import knn_grid
        from biocomp.plotutils import LazyPlotData
        for tc in figure.plot_tasks:
            try:
                pt = tc.copy(reroot=True).construct(context={"FIG": _DUMMY_FIG_AX})
            except Exception:
                continue
            if not hasattr(pt, "plot_method") or pt.plot_method is None:
                continue
            try:
                func_name = pt.plot_method.get_name()
            except Exception:
                continue
            if not (func_name.endswith("smooth") or func_name.endswith("smooth_2d")):
                continue
            kw = dict(pt.plot_method.kwargs)
            plot_data = kw.get("plot_data") or (pt.plot_method.args[0] if pt.plot_method.args else None)
            if plot_data is None or not isinstance(plot_data, LazyPlotData):
                continue
            rescaler = pt.plot_config.rescaler if pt.plot_config else None
            if rescaler is None:
                continue
            try:
                x_raw, y_raw = plot_data.x, plot_data.y
                if x_raw is None or x_raw.ndim != 2 or x_raw.shape[1] != 2:
                    continue
                xlims = kw.get("xlims") or [0.0, 0.65]
                ylims = kw.get("ylims") or xlims
                knn_grid(rescaler.fwd(x_raw), rescaler.fwd(y_raw), xlims, ylims, grid_resolution=250)
            except Exception:
                continue
    except Exception:
        pass


class _DummyFigAx:
    flat_ax = [None] * 100
    axes = [[None] * 10 for _ in range(10)]
    figure = None


_DUMMY_FIG_AX = _DummyFigAx()


def _result_path(f) -> str:
    return str(
        getattr(f.figure_spec, '_output_path_override', None)
        or f.figure_spec.output_path
        or "(no file output)"
    )


def run_figure(f, **kw) -> FigureResult:
    t0 = time.time()
    try:
        f.run(**kw)
        if not f.is_txt_output:
            plt.close('all')
    except Exception as e:
        logger.error(f"Error running figure: {e}")
        logger.exception(e)
        raise
    logger.debug(f"Figure {_result_path(f)} completed in {time.time() - t0:.2f}s")
    return FigureResult(
        path=_result_path(f),
        figure_spec=f.figure_spec,
        metadata=getattr(f.figure_spec, 'metadata', {}),
    )


def _worker_entry(task) -> FigureResult:
    node, output_path, text_mode, stdout_txt, overwrite = task
    fig = construct_figure(
        node,
        output_path_override=output_path,
        text_mode=text_mode,
        stdout_txt_plot=stdout_txt,
    )
    return run_figure(fig, overwrite=overwrite)


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS",
    "BIOCOMP_KNN_WORKERS",
)


def _lock_thread_env() -> None:
    """Pin numerics libs to 1 thread before BLAS imports - each loky worker = 1 core."""
    import os as _os
    for k in _THREAD_ENV_KEYS:
        _os.environ.setdefault(k, "1")
    _os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )


def _loky_pool(nworkers: int):
    from loky import get_reusable_executor
    return get_reusable_executor(
        max_workers=nworkers, initializer=_lock_thread_env, reuse=True,
    )


DEFAULT_TYPES = [
    Figure,
    BiocompPlotFigure,
    FigureResult,
    PlotResult,
    PlotConfig,
    PlotTask,
    *OVERLAY_TYPES,
    DataSource,
    DBSource,
    FigureSpec,
    FigAx,
    SimpleLayout,
    GridLayout,
    MultiRowGridLayout,
    MergeSpec,
    Regex,
    iRegex,
    ModelSelector,
    BiocompModel,
    NetworkModel,
    NetworkPrediction,
    PredictionSamplingConfig,
    NetworkSelector,
    NetworkSet,
    NetworkDataPair,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    PartialFunction,
    DataRescaler,
    UORFMatrixFigure,
    ShapleyDetailFigure,
    ShapleyDetailConfig,
    HorizontalHeatmapFigure,
    ClassSummaryHeatmapFigure,
    HeatmapConfig,
    GenViewConfig,
    load_metrics_csv,
    InnerNodesFigure,
    InnerNodesFigureSpec,
    BenchmarkData,
    BenchmarkItem,
    MeasuredVsPredictedData,
    bundle_uorf_data,
    get_uorf_values,
    extract_uorf_info,
    NetworkFilter,
    CleanupFilter,
    CustomFilter,
    UorfFilter,
    ndarray,
    linspace,
    array,
    arange,
    meshgrid,
    zeros,
    ones,
    full,
    empty,
    empty_like,
    full_like,
    zeros_like,
    ones_like,
    eye,
    diag,
    diagflat,
    triu,
    tril,
    vander,
    histogram,
    histogram2d,
    digitize,
    biocomp
]


_HELPER_FUNCS = {
    'get_pretty_axis_label': get_pretty_axis_label,
    'urlencoded': urlencoded,
    'sample_latent': sample_latent,
    'expand_panel_atomics': expand_panel_atomics,
    'compose_rows': compose_rows,
    'compose_atomics': compose_atomics,
    'build_rows': build_rows,
    'panel_plot_method': panel_plot_method,
    'panel_plot_config': panel_plot_config,
    'layout_dimensions': layout_dimensions,
    'extract_plot_data_metadata': extract_plot_data_metadata,
    'extract_model_metadata': extract_model_metadata,
    'extract_prediction_config': extract_prediction_config,
    'build_figure_metadata': build_figure_metadata,
    'predicted_stats': predicted_stats,
    'build_prediction_pipeline': build_prediction_pipeline,
    'maybe_build_mvp': maybe_build_mvp,
    'filter_compatible': filter_compatible,
    'format_z_label': format_z_label,
    'smart_title': smart_title,
    'training_set_count': training_set_count,
    'trained_on_status': trained_on_status,
    'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
    'get_cmap': plt.get_cmap,
    'diagonal_xy': diagonal_xy,
    'diagonal_xy_raw': diagonal_xy_raw,
    'plot_diagonal_paths': plot_diagonal_paths,
    'slice_panel_args': slice_panel_args,
    'plot_slice_overlay': plot_slice_overlay,
    'plot_slice_chords': plot_slice_chords,
    'plot_addition_vs_removal_overlay': plot_addition_vs_removal_overlay,
    'IDENTITY_RESCALER': IDENTITY_RESCALER,
}

DEFAULT_CONTEXT = {**make_context_from_types(DEFAULT_TYPES), **_HELPER_FUNCS}


@dracon_program(
    name='biocomp-plot',
    description='Generate plots from YAML configuration.',
    deferred_paths=['/figures.*'],
    context_types=DEFAULT_TYPES,
    context=_HELPER_FUNCS,
)
class PlotJob(BaseModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworkers: Annotated[int, Arg(help='Number of workers (processes) to use')] = 8
    skip_existing: Annotated[bool, Arg(help='Overwrite existing figures')] = False
    parallel_mode: Annotated[
        Literal['process', 'thread', 'ray', 'none'],
        Arg(help="'process' (loky, default) | 'thread' | 'none'. 'ray' is a legacy alias for 'process'."),
    ] = 'process'
    pred: Annotated[
        Optional[NetworkPrediction],
        Arg(help='NetworkPrediction handle for shard-by-network parallel mode'),
    ] = None

    clear_figure_context_keys: Annotated[
        List[str], Arg(help='Clear these keys from the figure context')
    ] = []

    max_batch_size: Annotated[int, Arg(help='Maximum batch size per dispatch')] = 32

    merge_spec: Annotated[MergeSpec | None, Arg(help='Merge all figures into a single output')] = (
        None
    )

    use_txt_plotting: Annotated[
        bool,
        Arg(help='Use ASCII text plotting instead of image plots'),
    ] = False

    no_stdout_txt_plot: Annotated[
        bool,
        Arg(help='Disable stdout output when using text plotting'),
    ] = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_temp_paths_for_merge(self, n: int) -> list[str | None]:
        assert self.merge_spec is not None
        out_dir = Path(self.merge_spec.output_path).parent
        ext = Path(self.merge_spec.output_file).suffix or '.png'
        return [str(out_dir / f"_merge_tmp_{i:04d}{ext}") for i in range(n)]

    def run(self):
        overwrite = not self.skip_existing
        total_figures = len(self.figures)
        logger.debug(f"Going to create {total_figures} figures")

        if total_figures == 0:
            return PlotResult(figures=[], merged_path=None)

        for fig in self.figures:
            for k in self.clear_figure_context_keys:
                if k in fig.context:
                    del fig.context[k]

        t0 = time.time()

        if self.merge_spec:
            order = list(range(total_figures))
            Path(self.merge_spec.output_path).parent.mkdir(parents=True, exist_ok=True)
            temp_paths: list[str | None] = self._get_temp_paths_for_merge(total_figures)
        else:
            order = list(np.random.permutation(total_figures))
            temp_paths = [None] * total_figures

        txt_mode = self.use_txt_plotting
        stdout_txt = not self.no_stdout_txt_plot

        is_parallel = (
            self.nworkers > 1
            and self.parallel_mode != 'none'
            and total_figures > 1
        )

        if is_parallel:
            out_paths = self._run_parallel(order, temp_paths, txt_mode, stdout_txt, overwrite)
        else:
            out_paths = self._run_sequential(order, temp_paths, txt_mode, stdout_txt, overwrite)

        elapsed = time.time() - t0
        written = [r for r in out_paths if r is not None and getattr(r, "path", None)]
        if written:
            paths_block = "\n  ".join(str(r.path) for r in written)
            logger.info(
                f"{total_figures} figures completed in {elapsed:.2f}s:\n  {paths_block}"
            )
        else:
            logger.info(f"{total_figures} figures completed in {elapsed:.2f}s (no output paths)")

        merged_path = None
        if self.merge_spec and out_paths:
            self._merge_figures([r.path for r in out_paths])
            merged_path = str(self.merge_spec.output_path)

        return PlotResult(figures=out_paths, merged_path=merged_path)

    def _make_tasks(self, batch_indices, temp_paths, txt_mode, stdout_txt, overwrite):
        return [
            (self.figures[i].copy(reroot=True), temp_paths[i], txt_mode, stdout_txt, overwrite)
            for i in batch_indices
        ]

    def _run_sequential(self, order, temp_paths, txt_mode, stdout_txt, overwrite):
        from concurrent.futures import ThreadPoolExecutor

        fig_copies = [
            self.figures[i].copy(reroot=True)
            for i in maybetqdm(order, min_len=20, desc='Copying figure tasks')
        ]

        def construct_and_warm(node, path):
            fig = construct_figure(node, output_path_override=path,
                                   text_mode=txt_mode, stdout_txt_plot=stdout_txt)
            warm_caches_for_figure(fig)
            return fig

        out_paths: list = []
        with ThreadPoolExecutor(max_workers=1) as exec_:
            fut = exec_.submit(construct_and_warm, fig_copies[0], temp_paths[order[0]])
            for j in maybetqdm(range(len(order)), min_len=2, desc='Running figures'):
                fig = fut.result()
                if j + 1 < len(order):
                    fut = exec_.submit(construct_and_warm, fig_copies[j + 1], temp_paths[order[j + 1]])
                out_paths.append(run_figure(fig, overwrite=overwrite))
        return out_paths

    def _run_parallel(self, order, temp_paths, txt_mode, stdout_txt, overwrite):
        ex = None if self.parallel_mode == 'thread' else _loky_pool(self.nworkers)
        warm_futs = [ex.submit(int, 0) for _ in range(self.nworkers)] if ex else []

        if self.pred is not None:
            if self.pred._yhats is None:
                self.pred.compute_all_network_predictions()
            nm = self.pred.network_model
            nm._batch_apply = nm._batch_apply_cpu = nm._batch_apply_gpu = None

        for f in warm_futs:
            f.result()

        out_paths: list = []
        for start in range(0, len(order), self.max_batch_size):
            batch = order[start:start + self.max_batch_size]
            tasks = self._make_tasks(batch, temp_paths, txt_mode, stdout_txt, overwrite)
            out_paths.extend(self._dispatch(tasks, ex))
        return out_paths

    def _dispatch(self, tasks, ex):
        if ex is None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.nworkers) as tex:
                return list(tex.map(_worker_entry, tasks))
        return list(ex.map(_worker_entry, tasks))

    def _merge_figures(self, paths: list[str]):
        spec = self.merge_spec
        if not paths:
            return

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)

        if spec.mode == "pages":
            self._merge_as_pdf_pages(paths)
        else:
            all_pdfs = all(p.lower().endswith('.pdf') for p in paths)
            output_is_pdf = str(spec.output_path).lower().endswith('.pdf')
            if all_pdfs and output_is_pdf:
                self._merge_pdfs_lossless(paths)
            else:
                self._merge_as_images(paths)

        if spec.delete_intermediates:
            for p in paths:
                try:
                    Path(p).unlink()
                except OSError:
                    pass

        logger.info(f"Merged {len(paths)} figures to {spec.output_path}")

    def _merge_pdfs_lossless(self, paths: list[str]):
        from pypdf import PdfReader, PdfWriter, PageObject, Transformation

        spec = self.merge_spec
        rows, cols = spec.rows, spec.cols
        n = len(paths)

        readers = [PdfReader(p) for p in paths]
        pages = [r.pages[0] for r in readers]

        widths = [float(p.mediabox.width) for p in pages]
        heights = [float(p.mediabox.height) for p in pages]

        col_w = [w * max(widths) for w in spec.col_widths] if spec.col_widths else [max(widths)] * cols
        row_h = [h * max(heights) for h in spec.row_heights] if spec.row_heights else [max(heights)] * rows

        total_w = sum(col_w) + spec.hspace * (cols - 1)
        total_h = sum(row_h) + spec.vspace * (rows - 1)

        merged_page = PageObject.create_blank_page(width=total_w, height=total_h)

        y_offset = total_h  # PDF origin is bottom-left
        idx = 0
        for r in range(rows):
            y_offset -= row_h[r]
            x_offset = 0
            for c in range(cols):
                if idx < n:
                    page = pages[idx]
                    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
                    scale = min(col_w[c] / pw, row_h[r] / ph)
                    dx = x_offset + (col_w[c] - pw * scale) / 2
                    dy = y_offset + (row_h[r] - ph * scale) / 2
                    merged_page.merge_transformed_page(
                        page, Transformation().scale(scale).translate(dx, dy)
                    )
                    idx += 1
                x_offset += col_w[c] + spec.hspace
            y_offset -= spec.vspace

        writer = PdfWriter()
        writer.add_page(merged_page)
        with open(spec.output_path, 'wb') as f:
            writer.write(f)

    def _merge_as_pdf_pages(self, paths: list[str]):
        from pypdf import PdfReader, PdfWriter
        from PIL import Image
        import io

        spec = self.merge_spec
        writer = PdfWriter()

        for p in paths:
            if p.lower().endswith('.pdf'):
                for page in PdfReader(p).pages:
                    writer.add_page(page)
            else:
                img = Image.open(p)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='PDF', resolution=300)
                buf.seek(0)
                writer.add_page(PdfReader(buf).pages[0])
                img.close()

        with open(spec.output_path, 'wb') as f:
            writer.write(f)

    def _merge_as_images(self, paths: list[str]):
        from PIL import Image
        import subprocess
        import tempfile

        spec = self.merge_spec
        rows, cols = spec.rows, spec.cols
        n = len(paths)

        images = []
        temp_files = []
        try:
            for p in paths:
                if p.lower().endswith('.pdf'):
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_files.append(tmp.name)
                    tmp.close()
                    subprocess.run(
                        ['pdftoppm', '-png', '-singlefile', '-r', '300', p, tmp.name[:-4]],
                        check=True,
                    )
                    images.append(Image.open(tmp.name))
                else:
                    images.append(Image.open(p))

            if spec.row_heights:
                row_h = [int(h * sum(im.height for im in images[:rows])) for h in spec.row_heights]
            else:
                row_h = [images[i].height if i < n else images[0].height for i in range(rows)]

            if spec.col_widths:
                col_w = [int(w * max(im.width for im in images)) for w in spec.col_widths]
            else:
                col_w = [max(im.width for im in images)] * cols

            total_w = sum(col_w) + spec.hspace * (cols - 1)
            total_h = sum(row_h) + spec.vspace * (rows - 1)

            merged = Image.new('RGB', (total_w, total_h), spec.bg_color)

            y_offset = 0
            idx = 0
            for r in range(rows):
                x_offset = 0
                for c in range(cols):
                    if idx < n:
                        im = images[idx].resize((col_w[c], row_h[r]), Image.LANCZOS)
                        merged.paste(im, (x_offset, y_offset))
                        idx += 1
                    x_offset += col_w[c] + spec.hspace
                y_offset += row_h[r] + spec.vspace

            if str(spec.output_path).lower().endswith('.pdf'):
                merged.save(spec.output_path, 'PDF', resolution=300)
            else:
                merged.save(spec.output_path)
        finally:
            for im in images:
                im.close()
            for tmp in temp_files:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass


def main():
    setup_logging()
    try:
        PlotJob.cli()
    except DraconError as e:
        handle_dracon_error(e, exit_code=1)
    except Exception as e:
        root = e
        while root.__cause__ is not None:
            root = root.__cause__
        if isinstance(root, DraconError):
            handle_dracon_error(root, exit_code=1)
        raise


if __name__ == '__main__':
    main()
