## {{{                          --     imports     --

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
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
)
from biocomptools.toollib.datasources import DataSource, DBSource

from biocomptools.toollib.networkprediction import NetworkPrediction, PredictionSamplingConfig
from biocomptools.toollib.typical_experimental_distribution import sample_latent

from biocomptools.toollib.common import config
from biocomptools.toollib.plot import PlotConfig, PlotTask, Figure
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
    layout_dimensions,
    extract_plot_data_metadata,
    extract_model_metadata,
    extract_prediction_config,
    build_figure_metadata,
    predicted_stats,
    build_prediction_pipeline,
    build_per_network_mvp,
    filter_compatible,
    format_z_label,
    smart_title,
    training_set_count,
    trained_on_status,
)
from biocomptools.toollib.figuremakers.measuredvspredicted import MeasuredVsPredictedData
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

# suppress the fork warning from jax
warnings.filterwarnings("ignore", message="os.fork()", module="subprocess")

logger = get_logger(__name__)


class FigureResult(BaseModel):
    """Result of running a single figure. Useful for chaining/composition."""

    path: str
    figure_spec: FigureSpec
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return self.path


class PlotResult(BaseModel):
    """Result of running PlotJob. Provides both paths and structured data for composition."""

    figures: List[FigureResult] = Field(default_factory=list)
    merged_path: Optional[str] = None

    @property
    def paths(self) -> List[str]:
        """List of output file paths (for backward compatibility)."""
        return [f.path for f in self.figures]

    def __iter__(self):
        """Iterate over paths for backward compatibility."""
        return iter(self.paths)

    def __len__(self):
        return len(self.figures)

    def __getitem__(self, idx):
        """Index access returns FigureResult for composition, or path for backward compat."""
        return self.figures[idx]


##────────────────────────────────────────────────────────────────────────────}}}


def debug_figures(figures):
    for fig in figures:
        ser_debug(fig, 'dill')
        ser_debug(fig, 'deepcopy')
        # ser_debug(fig, 'sizeof', max_size_mb=200)
        nr = dr.utils.node_repr(
            fig,
            enable_colors=True,
            show_biggest_context=5,
        )
        logger.debug(nr)


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


def run_figure(f, **kw) -> FigureResult:
    try:
        t0 = time.time()
        f.run(**kw)
        if not f.is_txt_output:
            plt.close('all')
        t1 = time.time()
        opath = getattr(f.figure_spec, '_output_path_override', None) or f.figure_spec.output_path or "(no file output)"
        logger.debug(f"Figure {opath} completed in {t1 - t0:.2f}s")
    except Exception as e:
        logger.error(f"Error running figure: {e}")
        logger.exception(e)
        raise
    opath = getattr(f.figure_spec, '_output_path_override', None) or f.figure_spec.output_path or "(no file output)"
    return FigureResult(
        path=str(opath),
        figure_spec=f.figure_spec,
        metadata=getattr(f.figure_spec, 'metadata', {}),
    )


def _run_figure_worker(fig, overwrite) -> FigureResult:
    return run_figure(fig, overwrite=overwrite)


DEFAULT_TYPES = [
    Figure,
    FigureResult,
    PlotResult,
    PlotConfig,
    PlotTask,
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


DEFAULT_CONTEXT = {
    **make_context_from_types(DEFAULT_TYPES),
    'get_pretty_axis_label': get_pretty_axis_label,
    'urlencoded': urlencoded,
    'expand_panel_atomics': expand_panel_atomics,
    'compose_rows': compose_rows,
    'compose_atomics': compose_atomics,
    'layout_dimensions': layout_dimensions,
    'extract_plot_data_metadata': extract_plot_data_metadata,
    'extract_model_metadata': extract_model_metadata,
    'extract_prediction_config': extract_prediction_config,
    'build_figure_metadata': build_figure_metadata,
    'predicted_stats': predicted_stats,
    'build_prediction_pipeline': build_prediction_pipeline,
    'build_per_network_mvp': build_per_network_mvp,
    'filter_compatible': filter_compatible,
    'format_z_label': format_z_label,
    'smart_title': smart_title,
    'training_set_count': training_set_count,
    'trained_on_status': trained_on_status,
    'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
}


@dracon_program(
    name='biocomp-plot',
    description='Generate plots from YAML configuration.',
    deferred_paths=['/figures.*'],
    context_types=DEFAULT_TYPES,
    context={
        'get_pretty_axis_label': get_pretty_axis_label,
        'urlencoded': urlencoded,
        'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
        'sample_latent': sample_latent,
        'expand_panel_atomics': expand_panel_atomics,
        'compose_rows': compose_rows,
        'compose_atomics': compose_atomics,
        'layout_dimensions': layout_dimensions,
        'extract_plot_data_metadata': extract_plot_data_metadata,
        'extract_model_metadata': extract_model_metadata,
        'extract_prediction_config': extract_prediction_config,
        'build_figure_metadata': build_figure_metadata,
        'predicted_stats': predicted_stats,
        'build_prediction_pipeline': build_prediction_pipeline,
        'filter_compatible': filter_compatible,
        'build_per_network_mvp': build_per_network_mvp,
        'format_z_label': format_z_label,
        'smart_title': smart_title,
        'training_set_count': training_set_count,
        'trained_on_status': trained_on_status,
    },
)
class PlotJob(BaseModel):
    figures: Annotated[List[DeferredNode[Figure]], Arg(help='List of figure objects to create')]
    nworkers: Annotated[int, Arg(help='Number of workers (processes) to use')] = 8
    skip_existing: Annotated[bool, Arg(help='Overwrite existing figures')] = False
    parallel_mode: Annotated[
        Literal['ray', 'none'],
        Arg(help='Parallel mode to use (multiprocess, ray, none)'),
    ] = 'ray'

    clear_figure_context_keys: Annotated[
        List[str], Arg(help='Clear these keys from the figure context')
    ] = []

    max_batch_size: Annotated[int, Arg(help='Maximum batch size for ray')] = 32

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
        """Generate temp file paths for merge mode when figures don't have unique outputs."""
        assert self.merge_spec is not None
        merge_out_dir = Path(self.merge_spec.output_path).parent
        # use same extension as final merge output
        ext = Path(self.merge_spec.output_file).suffix or '.png'
        return [str(merge_out_dir / f"_merge_tmp_{i:04d}{ext}") for i in range(n)]

    def run(self):
        overwrite = not self.skip_existing
        total_figures = len(self.figures)
        logger.debug(f"Going to create {total_figures} figures")

        for fig in self.figures:
            for k in self.clear_figure_context_keys:
                if k in fig.context:
                    del fig.context[k]

        t0 = time.time()
        out_paths = []
        temp_paths: list[str | None] = [None] * total_figures

        if self.merge_spec:
            order = list(range(total_figures))
            merge_out_dir = Path(self.merge_spec.output_path).parent
            merge_out_dir.mkdir(parents=True, exist_ok=True)
            temp_paths = self._get_temp_paths_for_merge(total_figures)
        else:
            order = list(np.random.permutation(total_figures))

        txt_mode = self.use_txt_plotting
        stdout_txt = not self.no_stdout_txt_plot

        if self.nworkers <= 1 or self.parallel_mode == 'none' or total_figures <= 1:
            fig_copies = [
                self.figures[i].copy(reroot=True)
                for i in maybetqdm(order, min_len=20, desc='Copying figure tasks')
            ]
            constructed_figures = [
                construct_figure(
                    fig,
                    output_path_override=temp_paths[order[j]],
                    text_mode=txt_mode,
                    stdout_txt_plot=stdout_txt,
                )
                for j, fig in enumerate(
                    maybetqdm(fig_copies, min_len=5, desc='Constructing figures')
                )
            ]

            out_paths = [
                run_figure(f, overwrite=overwrite)
                for f in maybetqdm(constructed_figures, min_len=2, desc='Running figures')
            ]
            outpathstr = '\n  - ' + '\n  - '.join(r.path for r in out_paths)
            logger.info(f"Generated {len(out_paths)} figures:{outpathstr}")

        elif self.parallel_mode == 'ray':
            logger.debug(f"Using joblib with {self.nworkers} workers")
            from joblib import Parallel, delayed
            import cloudpickle

            time_start = time.time()

            batch_idx = 0
            num_batches = (total_figures + self.max_batch_size - 1) // self.max_batch_size
            batch_start = 0
            while batch_start < total_figures:
                batch_idx += 1
                batch_len = min(self.max_batch_size, total_figures - batch_start)
                batch_order_indices = order[batch_start : batch_start + batch_len]
                fig_copies = [
                    self.figures[i].copy(reroot=True)
                    for i in maybetqdm(
                        batch_order_indices,
                        min_len=20,
                        desc=f'Copying figure tasks for batch {batch_idx}/{num_batches}',
                    )
                ]

                batch_start += batch_len

                constructed_figures = [
                    construct_figure(
                        fig,
                        output_path_override=temp_paths[batch_order_indices[j]],
                        text_mode=txt_mode,
                        stdout_txt_plot=stdout_txt,
                    )
                    for j, fig in enumerate(
                        maybetqdm(
                            fig_copies,
                            min_len=5,
                            desc=f'Constructing figures for batch {batch_idx}/{num_batches}',
                        )
                    )
                ]

                # for fig in constructed_figures:
                    # _detach_lazy_plot_data(fig)
                    # _convert_jax_to_numpy_inplace(fig)

                # try parallel execution with fallback to sequential if pickling fails
                parallel_figs, sequential_figs, sequential_indices = [], [], []
                for i, fig in enumerate(constructed_figures):
                    try:
                        cloudpickle.dumps(fig)
                        parallel_figs.append(fig)
                    except (TypeError, AttributeError) as e:
                        logger.debug(f"Figure {i} cannot be pickled ({e}), running sequentially")
                        sequential_figs.append(fig)
                        sequential_indices.append(i)

                results = [None] * len(constructed_figures)

                # run picklable figures in parallel
                if parallel_figs:
                    parallel_results = Parallel(n_jobs=self.nworkers, backend='loky')(
                        delayed(_run_figure_worker)(fig, overwrite)
                        for fig in tqdm(
                            parallel_figs,
                            total=len(parallel_figs),
                            desc=f'Parallel figures in batch {batch_idx}/{num_batches}',
                        )
                    )
                    j = 0
                    for i in range(len(constructed_figures)):
                        if i not in sequential_indices:
                            results[i] = parallel_results[j]
                            j += 1

                # run unpicklable figures sequentially
                if sequential_figs:
                    logger.info(
                        f"Running {len(sequential_figs)} figures sequentially (not picklable)"
                    )
                    for i, fig in zip(
                        sequential_indices,
                        tqdm(
                            sequential_figs,
                            desc=f'Sequential figures in batch {batch_idx}/{num_batches}',
                        ),
                        strict=True,
                    ):
                        results[i] = run_figure(fig, overwrite=overwrite)

                out_paths.extend(results)

                time_batch_end = time.time()
                logger.debug(f"Batch {batch_idx} completed in {time_batch_end - time_start:.2f}s")

                fpaths = '\n  - ' + '\n  - '.join(r.path for r in results if r)
                logger.debug(f"Generated {len(results)} figures:{fpaths}")

        t1 = time.time()
        logger.info(f"{total_figures} figures completed in {t1 - t0:.2f}s")

        merged_path = None
        if self.merge_spec and out_paths:
            paths = [r.path for r in out_paths]
            self._merge_figures(paths)
            merged_path = str(self.merge_spec.output_path)

        return PlotResult(figures=out_paths, merged_path=merged_path)

    def _merge_figures(self, paths: list[str]):
        """Merge generated figures into single output per MergeSpec."""
        spec = self.merge_spec
        if not paths:
            return

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)

        if spec.mode == "pages":
            self._merge_as_pdf_pages(paths)
        else:  # grid mode
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
        """Merge PDFs into grid layout without rasterization using pypdf."""
        from pypdf import PdfReader, PdfWriter, PageObject, Transformation

        spec = self.merge_spec
        rows, cols = spec.rows, spec.cols
        n = len(paths)

        readers = [PdfReader(p) for p in paths]
        pages = [r.pages[0] for r in readers]

        # get dimensions from source pages (in points, 72pt = 1 inch)
        widths = [float(p.mediabox.width) for p in pages]
        heights = [float(p.mediabox.height) for p in pages]

        if spec.col_widths:
            col_w = [w * max(widths) for w in spec.col_widths]
        else:
            col_w = [max(widths)] * cols

        if spec.row_heights:
            row_h = [h * max(heights) for h in spec.row_heights]
        else:
            row_h = [max(heights)] * rows

        total_w = sum(col_w) + spec.hspace * (cols - 1)
        total_h = sum(row_h) + spec.vspace * (rows - 1)

        merged_page = PageObject.create_blank_page(width=total_w, height=total_h)

        y_offset = total_h  # PDF coords: origin at bottom-left
        idx = 0
        for r in range(rows):
            y_offset -= row_h[r]
            x_offset = 0
            for c in range(cols):
                if idx < n:
                    page = pages[idx]
                    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
                    scale_x, scale_y = col_w[c] / pw, row_h[r] / ph
                    scale = min(scale_x, scale_y)  # preserve aspect ratio
                    # center within cell
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
        """Merge figures as separate pages in a PDF."""
        from pypdf import PdfReader, PdfWriter
        from PIL import Image
        import io

        spec = self.merge_spec
        writer = PdfWriter()

        for p in paths:
            if p.lower().endswith('.pdf'):
                reader = PdfReader(p)
                for page in reader.pages:
                    writer.add_page(page)
            else:
                # convert image to PDF page
                img = Image.open(p)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                pdf_bytes = io.BytesIO()
                img.save(pdf_bytes, format='PDF', resolution=300)
                pdf_bytes.seek(0)
                reader = PdfReader(pdf_bytes)
                writer.add_page(reader.pages[0])
                img.close()

        with open(spec.output_path, 'wb') as f:
            writer.write(f)

    def _merge_as_images(self, paths: list[str]):
        """Merge figures as images (fallback for non-PDF or mixed inputs)."""
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
        # check if root cause is a DraconError
        root = e
        while root.__cause__ is not None:
            root = root.__cause__
        if isinstance(root, DraconError):
            handle_dracon_error(root, exit_code=1)
        raise


if __name__ == '__main__':
    main()
