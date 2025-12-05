## {{{                          --     imports     --

from biocomptools.logging_config import get_logger, setup_logging
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from tqdm import tqdm
from biocomptools.toollib.common import maybetqdm
import numpy as np
import memray

import dracon as dr
from dracon.utils import ser_debug
from dracon.diagnostics import DraconError, handle_dracon_error
from pympler.asizeof import asizeof
import os

import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
from typing import List, Annotated, Literal
from dracon.commandline import make_program, Arg
from dracon.deferred import DeferredNode

from biocomp.utils import PartialFunction
from biocomp.datautils import DataRescaler
from biocomp.plotutils import FigureSpec, FigAx, SimpleLayout, MergeSpec
from biocomptools.toollib.datasources import DataSource, DBSource

from biocomptools.toollib.networkprediction import NetworkPrediction

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

from biocomptools.modelmodel import BiocompModel, get_shared_params, NetworkModel
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

DEFAULT_TYPES = [
    Figure,
    PlotConfig,
    PlotTask,
    DataSource,
    DBSource,
    FigureSpec,
    FigAx,
    SimpleLayout,
    MergeSpec,
    Regex,
    iRegex,
    ModelSelector,
    BiocompModel,
    NetworkModel,
    NetworkPrediction,
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
]


def make_context_from_types(types):
    return {t.__name__: t for t in types}


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


def construct_figure(figure_node):
    try:
        figure = figure_node.construct(deferred_paths=['/plot_tasks.*'])
        if dict_like(figure):
            figure = Figure(**figure)  # type: ignore
        assert isinstance(figure, Figure), f"Expected Figure, got {type(figure)}"
    except DraconError:
        # let DraconErrors propagate - they'll be formatted nicely at the top level
        raise
    except Exception as e:
        logger.error(f"Error constructing figure: {e}")
        logger.exception(e)
        raise
    return figure


def run_figure(f, **kw):
    try:
        t0 = time.time()
        f.run(**kw)
        plt.close('all')
        t1 = time.time()
        opath = f.figure_spec.output_path
        logger.debug(f"Figure {opath} completed in {t1 - t0:.2f}s")
    except Exception as e:
        logger.error(f"Error running figure: {e}")
        logger.exception(e)
        raise
    return str(opath)


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

    merge_spec: Annotated[
        MergeSpec | None, Arg(help='Merge all figures into a single output')
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        if self.merge_spec:
            order = list(range(total_figures))
        else:
            order = list(np.random.permutation(total_figures))

        if self.nworkers <= 1 or self.parallel_mode == 'none' or total_figures <= 1:
            fig_copies = [
                self.figures[i].copy(reroot=True)
                for i in maybetqdm(order, min_len=20, desc='Copying figure tasks')
            ]
            constructed_figures = [
                construct_figure(fig)
                for fig in maybetqdm(fig_copies, min_len=5, desc='Constructing figures')
            ]

            out_paths = [
                run_figure(f, overwrite=overwrite)
                for f in maybetqdm(constructed_figures, min_len=2, desc='Running figures')
            ]
            outpathstr = '\n  - ' + '\n  - '.join(out_paths)
            logger.info(f"Generated {len(out_paths)} figures:{outpathstr}")

        elif self.parallel_mode == 'ray':
            logger.debug(f"Using ray with {self.nworkers} workers")
            import ray

            time_ray_start = time.time()

            if not ray.is_initialized():
                context = ray.init(
                    num_cpus=self.nworkers,
                    runtime_env={
                        'env_vars': {
                            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                            'JAX_PLATFORMS': 'cpu',
                        }
                    },
                )
                logger.info(f"Ray context: {context}")
                logger.info(f"Ray dashboard at: {context.dashboard_url}")

            def wait_iterator(task_refs):
                while task_refs:
                    done, task_refs = ray.wait(task_refs, num_returns=1)
                    yield ray.get(done[0])

            batch_idx = 0
            num_batches = (total_figures + self.max_batch_size - 1) // self.max_batch_size
            batch_start = 0
            while batch_start < total_figures:
                batch_idx += 1
                batch_len = min(self.max_batch_size, total_figures - batch_start)
                fig_copies = [
                    self.figures[i].copy(reroot=True)
                    for i in maybetqdm(
                        order[batch_start : batch_start + batch_len],
                        min_len=20,
                        desc=f'Copying figure tasks for batch {batch_idx}/{num_batches}',
                    )
                ]

                batch_start += batch_len

                constructed_figures = [
                    construct_figure(fig)
                    for fig in maybetqdm(
                        fig_copies,
                        min_len=5,
                        desc=f'Constructing figures for batch {batch_idx}/{num_batches}',
                    )
                ]

                fig_refs = ray.put(constructed_figures)

                @ray.remote
                def run_figure_ray(i, fig_regs=fig_refs):
                    fig = ray.get(fig_regs)[i]
                    return run_figure(fig, overwrite=overwrite)

                task_refs = [run_figure_ray.remote(i) for i in range(len(constructed_figures))]

                time_ray_work_submit = time.time()
                logger.debug(f"Ray work submitted in {time_ray_work_submit - time_ray_start:.2f}s")

                results = list(
                    tqdm(
                        wait_iterator(task_refs),
                        total=batch_len,
                        desc=f'Generating figures in batch {batch_idx}/{num_batches}',
                    )
                )
                out_paths.extend(results)

                fpaths = '\n  - ' + '\n  - '.join(results)
                logger.debug(f"Generated {len(results)} figures:{fpaths}")

        t1 = time.time()
        logger.info(f"{total_figures} figures completed in {t1 - t0:.2f}s")

        if self.merge_spec and out_paths:
            self._merge_figures(out_paths)

    def _merge_figures(self, paths: list[str]):
        """Merge generated figures into single output per MergeSpec."""
        spec = self.merge_spec
        if not paths:
            return

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
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
                        check=True
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


DEFAULT_CONTEXT = {
    **make_context_from_types(DEFAULT_TYPES),
    'get_pretty_axis_label': get_pretty_axis_label,
    'urlencoded': urlencoded,
    'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
}


def main():
    setup_logging()

    prog = make_program(
        PlotJob,
        name='biocomp-plot',
        description='Make plots.',
    )

    pj, args = prog.parse_args(
        sys.argv[1:],
        deferred_paths=[
            '/figures.*',
        ],
        context=DEFAULT_CONTEXT,
        enable_shorthand_vars=False,
    )

    try:
        pj.run()
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
