# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
## {{{                          --     imports     --
from typing import Any, List, Dict, Optional, Annotated
import matplotlib as mpl
import os
from dracon.draconstructor import resolve_all_lazy
from biocomp.datautils import DataRescaler
from biocomp.utils import PartialFunction
from dracon.deferred import DeferredNode
from biocomp import utils as ut
from biocomp.plotutils import FigureSpec
from dracon.utils import dict_like
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from biocomptools.logging_config import get_logger
from biocomptools.trainutils import make_json_ready
from biocomp.tracing import is_plot_debug_enabled, save_debug_state

logger = get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
##────────────────────────────────────────────────────────────────────────────}}}

## {{{                        --     plot config     --


def load_default_plotconf():
    import dracon as dr
    from dracon.lazy import resolve_all_lazy

    plcontent = dr.load(
        "pkg:biocomptools:configs/plot_config/default_plotconf_v2",
        enable_interpolation=True,
        raw_dict=True,
    )
    resolve_all_lazy(plcontent)
    pc = PlotConfig(**plcontent)
    return pc


def load_default_rescaler() -> DataRescaler:
    import dracon as dr
    from dracon.lazy import resolve_all_lazy

    rescaler = dr.load(
        "pkg:biocomptools:configs/data_config/rescaler/EBFP2_compressed_v2",
        enable_interpolation=True,
    )
    resolve_all_lazy(rescaler)
    return rescaler


class PlotConfig(BaseModel):
    rc_context: Dict[str, Any] = {}
    callstack_params: Dict[str, Any] = {}  # nested parameters for the plotting function
    rescaler: Optional[DataRescaler] = Field(default_factory=load_default_rescaler)

    def _auto_extract_metadata(
        self,
        callstack_conf: dict,
        plot_method: PartialFunction,
        size_limit: int = 100000,
        *args,
        **kwargs,
    ):
        """
        Returns some possibly interesting metadata from the callstack configuration and the plot method.
        """
        import json

        def can_dump(obj):
            try:
                dmp = json.dumps(make_json_ready(obj))
                if len(dmp) > size_limit:
                    logger.debug(f"Object too large to dump: {len(dmp)} > {size_limit}: {dmp}")
                    return False
                logger.debug(f"Object dumped: {len(dmp)} < {size_limit}: {dmp}")
                return True
            except Exception as e:
                logger.debug(f"Error dumping json object: {e}")
                return False

        metadata = {}
        metadata["plot_method"] = plot_method.get_name()

        if can_dump(callstack_conf):
            metadata["callstack_conf"] = callstack_conf

        extra_args = list(args)

        for pval in plot_method.args + extra_args:
            if hasattr(pval, "metadata") and pval.metadata:
                for k, v in pval.metadata.items():
                    if can_dump(v):
                        metadata[k] = v

        extra_kwargs = list(kwargs.items())
        for _, pval in list(plot_method.kwargs.items()) + extra_kwargs:
            if hasattr(pval, "metadata") and pval.metadata:
                for k, v in pval.metadata.items():
                    if can_dump(v):
                        metadata[k] = v

        return metadata

    def prepare_func(
        self,
        plot_method: PartialFunction,
        auto_callstack_bind: bool = True,
        overwrite_kwargs: Optional[dict] = None,
    ):
        callstack_conf = {}
        if auto_callstack_bind:
            callstack_conf = ut.generate_full_nested_config(
                self.callstack_params, namespace="biocomp.plotting"
            ).get(f"{plot_method.get_name()}_params", {})

        if overwrite_kwargs:
            plot_method.set_missing_kwargs(overwrite_kwargs)

        def wrapped_plot_method(*args, **kwargs):
            from biocomp.plotutils import PlotFunctionResult

            # collect metadata after calling the plot method
            # (better than before, since some args may be modified e.g. lazy-loaded plot data)
            res = plot_method(*args, **kwargs)

            # extract output-side metadata from PlotFunctionResult
            extra_meta: dict = {}
            if isinstance(res, PlotFunctionResult):
                extra_meta = res.metadata
                res = res.rendering

            metadata = self._auto_extract_metadata(
                callstack_conf,
                plot_method,
                *args,
                **kwargs,
            )
            metadata.update(extra_meta)
            return res, metadata

        if self.rescaler is not None:
            plot_method.set_missing_kwargs({"rescaler": self.rescaler})

        def prepared_func(*args, rc=self.rc_context, cs=callstack_conf, **kwargs):
            full_kwargs = {**cs, **kwargs}
            with mpl.rc_context(rc=rc):
                return wrapped_plot_method(*args, **full_kwargs)

        return prepared_func

    def inherit_from(
        self, other: "PlotConfig", keep_rescaler: bool = True, key: str = "<<{+<}[~<]"
    ):
        from dracon.merge import MergeKey, merged

        k = MergeKey(raw=key)

        if not keep_rescaler or not self.rescaler:
            self.rescaler = other.rescaler

        self.callstack_params = merged(other.callstack_params, self.callstack_params, k)  # type: ignore
        self.rc_context = merged(other.rc_context, self.rc_context, k)  # type: ignore


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                         --     plot task     --


class PlotTask(BaseModel):
    plot_config: PlotConfig = Field(default_factory=PlotConfig)
    plot_method: Optional[PartialFunction] = None
    overlays: List[Any] = []  # list of Overlay instances; runs after plot_method
    # Optional separate data source for overlays. Useful when plot_method
    # consumes a transformed PlotData (e.g. KNN-smoothed centroids) but
    # overlays need the original raw points to compute forward maps. Falls
    # back to plot_method.kwargs.plot_data if unset.
    overlay_data: Any = None
    auto_callstack_bind: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _ax: Optional[mpl.axes.Axes] = None
    _overlay_results: List[Dict[str, Any]] = []

    def run(self):
        # generates some metadata and returns it
        metadata = {}
        if self.plot_method:
            kw = {"ax": self._ax} if self._ax else {}
            f = self.plot_config.prepare_func(
                plot_method=self.plot_method,
                auto_callstack_bind=self.auto_callstack_bind,
                overwrite_kwargs=kw,
            )

            result, metadata = f()

        if self.overlays and self._ax is not None:
            plot_data = self.overlay_data
            if plot_data is None and self.plot_method:
                plot_data = self.plot_method.kwargs.get("plot_data")
            self._overlay_results = []
            for ov in self.overlays:
                if not getattr(ov, "enabled", True):
                    continue
                try:
                    self._overlay_results.append(
                        ov.apply(self._ax, plot_data, self.plot_config)
                    )
                except Exception as e:
                    logger.error(f"overlay {type(ov).__name__} failed: {e}")
                    logger.exception(e)
            if self._overlay_results:
                metadata["overlays"] = self._overlay_results

        return metadata


##────────────────────────────────────────────────────────────────────────────}}}

## {{{                          --     figure     --{{{


def resolve(obj):
    resolve_all_lazy(obj)
    return obj


TXT_PLOT_FUNC_MAP = {
    "biocomp.plotutils.smooth": "biocomp.plotutils.smooth_txt",
    "jeanplot.plots.smooth_1d.smooth_1d": "jeanplot.plots.txt.smooth_1d_txt",
    "jeanplot.plots.smooth_2d.smooth_2d": "jeanplot.plots.txt.smooth_2d_txt",
    "jeanplot.plots.smooth_3d.smooth_3d": "jeanplot.plots.txt.smooth_3d_txt",
}


class BiocompPlotFigure(BaseModel):
    figure_spec: Annotated[FigureSpec, BeforeValidator(resolve)]
    plot_config: PlotConfig = Field(default_factory=load_default_plotconf)
    plot_tasks: List[DeferredNode[PlotTask]] = []

    text_mode: bool = False
    stdout_txt_plot: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ptasks: Optional[List[PlotTask]] = None
    _txt_output: Optional[str] = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.plot_tasks = [task.copy(reroot=True) for task in self.plot_tasks]

    @property
    def is_txt_output(self) -> bool:
        return self.text_mode or str(self.figure_spec.output_file).endswith(".txt")

    def prepare(self):
        if self.is_txt_output:
            self._prepare_txt()
        else:
            self._prepare_mpl()

    def _prepare_mpl(self):
        with mpl.rc_context(rc=self.plot_config.rc_context):
            try:
                self._figax = self.figure_spec.make_figure()
            except Exception as e:
                logger.error(f"Error making figure axes: {e}")
                logger.exception(e)
                return

            self._ptasks = []
            for i, tc in enumerate(self.plot_tasks):
                try:
                    pt = tc.construct(context={"FIG": self._figax})
                    if dict_like(pt):
                        pt = PlotTask(**pt)  # type: ignore
                    pt.plot_config.inherit_from(self.plot_config)

                    flat = self._figax.flat_ax
                    pt._ax = flat[i] if i < len(flat) else None
                    self._ptasks.append(pt)
                except Exception as e:
                    logger.error(f"Error constructing plot task {i}: {e}")
                    logger.exception(e)
                    continue

    def _prepare_txt(self):
        class DummyFigAx:
            flat_ax = [None] * 100
            axes = [[None] * 10 for _ in range(10)]
            figure = None

        self._figax = DummyFigAx()
        self._ptasks = []
        for i, tc in enumerate(self.plot_tasks):
            try:
                pt = tc.construct(context={"FIG": self._figax})
                if dict_like(pt):
                    pt = PlotTask(**pt)
                pt.plot_config.inherit_from(self.plot_config)
                pt._ax = None
                self._ptasks.append(pt)
            except Exception as e:
                logger.error(f"Error constructing txt plot task {i}: {e}")
                logger.exception(e)
                continue

    def run(self, overwrite: bool = True, finalize: bool = True):
        if not overwrite and self.figure_spec.output_path and self.figure_spec.output_path.exists():
            logger.info(f"Skipping existing figure {self.figure_spec.output_path}")
            return

        if self._ptasks is None:
            self.prepare()
        assert isinstance(self._ptasks, list)

        if self.is_txt_output:
            self._run_txt(overwrite=overwrite, finalize=finalize)
        else:
            self._run_mpl(overwrite=overwrite, finalize=finalize)

    def _run_mpl(self, overwrite: bool = True, finalize: bool = True):
        with mpl.rc_context(rc=self.plot_config.rc_context):
            metadata = dict(self.figure_spec.metadata) if self.figure_spec.metadata else {}
            embed_plot_tasks = _env_flag("BIOCOMP_EMBED_PLOT_TASK_METADATA", False)
            embed_grid_data = _env_flag("BIOCOMP_EMBED_GRID_DATA", False)
            if embed_plot_tasks:
                metadata["plot_tasks"] = []
            all_grid_data = []
            for i, pt in enumerate(self._ptasks):
                try:
                    resolve_all_lazy(pt)
                    pt_metadata = pt.run()
                    gd = pt_metadata.pop("grid_data", None)
                    if embed_grid_data and gd:
                        all_grid_data.extend(gd)
                    if embed_plot_tasks:
                        metadata["plot_tasks"].append(pt_metadata)
                except Exception as e:
                    logger.error(f"Error running plot task {i}: {e}")
                    logger.exception(e)
                    continue

            if all_grid_data:
                from jeanplot.data.grid import grid_data_to_b64

                metadata["grid_data"] = grid_data_to_b64(all_grid_data)

            if is_plot_debug_enabled():
                self._save_plot_debug_state(metadata)

            if finalize:
                self.figure_spec.metadata = metadata
                self.figure_spec.finalize(self._figax)

    def _run_txt(self, overwrite: bool = True, finalize: bool = True):
        from jeanplot.plots.txt import TextPlotResult

        txt_parts = []
        metadata = {"plot_tasks": []}

        for i, pt in enumerate(self._ptasks):
            try:
                resolve_all_lazy(pt)
                if pt.plot_method is None:
                    continue

                func_name = pt.plot_method.get_name()
                txt_func_name = TXT_PLOT_FUNC_MAP.get(func_name)

                if txt_func_name is None:
                    for key, val in TXT_PLOT_FUNC_MAP.items():
                        if func_name.endswith(key.split(".")[-1]):
                            txt_func_name = val
                            break

                if txt_func_name is None:
                    logger.warning(f"No txt plot function for {func_name}, skipping")
                    continue

                module_path, func_name_only = txt_func_name.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_path)
                txt_func = getattr(module, func_name_only)

                kwargs = dict(pt.plot_method.kwargs)
                kwargs["ax"] = None
                if pt.plot_config.rescaler is not None:
                    kwargs.setdefault("rescaler", pt.plot_config.rescaler)

                cs = ut.generate_full_nested_config(
                    pt.plot_config.callstack_params, namespace="biocomp.plotting"
                ).get(f"{func_name_only}_params", {})
                kwargs.update(cs)

                result = txt_func(**kwargs)
                if isinstance(result, TextPlotResult):
                    txt_parts.append(result.text)
                    metadata["plot_tasks"].append({"txt_result": True})
                elif isinstance(result, str):
                    txt_parts.append(result)
                    metadata["plot_tasks"].append({"txt_result": True})

            except Exception as e:
                logger.error(f"Error running txt plot task {i}: {e}")
                logger.exception(e)
                continue

        self._txt_output = "\n\n".join(txt_parts)

        if self.stdout_txt_plot and self._txt_output:
            print(self._txt_output)

        if finalize:
            self.figure_spec.metadata = metadata
            self._finalize_txt()

    def _finalize_txt(self):
        from pathlib import Path as PathLib

        output_path = self.figure_spec.output_path
        if output_path is None:
            return
        if not str(output_path).endswith(".txt"):
            output_path = PathLib(str(output_path).rsplit(".", 1)[0] + ".txt")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._txt_output:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self._txt_output)
            logger.debug(f"Text plot saved to {output_path}")

        self.figure_spec._output_path_override = str(output_path)

    def _save_plot_debug_state(self, metadata: dict):
        """Save comprehensive plot debug state after all tasks complete."""
        import numpy as np

        data = {}
        meta = {
            "output_path": str(self.figure_spec.output_path or ""),
            "output_dir": str(self.figure_spec.output_dir),
            "output_file": str(self.figure_spec.output_file),
            "n_tasks": len(self._ptasks) if self._ptasks else 0,
            "metadata": metadata,
        }

        # Extract plot data from each task's plot_method kwargs
        for i, pt in enumerate(self._ptasks or []):
            if pt.plot_method is None:
                continue

            # Look for common data containers in plot method kwargs
            for key, val in pt.plot_method.kwargs.items():
                if hasattr(val, "xval") and hasattr(val, "yval"):
                    # PlotData-like object
                    data[f"task_{i}_{key}_X"] = np.asarray(val.xval)
                    data[f"task_{i}_{key}_Y"] = np.asarray(val.yval)
                    if hasattr(val, "input_names"):
                        meta[f"task_{i}_{key}_input_names"] = val.input_names
                elif hasattr(val, "x") and hasattr(val, "y"):
                    # DataSource-like object
                    data[f"task_{i}_{key}_X"] = np.asarray(val.x)
                    data[f"task_{i}_{key}_Y"] = np.asarray(val.y)
                    if hasattr(val, "input_names"):
                        meta[f"task_{i}_{key}_input_names"] = val.input_names
                    if hasattr(val, "metadata"):
                        meta[f"task_{i}_{key}_metadata"] = val.metadata

        save_debug_state(
            stage="figure_complete",
            data=data,
            metadata=meta,
            output_dir=str(self.figure_spec.output_dir),
            mode="plot",
        )

    @property
    def fig(self):
        if hasattr(self, "_figax"):
            return self._figax.figure
        raise AttributeError("Figure not prepared yet. Call 'prepare()' first.")


##────────────────────────────────────────────────────────────────────────────}}}
