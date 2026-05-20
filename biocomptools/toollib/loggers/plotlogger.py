# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from __future__ import annotations

import copy
import matplotlib

matplotlib.use("Agg")

from dracon.deferred import DeferredNode
from typing import Any, Callable, Literal
from pathlib import Path
from pydantic import PrivateAttr
from biocomptools.plot import PlotJob
from biocomptools.toollib.loggers.logger import Logger
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger

logger = get_logger(__name__)


class PlotLogger(Logger):
    jobs: list[DeferredNode[PlotJob]] = []
    execution_mode: Literal["inline", "thread", "process"] = "process"
    extra_context: dict = {}
    required_arrays: list[str] = ["loss", "latest_params"]
    required_extra: list[str] = ["embedding_snapshots"]
    max_trajectory_points: int = 1000

    _save_dir: Path | None = PrivateAttr(default=None)
    _get_best_model_fn: Callable[..., object] | None = PrivateAttr(default=None)

    def model_post_init(self, __context):
        super().model_post_init(__context)

    def initialize(self, training_program):
        super().initialize(training_program)
        self._save_dir = getattr(training_program, "_save_dir", None)
        self._get_best_model_fn = training_program.get_best_model_func()

    def _build_base_context(
        self, step: int, best_model: object, extra_ctx: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        from biocomptools.toollib.figuremakers.innernodes import (
            InnerNodesFigure,
            InnerNodesFigureSpec,
        )
        from biocomptools.toollib.figuremakers.benchmarkutils import (
            BenchmarkData,
            BenchmarkItem,
            render_summary_header,
            render_metrics_panel,
        )
        from biocomp.plotutils import FigureSpec, FigureLayout

        ctx = {
            "InnerNodesFigure": InnerNodesFigure,
            "InnerNodesFigureSpec": InnerNodesFigureSpec,
            "BenchmarkData": BenchmarkData,
            "BenchmarkItem": BenchmarkItem,
            "render_summary_header": render_summary_header,
            "render_metrics_panel": render_metrics_panel,
            "FigureSpec": FigureSpec,
            "FigureLayout": FigureLayout,
            "best_model": best_model,
            "step": step,
            "save_dir": self._save_dir,
            "embedding_trajectories": None,
            **self.extra_context,
        }
        if extra_ctx:
            ctx.update(extra_ctx)
        return ctx

    def do_plot(
        self,
        step,
        best_model,
        job_idx,
        job: DeferredNode[PlotJob],
        extra_ctx: dict[str, Any] | None = None,
    ):
        j = copy.deepcopy(job)
        if best_model is None:
            logger.info("Skipping plot - no best model available")
            return True

        ctx = self._build_base_context(step, best_model, extra_ctx)

        try:
            logger.info(f"Job {job_idx + 1}: constructing deferred node")
            constructed = j.construct(context=ctx)
            if not isinstance(constructed, PlotJob):
                logger.info(f"Job {job_idx + 1}: converting dict to PlotJob")
                constructed = PlotJob(**constructed)
            logger.info(
                f"Job {job_idx + 1}: running PlotJob with {len(constructed.figures)} figures"
            )
            constructed.run()
            logger.info(f"Job {job_idx + 1} completed successfully")
        except Exception as e:
            logger.error(f"Job {job_idx + 1} failed: {e}")
            logger.exception(e)
            return False
        return True

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step_history = view.to_step_history()
        best_model = self._resolve_best_model(step_history)

        snapshots = context.extra.get("embedding_snapshots", [])
        embedding_trajectories = (
            self._build_trajectories(snapshots, max_points=self.max_trajectory_points)
            if snapshots
            else None
        )
        extra_ctx = {"embedding_trajectories": embedding_trajectories} if embedding_trajectories else None

        logger.info(f"PlotLogger on_batch at step {context.current_step}, {len(self.jobs)} jobs")
        for job_idx, job in enumerate(self.jobs):
            for attempt in range(2):
                try:
                    if self.do_plot(context.current_step, best_model, job_idx, job, extra_ctx=extra_ctx):
                        break
                except Exception as e:
                    logger.error(f"Job {job_idx + 1} on_batch attempt {attempt + 1} failed: {e}")

    def _resolve_best_model(self, step_history: dict[str, Any]) -> object | None:
        losses = step_history.get("loss")
        params = step_history.get("latest_params")
        if self._get_best_model_fn is not None:
            return self._get_best_model_fn(params, losses)
        return None

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        """End-of-training plot with embedding trajectories from accumulated snapshots."""
        step_history = view.to_step_history()
        best_model = self._resolve_best_model(step_history)

        snapshots = context.extra.get("embedding_snapshots", [])
        embedding_trajectories = (
            self._build_trajectories(snapshots, max_points=self.max_trajectory_points)
            if snapshots
            else None
        )

        extra_ctx = {
            "embedding_trajectories": embedding_trajectories,
        }

        logger.info(f"PlotLogger on_end at step {context.current_step}, {len(self.jobs)} jobs")
        for job_idx, job in enumerate(self.jobs):
            for attempt in range(2):
                try:
                    if self.do_plot(
                        context.current_step, best_model, job_idx, job, extra_ctx=extra_ctx
                    ):
                        break
                except Exception as e:
                    logger.error(f"Job {job_idx + 1} on_end attempt {attempt + 1} failed: {e}")

    @staticmethod
    def _build_trajectories(
        snapshots: list[tuple[int, dict[str, Any]]],
        max_points: int = 1000,
    ) -> dict[str, list[tuple[float, ...]]]:
        """Convert accumulated snapshots to per-embedding-name trajectory dicts.

        Returns:
            {emb_type: {name_index: [(v0, v1, ...), ...]}} flattened to
            {"{emb_type}_{idx}": [(v0, v1, ...), ...]}
        """
        if not snapshots:
            return {}

        if len(snapshots) > max_points:
            import numpy as np

            indices = np.linspace(0, len(snapshots) - 1, max_points, dtype=int)
            snapshots = [snapshots[i] for i in indices]

        # Build trajectories keyed by "{emb_type}_{idx}"
        trajectories: dict[str, list[tuple[float, ...]]] = {}
        for _step, snapshot in snapshots:
            for emb_type, values_list in snapshot.items():
                for idx, vals in enumerate(values_list):
                    key = f"{emb_type}_{idx}"
                    if key not in trajectories:
                        trajectories[key] = []
                    if isinstance(vals, (list, tuple)):
                        trajectories[key].append(tuple(float(v) for v in vals))
                    else:
                        trajectories[key].append((float(vals),))

        return trajectories
