"""Logger that generates a uORF matrix prediction figure at end of training.

Uses the in-memory BiocompModel from get_best_model_func() to run predictions
on the full PgU 9x9 matrix, then renders a UORFMatrixFigure with annotations
showing which uORF pairs were used in training.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Callable, Optional

from biocomptools.logging_config import get_logger
from biocomptools.toollib.loggers.logger import Logger

logger = get_logger(__name__)


class MatrixEvalLogger(Logger):
    """Generate uORF matrix prediction figure after training."""

    ern_name: str = "Pgu"
    grid_resolution: int = 32
    show_individual_rmse: bool = True
    show_overall_rmse: bool = True
    trained_uorf_annotation: list[list[int]] = []
    parallel_ok: bool = True

    _save_dir: Optional[Path] = None
    _get_best_model: Optional[Callable] = None

    def initialize(self, training_program):
        super().initialize(training_program)
        self._save_dir = getattr(training_program, "_save_dir", None)
        self._get_best_model = training_program.get_best_model_func()

    def get_callbacks(self, training_program):
        def callback(step, training_config, step_history=None, **kwargs):
            if step_history is None:
                return
            params = step_history.get("latest_params")
            losses = step_history.get("loss")
            if params is None:
                logger.debug("MatrixEvalLogger: no params available, skipping")
                return

            assert self._get_best_model is not None
            model = self._get_best_model(all_params=params, all_losses=losses)
            if model is None:
                logger.warning("MatrixEvalLogger: no best model available, skipping")
                return

            try:
                self._run_eval(model)
            except Exception as e:
                logger.error(f"MatrixEvalLogger: evaluation failed: {e}")
                logger.exception(e)

        if self.call_at_interval is not None:
            return [(self.call_at_interval, callback)]
        return []

    def _run_eval(self, model):
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.datasources import DBSource
        from biocomptools.toollib.figuremakers.uorfmatrixfigure import (
            UORFMatrixFigure,
            bundle_uorf_data,
        )
        from biocomptools.toollib.networkselector import iRegex
        from biocomptools.toollib.networkprediction import NetworkPrediction
        from biocomptools.toollib.plot import load_default_plotconf

        # 1. Load full PgU matrix data from DB
        ern_xp_name = self.ern_name if self.ern_name.lower() != "case" else ""
        db_source = DBSource(
            content=[
                {
                    "experiment_name": iRegex(f".*matrix{ern_xp_name}"),
                    "calibration_name": iRegex(".*[Ff][Ii][Nn][Aa][Ll].*"),
                }
            ],
        )
        all_plot_data = db_source.get_data()
        bundles = bundle_uorf_data(all_plot_data)
        if not bundles:
            logger.warning(f"MatrixEvalLogger: no complete {self.ern_name} matrix found in DB")
            return
        matrix_data = bundles[0]
        logger.info(
            f"MatrixEvalLogger: loaded {len(matrix_data)} networks for {self.ern_name} matrix"
        )

        # 2. Create NetworkPrediction with input_order=[[1, 0]]
        built_networks = [d.metadata["built_network"] for d in matrix_data]
        predict_at = [d.x for d in matrix_data]
        ground_truth = [d.y for d in matrix_data]

        nm = NetworkModel(
            model=model,
            network=built_networks,
        )

        pred = NetworkPrediction(
            input_order=[[1, 0]],
            predict_at=predict_at,
            ground_truth=ground_truth,
            network_model=nm,
            enable_gridstats=True,
            device="cpu",
        )
        prediction_data = pred.get_data()

        # 3. Render UORFMatrixFigure
        output_dir = Path(self._save_dir) / "matrix_eval" if self._save_dir else Path("matrix_eval")
        output_dir.mkdir(parents=True, exist_ok=True)

        annotation_tuples: list[tuple[int, int]] = [
            (pair[0], pair[1]) for pair in self.trained_uorf_annotation
        ]

        fig = UORFMatrixFigure(
            plot_data=prediction_data,
            plot_config=load_default_plotconf(),
            annotate=annotation_tuples,
            show_individual_rmse=self.show_individual_rmse,
            show_overall_rmse=self.show_overall_rmse,
            figure_spec={
                "title": f"Predicted {self.ern_name} matrix ({len(annotation_tuples)} training samples)",
                "output_file": f"{self.ern_name}_matrix_prediction.pdf",
                "output_dir": str(output_dir),
                "title_kwargs": {"fontsize": 25, "y": 1.01},
            },
            grid_plotconfigs=[
                {
                    "plot_config": {
                        "callstack_params": {
                            "smooth_2d_params": {
                                "knn_grid_params": {"grid_resolution": self.grid_resolution},
                            },
                        },
                    },
                }
            ],
        )
        fig.run()
        logger.info(f"MatrixEvalLogger: saved figure to {output_dir}")
