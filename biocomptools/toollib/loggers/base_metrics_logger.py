## {{{                          --     imports     --

from __future__ import annotations

from biocomptools.toollib.loggers.logger import Logger
from biocomptools.toollib.loggers.metrics_models import LoggerMetricsHistory, ReplicateMetrics
from biocomptools.toollib.loggers.plotting_utils import MetricsPlotter
from biocomptools.logger_history import HistoryView, LoggerContext
from biocomptools.logging_config import get_logger
from pathlib import Path
from typing import TYPE_CHECKING, Any
from pydantic import PrivateAttr
from rich.console import Console

if TYPE_CHECKING:
    from biocomptools.run_training import TrainingProgram

logger = get_logger(__name__)

##────────────────────────────────────────────────────────────────────────────}}}


class BaseMetricsLogger(Logger):
    """Base class for loggers that compute and optionally plot metrics."""

    name: str | None = None
    only_metrics: bool = False
    plot_at_the_end: bool = False
    save_plots: bool = True
    plot_dpi: int = 300
    plot_training_losses: bool = False
    required_arrays: list[str] = ["y", "yhat", "latest_params"]

    _training_program: TrainingProgram | None = PrivateAttr(default=None)
    _console: Console | None = PrivateAttr(default=None)
    _metrics_history: LoggerMetricsHistory | None = PrivateAttr(default=None)
    _plot_save_dir: Path | None = PrivateAttr(default=None)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._console = Console()

    # find_myself() inherited from Logger base class

    def initialize(self, training_program):
        """Initialize the logger with training program context."""
        if self.name is None:
            if training_program:
                idx = self.find_myself(training_program)
                self.name = f"{self.__class__.__name__.lower()}_{idx}"
            else:
                self.name = self.__class__.__name__.lower()

        if training_program:
            self._training_program = training_program
            if self.save_plots and not self.only_metrics:
                self._plot_save_dir = Path(training_program._save_dir) / f"plots/{self.name}"
                self._plot_save_dir.mkdir(exist_ok=True, parents=True)

        # Initialize metrics history
        self._metrics_history = LoggerMetricsHistory(
            logger_name=self.name, logger_type=self.__class__.__name__
        )

        # We'll use MetricsPlotter as a static class, no need to instantiate

        self.metadata = {
            "logger_name": self.name,
            "logger_type": self.__class__.__name__,
            "only_metrics": self.only_metrics,
            "plot_at_the_end": self.plot_at_the_end,
        }

        logger.info(f"{self.__class__.__name__} {self.name}: Initialized")

    def on_batch(self, view: HistoryView, context: LoggerContext) -> None:
        step_history = view.to_step_history()
        if "y" not in step_history or "yhat" not in step_history:
            if context.current_step > 0:
                logger.debug(
                    f"{type(self).__name__} {self.name}: y/yhat not available at step {context.current_step}"
                )
            return
        training_loss = step_history.get("loss")
        self._log_metrics_step(context.current_step, step_history, training_loss)

    def on_end(self, view: HistoryView, context: LoggerContext) -> None:
        self.on_batch(view, context)

    def _compute_metrics(self, step_data: dict[str, Any]) -> list[ReplicateMetrics]:
        """Override this method to compute metrics from step data."""
        raise NotImplementedError("Subclasses must implement _compute_metrics")

    def _print_metrics(self, step: int, metrics: list[ReplicateMetrics]):
        """Override this method to print metrics in a formatted way."""
        raise NotImplementedError("Subclasses must implement _print_metrics")

    def get_metrics(self, replicate: int | None = None) -> dict[str, Any] | None:
        """Return the latest metrics."""
        if self._metrics_history is None:
            return None
        return self._metrics_history.get_latest_metrics(replicate)

    def _should_plot_now(self, step: int) -> bool:
        """Determine if we should plot at this step."""
        if self.only_metrics:
            return False
        if self.plot_at_the_end:
            return False
        return True

    def _log_metrics_step(self, step: int, step_data: dict[str, Any], training_loss: Any = None):
        """Common logic for logging metrics at a step."""
        try:
            # Compute metrics
            metrics = self._compute_metrics(step_data)

            # Store in history
            if self._metrics_history is not None:
                self._metrics_history.add_step_metrics(step, metrics, training_loss)

            # Print metrics
            self._print_metrics(step, metrics)

            # Plot if needed
            if self._should_plot_now(step) and self.save_plots and not self.only_metrics:
                # Ensure plot save directory exists (for replay mode or standalone usage)
                if self._plot_save_dir is None:
                    self._plot_save_dir = Path.cwd() / f"plots/{self.name}"
                    self._plot_save_dir.mkdir(exist_ok=True, parents=True)

                output_path = self._plot_save_dir / f"{self.name}_{step:05d}.png"
                MetricsPlotter.plot_metrics_history(
                    self._metrics_history.history,
                    f"{self.__class__.__name__}",
                    output_path,
                    self.name,
                )

        except Exception as e:
            logger.error(f"{self.__class__.__name__} {self.name} failed at step {step}: {e}")
            logger.exception(e)

    def finalize(self):
        """Create final plots if plot_at_the_end is True."""
        if (
            self.plot_at_the_end
            and self._metrics_history is not None
            and len(self._metrics_history.history) > 0
        ):
            logger.info(f"{self.__class__.__name__} {self.name}: Creating final plot...")
            output_path = self._plot_save_dir / f"{self.name}_final.png"
            MetricsPlotter.plot_metrics_history(
                self._metrics_history.history, f"{self.__class__.__name__}", output_path, self.name
            )

        if self._metrics_history is not None:
            logger.info(
                f"{self.__class__.__name__} {self.name}: "
                f"Training completed with {len(self._metrics_history.history)} logged steps"
            )
