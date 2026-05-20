# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Rich training logger for hyperopt with live progress bar and loss display."""

import numpy as np
from typing import Callable, List, Tuple
from biocomptools.toollib.loggers.logger import Logger


class HyperoptTrainingLogger(Logger):
    """Live training progress logger for hyperopt using rich."""

    call_at_interval: int = 1  # log every step
    execution_mode: str = "inline"  # must be inline for live display
    n_replicates: int = 4

    def __init__(self, n_replicates: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_replicates = n_replicates
        self._progress = None
        self._task = None
        self._live = None
        self._total_steps = None
        self._current_step = 0

    def get_callbacks(self, training_program) -> List[Tuple[int, Callable]]:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        from rich.live import Live
        from rich.table import Table
        from rich.console import Group
        from rich.panel import Panel

        # store for later use
        self._Progress = Progress
        self._SpinnerColumn = SpinnerColumn
        self._BarColumn = BarColumn
        self._TextColumn = TextColumn
        self._TimeElapsedColumn = TimeElapsedColumn
        self._Live = Live
        self._Table = Table
        self._Group = Group
        self._Panel = Panel

        def start_callback(step, training_config, step_history=None, stack=None, **kwargs):
            """Initialize progress display at step 0."""
            if step != 0:
                return

            self._total_steps = int(
                training_config.n_epochs * training_config.n_batches / training_config.batches_per_step
            )

            self._progress = self._Progress(
                self._SpinnerColumn(),
                self._TextColumn("[bold blue]{task.description}"),
                self._BarColumn(bar_width=40),
                self._TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                self._TextColumn("•"),
                self._TimeElapsedColumn(),
                self._TextColumn("•"),
                self._TextColumn("[cyan]{task.fields[loss_info]}"),
            )
            self._task = self._progress.add_task(
                f"Training ({self.n_replicates} trials)",
                total=self._total_steps,
                loss_info="starting...",
            )
            self._progress.start()
            self._current_step = 0

        def update_callback(step, training_config, step_history=None, stack=None, **kwargs):
            """Update progress bar with current loss."""
            if self._progress is None or step == 0:
                return

            self._current_step = step
            loss_info = "..."

            if step_history is not None:
                losses = step_history.get('loss')
                if losses is not None:
                    losses = np.asarray(losses)
                    # losses shape: (n_replicates, n_batches_per_step)
                    mean_per_rep = np.mean(losses, axis=1)  # per replicate mean
                    overall_mean = np.mean(mean_per_rep)
                    best = np.min(mean_per_rep)
                    worst = np.max(mean_per_rep)
                    loss_info = f"loss: {overall_mean:.4f} (best={best:.4f}, worst={worst:.4f})"

            self._progress.update(self._task, completed=step, loss_info=loss_info)

        def end_callback(step, training_config, step_history=None, stack=None, **kwargs):
            """Finalize progress display."""
            if self._progress is not None:
                # Final update
                if step_history is not None:
                    losses = step_history.get('loss')
                    if losses is not None:
                        losses = np.asarray(losses)
                        mean_per_rep = np.mean(losses, axis=1)
                        overall_mean = np.mean(mean_per_rep)
                        best = np.min(mean_per_rep)
                        loss_info = f"final: {overall_mean:.4f} (best={best:.4f})"
                        self._progress.update(
                            self._task, completed=self._total_steps, loss_info=loss_info
                        )
                self._progress.stop()
                self._progress = None

        return [
            (0, start_callback),  # run at start
            (1, update_callback),  # run every step
            (-1, end_callback),  # run at end
        ]
