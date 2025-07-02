import asyncio
import hashlib
import signal
import subprocess
import threading
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.events import Resize
from textual.widgets import Footer, Header, ProgressBar, Static


class JobStatus(Enum):
    """Lifecycle state for each job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


STATUS_DETAILS: Dict[JobStatus, Dict[str, Any]] = {
    JobStatus.PENDING: {"icon": "-", "text": "PENDING", "color": "grey50"},
    JobStatus.RUNNING: {"icon": ">", "text": "RUNNING", "color": "blue"},
    JobStatus.SUCCESS: {"icon": "✓", "text": "SUCCESS", "color": "green"},
    JobStatus.FAILED: {"icon": "X", "text": "FAILED", "color": "red"},
    JobStatus.CANCELLED: {"icon": "🚫", "text": "CANCELLED", "color": "yellow"},
}


@dataclass
class JobResult:
    """Full result for a command execution."""

    command: List[str]
    status: JobStatus
    log_file: Path
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class JobCard(Container):
    """Card-style widget to display job state and details."""

    DEFAULT_CSS = """
    JobCard {
        width: 1fr;            /* take full column */
        height: auto;          /* intrinsic height (no compression) */
        border: round $secondary;
        padding: 1;
        margin-bottom: 1;
        transition: border 500ms;
    }

    .details-text {
        color: $text-muted;
    }

    JobCard.pending   { border: round $secondary-lighten-2; }
    JobCard.running   { border: round $primary; }
    JobCard.success   { border: round $success; }
    JobCard.failed    { border: round $error; }
    JobCard.cancelled { border: round $warning; }
    """

    def __init__(self, job_id: int, command: List[str]):
        super().__init__()
        self.job_id = job_id
        self.command_str = " ".join(command)
        self.status: JobStatus = JobStatus.PENDING

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(classes="status-line")
        yield Static(classes="command-text")
        yield Static(classes="details-text")

    def on_mount(self) -> None:
        self.query_one(".command-text", Static).update(Text(self.command_str, overflow="fold"))
        self.set_status(JobStatus.PENDING)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_status(
        self,
        status: JobStatus,
        result: Optional[JobResult] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        self.status = status
        meta = STATUS_DETAILS[status]

        # Update status line
        status_widget = self.query_one(".status-line", Static)
        txt = Text(f"{meta['icon']} {meta['text']}")
        txt.stylize(meta["color"])
        status_widget.update(txt)

        # Border colour via classes
        for s in JobStatus:
            self.remove_class(s.value)
        self.add_class(status.value)

        # Details line
        details_widget = self.query_one(".details-text", Static)
        if status == JobStatus.RUNNING and log_file is not None:
            details_widget.update(Text(f"Log: {log_file.name}", overflow="fold"))
        elif result is not None:
            detail = Text(f"Log: {result.log_file.name}", overflow="fold")
            detail.append(f"\nExit: {result.exit_code} | Duration: {result.duration:.2f}s")
            details_widget.update(detail)
        else:
            details_widget.update("Waiting to start…")


class CommandRunnerApp(App[Tuple[int, int]]):
    """Interactive TUI that runs commands and displays live status."""

    BINDINGS = [Binding("ctrl+c", "quit", "Quit", show=True)]

    CSS = """
    #main-container {
        layout: vertical;
        padding: 0 1;
    }

    #summary-panel {
        height: 11;
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }

    /* Scrollable area for job grid */
    VerticalScroll#job-scroll {
        height: 1fr;   /* take remaining space */
        min-height: 1fr;
    }

    #job-grid {
        layout: grid;
        grid-gutter: 1;
        grid-size-rows: 4;   /* prevent row compression */
    }
    """

    CARD_WIDTH = 50  # width of one JobCard (approx.)
    GUTTER = 1  # same as grid-gutter

    def __init__(self, runner: "CommandRunner") -> None:
        super().__init__()
        self.runner = runner
        self.start_time = datetime.now()
        self.job_cards: Dict[int, JobCard] = {}
        self.results: Dict[int, JobResult] = {}
        self._cancel = threading.Event()
        self.executor: Optional[futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        # yield Header(show_clock=True)
        with Container(id="main-container"):
            yield Static(id="summary-panel")
            yield ProgressBar(total=len(self.runner.commands), id="overall-progress")
            with VerticalScroll(id="job-scroll"):
                with Container(id="job-grid"):
                    for idx, cmd in enumerate(self.runner.commands):
                        yield JobCard(idx, cmd)
        # yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_mount(self) -> None:
        self.job_cards = {card.job_id: card for card in self.query(JobCard)}
        self.call_after_refresh(self._update_grid)
        self._refresh_summary()

        # Spawn worker thread for command execution
        self.run_worker(self._execute_commands, name="executor", thread=True)

    def on_resize(self, event: Resize) -> None:  # noqa: D401 – signature set by Textual
        self._update_grid()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _update_grid(self) -> None:
        grid = self.query_one("#job-grid", Container)
        num_cols = max(1, self.size.width // (self.CARD_WIDTH + self.GUTTER))
        grid.styles.grid_size_columns = num_cols
        grid.styles.grid_size_rows = 4

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------
    def _refresh_summary(self) -> None:
        tbl = Table(box=None, expand=True, show_header=False, padding=0)
        tbl.add_column(style="bold cyan")
        tbl.add_column()

        counts = {s: 0 for s in JobStatus}
        for card in self.job_cards.values():
            counts[card.status] += 1

        elapsed = (datetime.now() - self.start_time).total_seconds()
        tbl.add_row("Total Jobs", str(len(self.runner.commands)))
        tbl.add_row("Success", f"[green]{counts[JobStatus.SUCCESS]}[/]")
        tbl.add_row("Failed", f"[red]{counts[JobStatus.FAILED]}[/]")
        tbl.add_row("Running", f"[blue]{counts[JobStatus.RUNNING]}[/]")
        tbl.add_row("Pending", f"[grey50]{counts[JobStatus.PENDING]}[/]")
        tbl.add_row("Elapsed Time", f"{int(elapsed // 60):02}:{int(elapsed % 60):02}")

        self.query_one("#summary-panel", Static).update(
            Panel(tbl, title="[bold]Run Summary[/]", border_style="blue")
        )

    def action_quit(self):
        self.notify("Cancelling…", severity="warning", timeout=1)
        self._cancel.set()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.exit()

    def _handle_result(self, job_id: int, result: JobResult) -> None:
        self.results[job_id] = result
        self.job_cards[job_id].set_status(result.status, result=result)
        self.query_one(ProgressBar).advance()
        self._refresh_summary()

    # ------------------------------------------------------------------
    # Background execution
    # ------------------------------------------------------------------
    def _execute_commands(self) -> None:
        with futures.ThreadPoolExecutor(max_workers=self.runner.max_parallel_jobs) as pool:
            self.executor = pool
            in_flight: Dict[futures.Future, int] = {}
            cmd_iter = iter(enumerate(self.runner.commands))

            while not self._cancel.is_set():
                # Fill executor with work
                while len(in_flight) < self.runner.max_parallel_jobs:
                    try:
                        job_id, cmd = next(cmd_iter)
                    except StopIteration:
                        break

                    log_file = self.runner._get_log_filename(cmd)
                    fut = pool.submit(self.runner._run_single_command, cmd, log_file)
                    in_flight[fut] = job_id
                    self.call_from_thread(
                        self.job_cards[job_id].set_status,
                        JobStatus.RUNNING,
                        log_file=log_file,
                    )
                    self.call_from_thread(self._refresh_summary)

                if not in_flight:
                    break

                done, _ = futures.wait(
                    in_flight.keys(), timeout=0.5, return_when=futures.FIRST_COMPLETED
                )

                for fut in done:
                    job_id = in_flight.pop(fut)
                    try:
                        result = fut.result()
                        # Notify UI thread about completion
                        self.call_from_thread(self._handle_result, job_id, result)
                    except futures.CancelledError:
                        # Graceful shutdown – ignore
                        pass
                    except Exception as exc:
                        # Convert unexpected exception into a failure result so UI remains consistent
                        if not self._cancel.is_set():
                            log_file = self.runner._get_log_filename(self.runner.commands[job_id])
                            result = self.runner._create_failure_result(
                                self.runner.commands[job_id], exc, log_file
                            )
                            self.call_from_thread(self._handle_result, job_id, result)

        # All tasks complete or cancelled – finalise run
        self.call_from_thread(self._finish_run)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------
    def _finish_run(self) -> None:
        """Update UI once all jobs have finished (or we were cancelled)."""
        if self._cancel.is_set():
            # mark any still‑pending/running jobs as cancelled
            for card in self.job_cards.values():
                if card.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    card.set_status(JobStatus.CANCELLED)
            self._refresh_summary()
            return

        success = sum(1 for r in self.results.values() if r.status == JobStatus.SUCCESS)
        failed = sum(1 for r in self.results.values() if r.status == JobStatus.FAILED)

        self.notify(f"Run finished: {success} success, {failed} failed.", timeout=5)

        async def _exit_later():
            await asyncio.sleep(2)
            self.exit((success, failed))

        self.call_after_refresh(_exit_later)


class CommandRunner(BaseModel):
    """Encapsulates the mechanics of launching shell commands and logging output."""

    log_dir: Path = Field(default=Path("./command_logs"))
    max_parallel_jobs: int = Field(default=8, gt=0)
    commands: List[List[str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("log_dir")
    def _ensure_log_dir(cls, v: Path) -> Path:  # noqa: D401
        v.mkdir(exist_ok=True, parents=True)
        return v

    def _get_log_filename(self, command: List[str]) -> Path:
        cmd_hash = hashlib.md5(" ".join(command).encode()).hexdigest()[:8]
        safe_cmd = "".join(c if c.isalnum() else "_" for c in command[0])[:30]
        return self.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{safe_cmd}_{cmd_hash}.log"

    def _write_log(self, result: JobResult) -> None:
        content = f"""
Command: {' '.join(result.command)}
Status: {result.status.value.upper()}
Exit Code: {result.exit_code}
Start Time: {result.start_time}
End Time: {result.end_time}
Duration: {result.duration:.2f}s

--- STDOUT ---
{result.stdout}
--- STDERR ---
{result.stderr}
""".strip()
        result.log_file.write_text(content)

    def _create_failure_result(
        self, command: List[str], exc: Exception, log_file: Path
    ) -> JobResult:
        result_args = {
            "command": command,
            "status": JobStatus.FAILED,
            "log_file": log_file,
            "start_time": datetime.now(),
            "end_time": datetime.now(),
        }
        if isinstance(exc, subprocess.CalledProcessError):
            result_args.update(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=exc.returncode,
            )
        else:
            result_args.update(stderr=str(exc), exit_code=-1)

        result = JobResult(**result_args)
        self._write_log(result)
        return result

    def _run_single_command(self, command: List[str], log_file: Path) -> JobResult:
        start_time = datetime.now()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=True)
            result = JobResult(
                command=command,
                status=JobStatus.SUCCESS,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                log_file=log_file,
            )
        except Exception as exc:
            return self._create_failure_result(command, exc, log_file)

        result.start_time = start_time
        result.end_time = datetime.now()
        result.duration = (result.end_time - start_time).total_seconds()
        self._write_log(result)
        return result

    def run(self) -> Tuple[int, int]:
        """Launch the TUI application and return (success_count, failure_count)."""
        if not self.commands:
            print("No commands to run.")
            return (0, 0)

        # Allow Ctrl‑C to bubble up into Textual cleanly
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            app = CommandRunnerApp(self)
            final_counts = app.run()
            return final_counts or (0, 0)
        finally:
            signal.signal(signal.SIGINT, original_sigint)


if __name__ == "__main__":
    COMMANDS = [
        ["sleep", "2"],
        ["echo", "Fast job 1"],
        ["sleep", "3"],
        ["date"],
        ["ls", "-la", "/var/log"],
        ["echo", "This is a slightly longer command that might need to wrap onto a new line"],
        ["sleep", "1"],
        ["false"],
        ["echo", "Job after failure"],
        ["this-is-not-a-command"],
        ["sleep", "1"],
        ["uname", "-a"],
        ["echo", "one more"],
        ["sleep", "4"],
        ["echo", "and another"],
        ["find", "/", "-name", "private", "-type", "d", "-print", "-exec", "sleep", "0.1", ";"],
    ] * 2

    runner = CommandRunner(commands=COMMANDS, max_parallel_jobs=6, log_dir=Path("./runner_logs"))
    success, failed = runner.run()

    print("\n" + "=" * 40)
    print("  Execution complete.")
    print(f"  {success} successful jobs, {failed} failed jobs.")
    print(f"  Logs available in: {runner.log_dir.resolve()}")
    print("=" * 40)
