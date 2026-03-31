from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from mkv_compress.models import EncodeJob, EncodeResult

_GB = 1024 ** 3
_MB = 1024 ** 2


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.2f} GB"
    return f"{size_bytes / _MB:.1f} MB"


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class EncodingDisplay:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def show_scan_table(self, jobs: list[EncodeJob]) -> None:
        """Display a table of files found and the planned action for each."""
        table = Table(
            title="Files found",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("File", style="white", no_wrap=False)
        table.add_column("Size", style="yellow", justify="right", no_wrap=True)
        table.add_column("Output", style="dim white", no_wrap=False)
        table.add_column("Action", justify="center", no_wrap=True)

        for job in jobs:
            size_str = _fmt_size(job.source.stat().st_size)
            output_name = job.output.name

            if job.skip:
                action = Text("SKIP", style="dim")
                reason = f"  [dim]({job.skip_reason})[/dim]"
            elif job.dry_run:
                action = Text("DRY RUN", style="cyan")
                reason = ""
            else:
                action = Text("ENCODE", style="green bold")
                reason = ""

            table.add_row(
                job.source.name + (f"\n[dim]{job.skip_reason}[/dim]" if job.skip else ""),
                size_str,
                output_name,
                action,
            )

        self.console.print()
        self.console.print(table)

        to_encode = sum(1 for j in jobs if not j.skip)
        to_skip = sum(1 for j in jobs if j.skip)
        total_input_bytes = sum(j.source.stat().st_size for j in jobs if not j.skip)

        self.console.print(
            f"[bold]{len(jobs)}[/bold] file(s) found — "
            f"[green bold]{to_encode}[/green bold] to encode, "
            f"[dim]{to_skip}[/dim] to skip  "
            f"([yellow]{_fmt_size(total_input_bytes)}[/yellow] total input)"
        )
        self.console.print()

    def confirm_proceed(self, count: int) -> bool:
        """Prompt the user to confirm encoding. Returns True if confirmed."""
        return typer.confirm(
            f"Encode {count} file(s)?",
            default=True,
        )

    def make_progress_bar(self) -> Progress:
        """Return a Rich Progress bar for per-file encoding."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    def show_summary(self, results: list[EncodeResult]) -> None:
        """Display a before/after summary table across all processed files."""
        table = Table(
            title="Encoding summary",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("File", style="white", no_wrap=False)
        table.add_column("Before", style="yellow", justify="right", no_wrap=True)
        table.add_column("After", style="green", justify="right", no_wrap=True)
        table.add_column("Saved", style="bold green", justify="right", no_wrap=True)
        table.add_column("Reduction", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)

        total_input = 0
        total_output = 0
        total_saved = 0
        total_time = 0.0

        for r in results:
            if r.skipped:
                table.add_row(
                    r.job.source.name,
                    _fmt_size(r.input_size_bytes),
                    "—",
                    "—",
                    "—",
                    "—",
                    Text("SKIPPED", style="dim"),
                )
                continue

            if not r.success:
                table.add_row(
                    r.job.source.name,
                    _fmt_size(r.input_size_bytes),
                    "—",
                    "—",
                    "—",
                    _fmt_duration(r.duration_seconds),
                    Text("FAILED", style="red bold"),
                )
                continue

            if r.job.dry_run:
                table.add_row(
                    r.job.source.name,
                    _fmt_size(r.input_size_bytes),
                    "—",
                    "—",
                    "—",
                    "—",
                    Text("DRY RUN", style="cyan"),
                )
                continue

            saved = r.size_reduction_bytes
            pct = r.size_reduction_pct
            pct_str = f"{pct:.1f}%"
            pct_style = "green bold" if pct >= 30 else "yellow" if pct >= 10 else "red"

            table.add_row(
                r.job.source.name,
                _fmt_size(r.input_size_bytes),
                _fmt_size(r.output_size_bytes),
                _fmt_size(saved),
                Text(pct_str, style=pct_style),
                _fmt_duration(r.duration_seconds),
                Text("OK", style="green bold"),
            )

            total_input += r.input_size_bytes
            total_output += r.output_size_bytes
            total_saved += saved
            total_time += r.duration_seconds

        self.console.print()
        self.console.print(table)

        if total_input > 0:
            overall_pct = (total_saved / total_input) * 100
            self.console.print(
                f"\n[bold]Total:[/bold] "
                f"[yellow]{_fmt_size(total_input)}[/yellow] → "
                f"[green]{_fmt_size(total_output)}[/green]  "
                f"([bold green]{_fmt_size(total_saved)} saved, {overall_pct:.1f}%[/bold green])  "
                f"[dim]{_fmt_duration(total_time)} elapsed[/dim]"
            )
        self.console.print()
