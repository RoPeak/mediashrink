from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
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

_GB = 1024**3
_MB = 1024**2


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
        show_estimates = any(job.dry_run and job.estimated_output_bytes > 0 for job in jobs)

        table = Table(
            title="Files found",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("File", style="white", no_wrap=False)
        table.add_column("Codec", style="dim cyan", justify="center", no_wrap=True)
        table.add_column("Size", style="yellow", justify="right", no_wrap=True)
        if show_estimates:
            table.add_column("Est. Output", style="green", justify="right", no_wrap=True)
            table.add_column("Est. Saving", style="bold green", justify="right", no_wrap=True)
        table.add_column("Action", justify="center", no_wrap=True)

        for job in jobs:
            size_str = _fmt_size(job.source.stat().st_size)
            codec_str = job.source_codec or "?"

            if job.skip:
                action = Text("SKIP", style="dim")
            elif job.dry_run:
                action = Text("DRY RUN", style="cyan")
            else:
                action = Text("ENCODE", style="green bold")

            filename = job.source.name
            if job.skip and job.skip_reason:
                filename += f"\n[dim]{job.skip_reason}[/dim]"

            row: list[str | Text] = [filename, codec_str, size_str]
            if show_estimates:
                if job.skip or job.estimated_output_bytes == 0:
                    row += ["-", "-"]
                else:
                    est_out = job.estimated_output_bytes
                    est_saved = job.source.stat().st_size - est_out
                    est_pct = (est_saved / job.source.stat().st_size * 100) if job.source.stat().st_size else 0
                    row += [_fmt_size(est_out), f"{_fmt_size(est_saved)} ({est_pct:.0f}%)"]
            row.append(action)
            table.add_row(*row)

        self.console.print()
        self.console.print(table)

        to_encode = sum(1 for job in jobs if not job.skip)
        to_skip = sum(1 for job in jobs if job.skip)
        total_input_bytes = sum(job.source.stat().st_size for job in jobs if not job.skip)
        total_est_bytes = sum(
            job.estimated_output_bytes for job in jobs if not job.skip and job.estimated_output_bytes > 0
        )

        summary = (
            f"[bold]{len(jobs)}[/bold] file(s) found - "
            f"[green bold]{to_encode}[/green bold] to encode, "
            f"[dim]{to_skip}[/dim] to skip  "
            f"([yellow]{_fmt_size(total_input_bytes)}[/yellow] total input)"
        )
        if show_estimates and total_est_bytes > 0:
            total_saving = total_input_bytes - total_est_bytes
            total_pct = total_saving / total_input_bytes * 100 if total_input_bytes else 0
            summary += (
                f"  -> est. [green]{_fmt_size(total_est_bytes)}[/green] output"
                f"  ([bold green]{_fmt_size(total_saving)} saved, ~{total_pct:.0f}%[/bold green])"
            )
        self.console.print(summary, highlight=False)
        self.console.print()

    def confirm_proceed(self, count: int) -> bool:
        return typer.confirm(f"Encode {count} file(s)?", default=True)

    def make_progress_bar(self) -> Progress:
        from rich.progress import DownloadColumn

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", table_column=None),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            DownloadColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )

    def show_summary(self, results: list[EncodeResult]) -> None:
        is_dry_run = any(result.job.dry_run for result in results)

        table = Table(
            title="Encoding summary" if not is_dry_run else "Dry-run summary (estimates)",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("File", style="white", no_wrap=False)
        table.add_column("Before", style="yellow", justify="right", no_wrap=True)
        table.add_column("After", style="green", justify="right", no_wrap=True)
        table.add_column("Saving", style="bold green", justify="right", no_wrap=True)
        table.add_column("Reduction", justify="right", no_wrap=True)
        if not is_dry_run:
            table.add_column("Time", justify="right", no_wrap=True)
            table.add_column("Speed", justify="right", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)

        total_input = 0
        total_output = 0
        total_saved = 0
        total_time = 0.0

        for result in results:
            if result.skipped:
                row: list[str | Text] = [result.job.source.name, _fmt_size(result.input_size_bytes), "-", "-", "-"]
                if not is_dry_run:
                    row += ["-", "-"]
                row.append(Text("SKIPPED", style="dim"))
                table.add_row(*row)
                continue

            if not result.success:
                row = [result.job.source.name, _fmt_size(result.input_size_bytes), "-", "-", "-"]
                if not is_dry_run:
                    row += [_fmt_duration(result.duration_seconds), "-"]
                row.append(Text("FAILED", style="red bold"))
                table.add_row(*row)
                continue

            if result.job.dry_run:
                est = result.job.estimated_output_bytes
                if est > 0:
                    est_saved = result.input_size_bytes - est
                    est_pct = est_saved / result.input_size_bytes * 100 if result.input_size_bytes else 0
                    pct_style = "green bold" if est_pct >= 30 else "yellow" if est_pct >= 10 else "red"
                    row = [
                        result.job.source.name,
                        _fmt_size(result.input_size_bytes),
                        f"~{_fmt_size(est)}",
                        f"~{_fmt_size(est_saved)}",
                        Text(f"~{est_pct:.0f}%", style=pct_style),
                        Text("DRY RUN", style="cyan"),
                    ]
                    total_input += result.input_size_bytes
                    total_output += est
                    total_saved += est_saved
                else:
                    row = [result.job.source.name, _fmt_size(result.input_size_bytes), "-", "-", "-", Text("DRY RUN", style="cyan")]
                table.add_row(*row)
                continue

            saved = result.size_reduction_bytes
            pct = result.size_reduction_pct
            pct_style = "green bold" if pct >= 30 else "yellow" if pct >= 10 else "red"

            speed_str = "-"
            if result.duration_seconds > 0 and result.media_duration_seconds > 0:
                speed_x = result.media_duration_seconds / result.duration_seconds
                speed_str = f"{speed_x:.2f}x"

            row = [
                result.job.source.name,
                _fmt_size(result.input_size_bytes),
                _fmt_size(result.output_size_bytes),
                _fmt_size(saved),
                Text(f"{pct:.1f}%", style=pct_style),
                _fmt_duration(result.duration_seconds),
                speed_str,
                Text("OK", style="green bold"),
            ]
            table.add_row(*row)

            total_input += result.input_size_bytes
            total_output += result.output_size_bytes
            total_saved += saved
            total_time += result.duration_seconds

        self.console.print()
        self.console.print(table)

        if total_input > 0:
            overall_pct = (total_saved / total_input) * 100
            est_marker = "~" if is_dry_run else ""
            self.console.print(
                f"\n[bold]Total:[/bold] "
                f"[yellow]{_fmt_size(total_input)}[/yellow] -> "
                f"[green]{est_marker}{_fmt_size(total_output)}[/green]  "
                f"([bold green]{est_marker}{_fmt_size(total_saved)} saved, {est_marker}{overall_pct:.1f}%[/bold green])"
                + (f"  [dim]{_fmt_duration(total_time)} elapsed[/dim]" if total_time > 0 else ""),
                highlight=False,
            )
        self.console.print()
