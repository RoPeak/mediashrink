from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    ProgressColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from mediashrink.models import EncodeJob, EncodeResult

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


class _CompletedSizeColumn(ProgressColumn):
    def render(self, task) -> Text:
        total = int(task.total or 0)
        completed = min(int(task.completed or 0), total) if total > 0 else int(task.completed or 0)
        if total > 0:
            return Text(f"{_fmt_size(completed)}/{_fmt_size(total)}", style="progress.download")
        return Text(_fmt_size(completed), style="progress.download")


class _EtaColumn(ProgressColumn):
    def render(self, task) -> Text:
        remaining = task.time_remaining
        if remaining is None:
            return Text("ETA unavailable", style="progress.remaining")
        return Text(_fmt_duration(remaining), style="progress.remaining")


class _FileCountsColumn(ProgressColumn):
    def render(self, task) -> Text:
        completed_files = task.fields.get("completed_files")
        remaining_files = task.fields.get("remaining_files")
        if completed_files is None or remaining_files is None:
            return Text("-", style="dim")
        return Text(f"{completed_files} done / {remaining_files} left", style="dim")


class _HeartbeatColumn(ProgressColumn):
    def render(self, task) -> Text:
        last_update_at = task.fields.get("last_update_at")
        stall_warning_seconds = task.fields.get("stall_warning_seconds")
        if not isinstance(last_update_at, (int, float)):
            return Text("-", style="dim")
        if not isinstance(stall_warning_seconds, (int, float)):
            stall_warning_seconds = 90.0
        idle_for = max(time.monotonic() - last_update_at, 0.0)
        if idle_for >= stall_warning_seconds:
            return Text("stalled warning", style="yellow")
        if idle_for >= min(15.0, stall_warning_seconds / 2):
            return Text("quiet but alive", style="cyan")
        return Text("active", style="green")


class _LastUpdateColumn(ProgressColumn):
    def render(self, task) -> Text:
        last_update_at = task.fields.get("last_update_at")
        if not isinstance(last_update_at, (int, float)):
            return Text("-", style="dim")
        idle_for = max(time.monotonic() - last_update_at, 0.0)
        return Text(f"{int(idle_for)}s ago", style="dim")


def _is_preview_result(result: EncodeResult) -> bool:
    return result.job.output.stem.endswith("_preview")


def _preview_duration_note(results: list[EncodeResult]) -> str | None:
    preview_durations = [
        result.media_duration_seconds for result in results if result.media_duration_seconds > 0
    ]
    if (
        not results
        or not all(_is_preview_result(result) for result in results)
        or not preview_durations
    ):
        return None
    return _fmt_duration(min(preview_durations))


def _skip_bucket(result: EncodeResult) -> str:
    reason = result.skip_reason or ""
    if reason.startswith("incompatible:"):
        return "incompatible"
    if reason.startswith("skipped_by_policy:"):
        return "policy"
    return "other"


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
                    est_pct = (
                        (est_saved / job.source.stat().st_size * 100)
                        if job.source.stat().st_size
                        else 0
                    )
                    row += [_fmt_size(est_out), f"{_fmt_size(est_saved)} ({est_pct:.0f}%)"]
            row.append(action)
            table.add_row(*row)

        self.console.print()
        self.console.print(table)

        to_encode = sum(1 for job in jobs if not job.skip)
        to_skip = sum(1 for job in jobs if job.skip)
        total_input_bytes = sum(job.source.stat().st_size for job in jobs if not job.skip)
        total_est_bytes = sum(
            job.estimated_output_bytes
            for job in jobs
            if not job.skip and job.estimated_output_bytes > 0
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
        width = self.console.width
        bar_width = max(20, min(40, width // 4))
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", table_column=None),
            BarColumn(bar_width=bar_width),
            TaskProgressColumn(),
            _FileCountsColumn(),
            _HeartbeatColumn(),
            _LastUpdateColumn(),
            _CompletedSizeColumn(),
            TimeElapsedColumn(),
            _EtaColumn(),
            console=self.console,
            transient=False,
            expand=True,
            disable=not self.console.is_terminal,
        )

    def show_summary(
        self,
        results: list[EncodeResult],
        *,
        resumed_from_session: bool = False,
        previously_completed: int = 0,
        previously_skipped: int = 0,
    ) -> None:
        is_dry_run = any(result.job.dry_run for result in results)
        is_preview = bool(results) and all(_is_preview_result(result) for result in results)
        preview_duration = _preview_duration_note(results)

        table = Table(
            title=(
                "Preview summary"
                if is_preview
                else "Encoding summary"
                if not is_dry_run
                else "Dry-run summary (estimates)"
            ),
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("File", style="white", no_wrap=False)
        table.add_column("Before", style="yellow", justify="right", no_wrap=True)
        table.add_column(
            "Preview clip" if is_preview else "After",
            style="green",
            justify="right",
            no_wrap=True,
        )
        table.add_column("Saving", style="bold green", justify="right", no_wrap=True)
        table.add_column("Reduction", justify="right", no_wrap=True)
        if not is_dry_run:
            table.add_column("Time", justify="right", no_wrap=True)
            table.add_column("Speed", justify="right", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)

        successful_results = [result for result in results if result.success and not result.skipped]
        failed_results = [result for result in results if not result.success and not result.skipped]
        skipped_results = [result for result in results if result.skipped]
        skipped_incompatible = [
            result for result in skipped_results if _skip_bucket(result) == "incompatible"
        ]
        skipped_by_policy = [
            result for result in skipped_results if _skip_bucket(result) == "policy"
        ]
        skipped_other = [result for result in skipped_results if _skip_bucket(result) == "other"]
        total_input = 0
        total_output = 0
        total_saved = 0
        total_time = 0.0

        for result in successful_results + failed_results + skipped_results:
            if result.skipped:
                row: list[str | Text] = [
                    result.job.source.name,
                    _fmt_size(result.input_size_bytes),
                    "-",
                    "-",
                    "-",
                ]
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
                    est_pct = (
                        est_saved / result.input_size_bytes * 100 if result.input_size_bytes else 0
                    )
                    pct_style = (
                        "green bold" if est_pct >= 30 else "yellow" if est_pct >= 10 else "red"
                    )
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
                    row = [
                        result.job.source.name,
                        _fmt_size(result.input_size_bytes),
                        "-",
                        "-",
                        "-",
                        Text("DRY RUN", style="cyan"),
                    ]
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
        if resumed_from_session:
            self.console.print(
                f"[dim]Resumed run:[/dim] {previously_completed} file(s) were already complete"
                + (f", {previously_skipped} already skipped" if previously_skipped > 0 else "")
                + " before this run started.",
                highlight=False,
            )
        self.console.print(
            f"[bold]{len(successful_results)}[/bold] succeeded, "
            f"[red bold]{len(failed_results)}[/red bold] failed, "
            f"[dim]{len(skipped_results)}[/dim] skipped",
            highlight=False,
        )
        if skipped_results:
            details: list[str] = []
            if skipped_incompatible:
                details.append(f"{len(skipped_incompatible)} incompatible")
            if skipped_by_policy:
                details.append(f"{len(skipped_by_policy)} skipped by policy")
            if skipped_other:
                details.append(f"{len(skipped_other)} other skipped")
            self.console.print(
                f"[dim]Skip breakdown:[/dim] {', '.join(details)}",
                highlight=False,
            )
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

        # Show estimate vs actual when we have both and it's a real (non-dry-run) batch encode.
        if not is_dry_run and not is_preview and successful_results:
            est_output_total = sum(
                r.job.estimated_output_bytes
                for r in successful_results
                if r.job.estimated_output_bytes > 0
            )
            est_input_total = sum(
                r.input_size_bytes for r in successful_results if r.job.estimated_output_bytes > 0
            )
            if est_output_total > 0 and est_input_total > 0 and total_input > 0:
                est_saved = est_input_total - est_output_total
                est_pct = est_saved / est_input_total * 100
                self.console.print(
                    f"[dim]Estimated saving: ~{_fmt_size(est_saved)} ({est_pct:.0f}%)  "
                    f"→  Actual saving: {_fmt_size(total_saved)} ({overall_pct:.1f}%)[/dim]",
                    highlight=False,
                )

        if is_preview:
            clip_label = preview_duration or "this sample"
            self.console.print(
                f"[dim]Preview results cover only {clip_label}; size and speed are not representative of the full encode.[/dim]"
            )
        if failed_results:
            self.console.print("\n[bold red]Failure details[/bold red]")
            for result in failed_results:
                message = result.error_message or "unknown ffmpeg error"
                self.console.print(f"  [red]-[/red] {result.job.source.name}: {message}")
        if skipped_incompatible:
            self.console.print("\n[bold yellow]Incompatible files skipped[/bold yellow]")
            for result in skipped_incompatible:
                self.console.print(
                    f"  [yellow]-[/yellow] {result.job.source.name}: {result.skip_reason or 'incompatible with current output settings'}"
                )
        self.console.print()
