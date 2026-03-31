from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from mkv_compress.encoder import encode_file
from mkv_compress.models import EncodeJob
from mkv_compress.platform_utils import check_ffmpeg_available, find_ffmpeg, find_ffprobe
from mkv_compress.profiles import delete_profile, get_profile, load_profiles
from mkv_compress.progress import EncodingDisplay
from mkv_compress.scanner import build_jobs, scan_directory


class DefaultCommandGroup(TyperGroup):
    """Route bare `mkvcompress ...` invocations to the hidden encode command."""

    default_command_name = "encode"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in {"--help", "-h"}:
            args.insert(0, self.default_command_name)
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="mkvcompress",
    help="Re-encode MKV files to H.265/HEVC to reduce file size.",
    add_completion=False,
    cls=DefaultCommandGroup,
)
profiles_app = typer.Typer(help="Manage saved encoding profiles.")
app.add_typer(profiles_app, name="profiles")

console = Console()


def _run_encode_loop(
    jobs: list[EncodeJob],
    ffmpeg: Path,
    ffprobe: Path,
    display: EncodingDisplay,
) -> None:
    to_encode = [job for job in jobs if not job.skip]
    total_bytes = sum(job.source.stat().st_size for job in to_encode)
    results = []
    bytes_done = 0

    with display.make_progress_bar() as progress:
        overall_task = progress.add_task(
            f"[cyan]Overall ({len(to_encode)} file(s))",
            total=total_bytes,
        )
        file_task = progress.add_task("", total=100)

        for job in jobs:
            filename = job.source.name
            file_size = job.source.stat().st_size
            progress.update(file_task, description=f"[white]{filename}", completed=0)

            def make_callback(ft=file_task, fb=file_size, ot=overall_task, bd=bytes_done):
                def callback(pct: float) -> None:
                    progress.update(ft, completed=pct)
                    progress.update(ot, completed=bd + fb * pct / 100)

                return callback

            try:
                result = encode_file(
                    job,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    progress_callback=make_callback() if not job.skip else None,
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                sys.exit(1)

            results.append(result)

            if not job.skip:
                bytes_done += file_size
                progress.update(overall_task, completed=bytes_done)
                progress.update(file_task, completed=100)

    display.show_summary(results)


def _prepare_tools(output_dir: Optional[Path]) -> tuple[Path, Path]:
    ok, err = check_ffmpeg_available()
    if not ok:
        console.print(f"[red bold]Error:[/red bold] {err}")
        raise typer.Exit(code=1)

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    return ffmpeg, ffprobe


@app.command("encode", hidden=True)
def encode_cmd(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing .mkv files to compress.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        help="H.265 CRF quality value (0-51, lower = better quality). Default: 20.",
        min=0,
        max=51,
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Encoding preset. Software: ultrafast/faster/fast/medium/slow. "
            "Hardware (much faster): qsv (Intel), nvenc (Nvidia), amf (AMD). "
            "Default: fast."
        ),
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load saved CRF/preset defaults from a named profile.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Scan subdirectories for .mkv files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be encoded without actually encoding.",
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    display = EncodingDisplay(console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)

    effective_crf = 20
    effective_preset = "fast"
    if profile:
        saved_profile = get_profile(profile)
        if saved_profile is None:
            console.print(f"[red bold]Error:[/red bold] profile '{profile}' was not found.")
            raise typer.Exit(code=1)
        effective_crf = saved_profile.crf
        effective_preset = saved_profile.preset

    if crf is not None:
        effective_crf = crf
    if preset is not None:
        effective_preset = preset

    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(f"[yellow]No .mkv files found in[/yellow] {directory}")
        raise typer.Exit(code=0)

    jobs = build_jobs(
        files=files,
        output_dir=output_dir,
        overwrite=overwrite,
        crf=effective_crf,
        preset=effective_preset,
        dry_run=dry_run,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )

    display.show_scan_table(jobs)

    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        console.print("[dim]Nothing to encode.[/dim]")
        raise typer.Exit(code=0)

    if not dry_run and not yes:
        if not display.confirm_proceed(len(to_encode)):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=0)

    _run_encode_loop(jobs, ffmpeg, ffprobe, display)


@app.command()
def wizard(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing .mkv files to compress.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Scan subdirectories for .mkv files.",
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
) -> None:
    """Interactively detect hardware, choose settings, and optionally save a profile."""
    from mkv_compress.wizard import run_wizard

    ffmpeg, ffprobe = _prepare_tools(output_dir)
    display = EncodingDisplay(console)

    jobs, confirmed = run_wizard(
        directory=directory,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        recursive=recursive,
        output_dir=output_dir,
        overwrite=overwrite,
        no_skip=no_skip,
        console=console,
    )

    if not confirmed:
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=0)

    _run_encode_loop(jobs, ffmpeg, ffprobe, display)


@profiles_app.command("list")
def list_profiles() -> None:
    """List saved encoding profiles."""
    profiles = load_profiles()
    if not profiles:
        console.print("[dim]No saved profiles.[/dim]")
        raise typer.Exit(code=0)

    for profile in profiles:
        label = f" - {profile.label}" if profile.label else ""
        source = " (wizard)" if profile.created_from_wizard else ""
        console.print(f"{profile.name}: preset={profile.preset}, crf={profile.crf}{label}{source}")


@profiles_app.command("delete")
def delete_profile_cmd(name: str = typer.Argument(..., help="Profile name to delete.")) -> None:
    """Delete a saved encoding profile."""
    if not delete_profile(name):
        console.print(f"[red bold]Error:[/red bold] profile '{name}' was not found.")
        raise typer.Exit(code=1)
    console.print(f"[green]Deleted profile[/green] {name}")
