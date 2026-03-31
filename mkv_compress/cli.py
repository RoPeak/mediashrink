from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from mkv_compress.encoder import encode_file
from mkv_compress.platform_utils import check_ffmpeg_available, find_ffmpeg, find_ffprobe
from mkv_compress.progress import EncodingDisplay
from mkv_compress.scanner import build_jobs, scan_directory

app = typer.Typer(
    name="mkvcompress",
    help="Re-encode MKV files to H.265/HEVC to reduce file size.",
    add_completion=False,
)

console = Console()


@app.command()
def main(
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
    crf: int = typer.Option(
        20,
        "--crf",
        help="H.265 CRF quality value (0–51, lower = better quality). Default: 20.",
        min=0,
        max=51,
    ),
    preset: str = typer.Option(
        "slow",
        "--preset",
        help="FFmpeg encoding preset. Slower = better compression. Default: slow.",
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

    # 1. Check FFmpeg
    ok, err = check_ffmpeg_available()
    if not ok:
        console.print(f"[red bold]Error:[/red bold] {err}")
        raise typer.Exit(code=1)

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    # 2. Validate output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Scan
    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(f"[yellow]No .mkv files found in[/yellow] {directory}")
        raise typer.Exit(code=0)

    # 4. Build jobs
    jobs = build_jobs(
        files=files,
        output_dir=output_dir,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
        dry_run=dry_run,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )

    # 5. Show scan table
    display.show_scan_table(jobs)

    # 6. Confirm (skip if --yes or --dry-run)
    to_encode = [j for j in jobs if not j.skip]
    if not to_encode:
        console.print("[dim]Nothing to encode.[/dim]")
        raise typer.Exit(code=0)

    if not dry_run and not yes:
        if not display.confirm_proceed(len(to_encode)):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=0)

    # 7. Encode
    results = []
    with display.make_progress_bar() as progress:
        overall_task = progress.add_task(
            f"[cyan]Encoding {len(to_encode)} file(s)...",
            total=len(to_encode),
        )
        file_task = progress.add_task("", total=100)

        for i, job in enumerate(jobs):
            filename = job.source.name
            progress.update(file_task, description=f"[white]{filename}", completed=0)

            def make_callback(ft=file_task):
                def callback(pct: float) -> None:
                    progress.update(ft, completed=pct)
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
                progress.update(overall_task, advance=1)
                progress.update(file_task, completed=100)

    # 8. Summary
    display.show_summary(results)
