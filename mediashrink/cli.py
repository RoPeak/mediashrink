import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from mediashrink.analysis import (
    analyze_directory,
    build_manifest,
    display_analysis_summary,
    estimate_analysis_encode_seconds,
    load_manifest,
    save_manifest,
)
from mediashrink.cleanup import cleanup_successful_results, eligible_cleanup_results
from mediashrink.encoder import encode_file, encode_preview
from mediashrink.models import EncodeJob, EncodeResult, SessionManifest
from mediashrink.platform_utils import check_ffmpeg_available, find_ffmpeg, find_ffprobe
from mediashrink.profiles import delete_profile, get_profile, list_all_profiles
from mediashrink.progress import EncodingDisplay
from mediashrink.scanner import build_jobs, scan_directory, supported_formats_label
from mediashrink.session import (
    build_session,
    find_resumable_session,
    get_session_path,
    save_session,
    update_session_entry,
)

EXIT_SUCCESS = 0
EXIT_NO_FILES = 1
EXIT_ENCODE_FAILURES = 2
EXIT_USER_CANCELLED = 3
EXIT_FFMPEG_NOT_FOUND = 4


def _results_to_json(results: list[EncodeResult], exit_code: int) -> str:
    files = []
    total_saved = 0
    for r in results:
        if r.skipped:
            status = "skipped"
        elif r.success:
            status = "success"
        else:
            status = "failed"
        saved = r.size_reduction_bytes if r.success else 0
        total_saved += saved
        files.append(
            {
                "source": str(r.job.source),
                "status": status,
                "input_bytes": r.input_size_bytes,
                "output_bytes": r.output_size_bytes,
                "reduction_pct": round(r.size_reduction_pct, 1) if r.success else 0.0,
            }
        )
    return json.dumps({"exit_code": exit_code, "files": files, "total_saved_bytes": total_saved})


class DefaultCommandGroup(TyperGroup):
    """Route bare `mediashrink ...` invocations to the hidden encode command."""

    default_command_name = "encode"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in {"--help", "-h"}:
            args.insert(0, self.default_command_name)
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="mediashrink",
    help=f"Re-encode supported video files ({supported_formats_label()}) to H.265/HEVC to reduce file size.",
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
    session: SessionManifest | None = None,
    session_path: Path | None = None,
    log_path: Path | None = None,
) -> list[EncodeResult]:
    to_encode = [job for job in jobs if not job.skip]
    total_bytes = sum(job.source.stat().st_size for job in to_encode)
    results = []
    bytes_done = 0

    with display.make_progress_bar() as progress:
        overall_task = progress.add_task(
            f"[cyan]Overall ({len(to_encode)} file(s))",
            total=total_bytes,
        )
        file_task = progress.add_task("", total=1)

        for job in jobs:
            filename = job.source.name
            file_size = job.source.stat().st_size
            # Use input size as the task total so DownloadColumn shows GB-scale numbers
            task_total = max(file_size, 1)
            progress.update(
                file_task, description=f"[white]{filename}", completed=0, total=task_total
            )

            def make_callback(ft=file_task, fb=file_size, ot=overall_task, bd=bytes_done):
                def callback(pct: float) -> None:
                    progress.update(ft, completed=fb * pct / 100)
                    progress.update(ot, completed=bd + fb * pct / 100)

                return callback

            try:
                result = encode_file(
                    job,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    progress_callback=make_callback() if not job.skip else None,
                    log_path=log_path,
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                sys.exit(1)

            results.append(result)

            # Update session after each file so partial progress is persisted
            if session is not None and session_path is not None:
                if result.skipped:
                    status = "skipped"
                elif result.success:
                    status = "success"
                else:
                    status = "failed"
                update_session_entry(
                    session,
                    source=job.source,
                    status=status,
                    output=job.output if result.success else None,
                    error=result.error_message,
                )
                save_session(session, session_path)

            if not job.skip:
                bytes_done += file_size
                progress.update(overall_task, completed=bytes_done)
                progress.update(file_task, completed=100)

    display.show_summary(results)
    return results


def _prepare_tools(output_dir: Path | None) -> tuple[Path, Path]:
    ok, err = check_ffmpeg_available()
    if not ok:
        console.print(f"[red bold]Error:[/red bold] {err}")
        raise typer.Exit(code=EXIT_FFMPEG_NOT_FOUND)

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    return ffmpeg, ffprobe


def _maybe_prompt_for_cleanup(results: list[EncodeResult], assume_yes: bool) -> None:
    candidates = eligible_cleanup_results(results)
    if not candidates:
        return

    if not assume_yes:
        if not typer.confirm(
            "Delete the original source files for successful encodes and rename the compressed outputs back to the original filenames?",
            default=False,
        ):
            return

    cleaned = cleanup_successful_results(results)
    if cleaned:
        console.print(
            f"[green]Cleanup complete:[/green] restored original names for {len(cleaned)} file(s)."
        )


def _resolve_encode_settings(
    profile: str | None,
    crf: int | None,
    preset: str | None,
) -> tuple[int, str, str | None]:
    effective_crf = 20
    effective_preset = "fast"
    profile_name: str | None = None

    if profile:
        saved_profile = get_profile(profile)
        if saved_profile is None:
            console.print(f"[red bold]Error:[/red bold] profile '{profile}' was not found.")
            raise typer.Exit(code=EXIT_NO_FILES)
        effective_crf = saved_profile.crf
        effective_preset = saved_profile.preset
        profile_name = saved_profile.name

    if crf is not None:
        effective_crf = crf
    if preset is not None:
        effective_preset = preset

    return effective_crf, effective_preset, profile_name


@app.command("encode", hidden=True)
def encode_cmd(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to compress.",
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
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
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
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="After successful side-by-side encodes, delete originals and rename outputs back to the original filenames.",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Ignore any existing session file and start fresh.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a single JSON blob instead of Rich terminal output.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Write FFmpeg stderr to a log file alongside the output.",
    ),
) -> None:
    quiet_console = Console(quiet=True) if json_output else console
    display = EncodingDisplay(quiet_console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)

    effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)

    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(
            f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
        )
        raise typer.Exit(code=EXIT_NO_FILES)

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

    # Resume detection — skip files already completed in a previous session
    if not dry_run and not no_resume:
        prior = find_resumable_session(directory, output_dir, effective_preset, effective_crf)
        if prior is not None:
            done = {e.source for e in prior.entries if e.status == "success"}
            if done:
                console.print(
                    f"[cyan]Session found:[/cyan] {len(done)} file(s) already completed. "
                    "Resume? [Y/n] ",
                    end="",
                )
                if typer.confirm("", default=True):
                    for job in jobs:
                        if str(job.source) in done:
                            job.skip = True
                            job.skip_reason = "resumed (already done)"

    display.show_scan_table(jobs)

    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        console.print("[dim]Nothing to encode.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)

    if not dry_run and not yes:
        if not display.confirm_proceed(len(to_encode)):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)

    session_path = get_session_path(directory, output_dir) if not dry_run else None
    active_session = None
    if session_path is not None:
        active_session = build_session(
            directory, effective_preset, effective_crf, overwrite, output_dir, jobs
        )
        save_session(active_session, session_path)

    log_path: Path | None = None
    if verbose and not dry_run:
        log_dir = output_dir if output_dir is not None else directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"mediashrink_{timestamp}.log"
        console.print(f"[dim]Verbose log:[/dim] {log_path}")

    results = _run_encode_loop(
        jobs,
        ffmpeg,
        ffprobe,
        display,
        session=active_session,
        session_path=session_path,
        log_path=log_path,
    )
    if cleanup:
        _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not dry_run and not overwrite and output_dir is None and not yes:
        _maybe_prompt_for_cleanup(results, assume_yes=False)

    has_failures = any(not r.success and not r.skipped for r in results)
    exit_code = EXIT_ENCODE_FAILURES if has_failures else EXIT_SUCCESS
    if json_output:
        print(_results_to_json(results, exit_code))
    if has_failures:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def analyze(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to analyze.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load saved CRF/preset defaults from a named profile.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        min=0,
        max=51,
        help="H.265 CRF quality value used for analysis estimates.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Encoding preset used for analysis estimates.",
    ),
    manifest_out: Optional[Path] = typer.Option(
        None,
        "--manifest-out",
        help="Write recommended candidates to a JSON manifest.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit analysis results as a JSON blob instead of Rich terminal output.",
    ),
) -> None:
    quiet_console = Console(quiet=True) if json_output else console
    ffmpeg, ffprobe = _prepare_tools(None)
    effective_crf, effective_preset, profile_name = _resolve_encode_settings(profile, crf, preset)

    items = analyze_directory(directory, recursive=recursive, ffprobe=ffprobe)
    if not items:
        if json_output:
            print(json.dumps({"exit_code": EXIT_NO_FILES, "items": []}))
        else:
            console.print(
                f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
            )
        raise typer.Exit(code=EXIT_NO_FILES)

    estimated_total_encode_seconds = estimate_analysis_encode_seconds(
        items=items,
        preset=effective_preset,
        crf=effective_crf,
        ffmpeg=ffmpeg,
    )

    if json_output:
        manifest = build_manifest(
            directory=directory,
            recursive=recursive,
            preset=effective_preset,
            crf=effective_crf,
            profile_name=profile_name,
            estimated_total_encode_seconds=estimated_total_encode_seconds,
            items=items,
        )
        print(json.dumps(manifest.to_dict()))
    else:
        display_analysis_summary(items, estimated_total_encode_seconds, quiet_console)

    if manifest_out is not None:
        manifest = build_manifest(
            directory=directory,
            recursive=recursive,
            preset=effective_preset,
            crf=effective_crf,
            profile_name=profile_name,
            estimated_total_encode_seconds=estimated_total_encode_seconds,
            items=items,
        )
        save_manifest(manifest, manifest_out)
        if not json_output:
            console.print(f"[green]Wrote manifest[/green] {manifest_out}")


@app.command()
def apply(
    manifest: Path = typer.Argument(
        ...,
        help="Path to an analysis manifest JSON file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
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
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Override manifest settings using a saved profile.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        min=0,
        max=51,
        help="Override manifest CRF.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Override manifest preset.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="After successful side-by-side encodes, delete originals and rename outputs back to the original filenames.",
    ),
) -> None:
    display = EncodingDisplay(console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)
    loaded_manifest = load_manifest(manifest)

    effective_crf = loaded_manifest.crf
    effective_preset = loaded_manifest.preset
    if profile or crf is not None or preset is not None:
        effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)

    missing_files = [item.source for item in loaded_manifest.items if not item.source.exists()]
    existing_files = [item.source for item in loaded_manifest.items if item.source.exists()]
    for missing_path in missing_files:
        console.print(f"[yellow]Missing file from manifest:[/yellow] {missing_path}")

    if not existing_files:
        console.print("[yellow]No manifest files are available to encode.[/yellow]")
        raise typer.Exit(code=EXIT_NO_FILES)

    jobs = build_jobs(
        files=existing_files,
        output_dir=output_dir,
        overwrite=overwrite,
        crf=effective_crf,
        preset=effective_preset,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=False,
    )

    display.show_scan_table(jobs)
    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        console.print("[dim]Nothing to encode.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)

    if not yes and not display.confirm_proceed(len(to_encode)):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)

    results = _run_encode_loop(jobs, ffmpeg, ffprobe, display)
    if cleanup:
        _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not overwrite and output_dir is None and not yes:
        _maybe_prompt_for_cleanup(results, assume_yes=False)

    if any(not r.success and not r.skipped for r in results):
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def wizard(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to compress.",
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
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit encode results as a JSON blob instead of Rich terminal output.",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Non-interactive mode: auto-select the recommended profile, skip all prompts.",
    ),
) -> None:
    """Interactively detect hardware, choose settings, and optionally save a profile."""
    from mediashrink.wizard import run_wizard

    ffmpeg, ffprobe = _prepare_tools(output_dir)
    quiet_console = Console(quiet=True) if json_output else console
    display = EncodingDisplay(quiet_console)

    jobs, action, wizard_cleanup = run_wizard(
        directory=directory,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        recursive=recursive,
        output_dir=output_dir,
        overwrite=overwrite,
        no_skip=no_skip,
        console=quiet_console,
        auto=auto,
    )

    if action == "cancel":
        if not json_output:
            console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)
    if action == "export":
        raise typer.Exit(code=EXIT_SUCCESS)

    results = _run_encode_loop(jobs, ffmpeg, ffprobe, display)
    if wizard_cleanup:
        _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not json_output and not overwrite and output_dir is None:
        pass  # cleanup was already asked upfront; no second prompt

    has_failures = any(not r.success and not r.skipped for r in results)
    exit_code = EXIT_ENCODE_FAILURES if has_failures else EXIT_SUCCESS
    if json_output:
        print(_results_to_json(results, exit_code))
    if has_failures:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def preview(
    file: Path = typer.Argument(
        ...,
        help=f"A supported video file ({supported_formats_label()}) to preview-encode.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    minutes: float = typer.Option(
        2.0,
        "--minutes",
        help="How many minutes to encode for the preview (default: 2).",
        min=0.1,
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        help="H.265 CRF quality value. Default: 20.",
        min=0,
        max=51,
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Encoding preset. Default: fast.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load CRF/preset from a named profile.",
    ),
) -> None:
    """Test-encode the first N minutes of a single file to check quality before a full batch."""
    from mediashrink.scanner import SUPPORTED_EXTENSIONS

    if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[red bold]Error:[/red bold] {file.name} is not a supported format "
            f"({supported_formats_label()})."
        )
        raise typer.Exit(code=EXIT_NO_FILES)

    display = EncodingDisplay(console)
    ffmpeg, ffprobe = _prepare_tools(None)
    effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)

    console.print(f"[dim]Preview encoding first {minutes:.1f} minute(s) of[/dim] {file.name}...")
    result = encode_preview(
        source=file,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        duration_minutes=minutes,
        crf=effective_crf,
        preset=effective_preset,
    )
    display.show_summary([result])

    if not result.success:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@profiles_app.command("list")
def list_profiles() -> None:
    """List saved encoding profiles."""
    profiles = list_all_profiles()
    if not profiles:
        console.print("[dim]No saved profiles.[/dim]")
        raise typer.Exit(code=EXIT_SUCCESS)

    for profile in profiles:
        label = f" - {profile.label}" if profile.label else ""
        if profile.builtin:
            source = " [dim](builtin)[/dim]"
        elif profile.created_from_wizard:
            source = " (wizard)"
        else:
            source = ""
        console.print(f"{profile.name}: preset={profile.preset}, crf={profile.crf}{label}{source}")


@profiles_app.command("delete")
def delete_profile_cmd(name: str = typer.Argument(..., help="Profile name to delete.")) -> None:
    """Delete a saved encoding profile."""
    if not delete_profile(name):
        console.print(f"[red bold]Error:[/red bold] profile '{name}' was not found.")
        raise typer.Exit(code=EXIT_NO_FILES)
    console.print(f"[green]Deleted profile[/green] {name}")
