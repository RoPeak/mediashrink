from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mediashrink.analysis import (
    analyze_directory,
    build_manifest,
    display_analysis_summary,
    estimate_analysis_encode_seconds,
    save_manifest,
)
from mediashrink.constants import CRF_COMPRESSION_FACTOR
from mediashrink.encoder import _HW_ENCODERS, encode_preview, get_duration_seconds, probe_encoder_available, validate_encoder
from mediashrink.models import AnalysisItem, EncodeJob
from mediashrink.platform_utils import detect_device_labels
from mediashrink.profiles import SavedProfile, get_builtin_profiles, upsert_profile
from mediashrink.scanner import build_jobs, scan_directory, supported_formats_label

_GB = 1024**3
_MB = 1024**2

_CRF_COMPRESSION_FACTOR = CRF_COMPRESSION_FACTOR

_BENCHMARK_SECONDS = 8
_HARDWARE_DISPLAY_NAMES = {
    "qsv": "Intel Quick Sync",
    "nvenc": "Nvidia NVENC",
    "amf": "AMD AMF",
}
_HW_ENCODER_CAVEATS: dict[str, str] = {
    "nvenc": "NVENC: consumer GPUs allow max 3 concurrent encode sessions.",
    "amf":   "AMF: quality may vary on older Radeon GPUs; test output before batch use.",
}


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.2f} GB"
    return f"{size_bytes / _MB:.1f} MB"


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


@dataclass
class EncoderProfile:
    index: int
    name: str
    encoder_key: str
    crf: int
    sw_preset: str | None
    estimated_output_bytes: int
    estimated_encode_seconds: float
    quality_label: str
    is_recommended: bool
    is_custom: bool = False
    is_builtin: bool = False


def detect_available_encoders(
    ffmpeg: Path,
    console: Console,
    sample_file: Path | None = None,
    ffprobe: Path | None = None,
) -> list[str]:
    """Return available hardware encoder keys in stable display order.

    If sample_file and ffprobe are provided, each probed encoder is validated
    against real content to catch encoders that pass the synthetic test but
    fail on actual video input.
    """
    candidates = list(_HW_ENCODERS.keys())

    with console.status("[dim]Detecting hardware encoders...[/dim]", spinner="dots"):
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(probe_encoder_available, key, ffmpeg): key
                for key in candidates
            }
            detected: set[str] = set()
            for future in as_completed(futures):
                key = futures[future]
                try:
                    if future.result():
                        detected.add(key)
                except Exception:
                    pass

    # Validation pass — encode 3s of real content to catch broken encoders
    if sample_file is not None and ffprobe is not None:
        validated: set[str] = set()
        for key in detected:
            ok, err = validate_encoder(key, sample_file, ffmpeg, ffprobe)
            if ok:
                validated.add(key)
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] {_HARDWARE_DISPLAY_NAMES.get(key, key)} "
                    f"passed availability check but failed validation ({err}). Skipping."
                )
        detected = validated

    available: list[str] = []
    for key in ("qsv", "nvenc", "amf"):
        if key in detected:
            available.append(key)
    return available


def benchmark_encoder(
    encoder_key: str,
    sample_file: Path,
    sample_duration: float,
    crf: int,
    ffmpeg: Path,
) -> float | None:
    """Return benchmark speed as media-seconds per wall-second."""
    clip_len = min(_BENCHMARK_SECONDS, sample_duration * 0.9)
    if clip_len <= 0:
        return None
    seek_pos = max(sample_duration * 0.2, 0.0)

    if encoder_key in _HW_ENCODERS:
        encoder_name, quality_flag, extra_flags = _HW_ENCODERS[encoder_key]
        if encoder_key == "amf":
            quality_args = ["-qp_i", str(crf), "-qp_p", str(crf)]
        else:
            quality_args = [quality_flag, str(crf)]
        video_flags = ["-c:v", encoder_name] + quality_args + extra_flags
    else:
        video_flags = ["-c:v", "libx265", "-crf", str(crf), "-preset", encoder_key]

    cmd = [
        str(ffmpeg),
        "-ss",
        str(seek_pos),
        "-i",
        str(sample_file),
        "-t",
        str(clip_len),
    ] + video_flags + [
        "-an",
        "-f",
        "null",
        "-",
        "-loglevel",
        "error",
    ]

    try:
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.monotonic() - start
        if result.returncode != 0 or elapsed < 0.1:
            return None
        return clip_len / elapsed
    except (subprocess.TimeoutExpired, OSError):
        return None


def _estimate_output_bytes(total_input_bytes: int, crf: int) -> int:
    nearest = min(_CRF_COMPRESSION_FACTOR, key=lambda k: abs(k - crf))
    return int(total_input_bytes * _CRF_COMPRESSION_FACTOR[nearest])


def _estimate_time(total_media_seconds: float, speed: float | None) -> float:
    if speed is None or speed <= 0:
        return 0.0
    return total_media_seconds / speed


def _sum_media_durations(files: list[Path], ffprobe: Path) -> float:
    total = 0.0
    fallback_count = 0

    for path in files:
        duration = get_duration_seconds(path, ffprobe)
        if duration > 0:
            total += duration
        else:
            fallback_count += 1

    if total <= 0 and files:
        return 3600.0 * len(files)

    if fallback_count:
        avg_known = total / max(len(files) - fallback_count, 1)
        total += avg_known * fallback_count

    return total


_ENCODER_LABEL_MAX_CHARS = 22


def _encoder_display_name(encoder_key: str, device_labels: dict[str, str], truncate: bool = False) -> str:
    if encoder_key in _HW_ENCODERS:
        label = device_labels.get(encoder_key)
        if label:
            if truncate and len(label) > _ENCODER_LABEL_MAX_CHARS:
                label = label[:_ENCODER_LABEL_MAX_CHARS].rstrip() + "…"
            return f"{_HARDWARE_DISPLAY_NAMES[encoder_key]} ({label})"
        return _HARDWARE_DISPLAY_NAMES[encoder_key]
    return f"libx265 ({encoder_key})"


def build_profiles(
    available_hw: list[str],
    benchmark_speeds: dict[str, float | None],
    total_media_seconds: float,
    total_input_bytes: int,
) -> list[EncoderProfile]:
    profiles: list[EncoderProfile] = []
    idx = 1

    hardware_profiles: list[tuple[str, float | None]] = [
        (key, benchmark_speeds.get(key)) for key in ("qsv", "nvenc", "amf") if key in available_hw
    ]
    hardware_profiles.sort(key=lambda item: (item[1] or 0.0), reverse=True)

    for key, speed in hardware_profiles:
        profiles.append(
            EncoderProfile(
                index=idx,
                name=f"Fastest on this device",
                encoder_key=key,
                crf=20,
                sw_preset=None,
                estimated_output_bytes=_estimate_output_bytes(total_input_bytes, 20),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label="Good",
                is_recommended=False,
            )
        )
        idx += 1

    fast_speed = benchmark_speeds.get("fast")
    faster_speed = benchmark_speeds.get("faster") or (fast_speed * 1.3 if fast_speed else None)
    profiles.append(
        EncoderProfile(
            index=idx,
            name="Faster Encode",
            encoder_key="faster",
            crf=22,
            sw_preset="faster",
            estimated_output_bytes=_estimate_output_bytes(total_input_bytes, 22),
            estimated_encode_seconds=_estimate_time(total_media_seconds, faster_speed),
            quality_label="Very good",
            is_recommended=False,
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            name="Balanced",
            encoder_key="fast",
            crf=20,
            sw_preset="fast",
            estimated_output_bytes=_estimate_output_bytes(total_input_bytes, 20),
            estimated_encode_seconds=_estimate_time(total_media_seconds, fast_speed),
            quality_label="Excellent",
            is_recommended=False,
        )
    )
    idx += 1

    slow_speed = (fast_speed / 4) if fast_speed else None
    profiles.append(
        EncoderProfile(
            index=idx,
            name="Best Quality",
            encoder_key="slow",
            crf=18,
            sw_preset="slow",
            estimated_output_bytes=_estimate_output_bytes(total_input_bytes, 18),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Visually lossless",
            is_recommended=False,
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            name="Smallest File",
            encoder_key="slow",
            crf=28,
            sw_preset="slow",
            estimated_output_bytes=_estimate_output_bytes(total_input_bytes, 28),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Good",
            is_recommended=False,
        )
    )
    idx += 1

    # Built-in intent presets
    _BUILTIN_QUALITY_LABELS = {
        "TV Batch":           "Very good",
        "Archival":           "Visually lossless",
        "Fast GPU Transcode": "Good",
        "Smallest Acceptable": "Acceptable",
    }
    best_hw = hardware_profiles[0][0] if hardware_profiles else None
    best_hw_speed = hardware_profiles[0][1] if hardware_profiles else None

    for bp in get_builtin_profiles():
        # "Fast GPU Transcode" uses the best available HW encoder, or falls back to sw
        if bp.name == "Fast GPU Transcode":
            encoder_key = best_hw if best_hw else "faster"
            speed = best_hw_speed if best_hw else benchmark_speeds.get("faster")
            sw_preset = None if best_hw else "faster"
        else:
            encoder_key = bp.preset
            speed = benchmark_speeds.get(bp.preset)
            if encoder_key not in _HW_ENCODERS:
                # For slow preset use the slow-speed estimate derived from fast
                if encoder_key == "slow" and benchmark_speeds.get("fast"):
                    speed = (benchmark_speeds["fast"] or 0) / 4
            sw_preset = bp.preset if encoder_key not in _HW_ENCODERS else None

        profiles.append(
            EncoderProfile(
                index=idx,
                name=bp.name,
                encoder_key=encoder_key,
                crf=bp.crf,
                sw_preset=sw_preset,
                estimated_output_bytes=_estimate_output_bytes(total_input_bytes, bp.crf),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label=_BUILTIN_QUALITY_LABELS.get(bp.name, "Good"),
                is_recommended=False,
                is_builtin=True,
            )
        )
        idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            name="Custom",
            encoder_key="",
            crf=20,
            sw_preset=None,
            estimated_output_bytes=0,
            estimated_encode_seconds=0,
            quality_label="",
            is_recommended=False,
            is_custom=True,
        )
    )

    if hardware_profiles:
        recommended_key = hardware_profiles[0][0]
        for profile in profiles:
            if profile.encoder_key == recommended_key:
                profile.is_recommended = True
                break
    else:
        for profile in profiles:
            if profile.name == "Balanced":
                profile.is_recommended = True
                break

    return profiles


def display_profiles_table(
    profiles: list[EncoderProfile],
    total_input_bytes: int,
    device_labels: dict[str, str],
    console: Console,
) -> None:
    table = Table(
        title="Available encoding profiles",
        header_style="bold cyan",
        expand=True,
        show_lines=False,
    )
    table.add_column("#", justify="right", style="bold", no_wrap=True)
    table.add_column("Profile", no_wrap=True)
    table.add_column("Encoder", style="dim cyan")
    table.add_column("CRF", justify="center", no_wrap=True)
    table.add_column("Est. Output", justify="right", style="green", no_wrap=True)
    table.add_column("Est. Saving", justify="right", style="bold green", no_wrap=True)
    table.add_column("Est. Time", justify="right", no_wrap=True)
    table.add_column("Quality", no_wrap=True)
    table.add_column("", no_wrap=True)

    for profile in profiles:
        if profile.is_custom:
            table.add_row(str(profile.index), "Custom", "-", "-", "-", "-", "-", "-", "")
            continue

        encoder_display = _encoder_display_name(profile.encoder_key, device_labels, truncate=True)
        est_out = _fmt_size(profile.estimated_output_bytes)
        saved = total_input_bytes - profile.estimated_output_bytes
        saved_pct = saved / total_input_bytes * 100 if total_input_bytes else 0
        est_saving = f"~{_fmt_size(saved)} ({saved_pct:.0f}%)"
        est_time = (
            f"~{_fmt_duration(profile.estimated_encode_seconds)}"
            if profile.estimated_encode_seconds > 0
            else "~unknown"
        )
        tag = Text("RECOMMENDED", style="bold cyan") if profile.is_recommended else Text("")
        quality_style = {
            "Visually lossless": "green bold",
            "Excellent": "green",
            "Very good": "green",
            "Good": "yellow",
        }.get(profile.quality_label, "white")

        table.add_row(
            str(profile.index),
            profile.name,
            encoder_display,
            str(profile.crf),
            f"~{est_out}",
            est_saving,
            est_time,
            Text(profile.quality_label, style=quality_style),
            tag,
        )

    console.print()
    console.print(table)
    console.print(f"  [dim]Total input: {_fmt_size(total_input_bytes)}[/dim]")
    console.print("  [dim]Time and size numbers are approximate estimates.[/dim]")
    console.print("  [dim]Hardware presets are still full re-encodes; source bitrate, resolution, and runtime dominate total time.[/dim]")
    console.print("  [dim]For lower wait time, choose Fastest on this device or Faster Encode.[/dim]")
    console.print()


def prompt_profile_selection(profiles: list[EncoderProfile], console: Console) -> EncoderProfile:
    recommended = next((profile for profile in profiles if profile.is_recommended), profiles[0])
    max_idx = profiles[-1].index

    while True:
        choice = typer.prompt(
            f"Select a profile [1-{max_idx}, Enter for {recommended.index} ({recommended.name})]",
            default=str(recommended.index),
        ).strip()

        try:
            selected_index = int(choice)
        except ValueError:
            selected_index = -1

        for profile in profiles:
            if profile.index == selected_index:
                return profile

        console.print(f"[yellow]Please enter a number between 1 and {max_idx}.[/yellow]")


def run_custom_wizard(available_hw: list[str], console: Console) -> tuple[str, int, str | None]:
    """Walk through manual encoder/CRF selection."""
    encoder_choices: list[tuple[str, str]] = []
    for key in ("qsv", "nvenc", "amf"):
        if key in available_hw:
            encoder_choices.append((key, f"{_HARDWARE_DISPLAY_NAMES[key]} (hardware)"))
    encoder_choices.append(("libx265", "libx265 (software)"))

    console.print("\n[bold]Custom encoder:[/bold]")
    for idx, (_, label) in enumerate(encoder_choices, start=1):
        console.print(f"  {idx}. {label}")

    while True:
        try:
            enc_idx = int(typer.prompt(f"Choose encoder [1-{len(encoder_choices)}]", default="1")) - 1
        except ValueError:
            enc_idx = -1
        if 0 <= enc_idx < len(encoder_choices):
            break
        console.print("[yellow]Invalid choice.[/yellow]")

    chosen_key, _ = encoder_choices[enc_idx]

    while True:
        try:
            crf = int(typer.prompt("CRF quality value [0-51, lower = better quality]", default="20"))
        except ValueError:
            crf = -1
        if 0 <= crf <= 51:
            break
        console.print("[yellow]CRF must be between 0 and 51.[/yellow]")

    if chosen_key != "libx265":
        return chosen_key, crf, None

    sw_presets = ["ultrafast", "faster", "fast", "medium", "slow"]
    console.print("\n[bold]Software preset:[/bold] (slower = better compression)")
    for idx, sw_preset in enumerate(sw_presets, start=1):
        console.print(f"  {idx}. {sw_preset}")

    while True:
        try:
            preset_idx = int(typer.prompt("Choose preset [1-5]", default="3")) - 1
        except ValueError:
            preset_idx = -1
        if 0 <= preset_idx < len(sw_presets):
            selected_preset = sw_presets[preset_idx]
            return selected_preset, crf, selected_preset
        console.print("[yellow]Invalid choice.[/yellow]")


def maybe_save_profile(
    preset: str,
    crf: int,
    display_label: str,
    console: Console,
) -> None:
    if not typer.confirm("Save these settings as a named profile?", default=False):
        return

    while True:
        name = typer.prompt("Profile name").strip()
        if name:
            break
        console.print("[yellow]Profile name cannot be empty.[/yellow]")

    upsert_profile(
        SavedProfile(
            name=name,
            preset=preset,
            crf=crf,
            label=display_label,
            created_from_wizard=True,
        )
    )
    console.print(f"[green]Saved profile[/green] {name}")


def display_candidate_table(title: str, items: list[AnalysisItem], console: Console) -> None:
    table = Table(title=title, header_style="bold cyan", expand=True)
    table.add_column("File")
    table.add_column("Codec", justify="center", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Est. Saving", justify="right", no_wrap=True)
    table.add_column("Reason")

    for item in sorted(items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True)[:12]:
        saving = "-" if item.estimated_savings_bytes <= 0 else f"~{_fmt_size(item.estimated_savings_bytes)}"
        table.add_row(
            item.source.name,
            item.codec or "?",
            _fmt_size(item.size_bytes),
            saving,
            item.reason_text,
        )

    console.print()
    console.print(table)
    console.print()


def prompt_analysis_action(recommended_count: int, maybe_count: int, console: Console) -> str:
    console.print("[bold]Next step:[/bold]")
    console.print("  1. Compress recommended only")
    if maybe_count:
        console.print("  2. Review maybe files")
        console.print("  3. Export manifest")
        console.print("  4. Cancel")
        max_choice = 4
        export_choice = 3
        cancel_choice = 4
    else:
        console.print("  2. Export manifest")
        console.print("  3. Cancel")
        max_choice = 3
        export_choice = 2
        cancel_choice = 3

    while True:
        try:
            choice = int(
                typer.prompt(
                    f"Choose action [1-{max_choice}]",
                    default="1",
                )
            )
        except ValueError:
            choice = -1

        if choice == 1:
            return "compress_recommended"
        if maybe_count and choice == 2:
            return "review_maybe"
        if choice == export_choice:
            return "export"
        if choice == cancel_choice:
            return "cancel"

        console.print("[yellow]Invalid choice.[/yellow]")


def review_maybe_items(maybe_items: list[AnalysisItem], console: Console) -> bool:
    display_candidate_table("Maybe files", maybe_items, console)
    return typer.confirm("Include maybe files in this run?", default=False)


def run_wizard(
    directory: Path,
    ffmpeg: Path,
    ffprobe: Path,
    recursive: bool,
    output_dir: Path | None,
    overwrite: bool,
    no_skip: bool,
    console: Console,
    auto: bool = False,
) -> tuple[list[EncodeJob], str, bool]:
    """Run the interactive wizard and return (jobs, action)."""
    console.print("\n[bold cyan]mediashrink wizard[/bold cyan]")
    console.print("[dim]Scanning files and detecting hardware...[/dim]\n")

    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(
            f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
        )
        return [], "cancel", False

    total_input_bytes = sum(path.stat().st_size for path in files)
    console.print(
        f"Found [bold]{len(files)}[/bold] file(s) "
        f"([yellow]{_fmt_size(total_input_bytes)}[/yellow] total)\n"
    )

    total_media_seconds = _sum_media_durations(files, ffprobe)
    sample_file = files[0]
    sample_duration = get_duration_seconds(sample_file, ffprobe)
    if sample_duration <= 0:
        sample_duration = 3600.0

    available_hw = detect_available_encoders(ffmpeg, console, sample_file=sample_file, ffprobe=ffprobe)
    device_labels = detect_device_labels()

    if available_hw:
        console.print("[green]Hardware encoders available:[/green]")
        for key in available_hw:
            console.print(f"  - {_encoder_display_name(key, device_labels)}")
    else:
        console.print("[dim]No hardware encoders detected. Software only.[/dim]")

    candidates_to_bench = list(available_hw) + ["fast", "faster"]
    benchmark_speeds: dict[str, float | None] = {}

    with console.status("[dim]Benchmarking encoders (this takes ~10s)...[/dim]", spinner="dots"):
        for key in candidates_to_bench:
            benchmark_speeds[key] = benchmark_encoder(
                encoder_key=key,
                sample_file=sample_file,
                sample_duration=sample_duration,
                crf=20,
                ffmpeg=ffmpeg,
            )

    profiles = build_profiles(
        available_hw=available_hw,
        benchmark_speeds=benchmark_speeds,
        total_media_seconds=total_media_seconds,
        total_input_bytes=total_input_bytes,
    )
    display_profiles_table(profiles, total_input_bytes, device_labels, console)

    for hw_key in available_hw:
        if hw_key in _HW_ENCODER_CAVEATS:
            console.print(f"  [dim yellow]Note:[/dim yellow] [dim]{_HW_ENCODER_CAVEATS[hw_key]}[/dim]")

    if auto:
        selected = next((p for p in profiles if p.is_recommended), profiles[0])
        console.print(f"[dim]Auto mode: selected profile[/dim] {selected.name}")
    else:
        selected = prompt_profile_selection(profiles, console)
    if selected.is_custom:
        preset, crf, sw_preset = run_custom_wizard(available_hw, console)
        display_label = f"Custom ({preset}, CRF {crf})"
    else:
        preset = selected.encoder_key
        crf = selected.crf
        sw_preset = selected.sw_preset
        display_label = selected.name

    if not auto:
        maybe_save_profile(preset, crf, display_label, console)

    if not auto and typer.confirm("Test encode the first 2 minutes before the full batch?", default=False):
        console.print(f"[dim]Preview encoding[/dim] {sample_file.name}...")
        preview_result = encode_preview(
            source=sample_file,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            duration_minutes=2.0,
            crf=crf,
            preset=preset,
        )
        from mediashrink.progress import EncodingDisplay
        EncodingDisplay(console).show_summary([preview_result])
        if not preview_result.success:
            console.print("[yellow]Preview encode failed — continuing with full batch selection.[/yellow]")

    analysis_items = analyze_directory(directory, recursive=recursive, ffprobe=ffprobe)
    estimated_total_encode_seconds = estimate_analysis_encode_seconds(
        items=analysis_items,
        preset=preset,
        crf=crf,
        ffmpeg=ffmpeg,
        known_speed=benchmark_speeds.get(preset),
    )
    display_analysis_summary(analysis_items, estimated_total_encode_seconds, console)

    recommended_items = [item for item in analysis_items if item.recommendation == "recommended"]
    maybe_items = [item for item in analysis_items if item.recommendation == "maybe"]
    if not recommended_items:
        console.print("[dim]No recommended files were found for automatic compression.[/dim]")
        return [], "cancel", False

    action = "compress_recommended" if auto else prompt_analysis_action(len(recommended_items), len(maybe_items), console)
    if action == "cancel":
        return [], "cancel", False
    if action == "export":
        manifest = build_manifest(
            directory=directory,
            recursive=recursive,
            preset=preset,
            crf=crf,
            profile_name=None,
            estimated_total_encode_seconds=estimated_total_encode_seconds,
            items=analysis_items,
        )
        default_manifest_path = directory / "mediashrink-analysis.json"
        manifest_path = Path(typer.prompt("Manifest path", default=str(default_manifest_path)))
        save_manifest(manifest, manifest_path)
        console.print(f"[green]Wrote manifest[/green] {manifest_path}")
        return [], "export", False

    selected_items = list(recommended_items)
    if action == "review_maybe" and maybe_items:
        if review_maybe_items(maybe_items, console):
            selected_items.extend(maybe_items)

    jobs = build_jobs(
        files=[item.source for item in selected_items],
        output_dir=output_dir,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )

    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        console.print("[dim]Nothing to encode after re-checking the selected files.[/dim]")
        return [], "cancel", False

    console.print()
    console.print("[bold]Ready to encode[/bold]")
    console.print(f"  Files:    {len(to_encode)}")
    console.print(f"  Encoder:  {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}")
    console.print(f"  CRF:      {crf}")
    selected_input_bytes = sum(item.size_bytes for item in selected_items)
    selected_output_bytes = sum(item.estimated_output_bytes for item in selected_items if item.estimated_output_bytes > 0)
    selected_saved_bytes = sum(item.estimated_savings_bytes for item in selected_items)
    selected_saved_pct = selected_saved_bytes / selected_input_bytes * 100 if selected_input_bytes else 0.0
    console.print(f"  Input:    {_fmt_size(selected_input_bytes)}")
    if selected_output_bytes > 0:
        console.print(
            f"  Est. out: ~{_fmt_size(selected_output_bytes)}  "
            f"(~{_fmt_size(selected_saved_bytes)} saved, ~{selected_saved_pct:.0f}%)"
        )
    if estimated_total_encode_seconds is not None and estimated_total_encode_seconds > 0:
        console.print(f"  Est. time: ~{_fmt_duration(estimated_total_encode_seconds)}")
    if preset in _HW_ENCODERS:
        console.print("  [dim]Hardware encoding is faster, but source duration and bitrate still dominate total runtime.[/dim]")
    elif preset in {"faster", "ultrafast"}:
        console.print("  [dim]This favors faster completion over maximum compression efficiency.[/dim]")
    else:
        console.print("  [dim]Slower software presets trade more time for slightly smaller files.[/dim]")

    if not to_encode[0].output.parent.exists():
        console.print(f"  Output:   {to_encode[0].output.parent}")
    console.print("  [dim]Estimates are approximate.[/dim]")

    cleanup_after = False
    if not overwrite and output_dir is None and not auto:
        cleanup_after = typer.confirm(
            "  Delete originals after successful encodes?", default=False
        )
    console.print()

    if not auto:
        if not typer.confirm("Start encoding?", default=True):
            return [], "cancel", False
    return jobs, "encode", cleanup_after
