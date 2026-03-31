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

from mkv_compress.encoder import _HW_ENCODERS, get_duration_seconds, probe_encoder_available
from mkv_compress.models import EncodeJob
from mkv_compress.platform_utils import detect_device_labels
from mkv_compress.profiles import SavedProfile, upsert_profile
from mkv_compress.scanner import build_jobs, scan_directory

_GB = 1024**3
_MB = 1024**2

_CRF_COMPRESSION_FACTOR: dict[int, float] = {
    18: 0.50,
    20: 0.40,
    24: 0.30,
    28: 0.22,
}

_BENCHMARK_SECONDS = 8
_HARDWARE_DISPLAY_NAMES = {
    "qsv": "Intel Quick Sync",
    "nvenc": "Nvidia NVENC",
    "amf": "AMD AMF",
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


def detect_available_encoders(ffmpeg: Path, console: Console) -> list[str]:
    """Return available hardware encoder keys in stable display order."""
    available: list[str] = []
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


def _encoder_display_name(encoder_key: str, device_labels: dict[str, str]) -> str:
    if encoder_key in _HW_ENCODERS:
        label = device_labels.get(encoder_key)
        if label:
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

        encoder_display = _encoder_display_name(profile.encoder_key, device_labels)
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


def run_wizard(
    directory: Path,
    ffmpeg: Path,
    ffprobe: Path,
    recursive: bool,
    output_dir: Path | None,
    overwrite: bool,
    no_skip: bool,
    console: Console,
) -> tuple[list[EncodeJob], bool]:
    """Run the interactive wizard and return (jobs, confirmed)."""
    console.print("\n[bold cyan]mkvcompress wizard[/bold cyan]")
    console.print("[dim]Scanning files and detecting hardware...[/dim]\n")

    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(f"[yellow]No .mkv files found in[/yellow] {directory}")
        return [], False

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

    available_hw = detect_available_encoders(ffmpeg, console)
    device_labels = detect_device_labels()

    if available_hw:
        console.print("[green]Hardware encoders available:[/green]")
        for key in available_hw:
            console.print(f"  - {_encoder_display_name(key, device_labels)}")
    else:
        console.print("[dim]No hardware encoders detected. Software only.[/dim]")

    candidates_to_bench = list(available_hw) + ["fast"]
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

    selected = prompt_profile_selection(profiles, console)
    if selected.is_custom:
        preset, crf, sw_preset = run_custom_wizard(available_hw, console)
        display_label = f"Custom ({preset}, CRF {crf})"
    else:
        preset = selected.encoder_key
        crf = selected.crf
        sw_preset = selected.sw_preset
        display_label = selected.name

    jobs = build_jobs(
        files=files,
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
        console.print("[dim]All files are already H.265. Nothing to encode.[/dim]")
        return jobs, False

    maybe_save_profile(preset, crf, display_label, console)

    console.print()
    console.print("[bold]Ready to encode[/bold]")
    console.print(f"  Files:    {len(to_encode)}")
    console.print(f"  Encoder:  {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}")
    console.print(f"  CRF:      {crf}")
    console.print(f"  Input:    {_fmt_size(total_input_bytes)}")

    est_out = _estimate_output_bytes(total_input_bytes, crf)
    saved = total_input_bytes - est_out
    saved_pct = saved / total_input_bytes * 100 if total_input_bytes else 0
    console.print(f"  Est. out: ~{_fmt_size(est_out)}  (~{_fmt_size(saved)} saved, ~{saved_pct:.0f}%)")

    chosen_speed = benchmark_speeds.get(preset)
    if chosen_speed is None and sw_preset == "slow":
        fast_speed = benchmark_speeds.get("fast")
        chosen_speed = (fast_speed / 4) if fast_speed else None
    est_time = _estimate_time(total_media_seconds, chosen_speed)
    if est_time > 0:
        console.print(f"  Est. time: ~{_fmt_duration(est_time)}")

    if not to_encode[0].output.parent.exists():
        console.print(f"  Output:   {to_encode[0].output.parent}")
    console.print("  [dim]Estimates are approximate.[/dim]")
    console.print()

    confirmed = typer.confirm("Start encoding?", default=True)
    return jobs, confirmed
