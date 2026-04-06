from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from mediashrink.analysis import (
    analyze_files,
    build_manifest,
    describe_estimate_calibration,
    describe_estimate_confidence,
    display_analysis_summary,
    estimate_analysis_confidence,
    estimate_analysis_encode_seconds,
    save_manifest,
    select_representative_items,
)
from mediashrink.calibration import estimate_failure_rate, load_calibration_store
from mediashrink.constants import CRF_COMPRESSION_FACTOR
from mediashrink.encoder import (
    _HW_ENCODERS,
    encode_preview,
    estimate_output_size,
    get_duration_seconds,
    preflight_encode_job,
    probe_encoder_available,
    source_has_subtitle_streams,
    output_drops_subtitles,
    validate_encoder,
)
from mediashrink.models import AnalysisItem, EncodeJob, EncodeResult
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
    "amf": "AMF: quality may vary on older Radeon GPUs; test output before batch use.",
}
_DEFAULT_FALLBACK_PROFILE = ("faster", 22, "Fast", "faster")


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
    intent_label: str
    name: str
    encoder_key: str
    crf: int
    sw_preset: str | None
    estimated_output_bytes: int
    estimated_encode_seconds: float
    quality_label: str
    is_recommended: bool
    why_choose: str = ""
    is_custom: bool = False
    is_builtin: bool = False


_QUALITY_RANK = {
    "Acceptable": 0,
    "Good": 1,
    "Very good": 2,
    "Excellent": 3,
    "Visually lossless": 4,
}


def _quality_rank(label: str) -> int:
    return _QUALITY_RANK.get(label, -1)


def _is_profile_dominated(profile: EncoderProfile, peers: list[EncoderProfile]) -> bool:
    if profile.is_custom or profile.estimated_encode_seconds <= 0:
        return False

    profile_saved = profile.estimated_output_bytes
    profile_quality = _quality_rank(profile.quality_label)
    for peer in peers:
        if peer is profile or peer.is_custom or peer.estimated_encode_seconds <= 0:
            continue
        peer_saved = peer.estimated_output_bytes
        peer_quality = _quality_rank(peer.quality_label)
        if (
            peer.estimated_encode_seconds <= profile.estimated_encode_seconds
            and peer_saved <= profile_saved
            and peer_quality >= profile_quality
            and (
                peer.estimated_encode_seconds < profile.estimated_encode_seconds
                or peer_saved < profile_saved
                or peer_quality > profile_quality
            )
        ):
            return True
    return False


def _policy_sort_key(
    profile: EncoderProfile,
    *,
    policy: str,
    failure_rate: float,
) -> tuple[float, ...]:
    hardware_bias = 0.0 if profile.encoder_key in _HW_ENCODERS else 1.0
    quality_bias = float(_quality_rank(profile.quality_label) * -1)
    if policy == "best-compression":
        return (
            float(profile.estimated_output_bytes),
            quality_bias,
            float(profile.estimated_encode_seconds),
            failure_rate,
        )
    if policy == "lowest-cpu":
        return (
            hardware_bias,
            failure_rate,
            float(profile.estimated_encode_seconds),
            float(profile.estimated_output_bytes),
        )
    if policy == "highest-confidence":
        return (
            failure_rate,
            hardware_bias,
            float(profile.estimated_encode_seconds),
            quality_bias,
            float(profile.estimated_output_bytes),
        )
    return (
        float(profile.estimated_encode_seconds),
        failure_rate,
        quality_bias,
        float(profile.estimated_output_bytes),
    )


def _select_recommended_profile(
    profiles: list[EncoderProfile],
    *,
    policy: str = "fastest-wall-clock",
    failure_rates: dict[str, float] | None = None,
) -> EncoderProfile | None:
    candidates = [
        profile
        for profile in profiles
        if not profile.is_custom
        and profile.estimated_encode_seconds > 0
        and not _is_profile_dominated(profile, profiles)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda profile: _policy_sort_key(
            profile,
            policy=policy,
            failure_rate=(failure_rates or {}).get(profile.encoder_key, 0.0),
        ),
    )


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
            futures = {pool.submit(probe_encoder_available, key, ffmpeg): key for key in candidates}
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

    cmd = (
        [
            str(ffmpeg),
            "-ss",
            str(seek_pos),
            "-i",
            str(sample_file),
            "-t",
            str(clip_len),
        ]
        + video_flags
        + [
            "-an",
            "-f",
            "null",
            "-",
            "-loglevel",
            "error",
        ]
    )

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


def _sum_item_durations(items: list[AnalysisItem]) -> float:
    total = 0.0
    fallback_count = 0

    for item in items:
        if item.duration_seconds > 0:
            total += item.duration_seconds
        else:
            fallback_count += 1

    if total <= 0 and items:
        return 3600.0 * len(items)

    if fallback_count:
        avg_known = total / max(len(items) - fallback_count, 1)
        total += avg_known * fallback_count

    return total


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


def _encoder_display_name(
    encoder_key: str, device_labels: dict[str, str], truncate: bool = False
) -> str:
    if encoder_key in _HW_ENCODERS:
        label = device_labels.get(encoder_key)
        if label:
            if truncate and len(label) > _ENCODER_LABEL_MAX_CHARS:
                label = label[:_ENCODER_LABEL_MAX_CHARS].rstrip() + "…"
            return f"{_HARDWARE_DISPLAY_NAMES[encoder_key]} ({label})"
        return _HARDWARE_DISPLAY_NAMES[encoder_key]
    return f"libx265 ({encoder_key})"


def _profile_why_choose(profile: EncoderProfile, recommended: EncoderProfile | None) -> str:
    if profile.is_custom:
        return "Manual override for exact settings."
    if recommended is profile:
        return "Best default from the current time, size, and quality estimates."
    if profile.encoder_key in _HW_ENCODERS:
        return "Uses GPU hardware to reduce CPU load and keep the encode on the hardware path."
    if profile.intent_label == "GPU offload":
        return "Uses GPU hardware to reduce CPU load and stay on the hardware path."
    if profile.name == "Balanced":
        return "Higher quality at a moderate speed cost."
    if profile.name in {"Archival", "Archival+"}:
        return "Prioritizes retention over runtime."
    if profile.name in {"Smallest", "Smallest Acceptable"}:
        return "Pushes harder for smaller output sizes."
    return "Alternative trade-off if you prefer this balance."


def build_profiles(
    available_hw: list[str],
    benchmark_speeds: dict[str, float | None],
    total_media_seconds: float,
    total_input_bytes: int,
    *,
    candidate_items: list[AnalysisItem] | None = None,
    ffprobe: Path | None = None,
    policy: str = "fastest-wall-clock",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> list[EncoderProfile]:
    profiles: list[EncoderProfile] = []
    idx = 1
    active_calibration = (
        load_calibration_store()
        if use_calibration and calibration_store is None
        else calibration_store
    )

    def estimated_output_bytes_for(crf: int, preset: str) -> int:
        if not candidate_items or ffprobe is None:
            return _estimate_output_bytes(total_input_bytes, crf)
        total = 0
        for item in candidate_items:
            estimated = estimate_output_size(
                item.source,
                ffprobe,
                codec=item.codec,
                crf=crf,
                preset=preset,
                use_calibration=use_calibration,
                calibration_store=active_calibration,
            )
            total += estimated if estimated > 0 else item.estimated_output_bytes
        return total

    hardware_profiles: list[tuple[str, float | None]] = [
        (key, benchmark_speeds.get(key)) for key in ("qsv", "nvenc", "amf") if key in available_hw
    ]
    hardware_profiles.sort(key=lambda item: (item[1] or 0.0), reverse=True)

    for key, speed in hardware_profiles:
        profiles.append(
            EncoderProfile(
                index=idx,
                intent_label="GPU offload",
                name="Fastest GPU encode",
                encoder_key=key,
                crf=20,
                sw_preset=None,
                estimated_output_bytes=estimated_output_bytes_for(20, key),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label="Good",
                is_recommended=False,
                why_choose="",
            )
        )
        idx += 1

    fast_speed = benchmark_speeds.get("fast")
    faster_speed = benchmark_speeds.get("faster") or (fast_speed * 1.3 if fast_speed else None)
    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Fast",
            name="Fast",
            encoder_key="faster",
            crf=22,
            sw_preset="faster",
            estimated_output_bytes=estimated_output_bytes_for(22, "faster"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, faster_speed),
            quality_label="Very good",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Balanced",
            name="Balanced",
            encoder_key="fast",
            crf=20,
            sw_preset="fast",
            estimated_output_bytes=estimated_output_bytes_for(20, "fast"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, fast_speed),
            quality_label="Excellent",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    slow_speed = (fast_speed / 4) if fast_speed else None
    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Archival",
            name="Archival",
            encoder_key="slow",
            crf=18,
            sw_preset="slow",
            estimated_output_bytes=estimated_output_bytes_for(18, "slow"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Visually lossless",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Smallest",
            name="Smallest",
            encoder_key="slow",
            crf=28,
            sw_preset="slow",
            estimated_output_bytes=estimated_output_bytes_for(28, "slow"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Good",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    # Built-in intent presets
    _BUILTIN_QUALITY_LABELS = {
        "Fast Batch": "Very good",
        "Archival": "Visually lossless",
        "GPU Offload": "Good",
        "Smallest Acceptable": "Acceptable",
    }
    _BUILTIN_INTENTS = {
        "Fast Batch": "Fast",
        "Archival": "Archival",
        "GPU Offload": "GPU offload",
        "Smallest Acceptable": "Smallest",
    }
    best_hw = hardware_profiles[0][0] if hardware_profiles else None
    best_hw_speed = hardware_profiles[0][1] if hardware_profiles else None

    for bp in get_builtin_profiles():
        # "GPU Offload" uses the best available HW encoder, or falls back to sw
        if bp.name == "GPU Offload":
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
                intent_label=_BUILTIN_INTENTS.get(bp.name, "Balanced"),
                name=bp.name,
                encoder_key=encoder_key,
                crf=bp.crf,
                sw_preset=sw_preset,
                estimated_output_bytes=estimated_output_bytes_for(bp.crf, encoder_key),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label=_BUILTIN_QUALITY_LABELS.get(bp.name, "Good"),
                is_recommended=False,
                why_choose="",
                is_builtin=True,
            )
        )
        idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Custom",
            name="Custom",
            encoder_key="",
            crf=20,
            sw_preset=None,
            estimated_output_bytes=0,
            estimated_encode_seconds=0,
            quality_label="",
            is_recommended=False,
            why_choose="",
            is_custom=True,
        )
    )

    failure_rates = {
        profile.encoder_key: estimate_failure_rate(
            active_calibration,
            preset=profile.encoder_key,
            container=(candidate_items[0].source.suffix.lower() if candidate_items else ".mkv"),
        )
        for profile in profiles
        if not profile.is_custom
    }

    if hardware_profiles:
        timed_profiles = [
            profile
            for profile in profiles
            if not profile.is_custom and profile.estimated_encode_seconds > 0
        ]
        fastest_overall = (
            min(timed_profiles, key=lambda profile: profile.estimated_encode_seconds)
            if timed_profiles
            else None
        )
        hw_primary = next(
            (
                profile
                for profile in profiles
                if profile.encoder_key == best_hw and not profile.is_builtin
            ),
            None,
        )
        if hw_primary is not None and hw_primary is fastest_overall:
            hw_primary.name = "Fastest on this device"
        recommended = _select_recommended_profile(
            profiles,
            policy=policy,
            failure_rates=failure_rates,
        )
        if recommended is not None:
            recommended.is_recommended = True
        elif hw_primary is not None:
            hw_primary.is_recommended = True
    else:
        balanced = next((profile for profile in profiles if profile.name == "Balanced"), None)
        if balanced is not None and not _is_profile_dominated(balanced, profiles):
            balanced.is_recommended = True
        else:
            recommended = _select_recommended_profile(
                profiles,
                policy=policy,
                failure_rates=failure_rates,
            )
            if recommended is not None:
                recommended.is_recommended = True

    recommended = next((profile for profile in profiles if profile.is_recommended), None)
    for profile in profiles:
        profile.why_choose = _profile_why_choose(profile, recommended)

    return profiles


def display_profiles_table(
    profiles: list[EncoderProfile],
    total_input_bytes: int,
    candidate_count: int,
    device_labels: dict[str, str],
    console: Console,
    estimate_confidence: str | None = None,
    estimate_confidence_detail: str | None = None,
) -> None:
    compact = console.width < 120
    table = Table(
        title="Available encoding profiles",
        header_style="bold cyan",
        expand=True,
        show_lines=False,
    )
    table.add_column("#", justify="right", style="bold", no_wrap=True)
    table.add_column("Intent", no_wrap=True)
    table.add_column("Profile", no_wrap=True)
    table.add_column("Encoder", style="dim cyan")
    table.add_column("Est. Saving", justify="right", style="bold green", no_wrap=True)
    table.add_column("Est. Time", justify="right", no_wrap=True)
    table.add_column("Quality", no_wrap=True)
    table.add_column("Why choose this")
    if not compact:
        table.add_column("CRF", justify="center", no_wrap=True)

    for profile in profiles:
        if profile.is_custom:
            row: list[str | Text] = [
                str(profile.index),
                profile.intent_label,
                "Custom",
                "-",
                "-",
                "-",
                "-",
                "Manual override.",
            ]
            if not compact:
                row.append("-")
            table.add_row(*row)
            continue

        encoder_display = _encoder_display_name(profile.encoder_key, device_labels, truncate=True)
        saved = total_input_bytes - profile.estimated_output_bytes
        saved_pct = saved / total_input_bytes * 100 if total_input_bytes else 0
        est_saving = f"~{_fmt_size(saved)} ({saved_pct:.0f}%)"
        est_time = (
            f"~{_fmt_duration(profile.estimated_encode_seconds)}"
            if profile.estimated_encode_seconds > 0
            else "~unknown"
        )
        quality_style = {
            "Visually lossless": "green bold",
            "Excellent": "green",
            "Very good": "green",
            "Good": "yellow",
        }.get(profile.quality_label, "white")
        profile_name: str | Text = (
            Text.assemble(profile.name, " ", ("[recommended]", "bold cyan"))
            if profile.is_recommended
            else profile.name
        )

        row = [
            str(profile.index),
            profile.intent_label,
            profile_name,
            encoder_display,
            est_saving,
            est_time,
            Text(profile.quality_label, style=quality_style),
            profile.why_choose,
        ]
        if not compact:
            row.append(str(profile.crf))
        table.add_row(*row)

    recommended = next((profile for profile in profiles if profile.is_recommended), None)
    fastest = min(
        (
            profile
            for profile in profiles
            if not profile.is_custom and profile.estimated_encode_seconds > 0
        ),
        key=lambda profile: profile.estimated_encode_seconds,
        default=None,
    )

    console.print()
    console.print(table)
    console.print(
        f"  [dim]Likely encode candidates: {candidate_count} file(s) / {_fmt_size(total_input_bytes)}[/dim]"
    )
    console.print(
        "  [dim]Time and size numbers are approximate estimates for likely encode candidates, not already-skipped files.[/dim]"
    )
    console.print(
        "  [dim]Hardware presets are still full re-encodes; source bitrate, resolution, and runtime dominate total time.[/dim]"
    )
    if estimate_confidence is not None:
        console.print(
            f"  [dim]Estimate confidence: {estimate_confidence}"
            + (f" ({estimate_confidence_detail})" if estimate_confidence_detail else "")
            + ".[/dim]"
        )
    if fastest is not None:
        console.print(
            f"  [dim]Lowest estimated wait: {fastest.name} (~{_fmt_duration(fastest.estimated_encode_seconds)}).[/dim]"
        )
    if recommended is not None:
        console.print(
            f"  [dim]Default pick: {recommended.name} because it offers the best non-dominated trade-off for this batch.[/dim]"
        )
    if compact:
        console.print("  [dim]Compact view hides lower-priority columns on narrow terminals.[/dim]")
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
            enc_idx = (
                int(typer.prompt(f"Choose encoder [1-{len(encoder_choices)}]", default="1")) - 1
            )
        except ValueError:
            enc_idx = -1
        if 0 <= enc_idx < len(encoder_choices):
            break
        console.print("[yellow]Invalid choice.[/yellow]")

    chosen_key, _ = encoder_choices[enc_idx]

    while True:
        try:
            crf = int(
                typer.prompt("CRF quality value [0-51, lower = better quality]", default="20")
            )
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

    for item in sorted(
        items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True
    )[:12]:
        saving = (
            "-"
            if item.estimated_savings_bytes <= 0
            else f"~{_fmt_size(item.estimated_savings_bytes)}"
        )
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
    console.print(f"  1. Compress recommended only ({recommended_count} file(s))")
    if maybe_count:
        console.print(f"  2. Review maybe files ({maybe_count} file(s))")
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


def _subtitle_drop_warning(jobs: list[EncodeJob], ffprobe: Path) -> str | None:
    affected: list[str] = []
    for job in jobs:
        if job.skip or not output_drops_subtitles(job.output):
            continue
        if source_has_subtitle_streams(job.source, ffprobe):
            affected.append(job.source.name)
    if not affected:
        return None
    examples = ", ".join(affected[:3])
    return (
        f"{len(affected)} file(s) with subtitle streams will lose subtitles because "
        f"{jobs[0].output.suffix.lower()} outputs use '-sn' for compatibility."
        + (f" Examples: {examples}." if examples else "")
    )


def _preflight_candidates(jobs: list[EncodeJob]) -> list[EncodeJob]:
    non_mkv = [job for job in jobs if job.output.suffix.lower() != ".mkv"]
    if non_mkv:
        return non_mkv
    if not jobs:
        return []
    return [max(jobs, key=lambda job: job.source.stat().st_size)]


def _run_preflight_checks(
    jobs: list[EncodeJob],
    ffmpeg: Path,
    ffprobe: Path,
    *,
    crf: int,
    preset: str,
    console: Console,
) -> tuple[list[EncodeJob], list[tuple[EncodeJob, EncodeResult]]]:
    candidates = _preflight_candidates(jobs)
    compatible: list[EncodeJob] = []
    failed: list[tuple[EncodeJob, EncodeResult]] = []
    with console.status("[dim]Running final compatibility check...[/dim]", spinner="dots"):
        for job in candidates:
            result = preflight_encode_job(
                job.source,
                ffmpeg,
                ffprobe,
                crf=crf,
                preset=preset,
            )
            if result.success:
                compatible.append(job)
            else:
                failed.append((job, result))
    if len(candidates) == 1 and compatible:
        return jobs, failed
    compatible_sources = {job.source for job in compatible}
    passthrough = [job for job in jobs if job not in candidates]
    return passthrough + compatible, failed


def _select_profile_interactively(
    profiles: list[EncoderProfile],
    available_hw: list[str],
    device_labels: dict[str, str],
    auto: bool,
    console: Console,
) -> tuple[str, int, str | None, str]:
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

    console.print(
        f"[dim]Selected profile:[/dim] {display_label} "
        f"([dim]encoder:[/dim] {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}, "
        f"[dim]CRF:[/dim] {crf})"
    )
    if not selected.is_custom and selected.why_choose:
        console.print(f"  [dim]Why choose this:[/dim] {selected.why_choose}")
    return preset, crf, sw_preset, display_label


def _run_analysis_with_progress(
    files: list[Path],
    ffprobe: Path,
    console: Console,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
) -> list[AnalysisItem]:
    if not files:
        return []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console,
        transient=False,
        expand=True,
    ) as progress:
        task = progress.add_task(
            "[dim]Analyzing files (ffprobe + size estimates)...[/dim]", total=len(files)
        )

        def callback(completed: int, total: int, path: Path) -> None:
            name = path.name if len(path.name) <= 56 else path.name[:53] + "..."
            progress.update(
                task,
                total=total,
                completed=completed,
                description=f"[dim]Analyzing files (ffprobe + size estimates)...[/dim] [white]{name}[/white]",
            )

        items = analyze_files(
            files,
            ffprobe,
            progress_callback=callback,
            preset=preset,
            crf=crf,
            use_calibration=use_calibration,
        )
        progress.update(task, completed=len(files))
        return items


def _maybe_run_preview(
    preview_items: list[AnalysisItem],
    ffmpeg: Path,
    ffprobe: Path,
    preset: str,
    crf: int,
    auto: bool,
    console: Console,
) -> bool:
    if auto or not typer.confirm(
        "Test a 2-minute preview clip before the full batch?", default=False
    ):
        return True

    if not preview_items:
        return True

    preview_results: list[EncodeResult] = []
    console.print(f"[dim]Preview encoding {len(preview_items)} representative clip(s)...[/dim]")
    for item in preview_items:
        console.print(f"  [dim]Preview:[/dim] {item.source.name}")
        preview_results.append(
            encode_preview(
                source=item.source,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                duration_minutes=2.0,
                crf=crf,
                preset=preset,
            )
        )
    from mediashrink.progress import EncodingDisplay

    EncodingDisplay(console).show_summary(preview_results)
    successful_previews = [
        result for result in preview_results if result.success and result.job.output.exists()
    ]
    if successful_previews and all(result.success for result in preview_results):
        for result in successful_previews:
            console.print(f"  [dim]Preview saved:[/dim] {result.job.output}")
        console.print(
            "  [dim]Inspect video quality and verify audio/subtitle playback before continuing.[/dim]"
        )
        console.print(
            "  [dim]Use the preview clip as a quality check, not as a file-size estimate.[/dim]"
        )
        return True

    failed_preview = next((result for result in preview_results if not result.success), None)
    if failed_preview and failed_preview.error_message:
        console.print(f"[red]Preview encode failed:[/red] {failed_preview.error_message}")
    else:
        console.print("[red]Preview encode failed.[/red]")
    console.print(
        "[dim]This is likely an encoder configuration issue, not a problem with your files. "
        "If it persists, try a software profile (e.g. 'Fast').[/dim]"
    )
    return typer.confirm("Continue to full batch anyway?", default=False)


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
    policy: str = "fastest-wall-clock",
    on_file_failure: str = "retry",
    use_calibration: bool = True,
) -> tuple[list[EncodeJob], str, bool]:
    """Run the interactive wizard and return (jobs, action)."""
    console.print("\n[bold cyan]mediashrink wizard[/bold cyan]")
    console.print("[dim]Discovering supported files...[/dim]\n")

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

    analysis_items = _run_analysis_with_progress(
        files,
        ffprobe,
        console,
        use_calibration=use_calibration,
    )
    initial_confidence = estimate_analysis_confidence(analysis_items)
    display_analysis_summary(
        analysis_items,
        None,
        console,
        estimate_confidence=initial_confidence,
        estimate_confidence_detail=describe_estimate_confidence(analysis_items),
    )

    recommended_items = [item for item in analysis_items if item.recommendation == "recommended"]
    maybe_items = [item for item in analysis_items if item.recommendation == "maybe"]
    if not recommended_items:
        console.print("[dim]No recommended files were found for automatic compression.[/dim]")
        return [], "cancel", False

    candidate_items = recommended_items + maybe_items
    candidate_input_bytes = sum(item.size_bytes for item in candidate_items)
    candidate_media_seconds = _sum_item_durations(candidate_items)
    sample_pool = recommended_items or maybe_items or analysis_items
    sample_item = max(sample_pool, key=lambda item: item.size_bytes)
    sample_file = sample_item.source
    sample_duration = sample_item.duration_seconds if sample_item.duration_seconds > 0 else 3600.0
    preview_items = select_representative_items(candidate_items or sample_pool, limit=3)

    available_hw = detect_available_encoders(
        ffmpeg, console, sample_file=sample_file, ffprobe=ffprobe
    )
    device_labels = detect_device_labels()

    if available_hw:
        console.print("[green]Hardware encoders available:[/green]")
        for key in available_hw:
            console.print(f"  - {_encoder_display_name(key, device_labels)}")
    else:
        console.print("[dim]No hardware encoders detected. Software only.[/dim]")
    console.print(
        f"[dim]Next: benchmark a representative candidate, suggest profiles for likely encode files, and optionally preview "
        f"{sample_file.name} before batch encoding.[/dim]"
    )
    console.print(
        "[dim]Benchmark and profile estimates are sample-based and exclude files already expected to be skipped.[/dim]"
    )

    candidates_to_bench = list(available_hw) + ["fast", "faster"]
    benchmark_speeds: dict[str, float | None] = {}
    with console.status("[dim]Benchmarking profiles...[/dim]", spinner="dots"):
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
        total_media_seconds=candidate_media_seconds,
        total_input_bytes=candidate_input_bytes,
        candidate_items=candidate_items,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
    )
    profile_estimate_confidence = estimate_analysis_confidence(candidate_items, benchmarked_files=1)
    profile_calibration_detail = describe_estimate_calibration(
        candidate_items,
        preset=next(
            (profile.encoder_key for profile in profiles if profile.is_recommended), "fast"
        ),
        use_calibration=use_calibration,
    )
    display_profiles_table(
        profiles,
        candidate_input_bytes,
        len(candidate_items),
        device_labels,
        console,
        estimate_confidence=profile_estimate_confidence,
        estimate_confidence_detail=(
            describe_estimate_confidence(candidate_items, benchmarked_files=1)
            + (
                f"; local history: {profile_calibration_detail}"
                if profile_calibration_detail
                else ""
            )
        ),
    )

    for hw_key in available_hw:
        if hw_key in _HW_ENCODER_CAVEATS:
            console.print(
                f"  [dim yellow]Note:[/dim yellow] [dim]{_HW_ENCODER_CAVEATS[hw_key]}[/dim]"
            )

    action_taken = False
    profile_saved = False
    selected_items = list(recommended_items)
    estimated_total_encode_seconds: float | None = None
    while True:
        preset, crf, sw_preset, display_label = _select_profile_interactively(
            profiles, available_hw, device_labels, auto, console
        )

        if not auto and not profile_saved:
            maybe_save_profile(preset, crf, display_label, console)
            profile_saved = True

        if not _maybe_run_preview(preview_items, ffmpeg, ffprobe, preset, crf, auto, console):
            return [], "cancel", False

        if not action_taken:
            action = (
                "compress_recommended"
                if auto
                else prompt_analysis_action(len(recommended_items), len(maybe_items), console)
            )
            if action == "cancel":
                return [], "cancel", False
            if action == "export":
                export_estimate = estimate_analysis_encode_seconds(
                    items=recommended_items,
                    preset=preset,
                    crf=crf,
                    ffmpeg=ffmpeg,
                    known_speed=benchmark_speeds.get(preset),
                )
                manifest = build_manifest(
                    directory=directory,
                    recursive=recursive,
                    preset=preset,
                    crf=crf,
                    profile_name=None,
                    estimated_total_encode_seconds=export_estimate,
                    estimate_confidence=estimate_analysis_confidence(
                        recommended_items, benchmarked_files=1
                    ),
                    items=analysis_items,
                )
                default_manifest_path = directory / "mediashrink-analysis.json"
                manifest_path = Path(
                    typer.prompt("Manifest path", default=str(default_manifest_path))
                )
                save_manifest(manifest, manifest_path)
                console.print(f"[green]Wrote manifest[/green] {manifest_path}")
                return [], "export", False

            selected_items = list(recommended_items)
            if (
                action == "review_maybe"
                and maybe_items
                and review_maybe_items(maybe_items, console)
            ):
                selected_items.extend(maybe_items)
            action_taken = True

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

        estimated_total_encode_seconds = estimate_analysis_encode_seconds(
            items=selected_items,
            preset=preset,
            crf=crf,
            ffmpeg=ffmpeg,
            known_speed=benchmark_speeds.get(preset),
        )

        compatible_jobs, preflight_failures = _run_preflight_checks(
            to_encode,
            ffmpeg,
            ffprobe,
            crf=crf,
            preset=preset,
            console=console,
        )
        if not preflight_failures:
            break

        console.print()
        console.print(
            "[red]Selected settings failed a short compatibility check before batch encoding.[/red]"
        )
        console.print(
            f"[red]Profile:[/red] {display_label} "
            f"([red]encoder:[/red] {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'})"
        )
        for failed_job, preflight_result in preflight_failures[:5]:
            if preflight_result.error_message:
                console.print(
                    f"[red]{failed_job.source.name}:[/red] {preflight_result.error_message}"
                )
        if len(preflight_failures) > 5:
            console.print(
                f"[dim]...and {len(preflight_failures) - 5} more compatibility failure(s).[/dim]"
            )
        if any(job.output.suffix.lower() != ".mkv" for job, _ in preflight_failures):
            console.print(
                "[dim]These failures are often caused by non-video/data streams that .mp4/.m4v outputs cannot safely carry.[/dim]"
            )

        if compatible_jobs and len(compatible_jobs) < len(to_encode):
            compatible_sources = {job.source for job in compatible_jobs}
            skip_default = True
            skip_prompt = (
                f"Skip {len(preflight_failures)} incompatible file(s) and continue with "
                f"{len(compatible_jobs)} compatible file(s)?"
            )
            if (
                on_file_failure == "skip"
                or auto
                or typer.confirm(skip_prompt, default=skip_default)
            ):
                selected_items = [
                    item for item in selected_items if item.source in compatible_sources
                ]
                jobs = [job for job in jobs if job.skip or job.source in compatible_sources]
                to_encode = compatible_jobs
                estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                    items=selected_items,
                    preset=preset,
                    crf=crf,
                    ffmpeg=ffmpeg,
                    known_speed=benchmark_speeds.get(preset),
                )
                console.print(
                    f"[yellow]Skipping {len(preflight_failures)} incompatible file(s) for this run.[/yellow]"
                )
                break

        fallback_preset, fallback_crf, fallback_label, fallback_sw_preset = (
            _DEFAULT_FALLBACK_PROFILE
        )
        should_try_fallback = on_file_failure != "stop" and (
            auto
            or typer.confirm(
                f"Switch to {fallback_label} (libx265, CRF {fallback_crf}) and retry?",
                default=True,
            )
        )
        if preset != fallback_preset and should_try_fallback:
            fallback_jobs = build_jobs(
                files=[item.source for item in selected_items],
                output_dir=output_dir,
                overwrite=overwrite,
                crf=fallback_crf,
                preset=fallback_preset,
                dry_run=False,
                ffprobe=ffprobe,
                no_skip=no_skip,
            )
            fallback_to_encode = [job for job in fallback_jobs if not job.skip]
            console.print(f"[dim]Retrying with[/dim] {fallback_label}...")
            fallback_compatible, fallback_failures = _run_preflight_checks(
                fallback_to_encode,
                ffmpeg,
                ffprobe,
                crf=fallback_crf,
                preset=fallback_preset,
                console=console,
            )
            if not fallback_failures:
                jobs = fallback_jobs
                to_encode = fallback_compatible
                preset = fallback_preset
                crf = fallback_crf
                sw_preset = fallback_sw_preset
                display_label = fallback_label
                estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                    items=selected_items,
                    preset=preset,
                    crf=crf,
                    ffmpeg=ffmpeg,
                    known_speed=benchmark_speeds.get(preset),
                )
                break
            if fallback_compatible and len(fallback_compatible) < len(fallback_to_encode):
                fallback_sources = {job.source for job in fallback_compatible}
                if (
                    on_file_failure == "skip"
                    or auto
                    or typer.confirm(
                        f"Skip {len(fallback_failures)} incompatible file(s) and continue with {len(fallback_compatible)} compatible file(s) using {fallback_label}?",
                        default=True,
                    )
                ):
                    selected_items = [
                        item for item in selected_items if item.source in fallback_sources
                    ]
                    jobs = [
                        job for job in fallback_jobs if job.skip or job.source in fallback_sources
                    ]
                    to_encode = fallback_compatible
                    preset = fallback_preset
                    crf = fallback_crf
                    sw_preset = fallback_sw_preset
                    display_label = fallback_label
                    estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                        items=selected_items,
                        preset=preset,
                        crf=crf,
                        ffmpeg=ffmpeg,
                        known_speed=benchmark_speeds.get(preset),
                    )
                    console.print(
                        f"[yellow]Skipping {len(fallback_failures)} incompatible file(s) for this run.[/yellow]"
                    )
                    break
            console.print()
            console.print("[red]Fallback profile also failed compatibility checks.[/red]")
            for failed_job, fallback_result in fallback_failures[:5]:
                if fallback_result.error_message:
                    console.print(
                        f"[red]{failed_job.source.name}:[/red] {fallback_result.error_message}"
                    )

        if on_file_failure == "stop":
            return [], "cancel", False

        console.print(
            "[dim]Choose another profile. Software profiles are the safest fallback when hardware encoding is not compatible with the selected files.[/dim]"
        )

    console.print()
    console.print("[bold]Ready to encode[/bold]")
    console.print(f"  Files:    {len(to_encode)}")
    console.print(
        f"  Encoder:  {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}"
    )
    console.print(f"  CRF:      {crf}")
    selected_input_bytes = sum(item.size_bytes for item in selected_items)
    selected_output_bytes = sum(
        item.estimated_output_bytes for item in selected_items if item.estimated_output_bytes > 0
    )
    selected_saved_bytes = sum(item.estimated_savings_bytes for item in selected_items)
    selected_saved_pct = (
        selected_saved_bytes / selected_input_bytes * 100 if selected_input_bytes else 0.0
    )
    console.print(f"  Input:    {_fmt_size(selected_input_bytes)}")
    not_selected_count = len(analysis_items) - len(selected_items)
    if not_selected_count > 0:
        console.print(
            f"  [yellow]Not in this run:[/yellow] {not_selected_count} file(s) left out as maybe/skip candidates."
        )
    if selected_output_bytes > 0:
        console.print(
            f"  Est. out: ~{_fmt_size(selected_output_bytes)}  "
            f"(~{_fmt_size(selected_saved_bytes)} saved, ~{selected_saved_pct:.0f}%)"
        )
    if estimated_total_encode_seconds is not None and estimated_total_encode_seconds > 0:
        console.print(f"  Est. time: ~{_fmt_duration(estimated_total_encode_seconds)}")
    console.print(
        f"  Estimate confidence: {estimate_analysis_confidence(selected_items, benchmarked_files=1)}"
    )
    if preset in _HW_ENCODERS:
        console.print(
            "  [dim]Hardware encoding is faster, but source duration and bitrate still dominate total runtime.[/dim]"
        )
    elif preset in {"faster", "ultrafast"}:
        console.print(
            "  [dim]This favors faster completion over maximum compression efficiency.[/dim]"
        )
    else:
        console.print(
            "  [dim]Slower software presets trade more time for slightly smaller files.[/dim]"
        )

    if not to_encode[0].output.parent.exists():
        console.print(f"  Output:   {to_encode[0].output.parent}")
    subtitle_warning = _subtitle_drop_warning(to_encode, ffprobe)
    if subtitle_warning:
        console.print(f"  [yellow]{subtitle_warning}[/yellow]")
    console.print("  [dim]Estimates are approximate.[/dim]")
    console.print(
        "  [dim]Safe to stop with Ctrl+C: completed files stay done, the current temp output is discarded, and you can resume unfinished files later.[/dim]"
    )

    cleanup_after = False
    if not overwrite and output_dir is None and not auto:
        cleanup_after = typer.confirm(
            "  Delete originals only after successful side-by-side encodes?",
            default=False,
        )
    console.print()

    if not auto:
        if not typer.confirm("Start encoding?", default=True):
            return [], "cancel", False
    return jobs, "encode", cleanup_after
