from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mediashrink.encoder import estimate_output_size, get_duration_seconds, get_video_bitrate_kbps
from mediashrink.models import AnalysisItem, AnalysisManifest
from mediashrink.scanner import is_already_compressed, probe_video_codec, scan_directory

MANIFEST_VERSION = 1

_GB = 1024**3
_MB = 1024**2
_RECOMMENDED_CODECS = {"h264", "vc1", "mpeg2video"}

_MIN_SKIP_SAVINGS_BYTES = 250 * _MB
_MIN_SKIP_SAVINGS_PCT = 10.0
_MIN_RECOMMENDED_SAVINGS_BYTES = 1 * _GB
_MIN_RECOMMENDED_SAVINGS_PCT = 25.0


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


def build_analysis_item(path: Path, ffprobe: Path) -> AnalysisItem:
    codec = probe_video_codec(path, ffprobe)
    skip, skip_reason = is_already_compressed(path, ffprobe, codec=codec)
    size_bytes = path.stat().st_size
    duration_seconds = get_duration_seconds(path, ffprobe)
    bitrate_kbps = get_video_bitrate_kbps(path, ffprobe)
    estimated_output_bytes = 0 if skip else estimate_output_size(path, ffprobe)
    estimated_savings_bytes = (
        max(size_bytes - estimated_output_bytes, 0) if estimated_output_bytes > 0 else 0
    )

    if skip:
        reason_code = "already_hevc" if codec == "hevc" else "already_marked_compressed"
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="skip",
            reason_code=reason_code,
            reason_text=skip_reason,
        )

    if estimated_output_bytes <= 0:
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=0,
            estimated_savings_bytes=0,
            recommendation="maybe",
            reason_code="estimate_unavailable",
            reason_text="could not estimate compressed size confidently",
        )

    savings_pct = estimated_savings_bytes / size_bytes * 100 if size_bytes else 0.0
    if estimated_savings_bytes < _MIN_SKIP_SAVINGS_BYTES or savings_pct < _MIN_SKIP_SAVINGS_PCT:
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="skip",
            reason_code="savings_too_small",
            reason_text="estimated savings are too small to justify recompression",
        )

    if codec in _RECOMMENDED_CODECS and (
        estimated_savings_bytes >= _MIN_RECOMMENDED_SAVINGS_BYTES
        and savings_pct >= _MIN_RECOMMENDED_SAVINGS_PCT
    ):
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="recommended",
            reason_code="strong_savings_candidate",
            reason_text="legacy codec with strong projected space savings",
        )

    return AnalysisItem(
        source=path,
        codec=codec,
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        bitrate_kbps=bitrate_kbps,
        estimated_output_bytes=estimated_output_bytes,
        estimated_savings_bytes=estimated_savings_bytes,
        recommendation="maybe",
        reason_code="borderline_candidate",
        reason_text="projected savings may be worthwhile, but the case is not strong enough for auto-selection",
    )


def analyze_directory(directory: Path, recursive: bool, ffprobe: Path) -> list[AnalysisItem]:
    files = scan_directory(directory, recursive=recursive)
    return [build_analysis_item(path, ffprobe) for path in files]


def estimate_analysis_encode_seconds(
    items: list[AnalysisItem],
    preset: str,
    crf: int,
    ffmpeg: Path,
    known_speed: float | None = None,
) -> float | None:
    recommended = [item for item in items if item.recommendation == "recommended"]
    if not recommended:
        return 0.0

    speed: float | None
    if known_speed is not None and known_speed > 0:
        speed = known_speed
    else:
        from mediashrink.wizard import benchmark_encoder

        sample = recommended[0]
        if sample.duration_seconds <= 0:
            return None
        speed = benchmark_encoder(
            encoder_key=preset,
            sample_file=sample.source,
            sample_duration=sample.duration_seconds,
            crf=crf,
            ffmpeg=ffmpeg,
        )

    if speed is None or speed <= 0:
        return None

    total_media_seconds = sum(
        item.duration_seconds for item in recommended if item.duration_seconds > 0
    )
    return total_media_seconds / speed if total_media_seconds > 0 else None


def build_manifest(
    directory: Path,
    recursive: bool,
    preset: str,
    crf: int,
    profile_name: str | None,
    estimated_total_encode_seconds: float | None,
    items: list[AnalysisItem],
) -> AnalysisManifest:
    return AnalysisManifest(
        version=MANIFEST_VERSION,
        analyzed_directory=directory,
        recursive=recursive,
        preset=preset,
        crf=crf,
        profile_name=profile_name,
        estimated_total_encode_seconds=estimated_total_encode_seconds,
        items=[item for item in items if item.recommendation == "recommended"],
    )


def save_manifest(manifest: AnalysisManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> AnalysisManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")
    manifest = AnalysisManifest.from_dict(raw)
    if manifest.version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.version}")
    return manifest


def display_analysis_summary(
    items: list[AnalysisItem],
    estimated_total_encode_seconds: float | None,
    console: Console,
) -> None:
    recommended = [item for item in items if item.recommendation == "recommended"]
    maybe = [item for item in items if item.recommendation == "maybe"]
    skipped = [item for item in items if item.recommendation == "skip"]

    table = Table(title="Compression analysis", header_style="bold cyan", expand=True)
    table.add_column("File")
    table.add_column("Codec", justify="center", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Est. Saving", justify="right", no_wrap=True)
    table.add_column("Recommendation", justify="center", no_wrap=True)
    table.add_column("Reason")

    for item in sorted(
        items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True
    )[:12]:
        savings_text = (
            "-"
            if item.estimated_savings_bytes <= 0
            else f"~{_fmt_size(item.estimated_savings_bytes)}"
        )
        table.add_row(
            item.source.name,
            item.codec or "?",
            _fmt_size(item.size_bytes),
            savings_text,
            item.recommendation.upper(),
            item.reason_text,
        )

    total_current = sum(item.size_bytes for item in recommended)
    total_estimated = sum(item.estimated_output_bytes for item in recommended)
    total_saved = sum(item.estimated_savings_bytes for item in recommended)
    total_saved_pct = total_saved / total_current * 100 if total_current else 0.0

    console.print()
    console.print(table)
    console.print(
        f"[bold]{len(items)}[/bold] file(s) scanned - "
        f"[green bold]{len(recommended)}[/green bold] recommended, "
        f"[yellow]{len(maybe)}[/yellow] maybe, "
        f"[dim]{len(skipped)}[/dim] skip",
        highlight=False,
    )
    if recommended:
        console.print(
            f"Recommended set: [yellow]{_fmt_size(total_current)}[/yellow] -> "
            f"[green]~{_fmt_size(total_estimated)}[/green] "
            f"([bold green]~{_fmt_size(total_saved)} saved, ~{total_saved_pct:.0f}%[/bold green])",
            highlight=False,
        )
    if estimated_total_encode_seconds is not None and estimated_total_encode_seconds > 0:
        console.print(
            f"Rough encode time: [cyan]~{_fmt_duration(estimated_total_encode_seconds)}[/cyan]"
        )
    console.print("[dim]Analysis estimates are approximate.[/dim]")
    console.print(
        "[dim]Hardware encoders are faster, but source duration, bitrate, and resolution still dominate total runtime.[/dim]"
    )
    console.print()
