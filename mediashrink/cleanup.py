from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mediashrink.encoder import (
    get_duration_seconds,
    output_passes_safety_check,
    probe_stream_type_counts,
)
from mediashrink.models import EncodeResult


def eligible_cleanup_results(results: list[EncodeResult]) -> list[EncodeResult]:
    """Return successful side-by-side results that can be cleaned up safely."""
    eligible: list[EncodeResult] = []
    for result in results:
        if not result.success or result.skipped or result.job.dry_run:
            continue
        if not output_passes_safety_check(result.input_size_bytes, result.output_size_bytes):
            continue
        source = result.job.source
        output = result.job.output
        if source == output or not output.exists() or not source.exists():
            continue
        if source.suffix.lower() != output.suffix.lower():
            continue
        eligible.append(result)
    return eligible


def cleanup_successful_results(results: list[EncodeResult]) -> list[Path]:
    """
    Replace original sources with successful compressed outputs.

    This only operates on results from the current run whose compressed output exists
    separately from the source path. The source is moved to a temporary backup first
    so a failed move can be rolled back safely.
    """
    cleaned: list[Path] = []
    for result in eligible_cleanup_results(results):
        cleaned.append(_replace_source_with_sidecar(result.job.source, result.job.output))

    return cleaned


@dataclass(frozen=True)
class RecoverableSidecar:
    source: Path
    sidecar: Path
    reason: str


def _replace_source_with_sidecar(source: Path, sidecar: Path) -> Path:
    backup = source.with_name(f".cleanup_backup_{source.name}")

    if backup.exists():
        backup.unlink()

    source.replace(backup)
    try:
        shutil.move(str(sidecar), str(source))
    except Exception:
        if backup.exists():
            backup.replace(source)
        raise
    else:
        if backup.exists():
            backup.unlink()
    return source


def reconcile_recoverable_sidecars(pairs: list[RecoverableSidecar]) -> list[Path]:
    reconciled: list[Path] = []
    for pair in pairs:
        reconciled.append(_replace_source_with_sidecar(pair.source, pair.sidecar))
    return reconciled


def _duration_matches(source: Path, sidecar: Path, ffprobe: Path) -> bool:
    source_duration = get_duration_seconds(source, ffprobe)
    sidecar_duration = get_duration_seconds(sidecar, ffprobe)
    if source_duration <= 0 or sidecar_duration <= 0:
        return False
    tolerance = max(2.0, source_duration * 0.02)
    return abs(source_duration - sidecar_duration) <= tolerance


def _stream_layout_matches(source: Path, sidecar: Path, ffprobe: Path) -> bool:
    return probe_stream_type_counts(source, ffprobe) == probe_stream_type_counts(sidecar, ffprobe)


def find_recoverable_sidecars(files: list[Path], ffprobe: Path) -> list[RecoverableSidecar]:
    recoverable: list[RecoverableSidecar] = []
    seen_sources: set[Path] = set()

    for source in files:
        if source in seen_sources or "_compressed" in source.stem.lower():
            continue
        sidecar = source.with_stem(source.stem + "_compressed")
        if not source.exists() or not sidecar.exists():
            continue
        if source.suffix.lower() != sidecar.suffix.lower():
            continue
        try:
            source_size = source.stat().st_size
            sidecar_size = sidecar.stat().st_size
        except OSError:
            continue
        if sidecar_size <= 0 or not output_passes_safety_check(source_size, sidecar_size):
            continue
        if not _duration_matches(source, sidecar, ffprobe):
            continue
        if not _stream_layout_matches(source, sidecar, ffprobe):
            continue
        recoverable.append(
            RecoverableSidecar(
                source=source,
                sidecar=sidecar,
                reason="completed same-format sidecar output looks safe to restore",
            )
        )
        seen_sources.add(source)

    return recoverable
