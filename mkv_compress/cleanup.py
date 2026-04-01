from __future__ import annotations

import shutil
from pathlib import Path

from mkv_compress.models import EncodeResult


def eligible_cleanup_results(results: list[EncodeResult]) -> list[EncodeResult]:
    """Return successful side-by-side results that can be cleaned up safely."""
    eligible: list[EncodeResult] = []
    for result in results:
        if not result.success or result.skipped or result.job.dry_run:
            continue
        source = result.job.source
        output = result.job.output
        if source == output or not output.exists() or not source.exists():
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
        source = result.job.source
        output = result.job.output
        backup = source.with_name(f".cleanup_backup_{source.name}")

        if backup.exists():
            backup.unlink()

        source.replace(backup)
        try:
            shutil.move(str(output), str(source))
        except Exception:
            if backup.exists():
                backup.replace(source)
            raise
        else:
            if backup.exists():
                backup.unlink()
            cleaned.append(source)

    return cleaned
