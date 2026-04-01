from __future__ import annotations

from pathlib import Path

from rich.console import Console

from mediashrink.models import EncodeJob, EncodeResult
from mediashrink.progress import EncodingDisplay


def _job(source: Path, *, preview: bool = False, dry_run: bool = False) -> EncodeJob:
    suffix = "_preview" if preview else "_compressed"
    return EncodeJob(
        source=source,
        output=source.parent / f"{source.stem}{suffix}{source.suffix}",
        tmp_output=source.parent / f".tmp_{source.stem}{suffix}{source.suffix}",
        crf=20,
        preset="fast",
        dry_run=dry_run,
        skip=False,
    )


def test_show_summary_preview_uses_clip_language(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "movie.mkv"
    source.write_bytes(b"x" * 1000)
    result = EncodeResult(
        job=_job(source, preview=True),
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=2.0,
        media_duration_seconds=120.0,
    )

    EncodingDisplay(console).show_summary([result])
    output = console.export_text()

    assert "Preview summary" in output
    assert "Preview clip" in output
    assert "only 2m 00s" in output
    assert "not representative of the full encode" in output


def test_show_summary_prints_counts_and_failure_details(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    ok_source = tmp_path / "ok.mkv"
    failed_source = tmp_path / "failed.mkv"
    skipped_source = tmp_path / "skip.mkv"
    for path in (ok_source, failed_source, skipped_source):
        path.write_bytes(b"x" * 1000)

    ok = EncodeResult(
        job=_job(ok_source),
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=2.0,
        media_duration_seconds=120.0,
    )
    failed = EncodeResult(
        job=_job(failed_source),
        skipped=False,
        skip_reason=None,
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=1.0,
        error_message="Could not write header",
    )
    skipped = EncodeResult(
        job=_job(skipped_source),
        skipped=True,
        skip_reason="already compressed",
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=0.0,
    )

    EncodingDisplay(console).show_summary([failed, skipped, ok])
    output = console.export_text()

    assert "1 succeeded, 1 failed, 1 skipped" in output
    assert "Failure details" in output
    assert "failed.mkv: Could not write header" in output
