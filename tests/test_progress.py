from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from mediashrink.models import EncodeJob, EncodeResult
from mediashrink.progress import EncodingDisplay, _FileCountsColumn


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


def test_make_progress_bar_uses_stable_columns() -> None:
    console = Console(record=True, width=100)

    progress = EncodingDisplay(console).make_progress_bar()
    column_names = [type(column).__name__ for column in progress.columns]

    assert "DownloadColumn" not in column_names
    assert "_CompletedSizeColumn" in column_names
    assert "_EtaColumn" in column_names
    assert "_FileCountsColumn" in column_names
    assert "_HeartbeatColumn" in column_names
    assert "_LastUpdateColumn" in column_names
    assert progress.expand is False


def test_make_progress_bar_uses_resize_safe_windows_layout() -> None:
    console = Console(record=True, width=160)
    display = EncodingDisplay(console)

    with patch("mediashrink.progress.detect_os", return_value="Windows"):
        progress = display.make_progress_bar()

    assert display._progress_layout_width == 108
    assert progress.expand is False
    bar_column = next(column for column in progress.columns if type(column).__name__ == "BarColumn")
    assert bar_column.bar_width == 26


def test_show_summary_mentions_resumed_context(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "movie.mkv"
    source.write_bytes(b"x" * 1000)
    result = EncodeResult(
        job=_job(source),
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=2.0,
        media_duration_seconds=120.0,
    )

    EncodingDisplay(console).show_summary(
        [result],
        resumed_from_session=True,
        previously_completed=3,
        previously_skipped=1,
    )
    output = console.export_text()

    assert "Resumed run:" in output
    assert "3 file(s) were already complete, 1 already skipped" in output


def test_show_summary_breaks_down_incompatible_and_policy_skips(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    incompatible_source = tmp_path / "bad.mp4"
    policy_source = tmp_path / "policy.mkv"
    ok_source = tmp_path / "ok.mkv"
    for path in (incompatible_source, policy_source, ok_source):
        path.write_bytes(b"x" * 1000)

    incompatible = EncodeResult(
        job=_job(incompatible_source),
        skipped=True,
        skip_reason="incompatible: unsupported container/stream combination",
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=0.0,
    )
    skipped_by_policy = EncodeResult(
        job=_job(policy_source),
        skipped=True,
        skip_reason="skipped_by_policy: disk full",
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=0.0,
    )
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

    EncodingDisplay(console).show_summary([incompatible, skipped_by_policy, ok])
    output = console.export_text()

    assert "Skip breakdown:" in output
    assert "1 incompatible" in output
    assert "1 skipped by policy" in output
    assert "Incompatible files skipped" in output
    assert "bad.mp4" in output


def test_file_counts_column_uses_processed_batch_wording() -> None:
    class DummyTask:
        fields = {
            "task_kind": "overall",
            "processed_files": 3,
            "remaining_files": 2,
            "succeeded_files": 1,
            "failed_files": 1,
            "skipped_files": 1,
        }

    rendered = _FileCountsColumn().render(DummyTask())

    assert rendered.plain == "processed 3 / rem 2 / ok 1 / fail 1 / skip 1"
