from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mkv_compress.cli import app
from mkv_compress.models import EncodeJob, EncodeResult

runner = CliRunner()

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def _make_result(job: EncodeJob, **kwargs) -> EncodeResult:
    defaults = dict(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1_000_000_000,
        output_size_bytes=300_000_000,
        duration_seconds=120.0,
    )
    defaults.update(kwargs)
    return EncodeResult(**defaults)


def _make_job(source: Path) -> EncodeJob:
    return EncodeJob(
        source=source,
        output=source.with_stem(source.stem + "_compressed"),
        tmp_output=source.parent / f".tmp_{source.stem}_compressed.mkv",
        crf=20,
        preset="slow",
        dry_run=False,
        skip=False,
        skip_reason=None,
    )


# ---------------------------------------------------------------------------
# Missing directory
# ---------------------------------------------------------------------------

def test_missing_directory_error() -> None:
    result = runner.invoke(app, ["/nonexistent/path"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# FFmpeg not found
# ---------------------------------------------------------------------------

def test_ffmpeg_not_found_error(tmp_path: Path) -> None:
    (tmp_path / "ep01.mkv").write_bytes(b"fake")

    with patch(
        "mkv_compress.cli.check_ffmpeg_available",
        return_value=(False, "ffmpeg not found"),
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 1
    assert "ffmpeg not found" in result.stdout.lower() or "Error" in result.stdout


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------

def test_no_mkv_files(tmp_path: Path) -> None:
    (tmp_path / "movie.mp4").touch()

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    assert "No .mkv files" in result.stdout


# ---------------------------------------------------------------------------
# Dry run — no encoding
# ---------------------------------------------------------------------------

def test_dry_run_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.dry_run = True
    fake_result = _make_result(fake_job, output_size_bytes=0)

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file", return_value=fake_result) as mock_encode:
        result = runner.invoke(app, [str(tmp_path), "--dry-run", "--yes"])

    # encode_file is still called (it handles dry_run internally), but FFmpeg subprocess is not
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --yes flag skips confirmation prompt
# ---------------------------------------------------------------------------

def test_yes_flag_skips_prompt(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_result = _make_result(fake_job)

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
        # With --yes, should not prompt; without it, CliRunner would need input
        result = runner.invoke(app, [str(tmp_path), "--yes"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# All files skipped
# ---------------------------------------------------------------------------

def test_all_skipped_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.skip = True
    fake_job.skip_reason = "already HEVC"

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file") as mock_encode:
        result = runner.invoke(app, [str(tmp_path)])

    mock_encode.assert_not_called()
    assert result.exit_code == 0
    assert "Nothing to encode" in result.stdout
