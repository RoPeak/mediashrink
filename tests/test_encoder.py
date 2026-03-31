from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mkv_compress.encoder import (
    build_ffmpeg_command,
    encode_file,
    get_duration_seconds,
    is_hardware_preset,
    parse_progress_line,
)
from mkv_compress.models import EncodeJob

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def _make_job(tmp_path: Path, **kwargs) -> EncodeJob:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"fake mkv data")
    output = tmp_path / "ep01_compressed.mkv"
    tmp_output = tmp_path / ".tmp_ep01_compressed.mkv"
    defaults = dict(
        source=source,
        output=output,
        tmp_output=tmp_output,
        crf=20,
        preset="slow",
        dry_run=False,
        skip=False,
        skip_reason=None,
    )
    defaults.update(kwargs)
    return EncodeJob(**defaults)


# ---------------------------------------------------------------------------
# build_ffmpeg_command
# ---------------------------------------------------------------------------

def test_build_ffmpeg_command_structure(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    cmd = build_ffmpeg_command(job, FFMPEG)

    assert cmd[0] == str(FFMPEG)
    assert "-i" in cmd
    assert str(job.source) in cmd
    assert "-map" in cmd
    assert "0" in cmd
    assert "-c:v" in cmd
    assert "libx265" in cmd
    assert "-crf" in cmd
    assert "20" in cmd
    assert "-preset" in cmd
    assert "slow" in cmd
    assert "-c:a" in cmd
    assert "copy" in cmd
    assert "-c:s" in cmd
    assert "-tag:v" in cmd
    assert "hvc1" in cmd
    assert "-progress" in cmd
    assert "pipe:1" in cmd
    assert str(job.tmp_output) == cmd[-1]


def test_build_ffmpeg_command_uses_job_crf(tmp_path: Path) -> None:
    job = _make_job(tmp_path, crf=28, preset="fast")
    cmd = build_ffmpeg_command(job, FFMPEG)
    crf_idx = cmd.index("-crf")
    assert cmd[crf_idx + 1] == "28"
    preset_idx = cmd.index("-preset")
    assert cmd[preset_idx + 1] == "fast"


def test_build_ffmpeg_command_qsv(tmp_path: Path) -> None:
    job = _make_job(tmp_path, crf=20, preset="qsv")
    cmd = build_ffmpeg_command(job, FFMPEG)
    assert "hevc_qsv" in cmd
    assert "libx265" not in cmd
    assert "-preset" not in cmd
    assert "-global_quality" in cmd
    q_idx = cmd.index("-global_quality")
    assert cmd[q_idx + 1] == "20"
    assert str(job.tmp_output) == cmd[-1]


def test_build_ffmpeg_command_nvenc(tmp_path: Path) -> None:
    job = _make_job(tmp_path, crf=22, preset="nvenc")
    cmd = build_ffmpeg_command(job, FFMPEG)
    assert "hevc_nvenc" in cmd
    assert "libx265" not in cmd
    assert "-cq" in cmd


def test_is_hardware_preset() -> None:
    assert is_hardware_preset("qsv") is True
    assert is_hardware_preset("nvenc") is True
    assert is_hardware_preset("amf") is True
    assert is_hardware_preset("fast") is False
    assert is_hardware_preset("slow") is False


# ---------------------------------------------------------------------------
# parse_progress_line
# ---------------------------------------------------------------------------

def test_parse_progress_line_basic() -> None:
    assert parse_progress_line("out_time_ms=5000000") == {"out_time_ms": "5000000"}


def test_parse_progress_line_with_whitespace() -> None:
    assert parse_progress_line("  frame=120  \n") == {"frame": "120"}


def test_parse_progress_line_no_equals() -> None:
    assert parse_progress_line("progress") == {}


def test_parse_progress_line_progress_end() -> None:
    assert parse_progress_line("progress=end") == {"progress": "end"}


# ---------------------------------------------------------------------------
# encode_file — success path
# ---------------------------------------------------------------------------

def test_encode_file_success(tmp_path: Path) -> None:
    job = _make_job(tmp_path)

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = iter(["out_time_ms=1000000\n", "progress=end\n"])

    def fake_rename(self: Path, dest: Path) -> Path:
        dest.write_bytes(b"compressed output")
        return dest

    with patch("mkv_compress.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mkv_compress.encoder.get_duration_seconds", return_value=10.0), \
         patch("pathlib.Path.rename", fake_rename):
        result = encode_file(job, FFMPEG, FFPROBE)

    assert result.success is True
    assert result.skipped is False
    assert result.error_message is None


def test_encode_file_reports_progress(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    reported: list[float] = []

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = iter([
        "out_time_ms=5000000\n",
        "out_time_ms=10000000\n",
        "progress=end\n",
    ])

    def fake_rename(self: Path, dest: Path) -> Path:
        dest.write_bytes(b"x")
        return dest

    with patch("mkv_compress.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mkv_compress.encoder.get_duration_seconds", return_value=10.0), \
         patch("pathlib.Path.rename", fake_rename):
        encode_file(job, FFMPEG, FFPROBE, progress_callback=reported.append)

    assert len(reported) == 2
    assert abs(reported[0] - 50.0) < 0.1
    assert abs(reported[1] - 100.0) < 0.1


# ---------------------------------------------------------------------------
# encode_file — failure path
# ---------------------------------------------------------------------------

def test_encode_file_failure_cleans_up_tmp(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    job.tmp_output.write_bytes(b"partial")

    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = iter([])

    with patch("mkv_compress.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mkv_compress.encoder.get_duration_seconds", return_value=10.0):
        result = encode_file(job, FFMPEG, FFPROBE)

    assert result.success is False
    assert result.error_message is not None
    assert not job.tmp_output.exists()
    assert job.source.exists()


# ---------------------------------------------------------------------------
# encode_file — skip path
# ---------------------------------------------------------------------------

def test_encode_file_skip(tmp_path: Path) -> None:
    job = _make_job(tmp_path, skip=True, skip_reason="already HEVC")

    with patch("mkv_compress.encoder.subprocess.Popen") as mock_popen:
        result = encode_file(job, FFMPEG, FFPROBE)

    mock_popen.assert_not_called()
    assert result.skipped is True
    assert result.skip_reason == "already HEVC"


# ---------------------------------------------------------------------------
# encode_file — dry run
# ---------------------------------------------------------------------------

def test_encode_file_dry_run_no_subprocess(tmp_path: Path) -> None:
    job = _make_job(tmp_path, dry_run=True)

    with patch("mkv_compress.encoder.subprocess.Popen") as mock_popen:
        result = encode_file(job, FFMPEG, FFPROBE)

    mock_popen.assert_not_called()
    assert result.success is True
    assert result.output_size_bytes == 0


# ---------------------------------------------------------------------------
# encode_file — keyboard interrupt
# ---------------------------------------------------------------------------

def test_encode_file_interrupt_cleans_up_tmp(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    job.tmp_output.write_bytes(b"partial")

    def raising_stdout():
        raise KeyboardInterrupt
        yield  # make it a generator

    mock_process = MagicMock()
    mock_process.stdout = raising_stdout()

    with patch("mkv_compress.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mkv_compress.encoder.get_duration_seconds", return_value=10.0):
        with pytest.raises(KeyboardInterrupt):
            encode_file(job, FFMPEG, FFPROBE)

    assert not job.tmp_output.exists()
    assert job.source.exists()
