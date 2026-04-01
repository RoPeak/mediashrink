from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mediashrink.encoder import (
    _CODEC_BASE_FACTOR,
    build_ffmpeg_command,
    encode_file,
    estimate_output_size,
    get_duration_seconds,
    get_video_resolution,
    is_hardware_preset,
    parse_progress_line,
)
from mediashrink.models import EncodeJob

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

    with patch("mediashrink.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mediashrink.encoder.get_duration_seconds", return_value=10.0), \
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

    with patch("mediashrink.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mediashrink.encoder.get_duration_seconds", return_value=10.0), \
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

    with patch("mediashrink.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mediashrink.encoder.get_duration_seconds", return_value=10.0):
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

    with patch("mediashrink.encoder.subprocess.Popen") as mock_popen:
        result = encode_file(job, FFMPEG, FFPROBE)

    mock_popen.assert_not_called()
    assert result.skipped is True
    assert result.skip_reason == "already HEVC"


# ---------------------------------------------------------------------------
# encode_file — dry run
# ---------------------------------------------------------------------------

def test_encode_file_dry_run_no_subprocess(tmp_path: Path) -> None:
    job = _make_job(tmp_path, dry_run=True)

    with patch("mediashrink.encoder.subprocess.Popen") as mock_popen:
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

    with patch("mediashrink.encoder.subprocess.Popen", return_value=mock_process), \
         patch("mediashrink.encoder.get_duration_seconds", return_value=10.0):
        with pytest.raises(KeyboardInterrupt):
            encode_file(job, FFMPEG, FFPROBE)

    assert not job.tmp_output.exists()
    assert job.source.exists()


# ---------------------------------------------------------------------------
# estimate_output_size
# ---------------------------------------------------------------------------

def _mock_ffprobe_responses(duration: float, bitrate_kbps: float, width: int, height: int):
    """Return a side_effect callable for subprocess.run covering duration, bitrate, resolution calls."""
    def _run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        result = MagicMock()
        result.returncode = 0
        if "format=duration" in cmd_str:
            result.stdout = f"{duration}\n"
        elif "stream=bit_rate" in cmd_str:
            result.stdout = f"{int(bitrate_kbps * 1000)}\n"
        elif "stream=width,height" in cmd_str:
            result.stdout = f"width={width}\nheight={height}\n"
        else:
            result.stdout = ""
        return result
    return _run


def test_estimate_output_size_h264(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x" * 100)
    source_size = 5 * 1024 ** 3

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(2700, 15000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        result = estimate_output_size(source, FFPROBE, codec="h264", crf=20)

    # h264 factor = 0.50, CRF 20 = baseline scale 1.0, 1080p res factor = 1.0
    assert result > 0
    ratio = result / source_size
    assert 0.40 < ratio < 0.65, f"Expected ~50% ratio, got {ratio:.2f}"


def test_estimate_output_size_vc1_lower_than_h264(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x" * 100)
    source_size = 7 * 1024 ** 3

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(2700, 12000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        vc1_result = estimate_output_size(source, FFPROBE, codec="vc1", crf=20)

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(2700, 12000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        h264_result = estimate_output_size(source, FFPROBE, codec="h264", crf=20)

    # vc1 factor (0.48) < h264 factor (0.50) → vc1 estimate should be smaller
    assert vc1_result < h264_result


def test_estimate_output_size_4k_lower_than_1080p(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x" * 100)
    source_size = 10 * 1024 ** 3

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(7200, 20000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        result_1080 = estimate_output_size(source, FFPROBE, codec="h264", crf=20)

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(7200, 20000, 3840, 2160)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        result_4k = estimate_output_size(source, FFPROBE, codec="h264", crf=20)

    # 4K has 0.85 resolution factor vs 1.0 for 1080p → 4K estimate should be smaller
    assert result_4k < result_1080


def test_estimate_output_size_higher_crf_smaller(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x" * 100)
    source_size = 5 * 1024 ** 3

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(2700, 10000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        result_crf20 = estimate_output_size(source, FFPROBE, codec="h264", crf=20)

    with patch("mediashrink.encoder.subprocess.run", side_effect=_mock_ffprobe_responses(2700, 10000, 1920, 1080)), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = source_size
        result_crf28 = estimate_output_size(source, FFPROBE, codec="h264", crf=28)

    assert result_crf28 < result_crf20


def test_get_video_resolution(tmp_path: Path) -> None:
    source = tmp_path / "ep.mkv"
    source.write_bytes(b"x")

    mock_result = MagicMock()
    mock_result.stdout = "width=1920\nheight=1080\n"

    with patch("mediashrink.encoder.subprocess.run", return_value=mock_result):
        w, h = get_video_resolution(source, FFPROBE)

    assert w == 1920
    assert h == 1080


def test_codec_base_factor_keys_present() -> None:
    assert "h264" in _CODEC_BASE_FACTOR
    assert "vc1" in _CODEC_BASE_FACTOR
    assert "hevc" in _CODEC_BASE_FACTOR
    assert _CODEC_BASE_FACTOR["hevc"] == 1.00
