from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from mediashrink.models import EncodeJob, EncodeResult


def get_duration_seconds(path: Path, ffprobe: Path) -> float:
    """Return the total duration of the media file in seconds."""
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    raw = result.stdout.strip()
    return float(raw) if raw else 0.0


def get_video_bitrate_kbps(path: Path, ffprobe: Path) -> float:
    """Return the video stream bitrate in kbps, or 0.0 on failure."""
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        raw = result.stdout.strip()
        if raw and raw != "N/A":
            return float(raw) / 1000
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return 0.0


# Approximate bitrate reduction factor for H.265 vs common source codecs at CRF 20.
# H.265 typically achieves 40-60% of the source bitrate for equivalent quality.
_HEVC_COMPRESSION_FACTOR = 0.40


def estimate_output_size(path: Path, ffprobe: Path) -> int:
    """
    Estimate the output file size after H.265 encoding.

    Uses the video stream bitrate to estimate the new video size, then adds
    the non-video streams (audio, subtitles) which are copied unchanged.
    Returns 0 if estimation is not possible.
    """
    try:
        # Get total file size and duration
        input_size = path.stat().st_size
        duration = get_duration_seconds(path, ffprobe)
        if duration <= 0:
            return 0

        # Estimate non-video bytes from total bitrate vs video bitrate
        video_kbps = get_video_bitrate_kbps(path, ffprobe)
        if video_kbps <= 0:
            # Fall back: assume video is ~85% of file, apply factor to whole file
            return int(input_size * _HEVC_COMPRESSION_FACTOR)

        video_bytes = (video_kbps * 1000 / 8) * duration
        non_video_bytes = max(input_size - video_bytes, 0)

        estimated_video_bytes = video_bytes * _HEVC_COMPRESSION_FACTOR
        return int(estimated_video_bytes + non_video_bytes)
    except (OSError, ValueError):
        return 0


# Hardware encoder names recognised as --preset values.
# Maps preset alias -> (ffmpeg encoder name, quality flag, extra flags)
_HW_ENCODERS: dict[str, tuple[str, str, list[str]]] = {
    "qsv":   ("hevc_qsv",  "-global_quality", []),
    "nvenc": ("hevc_nvenc", "-cq",             ["-rc", "vbr"]),
    "amf":   ("hevc_amf",  "-qp_i",           ["-qp_p", str(0)]),  # placeholder; overridden below
}


def probe_encoder_available(encoder_key: str, ffmpeg: Path) -> bool:
    """
    Run a null-encode test with a synthetic 64x64 1-second source to check if
    encoder_key is usable on this machine. Returns True only if FFmpeg exits 0.
    encoder_key must be one of "qsv", "nvenc", "amf".
    """
    if encoder_key not in _HW_ENCODERS:
        return False

    encoder_name, quality_flag, extra_flags = _HW_ENCODERS[encoder_key]

    if encoder_key == "amf":
        quality_args = ["-qp_i", "28", "-qp_p", "28"]
    else:
        quality_args = [quality_flag, "28"]

    cmd = [
        str(ffmpeg),
        "-f", "lavfi",
        "-i", "color=black:s=64x64:r=1",
        "-t", "1",
        "-c:v", encoder_name,
    ] + quality_args + extra_flags + [
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def is_hardware_preset(preset: str) -> bool:
    return preset in _HW_ENCODERS


def build_ffmpeg_command(job: EncodeJob, ffmpeg: Path) -> list[str]:
    """Return the canonical FFmpeg encode command for a job."""
    if job.preset in _HW_ENCODERS:
        return _build_hw_command(job, ffmpeg)
    return _build_sw_command(job, ffmpeg)


def _build_sw_command(job: EncodeJob, ffmpeg: Path) -> list[str]:
    """Software encode via libx265."""
    return [
        str(ffmpeg),
        "-i", str(job.source),
        "-map", "0",
        "-c:v", "libx265",
        "-crf", str(job.crf),
        "-preset", job.preset,
        "-c:a", "copy",
        "-c:s", "copy",
        "-tag:v", "hvc1",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-stats_period", "2",
        str(job.tmp_output),
    ]


def _build_hw_command(job: EncodeJob, ffmpeg: Path) -> list[str]:
    """Hardware-accelerated encode (QSV / NVENC / AMF)."""
    encoder, quality_flag, extra = _HW_ENCODERS[job.preset]

    # QSV: global_quality maps roughly to CRF (same scale, lower = better)
    # NVENC: -cq is the CRF equivalent
    # AMF: use -qp_i / -qp_p
    if job.preset == "amf":
        quality_args = ["-qp_i", str(job.crf), "-qp_p", str(job.crf)]
    else:
        quality_args = [quality_flag, str(job.crf)]

    cmd = [
        str(ffmpeg),
        "-i", str(job.source),
        "-map", "0",
        "-c:v", encoder,
    ]
    cmd += quality_args
    cmd += [
        "-c:a", "copy",
        "-c:s", "copy",
        "-tag:v", "hvc1",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-stats_period", "2",
        str(job.tmp_output),
    ]
    return cmd


def parse_progress_line(line: str) -> dict[str, str]:
    """Parse a single 'key=value' line from ffmpeg -progress output."""
    line = line.strip()
    if "=" in line:
        key, _, value = line.partition("=")
        return {key.strip(): value.strip()}
    return {}


def encode_file(
    job: EncodeJob,
    ffmpeg: Path,
    ffprobe: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> EncodeResult:
    """
    Encode a single file.

    Lifecycle:
    - Writes to job.tmp_output
    - On success: renames tmp to job.output (replaces source if overwrite)
    - On failure or interrupt: deletes tmp, leaves source untouched
    """
    input_size = job.source.stat().st_size
    start_time = time.monotonic()

    if job.skip:
        return EncodeResult(
            job=job,
            skipped=True,
            skip_reason=job.skip_reason,
            success=False,
            input_size_bytes=input_size,
            output_size_bytes=0,
            duration_seconds=0.0,
        )

    if job.dry_run:
        return EncodeResult(
            job=job,
            skipped=False,
            skip_reason=None,
            success=True,
            input_size_bytes=input_size,
            output_size_bytes=0,
            duration_seconds=0.0,
        )

    total_duration = get_duration_seconds(job.source, ffprobe)
    cmd = build_ffmpeg_command(job, ffmpeg)
    media_duration = total_duration

    # Ensure output directory exists
    job.tmp_output.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            parsed = parse_progress_line(raw_line)
            if progress_callback and "out_time_ms" in parsed:
                try:
                    # FFmpeg names the field out_time_ms but unit is microseconds
                    out_us = float(parsed["out_time_ms"])
                    if total_duration > 0:
                        pct = min((out_us / 1_000_000) / total_duration * 100, 100.0)
                        progress_callback(pct)
                except ValueError:
                    pass

        process.wait()
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        if job.tmp_output.exists():
            job.tmp_output.unlink()
        raise

    duration = time.monotonic() - start_time

    if process.returncode != 0:
        if job.tmp_output.exists():
            job.tmp_output.unlink()
        return EncodeResult(
            job=job,
            skipped=False,
            skip_reason=None,
            success=False,
            input_size_bytes=input_size,
            output_size_bytes=0,
            duration_seconds=duration,
            error_message=f"FFmpeg exited with code {process.returncode}",
        )

    # Success: rename tmp → final output
    job.tmp_output.rename(job.output)

    # If overwrite mode: source == output, already replaced above
    output_size = job.output.stat().st_size

    return EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        duration_seconds=duration,
        media_duration_seconds=media_duration,
    )
