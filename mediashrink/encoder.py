from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from mediashrink.constants import CRF_BASELINE, CRF_COMPRESSION_FACTOR
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
    try:
        return float(raw) if raw else 0.0
    except ValueError:
        return 0.0


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


# Per-codec compression factor at CRF_BASELINE (20).
# Represents the expected output-to-input size ratio after H.265 encoding.
# Calibrated against real-world encodes (vc1/mpeg2 values from observed runs).
_CODEC_BASE_FACTOR: dict[str, float] = {
    "h264":       0.50,
    "vc1":        0.48,
    "mpeg2video": 0.48,
    "hevc":       1.00,  # already H.265 — no meaningful savings expected
}
_DEFAULT_CODEC_FACTOR = 0.45  # used for unknown/unlisted codecs


def get_video_resolution(path: Path, ffprobe: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream, or (0, 0) on failure."""
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=nw=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        width = height = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("width="):
                width = int(line.split("=", 1)[1])
            elif line.startswith("height="):
                height = int(line.split("=", 1)[1])
        return width, height
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0, 0


def _resolution_factor(width: int, height: int) -> float:
    """Return a size-estimate multiplier based on resolution.

    4K content tends to compress more efficiently (more spatial redundancy);
    SD/720p has less room for improvement.
    """
    if width > 2000:   # 4K / UHD
        return 0.85
    if width >= 1280:  # 1080p / 720p
        return 1.00
    return 1.10        # SD — less redundancy, harder to compress


def estimate_output_size(path: Path, ffprobe: Path, codec: str | None = None, crf: int = 20) -> int:
    """
    Estimate the output file size after H.265 encoding.

    Factors in source codec, resolution, CRF value, and video bitrate.
    Uses the video stream bitrate to estimate the new video size, then adds
    the non-video streams (audio, subtitles) which are copied unchanged.
    Returns 0 if estimation is not possible.
    """
    try:
        input_size = path.stat().st_size
        duration = get_duration_seconds(path, ffprobe)
        if duration <= 0:
            return 0

        # Codec factor — how much H.265 compresses this source at CRF baseline
        codec_factor = _CODEC_BASE_FACTOR.get(codec or "", _DEFAULT_CODEC_FACTOR)

        # CRF scaling — scale relative to the baseline CRF
        baseline_ratio = CRF_COMPRESSION_FACTOR.get(CRF_BASELINE, 0.40)
        crf_ratio = CRF_COMPRESSION_FACTOR.get(crf, baseline_ratio)
        crf_scale = crf_ratio / baseline_ratio

        # Resolution factor
        width, height = get_video_resolution(path, ffprobe)
        res_factor = _resolution_factor(width, height)

        combined_factor = codec_factor * crf_scale * res_factor

        video_kbps = get_video_bitrate_kbps(path, ffprobe)
        if video_kbps <= 0:
            # Fallback: apply factor to whole file
            return int(input_size * combined_factor)

        video_bytes = (video_kbps * 1000 / 8) * duration
        non_video_bytes = max(input_size - video_bytes, 0)

        # Extra bonus for very high-bitrate sources (more headroom to compress)
        if video_kbps > 8000:
            combined_factor *= 0.90

        estimated_video_bytes = video_bytes * combined_factor
        return int(estimated_video_bytes + non_video_bytes)
    except (OSError, ValueError):
        return 0


# Hardware encoder names recognised as --preset values.
# Maps preset alias -> (ffmpeg encoder name, quality flag, extra flags)
# Extra flags include per-encoder quality tuning derived from real-world defaults.
_HW_ENCODERS: dict[str, tuple[str, str, list[str]]] = {
    "qsv":   ("hevc_qsv",   "-global_quality", ["-preset", "medium", "-look_ahead", "1"]),
    "nvenc": ("hevc_nvenc", "-cq",              ["-rc", "vbr", "-preset", "p4", "-tune", "hq", "-bf", "3"]),
    "amf":   ("hevc_amf",  "-qp_i",            ["-qp_p", "0", "-quality", "balanced", "-bf_ref", "1"]),
}


def _hw_quality_args(encoder_key: str, crf: int) -> list[str]:
    """Return the quality flag(s) for a hardware encoder at the given CRF."""
    _, quality_flag, _ = _HW_ENCODERS[encoder_key]
    if encoder_key == "amf":
        return ["-qp_i", str(crf), "-qp_p", str(crf)]
    return [quality_flag, str(crf)]


def probe_encoder_available(encoder_key: str, ffmpeg: Path) -> bool:
    """
    Run a null-encode test with a synthetic 256x256 1-second source to check if
    encoder_key is usable on this machine. Returns True only if FFmpeg exits 0.
    Uses full tuning flags so probe results reflect real encode behaviour.
    encoder_key must be one of "qsv", "nvenc", "amf".
    """
    if encoder_key not in _HW_ENCODERS:
        return False

    encoder_name, _, extra_flags = _HW_ENCODERS[encoder_key]
    quality_args = _hw_quality_args(encoder_key, 28)

    cmd = [
        str(ffmpeg),
        "-f", "lavfi",
        "-i", "color=black:s=256x256:r=1",
        "-t", "1",
        "-c:v", encoder_name,
    ] + quality_args + extra_flags + [
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def validate_encoder(
    encoder_key: str,
    sample_file: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> tuple[bool, str]:
    """
    Encode 3 seconds of real content to /dev/null using full tuning flags.
    Returns (True, '') on success, (False, error_summary) on failure.
    Used to catch encoders that pass the synthetic probe but fail on real input.
    """
    if encoder_key not in _HW_ENCODERS:
        return False, f"Unknown encoder key: {encoder_key}"

    duration = get_duration_seconds(sample_file, ffprobe)
    seek = max(duration * 0.2, 0.0) if duration > 0 else 0.0

    encoder_name, _, extra_flags = _HW_ENCODERS[encoder_key]
    quality_args = _hw_quality_args(encoder_key, 28)

    cmd = [
        str(ffmpeg),
        "-ss", str(seek),
        "-i", str(sample_file),
        "-t", "3",
        "-c:v", encoder_name,
    ] + quality_args + extra_flags + [
        "-an", "-f", "null", "-",
        "-loglevel", "error",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, ""
        stderr = result.stderr.strip()
        summary = stderr.splitlines()[-1] if stderr else f"exit code {result.returncode}"
        return False, summary
    except subprocess.TimeoutExpired:
        return False, "validation timed out"
    except OSError as exc:
        return False, str(exc)


def is_hardware_preset(preset: str) -> bool:
    return preset in _HW_ENCODERS


def build_ffmpeg_command(
    job: EncodeJob,
    ffmpeg: Path,
    duration_limit_seconds: float | None = None,
) -> list[str]:
    """Return the canonical FFmpeg encode command for a job.

    If duration_limit_seconds is set, a `-t <seconds>` flag is inserted before
    the output path so only that many seconds of the source are encoded.
    """
    if job.preset in _HW_ENCODERS:
        return _build_hw_command(job, ffmpeg, duration_limit_seconds)
    return _build_sw_command(job, ffmpeg, duration_limit_seconds)


def _duration_flags(duration_limit_seconds: float | None) -> list[str]:
    if duration_limit_seconds is not None and duration_limit_seconds > 0:
        return ["-t", str(duration_limit_seconds)]
    return []


def _build_sw_command(
    job: EncodeJob,
    ffmpeg: Path,
    duration_limit_seconds: float | None = None,
) -> list[str]:
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
    ] + _duration_flags(duration_limit_seconds) + [str(job.tmp_output)]


def _build_hw_command(
    job: EncodeJob,
    ffmpeg: Path,
    duration_limit_seconds: float | None = None,
) -> list[str]:
    """Hardware-accelerated encode (QSV / NVENC / AMF)."""
    encoder, _, extra = _HW_ENCODERS[job.preset]
    quality_args = _hw_quality_args(job.preset, job.crf)

    cmd = [
        str(ffmpeg),
        "-i", str(job.source),
        "-map", "0",
        "-c:v", encoder,
    ]
    cmd += quality_args
    cmd += extra
    cmd += [
        "-c:a", "copy",
        "-c:s", "copy",
        "-tag:v", "hvc1",
        "-movflags", "+faststart",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-stats_period", "2",
    ]
    cmd += _duration_flags(duration_limit_seconds)
    cmd += [str(job.tmp_output)]
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
    duration_limit_seconds: float | None = None,
    log_path: Path | None = None,
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
    # For progress tracking, use the shorter of total duration and limit
    media_duration = (
        min(total_duration, duration_limit_seconds)
        if duration_limit_seconds is not None and total_duration > 0
        else total_duration
    )
    cmd = build_ffmpeg_command(job, ffmpeg, duration_limit_seconds)

    # Ensure output directory exists
    job.tmp_output.parent.mkdir(parents=True, exist_ok=True)

    stderr_target = subprocess.PIPE if log_path is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=stderr_target,
        text=True,
    )

    stderr_thread: threading.Thread | None = None
    if log_path is not None and process.stderr is not None:
        def _drain_stderr(src: object, dst: Path) -> None:
            import io
            assert hasattr(src, "read")
            with dst.open("a", encoding="utf-8", errors="replace") as fh:
                for line in src:  # type: ignore[union-attr]
                    fh.write(line)

        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(process.stderr, log_path), daemon=True
        )
        stderr_thread.start()

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
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        if stderr_thread is not None:
            stderr_thread.join(timeout=2)
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


def encode_preview(
    source: Path,
    ffmpeg: Path,
    ffprobe: Path,
    duration_minutes: float = 2.0,
    crf: int = 20,
    preset: str = "fast",
) -> EncodeResult:
    """Encode the first `duration_minutes` of a file to a `_preview` output.

    Output path: `<stem>_preview<suffix>` alongside the source.
    The source is never modified. The preview is NOT a session entry.
    """
    output = source.parent / f"{source.stem}_preview{source.suffix}"
    tmp_output = source.parent / f".tmp_{source.stem}_preview{source.suffix}"
    job = EncodeJob(
        source=source,
        output=output,
        tmp_output=tmp_output,
        crf=crf,
        preset=preset,
        dry_run=False,
        skip=False,
    )
    limit = duration_minutes * 60.0
    return encode_file(job, ffmpeg, ffprobe, duration_limit_seconds=limit)
