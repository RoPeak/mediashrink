from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from mkv_compress.models import EncodeJob, EncodeResult


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


def build_ffmpeg_command(job: EncodeJob, ffmpeg: Path) -> list[str]:
    """Return the canonical FFmpeg encode command for a job."""
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
    )
