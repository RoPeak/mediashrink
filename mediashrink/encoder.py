from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from mediashrink.calibration import (
    bitrate_bucket,
    load_calibration_store,
    lookup_estimate,
    resolution_bucket,
)
from mediashrink.constants import CRF_BASELINE, CRF_COMPRESSION_FACTOR
from mediashrink.models import EncodeAttempt, EncodeJob, EncodeResult


def get_duration_seconds(path: Path, ffprobe: Path) -> float:
    """Return the total duration of the media file in seconds."""
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
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
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=nw=1:nk=1",
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
# Calibrated against real-world batch encodes:
#   mpeg2video: 42-file batch achieved 73.3% savings → ~0.27 output ratio (updated from 0.48)
#   vc1: similar compressibility to mpeg2video (updated from 0.48)
#   h264: typical H.264→H.265 yields 40-50% savings → 0.50-0.60 output ratio
_CODEC_BASE_FACTOR: dict[str, float] = {
    "h264": 0.50,
    "vc1": 0.30,
    "mpeg2video": 0.28,
    "hevc": 1.00,  # already H.265 — no meaningful savings expected
}
_DEFAULT_CODEC_FACTOR = 0.45  # used for unknown/unlisted codecs


def get_video_resolution(path: Path, ffprobe: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream, or (0, 0) on failure."""
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=nw=1",
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
    if width > 2000:  # 4K / UHD
        return 0.85
    if width >= 1280:  # 1080p / 720p
        return 1.00
    return 1.10  # SD — less redundancy, harder to compress


def estimate_output_size(
    path: Path,
    ffprobe: Path,
    codec: str | None = None,
    crf: int = 20,
    *,
    preset: str = "fast",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> int:
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

        # Hardware encoders (AMF/NVENC/QSV) produce larger files than software libx265 at the
        # same CRF value because their rate control is less efficient. Apply a penalty factor so
        # the heuristic estimate reflects real-world hardware output ratios.
        # Observed: AMF CRF 22 on mpeg2video → ~0.50 output ratio vs ~0.28 for software.
        # Factor: 1.65 (conservative midpoint of observed 1.78; refined over time by calibration).
        if preset in _HW_ENCODERS:
            combined_factor *= 1.65

        video_kbps = get_video_bitrate_kbps(path, ffprobe)
        heuristic_estimate = 0
        if video_kbps <= 0:
            heuristic_estimate = int(input_size * combined_factor)
        else:
            video_bytes = (video_kbps * 1000 / 8) * duration
            non_video_bytes = max(input_size - video_bytes, 0)

            # Extra bonus for very high-bitrate sources (more headroom to compress)
            if video_kbps > 8000:
                combined_factor *= 0.90

            estimated_video_bytes = video_bytes * combined_factor
            heuristic_estimate = int(estimated_video_bytes + non_video_bytes)

        if use_calibration:
            active_store = (
                calibration_store if calibration_store is not None else load_calibration_store()
            )
            lookup = lookup_estimate(
                active_store,
                codec=codec,
                resolution=resolution_bucket(width, height),
                bitrate=bitrate_bucket(video_kbps),
                preset=preset,
                container=path.suffix.lower() or ".mkv",
            )
            if lookup is not None and lookup.output_ratio is not None and lookup.output_ratio > 0:
                corrected_ratio = lookup.output_ratio
                if lookup.average_size_error is not None:
                    corrected_ratio = max(0.05, corrected_ratio + lookup.average_size_error)
                calibrated_estimate = int(input_size * corrected_ratio)
                if lookup.confidence == "High":
                    return calibrated_estimate
                if lookup.confidence == "Medium":
                    return int((calibrated_estimate * 0.65) + (heuristic_estimate * 0.35))
                return int((calibrated_estimate * 0.40) + (heuristic_estimate * 0.60))

        return heuristic_estimate
    except (OSError, ValueError):
        return 0


# Hardware encoder names recognised as --preset values.
# Maps preset alias -> (ffmpeg encoder name, quality flag, extra flags)
# Extra flags include per-encoder quality tuning derived from real-world defaults.
_HW_ENCODERS: dict[str, tuple[str, str, list[str]]] = {
    "qsv": ("hevc_qsv", "-global_quality", ["-preset", "medium", "-look_ahead", "1"]),
    "nvenc": ("hevc_nvenc", "-cq", ["-rc", "vbr", "-preset", "p4", "-tune", "hq", "-bf", "3"]),
    "amf": ("hevc_amf", "-qp_i", ["-quality", "balanced"]),
}


def _hw_quality_args(encoder_key: str, crf: int) -> list[str]:
    """Return the quality flag(s) for a hardware encoder at the given CRF."""
    _, quality_flag, _ = _HW_ENCODERS[encoder_key]
    if encoder_key == "amf":
        return ["-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]
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

    cmd = (
        [
            str(ffmpeg),
            "-f",
            "lavfi",
            "-i",
            "color=black:s=256x256:r=1",
            "-t",
            "1",
            "-c:v",
            encoder_name,
        ]
        + quality_args
        + extra_flags
        + [
            "-f",
            "null",
            "-",
        ]
    )

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

    cmd = (
        [
            str(ffmpeg),
            "-ss",
            str(seek),
            "-i",
            str(sample_file),
            "-t",
            "3",
            "-c:v",
            encoder_name,
        ]
        + quality_args
        + extra_flags
        + [
            "-an",
            "-f",
            "null",
            "-",
            "-loglevel",
            "error",
        ]
    )

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


# MP4/M4V containers do not support most subtitle formats (ASS, SSA, PGS, DVB, etc.).
# Attempting to copy incompatible subtitle streams causes FFmpeg to fail at header-write
# time with "Invalid argument" before a single frame is encoded.
_MP4_CONTAINERS = {".mp4", ".m4v"}
_SAFE_MP4_AUDIO_CODECS = {"aac", "ac3", "eac3", "mp3", "alac"}


def _subtitle_args(output_path: Path) -> list[str]:
    """Return subtitle stream arguments appropriate for the output container.

    MP4/M4V containers only support mov_text; arbitrary subtitle copy fails.
    Drop all subtitle streams for MP4 outputs to avoid header-write errors.
    MKV accepts any subtitle codec — copy unchanged.
    """
    if output_path.suffix.lower() in _MP4_CONTAINERS:
        return ["-sn"]
    return ["-c:s", "copy"]


def _stream_map_args(output_path: Path) -> list[str]:
    """Return stream-selection args for the output container.

    MP4/M4V outputs are more fragile when copying arbitrary auxiliary/data streams.
    Restrict them to video/audio only and drop data streams explicitly.
    """
    if output_path.suffix.lower() in _MP4_CONTAINERS:
        return ["-map", "0:v", "-map", "0:a?", "-dn"]
    return ["-map", "0"]


def output_drops_subtitles(output_path: Path) -> bool:
    return output_path.suffix.lower() in _MP4_CONTAINERS


def _probe_streams(path: Path, ffprobe: Path) -> list[dict[str, str]]:
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return []

    streams: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        raw = [part.strip() for part in line.split(",") if part.strip()]
        if not raw:
            continue
        if len(raw) == 1:
            streams.append({"codec_type": raw[0], "codec_name": ""})
        else:
            streams.append({"codec_type": raw[0], "codec_name": raw[1]})
    return streams


def source_has_subtitle_streams(path: Path, ffprobe: Path) -> bool:
    for stream in _probe_streams(path, ffprobe):
        if stream.get("codec_type") == "subtitle":
            return True
    return False


def source_has_attachment_streams(path: Path, ffprobe: Path) -> bool:
    for stream in _probe_streams(path, ffprobe):
        if stream.get("codec_type") == "attachment":
            return True
    return False


def source_has_data_streams(path: Path, ffprobe: Path) -> bool:
    for stream in _probe_streams(path, ffprobe):
        if stream.get("codec_type") == "data":
            return True
    return False


def source_audio_codecs(path: Path, ffprobe: Path) -> set[str]:
    codecs: set[str] = set()
    for stream in _probe_streams(path, ffprobe):
        if stream.get("codec_type") == "audio" and stream.get("codec_name"):
            codecs.add(stream["codec_name"])
    return codecs


def output_may_require_audio_reencode(path: Path, output_path: Path, ffprobe: Path) -> set[str]:
    if output_path.suffix.lower() not in _MP4_CONTAINERS:
        return set()
    codecs = source_audio_codecs(path, ffprobe)
    return {codec for codec in codecs if codec not in _SAFE_MP4_AUDIO_CODECS}


def describe_output_container_constraints(
    source: Path,
    output_path: Path,
    ffprobe: Path,
) -> list[str]:
    if output_path.suffix.lower() not in _MP4_CONTAINERS:
        return []

    notes: list[str] = []
    if source_has_subtitle_streams(source, ffprobe):
        notes.append("subtitle streams will be dropped for MP4/M4V compatibility")
    if source_has_attachment_streams(source, ffprobe):
        notes.append("attachment streams will be dropped for MP4/M4V compatibility")
    if source_has_data_streams(source, ffprobe):
        notes.append("auxiliary data streams will be dropped for MP4/M4V compatibility")
    unsupported_audio = output_may_require_audio_reencode(source, output_path, ffprobe)
    if unsupported_audio:
        notes.append(
            "audio copy may fail in this container for codec(s): "
            + ", ".join(sorted(unsupported_audio))
        )
    return notes


def describe_container_incompatibility(
    source: Path,
    output_path: Path,
    ffprobe: Path,
) -> str | None:
    if output_path.suffix.lower() not in _MP4_CONTAINERS:
        return None
    unsupported_audio = output_may_require_audio_reencode(source, output_path, ffprobe)
    if unsupported_audio:
        return (
            "audio codec copy is not supported by the chosen output container ("
            + ", ".join(sorted(unsupported_audio))
            + ")"
        )
    if source_has_attachment_streams(source, ffprobe):
        return "attachment streams are not supported by the chosen output container"
    if source_has_data_streams(source, ffprobe):
        return "auxiliary data streams are not supported by the chosen output container"
    # Subtitle streams are safely dropped via -sn for MP4/M4V outputs — not an encode failure.
    return None


def _build_sw_command(
    job: EncodeJob,
    ffmpeg: Path,
    duration_limit_seconds: float | None = None,
) -> list[str]:
    """Software encode via libx265."""
    return (
        [
            str(ffmpeg),
            "-i",
            str(job.source),
        ]
        + _stream_map_args(job.tmp_output)
        + [
            "-c:v",
            "libx265",
            "-crf",
            str(job.crf),
            "-preset",
            job.preset,
            "-c:a",
            "copy",
        ]
        + _subtitle_args(job.tmp_output)
        + [
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-stats_period",
            "2",
        ]
        + _duration_flags(duration_limit_seconds)
        + [str(job.tmp_output)]
    )


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
        "-i",
        str(job.source),
    ]
    cmd += _stream_map_args(job.tmp_output)
    cmd += [
        "-c:v",
        encoder,
    ]
    cmd += quality_args
    cmd += extra
    cmd += [
        "-c:a",
        "copy",
    ]
    cmd += _subtitle_args(job.tmp_output)
    cmd += [
        "-tag:v",
        "hev1",  # hvc1 requires inline params (software only); hev1 = container header (hardware)
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-stats_period",
        "2",
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


def _summarize_stderr_lines(stderr_lines: list[str], returncode: int) -> str:
    """Return a concise ffmpeg failure summary from captured stderr."""
    cleaned = [line.strip() for line in stderr_lines if line.strip()]
    if not cleaned:
        return f"FFmpeg exited with code {returncode}"
    return "\n".join(cleaned[-5:])


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
    last_progress_pct = 0.0

    if job.skip:
        return EncodeResult(
            job=job,
            skipped=True,
            skip_reason=job.skip_reason,
            success=False,
            input_size_bytes=input_size,
            output_size_bytes=0,
            duration_seconds=0.0,
            attempts=[],
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
            attempts=[],
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

    stderr_tail: deque[str] = deque(maxlen=20)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_thread: threading.Thread | None = None
    if process.stderr is not None:

        def _drain_stderr(src: object, dst: Path | None, tail: deque[str]) -> None:
            if dst is None:
                for line in src:  # type: ignore[union-attr, attr-defined]
                    tail.append(line)
                return

            with dst.open("a", encoding="utf-8", errors="replace") as fh:
                for line in src:  # type: ignore[union-attr, attr-defined]
                    tail.append(line)
                    fh.write(line)

        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(process.stderr, log_path, stderr_tail), daemon=True
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
                        pct = min((out_us / 1_000_000) / media_duration * 100, 100.0)
                        last_progress_pct = pct
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
            error_message=_summarize_stderr_lines(list(stderr_tail), process.returncode),
            raw_error_message=_summarize_stderr_lines(list(stderr_tail), process.returncode),
            attempts=[
                EncodeAttempt(
                    preset=job.preset,
                    crf=job.crf,
                    success=False,
                    duration_seconds=duration,
                    progress_pct=last_progress_pct,
                    error_message=_summarize_stderr_lines(list(stderr_tail), process.returncode),
                )
            ],
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
        attempts=[
            EncodeAttempt(
                preset=job.preset,
                crf=job.crf,
                success=True,
                duration_seconds=duration,
                progress_pct=max(last_progress_pct, 100.0 if media_duration > 0 else 0.0),
            )
        ],
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
    Stderr is captured to a temp log file so failures include a useful error message.
    """
    import tempfile

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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
        log_path = Path(tf.name)

    try:
        result = encode_file(job, ffmpeg, ffprobe, duration_limit_seconds=limit, log_path=log_path)
        if not result.success and log_path.exists():
            stderr_text = log_path.read_text(encoding="utf-8", errors="replace").strip()
            if stderr_text:
                last_lines = "\n".join(stderr_text.splitlines()[-5:])
                result = EncodeResult(
                    job=result.job,
                    skipped=result.skipped,
                    skip_reason=result.skip_reason,
                    success=False,
                    input_size_bytes=result.input_size_bytes,
                    output_size_bytes=result.output_size_bytes,
                    duration_seconds=result.duration_seconds,
                    error_message=last_lines,
                )
        return result
    finally:
        if log_path.exists():
            log_path.unlink()


def preflight_encode_job(
    source: Path,
    ffmpeg: Path,
    ffprobe: Path,
    *,
    crf: int,
    preset: str,
    duration_seconds: float = 3.0,
) -> EncodeResult:
    """Run a short real-output encode to validate the final batch settings."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="mediashrink-preflight-") as temp_dir:
        temp_root = Path(temp_dir)
        output = temp_root / f"{source.stem}_preflight{source.suffix}"
        tmp_output = temp_root / f".tmp_{source.stem}_preflight{source.suffix}"
        job = EncodeJob(
            source=source,
            output=output,
            tmp_output=tmp_output,
            crf=crf,
            preset=preset,
            dry_run=False,
            skip=False,
        )
        result = encode_file(
            job,
            ffmpeg,
            ffprobe,
            duration_limit_seconds=duration_seconds,
        )
        if result.success and output.exists():
            output.unlink()
        return result
