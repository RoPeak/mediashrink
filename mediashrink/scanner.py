from __future__ import annotations

import re
import subprocess
from pathlib import Path

from mediashrink.models import EncodeJob

SUPPORTED_EXTENSIONS = (".mkv", ".mp4", ".m4v")


def _natural_sort_key(path: Path) -> list[int | str]:
    """Sort key that orders '2' before '10' (natural/human sort)."""
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.name):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


def scan_directory(directory: Path, recursive: bool = False) -> list[Path]:
    """Return naturally sorted list of supported video files in directory."""
    patterns = [f"**/*{ext}" if recursive else f"*{ext}" for ext in SUPPORTED_EXTENSIONS]
    files = [
        path
        for pattern in patterns
        for path in directory.glob(pattern)
        if path.is_file() and not path.name.startswith(".tmp_")
    ]
    return sorted(files, key=_natural_sort_key)


def supported_formats_label() -> str:
    """Return a user-facing list of supported input formats."""
    return ", ".join(SUPPORTED_EXTENSIONS)


def probe_video_codec(path: Path, ffprobe: Path) -> str | None:
    """Return the codec name of the first video stream, or None on failure."""
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        codec = result.stdout.strip()
        return codec if codec else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def is_already_compressed(
    path: Path,
    ffprobe: Path,
    no_skip: bool = False,
    codec: str | None = None,
) -> tuple[bool, str]:
    """Return (should_skip, reason). Always returns (False, '') when no_skip=True."""
    if no_skip:
        return False, ""

    if "_compressed" in path.stem.lower():
        return True, "filename contains '_compressed'"

    if codec is None:
        codec = probe_video_codec(path, ffprobe)
    if codec in {"hevc", "h265"}:
        return True, "video stream is already H.265/HEVC"

    return False, ""


def build_jobs(
    files: list[Path],
    output_dir: Path | None,
    overwrite: bool,
    crf: int,
    preset: str,
    dry_run: bool,
    ffprobe: Path,
    no_skip: bool = False,
) -> list[EncodeJob]:
    """Construct an EncodeJob for each file, skipping already-compressed ones."""
    from mediashrink.encoder import estimate_output_size

    jobs: list[EncodeJob] = []

    for source in files:
        codec = probe_video_codec(source, ffprobe)
        skip, skip_reason = is_already_compressed(source, ffprobe, no_skip, codec=codec)

        if overwrite:
            output = source
        elif output_dir is not None:
            output = output_dir / source.name
        else:
            output = source.with_stem(source.stem + "_compressed")

        tmp_output = output.parent / f".tmp_{output.stem}{output.suffix}"

        estimated = estimate_output_size(source, ffprobe, codec=codec, crf=crf) if not skip else 0

        jobs.append(
            EncodeJob(
                source=source,
                output=output,
                tmp_output=tmp_output,
                crf=crf,
                preset=preset,
                dry_run=dry_run,
                skip=skip,
                skip_reason=skip_reason if skip else None,
                source_codec=codec,
                estimated_output_bytes=estimated,
            )
        )

    return jobs
