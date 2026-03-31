from __future__ import annotations

import subprocess
from pathlib import Path

from mkv_compress.models import EncodeJob


def scan_directory(directory: Path, recursive: bool = False) -> list[Path]:
    """Return sorted list of .mkv files in directory."""
    pattern = "**/*.mkv" if recursive else "*.mkv"
    return sorted(directory.glob(pattern))


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
) -> tuple[bool, str]:
    """Return (should_skip, reason). Always returns (False, '') when no_skip=True."""
    if no_skip:
        return False, ""

    if "_compressed" in path.stem:
        return True, "filename contains '_compressed'"

    codec = probe_video_codec(path, ffprobe)
    if codec == "hevc":
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
    jobs: list[EncodeJob] = []

    for source in files:
        skip, skip_reason = is_already_compressed(source, ffprobe, no_skip)

        if overwrite:
            output = source
        elif output_dir is not None:
            output = output_dir / source.name
        else:
            output = source.with_stem(source.stem + "_compressed")

        tmp_output = output.parent / f".tmp_{output.stem}.mkv"

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
            )
        )

    return jobs
