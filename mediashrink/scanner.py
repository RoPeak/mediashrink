from __future__ import annotations

import re
import subprocess
from pathlib import Path

from mediashrink.models import EncodeJob

SUPPORTED_EXTENSIONS = (".mkv", ".mp4", ".m4v")
_DUPLICATE_POLICY_CHOICES = {"prefer-mkv", "all", "skip-title"}


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


def duplicate_policy_choices() -> tuple[str, ...]:
    return tuple(sorted(_DUPLICATE_POLICY_CHOICES))


def _normalize_title(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_compressed$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[\[\{].*?[\]\}]", " ", stem)
    stem = re.sub(
        r"\(.*?(extended|director|criterion|remaster|special|unrated|theatrical).*?\)",
        " ",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"\b(19|20)\d{2}\b", "", stem)
    stem = re.sub(
        r"\b(bluray|blu ray|brrip|webrip|web rip|dvdrip|x264|x265|hevc|h264|hdr|uhd|remux|proper|repack|10bit|2160p|1080p|720p)\b",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"[-–—]", " ", stem)
    stem = re.sub(r"\s+", " ", stem.replace(".", " ").replace("_", " ")).strip().lower()
    return stem


def apply_duplicate_title_policy(
    files: list[Path], policy: str = "prefer-mkv"
) -> tuple[list[Path], list[str], dict[str, list[Path]]]:
    if policy not in _DUPLICATE_POLICY_CHOICES:
        raise ValueError(f"unsupported duplicate policy {policy!r}")
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(_normalize_title(path), []).append(path)

    filtered: list[Path] = []
    warnings: list[str] = []
    deprioritized: dict[str, list[Path]] = {}
    for key, group in groups.items():
        if len(group) == 1:
            filtered.extend(group)
            continue
        ordered_group = sorted(group, key=_natural_sort_key)
        if policy == "all":
            filtered.extend(ordered_group)
            warnings.append(
                f"Possible duplicate title kept across formats: {', '.join(path.name for path in ordered_group)}"
            )
            continue
        if policy == "skip-title":
            warnings.append(
                f"Skipped duplicate title group: {', '.join(path.name for path in ordered_group)}"
            )
            deprioritized[key] = ordered_group
            continue

        preferred = next(
            (path for path in ordered_group if path.suffix.lower() == ".mkv"),
            ordered_group[0],
        )
        filtered.append(preferred)
        skipped = [path for path in ordered_group if path != preferred]
        if skipped:
            deprioritized[key] = skipped
            warnings.append(
                f"Preferred {preferred.name} over duplicate format(s): {', '.join(path.name for path in skipped)}"
            )
    return sorted(filtered, key=_natural_sort_key), warnings, deprioritized


def supported_formats_label() -> str:
    """Return a user-facing list of supported input formats."""
    return ", ".join(SUPPORTED_EXTENSIONS)


def probe_video_codec(path: Path, ffprobe: Path) -> str | None:
    """Return the codec name of the first video stream, or None on failure."""
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nw=1:nk=1",
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
