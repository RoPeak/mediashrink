from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mediashrink.scanner import (
    build_jobs,
    is_already_compressed,
    probe_video_codec,
    scan_directory,
    supported_formats_label,
)

FFPROBE = Path("/usr/bin/ffprobe")


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------

def test_scan_directory_finds_supported_video_files(tmp_path: Path) -> None:
    (tmp_path / "ep01.mkv").touch()
    (tmp_path / "ep02.mp4").touch()
    (tmp_path / "ep03.m4v").touch()
    (tmp_path / "cover.jpg").touch()

    result = scan_directory(tmp_path)

    assert len(result) == 3
    assert {p.suffix for p in result} == {".mkv", ".mp4", ".m4v"}


def test_scan_directory_returns_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.mkv").touch()
    (tmp_path / "a.mkv").touch()

    result = scan_directory(tmp_path)

    assert result == sorted(result)


def test_scan_directory_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "ep01.mkv").touch()
    (sub / "ep02.mp4").touch()

    non_recursive = scan_directory(tmp_path, recursive=False)
    recursive = scan_directory(tmp_path, recursive=True)

    assert len(non_recursive) == 1
    assert len(recursive) == 2


def test_scan_directory_empty(tmp_path: Path) -> None:
    assert scan_directory(tmp_path) == []


def test_supported_formats_label_lists_extensions() -> None:
    assert supported_formats_label() == ".mkv, .mp4, .m4v"


# ---------------------------------------------------------------------------
# is_already_compressed
# ---------------------------------------------------------------------------

def test_is_already_compressed_by_filename(tmp_path: Path) -> None:
    path = tmp_path / "Heroes_S01E01_COMPRESSED.mp4"
    path.touch()

    skip, reason = is_already_compressed(path, FFPROBE)

    assert skip is True
    assert "compressed" in reason


def test_is_already_compressed_by_codec(tmp_path: Path) -> None:
    path = tmp_path / "Heroes_S01E01.mkv"
    path.touch()

    with patch("mediashrink.scanner.probe_video_codec", return_value="hevc"):
        skip, reason = is_already_compressed(path, FFPROBE)

    assert skip is True
    assert "HEVC" in reason


def test_is_already_compressed_false_for_h264(tmp_path: Path) -> None:
    path = tmp_path / "Heroes_S01E01.mkv"
    path.touch()

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        skip, reason = is_already_compressed(path, FFPROBE)

    assert skip is False
    assert reason == ""


def test_is_already_compressed_no_skip_overrides(tmp_path: Path) -> None:
    path = tmp_path / "Heroes_S01E01_compressed.mkv"
    path.touch()

    skip, reason = is_already_compressed(path, FFPROBE, no_skip=True)

    assert skip is False


# ---------------------------------------------------------------------------
# build_jobs
# ---------------------------------------------------------------------------

def _make_files(tmp_path: Path, names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(p)
    return paths


def test_build_jobs_default_output(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["ep01.mkv"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        jobs = build_jobs(files, output_dir=None, overwrite=False,
                          crf=20, preset="slow", dry_run=False, ffprobe=FFPROBE)

    assert len(jobs) == 1
    assert jobs[0].output.name == "ep01_compressed.mkv"
    assert jobs[0].tmp_output.name == ".tmp_ep01_compressed.mkv"
    assert jobs[0].skip is False


def test_build_jobs_output_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    files = _make_files(tmp_path, ["ep01.mp4"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        jobs = build_jobs(files, output_dir=out_dir, overwrite=False,
                          crf=20, preset="slow", dry_run=False, ffprobe=FFPROBE)

    assert jobs[0].output.parent == out_dir
    assert jobs[0].output.name == "ep01.mp4"
    assert jobs[0].tmp_output.name == ".tmp_ep01.mp4"


def test_build_jobs_overwrite(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["ep01.mkv"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        jobs = build_jobs(files, output_dir=None, overwrite=True,
                          crf=20, preset="slow", dry_run=False, ffprobe=FFPROBE)

    assert jobs[0].output == jobs[0].source


def test_build_jobs_default_output_preserves_container(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["ep01.m4v"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        jobs = build_jobs(files, output_dir=None, overwrite=False,
                          crf=20, preset="slow", dry_run=False, ffprobe=FFPROBE)

    assert jobs[0].output.name == "ep01_compressed.m4v"
    assert jobs[0].tmp_output.name == ".tmp_ep01_compressed.m4v"


def test_build_jobs_skip_hevc(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["ep01.mkv"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="hevc"):
        jobs = build_jobs(files, output_dir=None, overwrite=False,
                          crf=20, preset="slow", dry_run=False, ffprobe=FFPROBE)

    assert jobs[0].skip is True
    assert jobs[0].skip_reason is not None


def test_build_jobs_crf_and_preset_passed_through(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["ep01.mkv"])

    with patch("mediashrink.scanner.probe_video_codec", return_value="h264"):
        jobs = build_jobs(files, output_dir=None, overwrite=False,
                          crf=22, preset="medium", dry_run=True, ffprobe=FFPROBE)

    assert jobs[0].crf == 22
    assert jobs[0].preset == "medium"
    assert jobs[0].dry_run is True
