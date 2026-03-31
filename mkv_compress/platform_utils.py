from __future__ import annotations

import platform
import shutil
from pathlib import Path


def detect_os() -> str:
    """Return 'Windows', 'Linux', or 'Darwin'."""
    return platform.system()


def _find_binary(name: str, windows_fallbacks: list[Path]) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)

    if detect_os() == "Windows":
        for candidate in windows_fallbacks:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"{name!r} not found on PATH. "
        "Install FFmpeg and ensure it is available on your PATH.\n"
        "  Windows: https://ffmpeg.org/download.html\n"
        "  Linux:   sudo apt install ffmpeg  (Debian/Ubuntu)\n"
        "           sudo dnf install ffmpeg  (Fedora)"
    )


def find_ffmpeg() -> Path:
    return _find_binary(
        "ffmpeg",
        [
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        ],
    )


def find_ffprobe() -> Path:
    return _find_binary(
        "ffprobe",
        [
            Path(r"C:\Program Files\ffmpeg\bin\ffprobe.exe"),
            Path(r"C:\ffmpeg\bin\ffprobe.exe"),
        ],
    )


def check_ffmpeg_available() -> tuple[bool, str]:
    """Return (True, '') if both ffmpeg and ffprobe are found, else (False, message)."""
    try:
        find_ffmpeg()
        find_ffprobe()
        return True, ""
    except FileNotFoundError as exc:
        return False, str(exc)
