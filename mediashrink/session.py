from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mediashrink.models import EncodeJob, SessionFileEntry, SessionManifest

SESSION_VERSION = 1
_SESSION_FILENAME = ".mediashrink-session.json"


def get_session_path(directory: Path, output_dir: Path | None) -> Path:
    return (output_dir or directory) / _SESSION_FILENAME


def load_session(path: Path) -> SessionManifest | None:
    """Load a session manifest from disk. Returns None if missing, unreadable, or wrong version."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        manifest = SessionManifest.from_dict(raw)
        if manifest.version != SESSION_VERSION:
            return None
        return manifest
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_session(manifest: SessionManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def build_session(
    directory: Path,
    preset: str,
    crf: int,
    overwrite: bool,
    output_dir: Path | None,
    jobs: list[EncodeJob],
) -> SessionManifest:
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    entries = [
        SessionFileEntry(
            source=str(job.source),
            status="skipped" if job.skip else "pending",
            output=str(job.output) if not job.skip else None,
        )
        for job in jobs
    ]
    return SessionManifest(
        version=SESSION_VERSION,
        directory=str(directory),
        timestamp=timestamp,
        preset=preset,
        crf=crf,
        overwrite=overwrite,
        output_dir=str(output_dir) if output_dir is not None else None,
        entries=entries,
    )


def update_session_entry(
    manifest: SessionManifest,
    source: Path,
    status: str,
    output: Path | None = None,
    error: str | None = None,
) -> None:
    """Mutate the entry matching `source` in place."""
    source_str = str(source)
    for entry in manifest.entries:
        if entry.source == source_str:
            entry.status = status
            if output is not None:
                entry.output = str(output)
            if error is not None:
                entry.error = error
            return


def find_resumable_session(
    directory: Path,
    output_dir: Path | None,
    preset: str,
    crf: int,
) -> SessionManifest | None:
    """Return a session if one exists for the same directory/preset/crf, else None."""
    path = get_session_path(directory, output_dir)
    manifest = load_session(path)
    if manifest is None:
        return None
    if manifest.preset != preset or manifest.crf != crf:
        return None
    if manifest.directory != str(directory):
        return None
    # Only resumable if there are pending or failed entries
    actionable = [e for e in manifest.entries if e.status in {"pending", "failed"}]
    if not actionable:
        return None
    return manifest
