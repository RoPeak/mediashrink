from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mediashrink.models import EncodeAttempt, EncodeJob, SessionFileEntry, SessionManifest

SESSION_VERSION = 4
_SUPPORTED_SESSION_VERSIONS = {1, 2, 3, 4}
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
        if manifest.version not in _SUPPORTED_SESSION_VERSIONS:
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
    *,
    policy: str | None = None,
    on_file_failure: str | None = None,
    use_calibration: bool = True,
    retry_mode: str | None = None,
    queue_strategy: str | None = None,
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
        policy=policy,
        on_file_failure=on_file_failure,
        use_calibration=use_calibration,
        retry_mode=retry_mode,
        queue_strategy=queue_strategy,
    )


def update_session_entry(
    manifest: SessionManifest,
    source: Path,
    status: str,
    output: Path | None = None,
    error: str | None = None,
    encoder: str | None = None,
    last_progress_pct: float | None = None,
    last_progress_at: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    fallback_used: bool | None = None,
    retry_count: int | None = None,
    first_error: str | None = None,
    last_error: str | None = None,
    cleanup_result: str | None = None,
    attempt_history: list[EncodeAttempt] | None = None,
) -> None:
    """Mutate the entry matching `source` in place."""
    source_str = str(source)
    for entry in manifest.entries:
        if entry.source == source_str:
            entry.status = status
            if output is not None:
                entry.output = str(output)
            entry.error = error
            if encoder is not None:
                entry.encoder = encoder
            if last_progress_pct is not None:
                entry.last_progress_pct = last_progress_pct
            if last_progress_at is not None:
                entry.last_progress_at = last_progress_at
            if started_at is not None:
                entry.started_at = started_at
            if finished_at is not None:
                entry.finished_at = finished_at
            if fallback_used is not None:
                entry.fallback_used = fallback_used
            if retry_count is not None:
                entry.retry_count = retry_count
            if first_error is not None:
                entry.first_error = first_error
            if last_error is not None:
                entry.last_error = last_error
            if cleanup_result is not None:
                entry.cleanup_result = cleanup_result
            if attempt_history is not None:
                entry.attempt_history = list(attempt_history)
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
    actionable = [e for e in manifest.entries if e.status in {"pending", "failed", "in_progress"}]
    if not actionable:
        return None
    return manifest
