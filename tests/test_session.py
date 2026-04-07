from __future__ import annotations

from pathlib import Path

from mediashrink.models import EncodeAttempt, EncodeJob, SessionFileEntry, SessionManifest
from mediashrink.session import (
    SESSION_VERSION,
    build_session,
    find_resumable_session,
    get_session_path,
    load_session,
    save_session,
    update_session_entry,
)

FFPROBE = Path("/usr/bin/ffprobe")


def _make_job(tmp_path: Path, name: str = "ep01.mkv") -> EncodeJob:
    source = tmp_path / name
    source.write_bytes(b"fake")
    output = tmp_path / name.replace(".mkv", "_compressed.mkv")
    tmp_output = tmp_path / f".tmp_{name}"
    return EncodeJob(
        source=source,
        output=output,
        tmp_output=tmp_output,
        crf=20,
        preset="fast",
        dry_run=False,
    )


def test_build_and_load_session_round_trip(tmp_path: Path) -> None:
    jobs = [_make_job(tmp_path, "ep01.mkv"), _make_job(tmp_path, "ep02.mkv")]
    manifest = build_session(
        tmp_path,
        "fast",
        20,
        False,
        None,
        jobs,
        retry_mode="aggressive",
        queue_strategy="safe-first",
    )
    session_path = get_session_path(tmp_path, None)
    save_session(manifest, session_path)

    loaded = load_session(session_path)
    assert loaded is not None
    assert loaded.version == SESSION_VERSION
    assert loaded.preset == "fast"
    assert loaded.crf == 20
    assert loaded.retry_mode == "aggressive"
    assert loaded.queue_strategy == "safe-first"
    assert len(loaded.entries) == 2
    assert loaded.entries[0].status == "pending"
    assert loaded.entries[1].status == "pending"


def test_update_session_entry_mutates_status(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])

    update_session_entry(manifest, job.source, "success", output=job.output)

    assert manifest.entries[0].status == "success"
    assert manifest.entries[0].output == str(job.output)


def test_update_session_entry_records_error(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])

    update_session_entry(manifest, job.source, "failed", error="FFmpeg exited with code 1")

    assert manifest.entries[0].status == "failed"
    assert manifest.entries[0].error == "FFmpeg exited with code 1"


def test_update_session_entry_records_progress_telemetry(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])

    update_session_entry(
        manifest,
        job.source,
        "in_progress",
        encoder="fast",
        last_progress_pct=47.5,
        last_progress_at="2026-04-06T10:00:00+00:00",
        started_at="2026-04-06T09:00:00+00:00",
    )

    entry = manifest.entries[0]
    assert entry.status == "in_progress"
    assert entry.encoder == "fast"
    assert entry.last_progress_pct == 47.5
    assert entry.last_progress_at == "2026-04-06T10:00:00+00:00"
    assert entry.started_at == "2026-04-06T09:00:00+00:00"


def test_update_session_entry_records_attempt_history(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])

    update_session_entry(
        manifest,
        job.source,
        "failed",
        fallback_used=True,
        attempt_history=[
            EncodeAttempt(
                preset="amf",
                crf=20,
                success=False,
                duration_seconds=5.0,
                progress_pct=0.0,
                error_message="Invalid argument",
            ),
            EncodeAttempt(
                preset="faster",
                crf=22,
                success=True,
                duration_seconds=10.0,
                progress_pct=100.0,
            ),
        ],
    )

    entry = manifest.entries[0]
    assert entry.fallback_used is True
    assert entry.retry_count == 0
    assert len(entry.attempt_history) == 2
    assert entry.attempt_history[0].preset == "amf"


def test_update_session_entry_records_retry_and_cleanup_metadata(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])

    update_session_entry(
        manifest,
        job.source,
        "success",
        retry_count=1,
        first_error="device busy",
        last_error="device busy",
        cleanup_result="compressed output kept side-by-side",
    )

    entry = manifest.entries[0]
    assert entry.retry_count == 1
    assert entry.first_error == "device busy"
    assert entry.last_error == "device busy"
    assert entry.cleanup_result == "compressed output kept side-by-side"


def test_find_resumable_session_returns_none_when_settings_differ(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])
    session_path = get_session_path(tmp_path, None)
    save_session(manifest, session_path)

    # Different preset
    result = find_resumable_session(tmp_path, None, "slow", 20)
    assert result is None

    # Different CRF
    result = find_resumable_session(tmp_path, None, "fast", 28)
    assert result is None


def test_find_resumable_session_matches_preset_and_crf(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])
    session_path = get_session_path(tmp_path, None)
    save_session(manifest, session_path)

    result = find_resumable_session(tmp_path, None, "fast", 20)
    assert result is not None
    assert result.preset == "fast"
    assert result.crf == 20


def test_find_resumable_session_accepts_in_progress_entries(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])
    update_session_entry(manifest, job.source, "in_progress")
    session_path = get_session_path(tmp_path, None)
    save_session(manifest, session_path)

    result = find_resumable_session(tmp_path, None, "fast", 20)
    assert result is not None


def test_find_resumable_session_returns_none_when_all_done(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])
    update_session_entry(manifest, job.source, "success")
    session_path = get_session_path(tmp_path, None)
    save_session(manifest, session_path)

    result = find_resumable_session(tmp_path, None, "fast", 20)
    assert result is None


def test_load_session_returns_none_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / ".mediashrink-session.json"
    assert load_session(path) is None


def test_load_session_returns_none_for_wrong_version(tmp_path: Path) -> None:
    path = tmp_path / ".mediashrink-session.json"
    path.write_text(
        '{"version": 99, "directory": "/x", "timestamp": "t", "preset": "fast", "crf": 20, "overwrite": false, "output_dir": null, "entries": []}',
        encoding="utf-8",
    )
    assert load_session(path) is None


def test_load_session_accepts_prior_supported_version(tmp_path: Path) -> None:
    path = tmp_path / ".mediashrink-session.json"
    path.write_text(
        '{"version": 1, "directory": "/x", "timestamp": "t", "preset": "fast", "crf": 20, "overwrite": false, "output_dir": null, "entries": []}',
        encoding="utf-8",
    )
    loaded = load_session(path)
    assert loaded is not None
    assert loaded.version == 1


def test_session_skipped_jobs_marked_skipped(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    job.skip = True
    job.skip_reason = "already HEVC"
    manifest = build_session(tmp_path, "fast", 20, False, None, [job])
    assert manifest.entries[0].status == "skipped"
