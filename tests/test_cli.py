from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from mediashrink.analysis import build_manifest, save_manifest
from mediashrink.cli import app
from mediashrink.models import AnalysisItem, EncodeAttempt, EncodeJob, EncodeResult
from mediashrink.profiles import SavedProfile, upsert_profile
from mediashrink.session import build_session, get_session_path, save_session

runner = CliRunner()

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def _make_result(job: EncodeJob, **kwargs) -> EncodeResult:
    defaults = dict(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1_000_000_000,
        output_size_bytes=300_000_000,
        duration_seconds=120.0,
    )
    defaults.update(kwargs)
    return EncodeResult(**defaults)


def _make_job(source: Path) -> EncodeJob:
    return EncodeJob(
        source=source,
        output=source.with_stem(source.stem + "_compressed"),
        tmp_output=source.parent / f".tmp_{source.stem}_compressed{source.suffix}",
        crf=20,
        preset="slow",
        dry_run=False,
        skip=False,
        skip_reason=None,
    )


def test_missing_directory_error() -> None:
    result = runner.invoke(app, ["/nonexistent/path"])
    assert result.exit_code != 0


def test_ffmpeg_not_found_error(tmp_path: Path) -> None:
    (tmp_path / "ep01.mkv").write_bytes(b"fake")

    with patch("mediashrink.cli.check_ffmpeg_available", return_value=(False, "ffmpeg not found")):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 4
    assert "ffmpeg not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_no_supported_video_files(tmp_path: Path) -> None:
    (tmp_path / "cover.jpg").touch()

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 1
    assert "No supported video files" in result.stdout


def test_dry_run_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.dry_run = True
    fake_result = _make_result(fake_job, output_size_bytes=0)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[mkv]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--dry-run", "--yes"])

    assert result.exit_code == 0


def test_yes_flag_skips_prompt(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_result = _make_result(fake_job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[mkv]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.progress.typer.confirm") as mock_confirm,
        patch("mediashrink.cli.typer.confirm") as mock_cleanup_confirm,
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes"])

    mock_confirm.assert_not_called()
    mock_cleanup_confirm.assert_not_called()
    assert result.exit_code == 0


def test_all_skipped_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.skip = True
    fake_job.skip_reason = "already HEVC"

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[mkv]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file") as mock_encode,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    mock_encode.assert_not_called()
    assert result.exit_code == 3
    assert "Nothing to encode" in result.stdout


def test_profile_option_applies_saved_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)

    upsert_profile(SavedProfile(name="tv", preset="nvenc", crf=24, created_from_wizard=True))

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--profile", "tv", "--yes"])

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["crf"] == 24
    assert mock_build_jobs.call_args.kwargs["preset"] == "nvenc"


def test_explicit_flags_override_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)

    upsert_profile(SavedProfile(name="tv", preset="nvenc", crf=24))

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(
            app,
            [str(tmp_path), "--profile", "tv", "--crf", "18", "--preset", "slow", "--yes"],
        )

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["crf"] == 18
    assert mock_build_jobs.call_args.kwargs["preset"] == "slow"


def test_profiles_list_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    upsert_profile(
        SavedProfile(name="tv", preset="fast", crf=20, label="Balanced", created_from_wizard=True)
    )

    list_result = runner.invoke(app, ["profiles", "list"])
    delete_result = runner.invoke(app, ["profiles", "delete", "tv"])
    list_after_delete = runner.invoke(app, ["profiles", "list"])

    assert list_result.exit_code == 0
    assert "tv: preset=fast, crf=20 - Balanced (wizard)" in list_result.stdout
    assert delete_result.exit_code == 0
    assert "Deleted profile" in delete_result.stdout
    # After deleting the only user profile, builtins are still listed
    assert "tv:" not in list_after_delete.stdout
    assert "Fast Batch" in list_after_delete.stdout


def test_wizard_subcommand_registered() -> None:
    result = runner.invoke(app, ["wizard", "--help"])
    assert result.exit_code == 0
    assert "wizard" in result.stdout.lower() or "encoding" in result.stdout.lower()


def test_wizard_is_recursive_by_default(tmp_path: Path) -> None:
    with (
        patch("mediashrink.cli._prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.cli.EncodingDisplay"),
        patch(
            "mediashrink.wizard.run_wizard", return_value=([], "cancel", False)
        ) as mock_run_wizard,
    ):
        result = runner.invoke(app, ["wizard", str(tmp_path)])

    assert result.exit_code == 3
    assert mock_run_wizard.call_args.kwargs["recursive"] is True


def test_analyze_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    manifest_path = tmp_path / "analysis.json"
    item = AnalysisItem(
        source=source,
        codec="h264",
        size_bytes=2 * 1024**3,
        duration_seconds=120.0,
        bitrate_kbps=12000.0,
        estimated_output_bytes=800 * 1024**2,
        estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
        recommendation="recommended",
        reason_code="strong_savings_candidate",
        reason_text="legacy codec with strong projected space savings",
    )

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli._analyze_with_optional_progress", return_value=[item]),
        patch("mediashrink.cli.estimate_analysis_encode_seconds", return_value=600.0),
    ):
        result = runner.invoke(
            app, ["analyze", str(tmp_path), "--manifest-out", str(manifest_path)]
        )

    assert result.exit_code == 0
    assert manifest_path.exists()
    assert "Wrote manifest" in result.stdout


def test_analyze_profile_and_explicit_overrides_apply(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    upsert_profile(SavedProfile(name="tv", preset="nvenc", crf=24))
    item = AnalysisItem(
        source=source,
        codec="h264",
        size_bytes=2 * 1024**3,
        duration_seconds=120.0,
        bitrate_kbps=12000.0,
        estimated_output_bytes=800 * 1024**2,
        estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
        recommendation="recommended",
        reason_code="strong_savings_candidate",
        reason_text="legacy codec with strong projected space savings",
    )

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli._analyze_with_optional_progress", return_value=[item]),
        patch("mediashrink.cli.estimate_analysis_encode_seconds", return_value=600.0),
        patch("mediashrink.cli.build_manifest") as mock_build_manifest,
        patch("mediashrink.cli.save_manifest") as mock_save_manifest,
    ):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path),
                "--profile",
                "tv",
                "--crf",
                "18",
                "--preset",
                "slow",
                "--manifest-out",
                str(tmp_path / "analysis.json"),
            ],
        )

    assert result.exit_code == 0
    assert mock_build_manifest.call_args.kwargs["crf"] == 18
    assert mock_build_manifest.call_args.kwargs["preset"] == "slow"
    assert mock_build_manifest.call_args.kwargs["profile_name"] == "tv"
    mock_save_manifest.assert_called_once()


def test_apply_uses_manifest_settings(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    manifest = build_manifest(
        directory=tmp_path,
        recursive=True,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=100.0,
        estimate_confidence=None,
        items=[
            AnalysisItem(
                source=source,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            )
        ],
    )
    manifest_path = tmp_path / "analysis.json"
    save_manifest(manifest, manifest_path)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["crf"] == 20
    assert mock_build_jobs.call_args.kwargs["preset"] == "fast"


def test_apply_explicit_overrides_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    upsert_profile(SavedProfile(name="tv", preset="nvenc", crf=24))
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    manifest = build_manifest(
        directory=tmp_path,
        recursive=False,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=None,
        estimate_confidence=None,
        items=[
            AnalysisItem(
                source=source,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            )
        ],
    )
    manifest_path = tmp_path / "analysis.json"
    save_manifest(manifest, manifest_path)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(
            app,
            [
                "apply",
                str(manifest_path),
                "--profile",
                "tv",
                "--crf",
                "18",
                "--preset",
                "slow",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["crf"] == 18
    assert mock_build_jobs.call_args.kwargs["preset"] == "slow"


def test_apply_reports_missing_manifest_files(tmp_path: Path) -> None:
    existing = tmp_path / "exists.mkv"
    existing.write_bytes(b"x" * 1000)
    missing = tmp_path / "missing.mkv"
    fake_job = _make_job(existing)
    fake_result = _make_result(fake_job)
    manifest = build_manifest(
        directory=tmp_path,
        recursive=False,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=None,
        estimate_confidence=None,
        items=[
            AnalysisItem(
                source=existing,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            ),
            AnalysisItem(
                source=missing,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            ),
        ],
    )
    manifest_path = tmp_path / "analysis.json"
    save_manifest(manifest, manifest_path)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["files"] == [existing]
    assert "Missing file from manifest" in result.stdout


def test_apply_warns_when_output_container_drops_subtitles(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    manifest = build_manifest(
        directory=tmp_path,
        recursive=False,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=None,
        estimate_confidence=None,
        items=[
            AnalysisItem(
                source=source,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            )
        ],
    )
    manifest_path = tmp_path / "analysis.json"
    save_manifest(manifest, manifest_path)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.source_has_subtitle_streams", return_value=True),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    assert "Subtitle warning:" in result.stdout


def test_encode_prompts_for_cleanup_after_successful_side_by_side_run(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    fake_job.output.write_bytes(b"compressed")

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed", return_value=True),
        patch("mediashrink.cli.typer.confirm", return_value=False) as mock_cleanup_confirm,
        patch("mediashrink.cli.cleanup_successful_results") as mock_cleanup,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    mock_cleanup_confirm.assert_called_once()
    mock_cleanup.assert_not_called()


def test_encode_cleanup_flag_runs_without_prompt(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    fake_job.output.write_bytes(b"compressed")

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed", return_value=True),
        patch("mediashrink.cli.typer.confirm") as mock_cleanup_confirm,
        patch("mediashrink.cli.cleanup_successful_results", return_value=[source]) as mock_cleanup,
    ):
        result = runner.invoke(app, [str(tmp_path), "--cleanup", "--yes"])

    assert result.exit_code == 0
    mock_cleanup_confirm.assert_not_called()
    mock_cleanup.assert_called_once()


def test_apply_yes_does_not_prompt_for_cleanup_without_flag(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mp4"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)
    manifest = build_manifest(
        directory=tmp_path,
        recursive=False,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=None,
        estimate_confidence=None,
        items=[
            AnalysisItem(
                source=source,
                codec="h264",
                size_bytes=1000,
                duration_seconds=120.0,
                bitrate_kbps=12000.0,
                estimated_output_bytes=400,
                estimated_savings_bytes=600,
                recommendation="recommended",
                reason_code="strong_savings_candidate",
                reason_text="legacy codec with strong projected space savings",
            )
        ],
    )
    manifest_path = tmp_path / "analysis.json"
    save_manifest(manifest, manifest_path)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.cli.typer.confirm") as mock_cleanup_confirm,
    ):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    mock_cleanup_confirm.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 6 — preview command
# ---------------------------------------------------------------------------


def test_preview_command_exits_zero(tmp_path: Path) -> None:
    source = tmp_path / "film.mkv"
    source.write_bytes(b"x" * 1000)

    job = EncodeJob(
        source=source,
        output=tmp_path / "film_preview.mkv",
        tmp_output=tmp_path / ".tmp_film_preview.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    fake_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=1.0,
    )

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.encode_preview", return_value=fake_result),
    ):
        result = runner.invoke(app, ["preview", str(source)])

    assert result.exit_code == 0


def test_preview_directory_uses_representative_files(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.mkv"
    h264 = tmp_path / "h264.mkv"
    maybe = tmp_path / "maybe.mkv"
    for path in (legacy, h264, maybe):
        path.write_bytes(b"x" * 1000)

    def _item(path: Path, codec: str, recommendation: str) -> AnalysisItem:
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=1000,
            duration_seconds=120.0,
            bitrate_kbps=1000.0,
            estimated_output_bytes=500,
            estimated_savings_bytes=500,
            recommendation=recommendation,
            reason_code="reason",
            reason_text="reason",
        )

    fake_result = _make_result(_make_job(legacy))
    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[legacy, h264, maybe]),
        patch(
            "mediashrink.cli.analyze_files",
            return_value=[
                _item(legacy, "vc1", "recommended"),
                _item(h264, "h264", "recommended"),
                _item(maybe, "hevc", "maybe"),
            ],
        ),
        patch("mediashrink.cli.encode_preview", return_value=fake_result) as mock_preview,
    ):
        result = runner.invoke(app, ["preview", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    assert mock_preview.call_count == 3


def test_preview_command_unsupported_ext(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"fake")

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
    ):
        result = runner.invoke(app, ["preview", str(source)])

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Stage 7 — Exit codes
# ---------------------------------------------------------------------------


def test_exit_no_files_is_1(tmp_path: Path) -> None:
    from mediashrink.cli import EXIT_NO_FILES

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.scan_directory", return_value=[]),
    ):
        result = runner.invoke(app, ["encode", str(tmp_path)])
    assert result.exit_code == EXIT_NO_FILES
    assert EXIT_NO_FILES == 1


def test_exit_ffmpeg_not_found_is_4(tmp_path: Path) -> None:
    from mediashrink.cli import EXIT_FFMPEG_NOT_FOUND

    with patch("mediashrink.cli.check_ffmpeg_available", return_value=(False, "ffmpeg not found")):
        result = runner.invoke(app, ["encode", str(tmp_path)])
    assert result.exit_code == EXIT_FFMPEG_NOT_FOUND
    assert EXIT_FFMPEG_NOT_FOUND == 4


def test_exit_success_is_0() -> None:
    from mediashrink.cli import EXIT_SUCCESS

    assert EXIT_SUCCESS == 0


def test_exit_encode_failures_is_2(tmp_path: Path) -> None:
    from mediashrink.cli import EXIT_ENCODE_FAILURES

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 100)

    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    failed_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=False,
        input_size_bytes=100,
        output_size_bytes=0,
        duration_seconds=0.1,
        error_message="FFmpeg exited with code 1",
    )

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[failed_result]),
    ):
        result = runner.invoke(app, ["encode", str(tmp_path), "--yes"])

    assert result.exit_code == EXIT_ENCODE_FAILURES
    assert EXIT_ENCODE_FAILURES == 2


# ---------------------------------------------------------------------------
# Stage 8 — JSON output and verbose logging
# ---------------------------------------------------------------------------


def test_json_flag_outputs_valid_json(tmp_path: Path) -> None:
    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)

    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    ok_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=1.0,
    )

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]),
    ):
        result = runner.invoke(app, ["encode", str(tmp_path), "--yes", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["exit_code"] == 0
    assert len(parsed["files"]) == 1
    assert parsed["files"][0]["status"] == "success"


def test_json_flag_suppresses_rich_markup(tmp_path: Path) -> None:
    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)

    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    ok_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=1.0,
    )

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]),
    ):
        result = runner.invoke(app, ["encode", str(tmp_path), "--yes", "--json"])

    # Output should be a single JSON line — no Rich markup tags
    assert "[" not in result.stdout.split("\n")[0].replace("[", "").replace(
        "]", ""
    ) or result.stdout.strip().startswith("{")
    assert result.stdout.strip().startswith("{")


def test_verbose_flag_creates_log_file(tmp_path: Path) -> None:
    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)

    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    ok_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=1.0,
    )

    with (
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]) as mock_loop,
    ):
        result = runner.invoke(app, ["encode", str(tmp_path), "--yes", "--verbose"])

    assert result.exit_code == 0
    # log_path should have been passed as a keyword arg to _run_encode_loop
    call_kwargs = mock_loop.call_args.kwargs
    assert call_kwargs.get("log_path") is not None
    assert str(call_kwargs["log_path"]).endswith(".log")
    assert "mediashrink_" in str(call_kwargs["log_path"])


# ---------------------------------------------------------------------------
# Stage 10 — UX polish regression tests
# ---------------------------------------------------------------------------


def test_progress_bar_nonzero_total(tmp_path: Path) -> None:
    """Per-file task total passed to progress.update must never be zero."""
    from mediashrink.cli import _run_encode_loop
    from unittest.mock import MagicMock, call

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 5_000_000)  # 5 MB
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=True,
        skip_reason="already HEVC",
    )
    ok_result = EncodeResult(
        job=job,
        skipped=True,
        skip_reason="already HEVC",
        success=False,
        input_size_bytes=5_000_000,
        output_size_bytes=0,
        duration_seconds=0.0,
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]  # overall_task=0, file_task=1

    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    with patch("mediashrink.cli.encode_file", return_value=ok_result):
        _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    # Collect all 'total' kwargs passed to progress.update
    totals = [
        kwargs["total"] for _, kwargs in mock_progress.update.call_args_list if "total" in kwargs
    ]
    assert totals, "progress.update was never called with a total="
    assert all(t >= 1 for t in totals), f"Found zero or negative total: {totals}"


def test_run_encode_loop_removes_file_task_and_updates_overall_only_on_completion(
    tmp_path: Path,
) -> None:
    from mediashrink.cli import _run_encode_loop
    from unittest.mock import MagicMock

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 5_000_000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    ok_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=5_000_000,
        output_size_bytes=2_500_000,
        duration_seconds=2.0,
        media_duration_seconds=120.0,
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]

    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    def fake_encode_file(*args: object, **kwargs: object) -> EncodeResult:
        progress_callback = kwargs["progress_callback"]
        assert callable(progress_callback)
        progress_callback(50.0)
        return ok_result

    with patch("mediashrink.cli.encode_file", side_effect=fake_encode_file):
        _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    overall_updates = [
        kwargs["completed"]
        for args, kwargs in mock_progress.update.call_args_list
        if args and args[0] == 0 and "completed" in kwargs
    ]
    assert overall_updates == [5_000_000]
    mock_progress.remove_task.assert_called_once_with(1)
    mock_display.show_summary.assert_called_once_with(
        [ok_result],
        resumed_from_session=False,
        previously_completed=0,
        previously_skipped=0,
    )


def test_run_encode_loop_interrupt_preserves_session_and_prints_resume_guidance(
    tmp_path: Path,
) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 5_000_000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    session = build_session(tmp_path, "fast", 20, False, None, [job])
    session_path = get_session_path(tmp_path, None)
    save_session(session, session_path)

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]

    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    def fake_encode_file(*args: object, **kwargs: object) -> EncodeResult:
        progress_callback = kwargs["progress_callback"]
        progress_callback(47.0)
        raise KeyboardInterrupt

    with patch("mediashrink.cli.encode_file", side_effect=fake_encode_file):
        with pytest.raises(typer.Exit) as exc:
            _run_encode_loop(
                [job],
                FFMPEG,
                FFPROBE,
                mock_display,
                session=session,
                session_path=session_path,
            )

    assert exc.value.exit_code == 3
    saved = json.loads(session_path.read_text(encoding="utf-8"))
    entry = saved["entries"][0]
    assert entry["status"] == "pending"
    assert entry["error"] == "Interrupted by user"
    assert entry["last_progress_pct"] == 47.0


def test_encode_command_shows_resume_counts_and_path(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    source2 = tmp_path / "ep02.mkv"
    source2.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    pending_job = _make_job(source2)
    fake_result = _make_result(pending_job)
    session = build_session(tmp_path, "slow", 20, False, None, [fake_job, pending_job])
    session.entries[0].status = "success"
    save_session(session, get_session_path(tmp_path, None))

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source, source2]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job, pending_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--preset", "slow"], input="y\ny\n")

    assert result.exit_code == 0
    assert "Session found: 1 done, 1 pending, 0 failed, 0 skipped" in result.stdout
    assert "Session path:" in result.stdout


def test_resume_command_uses_session_settings_and_resumes_pending(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    session = build_session(tmp_path, "slow", 18, False, None, [job])
    save_session(session, get_session_path(tmp_path, None))
    ok_result = _make_result(job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]) as mock_loop,
    ):
        result = runner.invoke(app, ["resume", str(tmp_path), "--yes"])

    assert result.exit_code == 0
    call_kwargs = mock_loop.call_args.kwargs
    assert call_kwargs["session"].preset == "slow"
    assert call_kwargs["session"].crf == 18


def test_resume_command_fails_when_session_missing(tmp_path: Path) -> None:
    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
    ):
        result = runner.invoke(app, ["resume", str(tmp_path)])

    assert result.exit_code == 1
    assert "no resumable session found" in result.stdout.lower()


def test_encode_warns_on_low_disk_and_aborts_when_declined(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_job.estimated_output_bytes = 800

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.shutil.disk_usage") as mock_disk_usage,
        patch("mediashrink.cli.typer.confirm", return_value=False),
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed") as mock_confirm_proceed,
    ):
        mock_disk_usage.return_value = shutil._ntuple_diskusage(1000, 990, 10)
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 3
    assert "Low disk space warning" in result.stdout
    mock_confirm_proceed.assert_not_called()


def test_encode_writes_json_and_text_reports(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes"])

    assert result.exit_code == 0
    report_json = next(tmp_path.glob("mediashrink_report_*.json"))
    report_text = next(tmp_path.glob("mediashrink_report_*.txt"))
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["mode"] == "encode"
    assert "warnings" in report_payload
    assert "retry_count" in report_payload["files"][0]
    assert report_payload["files"][0]["source"].endswith("ep01.mkv")
    assert "mediashrink batch report" in report_text.read_text(encoding="utf-8")


def test_write_batch_reports_include_skip_breakdown(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    incompatible_job = _make_job(tmp_path / "bad.mp4")
    policy_job = _make_job(tmp_path / "policy.mkv")
    for job in (incompatible_job, policy_job):
        job.source.write_bytes(b"x" * 1000)

    incompatible = _make_result(
        incompatible_job,
        skipped=True,
        success=False,
        skip_reason="incompatible: unsupported container/stream combination",
        output_size_bytes=0,
    )
    skipped_by_policy = _make_result(
        policy_job,
        skipped=True,
        success=False,
        skip_reason="skipped_by_policy: disk full",
        output_size_bytes=0,
    )

    json_path, text_path = _write_batch_reports(
        mode="encode",
        base_dir=tmp_path,
        output_dir=None,
        manifest_path=None,
        preset="fast",
        crf=20,
        overwrite=False,
        cleanup_requested=False,
        resumed_from_session=False,
        session_path=None,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T01:00:00+00:00",
        results=[incompatible, skipped_by_policy],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["totals"]["skipped_incompatible"] == 1
    assert payload["totals"]["skipped_by_policy"] == 1
    assert "Skipped incompatible: 1" in text
    assert "Skipped by policy: 1" in text


def test_run_encode_loop_retries_early_hardware_failure_with_software_fallback(
    tmp_path: Path,
) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="amf",
        dry_run=False,
        skip=False,
    )
    failed = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=5.0,
        error_message="Invalid argument",
        attempts=[
            EncodeAttempt(
                preset="amf",
                crf=20,
                success=False,
                duration_seconds=5.0,
                progress_pct=0.0,
                error_message="Invalid argument",
            )
        ],
    )
    fallback_job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=22,
        preset="faster",
        dry_run=False,
        skip=False,
    )
    succeeded = EncodeResult(
        job=fallback_job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=10.0,
        attempts=[
            EncodeAttempt(
                preset="faster",
                crf=22,
                success=True,
                duration_seconds=10.0,
                progress_pct=100.0,
            )
        ],
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]
    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    with patch("mediashrink.cli.encode_file", side_effect=[failed, succeeded]):
        results = _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].fallback_used is True
    assert results[0].job.preset == "faster"
    assert len(results[0].attempts) == 2


def test_run_encode_loop_normalizes_common_failure_message(tmp_path: Path) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    failed = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=5.0,
        error_message="No space left on device",
        attempts=[
            EncodeAttempt(
                preset="fast",
                crf=20,
                success=False,
                duration_seconds=5.0,
                progress_pct=50.0,
                error_message="No space left on device",
            )
        ],
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]
    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    with patch("mediashrink.cli.encode_file", return_value=failed):
        results = _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    assert results[0].error_message == "Disk full: no space left on device."


def test_run_encode_loop_retries_transient_software_failure_once(tmp_path: Path) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    failed = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=False,
        input_size_bytes=1000,
        output_size_bytes=0,
        duration_seconds=2.0,
        error_message="Resource temporarily unavailable",
        raw_error_message="Resource temporarily unavailable",
        attempts=[
            EncodeAttempt(
                preset="fast",
                crf=20,
                success=False,
                duration_seconds=2.0,
                progress_pct=0.0,
                error_message="Resource temporarily unavailable",
            )
        ],
    )
    succeeded = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=8.0,
        attempts=[
            EncodeAttempt(
                preset="fast",
                crf=20,
                success=True,
                duration_seconds=8.0,
                progress_pct=100.0,
            )
        ],
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]
    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    with patch("mediashrink.cli.encode_file", side_effect=[failed, succeeded]):
        results = _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    assert results[0].success is True
    assert results[0].retry_kind == "io_temporary"
    assert results[0].retry_count == 1
    assert len(results[0].attempts) == 2


def test_encode_passes_custom_stall_warning_seconds(tmp_path: Path) -> None:
    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    ok_result = _make_result(job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]) as mock_loop,
    ):
        result = runner.invoke(
            app,
            [str(tmp_path), "--yes", "--stall-warning-seconds", "12"],
        )

    assert result.exit_code == 0
    assert mock_loop.call_args.kwargs["stall_warning_seconds"] == 12.0


def test_run_encode_loop_warns_when_progress_stalls(tmp_path: Path) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 5_000_000)
    job = EncodeJob(
        source=source,
        output=tmp_path / "vid_out.mkv",
        tmp_output=tmp_path / ".tmp_vid_out.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    ok_result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=5_000_000,
        output_size_bytes=2_500_000,
        duration_seconds=2.0,
        media_duration_seconds=120.0,
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]

    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    def fake_encode_file(*args: object, **kwargs: object) -> EncodeResult:
        time.sleep(0.03)
        return ok_result

    with (
        patch("mediashrink.cli.encode_file", side_effect=fake_encode_file),
        patch("mediashrink.cli.STALL_WARNING_SECONDS", 0.01),
        patch("mediashrink.cli.STALL_POLL_SECONDS", 0.005),
        patch("mediashrink.cli.console.print") as mock_print,
    ):
        _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "No progress update from FFmpeg" in printed


def test_run_encode_loop_skip_policy_marks_failed_file_skipped(tmp_path: Path) -> None:
    from mediashrink.cli import _run_encode_loop

    source = tmp_path / "vid.mkv"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    failed = _make_result(job, success=False, output_size_bytes=0, error_message="disk full")

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]
    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    mock_display.show_summary = MagicMock()

    with patch("mediashrink.cli.encode_file", return_value=failed):
        results = _run_encode_loop(
            [job], FFMPEG, FFPROBE, mock_display, on_file_failure="skip", use_calibration=False
        )

    assert results[0].skipped is True
    assert "skipped_by_policy" in (results[0].skip_reason or "")


def test_encode_overnight_flag_forces_skip_policy_and_verbose(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    ok_result = _make_result(job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]) as mock_loop,
    ):
        result = runner.invoke(app, [str(tmp_path), "--overnight"])

    assert result.exit_code == 0
    assert mock_loop.call_args.kwargs["on_file_failure"] == "skip"
    assert mock_loop.call_args.kwargs["use_calibration"] is True


def test_encode_skip_policy_skips_incompatible_file_before_batch(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    ok_result = _make_result(
        job, skipped=True, skip_reason="incompatible: unsupported container/stream combination"
    )

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch(
            "mediashrink.cli.preflight_encode_job",
            return_value=_make_result(
                job, success=False, output_size_bytes=0, error_message="Could not write header"
            ),
        ),
        patch("mediashrink.cli.encode_file", return_value=ok_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes", "--on-file-failure", "skip"])

    assert result.exit_code == 0
    assert "Skipping 1 incompatible file(s) before batch start" in result.stdout


def test_encode_stop_policy_aborts_on_incompatible_file_before_batch(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch(
            "mediashrink.cli.preflight_encode_job",
            return_value=_make_result(
                job, success=False, output_size_bytes=0, error_message="Could not write header"
            ),
        ),
        patch("mediashrink.cli.encode_file") as mock_encode,
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes", "--on-file-failure", "stop"])

    assert result.exit_code == 2
    mock_encode.assert_not_called()
    assert "Compatibility check failed" in result.stdout


def test_encode_skip_policy_reports_specific_subtitle_incompatibility(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch(
            "mediashrink.cli.preflight_encode_job",
            return_value=_make_result(
                job,
                success=False,
                output_size_bytes=0,
                error_message="Subtitle codec 94213 is not supported in mov_text",
            ),
        ),
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes", "--on-file-failure", "skip"])

    assert result.exit_code == 0
    assert "subtitle codec is not supported by the chosen output container" in result.stdout


def test_overnight_command_runs_prepare_and_encode(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    ok_result = _make_result(job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch(
            "mediashrink.cli._prepare_overnight_jobs",
            return_value=([job], "fast", 20),
        ) as mock_prepare,
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]) as mock_loop,
    ):
        result = runner.invoke(app, ["overnight", str(tmp_path)])

    assert result.exit_code == 0
    mock_prepare.assert_called_once()
    assert mock_loop.call_args.kwargs["on_file_failure"] == "skip"
