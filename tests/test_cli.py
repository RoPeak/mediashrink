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


def test_bare_interactive_invocation_can_route_to_wizard(tmp_path: Path) -> None:
    with (
        patch("mediashrink.cli._choose_interactive_default_command", return_value="wizard"),
        patch("mediashrink.cli._prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.cli.EncodingDisplay"),
        patch(
            "mediashrink.wizard.run_wizard",
            return_value=([], "cancel", False, None),
        ) as mock_run_wizard,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 3
    assert mock_run_wizard.called


def test_explicit_subcommand_bypasses_default_mode_prompt(tmp_path: Path) -> None:
    with patch("mediashrink.cli._choose_interactive_default_command") as mock_choose:
        result = runner.invoke(app, ["wizard", str(tmp_path)])

    assert result.exit_code == 3
    mock_choose.assert_not_called()


def test_choose_interactive_default_command_skips_prompt_for_json() -> None:
    from mediashrink.cli import _choose_interactive_default_command

    with patch("mediashrink.cli.typer.prompt") as mock_prompt:
        chosen = _choose_interactive_default_command(["/tmp/library", "--json"])

    assert chosen == "encode"
    mock_prompt.assert_not_called()


def test_wizard_is_recursive_by_default(tmp_path: Path) -> None:
    with (
        patch("mediashrink.cli._prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.cli.EncodingDisplay"),
        patch(
            "mediashrink.wizard.run_wizard", return_value=([], "cancel", False, None)
        ) as mock_run_wizard,
    ):
        result = runner.invoke(app, ["wizard", str(tmp_path)])

    assert result.exit_code == 3
    assert mock_run_wizard.call_args.kwargs["recursive"] is True


def test_wizard_passes_new_reliability_flags(tmp_path: Path) -> None:
    with (
        patch("mediashrink.cli._prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.cli.EncodingDisplay"),
        patch(
            "mediashrink.wizard.run_wizard", return_value=([], "cancel", False, None)
        ) as mock_run_wizard,
    ):
        result = runner.invoke(
            app,
            [
                "wizard",
                str(tmp_path),
                "--non-interactive-wizard",
                "--debug-session-log",
            ],
        )

    assert result.exit_code == 3
    assert mock_run_wizard.call_args.kwargs["non_interactive_wizard"] is True
    assert mock_run_wizard.call_args.kwargs["debug_session_log"] is True


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


def test_analyze_surfaces_filename_hygiene_and_persists_notes(tmp_path: Path) -> None:
    source = tmp_path / "Mergers &amp; Acquisitions.mkv"
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
    assert "Filename hygiene warning" in result.stdout
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any(
        "Mergers &amp; Acquisitions.mkv -> Mergers & Acquisitions.mkv" in note
        for note in payload["notes"]
    )


def test_analyze_writes_split_manifest_index(tmp_path: Path) -> None:
    first = tmp_path / "Show A - s01e01 - Pilot.mkv"
    second = tmp_path / "Show B - s01e01 - Start.mkv"
    for path in (first, second):
        path.write_bytes(b"x" * 1000)
    manifest_path = tmp_path / "split-index.json"
    items = [
        AnalysisItem(
            source=first,
            codec="h264",
            size_bytes=2 * 1024**3,
            duration_seconds=120.0,
            bitrate_kbps=12000.0,
            estimated_output_bytes=800 * 1024**2,
            estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
            recommendation="recommended",
            reason_code="strong_savings_candidate",
            reason_text="legacy codec with strong projected space savings",
        ),
        AnalysisItem(
            source=second,
            codec="h264",
            size_bytes=2 * 1024**3,
            duration_seconds=120.0,
            bitrate_kbps=12000.0,
            estimated_output_bytes=800 * 1024**2,
            estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
            recommendation="recommended",
            reason_code="strong_savings_candidate",
            reason_text="legacy codec with strong projected space savings",
        ),
    ]

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli._analyze_with_optional_progress", return_value=items),
        patch("mediashrink.cli.estimate_analysis_encode_seconds", return_value=600.0),
    ):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path),
                "--manifest-out",
                str(manifest_path),
                "--manifest-split-by",
                "show",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["split_by"] == "show"
    assert len(payload["generated_manifests"]) == 2


def test_analyze_split_requires_manifest_out(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", str(tmp_path), "--manifest-split-by", "show"])

    assert result.exit_code != 0
    assert "manifest-out" in result.stdout.lower()


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
        patch(
            "mediashrink.cli._maybe_prepare_mkv_reroute",
            return_value=([fake_job], [], None, [], []),
        ),
        patch("mediashrink.cli.source_has_subtitle_streams", return_value=True),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
    ):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    assert "Subtitle warning:" in result.stdout


def test_encode_prompts_for_cleanup_before_confirmation(tmp_path: Path) -> None:
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
        patch("mediashrink.cli._maybe_prompt_for_cleanup") as mock_prompt_cleanup,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    mock_cleanup_confirm.assert_called_once()
    mock_prompt_cleanup.assert_not_called()
    assert "Cleanup decision:" in result.stdout
    assert "Mode: Direct encode" in result.stdout


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


def test_encode_prompts_separately_for_successful_mkv_replacements(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mp4"
    reroute_dir = tmp_path / "mediashrink_mkv_followup"
    reroute_dir.mkdir()
    source.write_bytes(b"x" * 1000)
    fake_job = EncodeJob(
        source=source,
        output=reroute_dir / "ep01.mkv",
        tmp_output=reroute_dir / ".tmp_ep01.mkv",
        crf=20,
        preset="slow",
        dry_run=False,
        skip=False,
        skip_reason=None,
        action_label="MKV REROUTE",
        batch_cohort="mkv_reroute",
    )
    fake_job.output.write_bytes(b"compressed")
    fake_result = _make_result(fake_job, input_size_bytes=1000, output_size_bytes=500)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.cli._maybe_prepare_mkv_reroute",
            return_value=([], [fake_job], None, [], ["rerouted"]),
        ),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed", return_value=True),
        patch("mediashrink.cli.typer.confirm", return_value=False) as mock_confirm,
        patch(
            "mediashrink.cli.eligible_mkv_replacement_results",
            return_value=[fake_result],
        ),
        patch("mediashrink.cli.replace_successful_mkv_results") as mock_replace,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    assert mock_confirm.call_count == 1
    mock_replace.assert_not_called()
    assert "Successful MKV follow-up outputs detected:" in result.stdout
    assert (
        "newly encoded MKV file(s) are ready to replace original .mp4 source file(s)."
        in result.stdout
    )


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


def test_run_encode_loop_writes_machine_readable_runtime_log(tmp_path: Path) -> None:
    from mediashrink.cli import _run_encode_loop
    from unittest.mock import MagicMock

    source = tmp_path / "episode.mkv"
    source.write_bytes(b"x" * 5_000_000)
    output = tmp_path / "episode_out.mkv"
    output.write_bytes(b"y" * 2_000_000)
    job = EncodeJob(
        source=source,
        output=output,
        tmp_output=tmp_path / ".tmp_episode_out.mkv",
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
        output_size_bytes=2_000_000,
        duration_seconds=20.0,
        media_duration_seconds=100.0,
    )

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task.side_effect = [0, 1]
    mock_display = MagicMock()
    mock_display.make_progress_bar.return_value = mock_progress
    runtime_log = tmp_path / "runtime.jsonl"

    with (
        patch("mediashrink.cli.encode_file", return_value=ok_result),
        patch("mediashrink.cli._record_success_calibration"),
        patch("mediashrink.cli._record_batch_bias_calibration"),
    ):
        _run_encode_loop(
            [job],
            FFMPEG,
            FFPROBE,
            mock_display,
            use_calibration=False,
            runtime_log_path=runtime_log,
        )

    lines = [json.loads(line) for line in runtime_log.read_text(encoding="utf-8").splitlines()]
    events = [entry["event"] for entry in lines]
    assert "batch_started" in events
    assert "file_started" in events
    assert "file_finished" in events
    assert "batch_finished" in events


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
    assert overall_updates[-1] == 5_000_000
    assert any(update < 5_000_000 for update in overall_updates)
    mock_progress.remove_task.assert_called_once_with(1)
    mock_display.show_summary.assert_called_once_with(
        [ok_result],
        resumed_from_session=False,
        previously_completed=0,
        previously_skipped=0,
    )


def test_run_encode_loop_uses_overall_processed_label(tmp_path: Path) -> None:
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

    with patch("mediashrink.cli.encode_file", return_value=ok_result):
        _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    assert "Overall processed" in mock_progress.add_task.call_args_list[0].args[0]


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
        patch("mediashrink.cli.typer.prompt", return_value="x"),
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed") as mock_confirm_proceed,
    ):
        mock_disk_usage.return_value = shutil._ntuple_diskusage(1000, 990, 10)
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 3
    assert "Low disk space warning" in result.stdout
    mock_confirm_proceed.assert_not_called()


def test_encode_low_disk_prompt_allows_rescan_before_proceeding(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_job.estimated_output_bytes = 800
    fake_result = _make_result(fake_job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=fake_result),
        patch("mediashrink.cli._maybe_prompt_for_cleanup", return_value=[]),
        patch("mediashrink.cli.shutil.disk_usage") as mock_disk_usage,
        patch("mediashrink.progress.EncodingDisplay.confirm_proceed", return_value=True),
    ):
        calls = iter(
            [
                shutil._ntuple_diskusage(1000, 990, 10),
                shutil._ntuple_diskusage(3000, 100, 2900),
            ]
        )

        def fake_disk_usage(_path: object) -> shutil._ntuple_diskusage:
            return next(calls, shutil._ntuple_diskusage(3000, 100, 2900))

        mock_disk_usage.side_effect = fake_disk_usage
        result = runner.invoke(app, [str(tmp_path)], input="r\n")

    assert result.exit_code == 0
    assert "Disk space rescan" in result.stdout


def test_encode_reconciles_completed_sidecars_before_building_jobs(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    sidecar = tmp_path / "ep01_compressed.mkv"
    source.write_bytes(b"original")
    sidecar.write_bytes(b"compressed")
    rebuilt_job = _make_job(source)
    rebuilt_job.skip = True
    rebuilt_job.skip_reason = "video stream is already H.265/HEVC"

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch(
            "mediashrink.cli.scan_directory", side_effect=[[source, sidecar], [source]]
        ) as mock_scan,
        patch(
            "mediashrink.cli._maybe_reconcile_recoverable_sidecars",
            return_value=[source],
        ) as mock_reconcile,
        patch("mediashrink.cli.build_jobs", return_value=[rebuilt_job]),
        patch("mediashrink.cli.encode_file") as mock_encode,
    ):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 3
    assert mock_scan.call_count == 2
    mock_reconcile.assert_called_once()
    mock_encode.assert_not_called()


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


def test_maybe_prepare_mkv_reroute_interactive_accepts_and_builds_followup_jobs(
    tmp_path: Path,
) -> None:
    from mediashrink.cli import _maybe_prepare_mkv_reroute

    source = tmp_path / "ep01.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)

    with (
        patch(
            "mediashrink.cli._find_mkv_reroute_candidates",
            return_value=([job], ["ep01.mp4: unsupported container/stream combination"]),
        ),
        patch("mediashrink.cli.typer.confirm", return_value=True) as mock_confirm,
        patch(
            "mediashrink.cli._write_followup_manifest_for_jobs",
            return_value=tmp_path / "followup.json",
        ),
    ):
        remaining, rerouted, manifest_path, details, notes = _maybe_prepare_mkv_reroute(
            [job],
            ffmpeg=FFMPEG,
            ffprobe=FFPROBE,
            base_dir=tmp_path,
            output_dir=None,
            recursive=True,
            preset="slow",
            crf=20,
            duplicate_policy=None,
            assume_yes=False,
        )

    assert remaining == []
    assert len(rerouted) == 1
    assert rerouted[0].output.parent == tmp_path / "mediashrink_mkv_followup"
    assert rerouted[0].output.suffix == ".mkv"
    assert manifest_path == tmp_path / "followup.json"
    assert details == ["ep01.mp4: unsupported container/stream combination"]
    assert notes and "Rerouted 1 incompatible file(s)" in notes[0]
    mock_confirm.assert_called_once()


def test_maybe_prepare_mkv_reroute_unattended_skips_prompt(tmp_path: Path) -> None:
    from mediashrink.cli import _maybe_prepare_mkv_reroute

    source = tmp_path / "ep01.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)

    with (
        patch(
            "mediashrink.cli._find_mkv_reroute_candidates",
            return_value=([job], ["ep01.mp4: unsupported container/stream combination"]),
        ),
        patch("mediashrink.cli.typer.confirm") as mock_confirm,
        patch(
            "mediashrink.cli._write_followup_manifest_for_jobs",
            return_value=tmp_path / "followup.json",
        ),
    ):
        remaining, rerouted, manifest_path, details, _ = _maybe_prepare_mkv_reroute(
            [job],
            ffmpeg=FFMPEG,
            ffprobe=FFPROBE,
            base_dir=tmp_path,
            output_dir=None,
            recursive=True,
            preset="slow",
            crf=20,
            duplicate_policy=None,
            assume_yes=True,
        )

    assert remaining == []
    assert len(rerouted) == 1
    assert manifest_path == tmp_path / "followup.json"
    assert details == ["ep01.mp4: unsupported container/stream combination"]
    mock_confirm.assert_not_called()


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


def test_write_batch_reports_include_retry_and_queue_metadata(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    job = _make_job(tmp_path / "episode.mkv")
    job.source.write_bytes(b"x" * 1000)
    result = _make_result(job)

    json_path, text_path = _write_batch_reports(
        mode="overnight",
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
        results=[result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
        retry_mode="conservative",
        queue_strategy="safe-first",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["retry_mode"] == "conservative"
    assert payload["queue_strategy"] == "safe-first"
    assert "Retry mode: conservative" in text
    assert "Queue strategy: safe-first" in text


def test_write_batch_reports_include_estimate_context_and_excluded_files(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    job = _make_job(tmp_path / "episode.mkv")
    job.source.write_bytes(b"x" * 1000)
    result = _make_result(job)

    json_path, text_path = _write_batch_reports(
        mode="wizard",
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
        results=[result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
        estimate_context={
            "initial_scope": "6 files",
            "initial_estimated_seconds": 7200.0,
            "selected_scope_label": "5 files after preflight",
            "selected_estimated_seconds": 5400.0,
            "rebenchmarked_after_split": True,
            "original_benchmark_source": "The Sound of Music (1965).mp4",
        },
        container_fallback_actions={
            "mkv_sidecar_outputs": 0,
            "mkv_retry_failed_count": 1,
            "followup_count": 1,
            "followup_manifest": str(tmp_path / "followup.json"),
            "excluded_files": [
                {
                    "name": "The Sound of Music (1965).mp4",
                    "reason": "container/copied-stream incompatibility",
                    "next_step": "Use MKV output first.",
                }
            ],
        },
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["estimate_context"]["rebenchmarked_after_split"] is True
    assert (
        payload["container_fallback_actions"]["excluded_files"][0]["name"]
        == "The Sound of Music (1965).mp4"
    )
    assert (
        "Final selected-scope estimate was re-benchmarked after The Sound of Music (1965).mp4 left the run."
        in text
    )
    assert "Excluded files:" in text
    assert "The Sound of Music (1965).mp4: container/copied-stream incompatibility" in text


def test_write_batch_reports_include_estimate_ranges_without_crashing(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    job = _make_job(tmp_path / "episode.mkv")
    job.source.write_bytes(b"x" * 1000)
    result = _make_result(job)

    json_path, text_path = _write_batch_reports(
        mode="wizard",
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
        results=[result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
        estimate_ranges={
            "output_bytes": {"low": 400_000_000, "high": 550_000_000},
            "saved_bytes": {"low": 450_000_000, "high": 600_000_000},
            "encode_seconds": {"low": 3600.0, "high": 5400.0},
            "bias_note": "recent runs have usually saved less space than forecast",
        },
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["estimate_ranges"]["encode_seconds"]["low"] == 3600.0
    assert "Estimate ranges" in text
    assert "Encode time: ~1h 00m to 1h 30m" in text
    assert "Bias note: recent runs have usually saved less space than forecast" in text


def test_write_batch_reports_include_cohort_summaries_and_outliers(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    first = _make_job(tmp_path / "Show A - s01e01 - Pilot.mkv")
    second = _make_job(tmp_path / "Show A - s01e02 - Two.mkv")
    for job in (first, second):
        job.source.write_bytes(b"x" * 1000)
        job.estimated_output_bytes = 400_000_000
    first_result = _make_result(first, duration_seconds=600.0, media_duration_seconds=2400.0)
    second_result = _make_result(
        second,
        duration_seconds=900.0,
        media_duration_seconds=2400.0,
        output_size_bytes=500_000_000,
    )

    json_path, text_path = _write_batch_reports(
        mode="wizard",
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
        results=[first_result, second_result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["cohort_summaries"]["show"][0]["label"] == "Show A"
    assert payload["runtime_outliers"][0]["name"].endswith("Two.mkv")
    assert "Top show cohorts" in text
    assert "Runtime outliers" in text


def test_write_batch_reports_cleanup_text_distinguishes_true_mkv_sidecars(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    mp4_job = EncodeJob(
        source=tmp_path / "movie.mp4",
        output=tmp_path / "mediashrink_mkv_followup" / "movie.mkv",
        tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_movie.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
        skip_reason=None,
    )
    mp4_job.output.parent.mkdir(parents=True)
    mp4_job.source.write_bytes(b"x" * 1000)
    mp4_job.output.write_bytes(b"y" * 500)
    result = _make_result(mp4_job, input_size_bytes=1000, output_size_bytes=500)

    _, text_path = _write_batch_reports(
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
        results=[result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
    )

    text = text_path.read_text(encoding="utf-8")
    assert "Cleanup: MKV sidecar output kept alongside the original source" in text


def test_write_batch_reports_compacts_repeated_cleanup_text(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    first_job = _make_job(tmp_path / "episode01.mkv")
    second_job = _make_job(tmp_path / "episode02.mkv")
    for job in (first_job, second_job):
        job.source.write_bytes(b"x" * 1000)
        job.output.write_bytes(b"y" * 500)
    first_result = _make_result(first_job, input_size_bytes=1000, output_size_bytes=500)
    second_result = _make_result(second_job, input_size_bytes=1000, output_size_bytes=500)

    _, text_path = _write_batch_reports(
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
        results=[first_result, second_result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
    )

    text = text_path.read_text(encoding="utf-8")
    assert "Cleanup summary" in text
    assert text.count("Cleanup: Side-by-side output kept") == 0


def test_write_batch_reports_include_launch_mode_cleanup_policy_and_reroute_metadata(
    tmp_path: Path,
) -> None:
    from mediashrink.cli import _write_batch_reports

    primary_job = _make_job(tmp_path / "episode01.mkv")
    primary_job.source.write_bytes(b"x" * 1000)
    primary_result = _make_result(primary_job)

    reroute_job = EncodeJob(
        source=tmp_path / "episode02.mp4",
        output=tmp_path / "mediashrink_mkv_followup" / "episode02.mkv",
        tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_episode02.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
        skip_reason=None,
        action_label="MKV REROUTE",
        batch_cohort="mkv_reroute",
    )
    reroute_job.output.parent.mkdir(parents=True)
    reroute_job.source.write_bytes(b"x" * 1000)
    reroute_job.output.write_bytes(b"y" * 500)
    reroute_result = _make_result(reroute_job, input_size_bytes=1000, output_size_bytes=500)

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
        results=[primary_result, reroute_result],
        cleaned_paths=[],
        log_path=None,
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
        launch_mode="direct_encode",
        cleanup_policy="keep_originals",
        batch_jobs=[primary_job, reroute_job],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["launch_mode"] == "direct_encode"
    assert payload["cleanup_policy"] == "keep_originals"
    assert payload["batch_cohorts"]["primary_count"] == 1
    assert payload["batch_cohorts"]["rerouted_count"] == 1
    assert payload["files"][1]["action_label"] == "MKV REROUTE"
    assert payload["files"][1]["batch_cohort"] == "mkv_reroute"
    assert "Cleanup policy: keep_originals" in text
    assert "Launch mode: direct_encode" in text
    assert "Batch cohorts: 1 primary, 1 rerouted" in text


def test_write_batch_reports_marks_when_original_is_replaced_with_mkv(tmp_path: Path) -> None:
    from mediashrink.cli import _write_batch_reports

    reroute_job = EncodeJob(
        source=tmp_path / "episode02.mp4",
        output=tmp_path / "mediashrink_mkv_followup" / "episode02.mkv",
        tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_episode02.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
        skip_reason=None,
        action_label="MKV REROUTE",
        batch_cohort="mkv_reroute",
    )
    reroute_job.output.parent.mkdir(parents=True)
    reroute_job.source.write_bytes(b"x" * 1000)
    reroute_job.output.write_bytes(b"y" * 500)
    reroute_result = _make_result(reroute_job, input_size_bytes=1000, output_size_bytes=500)

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
        results=[reroute_result],
        cleaned_paths=[],
        log_path=None,
        mkv_replaced_paths={reroute_job.source: tmp_path / "episode02.mkv"},
        warnings=[],
        policy="highest-confidence",
        on_file_failure="skip",
        batch_jobs=[reroute_job],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert payload["mkv_replacements_completed"] == 1
    assert payload["files"][0]["output"].endswith("episode02.mkv")
    assert payload["files"][0]["cleanup_result"].startswith("original .mp4 removed")
    assert "MKV replacements completed: 1" in text


def test_safe_first_queue_prioritizes_lower_risk_jobs(tmp_path: Path) -> None:
    from mediashrink.cli import _prioritize_jobs

    safe = _make_job(tmp_path / "safe.mkv")
    safe.source.write_bytes(b"x" * 1000)
    mp4_job = _make_job(tmp_path / "container.mp4")
    mp4_job.source.write_bytes(b"x" * 1000)
    hardware_job = _make_job(tmp_path / "hardware.mkv")
    hardware_job.source.write_bytes(b"x" * 1000)
    mp4_job.output = mp4_job.source.with_stem(mp4_job.source.stem + "_compressed")
    hardware_job.preset = "amf"

    ordered = _prioritize_jobs([mp4_job, hardware_job, safe], "safe-first")

    assert [job.source.name for job in ordered] == ["safe.mkv", "hardware.mkv", "container.mp4"]


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


def test_run_encode_loop_uses_output_growth_as_visible_fallback_progress(tmp_path: Path) -> None:
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
        duration_seconds=0.05,
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
        time.sleep(0.01)
        job.tmp_output.write_bytes(b"x" * 250_000)
        time.sleep(0.02)
        job.tmp_output.write_bytes(b"x" * 750_000)
        time.sleep(0.02)
        return ok_result

    with (
        patch("mediashrink.cli.encode_file", side_effect=fake_encode_file),
        patch("mediashrink.cli.STALL_WARNING_SECONDS", 0.5),
        patch("mediashrink.cli.STALL_POLL_SECONDS", 0.005),
    ):
        _run_encode_loop([job], FFMPEG, FFPROBE, mock_display)

    fallback_updates = [
        kwargs["completed"]
        for args, kwargs in mock_progress.update.call_args_list
        if args
        and args[0] in {0, 1}
        and "completed" in kwargs
        and 0 < kwargs["completed"] < job.source.stat().st_size
    ]
    assert fallback_updates


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
    assert mock_loop.call_args.kwargs["retry_mode"] == "conservative"


def test_encode_overnight_flag_uses_safe_first_queue_strategy(tmp_path: Path) -> None:
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
        patch("mediashrink.cli._prioritize_jobs", return_value=[job]) as mock_prioritize,
        patch("mediashrink.cli._run_encode_loop", return_value=[ok_result]),
    ):
        result = runner.invoke(app, [str(tmp_path), "--overnight"])

    assert result.exit_code == 0
    assert mock_prioritize.call_args.args[1] == "safe-first"


def test_review_command_summarizes_latest_report(tmp_path: Path) -> None:
    report = tmp_path / "mediashrink_report_20260406_120000.json"
    report.write_text(
        json.dumps(
            {
                "mode": "overnight",
                "directory": str(tmp_path),
                "policy": "highest-confidence",
                "retry_mode": "conservative",
                "queue_strategy": "safe-first",
                "warnings": ["Subtitle warning"],
                "totals": {
                    "succeeded": 4,
                    "failed": 1,
                    "skipped_incompatible": 2,
                    "skipped_by_policy": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["review", str(tmp_path)])

    assert result.exit_code == 0
    assert "Run review" in result.stdout
    assert "retry mode: conservative" in result.stdout
    assert "Skipped incompatible files likely need MKV outputs" in result.stdout


def test_review_command_json_includes_guidance(tmp_path: Path) -> None:
    report = tmp_path / "mediashrink_report_20260406_120000.json"
    report.write_text(
        json.dumps(
            {
                "mode": "encode",
                "directory": str(tmp_path),
                "policy": "fastest-wall-clock",
                "totals": {
                    "succeeded": 1,
                    "failed": 0,
                    "skipped_incompatible": 0,
                    "skipped_by_policy": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["review", str(report), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["review_guidance"] == ["No manual follow-up is suggested from this report."]
    assert payload["review_analysis"]["guidance"] == payload["review_guidance"]


def test_review_command_share_safe_redacts_paths_and_exports_json(tmp_path: Path) -> None:
    report = tmp_path / "mediashrink_report_20260406_120000.json"
    runtime_log = tmp_path / "mediashrink_runtime_encode_20260406_120000.jsonl"
    report.write_text(
        json.dumps(
            {
                "mode": "encode",
                "directory": str(tmp_path),
                "runtime_log_path": str(runtime_log),
                "files": [
                    {
                        "source": str(tmp_path / "episode01.mkv"),
                        "output": str(tmp_path / "episode01_out.mkv"),
                    }
                ],
                "totals": {
                    "succeeded": 1,
                    "failed": 0,
                    "skipped_incompatible": 0,
                    "skipped_by_policy": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    export_path = tmp_path / "share-safe.json"

    result = runner.invoke(
        app,
        ["review", str(report), "--json", "--share-safe", "--export-json", str(export_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["directory"].startswith("<redacted>/")
    assert payload["runtime_log_path"].startswith("<redacted>/")
    assert payload["files"][0]["source"].startswith("<redacted>/")
    assert exported["review_analysis"]["guidance"] == exported["review_guidance"]


def test_review_command_mentions_followup_manifest_and_estimate_miss(tmp_path: Path) -> None:
    report = tmp_path / "mediashrink_report_20260406_120000.json"
    followup = tmp_path / "mediashrink_followup_20260406_120000.json"
    followup.write_text('{"version": 1, "items": []}', encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "mode": "encode",
                "directory": str(tmp_path),
                "policy": "highest-confidence",
                "retry_mode": "balanced",
                "queue_strategy": "original",
                "split_followup_manifest": str(followup),
                "estimate_miss_summary": "Actual output size was 20% larger than estimated across successful files.",
                "grouped_incompatibilities": [
                    {
                        "reason": "unsupported container/stream combination",
                        "count": 2,
                        "examples": ["episode-a.mp4", "episode-b.mp4"],
                    }
                ],
                "totals": {
                    "succeeded": 3,
                    "failed": 0,
                    "skipped_incompatible": 2,
                    "skipped_by_policy": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["review", str(report)])

    assert result.exit_code == 0
    assert "follow-up manifest" in result.stdout.lower()


def test_review_command_mentions_top_cohort_and_slowest_file(tmp_path: Path) -> None:
    report = tmp_path / "mediashrink_report_20260406_120000.json"
    report.write_text(
        json.dumps(
            {
                "mode": "encode",
                "directory": str(tmp_path),
                "totals": {
                    "succeeded": 2,
                    "failed": 0,
                    "skipped_incompatible": 0,
                    "skipped_by_policy": 0,
                },
                "cohort_summaries": {
                    "show": [
                        {
                            "label": "Show A",
                            "file_count": 2,
                            "saved_bytes": 1234,
                            "elapsed_seconds": 1000.0,
                        }
                    ],
                    "season": [],
                },
                "runtime_outliers": [
                    {"name": "Show A - s01e02 - Two.mkv", "duration_seconds": 900.0}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["review", str(report)])

    assert result.exit_code == 0
    assert "Top time-saving cohort in this run: Show A." in result.stdout
    assert "Slowest file in this run: Show A - s01e02 - Two.mkv." in result.stdout


def test_calibration_command_summarizes_local_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    calibration_path = tmp_path / "appdata" / "mediashrink" / "calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(
            {
                "version": 1,
                "records": [
                    {
                        "codec": "h264",
                        "container": ".mkv",
                        "resolution_bucket": "1080p",
                        "bitrate_bucket": "high",
                        "preset": "fast",
                        "preset_family": "software",
                        "crf": 20,
                        "input_bytes": 1000,
                        "output_bytes": 500,
                        "duration_seconds": 100.0,
                        "wall_seconds": 50.0,
                        "effective_speed": 2.0,
                        "fallback_used": False,
                        "retry_used": False,
                        "accepted_output": True,
                    },
                    {
                        "codec": "h264",
                        "container": ".mp4",
                        "resolution_bucket": "1080p",
                        "bitrate_bucket": "high",
                        "preset": "amf",
                        "preset_family": "hardware",
                        "crf": 22,
                        "input_bytes": 1000,
                        "output_bytes": 3000,
                        "duration_seconds": 100.0,
                        "wall_seconds": 10.0,
                        "effective_speed": 10.0,
                        "fallback_used": False,
                        "retry_used": False,
                        "accepted_output": False,
                    },
                ],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["calibration"])

    assert result.exit_code == 0
    assert "Calibration summary" in result.stdout
    assert "rejected by safety checks" in result.stdout


def test_calibration_command_emits_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    calibration_path = tmp_path / "appdata" / "mediashrink" / "calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text('{"version":1,"records":[],"failures":[]}', encoding="utf-8")

    result = runner.invoke(app, ["calibration", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["records"] == 0


def test_encode_skip_policy_writes_followup_manifest(tmp_path: Path) -> None:
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
            "mediashrink.cli._maybe_prepare_mkv_reroute",
            return_value=([job], [], None, [], []),
        ),
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
    followup = next(tmp_path.glob("mediashrink_followup_*.json"))
    report_json = next(tmp_path.glob("mediashrink_report_*.json"))
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["split_followup_manifest"] == str(followup)
    assert payload["grouped_incompatibilities"] == [
        {
            "reason": "output container cannot safely carry one or more copied streams",
            "count": 1,
            "examples": ["episode.mp4"],
        }
    ]


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
            "mediashrink.cli._maybe_prepare_mkv_reroute",
            return_value=([job], [], None, [], []),
        ),
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
            "mediashrink.cli._maybe_prepare_mkv_reroute",
            return_value=([job], [], None, [], []),
        ),
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


def test_encode_warns_about_attachment_data_and_audio_container_constraints(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 1000)
    job = _make_job(source)
    ok_result = _make_result(job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[job]),
        patch(
            "mediashrink.cli.describe_output_container_constraints",
            return_value=[
                "attachment streams will be dropped for MP4/M4V compatibility",
                "auxiliary data streams will be dropped for MP4/M4V compatibility",
                "audio copy may fail in this container for codec(s): dts",
            ],
        ),
        patch("mediashrink.cli.encode_file", return_value=ok_result),
    ):
        result = runner.invoke(app, [str(tmp_path), "--yes"])

    assert result.exit_code == 0
    assert "Attachment warning:" in result.stdout
    assert "Auxiliary data warning:" in result.stdout
    assert "Audio compatibility warning:" in result.stdout


def test_wizard_followup_rerun_preflights_before_encoding(tmp_path: Path) -> None:
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"x" * 1000)
    followup = tmp_path / "mediashrink_followup_20260407_120000.json"
    save_manifest(
        build_manifest(
            directory=tmp_path,
            recursive=True,
            preset="amf",
            crf=22,
            profile_name=None,
            estimated_total_encode_seconds=None,
            estimate_confidence=None,
            size_confidence=None,
            size_confidence_detail=None,
            time_confidence=None,
            time_confidence_detail=None,
            duplicate_policy=None,
            items=[
                AnalysisItem(
                    source=source,
                    codec="h264",
                    size_bytes=1000,
                    duration_seconds=120.0,
                    bitrate_kbps=8000.0,
                    estimated_output_bytes=500,
                    estimated_savings_bytes=500,
                    recommendation="recommended",
                    reason_code="reason",
                    reason_text="reason",
                )
            ],
        ),
        followup,
    )
    primary_job = _make_job(source)
    primary_result = _make_result(primary_job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch(
            "mediashrink.wizard.run_wizard",
            return_value=([primary_job], "encode", False, followup),
        ),
        patch("mediashrink.cli._run_encode_loop", return_value=[primary_result]) as mock_loop,
        patch(
            "mediashrink.cli.preflight_encode_job",
            return_value=_make_result(
                _make_job(source),
                success=False,
                output_size_bytes=0,
                error_message="Could not write header (incorrect codec parameters ?): Invalid argument",
            ),
        ),
    ):
        result = runner.invoke(app, ["wizard", str(tmp_path)], input="y\n")

    assert result.exit_code == 2
    assert mock_loop.call_count == 1
    assert "Follow-up preflight found remaining incompatibilities." in result.stdout
    assert "before the output header could be initialized" in result.stdout


def test_wizard_followup_skips_software_retry_prompt_for_container_only_failures(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"x" * 100)
    followup = tmp_path / "followup.json"
    save_manifest(
        build_manifest(
            directory=tmp_path,
            recursive=True,
            preset="amf",
            crf=22,
            profile_name=None,
            estimated_total_encode_seconds=None,
            estimate_confidence=None,
            size_confidence=None,
            size_confidence_detail=None,
            time_confidence=None,
            time_confidence_detail=None,
            duplicate_policy=None,
            recommended_only=False,
            notes=[
                "Automatically generated from files left out by preflight compatibility checks.",
                "1 file(s): hardware encoder startup failure (movie.mp4)",
            ],
            items=[
                AnalysisItem(
                    source=source,
                    codec="h264",
                    size_bytes=1000,
                    duration_seconds=120.0,
                    bitrate_kbps=8000.0,
                    estimated_output_bytes=500,
                    estimated_savings_bytes=500,
                    recommendation="recommended",
                    reason_code="reason",
                    reason_text="reason",
                )
            ],
        ),
        followup,
    )
    primary_job = _make_job(source)
    primary_result = _make_result(primary_job)

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch(
            "mediashrink.wizard.run_wizard",
            return_value=([primary_job], "encode", False, followup),
        ),
        patch("mediashrink.cli._run_encode_loop", return_value=[primary_result]) as mock_loop,
        patch(
            "mediashrink.cli.preflight_encode_job",
            return_value=_make_result(
                _make_job(source),
                success=False,
                output_size_bytes=0,
                error_message="Could not write header (incorrect codec parameters ?): Invalid argument",
            ),
        ),
        patch(
            "mediashrink.cli.describe_container_incompatibilities",
            return_value=["attachment stream incompatibility"],
        ),
        patch("mediashrink.cli.typer.confirm") as mock_confirm,
    ):
        result = runner.invoke(app, ["wizard", str(tmp_path)])

    assert result.exit_code == 2
    assert mock_loop.call_count == 1
    mock_confirm.assert_not_called()
    assert "attachment" in result.stdout.lower()
    assert "hardware encoder startup failure" in result.stdout.lower()
    assert "output container cannot safely carry one or more copied streams" not in result.stdout


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


def test_require_net_savings_rejects_small_savings_outputs(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    weak_result = _make_result(
        fake_job,
        input_size_bytes=1_000_000,
        output_size_bytes=900_000,
    )

    with (
        patch("mediashrink.cli.check_ffmpeg_available", return_value=(True, "")),
        patch("mediashrink.cli.find_ffmpeg", return_value=FFMPEG),
        patch("mediashrink.cli.find_ffprobe", return_value=FFPROBE),
        patch("mediashrink.cli.scan_directory", return_value=[source]),
        patch("mediashrink.cli.build_jobs", return_value=[fake_job]),
        patch("mediashrink.cli.encode_file", return_value=weak_result),
    ):
        result = runner.invoke(
            app,
            [str(tmp_path), "--yes", "--require-net-savings", "20"],
        )

    assert result.exit_code == 2
    assert "Output acceptance check" in result.stdout
