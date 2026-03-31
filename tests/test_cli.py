from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mkv_compress.analysis import build_manifest, save_manifest
from mkv_compress.cli import app
from mkv_compress.models import AnalysisItem, EncodeJob, EncodeResult
from mkv_compress.profiles import SavedProfile, upsert_profile

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
        tmp_output=source.parent / f".tmp_{source.stem}_compressed.mkv",
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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(False, "ffmpeg not found")):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 1
    assert "ffmpeg not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_no_mkv_files(tmp_path: Path) -> None:
    (tmp_path / "movie.mp4").touch()

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE):
        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    assert "No .mkv files" in result.stdout


def test_dry_run_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.dry_run = True
    fake_result = _make_result(fake_job, output_size_bytes=0)

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
        result = runner.invoke(app, [str(tmp_path), "--dry-run", "--yes"])

    assert result.exit_code == 0


def test_yes_flag_skips_prompt(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_result = _make_result(fake_job)

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file", return_value=fake_result), \
         patch("mkv_compress.progress.typer.confirm") as mock_confirm:
        result = runner.invoke(app, [str(tmp_path), "--yes"])

    mock_confirm.assert_not_called()
    assert result.exit_code == 0


def test_all_skipped_no_encoding(tmp_path: Path) -> None:
    mkv = tmp_path / "ep01.mkv"
    mkv.write_bytes(b"x" * 1000)

    fake_job = _make_job(mkv)
    fake_job.skip = True
    fake_job.skip_reason = "already HEVC"

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[mkv]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]), \
         patch("mkv_compress.cli.encode_file") as mock_encode:
        result = runner.invoke(app, [str(tmp_path)])

    mock_encode.assert_not_called()
    assert result.exit_code == 0
    assert "Nothing to encode" in result.stdout


def test_profile_option_applies_saved_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    fake_job = _make_job(source)
    fake_result = _make_result(fake_job)

    upsert_profile(SavedProfile(name="tv", preset="nvenc", crf=24, created_from_wizard=True))

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[source]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs, \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.scan_directory", return_value=[source]), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs, \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
        result = runner.invoke(
            app,
            [str(tmp_path), "--profile", "tv", "--crf", "18", "--preset", "slow", "--yes"],
        )

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["crf"] == 18
    assert mock_build_jobs.call_args.kwargs["preset"] == "slow"


def test_profiles_list_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    upsert_profile(SavedProfile(name="tv", preset="fast", crf=20, label="Balanced", created_from_wizard=True))

    list_result = runner.invoke(app, ["profiles", "list"])
    delete_result = runner.invoke(app, ["profiles", "delete", "tv"])
    list_after_delete = runner.invoke(app, ["profiles", "list"])

    assert list_result.exit_code == 0
    assert "tv: preset=fast, crf=20 - Balanced (wizard)" in list_result.stdout
    assert delete_result.exit_code == 0
    assert "Deleted profile" in delete_result.stdout
    assert "No saved profiles" in list_after_delete.stdout


def test_wizard_subcommand_registered() -> None:
    result = runner.invoke(app, ["wizard", "--help"])
    assert result.exit_code == 0
    assert "wizard" in result.stdout.lower() or "encoding" in result.stdout.lower()


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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.analyze_directory", return_value=[item]), \
         patch("mkv_compress.cli.estimate_analysis_encode_seconds", return_value=600.0):
        result = runner.invoke(app, ["analyze", str(tmp_path), "--manifest-out", str(manifest_path)])

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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.analyze_directory", return_value=[item]), \
         patch("mkv_compress.cli.estimate_analysis_encode_seconds", return_value=600.0), \
         patch("mkv_compress.cli.build_manifest") as mock_build_manifest, \
         patch("mkv_compress.cli.save_manifest") as mock_save_manifest:
        result = runner.invoke(
            app,
            ["analyze", str(tmp_path), "--profile", "tv", "--crf", "18", "--preset", "slow", "--manifest-out", str(tmp_path / "analysis.json")],
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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs, \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs, \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
        result = runner.invoke(app, ["apply", str(manifest_path), "--profile", "tv", "--crf", "18", "--preset", "slow", "--yes"])

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

    with patch("mkv_compress.cli.check_ffmpeg_available", return_value=(True, "")), \
         patch("mkv_compress.cli.find_ffmpeg", return_value=FFMPEG), \
         patch("mkv_compress.cli.find_ffprobe", return_value=FFPROBE), \
         patch("mkv_compress.cli.build_jobs", return_value=[fake_job]) as mock_build_jobs, \
         patch("mkv_compress.cli.encode_file", return_value=fake_result):
        result = runner.invoke(app, ["apply", str(manifest_path), "--yes"])

    assert result.exit_code == 0
    assert mock_build_jobs.call_args.kwargs["files"] == [existing]
    assert "Missing file from manifest" in result.stdout
