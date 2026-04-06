from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from mediashrink.models import AnalysisItem, EncodeJob, EncodeResult
from mediashrink.wizard import (
    _sum_media_durations,
    EncoderProfile,
    benchmark_encoder,
    build_profiles,
    detect_available_encoders,
    display_profiles_table,
    maybe_save_profile,
    review_maybe_items,
    run_wizard,
    run_custom_wizard,
)

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def test_detect_available_encoders_returns_available_only() -> None:
    console = Console()

    def fake_probe(key: str, ffmpeg: Path) -> bool:
        return key == "qsv"

    with patch("mediashrink.wizard.probe_encoder_available", side_effect=fake_probe):
        result = detect_available_encoders(FFMPEG, console)

    assert result == ["qsv"]


def test_detect_available_encoders_stable_order_even_if_completion_varies() -> None:
    console = Console()

    def fake_probe(key: str, ffmpeg: Path) -> bool:
        return key in {"nvenc", "qsv"}

    with patch("mediashrink.wizard.probe_encoder_available", side_effect=fake_probe):
        result = detect_available_encoders(FFMPEG, console)

    assert result == ["qsv", "nvenc"]


def test_benchmark_encoder_returns_speed_on_success(tmp_path: Path) -> None:
    sample = tmp_path / "sample.mkv"
    sample.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("mediashrink.wizard.subprocess.run", return_value=mock_result),
        patch("mediashrink.wizard.time.monotonic", side_effect=[0.0, 4.0]),
    ):
        speed = benchmark_encoder("fast", sample, 40.0, 20, FFMPEG)

    assert speed == pytest.approx(2.0, rel=0.01)


def test_benchmark_encoder_hardware_key(tmp_path: Path) -> None:
    sample = tmp_path / "sample.mkv"
    sample.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("mediashrink.wizard.subprocess.run", return_value=mock_result) as mock_run,
        patch("mediashrink.wizard.time.monotonic", side_effect=[0.0, 2.0]),
    ):
        benchmark_encoder("qsv", sample, 40.0, 20, FFMPEG)

    cmd = mock_run.call_args[0][0]
    assert "hevc_qsv" in cmd
    assert "-global_quality" in cmd


def test_sum_media_durations_uses_average_fallback(tmp_path: Path) -> None:
    files = [tmp_path / "a.mkv", tmp_path / "b.mkv", tmp_path / "c.mkv"]
    for file in files:
        file.write_bytes(b"x")

    with patch("mediashrink.wizard.get_duration_seconds", side_effect=[100.0, 0.0, 200.0]):
        total = _sum_media_durations(files, FFPROBE)

    assert total == pytest.approx(450.0)


def test_build_profiles_recommends_fastest_hardware() -> None:
    profiles = build_profiles(
        available_hw=["qsv", "nvenc"],
        benchmark_speeds={"qsv": 5.0, "nvenc": 8.0, "fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    recommended = [profile for profile in profiles if profile.is_recommended]
    assert len(recommended) == 1
    assert recommended[0].encoder_key == "nvenc"


def test_build_profiles_renames_slower_hardware_and_recommends_software() -> None:
    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 2.0, "fast": 0.8, "faster": 3.0},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    gpu_profile = next(profile for profile in profiles if profile.encoder_key == "amf")
    recommended = next(profile for profile in profiles if profile.is_recommended)

    assert gpu_profile.name == "Fastest GPU encode"
    assert recommended.name == "Fast"


def test_build_profiles_does_not_recommend_dominated_profile() -> None:
    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 2.0, "fast": 0.8, "faster": 3.0},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    gpu_profile = next(profile for profile in profiles if profile.encoder_key == "amf")
    assert gpu_profile.is_recommended is False


def test_build_profiles_recommends_balanced_without_hw() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    recommended = [profile for profile in profiles if profile.is_recommended]
    assert len(recommended) == 1
    assert recommended[0].name == "Balanced"


def test_build_profiles_estimates_decrease_with_higher_crf() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    balanced = next(profile for profile in profiles if profile.name == "Balanced")
    smallest = next(profile for profile in profiles if profile.name == "Smallest")
    assert smallest.estimated_output_bytes < balanced.estimated_output_bytes


def test_display_profiles_table_uses_device_label() -> None:
    console = Console(record=True, width=140)
    profiles = build_profiles(
        available_hw=["qsv"],
        benchmark_speeds={"qsv": 5.0, "fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    display_profiles_table(profiles, 10 * 1024**3, 3, {"qsv": "Intel Arc Test"}, console)

    output = console.export_text()
    assert "Intel Quick" in output
    assert "Sync" in output
    assert "Intel" in output
    assert "Arc" in output
    assert "Test" in output
    assert "approximate estimates" in output
    assert "Estimate confidence:" not in output


def test_display_profiles_table_shows_fastest_and_default_guidance() -> None:
    console = Console(record=True, width=160)
    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 2.0, "fast": 0.8, "faster": 3.0},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    display_profiles_table(profiles, 10 * 1024**3, 3, {"amf": "AMD Test"}, console)

    output = console.export_text()
    assert "Why choose this" in output
    assert "Intent" in output
    assert "Lowest estimated wait: Fast" in output
    assert "Default pick: Fast" in output


def test_run_custom_wizard_returns_hardware_choice() -> None:
    console = Console()

    with patch("mediashrink.wizard.typer.prompt", side_effect=["1", "21"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "qsv"
    assert crf == 21
    assert sw_preset is None


def test_run_custom_wizard_returns_software_preset() -> None:
    console = Console()

    with patch("mediashrink.wizard.typer.prompt", side_effect=["2", "22", "4"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "medium"
    assert crf == 22
    assert sw_preset == "medium"


def test_maybe_save_profile_persists_choice(tmp_path: Path, monkeypatch) -> None:
    console = Console()
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    with (
        patch("mediashrink.wizard.typer.confirm", return_value=True),
        patch("mediashrink.wizard.typer.prompt", return_value="tv-batch"),
    ):
        maybe_save_profile("slow", 18, "Best Quality", console)

    from mediashrink.profiles import get_profile

    profile = get_profile("tv-batch")
    assert profile is not None
    assert profile.preset == "slow"
    assert profile.crf == 18
    assert profile.label == "Best Quality"


def test_review_maybe_items_defaults_to_not_include(tmp_path: Path) -> None:
    console = Console()
    item = _analysis_item(tmp_path / "maybe.mkv", "maybe")

    with patch("mediashrink.wizard.typer.confirm", return_value=False):
        included = review_maybe_items([item], console)

    assert included is False


def test_run_wizard_analyzes_and_builds_jobs_for_recommended_only(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    maybe = _analysis_item(tmp_path / "ep02.mkv", "maybe")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch(
            "mediashrink.wizard._run_analysis_with_progress", return_value=[recommended, maybe]
        ) as mock_analyze,
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table") as mock_profiles_table,
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
        patch("mediashrink.wizard.typer.confirm", side_effect=[True, True]),
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, True, None, False, False, console)

    assert action == "encode"
    assert jobs == [fake_job]
    mock_analyze.assert_called_once()
    assert mock_build_jobs.call_args.kwargs["files"] == [source]
    assert mock_profiles_table.call_args.args[1] == 4 * 1024**3
    assert mock_profiles_table.call_args.args[2] == 2


def test_run_wizard_can_export_manifest_and_exit(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )
    manifest_path = tmp_path / "analysis.json"

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="export"),
        patch("mediashrink.wizard.typer.prompt", return_value=str(manifest_path)),
        patch("mediashrink.wizard.save_manifest") as mock_save_manifest,
        patch("mediashrink.wizard.build_jobs") as mock_build_jobs,
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    assert action == "export"
    assert jobs == []
    mock_save_manifest.assert_called_once()
    mock_build_jobs.assert_not_called()


def test_run_wizard_aborts_cleanly_when_no_recommended_files(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    maybe = _analysis_item(source, "maybe")
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[maybe]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.build_jobs") as mock_build_jobs,
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    assert action == "cancel"
    assert jobs == []
    mock_build_jobs.assert_not_called()


def test_run_wizard_can_include_maybe_files_when_requested(tmp_path: Path) -> None:
    console = Console()
    recommended_path = tmp_path / "ep01.mkv"
    maybe_path = tmp_path / "ep02.mkv"
    recommended_path.write_bytes(b"x" * 1000)
    maybe_path.write_bytes(b"x" * 1000)
    recommended = _analysis_item(recommended_path, "recommended")
    maybe = _analysis_item(maybe_path, "maybe")
    fake_job = _job_for(recommended_path)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[recommended_path, maybe_path]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended, maybe]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="review_maybe"),
        patch("mediashrink.wizard.review_maybe_items", return_value=True),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(recommended_path, success=True),
        ),
        patch("mediashrink.wizard.typer.confirm", side_effect=[True, True]),
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    assert action == "encode"
    assert jobs == [fake_job]
    assert mock_build_jobs.call_args.kwargs["files"] == [recommended_path, maybe_path]


def test_run_wizard_prints_sample_profile_and_cleanup_guidance(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    smaller = tmp_path / "small.mkv"
    larger = tmp_path / "large.mkv"
    smaller.write_bytes(b"x" * 1000)
    larger.write_bytes(b"x" * 2000)
    recommended = AnalysisItem(
        source=larger,
        codec="h264",
        size_bytes=2 * 1024**3,
        duration_seconds=120.0,
        bitrate_kbps=12000.0,
        estimated_output_bytes=800 * 1024**2,
        estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
        recommendation="recommended",
        reason_code="reason",
        reason_text="reason text",
    )
    fake_job = _job_for(larger)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )
    preview_result = _fake_preview_result(larger)
    preview_result.job.output.write_bytes(b"preview")

    with (
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.scan_directory", return_value=[smaller, larger]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(larger, success=True),
        ),
        patch("mediashrink.wizard.typer.confirm", side_effect=[False, False]) as mock_confirm,
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    output = console.export_text()

    assert action == "cancel"
    assert jobs == []
    assert "Selected profile:" in output
    assert "exclude files already expected to be skipped" in output
    confirm_prompts = [call.args[0] for call in mock_confirm.call_args_list]
    assert "  Delete originals only after successful side-by-side encodes?" in confirm_prompts


# ---------------------------------------------------------------------------
# Built-in profiles in wizard
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = {"Fast Batch", "Archival", "GPU Offload", "Smallest Acceptable"}


def test_build_profiles_includes_builtins() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    names = {p.name for p in profiles}
    assert _BUILTIN_NAMES.issubset(names)


def test_builtin_profiles_have_is_builtin_flag() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    builtin_profiles = [p for p in profiles if p.is_builtin]
    assert len(builtin_profiles) == 4
    assert all(p.is_builtin for p in builtin_profiles)


def test_builtin_fast_gpu_substitutes_sw_when_no_hw() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3, "faster": 0.5},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    gpu_profile = next(p for p in profiles if p.name == "GPU Offload")
    # No HW available — should fall back to software preset
    assert gpu_profile.encoder_key not in {"qsv", "nvenc", "amf"}
    assert gpu_profile.sw_preset is not None


def test_builtin_fast_gpu_uses_best_hw_when_available() -> None:
    profiles = build_profiles(
        available_hw=["qsv", "nvenc"],
        benchmark_speeds={"qsv": 4.0, "nvenc": 8.0, "fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    gpu_profile = next(p for p in profiles if p.name == "GPU Offload")
    # Best HW is nvenc (speed 8.0)
    assert gpu_profile.encoder_key == "nvenc"


def test_build_profiles_custom_is_last() -> None:
    profiles = build_profiles(
        available_hw=[],
        benchmark_speeds={"fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )
    assert profiles[-1].is_custom


def _analysis_item(source: Path, recommendation: str) -> AnalysisItem:
    source.write_bytes(b"x")
    return AnalysisItem(
        source=source,
        codec="h264",
        size_bytes=2 * 1024**3,
        duration_seconds=120.0,
        bitrate_kbps=12000.0,
        estimated_output_bytes=800 * 1024**2,
        estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
        recommendation=recommendation,
        reason_code="reason",
        reason_text="reason text",
    )


def _job_for(source: Path) -> EncodeJob:
    return EncodeJob(
        source=source,
        output=source.with_stem(source.stem + "_compressed"),
        tmp_output=source.parent / f".tmp_{source.stem}_compressed{source.suffix}",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
        skip_reason=None,
    )


def _fake_preview_result(source: Path) -> EncodeResult:
    job = EncodeJob(
        source=source,
        output=source.parent / f"{source.stem}_preview{source.suffix}",
        tmp_output=source.parent / f".tmp_{source.stem}_preview{source.suffix}",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    return EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=500,
        duration_seconds=2.0,
    )


def _fake_encode_result(
    source: Path, *, success: bool, error_message: str | None = None
) -> EncodeResult:
    job = EncodeJob(
        source=source,
        output=source.with_stem(source.stem + "_compressed"),
        tmp_output=source.parent / f".tmp_{source.stem}_compressed{source.suffix}",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    return EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=success,
        input_size_bytes=1000,
        output_size_bytes=500 if success else 0,
        duration_seconds=2.0,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Stage 9 — --auto mode
# ---------------------------------------------------------------------------


def test_run_wizard_auto_selects_recommended_profile(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)

    # Balanced is recommended when no HW available
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch(
            "mediashrink.wizard.build_profiles", return_value=[selected_profile]
        ) as mock_build_profiles,
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action") as mock_action,
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
    ):
        jobs, action, _ = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console, auto=True
        )

    assert action == "encode"
    assert jobs == [fake_job]
    mock_action.assert_not_called()


def test_run_wizard_auto_returns_without_prompts(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.build_profiles", return_value=[selected_profile]),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
        patch("mediashrink.wizard.typer.confirm") as mock_confirm,
        patch("mediashrink.wizard.typer.prompt") as mock_prompt,
    ):
        jobs, action, _ = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console, auto=True
        )

    assert action == "encode"
    mock_confirm.assert_not_called()
    mock_prompt.assert_not_called()


def test_run_wizard_switches_to_fallback_when_preflight_encode_fails(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch(
            "mediashrink.wizard.preflight_encode_job",
            side_effect=[
                _fake_encode_result(
                    source,
                    success=False,
                    error_message="[mp4 @ 123] Could not write header\nInvalid argument",
                ),
                _fake_encode_result(source, success=True),
            ],
        ),
        patch("mediashrink.wizard.typer.confirm", side_effect=[True, True, True]),
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    output = console.export_text()
    assert action == "encode"
    assert jobs == [fake_job]
    assert "failed a short compatibility check" in output
    assert "Could not write header" in output
    assert "Retrying with" in output
    presets = [call.kwargs["preset"] for call in mock_build_jobs.call_args_list]
    assert presets == ["fast", "faster"]


def test_run_wizard_returns_to_profile_selection_when_fallback_declined(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    first_profile = EncoderProfile(
        1, "GPU offload", "Fastest", "qsv", 20, None, 0, 0.0, "Good", True
    )
    second_profile = EncoderProfile(
        2, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", False
    )
    fake_job = _job_for(source)

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=["qsv"]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch(
            "mediashrink.wizard.prompt_profile_selection",
            side_effect=[first_profile, second_profile],
        ),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]) as mock_build_jobs,
        patch(
            "mediashrink.wizard.preflight_encode_job",
            side_effect=[
                _fake_encode_result(source, success=False, error_message="Invalid argument"),
                _fake_encode_result(source, success=True),
            ],
        ),
        patch("mediashrink.wizard.typer.confirm", side_effect=[False, True, True]),
    ):
        jobs, action, _ = run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)

    assert action == "encode"
    assert jobs == [fake_job]
    presets = [call.kwargs["preset"] for call in mock_build_jobs.call_args_list]
    assert presets == ["qsv", "fast"]


# ---------------------------------------------------------------------------
# Stage 10 — UX polish regression tests
# ---------------------------------------------------------------------------


def test_next_step_menu_no_maybe_items() -> None:
    """When maybe_count=0, menu should be 1/2/3 not 1/3/4."""
    console = Console()
    output_lines: list[str] = []

    class CapturingConsole:
        def print(self, msg: str = "", **kwargs: object) -> None:
            output_lines.append(str(msg))

    # Always choose option 1 to avoid looping
    with patch("mediashrink.wizard.typer.prompt", return_value="1"):
        from mediashrink.wizard import prompt_analysis_action

        action = prompt_analysis_action(3, 0, CapturingConsole())  # type: ignore[arg-type]

    assert action == "compress_recommended"
    joined = "\n".join(output_lines)
    assert "Review maybe" not in joined
    # Options should be 1, 2, 3 (not 1, 3, 4)
    assert "2." in joined
    assert "3." in joined
    assert "4." not in joined


def test_next_step_menu_includes_counts_when_maybe_present() -> None:
    output_lines: list[str] = []

    class CapturingConsole:
        def print(self, msg: str = "", **kwargs: object) -> None:
            output_lines.append(str(msg))

    with patch("mediashrink.wizard.typer.prompt", return_value="1"):
        from mediashrink.wizard import prompt_analysis_action

        action = prompt_analysis_action(22, 5, CapturingConsole())  # type: ignore[arg-type]

    assert action == "compress_recommended"
    joined = "\n".join(output_lines)
    assert "Compress recommended only (22 file(s))" in joined
    assert "Review maybe files (5 file(s))" in joined
