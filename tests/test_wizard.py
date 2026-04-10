from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from mediashrink.models import AnalysisItem, EncodeJob, EncodeResult
from mediashrink.wizard import (
    _WizardFallbackRequested,
    _followup_manifest_notes,
    _followup_next_step_hint,
    _sum_media_durations,
    _summarize_mkv_suitable_candidates,
    _wizard_prompt,
    _wizard_readline,
    EncoderProfile,
    ProfilePlanningResult,
    benchmark_encoder,
    build_profiles,
    detect_available_encoders,
    display_profiles_table,
    maybe_save_profile,
    prepare_profile_planning,
    review_maybe_items,
    run_wizard,
    run_custom_wizard,
)

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


class _NonClosingStringIO(io.StringIO):
    def close(self) -> None:
        pass


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


def test_wizard_readline_prefers_linux_controlling_terminal() -> None:
    tty_in = _NonClosingStringIO("2\n")
    tty_out = _NonClosingStringIO()

    def fake_open(path: str, mode: str, **_: object) -> io.StringIO:
        if (path, mode) == ("/dev/tty", "r"):
            return tty_in
        if (path, mode) == ("/dev/tty", "w"):
            return tty_out
        raise AssertionError(f"Unexpected open call: {(path, mode)}")

    with (
        patch("mediashrink.wizard.detect_os", return_value="Linux"),
        patch("builtins.open", side_effect=fake_open),
    ):
        result = _wizard_readline("Select a profile: ")

    assert result == "2"
    assert tty_out.getvalue() == "Select a profile: "


def test_wizard_readline_falls_back_to_stdio_when_terminal_device_unavailable() -> None:
    stdin = _NonClosingStringIO("1\n")
    stdout = _NonClosingStringIO()

    with (
        patch("mediashrink.wizard.detect_os", return_value="Linux"),
        patch("builtins.open", side_effect=OSError("tty unavailable")),
        patch("sys.stdin", stdin),
        patch("sys.stdout", stdout),
    ):
        result = _wizard_readline("Select a profile: ")

    assert result == "1"
    assert stdout.getvalue() == "Select a profile: "


def test_wizard_prompt_echoes_accepted_answer() -> None:
    console = Console(record=True, width=120)
    from mediashrink import wizard as wizard_module

    session = wizard_module.WizardSessionState(
        console=console, directory=Path("/tmp"), output_dir=None
    )
    prior = wizard_module._ACTIVE_WIZARD_SESSION
    wizard_module._ACTIVE_WIZARD_SESSION = session
    try:
        with patch("mediashrink.wizard._wizard_readline", return_value="2"):
            result = _wizard_prompt(
                "Choose action [1-3]",
                prompt_id="next-step",
                acceptance_label="Next-step selection",
            )
    finally:
        wizard_module._ACTIVE_WIZARD_SESSION = prior

    assert result == "2"
    assert "Next-step selection:" in console.export_text()


def test_summarize_mkv_suitable_candidates_groups_container_constraints(tmp_path: Path) -> None:
    candidate = AnalysisItem(
        source=tmp_path / "movie.mp4",
        codec="h264",
        size_bytes=1_000,
        duration_seconds=100.0,
        bitrate_kbps=2_000.0,
        estimated_output_bytes=700,
        estimated_savings_bytes=300,
        recommendation="maybe",
        reason_code="borderline_candidate",
        reason_text="borderline",
    )
    candidate.source.write_bytes(b"fake")

    with patch(
        "mediashrink.wizard.describe_output_container_constraints",
        return_value=["attachment streams will be dropped for MP4/M4V compatibility"],
    ):
        count, grouped, examples = _summarize_mkv_suitable_candidates([candidate], FFPROBE)

    assert count == 1
    assert grouped == {"attachment streams need MKV output": 1}
    assert examples == ["movie.mp4"]


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


def test_build_profiles_highest_confidence_downranks_unreliable_hardware(tmp_path: Path) -> None:
    calibration_store = {
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
                "wall_seconds": 100.0,
                "effective_speed": 1.0,
                "fallback_used": False,
                "retry_used": False,
            }
        ],
        "failures": [
            {
                "encoder": "amf",
                "container": ".mkv",
                "stage": "encode",
                "reason": "invalid argument",
            },
            {
                "encoder": "amf",
                "container": ".mkv",
                "stage": "encode",
                "reason": "invalid argument",
            },
        ],
    }
    candidate = _analysis_item(tmp_path / "candidate.mkv", "recommended")

    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 10.0, "fast": 1.0, "faster": 1.1},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
        candidate_items=[candidate],
        ffprobe=FFPROBE,
        policy="highest-confidence",
        calibration_store=calibration_store,
    )

    recommended = next(profile for profile in profiles if profile.is_recommended)
    assert recommended.encoder_key in {"fast", "faster"}
    assert recommended.encoder_key != "amf"


def test_build_profiles_downranks_hardware_with_observed_probe_failures(tmp_path: Path) -> None:
    candidate = _analysis_item(tmp_path / "candidate.mp4", "recommended")

    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 10.0, "fast": 1.0, "faster": 1.1},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
        candidate_items=[candidate],
        ffprobe=FFPROBE,
        policy="fastest-wall-clock",
        observed_probe_failures={
            ("amf", 20): {candidate.source: "Could not write header"},
            ("amf", 22): {candidate.source: "Could not write header"},
        },
    )

    recommended = next(profile for profile in profiles if profile.is_recommended)
    assert recommended.encoder_key in {"fast", "faster"}
    assert recommended.encoder_key != "amf"


def test_build_profiles_downranks_highly_variable_hardware_size_estimates(tmp_path: Path) -> None:
    candidate = _analysis_item(tmp_path / "candidate.mkv", "recommended")
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "high",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 5.0,
                "effective_speed": 20.0,
                "fallback_used": False,
                "retry_used": False,
                "predicted_output_ratio": 0.3,
            },
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "high",
                "preset": "faster",
                "preset_family": "software",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 500,
                "duration_seconds": 100.0,
                "wall_seconds": 50.0,
                "effective_speed": 2.0,
                "fallback_used": False,
                "retry_used": False,
            },
        ],
        "failures": [],
    }

    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 12.0, "fast": 1.0, "faster": 1.2},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
        candidate_items=[candidate],
        ffprobe=FFPROBE,
        policy="fastest-wall-clock",
        calibration_store=calibration_store,
    )

    recommended = next(profile for profile in profiles if profile.is_recommended)
    assert recommended.encoder_key != "amf"


def test_build_profiles_partial_hardware_profile_mentions_high_variability(tmp_path: Path) -> None:
    safe_candidate = _analysis_item(tmp_path / "candidate.mkv", "recommended")
    risky_candidate = _analysis_item(tmp_path / "candidate.mp4", "recommended")
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "high",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 5.0,
                "effective_speed": 20.0,
                "fallback_used": False,
                "retry_used": False,
                "predicted_output_ratio": 0.3,
            }
        ],
        "failures": [],
    }

    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 12.0, "fast": 1.0, "faster": 1.2},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
        candidate_items=[safe_candidate, risky_candidate],
        ffprobe=FFPROBE,
        policy="fastest-wall-clock",
        calibration_store=calibration_store,
        observed_probe_failures={("amf", 22): {risky_candidate.source: "Could not write header"}},
    )

    amf_profile = next(profile for profile in profiles if profile.name == "GPU Offload")
    assert "highly variable" in amf_profile.why_choose


def test_build_profiles_adjusts_time_estimate_from_speed_error_history(tmp_path: Path) -> None:
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "1080p",
                "bitrate_bucket": "high",
                "preset": "faster",
                "preset_family": "software",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 500,
                "duration_seconds": 100.0,
                "wall_seconds": 200.0,
                "effective_speed": 0.5,
                "predicted_speed": 1.0,
                "fallback_used": False,
                "retry_used": False,
            }
        ],
        "failures": [],
    }
    candidate = _analysis_item(tmp_path / "candidate.mkv", "recommended")

    with patch("mediashrink.wizard.get_video_resolution", return_value=(1200, 1080)):
        profiles = build_profiles(
            available_hw=[],
            benchmark_speeds={"fast": 1.0, "faster": 1.0},
            total_media_seconds=120.0,
            total_input_bytes=2 * 1024**3,
            candidate_items=[candidate],
            ffprobe=FFPROBE,
            calibration_store=calibration_store,
        )

    fast = next(profile for profile in profiles if profile.name == "Fast")
    balanced = next(profile for profile in profiles if profile.name == "Balanced")
    assert fast.estimated_encode_seconds == pytest.approx(240.0)
    assert balanced.estimated_encode_seconds == pytest.approx(240.0)


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

    display_profiles_table(profiles, 10 * 1024**3, 3, 3, {"qsv": "Intel Arc Test"}, console)

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

    display_profiles_table(profiles, 10 * 1024**3, 3, 3, {"amf": "AMD Test"}, console)

    output = console.export_text()
    assert "Why choose" in output
    assert "Intent" in output
    assert "Lowest estimated wait: Fast" in output
    assert "Default pick: Fast" in output
    assert "Recommended-only default scope:" in output


def test_display_profiles_table_mentions_faster_profile_when_default_is_steadier(
    tmp_path: Path,
) -> None:
    console = Console(record=True, width=160)
    candidate = _analysis_item(tmp_path / "candidate.mkv", "recommended")
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "high",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 5.0,
                "effective_speed": 20.0,
                "fallback_used": False,
                "retry_used": False,
                "predicted_output_ratio": 0.3,
            },
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "high",
                "preset": "faster",
                "preset_family": "software",
                "crf": 22,
                "input_bytes": 1000,
                "output_bytes": 500,
                "duration_seconds": 100.0,
                "wall_seconds": 50.0,
                "effective_speed": 2.0,
                "fallback_used": False,
                "retry_used": False,
            },
        ],
        "failures": [],
    }
    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 12.0, "fast": 1.0, "faster": 1.2},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
        candidate_items=[candidate],
        ffprobe=FFPROBE,
        policy="fastest-wall-clock",
        calibration_store=calibration_store,
    )

    display_profiles_table(profiles, 10 * 1024**3, 1, 1, {"amf": "AMD Test"}, console)

    output = console.export_text()
    assert "Fastest wait is still" in output
    assert "steadier default" in output


def test_display_profiles_table_hides_more_duplicate_builtin_rows() -> None:
    console = Console(record=True, width=160)
    profiles = build_profiles(
        available_hw=["amf"],
        benchmark_speeds={"amf": 4.0, "fast": 1.0, "faster": 1.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    display_profiles_table(profiles, 10 * 1024**3, 3, 3, {"amf": "AMD Test"}, console)

    output = console.export_text()
    assert "Hidden 4 near-duplicate profile row(s)." in output


def test_run_custom_wizard_returns_hardware_choice() -> None:
    console = Console()

    with patch("mediashrink.wizard._wizard_prompt", side_effect=["1", "21"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "qsv"
    assert crf == 21
    assert sw_preset is None


def test_run_custom_wizard_returns_software_preset() -> None:
    console = Console()

    with patch("mediashrink.wizard._wizard_prompt", side_effect=["2", "22", "4"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "medium"
    assert crf == 22
    assert sw_preset == "medium"


def test_maybe_save_profile_persists_choice(tmp_path: Path, monkeypatch) -> None:
    console = Console()
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    with (
        patch("mediashrink.wizard._wizard_confirm", return_value=True),
        patch("mediashrink.wizard._wizard_prompt", return_value="tv-batch"),
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

    with patch("mediashrink.wizard._wizard_confirm", return_value=False):
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
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, True, None, False, False, console
        )

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
        patch("mediashrink.wizard._targeted_profile_probe_failures", return_value={}),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="export"),
        patch("mediashrink.wizard._wizard_prompt", return_value=str(manifest_path)),
        patch("mediashrink.wizard.save_manifest") as mock_save_manifest,
        patch("mediashrink.wizard.build_jobs") as mock_build_jobs,
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

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
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

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
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

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
        1,
        "Balanced",
        "Balanced",
        "fast",
        20,
        "fast",
        0,
        0.0,
        "Excellent",
        True,
        why_choose="Best default from the current time, size, quality, and compatibility estimates. Covers 1 file(s).",
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
        patch("mediashrink.wizard._wizard_confirm", side_effect=[False, False]) as mock_confirm,
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()

    assert action == "cancel"
    assert jobs == []
    assert "Selected profile:" in output
    assert "Why choose this for likely encode candidates:" in output
    assert "exclude files already expected to be skipped" in output


def test_run_wizard_cleanup_prompt_has_no_leading_indent(tmp_path: Path) -> None:
    console = Console()
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    def fake_confirm(text: str, **kwargs: object) -> bool:
        if kwargs.get("prompt_id") == "cleanup-after":
            assert not text.startswith(" ")
        return False

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
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
        patch("mediashrink.wizard._wizard_confirm", side_effect=fake_confirm),
    ):
        run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console)


def test_run_wizard_ready_summary_reframes_profile_for_selected_subset(tmp_path: Path) -> None:
    console = Console(record=True, width=160)
    recommended_path = tmp_path / "recommended.mkv"
    maybe_path = tmp_path / "maybe.mp4"
    recommended_path.write_bytes(b"x" * 1000)
    maybe_path.write_bytes(b"x" * 1000)
    recommended = _analysis_item(recommended_path, "recommended")
    maybe = _analysis_item(maybe_path, "maybe")
    fake_job = _job_for(recommended_path)
    selected_profile = EncoderProfile(
        1,
        "GPU offload",
        "GPU Offload",
        "amf",
        22,
        None,
        0,
        300.0,
        "Good",
        True,
        why_choose="Fastest partial-batch option: 1 file(s) can run now, while 1 likely need follow-up.",
        compatible_count=1,
        incompatible_count=1,
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[recommended_path, maybe_path]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended, maybe]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=["amf"]),
        patch("mediashrink.wizard.detect_device_labels", return_value={"amf": "AMD Test"}),
        patch("mediashrink.wizard.prepare_profile_planning") as mock_planning,
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch("mediashrink.wizard._run_preflight_checks", return_value=([fake_job], [])),
        patch("mediashrink.wizard.estimate_analysis_encode_seconds", return_value=180.0),
        patch("mediashrink.wizard._estimate_selected_output_bytes", return_value=500),
        patch("mediashrink.wizard._wizard_confirm", side_effect=[False, False]),
    ):
        mock_planning.return_value = ProfilePlanningResult(
            candidate_items=[recommended, maybe],
            candidate_input_bytes=recommended.size_bytes + maybe.size_bytes,
            candidate_media_seconds=recommended.duration_seconds + maybe.duration_seconds,
            sample_item=recommended,
            sample_duration=recommended.duration_seconds,
            preview_items=[recommended],
            available_hw=["amf"],
            benchmark_speeds={"amf": 3.0},
            observed_probe_failures={},
            profiles=[selected_profile],
            active_calibration={"version": 1, "records": [], "failures": []},
            size_error_by_preset={"amf": None},
            stage_messages=[],
        )
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "cancel"
    assert jobs == []
    assert "Why choose this for likely encode candidates:" in output
    assert (
        "Selected run scope: recommended files only; this profile currently looks compatible for all 1 selected file(s)."
        in output
    )
    assert "Not in this run: 1 maybe file(s) were left out by choice." in output


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
        jobs, action, _, _fm = run_wizard(
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
        patch("mediashrink.wizard._wizard_confirm") as mock_confirm,
        patch("mediashrink.wizard._wizard_prompt") as mock_prompt,
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console, auto=True
        )

    assert action == "encode"
    mock_confirm.assert_not_called()
    mock_prompt.assert_not_called()


def test_run_wizard_non_interactive_wizard_returns_without_prompts(tmp_path: Path) -> None:
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
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.build_profiles", return_value=[selected_profile]),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
        patch("mediashrink.wizard._wizard_confirm") as mock_confirm,
        patch("mediashrink.wizard._wizard_prompt") as mock_prompt,
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path,
            FFMPEG,
            FFPROBE,
            False,
            None,
            False,
            False,
            console,
            non_interactive_wizard=True,
        )

    assert action == "encode"
    assert jobs == [fake_job]
    mock_confirm.assert_not_called()
    mock_prompt.assert_not_called()


def test_run_wizard_auto_falls_back_after_prompt_failure(tmp_path: Path) -> None:
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
        patch("mediashrink.wizard.display_profiles_table"),
        patch(
            "mediashrink.wizard.prompt_profile_selection",
            side_effect=[_WizardFallbackRequested("prompt glitch")],
        ),
        patch("mediashrink.wizard.build_profiles", return_value=[selected_profile]),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "encode"
    assert jobs == [fake_job]
    assert "Detected unreliable terminal input." in output
    assert "Non-interactive mode:" in output


def test_run_wizard_writes_debug_session_log(tmp_path: Path) -> None:
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
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.build_profiles", return_value=[selected_profile]),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
    ):
        run_wizard(
            tmp_path,
            FFMPEG,
            FFPROBE,
            False,
            None,
            False,
            False,
            console,
            non_interactive_wizard=True,
            debug_session_log=True,
        )

    output = console.export_text()
    assert "Wizard debug log:" in output
    debug_logs = sorted(tmp_path.glob("mediashrink_wizard_debug_*.log"))
    assert debug_logs
    content = debug_logs[-1].read_text(encoding="utf-8")
    assert "mode=non-interactive" in content
    assert "events:" in content


def test_display_profiles_table_uses_block_layout_on_narrow_terminal() -> None:
    console = Console(record=True, width=90)
    profiles = [
        EncoderProfile(
            1,
            "Fast",
            "Fast",
            "faster",
            22,
            "faster",
            2 * 1024**3,
            3600.0,
            "Very good",
            True,
            why_choose="Best default.",
            compatible_count=9,
            incompatible_count=3,
            grouped_incompatibilities={"output header failure": 3},
        )
    ]

    display_profiles_table(
        profiles,
        total_input_bytes=4 * 1024**3,
        candidate_count=12,
        recommended_count=10,
        device_labels={},
        console=console,
    )

    output = console.export_text()
    assert "1. Fast" in output
    assert "Fit:" in output
    assert "Recommended-only:" in output
    assert "Why:" in output


def test_run_wizard_prints_hardware_before_benchmark_progress(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "GPU offload", "GPU Offload", "amf", 22, None, 0, 0.0, "Good", True
    )
    planning = ProfilePlanningResult(
        candidate_items=[recommended],
        candidate_input_bytes=recommended.size_bytes,
        candidate_media_seconds=recommended.duration_seconds,
        sample_item=recommended,
        sample_duration=recommended.duration_seconds,
        preview_items=[recommended],
        available_hw=["amf"],
        benchmark_speeds={"amf": 1.0, "fast": 0.8, "faster": 1.2},
        observed_probe_failures={},
        profiles=[selected_profile],
        active_calibration=None,
        size_error_by_preset={},
        stage_messages=[],
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=["amf"]),
        patch("mediashrink.wizard.prepare_profile_planning", return_value=planning),
        patch("mediashrink.wizard.detect_device_labels", return_value={"amf": "AMD Test"}),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
    ):
        run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console, auto=True)

    output = console.export_text()
    assert output.index("Hardware encoders available:") < output.index("Next: benchmark")


def test_run_wizard_prints_benchmark_and_probe_stage_summaries(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    fake_job = _job_for(source)
    selected_profile = EncoderProfile(
        1, "GPU offload", "GPU Offload", "amf", 22, None, 0, 0.0, "Good", True
    )
    planning = ProfilePlanningResult(
        candidate_items=[recommended],
        candidate_input_bytes=recommended.size_bytes,
        candidate_media_seconds=recommended.duration_seconds,
        sample_item=recommended,
        sample_duration=recommended.duration_seconds,
        preview_items=[recommended],
        available_hw=["amf"],
        benchmark_speeds={"amf": 1.0, "fast": 0.8, "faster": 1.2},
        observed_probe_failures={},
        profiles=[selected_profile],
        active_calibration=None,
        size_error_by_preset={},
        stage_messages=[],
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[source]),
        patch("mediashrink.wizard._run_analysis_with_progress", return_value=[recommended]),
        patch("mediashrink.wizard.detect_available_encoders", return_value=["amf"]),
        patch("mediashrink.wizard.prepare_profile_planning", return_value=planning),
        patch("mediashrink.wizard.detect_device_labels", return_value={"amf": "AMD Test"}),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.build_jobs", return_value=[fake_job]),
        patch(
            "mediashrink.wizard.preflight_encode_job",
            return_value=_fake_encode_result(source, success=True),
        ),
    ):
        run_wizard(tmp_path, FFMPEG, FFPROBE, False, None, False, False, console, auto=True)

    output = console.export_text()
    assert "Benchmarked 3 profile candidate(s)." in output
    assert "Smoke-probed 1 risky profile combination(s)." in output


def test_prepare_profile_planning_tracks_post_100_stage_messages(tmp_path: Path) -> None:
    source = tmp_path / "ep01.mkv"
    source.write_bytes(b"x" * 1000)
    recommended = _analysis_item(source, "recommended")
    provisional = [
        EncoderProfile(1, "GPU offload", "GPU Offload", "amf", 22, None, 0, 10.0, "Good", False)
    ]
    final = [
        EncoderProfile(1, "GPU offload", "GPU Offload", "amf", 22, None, 0, 10.0, "Good", True)
    ]

    with (
        patch("mediashrink.wizard.benchmark_encoder", side_effect=[1.0, 0.8, 1.2]),
        patch("mediashrink.wizard.build_profiles", side_effect=[provisional, final]),
        patch("mediashrink.wizard._targeted_profile_probe_failures", return_value={}),
    ):
        planning = prepare_profile_planning(
            analysis_items=[recommended],
            ffmpeg=FFMPEG,
            ffprobe=FFPROBE,
            console=None,
            available_hw=["amf"],
        )

    assert planning is not None
    assert "Building provisional profiles..." in planning.stage_messages
    assert "Preparing smoke probes..." in planning.stage_messages
    assert "Scoring recommendations..." in planning.stage_messages
    assert "Preparing profile table..." in planning.stage_messages


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
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "encode"
    assert jobs == [fake_job]
    assert "failed a short compatibility check" in output
    assert "output header failure" in output
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
        patch("mediashrink.wizard._wizard_confirm", side_effect=[False, True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    assert action == "encode"
    assert jobs == [fake_job]
    presets = [call.kwargs["preset"] for call in mock_build_jobs.call_args_list]
    assert presets == ["qsv", "fast"]


def test_run_wizard_can_skip_incompatible_files_and_continue(tmp_path: Path) -> None:
    console = Console(record=True, width=160)
    mp4_source = tmp_path / "movie.mp4"
    mkv_source = tmp_path / "movie.mkv"
    mp4_source.write_bytes(b"x" * 1000)
    mkv_source.write_bytes(b"x" * 1000)
    recommended_mp4 = _analysis_item(mp4_source, "recommended")
    recommended_mkv = _analysis_item(mkv_source, "recommended")
    mp4_job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "movie_compressed.mp4",
        tmp_output=tmp_path / ".tmp_movie_compressed.mp4",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    mkv_job = _job_for(mkv_source)
    mp4_mkv_job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "mediashrink_mkv_followup" / "movie.mkv",
        tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_movie.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[mp4_source, mkv_source]),
        patch(
            "mediashrink.wizard._run_analysis_with_progress",
            return_value=[recommended_mp4, recommended_mkv],
        ),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", return_value=1.0),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[mp4_job, mkv_job]),
        patch("mediashrink.wizard._build_mkv_followup_jobs", return_value=[mp4_mkv_job]),
        patch(
            "mediashrink.wizard._run_preflight_checks",
            side_effect=[
                (
                    [mkv_job],
                    [
                        (
                            mp4_job,
                            _fake_encode_result(
                                mp4_source, success=False, error_message="Invalid argument"
                            ),
                        )
                    ],
                ),
                ([mp4_mkv_job], []),
            ],
        ),
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "encode"
    assert jobs == [mkv_job, mp4_mkv_job]
    assert "1 file(s) can run now with Balanced." in output
    assert "switched to MKV sidecar output" in output
    assert "Predicted compatibility for this selection:" in output


def test_followup_guidance_prefers_mkv_for_software_container_failures(tmp_path: Path) -> None:
    mp4_source = tmp_path / "movie.mp4"
    mp4_source.write_bytes(b"x" * 1000)
    recommended_mp4 = _analysis_item(mp4_source, "recommended")
    job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "movie_compressed.mp4",
        tmp_output=tmp_path / ".tmp_movie_compressed.mp4",
        crf=22,
        preset="faster",
        dry_run=False,
        skip=False,
    )

    with (
        patch(
            "mediashrink.wizard.describe_container_incompatibilities",
            return_value=["unsupported copied audio codec: dts"],
        ),
        patch("mediashrink.wizard.describe_output_container_constraints", return_value=[]),
    ):
        notes = _followup_manifest_notes(
            {"output header failure": [job]},
            [recommended_mp4],
            ffprobe=FFPROBE,
        )

    hint = _followup_next_step_hint(
        preset="faster",
        grouped_failures={"output header failure": [job]},
    )

    assert "same folder with MKV output first" in hint
    assert any("unsupported copied audio codec: dts" in note for note in notes)


def test_run_wizard_rebenchmarks_when_original_sample_moves_to_followup(tmp_path: Path) -> None:
    console = Console(record=True, width=160)
    mp4_source = tmp_path / "movie.mp4"
    mkv_source = tmp_path / "episode.mkv"
    mp4_source.write_bytes(b"x" * 1000)
    mkv_source.write_bytes(b"x" * 1000)
    recommended_mp4 = _analysis_item(mp4_source, "recommended")
    recommended_mkv = _analysis_item(mkv_source, "recommended")
    mp4_job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "movie_compressed.mp4",
        tmp_output=tmp_path / ".tmp_movie_compressed.mp4",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    mkv_job = _job_for(mkv_source)
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[mp4_source, mkv_source]),
        patch(
            "mediashrink.wizard._run_analysis_with_progress",
            return_value=[recommended_mp4, recommended_mkv],
        ),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", side_effect=[1.0, 1.0, 0.8]),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[mp4_job, mkv_job]),
        patch("mediashrink.wizard._build_mkv_followup_jobs", return_value=[]),
        patch(
            "mediashrink.wizard._run_preflight_checks",
            return_value=(
                [mkv_job],
                [
                    (
                        mp4_job,
                        _fake_encode_result(
                            mp4_source,
                            success=False,
                            error_message="[mp4 @ 123] Could not write header\nInvalid argument",
                        ),
                    )
                ],
            ),
        ),
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "encode"
    assert any(job.source == mkv_source for job in jobs)
    assert (
        "Time estimate was re-benchmarked after the original sample file moved to follow-up."
        in output
    )


def test_run_wizard_reports_failed_mkv_retry_and_single_followup_name(tmp_path: Path) -> None:
    console = Console(record=True, width=160)
    mp4_source = tmp_path / "movie.mp4"
    mkv_source = tmp_path / "episode.mkv"
    mp4_source.write_bytes(b"x" * 1000)
    mkv_source.write_bytes(b"x" * 1000)
    recommended_mp4 = _analysis_item(mp4_source, "recommended")
    recommended_mkv = _analysis_item(mkv_source, "recommended")
    mp4_job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "movie_compressed.mp4",
        tmp_output=tmp_path / ".tmp_movie_compressed.mp4",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    mkv_job = _job_for(mkv_source)
    mp4_mkv_job = EncodeJob(
        source=mp4_source,
        output=tmp_path / "mediashrink_mkv_followup" / "movie.mkv",
        tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_movie.mkv",
        crf=20,
        preset="fast",
        dry_run=False,
        skip=False,
    )
    selected_profile = EncoderProfile(
        1, "Balanced", "Balanced", "fast", 20, "fast", 0, 0.0, "Excellent", True
    )

    with (
        patch("mediashrink.wizard.scan_directory", return_value=[mp4_source, mkv_source]),
        patch(
            "mediashrink.wizard._run_analysis_with_progress",
            return_value=[recommended_mp4, recommended_mkv],
        ),
        patch("mediashrink.wizard.detect_available_encoders", return_value=[]),
        patch("mediashrink.wizard.detect_device_labels", return_value={}),
        patch("mediashrink.wizard.benchmark_encoder", side_effect=[1.0, 1.0, 0.8]),
        patch("mediashrink.wizard.display_profiles_table"),
        patch("mediashrink.wizard.prompt_profile_selection", return_value=selected_profile),
        patch("mediashrink.wizard.maybe_save_profile"),
        patch("mediashrink.wizard._maybe_run_preview", return_value=True),
        patch("mediashrink.wizard.display_analysis_summary"),
        patch("mediashrink.wizard.prompt_analysis_action", return_value="compress_recommended"),
        patch("mediashrink.wizard.build_jobs", return_value=[mp4_job, mkv_job]),
        patch("mediashrink.wizard._build_mkv_followup_jobs", return_value=[mp4_mkv_job]),
        patch(
            "mediashrink.wizard._run_preflight_checks",
            side_effect=[
                (
                    [mkv_job],
                    [
                        (
                            mp4_job,
                            _fake_encode_result(
                                mp4_source,
                                success=False,
                                error_message="[mp4 @ 123] Could not write header\nInvalid argument",
                            ),
                        )
                    ],
                ),
                (
                    [],
                    [
                        (
                            mp4_mkv_job,
                            _fake_encode_result(
                                mp4_source,
                                success=False,
                                error_message="attachment stream incompatibility",
                            ),
                        )
                    ],
                ),
            ],
        ),
        patch("mediashrink.wizard._wizard_confirm", side_effect=[True, True, True]),
    ):
        jobs, action, _, _fm = run_wizard(
            tmp_path, FFMPEG, FFPROBE, False, None, False, False, console
        )

    output = console.export_text()
    assert action == "encode"
    assert jobs == [mkv_job]
    assert "Tried MKV sidecar output for 1 file(s), but they still needed follow-up." in output
    assert "Follow-up file:" in output
    assert "movie.mp4" in output
    assert "MKV retry still left out:" in output
    assert "MKV-first retry command:" in output


def test_cleanup_expectation_lines_distinguish_same_format_outputs_from_true_sidecars(
    tmp_path: Path,
) -> None:
    from mediashrink.wizard import _cleanup_expectation_lines

    mkv_source = tmp_path / "episode.mkv"
    mp4_source = tmp_path / "movie.mp4"
    mkv_source.write_bytes(b"x")
    mp4_source.write_bytes(b"x")
    jobs = [
        _job_for(mkv_source),
        EncodeJob(
            source=mp4_source,
            output=tmp_path / "mediashrink_mkv_followup" / "movie.mkv",
            tmp_output=tmp_path / "mediashrink_mkv_followup" / ".tmp_movie.mkv",
            crf=20,
            preset="fast",
            dry_run=False,
            skip=False,
        ),
    ]

    lines = _cleanup_expectation_lines(jobs, cleanup_after=True)

    assert any("same-format output" in line for line in lines)
    assert any("true MKV sidecar output" in line for line in lines)


def test_build_profiles_outlier_hint_calls_out_single_risky_mp4(tmp_path: Path) -> None:
    console = Console(record=True, width=160)
    mkv_item = _analysis_item(tmp_path / "episode.mkv", "recommended")
    mp4_item = _analysis_item(tmp_path / "movie.mp4", "recommended")

    with patch(
        "mediashrink.wizard._cached_container_incompatibility",
        return_value="audio codec copy incompatibility",
    ):
        profiles = build_profiles(
            available_hw=[],
            benchmark_speeds={"fast": 1.0},
            total_media_seconds=3600.0,
            total_input_bytes=10 * 1024**3,
            candidate_items=[mkv_item, mp4_item],
            ffprobe=FFPROBE,
        )

    recommended = next(profile for profile in profiles if profile.is_recommended)
    assert (
        recommended.outlier_hint
        == "All MKV files look compatible; the single MP4 likely needs MKV output."
    )

    display_profiles_table(
        [recommended],
        total_input_bytes=1_000,
        candidate_count=2,
        recommended_count=2,
        device_labels={},
        console=console,
        plain_output=True,
    )

    assert "single MP4 likely needs MKV output" in console.export_text()


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
    with patch("mediashrink.wizard._wizard_prompt", return_value="1"):
        from mediashrink.wizard import prompt_analysis_action

        action = prompt_analysis_action(3, 0, CapturingConsole())  # type: ignore[arg-type]

    assert action == "compress_recommended"
    joined = "\n".join(output_lines)
    assert "Review maybe" not in joined
    assert "Compress recommended files (3 file(s))" in joined
    # Options should be 1, 2, 3 (not 1, 3, 4)
    assert "2." in joined
    assert "3." in joined
    assert "4." not in joined


def test_next_step_menu_includes_counts_when_maybe_present() -> None:
    output_lines: list[str] = []

    class CapturingConsole:
        def print(self, msg: str = "", **kwargs: object) -> None:
            output_lines.append(str(msg))

    with patch("mediashrink.wizard._wizard_prompt", return_value="1"):
        from mediashrink.wizard import prompt_analysis_action

        action = prompt_analysis_action(22, 5, CapturingConsole())  # type: ignore[arg-type]

    assert action == "compress_recommended"
    joined = "\n".join(output_lines)
    assert "Compress recommended only (22 file(s))" in joined
    assert "Review maybe files (5 file(s))" in joined


def test_display_profiles_table_warns_when_default_is_only_partial_batch() -> None:
    console = Console(record=True, width=160)
    profile = EncoderProfile(
        1,
        "Fast",
        "Fast",
        "faster",
        22,
        "faster",
        500,
        100.0,
        "Very good",
        True,
        why_choose="Partial-batch default only: 2 file(s) can run now, while 2 likely need follow-up.",
        compatible_count=2,
        incompatible_count=2,
        recommended_compatible_count=2,
        recommended_incompatible_count=2,
    )

    display_profiles_table(
        [profile],
        total_input_bytes=1_000,
        candidate_count=4,
        recommended_count=4,
        device_labels={},
        console=console,
        plain_output=True,
    )

    output = console.export_text()
    assert "partial-batch starting point" in output.lower()
