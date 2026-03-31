from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from mkv_compress.wizard import (
    _sum_media_durations,
    benchmark_encoder,
    build_profiles,
    detect_available_encoders,
    display_profiles_table,
    maybe_save_profile,
    run_custom_wizard,
)

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def test_detect_available_encoders_returns_available_only() -> None:
    console = Console()

    def fake_probe(key: str, ffmpeg: Path) -> bool:
        return key == "qsv"

    with patch("mkv_compress.wizard.probe_encoder_available", side_effect=fake_probe):
        result = detect_available_encoders(FFMPEG, console)

    assert result == ["qsv"]


def test_detect_available_encoders_stable_order_even_if_completion_varies() -> None:
    console = Console()

    def fake_probe(key: str, ffmpeg: Path) -> bool:
        return key in {"nvenc", "qsv"}

    with patch("mkv_compress.wizard.probe_encoder_available", side_effect=fake_probe):
        result = detect_available_encoders(FFMPEG, console)

    assert result == ["qsv", "nvenc"]


def test_benchmark_encoder_returns_speed_on_success(tmp_path: Path) -> None:
    sample = tmp_path / "sample.mkv"
    sample.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("mkv_compress.wizard.subprocess.run", return_value=mock_result), \
         patch("mkv_compress.wizard.time.monotonic", side_effect=[0.0, 4.0]):
        speed = benchmark_encoder("fast", sample, 40.0, 20, FFMPEG)

    assert speed == pytest.approx(2.0, rel=0.01)


def test_benchmark_encoder_hardware_key(tmp_path: Path) -> None:
    sample = tmp_path / "sample.mkv"
    sample.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("mkv_compress.wizard.subprocess.run", return_value=mock_result) as mock_run, \
         patch("mkv_compress.wizard.time.monotonic", side_effect=[0.0, 2.0]):
        benchmark_encoder("qsv", sample, 40.0, 20, FFMPEG)

    cmd = mock_run.call_args[0][0]
    assert "hevc_qsv" in cmd
    assert "-global_quality" in cmd


def test_sum_media_durations_uses_average_fallback(tmp_path: Path) -> None:
    files = [tmp_path / "a.mkv", tmp_path / "b.mkv", tmp_path / "c.mkv"]
    for file in files:
        file.write_bytes(b"x")

    with patch("mkv_compress.wizard.get_duration_seconds", side_effect=[100.0, 0.0, 200.0]):
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
    smallest = next(profile for profile in profiles if profile.name == "Smallest File")
    assert smallest.estimated_output_bytes < balanced.estimated_output_bytes


def test_display_profiles_table_uses_device_label() -> None:
    console = Console(record=True, width=140)
    profiles = build_profiles(
        available_hw=["qsv"],
        benchmark_speeds={"qsv": 5.0, "fast": 0.3},
        total_media_seconds=3600.0,
        total_input_bytes=10 * 1024**3,
    )

    display_profiles_table(profiles, 10 * 1024**3, {"qsv": "Intel Arc Test"}, console)

    output = console.export_text()
    assert "Intel Quick Sync" in output
    assert "Intel" in output
    assert "Arc Test" in output
    assert "approximate estimates" in output


def test_run_custom_wizard_returns_hardware_choice() -> None:
    console = Console()

    with patch("mkv_compress.wizard.typer.prompt", side_effect=["1", "21"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "qsv"
    assert crf == 21
    assert sw_preset is None


def test_run_custom_wizard_returns_software_preset() -> None:
    console = Console()

    with patch("mkv_compress.wizard.typer.prompt", side_effect=["2", "22", "4"]):
        preset, crf, sw_preset = run_custom_wizard(["qsv"], console)

    assert preset == "medium"
    assert crf == 22
    assert sw_preset == "medium"


def test_maybe_save_profile_persists_choice(tmp_path: Path, monkeypatch) -> None:
    console = Console()
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    with patch("mkv_compress.wizard.typer.confirm", return_value=True), \
         patch("mkv_compress.wizard.typer.prompt", return_value="tv-batch"):
        maybe_save_profile("slow", 18, "Best Quality", console)

    from mkv_compress.profiles import get_profile

    profile = get_profile("tv-batch")
    assert profile is not None
    assert profile.preset == "slow"
    assert profile.crf == 18
    assert profile.label == "Best Quality"
