from __future__ import annotations

import json
from pathlib import Path

from mediashrink.calibration import (
    CalibrationRecord,
    append_success_record,
    describe_calibration_estimate,
    load_calibration_store,
    lookup_estimate,
    summarize_calibration_store,
)


def test_success_records_round_trip_and_lookup_prefers_exact_bucket(tmp_path: Path) -> None:
    store_path = tmp_path / "calibration.json"
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mkv",
            resolution_bucket="1080p",
            bitrate_bucket="high",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=450_000,
            duration_seconds=100.0,
            wall_seconds=50.0,
            effective_speed=2.0,
            fallback_used=False,
            retry_used=False,
        ),
        path=store_path,
    )
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mkv",
            resolution_bucket="1080p",
            bitrate_bucket="medium",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=700_000,
            duration_seconds=100.0,
            wall_seconds=100.0,
            effective_speed=1.0,
            fallback_used=False,
            retry_used=False,
        ),
        path=store_path,
    )

    store = load_calibration_store(store_path)
    estimate = lookup_estimate(
        store,
        codec="h264",
        resolution="1080p",
        bitrate="high",
        preset="fast",
        container=".mkv",
    )

    assert estimate is not None
    assert estimate.output_ratio is not None
    assert 0.45 < estimate.output_ratio < 0.60
    assert estimate.speed is not None
    assert 1.5 < estimate.speed < 2.0


def test_load_calibration_store_returns_empty_for_missing_file(tmp_path: Path) -> None:
    store = load_calibration_store(tmp_path / "missing.json")
    assert store["records"] == []
    assert store["failures"] == []


def test_lookup_estimate_blends_exact_and_related_matches(tmp_path: Path) -> None:
    store_path = tmp_path / "calibration.json"
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mkv",
            resolution_bucket="1080p",
            bitrate_bucket="high",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=500_000,
            duration_seconds=100.0,
            wall_seconds=50.0,
            effective_speed=2.0,
            fallback_used=False,
            retry_used=False,
        ),
        path=store_path,
    )
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mkv",
            resolution_bucket="1080p",
            bitrate_bucket="medium",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=800_000,
            duration_seconds=100.0,
            wall_seconds=100.0,
            effective_speed=1.0,
            fallback_used=False,
            retry_used=False,
        ),
        path=store_path,
    )

    estimate = lookup_estimate(
        load_calibration_store(store_path),
        codec="h264",
        resolution="1080p",
        bitrate="high",
        preset="fast",
        container=".mkv",
    )

    assert estimate is not None
    assert estimate.output_ratio is not None
    assert 0.50 < estimate.output_ratio < 0.65
    assert estimate.weighted_samples > 1.0
    assert describe_calibration_estimate(estimate) is not None


def test_lookup_estimate_accepts_legacy_records_without_codec_family(tmp_path: Path) -> None:
    store_path = tmp_path / "calibration.json"
    append_success_record(
        CalibrationRecord(
            codec="mpeg2video",
            container=".mkv",
            resolution_bucket="sd",
            bitrate_bucket="low",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=280_000,
            duration_seconds=100.0,
            wall_seconds=40.0,
            effective_speed=2.5,
            fallback_used=False,
            retry_used=False,
            codec_family="unknown",
        ),
        path=store_path,
    )

    estimate = lookup_estimate(
        load_calibration_store(store_path),
        codec="mpeg2video",
        resolution="sd",
        bitrate="low",
        preset="fast",
        container=".mkv",
    )

    assert estimate is not None
    assert estimate.output_ratio is not None
    assert estimate.output_ratio < 0.40


def test_calibration_summary_counts_rejected_outputs(tmp_path: Path) -> None:
    store_path = tmp_path / "calibration.json"
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mp4",
            resolution_bucket="1080p",
            bitrate_bucket="high",
            preset="amf",
            preset_family="hardware",
            crf=22,
            input_bytes=1_000_000,
            output_bytes=3_000_000,
            duration_seconds=100.0,
            wall_seconds=10.0,
            effective_speed=10.0,
            fallback_used=False,
            retry_used=False,
            accepted_output=False,
            safety_rejection_reason="oversized output",
        ),
        path=store_path,
    )

    summary = summarize_calibration_store(load_calibration_store(store_path))

    assert summary["records"] == 1
    assert summary["accepted_records"] == 0
    assert summary["rejected_records"] == 1


def test_calibration_summary_exposes_family_mix_and_bias(tmp_path: Path) -> None:
    store_path = tmp_path / "calibration.json"
    append_success_record(
        CalibrationRecord(
            codec="h264",
            container=".mkv",
            resolution_bucket="1080p",
            bitrate_bucket="high",
            preset="fast",
            preset_family="software",
            crf=20,
            input_bytes=1_000_000,
            output_bytes=300_000,
            duration_seconds=100.0,
            wall_seconds=140.0,
            effective_speed=0.7,
            fallback_used=False,
            retry_used=False,
            predicted_output_ratio=0.5,
            predicted_speed=1.0,
        ),
        path=store_path,
    )

    summary = summarize_calibration_store(load_calibration_store(store_path))

    assert summary["family_container_summary_text"] is not None
    bias_summary = summary["bias_summary"]
    assert isinstance(bias_summary, dict)
    assert "saved more space than forecast" in str(bias_summary.get("summary"))
    assert "slower than forecast" in str(bias_summary.get("summary"))
