from __future__ import annotations

import json
from pathlib import Path

from mediashrink.calibration import (
    CalibrationRecord,
    append_success_record,
    load_calibration_store,
    lookup_estimate,
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
    assert estimate.output_ratio == 0.45
    assert estimate.speed == 2.0


def test_load_calibration_store_returns_empty_for_missing_file(tmp_path: Path) -> None:
    store = load_calibration_store(tmp_path / "missing.json")
    assert store["records"] == []
    assert store["failures"] == []
