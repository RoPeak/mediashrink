from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from mediashrink.analysis import (
    analyze_files,
    build_analysis_item,
    build_manifest,
    describe_estimate_calibration,
    describe_estimate_confidence,
    display_analysis_summary,
    estimate_analysis_confidence,
    estimate_analysis_encode_seconds,
    load_manifest,
    save_manifest,
    select_representative_items,
)

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def test_build_analysis_item_skips_hevc(tmp_path: Path) -> None:
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"x" * 1024)

    with (
        patch("mediashrink.analysis.probe_video_codec", return_value="hevc"),
        patch(
            "mediashrink.analysis.is_already_compressed",
            return_value=(True, "video stream is already H.265/HEVC"),
        ),
        patch("mediashrink.analysis.get_duration_seconds", return_value=100.0),
        patch("mediashrink.analysis.get_video_bitrate_kbps", return_value=10000.0),
    ):
        item = build_analysis_item(path, FFPROBE)

    assert item.recommendation == "skip"
    assert item.reason_code == "already_hevc"


def test_build_analysis_item_skips_compressed_filename(tmp_path: Path) -> None:
    path = tmp_path / "movie_compressed.mkv"
    path.write_bytes(b"x" * 1024)

    with (
        patch("mediashrink.analysis.probe_video_codec", return_value="h264"),
        patch(
            "mediashrink.analysis.is_already_compressed",
            return_value=(True, "filename contains '_compressed'"),
        ),
        patch("mediashrink.analysis.get_duration_seconds", return_value=100.0),
        patch("mediashrink.analysis.get_video_bitrate_kbps", return_value=10000.0),
    ):
        item = build_analysis_item(path, FFPROBE)

    assert item.recommendation == "skip"
    assert item.reason_code == "already_marked_compressed"


def test_build_analysis_item_recommends_strong_candidate(tmp_path: Path) -> None:
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"x" * (2 * 1024**3))

    with (
        patch("mediashrink.analysis.probe_video_codec", return_value="h264"),
        patch("mediashrink.analysis.is_already_compressed", return_value=(False, "")),
        patch("mediashrink.analysis.get_duration_seconds", return_value=3600.0),
        patch("mediashrink.analysis.get_video_bitrate_kbps", return_value=12000.0),
        patch("mediashrink.analysis.estimate_output_size", return_value=700 * 1024**2),
    ):
        item = build_analysis_item(path, FFPROBE)

    assert item.recommendation == "recommended"
    assert item.reason_code == "strong_savings_candidate"


def test_build_analysis_item_marks_borderline_candidate(tmp_path: Path) -> None:
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"x" * (900 * 1024**2))

    with (
        patch("mediashrink.analysis.probe_video_codec", return_value="h264"),
        patch("mediashrink.analysis.is_already_compressed", return_value=(False, "")),
        patch("mediashrink.analysis.get_duration_seconds", return_value=1800.0),
        patch("mediashrink.analysis.get_video_bitrate_kbps", return_value=6000.0),
        patch("mediashrink.analysis.estimate_output_size", return_value=500 * 1024**2),
    ):
        item = build_analysis_item(path, FFPROBE)

    assert item.recommendation == "maybe"
    assert item.reason_code == "borderline_candidate"


def test_analyze_files_reports_progress_for_each_file(tmp_path: Path) -> None:
    files = [tmp_path / "a.mkv", tmp_path / "b.mkv"]
    for file in files:
        file.write_bytes(b"x")

    reported: list[tuple[int, int, str]] = []

    def callback(completed: int, total: int, path: Path) -> None:
        reported.append((completed, total, path.name))

    with patch(
        "mediashrink.analysis.build_analysis_item",
        side_effect=lambda path, _: build_analysis_item_dict_item(
            source=path, recommendation="recommended"
        ),
    ):
        items = analyze_files(files, FFPROBE, progress_callback=callback)

    assert len(items) == 2
    assert reported == [(1, 2, "a.mkv"), (2, 2, "b.mkv")]


def test_manifest_round_trip_keeps_recommended_only(tmp_path: Path) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
        reason_code="strong_savings_candidate",
    )
    maybe = build_analysis_item_dict_item(
        source=tmp_path / "maybe.mkv",
        recommendation="maybe",
        reason_code="borderline_candidate",
    )

    manifest = build_manifest(
        directory=tmp_path,
        recursive=True,
        preset="fast",
        crf=20,
        profile_name="tv",
        estimated_total_encode_seconds=1234.0,
        estimate_confidence="Medium",
        items=[recommended, maybe],
    )
    path = tmp_path / "analysis.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded.version == 1
    assert loaded.profile_name == "tv"
    assert loaded.estimate_confidence == "Medium"
    assert loaded.preset == "fast"
    assert loaded.crf == 20
    assert len(loaded.items) == 1
    assert loaded.items[0].recommendation == "recommended"


def test_manifest_load_rejects_invalid_shape(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(
        '{"version": 1, "analyzed_directory": "C:/tmp", "preset": "fast", "crf": 20, "items": "bad"}',
        encoding="utf-8",
    )

    try:
        load_manifest(path)
    except ValueError as exc:
        assert "items" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_estimate_analysis_encode_seconds_uses_recommended_only(tmp_path: Path) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
        duration_seconds=100.0,
    )
    maybe = build_analysis_item_dict_item(
        source=tmp_path / "maybe.mkv",
        recommendation="maybe",
        duration_seconds=500.0,
    )

    with patch("mediashrink.wizard.benchmark_encoder", return_value=2.0):
        estimate = estimate_analysis_encode_seconds([recommended, maybe], "fast", 20, FFMPEG)

    assert estimate == 50.0


def test_display_analysis_summary_prints_counts(tmp_path: Path) -> None:
    console = Console(record=True, width=140)
    items = [
        build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended"),
        build_analysis_item_dict_item(source=tmp_path / "b.mkv", recommendation="maybe"),
        build_analysis_item_dict_item(source=tmp_path / "c.mkv", recommendation="skip"),
    ]

    display_analysis_summary(
        items,
        600.0,
        console,
        estimate_confidence="High",
        estimate_confidence_detail="1 benchmark sample, 3/3 file durations known, 1 codec group",
    )
    output = console.export_text()

    assert "recommended" in output.lower()
    assert "Rollup:" in output
    assert "maybe 1" in output
    assert "skip 1" in output
    assert "Rough encode time" in output
    assert "Estimate confidence: High" in output


def test_estimate_analysis_confidence_prefers_known_durations_and_low_codec_mix(
    tmp_path: Path,
) -> None:
    items = [
        build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended"),
        build_analysis_item_dict_item(source=tmp_path / "b.mkv", recommendation="recommended"),
    ]

    assert estimate_analysis_confidence(items, benchmarked_files=1) == "High"


def test_describe_estimate_confidence_mentions_benchmark_and_codecs(tmp_path: Path) -> None:
    items = [build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended")]

    detail = describe_estimate_confidence(items, benchmarked_files=1)

    assert "1 benchmark sample" in detail
    assert "1 codec group" in detail


def test_select_representative_items_prefers_legacy_h264_and_maybe(tmp_path: Path) -> None:
    legacy = build_analysis_item_dict_item(
        source=tmp_path / "legacy.mkv", recommendation="recommended"
    )
    legacy.codec = "vc1"
    h264 = build_analysis_item_dict_item(source=tmp_path / "h264.mkv", recommendation="recommended")
    h264.codec = "h264"
    maybe = build_analysis_item_dict_item(source=tmp_path / "maybe.mkv", recommendation="maybe")
    maybe.codec = "hevc"
    extras = build_analysis_item_dict_item(
        source=tmp_path / "other.mkv", recommendation="recommended"
    )
    extras.codec = "mpeg4"

    selected = select_representative_items([extras, maybe, h264, legacy], limit=3)

    assert [item.source.name for item in selected] == ["legacy.mkv", "h264.mkv", "maybe.mkv"]


def test_describe_estimate_calibration_mentions_local_history(tmp_path: Path) -> None:
    item = build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended")
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
                "wall_seconds": 50.0,
                "effective_speed": 2.0,
                "fallback_used": False,
                "retry_used": False,
            }
        ],
        "failures": [],
    }

    detail = describe_estimate_calibration(
        [item],
        preset="fast",
        calibration_store=calibration_store,
    )

    assert detail is not None
    assert "close local match" in detail


def build_analysis_item_dict_item(
    *,
    source: Path,
    recommendation: str,
    reason_code: str = "reason",
    reason_text: str = "reason text",
    duration_seconds: float = 120.0,
) -> object:
    source.write_bytes(b"x")
    from mediashrink.models import AnalysisItem

    return AnalysisItem(
        source=source,
        codec="h264",
        size_bytes=2 * 1024**3,
        duration_seconds=duration_seconds,
        bitrate_kbps=12000.0,
        estimated_output_bytes=800 * 1024**2,
        estimated_savings_bytes=(2 * 1024**3) - (800 * 1024**2),
        recommendation=recommendation,
        reason_code=reason_code,
        reason_text=reason_text,
    )
