from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from mediashrink.analysis import (
    adjust_time_confidence_for_scope,
    apply_duplicate_policy_to_items,
    analyze_files,
    build_analysis_item,
    build_manifest,
    describe_estimate_calibration,
    describe_estimate_confidence,
    describe_size_confidence,
    describe_time_confidence,
    display_analysis_summary,
    estimate_analysis_confidence,
    estimate_analysis_encode_seconds,
    estimate_size_confidence,
    estimate_time_confidence,
    estimate_time_range_widening,
    load_manifest,
    rank_maybe_candidates,
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


def test_build_analysis_item_recommends_tv_sized_candidate(tmp_path: Path) -> None:
    path = tmp_path / "episode.mkv"
    path.write_bytes(b"x" * (1700 * 1024**2))

    with (
        patch("mediashrink.analysis.probe_video_codec", return_value="h264"),
        patch("mediashrink.analysis.is_already_compressed", return_value=(False, "")),
        patch("mediashrink.analysis.get_duration_seconds", return_value=2700.0),
        patch("mediashrink.analysis.get_video_bitrate_kbps", return_value=7000.0),
        patch("mediashrink.analysis.estimate_output_size", return_value=900 * 1024**2),
    ):
        item = build_analysis_item(path, FFPROBE)

    assert item.recommendation == "recommended"
    assert item.reason_code == "strong_savings_candidate"
    assert "TV/library cleanup" in item.reason_text


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


def test_estimate_analysis_encode_seconds_blends_known_speed_with_hardware_calibration(
    tmp_path: Path,
) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
        duration_seconds=120.0,
    )
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "unknown",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 22,
                "input_bytes": 1_000_000,
                "output_bytes": 400_000,
                "duration_seconds": 120.0,
                "wall_seconds": 30.0,
                "effective_speed": 4.0,
                "fallback_used": False,
                "retry_used": False,
            }
        ],
        "failures": [],
    }

    estimate = estimate_analysis_encode_seconds(
        [recommended],
        "amf",
        22,
        FFMPEG,
        known_speed=1.0,
        calibration_store=calibration_store,
    )

    assert estimate is not None
    assert estimate < 120.0


def test_estimate_analysis_encode_seconds_hardware_blend_stays_close_to_slower_signal(
    tmp_path: Path,
) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
        duration_seconds=120.0,
    )
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "unknown",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 22,
                "input_bytes": 1_000_000,
                "output_bytes": 500_000,
                "duration_seconds": 120.0,
                "wall_seconds": 120.0,
                "effective_speed": 1.0,
                "fallback_used": False,
                "retry_used": False,
            }
        ],
        "failures": [],
    }

    estimate = estimate_analysis_encode_seconds(
        [recommended],
        "amf",
        22,
        FFMPEG,
        known_speed=4.0,
        calibration_store=calibration_store,
    )

    assert estimate is not None
    assert estimate > 60.0
    assert estimate < 120.0


def test_estimate_analysis_encode_seconds_software_blend_leans_toward_slower_history(
    tmp_path: Path,
) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
        duration_seconds=120.0,
    )
    recommended.width = 1920
    recommended.height = 1080
    recommended.bitrate_kbps = 0.0
    with patch(
        "mediashrink.analysis.lookup_estimate",
        return_value=SimpleNamespace(
            speed=1.0,
            average_speed_error=0.0,
        ),
    ):
        estimate = estimate_analysis_encode_seconds(
            [recommended],
            "faster",
            22,
            FFMPEG,
            known_speed=4.0,
            calibration_store={"version": 1, "records": [], "failures": []},
        )

    assert estimate is not None
    assert estimate > 48.0
    assert estimate < 120.0


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
    assert "Size confidence: High" in output
    assert "Time confidence: High" in output


def test_estimate_analysis_confidence_prefers_known_durations_and_low_codec_mix(
    tmp_path: Path,
) -> None:
    items = [
        build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended"),
        build_analysis_item_dict_item(source=tmp_path / "b.mkv", recommendation="recommended"),
    ]

    assert estimate_analysis_confidence(items, benchmarked_files=1) == "High"


def test_estimate_time_confidence_caps_high_before_benchmark_on_mixed_batch(
    tmp_path: Path,
) -> None:
    mkv_item = build_analysis_item_dict_item(
        source=tmp_path / "episode.mkv", recommendation="recommended"
    )
    mp4_item = build_analysis_item_dict_item(
        source=tmp_path / "movie.mp4", recommendation="recommended"
    )
    mp4_item.codec = "mpeg2video"

    confidence = estimate_time_confidence([mkv_item, mp4_item], benchmarked_files=0)

    assert confidence == "Medium"


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


def test_rank_maybe_candidates_prefers_tv_near_recommended_items(tmp_path: Path) -> None:
    stronger = build_analysis_item_dict_item(
        source=tmp_path / "stronger.mkv",
        recommendation="maybe",
        duration_seconds=2700.0,
    )
    stronger.size_bytes = 1500 * 1024**2
    stronger.estimated_output_bytes = 760 * 1024**2
    stronger.estimated_savings_bytes = stronger.size_bytes - stronger.estimated_output_bytes
    stronger.reason_text = "episode-scale file with worthwhile projected savings; near recommended for TV/library cleanup"

    weaker = build_analysis_item_dict_item(
        source=tmp_path / "weaker.mkv",
        recommendation="maybe",
        duration_seconds=5400.0,
    )
    weaker.size_bytes = 1400 * 1024**2
    weaker.estimated_output_bytes = 900 * 1024**2
    weaker.estimated_savings_bytes = weaker.size_bytes - weaker.estimated_output_bytes

    ranked = rank_maybe_candidates([weaker, stronger])

    assert [item.source.name for item in ranked] == ["stronger.mkv", "weaker.mkv"]


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


def test_describe_time_confidence_uses_user_facing_bias_and_history_slices(tmp_path: Path) -> None:
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
                "output_bytes": 700,
                "predicted_output_ratio": 0.5,
                "duration_seconds": 100.0,
                "wall_seconds": 80.0,
                "predicted_speed": 2.0,
                "effective_speed": 1.5,
                "accepted_output": True,
                "fallback_used": False,
                "retry_used": False,
            }
            for _ in range(4)
        ],
        "failures": [],
    }

    detail = describe_time_confidence(
        [item],
        benchmarked_files=1,
        preset="fast",
        calibration_store=calibration_store,
    )

    assert "closest preset history:" in detail
    assert "current container mix:" in detail
    assert "overall machine history:" in detail
    assert "saved less space than forecast" in detail


def test_duplicate_policy_prefers_mkv_copy(tmp_path: Path) -> None:
    mkv_item = build_analysis_item_dict_item(
        source=tmp_path / "Film (2005).mkv",
        recommendation="recommended",
    )
    mp4_item = build_analysis_item_dict_item(
        source=tmp_path / "Film (2005).mp4",
        recommendation="recommended",
    )

    updated, notes = apply_duplicate_policy_to_items([mp4_item, mkv_item], policy="prefer-mkv")

    assert [item.source.suffix for item in updated if item.recommendation != "skip"] == [".mkv"]
    assert any("Preferred" in note for note in notes)


def test_size_and_time_confidence_can_diverge(tmp_path: Path) -> None:
    item = build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended")
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "unknown",
                "preset": "fast",
                "preset_family": "software",
                "crf": 20,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 200.0,
                "effective_speed": 0.5,
                "fallback_used": False,
                "retry_used": False,
                "predicted_output_ratio": 0.4,
                "predicted_speed": 1.0,
            }
        ],
        "failures": [],
    }

    size_conf = estimate_size_confidence([item], preset="fast", calibration_store=calibration_store)
    time_conf = estimate_time_confidence(
        [item],
        benchmarked_files=1,
        preset="fast",
        calibration_store=calibration_store,
    )
    size_detail = describe_size_confidence(
        [item], preset="fast", calibration_store=calibration_store
    )
    time_detail = describe_time_confidence(
        [item],
        benchmarked_files=1,
        preset="fast",
        calibration_store=calibration_store,
    )

    assert size_conf in {"Medium", "High"}
    assert time_conf in {"Medium", "High"}
    assert "local history" in size_detail
    assert "benchmark sample" in time_detail


def test_size_confidence_drops_when_local_history_is_highly_volatile(tmp_path: Path) -> None:
    item = build_analysis_item_dict_item(source=tmp_path / "a.mkv", recommendation="recommended")
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "unknown",
                "preset": "amf",
                "preset_family": "hardware",
                "crf": 20,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 40.0,
                "effective_speed": 2.5,
                "fallback_used": False,
                "retry_used": False,
                "predicted_output_ratio": 0.4,
                "predicted_speed": 2.0,
            }
        ],
        "failures": [],
    }

    with patch(
        "mediashrink.analysis.lookup_estimate",
        return_value=SimpleNamespace(
            output_ratio=0.5,
            average_size_error=0.28,
        ),
    ):
        size_conf = estimate_size_confidence(
            [item], preset="amf", calibration_store=calibration_store
        )

    assert size_conf == "Medium"


def test_estimate_time_confidence_downgrades_large_software_batch_with_slow_bias(
    tmp_path: Path,
) -> None:
    items = [
        build_analysis_item_dict_item(
            source=tmp_path / f"ep{i:02d}.mkv",
            recommendation="recommended",
            duration_seconds=1800.0 + (i % 3) * 1200.0,
        )
        for i in range(30)
    ]
    calibration_store = {
        "version": 1,
        "records": [
            {
                "codec": "h264",
                "container": ".mkv",
                "resolution_bucket": "unknown",
                "bitrate_bucket": "unknown",
                "preset": "fast",
                "preset_family": "software",
                "crf": 20,
                "input_bytes": 1000,
                "output_bytes": 700,
                "duration_seconds": 100.0,
                "wall_seconds": 120.0,
                "predicted_speed": 2.0,
                "effective_speed": 1.4,
                "fallback_used": False,
                "retry_used": False,
                "accepted_output": True,
            }
            for _ in range(6)
        ],
        "failures": [],
    }

    confidence = estimate_time_confidence(
        items,
        benchmarked_files=1,
        preset="fast",
        calibration_store=calibration_store,
    )

    assert confidence == "Medium"


def test_estimate_time_range_widening_grows_for_large_software_batches(tmp_path: Path) -> None:
    items = [
        build_analysis_item_dict_item(
            source=tmp_path / f"ep{i:02d}.mkv",
            recommendation="recommended",
            duration_seconds=1800.0 + (i % 4) * 1500.0,
        )
        for i in range(36)
    ]

    widen = estimate_time_range_widening(items, preset="fast", benchmarked_files=1)

    assert widen >= 0.17


def test_manifest_round_trip_keeps_duplicate_policy_and_notes(tmp_path: Path) -> None:
    recommended = build_analysis_item_dict_item(
        source=tmp_path / "recommended.mkv",
        recommendation="recommended",
    )

    manifest = build_manifest(
        directory=tmp_path,
        recursive=True,
        preset="fast",
        crf=20,
        profile_name=None,
        estimated_total_encode_seconds=1234.0,
        estimate_confidence="Medium",
        size_confidence="Low",
        size_confidence_detail="heuristic only",
        time_confidence="High",
        time_confidence_detail="1 benchmark sample",
        duplicate_policy="prefer-mkv",
        notes=["Preferred MKV copy over duplicate MP4"],
        items=[recommended],
    )
    path = tmp_path / "analysis.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded.duplicate_policy == "prefer-mkv"
    assert loaded.notes == ["Preferred MKV copy over duplicate MP4"]


def test_time_confidence_downgrades_for_mixed_sidecar_scope(tmp_path: Path) -> None:
    mkv_item = build_analysis_item_dict_item(
        source=tmp_path / "movie.mkv",
        recommendation="recommended",
    )
    mp4_item = build_analysis_item_dict_item(
        source=tmp_path / "movie.mp4",
        recommendation="recommended",
    )
    mp4_item.codec = "mpeg2video"

    adjusted = adjust_time_confidence_for_scope(
        "High",
        [mkv_item, mp4_item],
        original_items=[mkv_item, mp4_item],
        sidecar_count=1,
        followup_count=1,
    )

    assert adjusted == "Medium"


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
