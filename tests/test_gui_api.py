from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mediashrink.gui_api import (
    EncodeProgress,
    auto_select_profile,
    prepare_encode_run,
    run_encode_plan,
)
from mediashrink.models import AnalysisItem, EncodeJob, EncodeResult
from mediashrink.wizard import EncoderProfile, ProfilePlanningResult

FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")


def _analysis_item(path: Path, recommendation: str = "recommended") -> AnalysisItem:
    return AnalysisItem(
        source=path,
        codec="h264",
        size_bytes=5 * 1024**3,
        duration_seconds=1800.0,
        bitrate_kbps=5000.0,
        estimated_output_bytes=2 * 1024**3,
        estimated_savings_bytes=3 * 1024**3,
        recommendation=recommendation,
        reason_code="strong_savings_candidate",
        reason_text="legacy codec with strong projected space savings",
    )


def test_prepare_encode_run_returns_empty_plan_for_empty_directory(tmp_path: Path) -> None:
    with (
        patch("mediashrink.gui_api.prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.gui_api.scan_directory", return_value=[]),
    ):
        result = prepare_encode_run(directory=tmp_path)

    assert result.items == []
    assert result.jobs == []
    assert result.profile is None
    assert result.selected_count == 0
    assert result.total_input_bytes == 0


def test_auto_select_profile_returns_recommended_profile(tmp_path: Path) -> None:
    item = _analysis_item(tmp_path / "movie.mkv")
    planning = ProfilePlanningResult(
        candidate_items=[item],
        candidate_input_bytes=item.size_bytes,
        candidate_media_seconds=item.duration_seconds,
        sample_item=item,
        sample_duration=item.duration_seconds,
        preview_items=[item],
        available_hw=[],
        benchmark_speeds={"fast": 1.0, "faster": 2.0},
        observed_probe_failures={},
        profiles=[
            EncoderProfile(
                index=1,
                intent_label="Fast",
                name="Fast",
                encoder_key="faster",
                crf=22,
                sw_preset="faster",
                estimated_output_bytes=item.estimated_output_bytes,
                estimated_encode_seconds=900.0,
                quality_label="Very good",
                is_recommended=True,
            )
        ],
        active_calibration=None,
        size_error_by_preset={},
        stage_messages=["Benchmarking profiles... 2/2"],
    )
    with patch("mediashrink.gui_api.prepare_profile_planning", return_value=planning):
        profile = auto_select_profile([item], ffmpeg=FFMPEG, ffprobe=FFPROBE)

    assert profile is not None
    assert profile.is_recommended is True
    assert profile.encoder_key == "faster"


def test_run_encode_plan_forwards_progress_and_results(tmp_path: Path) -> None:
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"data")
    job = EncodeJob(
        source=source,
        output=source,
        tmp_output=tmp_path / ".tmp_episode.mkv",
        crf=22,
        preset="faster",
        dry_run=False,
    )
    preparation = (
        prepare_encode_run.__annotations__
    )  # keep type checker quiet for local construction
    from mediashrink.gui_api import EncodePreparation

    prep = EncodePreparation(
        directory=tmp_path,
        ffmpeg=FFMPEG,
        ffprobe=FFPROBE,
        items=[_analysis_item(source)],
        duplicate_warnings=[],
        profile=None,
        jobs=[job],
        recommended_count=1,
        maybe_count=0,
        skip_count=0,
        selected_count=1,
        total_input_bytes=1000,
        selected_input_bytes=1000,
        selected_estimated_output_bytes=400,
        estimated_total_seconds=60.0,
        on_file_failure="retry",
        use_calibration=True,
        stage_messages=[],
    )
    progress_events: list[EncodeProgress] = []
    result = EncodeResult(
        job=job,
        skipped=False,
        skip_reason=None,
        success=True,
        input_size_bytes=1000,
        output_size_bytes=400,
        duration_seconds=5.0,
    )

    def fake_run_encode_loop(jobs, ffmpeg, ffprobe, display, **kwargs):
        with display.make_progress_bar() as progress:
            overall = progress.add_task("Overall (1 file(s))", total=1000, completed=0)
            file_task = progress.add_task("In progress: episode.mkv", total=100, completed=0)
            progress.update(
                file_task, description="In progress: episode.mkv", completed=50, total=100
            )
            progress.update(
                overall, completed=500, total=1000, completed_files=0, remaining_files=1
            )
        return [result]

    with patch("mediashrink.gui_api._run_encode_loop", side_effect=fake_run_encode_loop):
        results = run_encode_plan(prep, on_progress=progress_events.append)

    assert results == [result]
    assert progress_events
    assert progress_events[-1].overall_progress == 0.5


def test_prepare_encode_run_exposes_structured_planning_fields(tmp_path: Path) -> None:
    source = tmp_path / "movie.mp4"
    item = _analysis_item(source)
    planning = ProfilePlanningResult(
        candidate_items=[item],
        candidate_input_bytes=item.size_bytes,
        candidate_media_seconds=item.duration_seconds,
        sample_item=item,
        sample_duration=item.duration_seconds,
        preview_items=[item],
        available_hw=["amf"],
        benchmark_speeds={"amf": 6.0, "fast": 1.0, "faster": 2.0},
        observed_probe_failures={},
        profiles=[
            EncoderProfile(
                index=1,
                intent_label="Fast",
                name="Fast",
                encoder_key="faster",
                crf=22,
                sw_preset="faster",
                estimated_output_bytes=item.estimated_output_bytes,
                estimated_encode_seconds=900.0,
                quality_label="Very good",
                is_recommended=True,
                compatible_count=1,
                incompatible_count=0,
                grouped_incompatibilities={"attachment stream incompatibility": 1},
                why_choose="Fastest wait: AMF, but Fast covers all 1 file(s) while AMF likely leaves 1 for follow-up.",
            )
        ],
        active_calibration=None,
        size_error_by_preset={},
        stage_messages=[
            "Benchmarking profiles... 3/3",
            "Smoke-probing risky container/profile combinations... 2/2",
        ],
    )
    with (
        patch("mediashrink.gui_api.prepare_tools", return_value=(FFMPEG, FFPROBE)),
        patch("mediashrink.gui_api.scan_directory", return_value=[source]),
        patch("mediashrink.gui_api.analyze_files", return_value=[item]),
        patch("mediashrink.gui_api.prepare_profile_planning", return_value=planning),
        patch("mediashrink.gui_api.build_jobs", return_value=[]),
        patch("mediashrink.gui_api.estimate_analysis_encode_seconds", return_value=120.0),
    ):
        result = prepare_encode_run(directory=tmp_path)

    assert result.recommendation_reason is not None
    assert result.compatible_count == 1
    assert result.grouped_incompatibilities == {"attachment stream incompatibility": 1}
    assert result.stage_messages == planning.stage_messages
