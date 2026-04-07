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

    with (
        patch("mediashrink.gui_api.detect_available_encoders", return_value=[]),
        patch(
            "mediashrink.gui_api.benchmark_encoder",
            side_effect=lambda key, *_args: {"fast": 1.0, "faster": 2.0}.get(key, None),
        ),
    ):
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
