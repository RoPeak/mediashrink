from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mediashrink.analysis import (
    AnalysisItem,
    analyze_files,
    apply_duplicate_policy_to_items,
    estimate_analysis_encode_seconds,
    estimate_size_confidence,
    estimate_time_confidence,
)
from mediashrink.cli import _run_encode_loop
from mediashrink.models import EncodeResult
from mediashrink.platform_utils import (
    check_ffmpeg_available,
    find_ffmpeg,
    find_ffprobe,
)
from mediashrink.scanner import build_jobs, scan_directory
from mediashrink.wizard import (
    EncoderProfile,
    prepare_profile_planning,
)


@dataclass(frozen=True)
class EncodeProgress:
    current_file: str
    current_file_progress: float
    overall_progress: float
    completed_files: int
    remaining_files: int
    bytes_processed: int
    total_bytes: int
    heartbeat_state: str


@dataclass(frozen=True)
class EncodePreparation:
    directory: Path
    ffmpeg: Path
    ffprobe: Path
    items: list[AnalysisItem]
    duplicate_warnings: list[str]
    profile: EncoderProfile | None
    jobs: list
    recommended_count: int
    maybe_count: int
    skip_count: int
    selected_count: int
    total_input_bytes: int
    selected_input_bytes: int
    selected_estimated_output_bytes: int
    estimated_total_seconds: float | None
    on_file_failure: str
    use_calibration: bool
    size_confidence: str | None = None
    time_confidence: str | None = None
    compatible_count: int = 0
    incompatible_count: int = 0
    grouped_incompatibilities: dict[str, int] | None = None
    followup_manifest_path: Path | None = None
    recommendation_reason: str | None = None
    stage_messages: list[str] | None = None


def prepare_tools() -> tuple[Path, Path]:
    ok, message = check_ffmpeg_available()
    if not ok:
        raise RuntimeError(message)
    return find_ffmpeg(), find_ffprobe()


def auto_select_profile(
    items: list[AnalysisItem],
    *,
    ffmpeg: Path,
    ffprobe: Path,
    policy: str = "fastest-wall-clock",
    use_calibration: bool = True,
) -> EncoderProfile | None:
    planning = prepare_profile_planning(
        analysis_items=items,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
        console=None,
    )
    if planning is None:
        return None
    return next((profile for profile in planning.profiles if profile.is_recommended), None)


def prepare_encode_run(
    *,
    directory: Path,
    recursive: bool = True,
    overwrite: bool = True,
    no_skip: bool = False,
    policy: str = "fastest-wall-clock",
    on_file_failure: str = "retry",
    use_calibration: bool = True,
    duplicate_policy: str = "prefer-mkv",
    progress_callback: Callable[[tuple[int, int, Path]], None] | None = None,
) -> EncodePreparation:
    ffmpeg, ffprobe = prepare_tools()
    files = scan_directory(directory, recursive=recursive)
    if not files:
        return EncodePreparation(
            directory=directory,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            items=[],
            duplicate_warnings=[],
            profile=None,
            jobs=[],
            recommended_count=0,
            maybe_count=0,
            skip_count=0,
            selected_count=0,
            total_input_bytes=0,
            selected_input_bytes=0,
            selected_estimated_output_bytes=0,
            estimated_total_seconds=0.0,
            on_file_failure=on_file_failure,
            use_calibration=use_calibration,
            stage_messages=[],
        )
    items = analyze_files(
        files,
        ffprobe,
        progress_callback=lambda completed, total, path: progress_callback((completed, total, path))
        if progress_callback is not None
        else None,
        preset="fast",
        crf=20,
        use_calibration=use_calibration,
    )
    items, duplicate_warnings = apply_duplicate_policy_to_items(items, policy=duplicate_policy)
    planning = prepare_profile_planning(
        analysis_items=items,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
        console=None,
    )
    profile = (
        next((candidate for candidate in planning.profiles if candidate.is_recommended), None)
        if planning is not None
        else None
    )
    selected_items = [item for item in items if item.recommendation == "recommended"]
    if not selected_items:
        selected_items = [item for item in items if item.recommendation == "maybe"]
    if profile is None:
        jobs = []
        estimated_total_seconds = 0.0
        selected_input_bytes = 0
        selected_estimated_output_bytes = 0
    else:
        jobs = build_jobs(
            files=[item.source for item in selected_items],
            output_dir=None,
            overwrite=overwrite,
            crf=profile.crf,
            preset=profile.encoder_key,
            dry_run=False,
            ffprobe=ffprobe,
            no_skip=no_skip,
        )
        estimated_total_seconds = estimate_analysis_encode_seconds(
            selected_items,
            preset=profile.encoder_key,
            crf=profile.crf,
            ffmpeg=ffmpeg,
            known_speed=None,
            use_calibration=use_calibration,
            calibration_store=planning.active_calibration if planning is not None else None,
        )
        selected_input_bytes = sum(item.size_bytes for item in selected_items)
        selected_estimated_output_bytes = sum(
            item.estimated_output_bytes
            for item in selected_items
            if item.estimated_output_bytes > 0
        )
    return EncodePreparation(
        directory=directory,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        items=items,
        duplicate_warnings=duplicate_warnings,
        profile=profile,
        jobs=jobs,
        recommended_count=sum(1 for item in items if item.recommendation == "recommended"),
        maybe_count=sum(1 for item in items if item.recommendation == "maybe"),
        skip_count=sum(1 for item in items if item.recommendation == "skip"),
        selected_count=len(selected_items),
        total_input_bytes=sum(item.size_bytes for item in items),
        selected_input_bytes=selected_input_bytes,
        selected_estimated_output_bytes=selected_estimated_output_bytes,
        estimated_total_seconds=estimated_total_seconds,
        on_file_failure=on_file_failure,
        use_calibration=use_calibration,
        size_confidence=(
            estimate_size_confidence(
                selected_items, preset=profile.encoder_key, use_calibration=use_calibration
            )
            if profile is not None
            else None
        ),
        time_confidence=(
            estimate_time_confidence(
                selected_items,
                benchmarked_files=1 if planning is not None and planning.benchmark_speeds else 0,
                preset=profile.encoder_key,
                use_calibration=use_calibration,
            )
            if profile is not None
            else None
        ),
        compatible_count=profile.compatible_count if profile is not None else 0,
        incompatible_count=profile.incompatible_count if profile is not None else 0,
        grouped_incompatibilities=profile.grouped_incompatibilities
        if profile is not None
        else None,
        followup_manifest_path=None,
        recommendation_reason=profile.why_choose if profile is not None else None,
        stage_messages=planning.stage_messages if planning is not None else [],
    )


class _CallbackProgressBar:
    def __init__(self, on_update: Callable[[EncodeProgress], None] | None) -> None:
        self.on_update = on_update
        self._tasks: dict[int, dict[str, object]] = {}
        self._next_id = 1
        self._file_task_id: int | None = None
        self._overall_task_id: int | None = None

    def __enter__(self) -> "_CallbackProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def add_task(self, description: str, total: int = 0, **fields) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            **fields,
        }
        if "Overall" in description:
            self._overall_task_id = task_id
        else:
            self._file_task_id = task_id
        return task_id

    def update(self, task_id: int, **fields) -> None:
        task = self._tasks.setdefault(task_id, {})
        task.update(fields)
        if self.on_update is None or self._file_task_id is None or self._overall_task_id is None:
            return
        file_task = self._tasks.get(self._file_task_id, {})
        overall_task = self._tasks.get(self._overall_task_id, {})
        file_total = float(file_task.get("total", 0) or 0)
        file_completed = float(file_task.get("completed", 0) or 0)
        overall_total = int(overall_task.get("total", 0) or 0)
        overall_completed = int(overall_task.get("completed", 0) or 0)
        current_file_progress = 0.0 if file_total <= 0 else min(file_completed / file_total, 1.0)
        overall_progress = (
            0.0 if overall_total <= 0 else min(overall_completed / overall_total, 1.0)
        )
        description = str(file_task.get("description", ""))
        self.on_update(
            EncodeProgress(
                current_file=description,
                current_file_progress=current_file_progress,
                overall_progress=overall_progress,
                completed_files=int(overall_task.get("completed_files", 0) or 0),
                remaining_files=int(overall_task.get("remaining_files", 0) or 0),
                bytes_processed=overall_completed,
                total_bytes=overall_total,
                heartbeat_state=str(file_task.get("heartbeat_state", "active")),
            )
        )

    def remove_task(self, task_id: int) -> None:
        self._tasks.pop(task_id, None)


class _CallbackDisplay:
    def __init__(self, on_update: Callable[[EncodeProgress], None] | None) -> None:
        self.on_update = on_update

    def make_progress_bar(self) -> _CallbackProgressBar:
        return _CallbackProgressBar(self.on_update)

    def show_summary(self, results: list[EncodeResult], **kwargs) -> None:
        return None


def run_encode_plan(
    preparation: EncodePreparation,
    *,
    on_progress: Callable[[EncodeProgress], None] | None = None,
    on_file_failure: str = "retry",
    use_calibration: bool = True,
) -> list[EncodeResult]:
    if not preparation.jobs:
        return []
    display = _CallbackDisplay(on_progress)
    results = _run_encode_loop(
        preparation.jobs,
        preparation.ffmpeg,
        preparation.ffprobe,
        display,
        on_file_failure=on_file_failure,
        use_calibration=use_calibration,
    )
    return list(results)
