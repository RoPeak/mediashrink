from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from mediashrink.analysis import (
    AnalysisItem,
    analyze_files,
    apply_duplicate_policy_to_items,
    estimate_analysis_encode_seconds,
)
from mediashrink.cli import _run_encode_loop
from mediashrink.models import EncodeResult
from mediashrink.platform_utils import (
    check_ffmpeg_available,
    detect_device_labels,
    find_ffmpeg,
    find_ffprobe,
)
from mediashrink.scanner import build_jobs, scan_directory
from mediashrink.wizard import (
    EncoderProfile,
    benchmark_encoder,
    build_profiles,
    detect_available_encoders,
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
    candidate_items = [item for item in items if item.recommendation in {"recommended", "maybe"}]
    if not candidate_items:
        return None
    total_input_bytes = sum(item.size_bytes for item in candidate_items)
    total_media_seconds = sum(
        item.duration_seconds for item in candidate_items if item.duration_seconds > 0
    )
    if total_media_seconds <= 0:
        total_media_seconds = 3600.0 * len(candidate_items)
    sample_item = max(candidate_items, key=lambda item: item.size_bytes)
    sample_duration = sample_item.duration_seconds if sample_item.duration_seconds > 0 else 3600.0
    quiet_console = Console(quiet=True)
    available_hw = detect_available_encoders(
        ffmpeg,
        quiet_console,
        sample_file=sample_item.source,
        ffprobe=ffprobe,
    )
    benchmark_speeds: dict[str, float | None] = {}
    for key in list(available_hw) + ["fast", "faster"]:
        benchmark_speeds[key] = benchmark_encoder(
            key, sample_item.source, sample_duration, 20, ffmpeg
        )
    profiles = build_profiles(
        available_hw=available_hw,
        benchmark_speeds=benchmark_speeds,
        total_media_seconds=total_media_seconds,
        total_input_bytes=total_input_bytes,
        candidate_items=candidate_items,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
    )
    return next((profile for profile in profiles if profile.is_recommended), None)


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
    profile = auto_select_profile(
        items,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
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
