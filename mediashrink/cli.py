import json
import shutil
import sys
import threading
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from mediashrink.analysis import (
    apply_duplicate_policy_to_items,
    analyze_files,
    build_manifest,
    describe_size_confidence,
    describe_time_confidence,
    describe_estimate_calibration,
    describe_estimate_confidence,
    display_analysis_summary,
    estimate_analysis_confidence,
    estimate_analysis_encode_seconds,
    estimate_size_confidence,
    estimate_time_confidence,
    load_manifest,
    save_manifest,
    select_representative_items,
)
from mediashrink.calibration import (
    CalibrationRecord,
    FailureRecord,
    append_failure_record,
    append_success_record,
    bitrate_bucket,
    estimate_failure_rate,
    load_calibration_store,
    lookup_estimate,
    resolution_bucket,
)
from mediashrink.cleanup import cleanup_successful_results, eligible_cleanup_results
from mediashrink.encoder import (
    describe_container_incompatibility,
    describe_output_container_constraints,
    encode_file,
    encode_preview,
    get_duration_seconds,
    get_video_bitrate_kbps,
    get_video_resolution,
    is_hardware_preset,
    output_drops_subtitles,
    preflight_encode_job,
    source_has_attachment_streams,
    source_has_data_streams,
    source_has_subtitle_streams,
)
from mediashrink.models import (
    AnalysisItem,
    EncodeAttempt,
    EncodeJob,
    EncodeResult,
    SessionFileEntry,
    SessionManifest,
)
from mediashrink.platform_utils import check_ffmpeg_available, find_ffmpeg, find_ffprobe
from mediashrink.profiles import delete_profile, get_profile, list_all_profiles
from mediashrink.progress import EncodingDisplay
from mediashrink.scanner import (
    build_jobs,
    duplicate_policy_choices,
    scan_directory,
    supported_formats_label,
)
from mediashrink.session import (
    build_session,
    find_resumable_session,
    get_session_path,
    load_session,
    save_session,
    update_session_entry,
)

EXIT_SUCCESS = 0
EXIT_NO_FILES = 1
EXIT_ENCODE_FAILURES = 2
EXIT_USER_CANCELLED = 3
EXIT_FFMPEG_NOT_FOUND = 4
REPORT_VERSION = 1
STALL_WARNING_SECONDS = 90.0
STALL_POLL_SECONDS = 5.0
_FALLBACK_PRESET = "faster"
_FALLBACK_CRF = 22
_FALLBACK_PROGRESS_THRESHOLD = 5.0
_FALLBACK_DURATION_THRESHOLD = 60.0
_TRANSIENT_RETRY_PATTERNS: dict[str, tuple[str, ...]] = {
    "hardware_api_startup": (
        "device busy",
        "device lost",
        "resource busy",
        "cannot init",
        "cannot load libcuda",
        "initialization failed",
    ),
    "io_temporary": (
        "input/output error",
        "resource temporarily unavailable",
        "broken pipe",
        "temporarily unavailable",
        "i/o error",
    ),
    "timeout": (
        "timed out",
        "timeout",
    ),
}
_DISK_ESTIMATE_FALLBACK_RATIO = 0.70
_GB = 1024**3
_MB = 1024**2
_POLICY_CHOICES = {
    "fastest-wall-clock",
    "lowest-cpu",
    "best-compression",
    "highest-confidence",
}
_FAILURE_POLICY_CHOICES = {"skip", "retry", "stop"}
_RETRY_MODE_CHOICES = {"conservative", "balanced", "aggressive"}
_QUEUE_STRATEGY_CHOICES = {"original", "safe-first"}
_DUPLICATE_POLICY_CHOICES = set(duplicate_policy_choices())


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.2f} GB"
    return f"{size_bytes / _MB:.1f} MB"


def _format_resume_counts(session: SessionManifest) -> str:
    counts = {"success": 0, "pending": 0, "failed": 0, "skipped": 0}
    for entry in session.entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return (
        f"{counts['success']} done, "
        f"{counts['pending']} pending, "
        f"{counts['failed']} failed, "
        f"{counts['skipped']} skipped"
    )


def _print_safe_interrupt_guidance() -> None:
    console.print(
        "[dim]Safe to stop with Ctrl+C: completed files stay done, the current temp output is discarded, and you can resume the unfinished files later.[/dim]"
    )


def _normalize_failure_message(message: str | None) -> str | None:
    if not message:
        return message
    lowered = message.lower()
    if "no space left on device" in lowered:
        return "Disk full: no space left on device."
    if "permission denied" in lowered:
        return "Permission denied while reading input or writing output."
    if "could not write header" in lowered and "invalid argument" in lowered:
        return "Container or stream compatibility issue: FFmpeg could not write the output header."
    if "invalid argument" in lowered:
        return "FFmpeg rejected the current encoder or container settings."
    if "device type" in lowered or "unsupported" in lowered and "encoder" in lowered:
        return "Hardware encoder is not supported or is unavailable for this file."
    if "error opening output" in lowered:
        return "Output path is not writable or could not be created."
    return message


def _validate_policy(value: str) -> str:
    if value not in _POLICY_CHOICES:
        raise typer.BadParameter(f"invalid policy '{value}'")
    return value


def _validate_failure_policy(value: str) -> str:
    if value not in _FAILURE_POLICY_CHOICES:
        raise typer.BadParameter(f"invalid failure policy '{value}'")
    return value


def _validate_retry_mode(value: str) -> str:
    if value not in _RETRY_MODE_CHOICES:
        raise typer.BadParameter(f"invalid retry mode '{value}'")
    return value


def _validate_queue_strategy(value: str) -> str:
    if value not in _QUEUE_STRATEGY_CHOICES:
        raise typer.BadParameter(f"invalid queue strategy '{value}'")
    return value


def _validate_duplicate_policy(value: str) -> str:
    if value not in _DUPLICATE_POLICY_CHOICES:
        raise typer.BadParameter(f"invalid duplicate policy '{value}'")
    return value


def _resolve_runtime_settings(
    *,
    overnight: bool,
    policy: str,
    on_file_failure: str,
    verbose: bool,
    cleanup: bool,
    yes: bool,
    use_calibration: bool,
    retry_mode: str,
    queue_strategy: str,
) -> dict[str, object]:
    runtime_policy = _validate_policy(policy)
    runtime_failure_policy = _validate_failure_policy(on_file_failure)
    runtime_retry_mode = _validate_retry_mode(retry_mode)
    runtime_queue_strategy = _validate_queue_strategy(queue_strategy)
    if overnight:
        runtime_failure_policy = "skip"
        return {
            "policy": runtime_policy,
            "on_file_failure": runtime_failure_policy,
            "verbose": True,
            "cleanup": False if not cleanup else cleanup,
            "yes": True,
            "use_calibration": use_calibration,
            "retry_mode": "conservative",
            "queue_strategy": "safe-first",
        }
    return {
        "policy": runtime_policy,
        "on_file_failure": runtime_failure_policy,
        "verbose": verbose,
        "cleanup": cleanup,
        "yes": yes,
        "use_calibration": use_calibration,
        "retry_mode": runtime_retry_mode,
        "queue_strategy": runtime_queue_strategy,
    }


def _classify_result_status(result: EncodeResult) -> str:
    if result.success and not result.skipped:
        return "success"
    if result.skipped:
        if result.skip_reason and result.skip_reason.startswith("incompatible:"):
            return "skipped_incompatible"
        return "skipped_by_policy"
    return "failed_after_retries"


def _normalize_loop_result(
    loop_result: list[EncodeResult] | tuple[list[EncodeResult], bool],
) -> tuple[list[EncodeResult], bool]:
    if isinstance(loop_result, tuple):
        return loop_result
    return loop_result, bool(getattr(loop_result, "stopped_early", False))


def _apply_failure_diagnostics(result: EncodeResult) -> EncodeResult:
    if result.success or result.skipped:
        return result
    normalized = _normalize_failure_message(result.error_message)
    if normalized == result.error_message:
        return result
    result.raw_error_message = result.error_message
    result.error_message = normalized
    if result.attempts:
        last_attempt = result.attempts[-1]
        result.attempts[-1] = EncodeAttempt(
            preset=last_attempt.preset,
            crf=last_attempt.crf,
            success=last_attempt.success,
            duration_seconds=last_attempt.duration_seconds,
            progress_pct=last_attempt.progress_pct,
            error_message=normalized,
            retry_kind=last_attempt.retry_kind,
        )
    return result


def _estimate_required_free_space(jobs: list[EncodeJob], overwrite: bool) -> int:
    to_encode = [job for job in jobs if not job.skip and not job.dry_run]
    if not to_encode:
        return 0
    estimated_outputs = 0
    largest_buffer = 0
    for job in to_encode:
        source_size = job.source.stat().st_size
        estimated_output = (
            job.estimated_output_bytes
            if job.estimated_output_bytes > 0
            else int(source_size * _DISK_ESTIMATE_FALLBACK_RATIO)
        )
        estimated_outputs += estimated_output
        largest_buffer = max(largest_buffer, max(source_size, estimated_output))
    if overwrite:
        estimated_outputs = 0
    return estimated_outputs + largest_buffer


def _maybe_warn_low_disk_space(
    jobs: list[EncodeJob],
    output_root: Path,
    overwrite: bool,
    assume_yes: bool,
) -> None:
    required_bytes = _estimate_required_free_space(jobs, overwrite)
    if required_bytes <= 0:
        return
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes >= required_bytes:
        return
    console.print(
        f"[yellow]Low disk space warning:[/yellow] estimated free space {_fmt_size(free_bytes)} vs roughly {_fmt_size(required_bytes)} required for this batch."
    )
    console.print(
        "[dim]This is a warning only. Side-by-side encodes and temporary outputs may need more space than is currently free.[/dim]"
    )
    if not assume_yes and not typer.confirm("Continue anyway?", default=False):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)


def _collect_preflight_warnings(jobs: list[EncodeJob], ffprobe: Path) -> list[str]:
    subtitle_examples: list[str] = []
    subtitle_count = 0
    attachment_examples: list[str] = []
    attachment_count = 0
    data_examples: list[str] = []
    data_count = 0
    audio_examples: list[str] = []
    for job in jobs:
        if job.skip:
            continue
        if output_drops_subtitles(job.output) and source_has_subtitle_streams(job.source, ffprobe):
            subtitle_count += 1
            if len(subtitle_examples) < 3:
                subtitle_examples.append(job.source.name)
        for note in describe_output_container_constraints(job.source, job.output, ffprobe):
            if note.startswith("attachment") and len(attachment_examples) < 3:
                attachment_count += 1
                attachment_examples.append(job.source.name)
            elif note.startswith("auxiliary data") and len(data_examples) < 3:
                data_count += 1
                data_examples.append(job.source.name)
            elif note.startswith("audio copy may fail") and len(audio_examples) < 3:
                audio_examples.append(f"{job.source.name}: {note}")

    warnings: list[str] = []
    if subtitle_count:
        container = next(
            (
                job.output.suffix.lower()
                for job in jobs
                if not job.skip and output_drops_subtitles(job.output)
            ),
            ".mp4/.m4v",
        )
        warning = (
            f"Subtitle warning: {subtitle_count} file(s) contain subtitle streams that will be "
            f"dropped because {container} outputs use '-sn' for compatibility."
        )
        if subtitle_examples:
            warning += f" Examples: {', '.join(subtitle_examples)}."
        warnings.append(warning)
    if attachment_count:
        warning = (
            f"Attachment warning: {attachment_count} file(s) contain attachment streams that will "
            "be dropped for MP4/M4V compatibility."
        )
        if attachment_examples:
            warning += f" Examples: {', '.join(attachment_examples)}."
        warnings.append(warning)
    if data_count:
        warning = (
            f"Auxiliary data warning: {data_count} file(s) contain data streams that will be "
            "dropped for MP4/M4V compatibility."
        )
        if data_examples:
            warning += f" Examples: {', '.join(data_examples)}."
        warnings.append(warning)
    if audio_examples:
        warnings.append(
            "Audio compatibility warning: some copied audio streams may fail in MP4/M4V outputs. "
            + " Examples: "
            + "; ".join(audio_examples)
            + "."
        )
    return warnings


def _classify_incompatible_reason(
    message: str | None,
    job: EncodeJob,
    *,
    ffprobe: Path | None = None,
) -> str:
    lowered = (message or "").lower()
    if "attachment" in lowered:
        return "attachment streams are not supported by the chosen output container"
    if "mov_text" in lowered or "subtitle" in lowered or "subrip" in lowered or "ass" in lowered:
        return "subtitle codec is not supported by the chosen output container"
    if "data" in lowered or "bin_data" in lowered:
        return "auxiliary data streams are not supported by the chosen output container"
    if "audio" in lowered and "not currently supported in container" in lowered:
        return "audio codec is not supported by the chosen output container"
    if "could not write header" in lowered and "invalid argument" in lowered:
        if ffprobe is not None:
            classified = describe_container_incompatibility(job.source, job.output, ffprobe)
            if classified is not None:
                return classified
        return "unsupported container/stream combination"
    if "invalid argument" in lowered and is_hardware_preset(job.preset):
        return "encoder/container combination appears unreliable on this device"
    if job.output.suffix.lower() in {".mp4", ".m4v"}:
        if ffprobe is not None:
            classified = describe_container_incompatibility(job.source, job.output, ffprobe)
            if classified is not None:
                return classified
        return "output container cannot safely carry one or more copied streams"
    if is_hardware_preset(job.preset) and "invalid argument" in lowered:
        return "encoder/container combination appears unreliable on this device"
    return "incompatible stream layout"


def _preflight_candidates(jobs: list[EncodeJob]) -> list[EncodeJob]:
    return [
        job
        for job in jobs
        if not job.skip and (job.output.suffix.lower() != ".mkv" or is_hardware_preset(job.preset))
    ]


def _apply_batch_preflight_policy(
    jobs: list[EncodeJob],
    *,
    ffmpeg: Path,
    ffprobe: Path,
    on_file_failure: str,
) -> tuple[list[EncodeJob], list[str], list[str]]:
    candidates = _preflight_candidates(jobs)
    if not candidates:
        return jobs, [], []

    warnings: list[str] = []
    skipped: list[str] = []
    stop_errors: list[str] = []
    for job in candidates:
        result = preflight_encode_job(
            job.source,
            ffmpeg,
            ffprobe,
            crf=job.crf,
            preset=job.preset,
        )
        if result.success:
            continue
        reason = _classify_incompatible_reason(result.error_message, job, ffprobe=ffprobe)
        detail = f"{job.source.name}: {reason}"
        if on_file_failure == "skip":
            job.skip = True
            job.skip_reason = f"incompatible: {reason}"
            skipped.append(detail)
        elif on_file_failure == "stop":
            stop_errors.append(detail)
        else:
            warnings.append(detail)
    return jobs, warnings, skipped or stop_errors


def _group_incompatibility_details(details: list[str]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for detail in details:
        name, _, reason = detail.partition(": ")
        label = reason or detail
        grouped.setdefault(label, []).append(name or detail)
    summaries: list[str] = []
    for reason, names in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        examples = ", ".join(names[:3])
        summary = f"{len(names)} file(s): {reason}"
        if examples:
            summary += f" ({examples})"
        summaries.append(summary)
    return summaries


def _print_grouped_preflight_details(details: list[str], *, style: str, prefix: str) -> None:
    for summary in _group_incompatibility_details(details):
        console.print(f"[{style}]{prefix}[/]{summary}")


def _followup_notes_for_incompatible_details(details: list[str]) -> list[str]:
    if not details:
        return []
    notes = ["Automatically generated from files left out by preflight compatibility checks."]
    notes.extend(_group_incompatibility_details(details))
    notes.append("Suggested retry: prefer MKV outputs or rerun with --policy highest-confidence.")
    return notes


def _write_followup_manifest_for_jobs(
    *,
    directory: Path,
    recursive: bool,
    preset: str,
    crf: int,
    jobs: list[EncodeJob],
    ffprobe: Path,
    duplicate_policy: str | None,
    details: list[str],
) -> Path | None:
    sources = [job.source for job in jobs if job.source.exists()]
    if not sources:
        return None
    items = analyze_files(sources, ffprobe, preset=preset, crf=crf)
    manifest = build_manifest(
        directory=directory,
        recursive=recursive,
        preset=preset,
        crf=crf,
        profile_name=None,
        estimated_total_encode_seconds=None,
        estimate_confidence=None,
        size_confidence=None,
        size_confidence_detail=None,
        time_confidence=None,
        time_confidence_detail=None,
        duplicate_policy=duplicate_policy,
        recommended_only=False,
        notes=_followup_notes_for_incompatible_details(details),
        items=items,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"mediashrink_followup_{timestamp}.json"
    save_manifest(manifest, path)
    return path


def _print_preflight_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")


def _reconcile_session_with_jobs(session: SessionManifest, jobs: list[EncodeJob]) -> None:
    known_sources = {entry.source for entry in session.entries}
    for job in jobs:
        if str(job.source) in known_sources:
            continue
        session.entries.append(
            SessionFileEntry(
                source=str(job.source),
                status="skipped" if job.skip else "pending",
                output=str(job.output) if not job.skip else None,
            )
        )


def _report_output_dir(base_dir: Path, output_dir: Path | None, log_path: Path | None) -> Path:
    if log_path is not None:
        return log_path.parent
    return output_dir or base_dir


def _find_latest_report(path: Path) -> Path | None:
    if path.is_file():
        return path if path.suffix.lower() == ".json" else None
    candidates = sorted(path.glob("mediashrink_report_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _load_report_payload(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("report root must be an object")
    return raw


def _review_guidance(payload: dict[str, object]) -> list[str]:
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        return []
    guidance: list[str] = []
    failed = int(totals.get("failed", 0))
    skipped_incompatible = int(totals.get("skipped_incompatible", 0))
    mode = payload.get("mode") if isinstance(payload.get("mode"), str) else "encode"
    directory = payload.get("directory") if isinstance(payload.get("directory"), str) else None
    if failed:
        if mode in {"encode", "resume", "overnight"} and directory:
            guidance.append(
                f'Retry unresolved files with: mediashrink resume "{directory}" --policy highest-confidence --retry-mode aggressive'
            )
        else:
            guidance.append(
                "Review the failed files and rerun them with a safer policy or software preset."
            )
    if skipped_incompatible and directory:
        guidance.append(
            f"Skipped incompatible files likely need MKV outputs or a different preset path for: {directory}"
        )
    followup_manifest = payload.get("split_followup_manifest")
    if isinstance(followup_manifest, str) and followup_manifest:
        guidance.append(f"Review the follow-up manifest for left-out files: {followup_manifest}")
    if not guidance:
        guidance.append("No manual follow-up is suggested from this report.")
    return guidance


def _record_success_calibration(result: EncodeResult, ffprobe: Path) -> None:
    if not result.success or result.skipped or result.job.dry_run:
        return
    width, height = get_video_resolution(result.job.source, ffprobe)
    bitrate_kbps = get_video_bitrate_kbps(result.job.source, ffprobe)
    duration_seconds = (
        result.media_duration_seconds
        if result.media_duration_seconds > 0
        else get_duration_seconds(result.job.source, ffprobe)
    )
    codec = result.job.source_codec or "unknown"
    effective_speed = (
        duration_seconds / result.duration_seconds if result.duration_seconds > 0 else 0.0
    )
    active_store = load_calibration_store()
    width, height = get_video_resolution(result.job.source, ffprobe)
    lookup = lookup_estimate(
        active_store,
        codec=codec,
        resolution=resolution_bucket(width, height),
        bitrate=bitrate_bucket(bitrate_kbps),
        preset=result.job.preset,
        container=result.job.output.suffix.lower() or ".mkv",
    )
    predicted_output_ratio = (
        result.job.estimated_output_bytes / result.input_size_bytes
        if result.input_size_bytes > 0 and result.job.estimated_output_bytes > 0
        else None
    )
    append_success_record(
        CalibrationRecord(
            codec=codec,
            container=result.job.output.suffix.lower() or ".mkv",
            resolution_bucket=resolution_bucket(width, height),
            bitrate_bucket=bitrate_bucket(bitrate_kbps),
            preset=result.job.preset,
            preset_family="hardware" if is_hardware_preset(result.job.preset) else "software",
            crf=result.job.crf,
            input_bytes=result.input_size_bytes,
            output_bytes=result.output_size_bytes,
            duration_seconds=duration_seconds,
            wall_seconds=result.duration_seconds,
            effective_speed=effective_speed,
            fallback_used=result.fallback_used,
            retry_used=result.retry_count > 0,
            predicted_output_ratio=predicted_output_ratio,
            predicted_speed=lookup.speed if lookup is not None else None,
        )
    )


def _record_failure_calibration(result: EncodeResult) -> None:
    if result.success or result.skipped or result.job.dry_run:
        return
    append_failure_record(
        FailureRecord(
            encoder="hardware" if is_hardware_preset(result.job.preset) else result.job.preset,
            container=result.job.output.suffix.lower() or ".mkv",
            stage="encode",
            reason=result.error_message or "unknown",
        )
    )


def _job_risk_score(job: EncodeJob) -> tuple[int, int]:
    container_risk = 1 if job.output.suffix.lower() in {".mp4", ".m4v"} else 0
    encoder_risk = 1 if is_hardware_preset(job.preset) else 0
    return container_risk, encoder_risk


def _prioritize_jobs(jobs: list[EncodeJob], queue_strategy: str) -> list[EncodeJob]:
    if queue_strategy != "safe-first":
        return jobs
    skipped = [job for job in jobs if job.skip]
    active = [job for job in jobs if not job.skip]
    active.sort(
        key=lambda job: (
            _job_risk_score(job),
            job.estimated_output_bytes
            if job.estimated_output_bytes > 0
            else job.source.stat().st_size,
            job.source.name.lower(),
        )
    )
    return skipped + active


def _results_totals(results: list[EncodeResult]) -> dict[str, int | float]:
    succeeded = [result for result in results if result.success and not result.skipped]
    failed = [result for result in results if not result.success and not result.skipped]
    skipped = [result for result in results if result.skipped]
    input_total = sum(result.input_size_bytes for result in succeeded)
    output_total = sum(result.output_size_bytes for result in succeeded)
    return {
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": len(skipped),
        "skipped_incompatible": sum(
            1 for result in skipped if _classify_result_status(result) == "skipped_incompatible"
        ),
        "skipped_by_policy": sum(
            1 for result in skipped if _classify_result_status(result) == "skipped_by_policy"
        ),
        "input_bytes": input_total,
        "output_bytes": output_total,
        "saved_bytes": input_total - output_total,
    }


def _estimate_miss_summary(results: list[EncodeResult]) -> str | None:
    successful = [result for result in results if result.success and not result.skipped]
    estimated_inputs = [
        result.job.estimated_output_bytes
        for result in successful
        if result.job.estimated_output_bytes > 0 and result.input_size_bytes > 0
    ]
    if not successful or not estimated_inputs:
        return None
    estimated_output = sum(
        result.job.estimated_output_bytes
        for result in successful
        if result.job.estimated_output_bytes > 0
    )
    actual_output = sum(result.output_size_bytes for result in successful)
    if estimated_output <= 0:
        return None
    delta_pct = ((actual_output - estimated_output) / estimated_output) * 100
    if abs(delta_pct) < 5:
        return "Actual output size stayed close to the estimate."
    direction = "larger" if delta_pct > 0 else "smaller"
    return f"Actual output size was {abs(delta_pct):.0f}% {direction} than estimated across successful files."


def _group_incompatible_results(results: list[EncodeResult]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for result in results:
        if _classify_result_status(result) != "skipped_incompatible":
            continue
        reason = result.skip_reason or "incompatible"
        if reason.startswith("incompatible: "):
            reason = reason[len("incompatible: ") :]
        entry = grouped.setdefault(reason, {"reason": reason, "count": 0, "examples": []})
        entry["count"] = int(entry["count"]) + 1
        examples = entry["examples"]
        if isinstance(examples, list) and len(examples) < 3:
            examples.append(result.job.source.name)
    return sorted(
        grouped.values(),
        key=lambda item: (-int(item["count"]), str(item["reason"])),
    )


def _write_batch_reports(
    *,
    mode: str,
    base_dir: Path,
    output_dir: Path | None,
    manifest_path: Path | None,
    preset: str,
    crf: int,
    overwrite: bool,
    cleanup_requested: bool,
    resumed_from_session: bool,
    session_path: Path | None,
    started_at: str,
    finished_at: str,
    results: list[EncodeResult],
    cleaned_paths: list[Path],
    log_path: Path | None,
    warnings: list[str] | None = None,
    policy: str | None = None,
    on_file_failure: str | None = None,
    retry_mode: str | None = None,
    queue_strategy: str | None = None,
    size_confidence: str | None = None,
    time_confidence: str | None = None,
    split_followup_manifest: Path | None = None,
    estimate_miss_summary: str | None = None,
) -> tuple[Path, Path]:
    report_dir = _report_output_dir(base_dir, output_dir, log_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"mediashrink_report_{timestamp}.json"
    text_path = report_dir / f"mediashrink_report_{timestamp}.txt"
    totals = _results_totals(results)
    grouped_incompatibilities = _group_incompatible_results(results)
    files = []
    for result in results:
        status = _classify_result_status(result)
        files.append(
            {
                "source": str(result.job.source),
                "output": str(result.job.output),
                "status": status,
                "skipped_reason": result.skip_reason,
                "error_message": result.error_message,
                "raw_error_message": result.raw_error_message,
                "duration_seconds": result.duration_seconds,
                "input_bytes": result.input_size_bytes,
                "output_bytes": result.output_size_bytes,
                "reduction_pct": round(result.size_reduction_pct, 1) if result.success else 0.0,
                "fallback_used": result.fallback_used,
                "retry_kind": result.retry_kind,
                "retry_count": result.retry_count,
                "first_error": result.first_error,
                "last_error": result.last_error,
                "cleanup_result": (
                    "original removed; compressed file restored to original name"
                    if result.job.source in cleaned_paths
                    else (
                        "compressed output kept side-by-side"
                        if result.success and result.job.output != result.job.source
                        else None
                    )
                ),
                "attempts": [attempt.to_dict() for attempt in result.attempts],
            }
        )
    payload = {
        "report_version": REPORT_VERSION,
        "mode": mode,
        "directory": str(base_dir),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "preset": preset,
        "crf": crf,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "overwrite": overwrite,
        "cleanup_requested": cleanup_requested,
        "cleanup_completed": len(cleaned_paths),
        "cleaned_paths": [str(path) for path in cleaned_paths],
        "started_at": started_at,
        "finished_at": finished_at,
        "resumed_from_session": resumed_from_session,
        "session_path": str(session_path) if session_path is not None else None,
        "totals": totals,
        "warnings": warnings or [],
        "policy": policy,
        "on_file_failure": on_file_failure,
        "retry_mode": retry_mode,
        "queue_strategy": queue_strategy,
        "size_confidence": size_confidence,
        "time_confidence": time_confidence,
        "split_followup_manifest": (
            str(split_followup_manifest) if split_followup_manifest is not None else None
        ),
        "estimate_miss_summary": estimate_miss_summary,
        "grouped_incompatibilities": grouped_incompatibilities,
        "files": files,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "mediashrink batch report",
        f"Mode: {mode}",
        f"Directory: {base_dir}",
        f"Manifest: {manifest_path}" if manifest_path is not None else "Manifest: -",
        f"Preset / CRF: {preset} / {crf}",
        f"Output dir: {output_dir}" if output_dir is not None else "Output dir: alongside sources",
        f"Overwrite: {'yes' if overwrite else 'no'}",
        f"Cleanup requested: {'yes' if cleanup_requested else 'no'}",
        f"Cleanup completed: {len(cleaned_paths)}",
        f"Resumed from session: {'yes' if resumed_from_session else 'no'}",
        f"Session path: {session_path}" if session_path is not None else "Session path: -",
        f"Policy: {policy}" if policy is not None else "Policy: -",
        f"On file failure: {on_file_failure}"
        if on_file_failure is not None
        else "On file failure: -",
        f"Retry mode: {retry_mode}" if retry_mode is not None else "Retry mode: -",
        f"Queue strategy: {queue_strategy}" if queue_strategy is not None else "Queue strategy: -",
        f"Size confidence: {size_confidence}"
        if size_confidence is not None
        else "Size confidence: -",
        f"Time confidence: {time_confidence}"
        if time_confidence is not None
        else "Time confidence: -",
        (
            f"Split follow-up manifest: {split_followup_manifest}"
            if split_followup_manifest is not None
            else "Split follow-up manifest: -"
        ),
        f"Started: {started_at}",
        f"Finished: {finished_at}",
        "",
        "Warnings",
    ]
    if warnings:
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("  - none")
    lines.extend(
        [
            "",
            "Totals",
            f"  Succeeded: {totals['succeeded']}",
            f"  Failed: {totals['failed']}",
            f"  Skipped: {totals['skipped']}",
            f"  Skipped incompatible: {totals['skipped_incompatible']}",
            f"  Skipped by policy: {totals['skipped_by_policy']}",
            f"  Input: {_fmt_size(int(totals['input_bytes']))}",
            f"  Output: {_fmt_size(int(totals['output_bytes']))}",
            f"  Saved: {_fmt_size(int(totals['saved_bytes']))}",
        ]
    )
    if grouped_incompatibilities:
        lines.extend(["", "Grouped incompatibilities"])
        for entry in grouped_incompatibilities:
            examples = entry.get("examples") or []
            example_text = (
                f" (examples: {', '.join(str(name) for name in examples)})"
                if isinstance(examples, list) and examples
                else ""
            )
            lines.append(f"  - {entry['count']}: {entry['reason']}{example_text}")
    lines.extend(["", "Files"])
    for result in results:
        status = _classify_result_status(result).upper()
        lines.append(f"- {result.job.source.name}: {status}")
        if result.skip_reason:
            lines.append(f"  Reason: {result.skip_reason}")
        if result.error_message:
            lines.append(f"  Error: {result.error_message}")
        if result.retry_count:
            lines.append(
                f"  Retries: {result.retry_count}"
                + (f" ({result.retry_kind})" if result.retry_kind else "")
            )
        if result.first_error and result.first_error != result.error_message:
            lines.append(f"  First error: {result.first_error}")
        if result.last_error and result.last_error != result.error_message:
            lines.append(f"  Last error: {result.last_error}")
        if result.fallback_used:
            lines.append("  Fallback: hardware retry switched to libx265 faster / CRF 22")
        cleanup_result = (
            "original removed; compressed file restored to original name"
            if result.job.source in cleaned_paths
            else (
                "compressed output kept side-by-side"
                if result.success and result.job.output != result.job.source
                else None
            )
        )
        if cleanup_result:
            lines.append(f"  Cleanup: {cleanup_result}")
        if result.attempts:
            for index, attempt in enumerate(result.attempts, start=1):
                status = "ok" if attempt.success else "failed"
                extra = f", retry={attempt.retry_kind}" if attempt.retry_kind else ""
                lines.append(
                    f"  Attempt {index}: {attempt.preset} / CRF {attempt.crf} - {status}{extra}"
                )
    if estimate_miss_summary:
        lines.extend(["", "Estimate miss summary", f"  - {estimate_miss_summary}"])
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def _clone_job_with_settings(job: EncodeJob, *, preset: str, crf: int) -> EncodeJob:
    return EncodeJob(
        source=job.source,
        output=job.output,
        tmp_output=job.tmp_output,
        crf=crf,
        preset=preset,
        dry_run=job.dry_run,
        skip=job.skip,
        skip_reason=job.skip_reason,
        source_codec=job.source_codec,
        estimated_output_bytes=job.estimated_output_bytes,
    )


def _should_retry_hardware_failure(result: EncodeResult) -> bool:
    if result.success or result.skipped or not is_hardware_preset(result.job.preset):
        return False
    if result.job.preset == _FALLBACK_PRESET:
        return False
    progress_pct = max((attempt.progress_pct for attempt in result.attempts), default=0.0)
    return (
        progress_pct <= _FALLBACK_PROGRESS_THRESHOLD
        or result.duration_seconds <= _FALLBACK_DURATION_THRESHOLD
    )


def _retry_mode_allows_hardware_fallback(result: EncodeResult, retry_mode: str) -> bool:
    if retry_mode == "conservative":
        return False
    if retry_mode == "aggressive":
        return _should_retry_hardware_failure(result)
    return _should_retry_hardware_failure(result)


def _classify_transient_failure(result: EncodeResult) -> str | None:
    if result.success or result.skipped:
        return None
    messages = [result.error_message or "", result.raw_error_message or "", result.last_error or ""]
    haystack = "\n".join(messages).lower()
    if not haystack.strip():
        return None
    for retry_kind, patterns in _TRANSIENT_RETRY_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return retry_kind
    return None


def _should_retry_transient_failure(result: EncodeResult) -> str | None:
    retry_kind = _classify_transient_failure(result)
    if retry_kind is None:
        return None
    if result.retry_count >= 1:
        return None
    if is_hardware_preset(result.job.preset):
        return retry_kind
    if retry_kind == "io_temporary":
        return retry_kind
    return None


def _retry_mode_transient_kind(result: EncodeResult, retry_mode: str) -> str | None:
    retry_kind = _classify_transient_failure(result)
    if retry_kind is None or result.retry_count >= 1:
        return None
    if retry_mode == "conservative":
        if retry_kind == "hardware_api_startup" and is_hardware_preset(result.job.preset):
            return retry_kind
        return retry_kind if retry_kind == "io_temporary" else None
    if retry_mode == "aggressive":
        return retry_kind
    return _should_retry_transient_failure(result)


def _attempts_with_retry_kind(
    attempts: list[EncodeAttempt], retry_kind: str
) -> list[EncodeAttempt]:
    if not attempts:
        return attempts
    patched = list(attempts)
    last = patched[-1]
    patched[-1] = EncodeAttempt(
        preset=last.preset,
        crf=last.crf,
        success=last.success,
        duration_seconds=last.duration_seconds,
        progress_pct=last.progress_pct,
        error_message=last.error_message,
        retry_kind=retry_kind,
    )
    return patched


def _record_cleanup_results(
    session: SessionManifest | None,
    session_path: Path | None,
    results: list[EncodeResult],
    cleaned_paths: list[Path],
) -> None:
    if session is None or session_path is None:
        return
    cleaned_sources = {path for path in cleaned_paths}
    for result in results:
        if not result.success or result.skipped:
            continue
        cleanup_result = (
            "original removed; compressed file restored to original name"
            if result.job.source in cleaned_sources
            else (
                "compressed output kept side-by-side"
                if result.job.output != result.job.source
                else "output replaced original in place"
            )
        )
        update_session_entry(
            session,
            source=result.job.source,
            status="success",
            cleanup_result=cleanup_result,
        )
    save_session(session, session_path)


def _results_to_json(results: list[EncodeResult], exit_code: int) -> str:
    files = []
    total_saved = 0
    for r in results:
        if r.skipped:
            status = "skipped"
        elif r.success:
            status = "success"
        else:
            status = "failed"
        saved = r.size_reduction_bytes if r.success else 0
        total_saved += saved
        files.append(
            {
                "source": str(r.job.source),
                "status": status,
                "input_bytes": r.input_size_bytes,
                "output_bytes": r.output_size_bytes,
                "reduction_pct": round(r.size_reduction_pct, 1) if r.success else 0.0,
            }
        )
    return json.dumps({"exit_code": exit_code, "files": files, "total_saved_bytes": total_saved})


def _resume_summary_counts(session: SessionManifest | None) -> tuple[int, int]:
    if session is None:
        return 0, 0
    completed = sum(1 for entry in session.entries if entry.status == "success")
    skipped = sum(1 for entry in session.entries if entry.status == "skipped")
    return completed, skipped


def _build_skipped_results(jobs: list[EncodeJob]) -> list[EncodeResult]:
    results: list[EncodeResult] = []
    for job in jobs:
        if not job.skip:
            continue
        input_size = job.source.stat().st_size if job.source.exists() else 0
        results.append(
            EncodeResult(
                job=job,
                skipped=True,
                skip_reason=job.skip_reason,
                success=False,
                input_size_bytes=input_size,
                output_size_bytes=0,
                duration_seconds=0.0,
            )
        )
    return results


def _has_incompatible_skips(jobs: list[EncodeJob]) -> bool:
    return any((job.skip_reason or "").startswith("incompatible:") for job in jobs if job.skip)


class DefaultCommandGroup(TyperGroup):
    """Route bare `mediashrink ...` invocations to the hidden encode command."""

    default_command_name = "encode"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in {"--help", "-h"}:
            args.insert(0, self.default_command_name)
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="mediashrink",
    help=f"Re-encode supported video files ({supported_formats_label()}) to H.265/HEVC to reduce file size.",
    add_completion=False,
    cls=DefaultCommandGroup,
)
profiles_app = typer.Typer(help="Manage saved encoding profiles.")
app.add_typer(profiles_app, name="profiles")

console = Console()


class EncodeLoopResults(list):
    def __init__(self, results: list[EncodeResult], *, stopped_early: bool = False) -> None:
        super().__init__(results)
        self.stopped_early = stopped_early


def _run_encode_loop(
    jobs: list[EncodeJob],
    ffmpeg: Path,
    ffprobe: Path,
    display: EncodingDisplay,
    session: SessionManifest | None = None,
    session_path: Path | None = None,
    log_path: Path | None = None,
    stall_warning_seconds: float | None = None,
    allow_hardware_fallback: bool = True,
    resumed_from_session: bool = False,
    previously_completed: int = 0,
    previously_skipped: int = 0,
    on_file_failure: str = "retry",
    use_calibration: bool = True,
    retry_mode: str = "balanced",
) -> list[EncodeResult]:
    if stall_warning_seconds is None:
        stall_warning_seconds = STALL_WARNING_SECONDS
    to_encode = [job for job in jobs if not job.skip]
    total_bytes = sum(job.source.stat().st_size for job in to_encode)
    results = []
    bytes_done = 0
    stopped_early = False

    with display.make_progress_bar() as progress:
        overall_task = progress.add_task(
            f"[cyan]Overall ({len(to_encode)} file(s))",
            total=total_bytes,
        )
        file_task = progress.add_task("", total=1)

        for job in jobs:
            filename = job.source.name
            file_size = job.source.stat().st_size
            files_done = len(results)
            files_remaining = max(len(jobs) - files_done - 1, 0)
            # Use input size as the task total so DownloadColumn shows GB-scale numbers
            task_total = max(file_size, 1)
            progress.update(
                file_task,
                description=f"[dim]In progress:[/dim] [white]{filename}",
                completed=0,
                total=task_total,
                completed_files=files_done,
                remaining_files=files_remaining,
                last_update_at=time.monotonic(),
                stall_warning_seconds=stall_warning_seconds,
                heartbeat_state="active",
                eta_confident=False,
            )
            started_at = _now_iso()
            stall_state = {
                "last_update": time.monotonic(),
                "last_percent": 0.0,
                "warned": False,
                "last_output_growth": time.monotonic(),
                "last_output_size": 0,
                "heartbeat_state": "active",
            }
            stall_stop = threading.Event()

            if session is not None and session_path is not None:
                update_session_entry(
                    session,
                    source=job.source,
                    status="in_progress" if not job.skip else "skipped",
                    encoder=job.preset,
                    started_at=started_at,
                    last_progress_pct=0.0 if not job.skip else None,
                    last_progress_at=started_at if not job.skip else None,
                )
                save_session(session, session_path)

            def watch_for_stall() -> None:
                grace_multiplier = (
                    2.0 if is_hardware_preset(job.preset) or file_size >= 4 * _GB else 1.0
                )
                while not stall_stop.wait(STALL_POLL_SECONDS):
                    try:
                        current_size = (
                            job.tmp_output.stat().st_size if job.tmp_output.exists() else 0
                        )
                    except OSError:
                        current_size = 0
                    if current_size > stall_state["last_output_size"]:
                        stall_state["last_output_size"] = current_size
                        stall_state["last_output_growth"] = time.monotonic()
                    idle_for = time.monotonic() - stall_state["last_update"]
                    since_growth = time.monotonic() - stall_state["last_output_growth"]
                    if idle_for < min(15.0, stall_warning_seconds / 2):
                        state = "active"
                    elif since_growth < stall_warning_seconds * grace_multiplier:
                        state = "quiet"
                    else:
                        state = "stalled"
                    stall_state["heartbeat_state"] = state
                    progress.update(
                        file_task,
                        heartbeat_state=state,
                        last_update_at=stall_state["last_update"],
                        stall_warning_seconds=stall_warning_seconds,
                    )
                    if stall_state["warned"] or state != "stalled":
                        continue
                    if idle_for >= stall_warning_seconds * grace_multiplier:
                        console.print(
                            f"\n[yellow]No progress update from FFmpeg for about {int(idle_for)}s while encoding {filename}.[/yellow]"
                        )
                        console.print(
                            "[dim]FFmpeg still appears to be alive, but progress looks sparse. If you stop now, completed files stay done and the next run can resume from the session file.[/dim]"
                        )
                        stall_state["warned"] = True

            stall_thread = None
            if not job.skip:
                stall_thread = threading.Thread(target=watch_for_stall, daemon=True)
                stall_thread.start()

            def make_callback(ft=file_task, fb=file_size):
                def callback(pct: float) -> None:
                    stall_state["last_update"] = time.monotonic()
                    stall_state["last_percent"] = pct
                    progress.update(
                        ft,
                        completed=fb * pct / 100,
                        completed_files=files_done,
                        remaining_files=files_remaining,
                        last_update_at=stall_state["last_update"],
                        stall_warning_seconds=stall_warning_seconds,
                        heartbeat_state="active",
                        eta_confident=pct >= 5.0,
                    )
                    if session is not None and session_path is not None:
                        update_session_entry(
                            session,
                            source=job.source,
                            status="in_progress",
                            last_progress_pct=pct,
                            last_progress_at=_now_iso(),
                        )
                        save_session(session, session_path)

                return callback

            try:
                result = encode_file(
                    job,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    progress_callback=make_callback() if not job.skip else None,
                    log_path=log_path,
                )
                result = _apply_failure_diagnostics(result)
                transient_retry_kind = _retry_mode_transient_kind(result, retry_mode)
                if transient_retry_kind is not None:
                    console.print(
                        f"[yellow]Retrying {filename} after transient {transient_retry_kind.replace('_', ' ')} failure[/yellow]"
                    )
                    retried_result = encode_file(
                        job,
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        progress_callback=make_callback(),
                        log_path=log_path,
                    )
                    retried_result = _apply_failure_diagnostics(retried_result)
                    attempts = _attempts_with_retry_kind(
                        result.attempts, transient_retry_kind
                    ) + list(retried_result.attempts)
                    result = EncodeResult(
                        job=retried_result.job,
                        skipped=retried_result.skipped,
                        skip_reason=retried_result.skip_reason,
                        success=retried_result.success,
                        input_size_bytes=retried_result.input_size_bytes,
                        output_size_bytes=retried_result.output_size_bytes,
                        duration_seconds=retried_result.duration_seconds,
                        error_message=retried_result.error_message,
                        raw_error_message="\n".join(
                            filter(
                                None,
                                [result.raw_error_message, retried_result.raw_error_message],
                            )
                        )
                        or retried_result.raw_error_message,
                        media_duration_seconds=retried_result.media_duration_seconds,
                        fallback_used=retried_result.fallback_used,
                        retry_kind=transient_retry_kind,
                        attempts=attempts,
                    )
                if allow_hardware_fallback and _retry_mode_allows_hardware_fallback(
                    result, retry_mode
                ):
                    console.print(
                        f"[yellow]Retrying {filename} with software fallback[/yellow] [dim](libx265 faster, CRF {_FALLBACK_CRF})[/dim]"
                    )
                    fallback_job = _clone_job_with_settings(
                        job, preset=_FALLBACK_PRESET, crf=_FALLBACK_CRF
                    )
                    fallback_result = encode_file(
                        fallback_job,
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        progress_callback=make_callback(),
                        log_path=log_path,
                    )
                    fallback_result = _apply_failure_diagnostics(fallback_result)
                    attempts = list(result.attempts) + list(fallback_result.attempts)
                    if fallback_result.success:
                        result = EncodeResult(
                            job=fallback_result.job,
                            skipped=False,
                            skip_reason=None,
                            success=True,
                            input_size_bytes=fallback_result.input_size_bytes,
                            output_size_bytes=fallback_result.output_size_bytes,
                            duration_seconds=fallback_result.duration_seconds,
                            error_message=None,
                            raw_error_message=result.raw_error_message,
                            media_duration_seconds=fallback_result.media_duration_seconds,
                            fallback_used=True,
                            retry_kind=result.retry_kind,
                            attempts=attempts,
                        )
                    else:
                        combined_error = (
                            f"Hardware attempt failed: {result.error_message} | "
                            f"Fallback failed: {fallback_result.error_message}"
                        )
                        result = EncodeResult(
                            job=fallback_result.job,
                            skipped=False,
                            skip_reason=None,
                            success=False,
                            input_size_bytes=fallback_result.input_size_bytes,
                            output_size_bytes=fallback_result.output_size_bytes,
                            duration_seconds=fallback_result.duration_seconds,
                            error_message=combined_error,
                            raw_error_message="\n".join(
                                filter(
                                    None,
                                    [result.raw_error_message, fallback_result.raw_error_message],
                                )
                            )
                            or None,
                            media_duration_seconds=fallback_result.media_duration_seconds,
                            fallback_used=True,
                            retry_kind=result.retry_kind,
                            attempts=attempts,
                        )
            except KeyboardInterrupt:
                stall_stop.set()
                if stall_thread is not None:
                    stall_thread.join(timeout=1)
                if session is not None and session_path is not None and not job.skip:
                    update_session_entry(
                        session,
                        source=job.source,
                        status="pending",
                        encoder=job.preset,
                        last_progress_pct=stall_state["last_percent"],
                        last_progress_at=_now_iso(),
                        error="Interrupted by user",
                        fallback_used=False,
                        retry_count=0,
                    )
                    save_session(session, session_path)
                if results:
                    display.show_summary(
                        results,
                        resumed_from_session=resumed_from_session,
                        previously_completed=previously_completed,
                        previously_skipped=previously_skipped,
                    )
                console.print("\n[yellow]Interrupted.[/yellow]")
                if session_path is not None:
                    console.print(
                        f"[dim]Completed files are preserved. Resume later with the same command; session state is stored in[/dim] {session_path}"
                    )
                raise typer.Exit(code=EXIT_USER_CANCELLED)

            results.append(result)
            stall_stop.set()
            if stall_thread is not None:
                stall_thread.join(timeout=1)

            if use_calibration:
                if result.success and not result.skipped:
                    _record_success_calibration(result, ffprobe)
                elif not result.success and not result.skipped:
                    _record_failure_calibration(result)

            if not result.success and not result.skipped:
                if on_file_failure == "skip":
                    result.skipped = True
                    result.skip_reason = (
                        f"skipped_by_policy: {result.error_message or 'encode failed'}"
                    )
                elif on_file_failure == "stop":
                    stopped_early = True

            # Update session after each file so partial progress is persisted
            if session is not None and session_path is not None:
                if result.skipped:
                    status = "skipped"
                elif result.success:
                    status = "success"
                else:
                    status = "failed"
                update_session_entry(
                    session,
                    source=job.source,
                    status=status,
                    output=job.output if result.success else None,
                    error=result.error_message,
                    encoder=job.preset,
                    last_progress_pct=100.0 if result.success else stall_state["last_percent"],
                    last_progress_at=_now_iso() if not job.skip else None,
                    finished_at=_now_iso() if not job.skip else None,
                    fallback_used=result.fallback_used,
                    retry_count=result.retry_count,
                    first_error=result.first_error,
                    last_error=result.last_error,
                    attempt_history=result.attempts,
                )
                save_session(session, session_path)

            if not job.skip:
                bytes_done += file_size
                progress.update(overall_task, completed=bytes_done)
                progress.update(file_task, completed=task_total)
            if stopped_early:
                break

        progress.remove_task(file_task)

    display.show_summary(
        results,
        resumed_from_session=resumed_from_session,
        previously_completed=previously_completed,
        previously_skipped=previously_skipped,
    )
    return EncodeLoopResults(results, stopped_early=stopped_early)


def _prepare_tools(output_dir: Path | None) -> tuple[Path, Path]:
    ok, err = check_ffmpeg_available()
    if not ok:
        console.print(f"[red bold]Error:[/red bold] {err}")
        raise typer.Exit(code=EXIT_FFMPEG_NOT_FOUND)

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    return ffmpeg, ffprobe


def _analyze_with_optional_progress(
    directory: Path,
    recursive: bool,
    ffprobe: Path,
    ui_console: Console,
    show_progress: bool,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
) -> list[AnalysisItem]:
    files = scan_directory(directory, recursive=recursive)
    if not files:
        return []
    if not show_progress:
        return analyze_files(
            files,
            ffprobe,
            preset=preset,
            crf=crf,
            use_calibration=use_calibration,
        )

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=ui_console,
        transient=False,
        expand=True,
    ) as progress:
        task = progress.add_task(
            "[dim]Analyzing files (ffprobe + size estimates)...[/dim]", total=len(files)
        )

        def callback(completed: int, total: int, path: Path) -> None:
            name = path.name if len(path.name) <= 56 else path.name[:53] + "..."
            progress.update(
                task,
                total=total,
                completed=completed,
                description=f"[dim]Analyzing files (ffprobe + size estimates)...[/dim] [white]{name}[/white]",
            )

        items = analyze_files(
            files,
            ffprobe,
            progress_callback=callback,
            preset=preset,
            crf=crf,
            use_calibration=use_calibration,
        )
        progress.update(task, completed=len(files))
        return items


def _maybe_prompt_for_cleanup(results: list[EncodeResult], assume_yes: bool) -> list[Path]:
    candidates = eligible_cleanup_results(results)
    if not candidates:
        return []

    if not assume_yes:
        if not typer.confirm(
            "Delete the original source files for successful encodes and rename the compressed outputs back to the original filenames?",
            default=False,
        ):
            return []

    cleaned = cleanup_successful_results(results)
    if cleaned:
        console.print(
            f"[green]Cleanup complete:[/green] restored original names for {len(cleaned)} file(s)."
        )
    return cleaned


def _resolve_encode_settings(
    profile: str | None,
    crf: int | None,
    preset: str | None,
) -> tuple[int, str, str | None]:
    effective_crf = 20
    effective_preset = "fast"
    profile_name: str | None = None

    if profile:
        saved_profile = get_profile(profile)
        if saved_profile is None:
            console.print(f"[red bold]Error:[/red bold] profile '{profile}' was not found.")
            raise typer.Exit(code=EXIT_NO_FILES)
        effective_crf = saved_profile.crf
        effective_preset = saved_profile.preset
        profile_name = saved_profile.name

    if crf is not None:
        effective_crf = crf
    if preset is not None:
        effective_preset = preset

    return effective_crf, effective_preset, profile_name


def _build_resume_jobs(
    session: SessionManifest,
    *,
    ffprobe: Path,
) -> list[EncodeJob]:
    files: list[Path] = []
    for entry in session.entries:
        source = Path(entry.source)
        if entry.status not in {"pending", "failed", "in_progress"}:
            continue
        if source.exists():
            files.append(source)
    if not files:
        return []
    output_dir = Path(session.output_dir) if session.output_dir is not None else None
    return build_jobs(
        files=files,
        output_dir=output_dir,
        overwrite=session.overwrite,
        crf=session.crf,
        preset=session.preset,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=False,
    )


def _prepare_overnight_jobs(
    *,
    directory: Path,
    recursive: bool,
    output_dir: Path | None,
    overwrite: bool,
    no_skip: bool,
    ffmpeg: Path,
    ffprobe: Path,
    policy: str,
    use_calibration: bool,
    duplicate_policy: str,
) -> tuple[list[EncodeJob], str, int]:
    from mediashrink.wizard import (
        _run_preflight_checks,
        _sum_item_durations,
        benchmark_encoder,
        build_profiles,
        detect_available_encoders,
    )

    files = scan_directory(directory, recursive=recursive)
    if not files:
        return [], "fast", 20
    items = analyze_files(files, ffprobe, use_calibration=use_calibration)
    items, _ = apply_duplicate_policy_to_items(items, policy=duplicate_policy)
    recommended_items = [item for item in items if item.recommendation == "recommended"]
    maybe_items = [item for item in items if item.recommendation == "maybe"]
    selected_items = recommended_items or maybe_items
    if not selected_items:
        selected_items = items
    sample_item = max(selected_items, key=lambda item: item.size_bytes)
    available_hw = detect_available_encoders(
        ffmpeg, Console(quiet=True), sample_item.source, ffprobe
    )
    benchmark_speeds: dict[str, float | None] = {}
    for key in list(available_hw) + ["fast", "faster"]:
        benchmark_speeds[key] = benchmark_encoder(
            encoder_key=key,
            sample_file=sample_item.source,
            sample_duration=sample_item.duration_seconds or 3600.0,
            crf=20,
            ffmpeg=ffmpeg,
        )
    profiles = build_profiles(
        available_hw=available_hw,
        benchmark_speeds=benchmark_speeds,
        total_media_seconds=_sum_item_durations(selected_items),
        total_input_bytes=sum(item.size_bytes for item in selected_items),
        candidate_items=selected_items,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
    )
    selected_profile = next(
        (profile for profile in profiles if profile.is_recommended), profiles[0]
    )
    jobs = build_jobs(
        files=[item.source for item in selected_items],
        output_dir=output_dir,
        overwrite=overwrite,
        crf=selected_profile.crf,
        preset=selected_profile.encoder_key,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )
    compatible_jobs, preflight_failures = _run_preflight_checks(
        [job for job in jobs if not job.skip],
        ffmpeg,
        ffprobe,
        crf=selected_profile.crf,
        preset=selected_profile.encoder_key,
        console=Console(quiet=True),
    )
    if preflight_failures:
        incompatible_sources = {job.source for job, _ in preflight_failures}
        for job in jobs:
            if job.source in incompatible_sources:
                job.skip = True
                job.skip_reason = "incompatible: preflight compatibility check failed"
        if not compatible_jobs and selected_profile.encoder_key != _FALLBACK_PRESET:
            jobs = build_jobs(
                files=[item.source for item in selected_items],
                output_dir=output_dir,
                overwrite=overwrite,
                crf=_FALLBACK_CRF,
                preset=_FALLBACK_PRESET,
                dry_run=False,
                ffprobe=ffprobe,
                no_skip=no_skip,
            )
            selected_profile = next(
                (
                    profile
                    for profile in profiles
                    if profile.encoder_key == _FALLBACK_PRESET and profile.crf == _FALLBACK_CRF
                ),
                selected_profile,
            )
    return jobs, selected_profile.encoder_key, selected_profile.crf


@app.command("encode", hidden=True)
def encode_cmd(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to compress.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        help="H.265 CRF quality value (0-51, lower = better quality). Default: 20.",
        min=0,
        max=51,
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Encoding preset. Software: ultrafast/faster/fast/medium/slow. "
            "Hardware (much faster): qsv (Intel), nvenc (Nvidia), amf (AMD). "
            "Default: fast."
        ),
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load saved CRF/preset defaults from a named profile.",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be encoded without actually encoding.",
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="After successful side-by-side encodes, delete originals and rename outputs back to the original filenames.",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Ignore any existing session file and start fresh.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a single JSON blob instead of Rich terminal output.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Write FFmpeg stderr to a log file alongside the output.",
    ),
    policy: str = typer.Option(
        "fastest-wall-clock",
        "--policy",
        help="Encoder selection policy: fastest-wall-clock, lowest-cpu, best-compression, or highest-confidence.",
    ),
    on_file_failure: str = typer.Option(
        "retry",
        "--on-file-failure",
        help="What to do after an unrecovered file failure: skip, retry, or stop.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results to improve estimates and policy ranking.",
    ),
    retry_mode: str = typer.Option(
        "balanced",
        "--retry-mode",
        help="Retry strategy: conservative, balanced, or aggressive.",
    ),
    queue_strategy: str = typer.Option(
        "original",
        "--queue-strategy",
        help="Batch order: original or safe-first.",
    ),
    overnight: bool = typer.Option(
        False,
        "--overnight",
        help="Run unattended with safe overnight defaults: auto-resume, verbose logging, skip failed files, and keep cleanup off.",
    ),
    stall_warning_seconds: int = typer.Option(
        int(STALL_WARNING_SECONDS),
        "--stall-warning-seconds",
        min=1,
        help="Warn if FFmpeg produces no progress update for this many seconds.",
    ),
) -> None:
    runtime = _resolve_runtime_settings(
        overnight=overnight,
        policy=policy,
        on_file_failure=on_file_failure,
        verbose=verbose,
        cleanup=cleanup,
        yes=yes,
        use_calibration=use_calibration,
        retry_mode=retry_mode,
        queue_strategy=queue_strategy,
    )
    verbose = bool(runtime["verbose"])
    cleanup = bool(runtime["cleanup"])
    yes = bool(runtime["yes"])
    use_calibration = bool(runtime["use_calibration"])
    on_file_failure = str(runtime["on_file_failure"])
    policy = str(runtime["policy"])
    retry_mode = str(runtime["retry_mode"])
    queue_strategy = str(runtime["queue_strategy"])
    quiet_console = Console(quiet=True) if json_output else console
    display = EncodingDisplay(quiet_console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)
    started_at = _now_iso()

    effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)

    files = scan_directory(directory, recursive=recursive)
    if not files:
        console.print(
            f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
        )
        raise typer.Exit(code=EXIT_NO_FILES)

    jobs = build_jobs(
        files=files,
        output_dir=output_dir,
        overwrite=overwrite,
        crf=effective_crf,
        preset=effective_preset,
        dry_run=dry_run,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )
    jobs = _prioritize_jobs(jobs, queue_strategy)
    split_followup_manifest: Path | None = None

    prior: SessionManifest | None = None
    resumed_from_session = False
    # Resume detection — skip files already completed in a previous session
    if not dry_run and not no_resume:
        prior = find_resumable_session(directory, output_dir, effective_preset, effective_crf)
        if prior is not None:
            done = {e.source for e in prior.entries if e.status == "success"}
            console.print(
                f"[cyan]Session found:[/cyan] {_format_resume_counts(prior)}",
                highlight=False,
            )
            console.print(f"[dim]Session path:[/dim] {get_session_path(directory, output_dir)}")
            if done:
                should_resume = overnight or typer.confirm(
                    "Resume from the last completed file?",
                    default=True,
                )
                if should_resume:
                    resumed_from_session = True
                    for job in jobs:
                        if str(job.source) in done:
                            job.skip = True
                            job.skip_reason = "resumed (already done)"
                    _reconcile_session_with_jobs(prior, jobs)

    display.show_scan_table(jobs)

    if not dry_run:
        jobs, preflight_notes, incompatible = _apply_batch_preflight_policy(
            jobs,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            on_file_failure=on_file_failure,
        )
        if preflight_notes:
            _print_grouped_preflight_details(
                preflight_notes,
                style="yellow",
                prefix="Compatibility summary: ",
            )
        if incompatible and on_file_failure == "skip":
            incompatible_jobs = [
                job for job in jobs if (job.skip_reason or "").startswith("incompatible:")
            ]
            split_followup_manifest = _write_followup_manifest_for_jobs(
                directory=directory,
                recursive=recursive,
                preset=effective_preset,
                crf=effective_crf,
                jobs=incompatible_jobs,
                ffprobe=ffprobe,
                duplicate_policy=None,
                details=incompatible,
            )
            console.print(
                f"[yellow]Skipping {len(incompatible)} incompatible file(s) before batch start because --on-file-failure=skip.[/yellow]"
            )
            _print_grouped_preflight_details(
                incompatible,
                style="yellow",
                prefix="  ",
            )
            if split_followup_manifest is not None:
                console.print(f"[dim]Follow-up manifest:[/dim] {split_followup_manifest}")
        if incompatible and on_file_failure == "stop":
            _print_grouped_preflight_details(
                incompatible,
                style="red",
                prefix="Compatibility check failed: ",
            )
            raise typer.Exit(code=EXIT_ENCODE_FAILURES)

    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        results = _build_skipped_results(jobs)
        if not results or not _has_incompatible_skips(jobs):
            console.print("[dim]Nothing to encode.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)
        console.print("[dim]All remaining files were skipped before encode start.[/dim]")
        stopped_early = False
    else:
        stopped_early = False
    preflight_warnings = _collect_preflight_warnings(jobs, ffprobe)

    if to_encode and not dry_run and not yes:
        if not json_output:
            _print_safe_interrupt_guidance()
            _print_preflight_warnings(preflight_warnings)
        _maybe_warn_low_disk_space(jobs, output_dir or directory, overwrite, assume_yes=json_output)
        if not display.confirm_proceed(len(to_encode)):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)
    elif to_encode and not dry_run:
        if not json_output:
            _print_safe_interrupt_guidance()
            _print_preflight_warnings(preflight_warnings)
        _maybe_warn_low_disk_space(jobs, output_dir or directory, overwrite, assume_yes=True)
    elif not dry_run and not json_output:
        _print_preflight_warnings(preflight_warnings)

    session_path = get_session_path(directory, output_dir) if not dry_run else None
    active_session = None
    if session_path is not None:
        if resumed_from_session and prior is not None:
            active_session = prior
        else:
            active_session = build_session(
                directory,
                effective_preset,
                effective_crf,
                overwrite,
                output_dir,
                jobs,
                policy=policy,
                on_file_failure=on_file_failure,
                use_calibration=use_calibration,
                retry_mode=retry_mode,
                queue_strategy=queue_strategy,
            )
        save_session(active_session, session_path)

    log_path: Path | None = None
    if verbose and not dry_run:
        log_dir = output_dir if output_dir is not None else directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"mediashrink_{timestamp}.log"
        console.print(f"[dim]Verbose log:[/dim] {log_path}")

    resume_completed, resume_skipped = _resume_summary_counts(
        prior if resumed_from_session else None
    )

    if to_encode:
        results, stopped_early = _normalize_loop_result(
            _run_encode_loop(
                jobs,
                ffmpeg,
                ffprobe,
                display,
                session=active_session,
                session_path=session_path,
                log_path=log_path,
                stall_warning_seconds=float(stall_warning_seconds),
                resumed_from_session=resumed_from_session,
                previously_completed=resume_completed,
                previously_skipped=resume_skipped,
                on_file_failure=on_file_failure,
                use_calibration=use_calibration,
                retry_mode=retry_mode,
            )
        )
    cleaned_paths: list[Path] = []
    if cleanup:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not dry_run and not overwrite and output_dir is None and not yes:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=False)
    _record_cleanup_results(active_session, session_path, results, cleaned_paths)

    if not dry_run:
        json_report, text_report = _write_batch_reports(
            mode="encode",
            base_dir=directory,
            output_dir=output_dir,
            manifest_path=None,
            preset=effective_preset,
            crf=effective_crf,
            overwrite=overwrite,
            cleanup_requested=cleanup,
            resumed_from_session=resumed_from_session,
            session_path=session_path,
            started_at=started_at,
            finished_at=_now_iso(),
            results=results,
            cleaned_paths=cleaned_paths,
            log_path=log_path,
            warnings=preflight_warnings,
            policy=policy,
            on_file_failure=on_file_failure,
            retry_mode=retry_mode,
            queue_strategy=queue_strategy,
            split_followup_manifest=split_followup_manifest,
            estimate_miss_summary=_estimate_miss_summary(results),
        )
        if not json_output:
            console.print(f"[dim]Reports:[/dim] {json_report}  {text_report}")

    has_failures = any(not r.success and not r.skipped for r in results) or stopped_early
    exit_code = EXIT_ENCODE_FAILURES if has_failures else EXIT_SUCCESS
    if json_output:
        print(_results_to_json(results, exit_code))
    if has_failures:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def analyze(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to analyze.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load saved CRF/preset defaults from a named profile.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        min=0,
        max=51,
        help="H.265 CRF quality value used for analysis estimates.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Encoding preset used for analysis estimates.",
    ),
    manifest_out: Optional[Path] = typer.Option(
        None,
        "--manifest-out",
        help="Write recommended candidates to a JSON manifest.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit analysis results as a JSON blob instead of Rich terminal output.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results to improve estimates.",
    ),
    duplicate_policy: str = typer.Option(
        "prefer-mkv",
        "--duplicate-policy",
        help="How to handle likely duplicate titles across formats: prefer-mkv, all, or skip-title.",
    ),
) -> None:
    quiet_console = Console(quiet=True) if json_output else console
    ffmpeg, ffprobe = _prepare_tools(None)
    effective_crf, effective_preset, profile_name = _resolve_encode_settings(profile, crf, preset)

    items = _analyze_with_optional_progress(
        directory=directory,
        recursive=recursive,
        ffprobe=ffprobe,
        ui_console=quiet_console,
        show_progress=not json_output,
        preset=effective_preset,
        crf=effective_crf,
        use_calibration=use_calibration,
    )
    items, duplicate_notes = apply_duplicate_policy_to_items(
        items,
        policy=_validate_duplicate_policy(duplicate_policy),
    )
    if not items:
        if json_output:
            print(json.dumps({"exit_code": EXIT_NO_FILES, "items": []}))
        else:
            console.print(
                f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
            )
        raise typer.Exit(code=EXIT_NO_FILES)

    estimated_total_encode_seconds = estimate_analysis_encode_seconds(
        items=items,
        preset=effective_preset,
        crf=effective_crf,
        ffmpeg=ffmpeg,
        use_calibration=use_calibration,
    )
    estimate_confidence = estimate_analysis_confidence(items)
    estimate_confidence_detail = describe_estimate_confidence(items)
    size_confidence = estimate_size_confidence(
        items,
        preset=effective_preset,
        use_calibration=use_calibration,
    )
    size_confidence_detail = describe_size_confidence(
        items,
        preset=effective_preset,
        use_calibration=use_calibration,
    )
    time_confidence = estimate_time_confidence(
        items,
        benchmarked_files=0,
        preset=effective_preset,
        use_calibration=use_calibration,
    )
    time_confidence_detail = describe_time_confidence(
        items,
        benchmarked_files=0,
        preset=effective_preset,
        use_calibration=use_calibration,
    )
    calibration_detail = describe_estimate_calibration(
        items,
        preset=effective_preset,
        use_calibration=use_calibration,
    )
    if calibration_detail:
        estimate_confidence_detail += f"; local history: {calibration_detail}"

    if json_output:
        manifest = build_manifest(
            directory=directory,
            recursive=recursive,
            preset=effective_preset,
            crf=effective_crf,
            profile_name=profile_name,
            estimated_total_encode_seconds=estimated_total_encode_seconds,
            estimate_confidence=estimate_confidence,
            size_confidence=size_confidence,
            size_confidence_detail=size_confidence_detail,
            time_confidence=time_confidence,
            time_confidence_detail=time_confidence_detail,
            duplicate_policy=duplicate_policy,
            items=items,
        )
        print(json.dumps(manifest.to_dict()))
    else:
        display_analysis_summary(
            items,
            estimated_total_encode_seconds,
            quiet_console,
            estimate_confidence=estimate_confidence,
            estimate_confidence_detail=estimate_confidence_detail,
            size_confidence=size_confidence,
            size_confidence_detail=size_confidence_detail,
            time_confidence=time_confidence,
            time_confidence_detail=time_confidence_detail,
            notes=duplicate_notes,
        )

    if manifest_out is not None:
        manifest = build_manifest(
            directory=directory,
            recursive=recursive,
            preset=effective_preset,
            crf=effective_crf,
            profile_name=profile_name,
            estimated_total_encode_seconds=estimated_total_encode_seconds,
            estimate_confidence=estimate_confidence,
            size_confidence=size_confidence,
            size_confidence_detail=size_confidence_detail,
            time_confidence=time_confidence,
            time_confidence_detail=time_confidence_detail,
            duplicate_policy=duplicate_policy,
            items=items,
        )
        save_manifest(manifest, manifest_out)
        if not json_output:
            console.print(f"[green]Wrote manifest[/green] {manifest_out}")


@app.command()
def apply(
    manifest: Path = typer.Argument(
        ...,
        help="Path to an analysis manifest JSON file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Override manifest settings using a saved profile.",
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        min=0,
        max=51,
        help="Override manifest CRF.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Override manifest preset.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="After successful side-by-side encodes, delete originals and rename outputs back to the original filenames.",
    ),
    policy: str = typer.Option(
        "fastest-wall-clock",
        "--policy",
        help="Encoder policy metadata recorded with the run.",
    ),
    on_file_failure: str = typer.Option(
        "retry",
        "--on-file-failure",
        help="What to do after an unrecovered file failure: skip, retry, or stop.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results to improve estimates and policy ranking.",
    ),
    retry_mode: str = typer.Option(
        "balanced",
        "--retry-mode",
        help="Retry strategy: conservative, balanced, or aggressive.",
    ),
    queue_strategy: str = typer.Option(
        "original",
        "--queue-strategy",
        help="Batch order: original or safe-first.",
    ),
    stall_warning_seconds: int = typer.Option(
        int(STALL_WARNING_SECONDS),
        "--stall-warning-seconds",
        min=1,
        help="Warn if FFmpeg produces no progress update for this many seconds.",
    ),
) -> None:
    display = EncodingDisplay(console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)
    started_at = _now_iso()
    loaded_manifest = load_manifest(manifest)

    effective_crf = loaded_manifest.crf
    effective_preset = loaded_manifest.preset
    if profile or crf is not None or preset is not None:
        effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)

    missing_files = [item.source for item in loaded_manifest.items if not item.source.exists()]
    existing_files = [item.source for item in loaded_manifest.items if item.source.exists()]
    for missing_path in missing_files:
        console.print(f"[yellow]Missing file from manifest:[/yellow] {missing_path}")

    if not existing_files:
        console.print("[yellow]No manifest files are available to encode.[/yellow]")
        raise typer.Exit(code=EXIT_NO_FILES)

    jobs = build_jobs(
        files=existing_files,
        output_dir=output_dir,
        overwrite=overwrite,
        crf=effective_crf,
        preset=effective_preset,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=False,
    )
    jobs = _prioritize_jobs(jobs, _validate_queue_strategy(queue_strategy))
    split_followup_manifest: Path | None = None

    jobs, preflight_notes, incompatible = _apply_batch_preflight_policy(
        jobs,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        on_file_failure=_validate_failure_policy(on_file_failure),
    )
    if preflight_notes:
        _print_grouped_preflight_details(
            preflight_notes,
            style="yellow",
            prefix="Compatibility summary: ",
        )
    if incompatible and on_file_failure == "skip":
        incompatible_jobs = [
            job for job in jobs if (job.skip_reason or "").startswith("incompatible:")
        ]
        split_followup_manifest = _write_followup_manifest_for_jobs(
            directory=loaded_manifest.analyzed_directory,
            recursive=loaded_manifest.recursive,
            preset=effective_preset,
            crf=effective_crf,
            jobs=incompatible_jobs,
            ffprobe=ffprobe,
            duplicate_policy=loaded_manifest.duplicate_policy,
            details=incompatible,
        )
        console.print(
            f"[yellow]Skipping {len(incompatible)} incompatible file(s) before batch start because --on-file-failure=skip.[/yellow]"
        )
        _print_grouped_preflight_details(incompatible, style="yellow", prefix="  ")
        if split_followup_manifest is not None:
            console.print(f"[dim]Follow-up manifest:[/dim] {split_followup_manifest}")
    if incompatible and on_file_failure == "stop":
        _print_grouped_preflight_details(
            incompatible,
            style="red",
            prefix="Compatibility check failed: ",
        )
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)

    display.show_scan_table(jobs)
    to_encode = [job for job in jobs if not job.skip]
    if not to_encode:
        results = _build_skipped_results(jobs)
        if not results or not _has_incompatible_skips(jobs):
            console.print("[dim]Nothing to encode.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)
        console.print("[dim]All manifest files were skipped before encode start.[/dim]")
        stopped_early = False
    else:
        stopped_early = False
    preflight_warnings = _collect_preflight_warnings(jobs, ffprobe)

    if to_encode:
        _print_safe_interrupt_guidance()
        _print_preflight_warnings(preflight_warnings)
        _maybe_warn_low_disk_space(
            jobs,
            output_dir or loaded_manifest.analyzed_directory,
            overwrite,
            assume_yes=yes,
        )
        if not yes and not display.confirm_proceed(len(to_encode)):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)
    else:
        _print_preflight_warnings(preflight_warnings)

    if to_encode:
        results, stopped_early = _normalize_loop_result(
            _run_encode_loop(
                jobs,
                ffmpeg,
                ffprobe,
                display,
                stall_warning_seconds=float(stall_warning_seconds),
                on_file_failure=_validate_failure_policy(on_file_failure),
                use_calibration=use_calibration,
                retry_mode=_validate_retry_mode(retry_mode),
            )
        )
    cleaned_paths: list[Path] = []
    if cleanup:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not overwrite and output_dir is None and not yes:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=False)

    json_report, text_report = _write_batch_reports(
        mode="apply",
        base_dir=loaded_manifest.analyzed_directory,
        output_dir=output_dir,
        manifest_path=manifest,
        preset=effective_preset,
        crf=effective_crf,
        overwrite=overwrite,
        cleanup_requested=cleanup,
        resumed_from_session=False,
        session_path=None,
        started_at=started_at,
        finished_at=_now_iso(),
        results=results,
        cleaned_paths=cleaned_paths,
        log_path=None,
        warnings=preflight_warnings,
        policy=_validate_policy(policy),
        on_file_failure=on_file_failure,
        retry_mode=_validate_retry_mode(retry_mode),
        queue_strategy=_validate_queue_strategy(queue_strategy),
        split_followup_manifest=split_followup_manifest,
        estimate_miss_summary=_estimate_miss_summary(results),
    )
    console.print(f"[dim]Reports:[/dim] {json_report}  {text_report}")

    if stopped_early or any(not r.success and not r.skipped for r in results):
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def wizard(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to compress.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help=(
            f"Scan subdirectories for supported video files ({supported_formats_label()}). "
            "Enabled by default for the wizard."
        ),
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit encode results as a JSON blob instead of Rich terminal output.",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Non-interactive mode: auto-select the recommended profile, skip all prompts.",
    ),
    policy: str = typer.Option(
        "fastest-wall-clock",
        "--policy",
        help="Encoder recommendation policy: fastest-wall-clock, lowest-cpu, best-compression, or highest-confidence.",
    ),
    on_file_failure: str = typer.Option(
        "retry",
        "--on-file-failure",
        help="What to do after an unrecovered file failure: skip, retry, or stop.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results to improve estimates and recommendations.",
    ),
    duplicate_policy: str = typer.Option(
        "prefer-mkv",
        "--duplicate-policy",
        help="How to handle likely duplicate titles across formats: prefer-mkv, all, or skip-title.",
    ),
    show_all_profiles: bool = typer.Option(
        False,
        "--show-all-profiles",
        help="Show every profile row instead of hiding near-duplicate trade-offs.",
    ),
    stall_warning_seconds: int = typer.Option(
        int(STALL_WARNING_SECONDS),
        "--stall-warning-seconds",
        min=1,
        help="Warn if FFmpeg produces no progress update for this many seconds.",
    ),
) -> None:
    """Interactively detect hardware, choose settings, and optionally save a profile."""
    from mediashrink.wizard import run_wizard

    ffmpeg, ffprobe = _prepare_tools(output_dir)
    quiet_console = Console(quiet=True) if json_output else console
    display = EncodingDisplay(quiet_console)
    started_at = _now_iso()

    jobs, action, wizard_cleanup = run_wizard(
        directory=directory,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        recursive=recursive,
        output_dir=output_dir,
        overwrite=overwrite,
        no_skip=no_skip,
        console=quiet_console,
        auto=auto,
        policy=_validate_policy(policy),
        on_file_failure=_validate_failure_policy(on_file_failure),
        use_calibration=use_calibration,
        duplicate_policy=_validate_duplicate_policy(duplicate_policy),
        show_all_profiles=show_all_profiles,
    )

    if action == "cancel":
        if not json_output:
            console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)
    if action == "export":
        raise typer.Exit(code=EXIT_SUCCESS)

    if not json_output:
        _print_safe_interrupt_guidance()
    preflight_warnings = _collect_preflight_warnings(jobs, ffprobe)
    if not json_output:
        _print_preflight_warnings(preflight_warnings)
    _maybe_warn_low_disk_space(
        jobs,
        output_dir or directory,
        overwrite,
        assume_yes=auto,
    )

    results, stopped_early = _normalize_loop_result(
        _run_encode_loop(
            jobs,
            ffmpeg,
            ffprobe,
            display,
            stall_warning_seconds=float(stall_warning_seconds),
            on_file_failure=on_file_failure,
            use_calibration=use_calibration,
        )
    )
    cleaned_paths: list[Path] = []
    if wizard_cleanup:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not json_output and not overwrite and output_dir is None:
        pass  # cleanup was already asked upfront; no second prompt

    json_report, text_report = _write_batch_reports(
        mode="wizard",
        base_dir=directory,
        output_dir=output_dir,
        manifest_path=None,
        preset=jobs[0].preset if jobs else "fast",
        crf=jobs[0].crf if jobs else 20,
        overwrite=overwrite,
        cleanup_requested=wizard_cleanup,
        resumed_from_session=False,
        session_path=None,
        started_at=started_at,
        finished_at=_now_iso(),
        results=results,
        cleaned_paths=cleaned_paths,
        log_path=None,
        warnings=preflight_warnings,
        policy=policy,
        on_file_failure=on_file_failure,
        estimate_miss_summary=_estimate_miss_summary(results),
    )
    if not json_output:
        console.print(f"[dim]Reports:[/dim] {json_report}  {text_report}")

    has_failures = any(not r.success and not r.skipped for r in results) or stopped_early
    exit_code = EXIT_ENCODE_FAILURES if has_failures else EXIT_SUCCESS
    if json_output:
        print(_results_to_json(results, exit_code))
    if has_failures:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def resume(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing the resumable session for supported video files ({supported_formats_label()}).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Locate the session/output files here instead of alongside originals.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="After successful side-by-side encodes, delete originals and rename outputs back to the original filenames.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit encode results as a JSON blob instead of Rich terminal output.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Write FFmpeg stderr to a log file alongside the output.",
    ),
    policy: str = typer.Option(
        "fastest-wall-clock",
        "--policy",
        help="Encoder policy metadata recorded with the resumed run.",
    ),
    on_file_failure: str = typer.Option(
        "retry",
        "--on-file-failure",
        help="What to do after an unrecovered file failure: skip, retry, or stop.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results during resumed runs.",
    ),
    retry_mode: str = typer.Option(
        "balanced",
        "--retry-mode",
        help="Retry strategy: conservative, balanced, or aggressive.",
    ),
    queue_strategy: str = typer.Option(
        "original",
        "--queue-strategy",
        help="Batch order: original or safe-first.",
    ),
    stall_warning_seconds: int = typer.Option(
        int(STALL_WARNING_SECONDS),
        "--stall-warning-seconds",
        min=1,
        help="Warn if FFmpeg produces no progress update for this many seconds.",
    ),
) -> None:
    quiet_console = Console(quiet=True) if json_output else console
    display = EncodingDisplay(quiet_console)
    ffmpeg, ffprobe = _prepare_tools(output_dir)
    started_at = _now_iso()

    session_path = get_session_path(directory, output_dir)
    session = load_session(session_path)
    if session is None:
        console.print(f"[red bold]Error:[/red bold] no resumable session found at {session_path}")
        raise typer.Exit(code=EXIT_NO_FILES)
    if session.directory != str(directory):
        console.print(
            "[red bold]Error:[/red bold] session directory does not match the requested directory."
        )
        raise typer.Exit(code=EXIT_NO_FILES)
    if output_dir is not None and session.output_dir != str(output_dir):
        console.print(
            "[red bold]Error:[/red bold] session output directory does not match --output-dir."
        )
        raise typer.Exit(code=EXIT_NO_FILES)
    jobs = _build_resume_jobs(session, ffprobe=ffprobe)
    jobs = _prioritize_jobs(jobs, _validate_queue_strategy(queue_strategy))
    split_followup_manifest: Path | None = None
    if not jobs:
        console.print("[yellow]No resumable files remain in the saved session.[/yellow]")
        raise typer.Exit(code=EXIT_NO_FILES)

    jobs, preflight_notes, incompatible = _apply_batch_preflight_policy(
        jobs,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        on_file_failure=_validate_failure_policy(on_file_failure),
    )
    if preflight_notes:
        _print_grouped_preflight_details(
            preflight_notes,
            style="yellow",
            prefix="Compatibility summary: ",
        )
    if incompatible and on_file_failure == "skip":
        incompatible_jobs = [
            job for job in jobs if (job.skip_reason or "").startswith("incompatible:")
        ]
        split_followup_manifest = _write_followup_manifest_for_jobs(
            directory=directory,
            recursive=False,
            preset=session.preset,
            crf=session.crf,
            jobs=incompatible_jobs,
            ffprobe=ffprobe,
            duplicate_policy=None,
            details=incompatible,
        )
        console.print(
            f"[yellow]Skipping {len(incompatible)} incompatible file(s) before batch start because --on-file-failure=skip.[/yellow]"
        )
        _print_grouped_preflight_details(incompatible, style="yellow", prefix="  ")
        if split_followup_manifest is not None:
            console.print(f"[dim]Follow-up manifest:[/dim] {split_followup_manifest}")
    if incompatible and on_file_failure == "stop":
        _print_grouped_preflight_details(
            incompatible,
            style="red",
            prefix="Compatibility check failed: ",
        )
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)

    _reconcile_session_with_jobs(session, jobs)
    console.print(
        f"[cyan]Resuming session:[/cyan] {_format_resume_counts(session)}",
        highlight=False,
    )
    console.print(f"[dim]Session path:[/dim] {session_path}")
    display.show_scan_table(jobs)
    to_encode = [job for job in jobs if not job.skip]
    preflight_warnings = _collect_preflight_warnings(jobs, ffprobe)
    if not json_output and to_encode:
        _print_safe_interrupt_guidance()
    if not json_output:
        _print_preflight_warnings(preflight_warnings)
    if to_encode:
        _maybe_warn_low_disk_space(
            jobs,
            output_dir or directory,
            session.overwrite,
            assume_yes=yes,
        )

    if not to_encode:
        results = _build_skipped_results(jobs)
        if not results or not _has_incompatible_skips(jobs):
            console.print("[dim]Nothing to encode.[/dim]")
            raise typer.Exit(code=EXIT_USER_CANCELLED)
        console.print("[dim]All resumable files were skipped before encode start.[/dim]")
        stopped_early = False
    else:
        stopped_early = False
    if to_encode and not yes and not display.confirm_proceed(len(to_encode)):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=EXIT_USER_CANCELLED)

    log_path: Path | None = None
    if verbose:
        log_dir = output_dir if output_dir is not None else directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"mediashrink_{timestamp}.log"
        console.print(f"[dim]Verbose log:[/dim] {log_path}")

    resume_completed, resume_skipped = _resume_summary_counts(session)

    if to_encode:
        results, stopped_early = _normalize_loop_result(
            _run_encode_loop(
                jobs,
                ffmpeg,
                ffprobe,
                display,
                session=session,
                session_path=session_path,
                log_path=log_path,
                stall_warning_seconds=float(stall_warning_seconds),
                resumed_from_session=True,
                previously_completed=resume_completed,
                previously_skipped=resume_skipped,
                on_file_failure=_validate_failure_policy(on_file_failure),
                use_calibration=use_calibration,
                retry_mode=_validate_retry_mode(retry_mode),
            )
        )
    cleaned_paths: list[Path] = []
    if cleanup:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=True)
    elif not session.overwrite and output_dir is None and not yes:
        cleaned_paths = _maybe_prompt_for_cleanup(results, assume_yes=False)
    _record_cleanup_results(session, session_path, results, cleaned_paths)

    json_report, text_report = _write_batch_reports(
        mode="resume",
        base_dir=directory,
        output_dir=output_dir,
        manifest_path=None,
        preset=session.preset,
        crf=session.crf,
        overwrite=session.overwrite,
        cleanup_requested=cleanup,
        resumed_from_session=True,
        session_path=session_path,
        started_at=started_at,
        finished_at=_now_iso(),
        results=results,
        cleaned_paths=cleaned_paths,
        log_path=log_path,
        warnings=preflight_warnings,
        policy=_validate_policy(policy),
        on_file_failure=on_file_failure,
        split_followup_manifest=split_followup_manifest,
        estimate_miss_summary=_estimate_miss_summary(results),
        retry_mode=_validate_retry_mode(retry_mode),
        queue_strategy=_validate_queue_strategy(queue_strategy),
    )
    if not json_output:
        console.print(f"[dim]Reports:[/dim] {json_report}  {text_report}")

    has_failures = any(not r.success and not r.skipped for r in results) or stopped_early
    exit_code = EXIT_ENCODE_FAILURES if has_failures else EXIT_SUCCESS
    if json_output:
        print(_results_to_json(results, exit_code))
    if has_failures:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def overnight(
    directory: Path = typer.Argument(
        ...,
        help=f"Directory containing supported video files ({supported_formats_label()}) to compress overnight.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Write output files here instead of alongside originals.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace original files after successful encoding.",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Scan subdirectories for supported files.",
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Encode files even if they appear to already be H.265.",
    ),
    policy: str = typer.Option(
        "highest-confidence",
        "--policy",
        help="Encoder selection policy: fastest-wall-clock, lowest-cpu, best-compression, or highest-confidence.",
    ),
    use_calibration: bool = typer.Option(
        True,
        "--use-calibration/--no-calibration",
        help="Use local historical encode results to improve estimates and profile ranking.",
    ),
    duplicate_policy: str = typer.Option(
        "prefer-mkv",
        "--duplicate-policy",
        help="How to handle likely duplicate titles across formats: prefer-mkv, all, or skip-title.",
    ),
    retry_mode: str = typer.Option(
        "conservative",
        "--retry-mode",
        help="Retry strategy: conservative, balanced, or aggressive.",
    ),
    queue_strategy: str = typer.Option(
        "safe-first",
        "--queue-strategy",
        help="Batch order: original or safe-first.",
    ),
    stall_warning_seconds: int = typer.Option(
        int(STALL_WARNING_SECONDS),
        "--stall-warning-seconds",
        min=1,
        help="Warn if FFmpeg produces no progress update for this many seconds.",
    ),
) -> None:
    ffmpeg, ffprobe = _prepare_tools(output_dir)
    started_at = _now_iso()
    display = EncodingDisplay(console)
    policy = _validate_policy(policy)
    jobs, effective_preset, effective_crf = _prepare_overnight_jobs(
        directory=directory,
        recursive=recursive,
        output_dir=output_dir,
        overwrite=overwrite,
        no_skip=no_skip,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
        duplicate_policy=_validate_duplicate_policy(duplicate_policy),
    )
    queue_strategy = _validate_queue_strategy(queue_strategy)
    retry_mode = _validate_retry_mode(retry_mode)
    jobs = _prioritize_jobs(jobs, queue_strategy)
    split_followup_manifest: Path | None = None
    if not jobs:
        console.print(
            f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
        )
        raise typer.Exit(code=EXIT_NO_FILES)

    jobs, preflight_notes, incompatible = _apply_batch_preflight_policy(
        jobs,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        on_file_failure="skip",
    )
    if preflight_notes:
        _print_grouped_preflight_details(
            preflight_notes,
            style="yellow",
            prefix="Compatibility summary: ",
        )
    if incompatible:
        incompatible_jobs = [
            job for job in jobs if (job.skip_reason or "").startswith("incompatible:")
        ]
        split_followup_manifest = _write_followup_manifest_for_jobs(
            directory=directory,
            recursive=recursive,
            preset=effective_preset,
            crf=effective_crf,
            jobs=incompatible_jobs,
            ffprobe=ffprobe,
            duplicate_policy=duplicate_policy,
            details=incompatible,
        )
        console.print(
            f"[yellow]Skipping {len(incompatible)} incompatible file(s) before batch start because overnight mode continues past file-level issues.[/yellow]"
        )
        _print_grouped_preflight_details(incompatible, style="yellow", prefix="  ")
        if split_followup_manifest is not None:
            console.print(f"[dim]Follow-up manifest:[/dim] {split_followup_manifest}")

    prior = find_resumable_session(directory, output_dir, effective_preset, effective_crf)
    resumed_from_session = False
    if prior is not None:
        done = {e.source for e in prior.entries if e.status == "success"}
        for job in jobs:
            if str(job.source) in done:
                job.skip = True
                job.skip_reason = "resumed (already done)"
        _reconcile_session_with_jobs(prior, jobs)
        resumed_from_session = True

    session_path = get_session_path(directory, output_dir)
    session = (
        prior
        if resumed_from_session and prior is not None
        else build_session(
            directory,
            effective_preset,
            effective_crf,
            overwrite,
            output_dir,
            jobs,
            policy=policy,
            on_file_failure="skip",
            use_calibration=use_calibration,
            retry_mode=retry_mode,
            queue_strategy=queue_strategy,
        )
    )
    save_session(session, session_path)
    preflight_warnings = _collect_preflight_warnings(jobs, ffprobe)
    _print_safe_interrupt_guidance()
    _print_preflight_warnings(preflight_warnings)
    _maybe_warn_low_disk_space(jobs, output_dir or directory, overwrite, assume_yes=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir if output_dir is not None else directory
    log_path = log_dir / f"mediashrink_{timestamp}.log"
    console.print(f"[dim]Overnight mode:[/dim] policy={policy}, on-file-failure=skip")
    console.print(f"[dim]Verbose log:[/dim] {log_path}")

    resume_completed, resume_skipped = _resume_summary_counts(
        session if resumed_from_session else None
    )
    to_encode = [job for job in jobs if not job.skip]
    if to_encode:
        results, stopped_early = _normalize_loop_result(
            _run_encode_loop(
                jobs,
                ffmpeg,
                ffprobe,
                display,
                session=session,
                session_path=session_path,
                log_path=log_path,
                stall_warning_seconds=float(stall_warning_seconds),
                resumed_from_session=resumed_from_session,
                previously_completed=resume_completed,
                previously_skipped=resume_skipped,
                on_file_failure="skip",
                use_calibration=use_calibration,
                retry_mode=retry_mode,
            )
        )
    else:
        results = _build_skipped_results(jobs)
        stopped_early = False
        console.print("[dim]All overnight candidates were skipped before encode start.[/dim]")
    _record_cleanup_results(session, session_path, results, [])
    json_report, text_report = _write_batch_reports(
        mode="overnight",
        base_dir=directory,
        output_dir=output_dir,
        manifest_path=None,
        preset=effective_preset,
        crf=effective_crf,
        overwrite=overwrite,
        cleanup_requested=False,
        resumed_from_session=resumed_from_session,
        session_path=session_path,
        started_at=started_at,
        finished_at=_now_iso(),
        results=results,
        cleaned_paths=[],
        log_path=log_path,
        warnings=preflight_warnings,
        policy=policy,
        on_file_failure="skip",
        retry_mode=retry_mode,
        queue_strategy=queue_strategy,
        split_followup_manifest=split_followup_manifest,
        estimate_miss_summary=_estimate_miss_summary(results),
    )
    console.print(f"[dim]Reports:[/dim] {json_report}  {text_report}")
    if stopped_early or any(not r.success and not r.skipped for r in results):
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@app.command()
def review(
    path: Path = typer.Argument(
        ...,
        help="Report JSON file or directory containing mediashrink_report_*.json files.",
        exists=True,
        readable=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the loaded report plus generated guidance as JSON.",
    ),
) -> None:
    report_path = _find_latest_report(path)
    if report_path is None:
        console.print(f"[red bold]Error:[/red bold] no mediashrink JSON report found in {path}")
        raise typer.Exit(code=EXIT_NO_FILES)

    payload = _load_report_payload(report_path)
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    guidance = _review_guidance(payload)

    if json_output:
        output = dict(payload)
        output["review_guidance"] = guidance
        print(json.dumps(output))
        return

    console.print(f"[bold]Run review[/bold] [dim]({report_path.name})[/dim]")
    console.print(
        f"Mode: {payload.get('mode', '-')}, policy: {payload.get('policy', '-')}, retry mode: {payload.get('retry_mode', '-')}, queue: {payload.get('queue_strategy', '-')}",
        highlight=False,
    )
    if payload.get("size_confidence") or payload.get("time_confidence"):
        console.print(
            f"Size confidence: {payload.get('size_confidence', '-')}, time confidence: {payload.get('time_confidence', '-')}",
            highlight=False,
        )
    console.print(
        f"Succeeded: {totals.get('succeeded', 0)}, failed: {totals.get('failed', 0)}, skipped incompatible: {totals.get('skipped_incompatible', 0)}, skipped by policy: {totals.get('skipped_by_policy', 0)}",
        highlight=False,
    )
    estimate_miss = payload.get("estimate_miss_summary")
    if isinstance(estimate_miss, str) and estimate_miss:
        console.print(f"Estimate miss: {estimate_miss}", highlight=False)
    grouped_incompatibilities = payload.get("grouped_incompatibilities")
    if isinstance(grouped_incompatibilities, list) and grouped_incompatibilities:
        console.print("[bold]Grouped incompatibilities[/bold]")
        for entry in grouped_incompatibilities[:5]:
            if not isinstance(entry, dict):
                continue
            reason = entry.get("reason", "incompatible")
            count = entry.get("count", 0)
            examples = entry.get("examples")
            suffix = (
                f" (examples: {', '.join(str(name) for name in examples[:3])})"
                if isinstance(examples, list) and examples
                else ""
            )
            console.print(f"  - {count}: {reason}{suffix}", highlight=False)
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        console.print("[bold]Warnings[/bold]")
        for warning in warnings[:5]:
            console.print(f"  - {warning}")
    console.print("[bold]Suggested next step[/bold]")
    for line in guidance:
        console.print(f"  - {line}")


@app.command()
def preview(
    file: Optional[Path] = typer.Argument(
        None,
        help=f"A supported video file ({supported_formats_label()}) to preview-encode.",
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    directory: Optional[Path] = typer.Option(
        None,
        "--directory",
        help="Preview up to three representative files from this directory instead of a single file.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    minutes: float = typer.Option(
        2.0,
        "--minutes",
        help="How many minutes to encode for the preview (default: 2).",
        min=0.1,
    ),
    crf: Optional[int] = typer.Option(
        None,
        "--crf",
        help="H.265 CRF quality value. Default: 20.",
        min=0,
        max=51,
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Encoding preset. Default: fast.",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load CRF/preset from a named profile.",
    ),
) -> None:
    """Test-encode a single file or a representative set before a full batch."""
    from mediashrink.scanner import SUPPORTED_EXTENSIONS

    if file is None and directory is None:
        console.print("[red bold]Error:[/red bold] provide a file or --directory.")
        raise typer.Exit(code=EXIT_NO_FILES)
    if file is not None and directory is not None:
        console.print("[red bold]Error:[/red bold] use either a file or --directory, not both.")
        raise typer.Exit(code=EXIT_NO_FILES)
    if file is not None and not file.exists():
        console.print(f"[red bold]Error:[/red bold] {file} does not exist.")
        raise typer.Exit(code=EXIT_NO_FILES)
    if file is not None and file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[red bold]Error:[/red bold] {file.name} is not a supported format "
            f"({supported_formats_label()})."
        )
        raise typer.Exit(code=EXIT_NO_FILES)

    display = EncodingDisplay(console)
    ffmpeg, ffprobe = _prepare_tools(None)
    effective_crf, effective_preset, _ = _resolve_encode_settings(profile, crf, preset)
    if directory is not None:
        files = scan_directory(directory, recursive=True)
        if not files:
            console.print(
                f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
            )
            raise typer.Exit(code=EXIT_NO_FILES)
        items = analyze_files(files, ffprobe)
        representative = select_representative_items(items, limit=3)
        console.print(
            f"[dim]Preview encoding first {minutes:.1f} minute(s) of {len(representative)} representative file(s) from[/dim] {directory}..."
        )
        results = [
            encode_preview(
                source=item.source,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                duration_minutes=minutes,
                crf=effective_crf,
                preset=effective_preset,
            )
            for item in representative
        ]
        display.show_summary(results)
        if any(not result.success for result in results):
            raise typer.Exit(code=EXIT_ENCODE_FAILURES)
        return

    assert file is not None
    console.print(f"[dim]Preview encoding first {minutes:.1f} minute(s) of[/dim] {file.name}...")
    result = encode_preview(
        source=file,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        duration_minutes=minutes,
        crf=effective_crf,
        preset=effective_preset,
    )
    display.show_summary([result])

    if not result.success:
        raise typer.Exit(code=EXIT_ENCODE_FAILURES)


@profiles_app.command("list")
def list_profiles() -> None:
    """List saved encoding profiles."""
    profiles = list_all_profiles()
    if not profiles:
        console.print("[dim]No saved profiles.[/dim]")
        raise typer.Exit(code=EXIT_SUCCESS)

    for profile in profiles:
        label = f" - {profile.label}" if profile.label else ""
        if profile.builtin:
            source = " [dim](builtin)[/dim]"
        elif profile.created_from_wizard:
            source = " (wizard)"
        else:
            source = ""
        console.print(f"{profile.name}: preset={profile.preset}, crf={profile.crf}{label}{source}")


@profiles_app.command("delete")
def delete_profile_cmd(name: str = typer.Argument(..., help="Profile name to delete.")) -> None:
    """Delete a saved encoding profile."""
    if not delete_profile(name):
        console.print(f"[red bold]Error:[/red bold] profile '{name}' was not found.")
        raise typer.Exit(code=EXIT_NO_FILES)
    console.print(f"[green]Deleted profile[/green] {name}")
