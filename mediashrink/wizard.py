from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from mediashrink.analysis import (
    adjust_time_confidence_for_scope,
    apply_duplicate_policy_to_items,
    analyze_files,
    build_manifest,
    collect_container_risk_signals,
    describe_estimate_calibration,
    describe_estimate_confidence,
    describe_size_confidence,
    describe_time_confidence,
    display_analysis_summary,
    estimate_value_range,
    estimate_analysis_confidence,
    estimate_analysis_encode_seconds,
    estimate_size_confidence,
    estimate_time_confidence,
    estimate_time_range_widening,
    format_tv_cohort_lines,
    maybe_priority_score,
    rank_maybe_candidates,
    save_manifest,
    select_representative_items,
    summarize_tv_cohorts,
    write_split_manifests,
    summarize_container_risks,
    describe_time_confidence_scope_adjustment,
)
from mediashrink.calibration import (
    bitrate_bucket,
    codec_family,
    describe_history_slices,
    estimate_display_uncertainty,
    estimate_failure_rate,
    format_family_container_summary,
    load_calibration_store,
    lookup_estimate,
    recent_bias_summary,
    resolution_bucket,
    summarize_calibration_store,
)
from mediashrink.constants import CRF_COMPRESSION_FACTOR
from mediashrink.encoder import (
    _HW_ENCODERS,
    describe_container_incompatibilities,
    describe_container_incompatibility,
    describe_output_container_constraints,
    encode_preview,
    estimate_output_size,
    get_duration_seconds,
    get_video_resolution,
    preflight_encode_job,
    probe_encoder_available,
    source_has_subtitle_streams,
    output_drops_subtitles,
    validate_encoder,
)
from mediashrink.models import AnalysisItem, EncodeJob, EncodeResult
from mediashrink.platform_utils import detect_device_labels, detect_os
from mediashrink.profiles import SavedProfile, get_builtin_profiles, upsert_profile
from mediashrink.scanner import (
    build_jobs,
    duplicate_policy_choices,
    parse_episode_grouping,
    scan_directory,
    supported_formats_label,
)

_GB = 1024**3
_MB = 1024**2

_CRF_COMPRESSION_FACTOR = CRF_COMPRESSION_FACTOR

_BENCHMARK_SECONDS = 8
_HARDWARE_DISPLAY_NAMES = {
    "qsv": "Intel Quick Sync",
    "nvenc": "Nvidia NVENC",
    "amf": "AMD AMF",
}
_HW_ENCODER_CAVEATS: dict[str, str] = {
    "nvenc": "NVENC: consumer GPUs allow max 3 concurrent encode sessions.",
    "amf": "AMF: quality may vary on older Radeon GPUs; test output before batch use.",
}
_DEFAULT_FALLBACK_PROFILE = ("faster", 22, "Fast", "faster")
_CONFIDENCE_LEVELS = ("Low", "Medium", "High")
_DEGRADED_PROMPT_THRESHOLD = 3
_LAST_WIZARD_REPORT_CONTEXT: dict[str, object] | None = None
_STRONGEST_MAYBE_LIMIT = 12
_LARGE_BATCH_FILE_THRESHOLD = 24
_OVERNIGHT_BATCH_SECONDS = 8 * 60 * 60


class _WizardFallbackRequested(Exception):
    """Raised when interactive input looks unreliable and we should switch modes."""


@dataclass
class WizardPromptRecord:
    prompt_id: str
    prompt_text: str
    raw_value: str
    normalized_value: str
    accepted: bool
    source: str
    note: str = ""


@dataclass
class WizardSessionState:
    console: Console
    directory: Path
    output_dir: Path | None
    debug_session_log: bool = False
    input_mode: str = "interactive"
    plain_prompts: bool = False
    last_prompt_had_newline: bool = True
    degraded_prompt_score: int = 0
    fallback_triggered: bool = False
    prompt_records: list[WizardPromptRecord] | None = None
    event_log: list[str] | None = None
    debug_log_path: Path | None = None
    last_io_source: str = "stdio"

    def __post_init__(self) -> None:
        if self.prompt_records is None:
            self.prompt_records = []
        if self.event_log is None:
            self.event_log = []

    def add_event(self, message: str) -> None:
        self.event_log.append(message)


def consume_last_wizard_report_context() -> dict[str, object] | None:
    global _LAST_WIZARD_REPORT_CONTEXT
    context = _LAST_WIZARD_REPORT_CONTEXT
    _LAST_WIZARD_REPORT_CONTEXT = None
    return context


_ACTIVE_WIZARD_SESSION: WizardSessionState | None = None


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= _GB:
        return f"{size_bytes / _GB:.2f} GB"
    return f"{size_bytes / _MB:.1f} MB"


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _get_wizard_session() -> WizardSessionState | None:
    return _ACTIVE_WIZARD_SESSION


def _note_prompt_boundary() -> None:
    session = _get_wizard_session()
    if session is None:
        return
    if not session.last_prompt_had_newline:
        session.console.print()
        session.last_prompt_had_newline = True
    try:
        sys.stdout.flush()
    except OSError:
        pass


def _echo_prompt_acceptance(label: str, value: str) -> None:
    session = _get_wizard_session()
    if session is None:
        return
    session.console.print(f"[dim]{label}:[/dim] {value}")
    session.last_prompt_had_newline = True


def _track_prompt_anomaly(note: str) -> None:
    session = _get_wizard_session()
    if session is None:
        return
    session.degraded_prompt_score += 1
    session.add_event(f"Prompt anomaly: {note}")
    if (
        session.degraded_prompt_score >= _DEGRADED_PROMPT_THRESHOLD
        and not session.fallback_triggered
    ):
        session.fallback_triggered = True
        raise _WizardFallbackRequested(note)


def _record_prompt(
    prompt_id: str,
    prompt_text: str,
    raw_value: str,
    normalized_value: str,
    *,
    accepted: bool,
    note: str = "",
) -> None:
    session = _get_wizard_session()
    if session is None:
        return
    session.prompt_records.append(
        WizardPromptRecord(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            raw_value=raw_value,
            normalized_value=normalized_value,
            accepted=accepted,
            source=session.last_io_source,
            note=note,
        )
    )
    if note:
        session.add_event(f"{prompt_id}: {note}")


def _write_debug_session_log() -> Path | None:
    session = _get_wizard_session()
    if session is None or not session.debug_session_log:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target_dir = session.output_dir or session.directory
    path = target_dir / f"mediashrink_wizard_debug_{timestamp}.log"
    lines = [
        "mediashrink wizard debug log",
        f"mode={session.input_mode}",
        f"plain_prompts={'yes' if session.plain_prompts else 'no'}",
        f"last_io_source={session.last_io_source}",
        f"fallback_triggered={'yes' if session.fallback_triggered else 'no'}",
        "",
        "events:",
    ]
    lines.extend(f"- {event}" for event in session.event_log)
    lines.append("")
    lines.append("prompts:")
    for record in session.prompt_records:
        lines.append(
            f"- id={record.prompt_id} source={record.source} accepted={'yes' if record.accepted else 'no'} raw={record.raw_value!r} normalized={record.normalized_value!r}"
            + (f" note={record.note}" if record.note else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    session.debug_log_path = path
    return path


def _wizard_readline(prompt_text: str) -> str:
    _note_prompt_boundary()
    input_stream = sys.stdin
    output_stream = sys.stdout
    opened_streams: list[object] = []

    os_name = detect_os()
    terminal_input: str | None = None
    terminal_output: str | None = None
    if os_name == "Windows":
        terminal_input = "CONIN$"
        terminal_output = "CONOUT$"
    elif os_name in {"Linux", "Darwin"}:
        terminal_input = "/dev/tty"
        terminal_output = "/dev/tty"

    if terminal_input is not None and terminal_output is not None:
        try:
            input_stream = open(
                terminal_input,
                "r",
                encoding=getattr(sys.stdin, "encoding", None) or "utf-8",
                errors="replace",
            )
            opened_streams.append(input_stream)
        except OSError:
            input_stream = sys.stdin
        try:
            output_stream = open(
                terminal_output,
                "w",
                encoding=getattr(sys.stdout, "encoding", None) or "utf-8",
                errors="replace",
            )
            opened_streams.append(output_stream)
        except OSError:
            output_stream = sys.stdout

    try:
        session = _get_wizard_session()
        if session is not None:
            if input_stream is sys.stdin and output_stream is sys.stdout:
                session.last_io_source = "stdio"
            elif os_name == "Windows":
                session.last_io_source = "conin/conout"
            else:
                session.last_io_source = "tty"
        output_stream.write(prompt_text)
        output_stream.flush()
        try:
            value = input_stream.readline()
        except KeyboardInterrupt as exc:
            raise typer.Abort() from exc
        if value == "":
            raise typer.Abort()
        if _get_wizard_session() is not None:
            _get_wizard_session().last_prompt_had_newline = value.endswith(("\n", "\r"))
        return value.rstrip("\r\n")
    finally:
        for stream in opened_streams:
            try:
                stream.close()
            except OSError:
                pass


def _wizard_prompt(
    text: str,
    default: str | None = None,
    *,
    show_default: bool = True,
    prompt_id: str = "prompt",
    acceptance_label: str = "Answer received",
) -> str:
    prompt_text = text
    if default is not None and show_default:
        prompt_text += f" [{default}]"
    prompt_text += ": "
    blank_attempts = 0
    while True:
        raw_value = _wizard_readline(prompt_text)
        normalized_value = raw_value.strip()
        if prompt_text.strip() and prompt_text.strip() in normalized_value:
            _record_prompt(
                prompt_id,
                prompt_text,
                raw_value,
                normalized_value,
                accepted=False,
                note="input included prompt text",
            )
            _track_prompt_anomaly(f"{prompt_id}: input included prompt text")
            normalized_value = normalized_value.replace(prompt_text.strip(), "").strip()
        if not normalized_value:
            blank_attempts += 1
            if default is not None:
                normalized_value = default
            elif blank_attempts >= 2:
                _record_prompt(
                    prompt_id,
                    prompt_text,
                    raw_value,
                    normalized_value,
                    accepted=False,
                    note="repeated blank input",
                )
                _track_prompt_anomaly(f"{prompt_id}: repeated blank input")
                continue
        else:
            blank_attempts = 0
        _record_prompt(prompt_id, prompt_text, raw_value, normalized_value, accepted=True)
        _echo_prompt_acceptance(acceptance_label, normalized_value)
        return normalized_value


def _wizard_confirm(
    text: str,
    *,
    default: bool = False,
    prompt_id: str = "confirm",
    acceptance_label: str = "Answer received",
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    blank_attempts = 0
    while True:
        raw_value = _wizard_readline(f"{text} {suffix}: ")
        value = raw_value.strip().lower()
        if not value:
            blank_attempts += 1
            if blank_attempts >= 2:
                _record_prompt(
                    prompt_id,
                    text,
                    raw_value,
                    "",
                    accepted=False,
                    note="repeated blank confirmation",
                )
                _track_prompt_anomaly(f"{prompt_id}: repeated blank confirmation")
            return default
        if value in {"y", "yes"}:
            _record_prompt(prompt_id, text, raw_value, "yes", accepted=True)
            _echo_prompt_acceptance(acceptance_label, "Yes")
            return True
        if value in {"n", "no"}:
            _record_prompt(prompt_id, text, raw_value, "no", accepted=True)
            _echo_prompt_acceptance(acceptance_label, "No")
            return False
        _record_prompt(
            prompt_id,
            text,
            raw_value,
            value,
            accepted=False,
            note="invalid confirmation input",
        )
        _track_prompt_anomaly(f"{prompt_id}: invalid confirmation input")


def _render_mode(console: Console, *, plain_output: bool = False) -> str:
    if plain_output or (not console.is_terminal and not getattr(console, "record", False)):
        return "plain"
    if console.width < 125:
        return "narrow"
    if console.width < 165:
        return "compact"
    return "wide"


@dataclass
class EncoderProfile:
    index: int
    intent_label: str
    name: str
    encoder_key: str
    crf: int
    sw_preset: str | None
    estimated_output_bytes: int
    estimated_encode_seconds: float
    quality_label: str
    is_recommended: bool
    why_choose: str = ""
    is_custom: bool = False
    is_builtin: bool = False
    compatible_count: int = 0
    incompatible_count: int = 0
    compatibility_summary: str = ""
    grouped_incompatibilities: dict[str, int] | None = None
    effective_input_bytes: int = 0
    size_uncertainty: float | None = None
    recommended_compatible_count: int = 0
    recommended_incompatible_count: int = 0
    outlier_hint: str | None = None


@dataclass
class ProfilePlanningResult:
    candidate_items: list[AnalysisItem]
    candidate_input_bytes: int
    candidate_media_seconds: float
    sample_item: AnalysisItem
    benchmark_items: list[AnalysisItem] = field(default_factory=list)
    sample_duration: float = 0.0
    preview_items: list[AnalysisItem] = field(default_factory=list)
    available_hw: list[str] = field(default_factory=list)
    benchmark_speeds: dict[str, float | None] = field(default_factory=dict)
    observed_probe_failures: dict[tuple[str, int], dict[Path, str]] = field(default_factory=dict)
    profiles: list[EncoderProfile] = field(default_factory=list)
    active_calibration: dict[str, object] | None = None
    size_error_by_preset: dict[str, float | None] = field(default_factory=dict)
    stage_messages: list[str] = field(default_factory=list)


_QUALITY_RANK = {
    "Acceptable": 0,
    "Good": 1,
    "Very good": 2,
    "Excellent": 3,
    "Visually lossless": 4,
}


def _quality_rank(label: str) -> int:
    return _QUALITY_RANK.get(label, -1)


def _grouped_incompatibility_summary(grouped: dict[str, int]) -> str:
    if not grouped:
        return ""
    parts = [
        f"{count} {reason}"
        for reason, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]
    return "; ".join(parts[:2])


def _format_grouped_reason_list(grouped: dict[str, int] | None) -> str:
    if not grouped:
        return "No likely follow-up."
    parts = [
        f"{count} {reason}"
        for reason, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]
    return "; ".join(parts[:2])


def _summarize_candidate_mix(items: list[AnalysisItem], limit: int = 3) -> str:
    grouped: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.source.suffix.lower() or "?", item.codec or "?")
        grouped[key] = grouped.get(key, 0) + 1
    if not grouped:
        return ""
    parts = []
    for (container, codec), count in sorted(grouped.items(), key=lambda item: (-item[1], item[0])):
        parts.append(f"{count} {container} {codec}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _compact_split_summary(
    selected_count: int,
    compatible_count: int,
    mkv_count: int,
    followup_count: int,
) -> str:
    parts = [f"{selected_count} selected -> {compatible_count} normal"]
    if mkv_count:
        parts.append(f"{mkv_count} MKV sidecar")
    if followup_count:
        parts.append(f"{followup_count} follow-up")
    return ", ".join(parts)


def _looks_like_escaped_filename(name: str) -> bool:
    if "&" not in name or ";" not in name:
        return False
    normalized = unescape(name)
    return normalized != name and normalized.strip()


def _collect_filename_hygiene_candidates(items: list[AnalysisItem]) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for item in items:
        source = item.source
        normalized_name = unescape(source.name)
        if normalized_name != source.name and normalized_name.strip():
            candidates.append((source, normalized_name))
    return candidates


def _maybe_fix_filename_hygiene(
    items: list[AnalysisItem],
    *,
    console: Console,
    active_auto: bool,
) -> list[dict[str, str]]:
    candidates = _collect_filename_hygiene_candidates(items)
    if not candidates:
        return []
    console.print(
        f"[yellow]Filename hygiene warning:[/yellow] {len(candidates)} file(s) look HTML-escaped or otherwise suspicious."
    )
    for source, normalized_name in candidates[:5]:
        console.print(f"  [dim]{source.name} -> {normalized_name}[/dim]")
    if active_auto:
        console.print(
            "[dim]Non-interactive mode will not rename files automatically. Review these names later if needed.[/dim]"
        )
        return [
            {"source": str(source), "suggested_name": normalized_name, "renamed": "no"}
            for source, normalized_name in candidates
        ]
    if not _wizard_confirm(
        "Rename suspicious filenames before encoding?",
        default=False,
        prompt_id="rename-filenames",
        acceptance_label="Filename rename decision",
    ):
        return [
            {"source": str(source), "suggested_name": normalized_name, "renamed": "no"}
            for source, normalized_name in candidates
        ]

    actions: list[dict[str, str]] = []
    for source, normalized_name in candidates:
        target = source.with_name(normalized_name)
        if target == source:
            continue
        try:
            source.rename(target)
        except OSError:
            actions.append(
                {
                    "source": str(source),
                    "suggested_name": normalized_name,
                    "renamed": "failed",
                }
            )
            continue
        for item in items:
            if item.source == source:
                item.source = target
        actions.append({"source": str(source), "suggested_name": normalized_name, "renamed": "yes"})
    return actions


def _auto_queue_strategy_for_items(items: list[AnalysisItem]) -> tuple[str, str]:
    if len(items) < 2:
        return "original", "Original order is fine for this run size."
    if any(item.source.suffix.lower() in {".mp4", ".m4v"} for item in items):
        return (
            "safe-first",
            "Safe-first reduces the chance of container-sensitive files blocking early progress.",
        )
    sizes = sorted((item.size_bytes for item in items), reverse=True)
    if (
        sizes
        and len(sizes) >= 3
        and sizes[0] >= max(sum(sizes) * 0.22, sizes[min(2, len(sizes) - 1)] * 1.5)
    ):
        return (
            "largest-first",
            "Largest-first should reclaim space earlier because a few files dominate the batch.",
        )
    return (
        "safe-first",
        "Safe-first favors lower-risk files first and is the best default for mixed TV batches.",
    )


def _cleanup_expectation_lines(
    jobs: list[EncodeJob],
    *,
    cleanup_after: bool,
) -> list[str]:
    if not jobs:
        return []
    true_mkv_sidecars = [
        job
        for job in jobs
        if job.output.suffix.lower() == ".mkv" and job.source.suffix.lower() != ".mkv"
    ]
    same_format_outputs = [
        job
        for job in jobs
        if job.output != job.source and job.output.suffix.lower() == job.source.suffix.lower()
    ]
    lines: list[str] = []
    if cleanup_after:
        if same_format_outputs:
            lines.append(
                f"Cleanup will restore {len(same_format_outputs)} same-format output"
                + ("s" if len(same_format_outputs) != 1 else "")
                + " to the original filename."
            )
        if true_mkv_sidecars:
            lines.append(
                f"{len(true_mkv_sidecars)} true MKV sidecar output"
                + ("s" if len(true_mkv_sidecars) != 1 else "")
                + " will get a separate end-of-run prompt about replacing the original non-MKV files."
            )
    else:
        if same_format_outputs:
            lines.append(
                "Cleanup is off, so same-format outputs will stay side-by-side until you review them."
            )
        if true_mkv_sidecars:
            lines.append(
                f"{len(true_mkv_sidecars)} file(s) are already planned as true MKV sidecars and will get a separate end-of-run replacement prompt."
            )
    return lines


def _queue_strategy_recommendation(items: list[AnalysisItem]) -> str | None:
    if len(items) < 2:
        return None
    total = sum(max(item.duration_seconds, 0.0) for item in items)
    if total <= 0:
        total = float(sum(item.size_bytes for item in items))
        dominant = max(items, key=lambda item: item.size_bytes)
        share = dominant.size_bytes / max(int(total), 1)
    else:
        dominant = max(items, key=lambda item: item.duration_seconds)
        share = dominant.duration_seconds / total
    if share < 0.45:
        return None
    return f"Largest-first queueing would front-load {dominant.source.name}, which looks like the longest runtime contributor."


def _should_multi_sample_benchmark(items: list[AnalysisItem]) -> bool:
    if len(items) < _LARGE_BATCH_FILE_THRESHOLD:
        return False
    episodes = [parse_episode_grouping(item.source) for item in items]
    valid = [episode for episode in episodes if episode is not None]
    if len(valid) != len(items) or not valid:
        return False
    return len({episode.show for episode in valid}) == 1


def _calibration_trust_line(calibration_store: dict[str, object] | None) -> str | None:
    summary = summarize_calibration_store(calibration_store)
    if not isinstance(summary, dict) or int(summary.get("records", 0) or 0) <= 0:
        return None
    line = (
        f"This machine has {summary['accepted_records']} accepted sample(s) and "
        f"{summary['rejected_records']} rejected safety-check sample(s)."
    )
    family_mix = format_family_container_summary(summary.get("family_container_summaries"))
    if family_mix:
        line += f" History mix: {family_mix}."
    bias_summary = summary.get("bias_summary")
    if isinstance(bias_summary, dict) and isinstance(bias_summary.get("summary"), str):
        line += f" Recent bias: {bias_summary['summary']}."
    return line


def _closest_history_line(
    calibration_store: dict[str, object] | None,
    *,
    preset: str,
    items: list[AnalysisItem],
) -> str | None:
    if not calibration_store or not items:
        return None
    history = describe_history_slices(
        calibration_store,
        preset=preset,
        containers={item.source.suffix.lower() or ".mkv" for item in items},
    )
    closest = history.get("closest_preset_history")
    if isinstance(closest, str) and closest:
        return f"Closest history: {closest}."
    return None


def _mkv_first_guidance(profile: EncoderProfile, *, recommended_count: int) -> str | None:
    grouped = profile.grouped_incompatibilities or {}
    if not grouped:
        return None
    mkv_reasons = {
        "unsupported copied audio codec",
        "unsupported copied subtitle codec",
        "attachment stream incompatibility",
        "auxiliary data stream incompatibility",
        "output header failure",
    }
    blocked = sum(count for reason, count in grouped.items() if reason in mkv_reasons)
    if blocked <= 0:
        return None
    if recommended_count > 0 and profile.compatible_count > (recommended_count // 2):
        return None
    return "MKV output first is the clearer default here because container/copied-stream issues, not encoder speed, are blocking most of this batch."


def _profile_predicted_incompatibility(
    item: AnalysisItem,
    profile: EncoderProfile,
    ffprobe: Path | None,
    failure_rate: float,
    *,
    container_incompatibility_cache: dict[Path, str | None] | None = None,
) -> str | None:
    output_suffix = item.source.suffix.lower() or ".mkv"
    if output_suffix in {".mp4", ".m4v"} and ffprobe is not None:
        reason = _cached_container_incompatibility(
            item.source,
            ffprobe,
            container_incompatibility_cache,
        )
        if reason is not None:
            if "audio codec copy" in reason:
                return "unsupported copied audio codec"
            if "subtitle" in reason:
                return "unsupported copied subtitle codec"
            if "attachment" in reason:
                return "attachment stream incompatibility"
            if "auxiliary data" in reason:
                return "auxiliary data stream incompatibility"
            return "output header failure"
    if (
        profile.encoder_key in _HW_ENCODERS
        and output_suffix in {".mp4", ".m4v"}
        and failure_rate >= 0.35
    ):
        return "hardware encoder startup failure"
    return None


def _downgrade_confidence(label: str, steps: int = 1) -> str:
    try:
        index = _CONFIDENCE_LEVELS.index(label)
    except ValueError:
        return label
    return _CONFIDENCE_LEVELS[max(index - steps, 0)]


def _estimate_selected_output_bytes(
    items: list[AnalysisItem],
    *,
    ffprobe: Path,
    preset: str,
    crf: int,
    use_calibration: bool,
    calibration_store: dict[str, object] | None,
) -> int:
    total = 0
    selected_size_error = _average_size_error_for_items(
        preset=preset,
        items=items,
        ffprobe=ffprobe,
        calibration_store=calibration_store,
    )
    for item in items:
        estimated = estimate_output_size(
            item.source,
            ffprobe,
            codec=item.codec,
            crf=crf,
            preset=preset,
            use_calibration=use_calibration,
            calibration_store=calibration_store,
        )
        if (
            preset in _HW_ENCODERS
            and use_calibration
            and selected_size_error is not None
            and abs(selected_size_error) >= 0.18
        ):
            heuristic_estimated = estimate_output_size(
                item.source,
                ffprobe,
                codec=item.codec,
                crf=crf,
                preset=preset,
                use_calibration=False,
                calibration_store=calibration_store,
            )
            if heuristic_estimated > 0 and estimated > 0:
                estimated = int((estimated * 0.30) + (heuristic_estimated * 0.70))
        total += estimated if estimated > 0 else item.estimated_output_bytes
    return total


def _post_split_confidence_labels(
    items: list[AnalysisItem],
    *,
    original_items: list[AnalysisItem],
    preset: str,
    use_calibration: bool,
    benchmarked_files: int,
    sidecar_count: int = 0,
    followup_count: int = 0,
) -> tuple[str, str]:
    size_conf = estimate_size_confidence(
        items,
        preset=preset,
        use_calibration=use_calibration,
    )
    time_conf = estimate_time_confidence(
        items,
        benchmarked_files=benchmarked_files,
        preset=preset,
        use_calibration=use_calibration,
    )
    time_conf = adjust_time_confidence_for_scope(
        time_conf,
        items,
        original_items=original_items,
        sidecar_count=sidecar_count,
        followup_count=followup_count,
    )
    if len(items) <= 1 < len(original_items):
        return _downgrade_confidence(size_conf), _downgrade_confidence(time_conf)
    if len(items) < len(original_items):
        size_codecs = {item.codec or "unknown" for item in items}
        original_codecs = {item.codec or "unknown" for item in original_items}
        if size_codecs != original_codecs or len(items) <= max(1, len(original_items) // 2):
            return _downgrade_confidence(size_conf), _downgrade_confidence(time_conf)
    return size_conf, time_conf


def _describe_selected_scope(
    selected_items: list[AnalysisItem],
    *,
    recommended_items: list[AnalysisItem],
    maybe_items: list[AnalysisItem],
) -> str:
    selected_sources = {item.source for item in selected_items}
    recommended_sources = {item.source for item in recommended_items}
    maybe_sources = {item.source for item in maybe_items}
    selected_maybe = len(selected_sources & maybe_sources)
    if selected_sources and selected_sources <= recommended_sources:
        return "recommended files only"
    if recommended_sources == set() and selected_maybe:
        return f"strongest maybe files ({selected_maybe} file(s))"
    if selected_maybe:
        return f"recommended files plus {selected_maybe} chosen maybe file(s)"
    return "selected files"


def _strongest_maybe_items(
    maybe_items: list[AnalysisItem], *, limit: int = _STRONGEST_MAYBE_LIMIT
) -> list[AnalysisItem]:
    ranked = rank_maybe_candidates(maybe_items, limit=limit)
    if len(ranked) <= 1:
        return ranked
    top_score = maybe_priority_score(ranked[0])
    shortlisted = [
        item for item in ranked if maybe_priority_score(item) >= max(3.8, top_score - 1.0)
    ]
    return shortlisted or ranked[:1]


def _large_batch_guidance(
    *,
    selected_count: int,
    total_candidates: int,
    estimated_seconds: float | None = None,
) -> str | None:
    if selected_count < _LARGE_BATCH_FILE_THRESHOLD and (
        estimated_seconds is None or estimated_seconds < _OVERNIGHT_BATCH_SECONDS
    ):
        return None
    left_out = max(total_candidates - selected_count, 0)
    if estimated_seconds is not None and estimated_seconds >= _OVERNIGHT_BATCH_SECONDS:
        return (
            f"This looks like an overnight-scale batch. Running {selected_count} now with {left_out} left out is sensible; "
            "split by season or smaller chunks if you want easier checkpoints."
        )
    return (
        f"This is a large batch ({selected_count} selected, {left_out} left out). "
        "Consider splitting by season or smaller chunks if you want quicker checkpoints."
    )


def _cohort_guidance_lines(
    *,
    selected_items: list[AnalysisItem],
    left_out_items: list[AnalysisItem],
) -> list[str]:
    lines: list[str] = []
    top_shows = format_tv_cohort_lines(selected_items, group_by="show", limit=2)
    top_seasons = format_tv_cohort_lines(selected_items, group_by="season", limit=2)
    if top_shows:
        lines.append("Selected shows: " + " | ".join(top_shows))
    if top_seasons:
        lines.append("Selected seasons: " + " | ".join(top_seasons))
    if left_out_items:
        left_show_count = len(summarize_tv_cohorts(left_out_items, group_by="show"))
        left_season_count = len(summarize_tv_cohorts(left_out_items, group_by="season"))
        if left_show_count or left_season_count:
            lines.append(
                f"Left out for later review: {left_show_count} show cohort(s), {left_season_count} season cohort(s)."
            )
    risky = [
        item
        for item in selected_items
        if item.source.suffix.lower() in {".mp4", ".m4v"} and (item.codec or "") == "h264"
    ]
    if risky:
        lines.append(
            f"Tail-risk hint: {len(risky)} MP4/H.264 file(s) in this selection may dominate the end of the batch."
        )
    return lines


def _select_risky_probe_items(
    items: list[AnalysisItem],
    ffprobe: Path,
    *,
    limit: int = 8,
    container_incompatibility_cache: dict[Path, str | None] | None = None,
) -> list[AnalysisItem]:
    risky: list[AnalysisItem] = []
    for item in items:
        suffix = item.source.suffix.lower()
        if suffix in {".mp4", ".m4v"}:
            risky.append(item)
            continue
        if (
            _cached_container_incompatibility(
                item.source,
                ffprobe,
                container_incompatibility_cache,
            )
            is not None
        ):
            risky.append(item)
    risky.sort(key=lambda item: item.size_bytes, reverse=True)
    return risky[:limit]


def _cached_container_incompatibility(
    source: Path,
    ffprobe: Path,
    cache: dict[Path, str | None] | None,
) -> str | None:
    if cache is None:
        return describe_container_incompatibility(source, source, ffprobe)
    if source not in cache:
        cache[source] = describe_container_incompatibility(source, source, ffprobe)
    return cache[source]


def _targeted_profile_probe_failures(
    *,
    items: list[AnalysisItem],
    profiles: list[EncoderProfile],
    ffmpeg: Path,
    ffprobe: Path,
    progress_callback: Callable[[str, int, int], None] | None = None,
    container_incompatibility_cache: dict[Path, str | None] | None = None,
) -> dict[tuple[str, int], dict[Path, str]]:
    risky_items = _select_risky_probe_items(
        items,
        ffprobe,
        container_incompatibility_cache=container_incompatibility_cache,
    )
    probe_targets = _iter_probe_targets(profiles)
    if not risky_items or not probe_targets:
        return {}
    failures: dict[tuple[str, int], dict[Path, str]] = {}

    def _probe(target: tuple[str, int]) -> tuple[tuple[str, int], dict[Path, str]]:
        preset, crf = target
        target_failures: dict[Path, str] = {}
        for item in risky_items:
            try:
                result = preflight_encode_job(
                    item.source,
                    ffmpeg,
                    ffprobe,
                    crf=crf,
                    preset=preset,
                )
            except (OSError, ValueError):
                continue
            if result.success:
                continue
            target_failures[item.source] = result.error_message or ""
        return target, target_failures

    completed = 0
    with ThreadPoolExecutor(max_workers=min(4, len(probe_targets))) as executor:
        future_map = {executor.submit(_probe, target): target for target in probe_targets}
        for future in as_completed(future_map):
            target, target_failures = future.result()
            if target_failures:
                failures[target] = target_failures
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    "Smoke-probing risky container/profile combinations",
                    completed,
                    len(probe_targets),
                )
    return failures


def _adjust_profile_speed_with_calibration(
    *,
    base_speed: float | None,
    preset: str,
    items: list[AnalysisItem],
    ffprobe: Path | None,
    calibration_store: dict[str, object] | None,
) -> float | None:
    if (
        base_speed is None
        or base_speed <= 0
        or ffprobe is None
        or not items
        or not calibration_store
    ):
        return base_speed
    factors: list[float] = []
    for item in items[:5]:
        width, height = get_video_resolution(item.source, ffprobe)
        if width <= 0 or height <= 0:
            continue
        lookup = lookup_estimate(
            calibration_store,
            codec=item.codec,
            resolution=resolution_bucket(width, height),
            bitrate=bitrate_bucket(item.bitrate_kbps),
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if lookup is None or lookup.average_speed_error is None:
            continue
        factors.append(max(0.2, 1.0 + lookup.average_speed_error))
    if not factors:
        return base_speed
    return base_speed * (sum(factors) / len(factors))


def _emit_stage_progress(
    stage: str,
    current: int,
    total: int,
    *,
    console: Console | None = None,
    stage_messages: list[str] | None = None,
    stage_callback: Callable[[str, str, int | None, int | None], None] | None = None,
) -> None:
    message = f"{stage}... {current}/{total}"
    if stage_messages is not None and (not stage_messages or stage_messages[-1] != message):
        stage_messages.append(message)
    if stage_callback is not None:
        stage_callback(stage, message, current, total)
    if console is not None:
        console.print(f"[dim]{message}[/dim]")


def _emit_stage_status(
    message: str,
    *,
    console: Console | None = None,
    stage_messages: list[str] | None = None,
    stage_callback: Callable[[str, str, int | None, int | None], None] | None = None,
) -> None:
    if stage_messages is not None and (not stage_messages or stage_messages[-1] != message):
        stage_messages.append(message)
    if stage_callback is not None:
        stage_callback(message, message, None, None)
    if console is not None:
        console.print(f"[dim]{message}[/dim]")


def _iter_probe_targets(profiles: list[EncoderProfile]) -> list[tuple[str, int]]:
    probe_targets: list[tuple[str, int]] = []
    seen_targets: set[tuple[str, int]] = set()
    for profile in profiles:
        if profile.is_custom or not profile.encoder_key:
            continue
        target = (profile.encoder_key, profile.crf)
        if target in seen_targets:
            continue
        seen_targets.add(target)
        probe_targets.append(target)
    return probe_targets


def _predict_profile_compatibility(
    *,
    profile: EncoderProfile,
    items: list[AnalysisItem],
    ffprobe: Path,
    failure_rate: float,
    observed_probe_failures: dict[tuple[str, int], dict[Path, str]] | None = None,
    container_incompatibility_cache: dict[Path, str | None] | None = None,
) -> tuple[int, int, dict[str, int]]:
    grouped: dict[str, int] = {}
    compatible_count = 0
    observed_for_profile = (observed_probe_failures or {}).get(
        (profile.encoder_key, profile.crf), {}
    )
    for item in items:
        observed_reason = observed_for_profile.get(item.source)
        if observed_reason:
            lowered = observed_reason.lower()
            if "audio codec copy" in lowered:
                reason = "unsupported copied audio codec"
            elif "subtitle" in lowered or "mov_text" in lowered:
                reason = "unsupported copied subtitle codec"
            elif "attachment" in lowered:
                reason = "attachment stream incompatibility"
            elif "data" in lowered or "bin_data" in lowered:
                reason = "auxiliary data stream incompatibility"
            elif "could not open encoder before eof" in lowered or "internal bug" in lowered:
                reason = "hardware encoder startup failure"
            else:
                reason = "output header failure"
        else:
            reason = _profile_predicted_incompatibility(
                item,
                profile,
                ffprobe,
                failure_rate,
                container_incompatibility_cache=container_incompatibility_cache,
            )
        if reason is None:
            compatible_count += 1
        else:
            grouped[reason] = grouped.get(reason, 0) + 1
    return compatible_count, sum(grouped.values()), grouped


def _format_ready_size_estimate(
    *,
    input_bytes: int,
    output_bytes: int,
    confidence: str | None,
    size_error: float | None,
) -> str:
    output_low_f, output_high_f = estimate_value_range(
        float(output_bytes),
        confidence=confidence,
        average_error=size_error,
    )
    output_low = max(0, int(output_low_f))
    output_high = max(output_low, int(output_high_f))
    saved_low = max(input_bytes - output_high, 0)
    saved_high = max(input_bytes - output_low, 0)
    saved_pct_low = saved_low / input_bytes * 100 if input_bytes else 0.0
    saved_pct_high = saved_high / input_bytes * 100 if input_bytes else 0.0
    if (saved_pct_high - saved_pct_low) >= 40:
        return "  Est. out: ~highly variable"
    return (
        f"  Est. out: ~{_fmt_size(output_low)}-{_fmt_size(output_high)}  "
        f"(~{_fmt_size(saved_low)}-{_fmt_size(saved_high)} saved, ~{saved_pct_low:.0f}-{saved_pct_high:.0f}%)"
    )


def _average_size_error_for_items(
    *,
    preset: str,
    items: list[AnalysisItem],
    ffprobe: Path | None,
    calibration_store: dict[str, object] | None,
) -> float | None:
    if ffprobe is None or not items or not calibration_store:
        return None
    errors: list[float] = []
    for item in items[:8]:
        width, height = get_video_resolution(item.source, ffprobe)
        lookup = lookup_estimate(
            calibration_store,
            codec=item.codec,
            resolution=resolution_bucket(width, height) if width > 0 and height > 0 else "unknown",
            bitrate=bitrate_bucket(item.bitrate_kbps),
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if lookup is not None and lookup.average_size_error is not None:
            errors.append(lookup.average_size_error)
    if not errors:
        return None
    return sum(errors) / len(errors)


def _average_speed_error_for_items(
    *,
    preset: str,
    items: list[AnalysisItem],
    ffprobe: Path | None,
    calibration_store: dict[str, object] | None,
) -> float | None:
    if ffprobe is None or not items or not calibration_store:
        return None
    errors: list[float] = []
    for item in items[:8]:
        width, height = get_video_resolution(item.source, ffprobe)
        lookup = lookup_estimate(
            calibration_store,
            codec=item.codec,
            resolution=resolution_bucket(width, height) if width > 0 and height > 0 else "unknown",
            bitrate=bitrate_bucket(item.bitrate_kbps),
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if lookup is not None and lookup.average_speed_error is not None:
            errors.append(lookup.average_speed_error)
    if not errors:
        return None
    return sum(errors) / len(errors)


def _format_duration_range(low_seconds: float, high_seconds: float) -> str:
    return f"~{_fmt_duration(low_seconds)}-{_fmt_duration(high_seconds)}"


def _summarize_mkv_suitable_candidates(
    items: list[AnalysisItem],
    ffprobe: Path,
) -> tuple[int, dict[str, int], list[str]]:
    grouped: dict[str, int] = {}
    examples: list[str] = []
    for item in items:
        if item.source.suffix.lower() not in {".mp4", ".m4v"}:
            continue
        notes = describe_output_container_constraints(item.source, item.source, ffprobe)
        actionable = [
            note
            for note in notes
            if note.startswith("subtitle")
            or note.startswith("attachment")
            or note.startswith("auxiliary")
            or note.startswith("audio copy")
        ]
        if not actionable:
            continue
        examples.append(item.source.name)
        for note in actionable:
            if note.startswith("subtitle"):
                label = "subtitle streams would be dropped in MP4/M4V output"
            elif note.startswith("attachment"):
                label = "attachment streams need MKV output"
            elif note.startswith("auxiliary"):
                label = "auxiliary data streams need MKV output"
            else:
                label = "some copied audio streams may need MKV output or re-encode"
            grouped[label] = grouped.get(label, 0) + 1
    return len(examples), grouped, examples[:3]


def _default_mkv_followup_dir(directory: Path, output_dir: Path | None) -> Path:
    base = output_dir if output_dir is not None else directory
    return base / "mediashrink_mkv_followup"


def _build_mkv_followup_jobs(
    items: list[AnalysisItem],
    *,
    output_dir: Path,
    overwrite: bool,
    crf: int,
    preset: str,
    ffprobe: Path,
    no_skip: bool,
) -> list[EncodeJob]:
    source_set = {item.source for item in items}
    jobs = build_jobs(
        files=list(source_set),
        output_dir=output_dir,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
        dry_run=False,
        ffprobe=ffprobe,
        no_skip=no_skip,
    )
    filtered_jobs = [job for job in jobs if job.source in source_set]
    for job in filtered_jobs:
        mkv_name = job.source.with_suffix(".mkv").name
        job.output = output_dir / mkv_name
        job.tmp_output = job.output.parent / f".tmp_{job.output.stem}{job.output.suffix}"
    return filtered_jobs


def prepare_profile_planning(
    *,
    analysis_items: list[AnalysisItem],
    ffmpeg: Path,
    ffprobe: Path,
    policy: str = "fastest-wall-clock",
    use_calibration: bool = True,
    console: Console | None = None,
    available_hw: list[str] | None = None,
    stage_callback: Callable[[str, str, int | None, int | None], None] | None = None,
) -> ProfilePlanningResult | None:
    recommended_items = [item for item in analysis_items if item.recommendation == "recommended"]
    maybe_items = [item for item in analysis_items if item.recommendation == "maybe"]
    candidate_items = recommended_items + maybe_items
    if not candidate_items:
        return None

    candidate_input_bytes = sum(item.size_bytes for item in candidate_items)
    candidate_media_seconds = _sum_item_durations(candidate_items)
    sample_pool = recommended_items or maybe_items or analysis_items
    representative_pool = select_representative_items(sample_pool, limit=3) or sample_pool
    benchmark_items = (
        representative_pool[:3]
        if _should_multi_sample_benchmark(candidate_items)
        else representative_pool[:1]
    )
    sample_item = max(
        benchmark_items,
        key=lambda item: (
            item.duration_seconds if item.duration_seconds > 0 else 0.0,
            item.size_bytes,
        ),
    )
    sample_duration = sample_item.duration_seconds if sample_item.duration_seconds > 0 else 3600.0
    preview_items = select_representative_items(candidate_items or sample_pool, limit=3)
    detect_console = console if console is not None else Console(quiet=True)
    if available_hw is None:
        available_hw = detect_available_encoders(
            ffmpeg,
            detect_console,
            sample_file=sample_item.source,
            ffprobe=ffprobe,
        )

    candidates_to_bench = list(available_hw) + ["fast", "faster"]
    benchmark_speeds: dict[str, float | None] = {}
    stage_messages: list[str] = []

    active_calibration = load_calibration_store() if use_calibration else None
    container_incompatibility_cache: dict[Path, str | None] = {}

    def _benchmark_target(key: str) -> tuple[str, float | None]:
        sample_speeds: list[float] = []
        sample_duration_target = max(
            (item.duration_seconds if item.duration_seconds > 0 else sample_duration)
            for item in benchmark_items
        )
        for benchmark_item in benchmark_items:
            speed = benchmark_encoder(
                encoder_key=key,
                sample_file=benchmark_item.source,
                sample_duration=min(
                    benchmark_item.duration_seconds
                    if benchmark_item.duration_seconds > 0
                    else sample_duration_target,
                    sample_duration_target,
                ),
                crf=20,
                ffmpeg=ffmpeg,
            )
            if speed is not None and speed > 0:
                sample_speeds.append(speed)
        return (
            key,
            (sum(sample_speeds) / len(sample_speeds)) if sample_speeds else None,
        )

    if console is not None and candidates_to_bench:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            console=console,
            transient=False,
            expand=True,
        ) as progress:
            task = progress.add_task("Benchmarking profiles...", total=len(candidates_to_bench))
            completed = 0
            with ThreadPoolExecutor(max_workers=min(4, len(candidates_to_bench))) as executor:
                future_map = {
                    executor.submit(_benchmark_target, key): key for key in candidates_to_bench
                }
                for future in as_completed(future_map):
                    key, speed = future.result()
                    benchmark_speeds[key] = speed
                    completed += 1
                    _emit_stage_progress(
                        "Benchmarking profiles",
                        completed,
                        len(candidates_to_bench),
                        stage_messages=stage_messages,
                        stage_callback=stage_callback,
                    )
                    progress.update(task, completed=completed, total=len(candidates_to_bench))
        benchmark_speeds = {key: benchmark_speeds.get(key) for key in candidates_to_bench}
        _emit_stage_status(
            "Building provisional profiles...",
            console=console,
            stage_messages=stage_messages,
            stage_callback=stage_callback,
        )

        provisional_profiles = build_profiles(
            available_hw=available_hw,
            benchmark_speeds=benchmark_speeds,
            total_media_seconds=candidate_media_seconds,
            total_input_bytes=candidate_input_bytes,
            candidate_items=candidate_items,
            ffprobe=ffprobe,
            policy=policy,
            use_calibration=use_calibration,
            calibration_store=active_calibration,
            container_incompatibility_cache=container_incompatibility_cache,
        )
        probe_targets = _iter_probe_targets(provisional_profiles)
        if probe_targets:
            _emit_stage_status(
                "Preparing smoke probes...",
                console=console,
                stage_messages=stage_messages,
                stage_callback=stage_callback,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                console=console,
                transient=False,
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "Smoke-probing risky container/profile combinations...",
                    total=max(len(probe_targets), 1),
                )

                def _update_probe_progress(stage: str, current: int, total: int) -> None:
                    _emit_stage_progress(
                        stage,
                        current,
                        total,
                        stage_messages=stage_messages,
                        stage_callback=stage_callback,
                    )
                    progress.update(task, completed=current, total=max(total, 1))

                observed_probe_failures = _targeted_profile_probe_failures(
                    items=candidate_items,
                    profiles=provisional_profiles,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    progress_callback=_update_probe_progress,
                    container_incompatibility_cache=container_incompatibility_cache,
                )
        else:
            observed_probe_failures = {}
        _emit_stage_status(
            "Scoring recommendations...",
            console=console,
            stage_messages=stage_messages,
            stage_callback=stage_callback,
        )
    else:
        if candidates_to_bench:
            _emit_stage_progress(
                "Benchmarking profiles",
                0,
                len(candidates_to_bench),
                stage_messages=stage_messages,
                stage_callback=stage_callback,
            )
            completed = 0
            with ThreadPoolExecutor(max_workers=min(4, len(candidates_to_bench))) as executor:
                future_map = {
                    executor.submit(_benchmark_target, key): key for key in candidates_to_bench
                }
                for future in as_completed(future_map):
                    key, speed = future.result()
                    benchmark_speeds[key] = speed
                    completed += 1
                    _emit_stage_progress(
                        "Benchmarking profiles",
                        completed,
                        len(candidates_to_bench),
                        stage_messages=stage_messages,
                        stage_callback=stage_callback,
                    )
            benchmark_speeds = {key: benchmark_speeds.get(key) for key in candidates_to_bench}
            _emit_stage_status(
                "Building provisional profiles...",
                console=console,
                stage_messages=stage_messages,
                stage_callback=stage_callback,
            )

        provisional_profiles = build_profiles(
            available_hw=available_hw,
            benchmark_speeds=benchmark_speeds,
            total_media_seconds=candidate_media_seconds,
            total_input_bytes=candidate_input_bytes,
            candidate_items=candidate_items,
            ffprobe=ffprobe,
            policy=policy,
            use_calibration=use_calibration,
            calibration_store=active_calibration,
            container_incompatibility_cache=container_incompatibility_cache,
        )
        if _iter_probe_targets(provisional_profiles):
            _emit_stage_status(
                "Preparing smoke probes...",
                console=console,
                stage_messages=stage_messages,
                stage_callback=stage_callback,
            )
        observed_probe_failures = _targeted_profile_probe_failures(
            items=candidate_items,
            profiles=provisional_profiles,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            progress_callback=lambda stage, current, total: _emit_stage_progress(
                stage,
                current,
                total,
                stage_messages=stage_messages,
                stage_callback=stage_callback,
            ),
            container_incompatibility_cache=container_incompatibility_cache,
        )
        _emit_stage_status(
            "Scoring recommendations...",
            console=console,
            stage_messages=stage_messages,
            stage_callback=stage_callback,
        )
    profiles = build_profiles(
        available_hw=available_hw,
        benchmark_speeds=benchmark_speeds,
        total_media_seconds=candidate_media_seconds,
        total_input_bytes=candidate_input_bytes,
        candidate_items=candidate_items,
        ffprobe=ffprobe,
        policy=policy,
        use_calibration=use_calibration,
        calibration_store=active_calibration,
        observed_probe_failures=observed_probe_failures,
        container_incompatibility_cache=container_incompatibility_cache,
    )
    _emit_stage_status(
        "Preparing profile table...",
        console=console,
        stage_messages=stage_messages,
    )

    seen_presets: set[str] = set()
    size_error_by_preset: dict[str, float | None] = {}
    for profile in profiles:
        key = profile.encoder_key
        if key in seen_presets or profile.is_custom:
            continue
        seen_presets.add(key)
        errors: list[float] = []
        for item in candidate_items[:5]:
            estimate = lookup_estimate(
                active_calibration,
                codec=item.codec,
                resolution="unknown",
                bitrate="unknown",
                preset=key,
                container=item.source.suffix.lower() or ".mkv",
            )
            if estimate is not None and estimate.average_size_error is not None:
                errors.append(estimate.average_size_error)
        size_error_by_preset[key] = sum(errors) / len(errors) if errors else None

    return ProfilePlanningResult(
        candidate_items=candidate_items,
        candidate_input_bytes=candidate_input_bytes,
        candidate_media_seconds=candidate_media_seconds,
        sample_item=sample_item,
        benchmark_items=benchmark_items,
        sample_duration=sample_duration,
        preview_items=preview_items,
        available_hw=available_hw,
        benchmark_speeds=benchmark_speeds,
        observed_probe_failures=observed_probe_failures,
        profiles=profiles,
        active_calibration=active_calibration,
        size_error_by_preset=size_error_by_preset,
        stage_messages=stage_messages,
    )


def _is_profile_dominated(profile: EncoderProfile, peers: list[EncoderProfile]) -> bool:
    if profile.is_custom or profile.estimated_encode_seconds <= 0:
        return False

    profile_saved = profile.estimated_output_bytes
    profile_quality = _quality_rank(profile.quality_label)
    for peer in peers:
        if peer is profile or peer.is_custom or peer.estimated_encode_seconds <= 0:
            continue
        peer_saved = peer.estimated_output_bytes
        peer_quality = _quality_rank(peer.quality_label)
        if (
            peer.compatible_count >= profile.compatible_count
            and peer.estimated_encode_seconds <= profile.estimated_encode_seconds
            and peer_saved <= profile_saved
            and peer_quality >= profile_quality
            and (
                peer.compatible_count > profile.compatible_count
                or peer.estimated_encode_seconds < profile.estimated_encode_seconds
                or peer_saved < profile_saved
                or peer_quality > profile_quality
            )
        ):
            return True
    return False


def _policy_sort_key(
    profile: EncoderProfile,
    *,
    policy: str,
    failure_rate: float,
) -> tuple[float, ...]:
    hardware_bias = 0.0 if profile.encoder_key in _HW_ENCODERS else 1.0
    quality_bias = float(_quality_rank(profile.quality_label) * -1)
    compatibility_penalty = float(max(profile.incompatible_count, 0))
    if profile.compatible_count > 0 and profile.incompatible_count >= profile.compatible_count:
        compatibility_penalty += 2.0
    if (
        profile.recommended_compatible_count > 0
        and profile.recommended_incompatible_count >= profile.recommended_compatible_count
    ):
        compatibility_penalty += 2.0
    size_uncertainty_penalty = (
        abs(profile.size_uncertainty)
        if profile.size_uncertainty is not None and abs(profile.size_uncertainty) >= 0.18
        else 0.0
    )
    effective_wait = float(profile.estimated_encode_seconds or 0.0)
    effective_size = float(profile.estimated_output_bytes or 0.0)
    if policy == "best-compression":
        return (
            compatibility_penalty,
            size_uncertainty_penalty,
            effective_size,
            quality_bias,
            effective_wait,
            failure_rate,
        )
    if policy == "lowest-cpu":
        return (
            compatibility_penalty,
            hardware_bias,
            failure_rate,
            effective_wait,
            effective_size,
        )
    if policy == "highest-confidence":
        return (
            compatibility_penalty,
            failure_rate,
            size_uncertainty_penalty,
            hardware_bias,
            effective_wait,
            quality_bias,
            effective_size,
        )
    return (
        compatibility_penalty,
        size_uncertainty_penalty,
        effective_wait,
        failure_rate,
        quality_bias,
        effective_size,
    )


def _is_highly_variable_profile(profile: EncoderProfile) -> bool:
    return (
        profile.encoder_key in _HW_ENCODERS
        and profile.size_uncertainty is not None
        and abs(profile.size_uncertainty) >= 0.25
    )


def _prefer_stable_software_alternative(
    recommended: EncoderProfile,
    candidates: list[EncoderProfile],
) -> EncoderProfile:
    if not _is_highly_variable_profile(recommended):
        return recommended

    stable_software_candidates = [
        profile
        for profile in candidates
        if profile.encoder_key not in _HW_ENCODERS
        and profile.compatible_count >= recommended.compatible_count
        and profile.incompatible_count <= recommended.incompatible_count
        and profile.size_uncertainty is not None
        and abs(profile.size_uncertainty) < 0.18
    ]
    if not stable_software_candidates:
        stable_software_candidates = [
            profile
            for profile in candidates
            if profile.encoder_key not in _HW_ENCODERS
            and profile.compatible_count >= recommended.compatible_count
            and profile.incompatible_count <= recommended.incompatible_count
            and (profile.size_uncertainty is None or abs(profile.size_uncertainty) < 0.25)
        ]
    if not stable_software_candidates:
        return recommended

    return min(
        stable_software_candidates,
        key=lambda profile: (
            profile.estimated_encode_seconds,
            -_quality_rank(profile.quality_label),
            profile.estimated_output_bytes,
        ),
    )


def _select_recommended_profile(
    profiles: list[EncoderProfile],
    *,
    policy: str = "fastest-wall-clock",
    failure_rates: dict[str, float] | None = None,
) -> EncoderProfile | None:
    candidates = [
        profile
        for profile in profiles
        if not profile.is_custom
        and profile.estimated_encode_seconds > 0
        and not _is_profile_dominated(profile, profiles)
    ]
    if not candidates:
        return None
    recommended = min(
        candidates,
        key=lambda profile: _policy_sort_key(
            profile,
            policy=policy,
            failure_rate=(failure_rates or {}).get(profile.encoder_key, 0.0),
        ),
    )
    if policy == "fastest-wall-clock":
        return _prefer_stable_software_alternative(recommended, candidates)
    return recommended


def detect_available_encoders(
    ffmpeg: Path,
    console: Console,
    sample_file: Path | None = None,
    ffprobe: Path | None = None,
) -> list[str]:
    """Return available hardware encoder keys in stable display order.

    If sample_file and ffprobe are provided, each probed encoder is validated
    against real content to catch encoders that pass the synthetic test but
    fail on actual video input.
    """
    candidates = list(_HW_ENCODERS.keys())

    with console.status("[dim]Detecting hardware encoders...[/dim]", spinner="dots"):
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {pool.submit(probe_encoder_available, key, ffmpeg): key for key in candidates}
            detected: set[str] = set()
            for future in as_completed(futures):
                key = futures[future]
                try:
                    if future.result():
                        detected.add(key)
                except Exception:
                    pass

    # Validation pass — encode 3s of real content to catch broken encoders
    if sample_file is not None and ffprobe is not None:
        validated: set[str] = set()
        for key in detected:
            ok, err = validate_encoder(key, sample_file, ffmpeg, ffprobe)
            if ok:
                validated.add(key)
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] {_HARDWARE_DISPLAY_NAMES.get(key, key)} "
                    f"passed availability check but failed validation ({err}). Skipping."
                )
        detected = validated

    available: list[str] = []
    for key in ("qsv", "nvenc", "amf"):
        if key in detected:
            available.append(key)
    return available


def benchmark_encoder(
    encoder_key: str,
    sample_file: Path,
    sample_duration: float,
    crf: int,
    ffmpeg: Path,
) -> float | None:
    """Return benchmark speed as media-seconds per wall-second."""
    clip_len = min(_BENCHMARK_SECONDS, sample_duration * 0.9)
    if clip_len <= 0:
        return None
    seek_pos = max(sample_duration * 0.2, 0.0)

    if encoder_key in _HW_ENCODERS:
        encoder_name, quality_flag, extra_flags = _HW_ENCODERS[encoder_key]
        if encoder_key == "amf":
            quality_args = ["-qp_i", str(crf), "-qp_p", str(crf)]
        else:
            quality_args = [quality_flag, str(crf)]
        video_flags = ["-c:v", encoder_name] + quality_args + extra_flags
    else:
        video_flags = ["-c:v", "libx265", "-crf", str(crf), "-preset", encoder_key]

    cmd = (
        [
            str(ffmpeg),
            "-ss",
            str(seek_pos),
            "-i",
            str(sample_file),
            "-t",
            str(clip_len),
        ]
        + video_flags
        + [
            "-an",
            "-f",
            "null",
            "-",
            "-loglevel",
            "error",
        ]
    )

    try:
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.monotonic() - start
        if result.returncode != 0 or elapsed < 0.1:
            return None
        return clip_len / elapsed
    except (subprocess.TimeoutExpired, OSError):
        return None


def _estimate_output_bytes(total_input_bytes: int, crf: int) -> int:
    nearest = min(_CRF_COMPRESSION_FACTOR, key=lambda k: abs(k - crf))
    return int(total_input_bytes * _CRF_COMPRESSION_FACTOR[nearest])


def _estimate_time(total_media_seconds: float, speed: float | None) -> float:
    if speed is None or speed <= 0:
        return 0.0
    return total_media_seconds / speed


def _sum_item_durations(items: list[AnalysisItem]) -> float:
    total = 0.0
    fallback_count = 0

    for item in items:
        if item.duration_seconds > 0:
            total += item.duration_seconds
        else:
            fallback_count += 1

    if total <= 0 and items:
        return 3600.0 * len(items)

    if fallback_count:
        avg_known = total / max(len(items) - fallback_count, 1)
        total += avg_known * fallback_count

    return total


def _sum_media_durations(files: list[Path], ffprobe: Path) -> float:
    total = 0.0
    fallback_count = 0

    for path in files:
        duration = get_duration_seconds(path, ffprobe)
        if duration > 0:
            total += duration
        else:
            fallback_count += 1

    if total <= 0 and files:
        return 3600.0 * len(files)

    if fallback_count:
        avg_known = total / max(len(files) - fallback_count, 1)
        total += avg_known * fallback_count

    return total


_ENCODER_LABEL_MAX_CHARS = 22


def _encoder_display_name(
    encoder_key: str, device_labels: dict[str, str], truncate: bool = False
) -> str:
    if encoder_key in _HW_ENCODERS:
        label = device_labels.get(encoder_key)
        if label:
            if truncate and len(label) > _ENCODER_LABEL_MAX_CHARS:
                label = label[:_ENCODER_LABEL_MAX_CHARS].rstrip() + "…"
            return f"{_HARDWARE_DISPLAY_NAMES[encoder_key]} ({label})"
        return _HARDWARE_DISPLAY_NAMES[encoder_key]
    return f"libx265 ({encoder_key})"


def _profile_why_choose(
    profile: EncoderProfile,
    recommended: EncoderProfile | None,
    *,
    fastest: EncoderProfile | None = None,
) -> str:
    highly_variable = (
        profile.encoder_key in _HW_ENCODERS
        and profile.size_uncertainty is not None
        and abs(profile.size_uncertainty) >= 0.25
    )
    if profile.is_custom:
        return "Manual override for exact settings."
    if recommended is profile:
        if profile.incompatible_count:
            mkv_first = _mkv_first_guidance(
                profile,
                recommended_count=profile.recommended_compatible_count
                + profile.recommended_incompatible_count,
            )
            outlier_note = f" {profile.outlier_hint}" if profile.outlier_hint else ""
            if highly_variable:
                return (
                    f"Partial-batch default only: {profile.compatible_count} file(s) can run now, "
                    f"{profile.incompatible_count} likely need follow-up, and output size is highly variable on similar files.{outlier_note}"
                )
            if mkv_first:
                return (
                    f"Partial-batch default only: {profile.compatible_count} file(s) can run now, "
                    f"{profile.incompatible_count} likely need follow-up. {mkv_first}{outlier_note}"
                )
            return (
                f"Partial-batch default only: {profile.compatible_count} file(s) can run now, "
                f"while {profile.incompatible_count} likely need follow-up.{outlier_note}"
            )
        if highly_variable:
            return (
                f"Fastest wall-clock option for {profile.compatible_count} file(s), "
                "but output size is highly variable on similar files."
            )
        if (
            fastest is not None
            and fastest is not profile
            and fastest.estimated_encode_seconds > 0
            and profile.compatible_count > fastest.compatible_count
        ):
            return (
                f"Fastest wait: {fastest.name}, but {profile.name} covers "
                f"{profile.compatible_count} file(s) while {fastest.name} likely leaves "
                f"{fastest.incompatible_count} for follow-up."
            )
        coverage = f" Covers {profile.compatible_count} file(s)" if profile.compatible_count else ""
        return (
            "Best default from the current time, size, quality, and compatibility estimates."
            + coverage
            + "."
        )
    if profile.incompatible_count:
        if highly_variable:
            return (
                f"Fastest wall-clock option, but output size is highly variable; "
                f"{profile.incompatible_count} file(s) may still need follow-up."
                + (f" {profile.outlier_hint}" if profile.outlier_hint else "")
            )
        return (
            f"Likely works for {profile.compatible_count} file(s); "
            f"{profile.incompatible_count} may need a safer follow-up profile."
            + (f" {profile.outlier_hint}" if profile.outlier_hint else "")
        )
    if profile.encoder_key in _HW_ENCODERS:
        return "Uses GPU hardware to reduce CPU load and keep the encode on the hardware path."
    if profile.intent_label == "GPU offload":
        return "Uses GPU hardware to reduce CPU load and stay on the hardware path."
    if profile.name == "Balanced":
        return "Higher quality at a moderate speed cost."
    if profile.name in {"Archival", "Archival+"}:
        return "Prioritizes retention over runtime."
    if profile.name in {"Smallest", "Smallest Acceptable"}:
        return "Pushes harder for smaller output sizes."
    return "Alternative trade-off if you prefer this balance."


def build_profiles(
    available_hw: list[str],
    benchmark_speeds: dict[str, float | None],
    total_media_seconds: float,
    total_input_bytes: int,
    *,
    candidate_items: list[AnalysisItem] | None = None,
    ffprobe: Path | None = None,
    policy: str = "fastest-wall-clock",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
    observed_probe_failures: dict[tuple[str, int], dict[Path, str]] | None = None,
    container_incompatibility_cache: dict[Path, str | None] | None = None,
) -> list[EncoderProfile]:
    profiles: list[EncoderProfile] = []
    idx = 1
    active_calibration = (
        load_calibration_store()
        if use_calibration and calibration_store is None
        else calibration_store
    )

    def estimated_output_bytes_for(crf: int, preset: str) -> int:
        if not candidate_items or ffprobe is None:
            return _estimate_output_bytes(total_input_bytes, crf)
        total = 0
        for item in candidate_items:
            estimated = estimate_output_size(
                item.source,
                ffprobe,
                codec=item.codec,
                crf=crf,
                preset=preset,
                use_calibration=use_calibration,
                calibration_store=active_calibration,
            )
            total += estimated if estimated > 0 else item.estimated_output_bytes
        return total

    hardware_profiles: list[tuple[str, float | None]] = [
        (key, benchmark_speeds.get(key)) for key in ("qsv", "nvenc", "amf") if key in available_hw
    ]
    hardware_profiles.sort(key=lambda item: (item[1] or 0.0), reverse=True)

    for key, speed in hardware_profiles:
        profiles.append(
            EncoderProfile(
                index=idx,
                intent_label="GPU offload",
                name="Fastest GPU encode",
                encoder_key=key,
                crf=20,
                sw_preset=None,
                estimated_output_bytes=estimated_output_bytes_for(20, key),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label="Good",
                is_recommended=False,
                why_choose="",
            )
        )
        idx += 1

    fast_speed = benchmark_speeds.get("fast")
    faster_speed = benchmark_speeds.get("faster") or (fast_speed * 1.3 if fast_speed else None)
    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Fast",
            name="Fast",
            encoder_key="faster",
            crf=22,
            sw_preset="faster",
            estimated_output_bytes=estimated_output_bytes_for(22, "faster"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, faster_speed),
            quality_label="Very good",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Balanced",
            name="Balanced",
            encoder_key="fast",
            crf=20,
            sw_preset="fast",
            estimated_output_bytes=estimated_output_bytes_for(20, "fast"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, fast_speed),
            quality_label="Excellent",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    slow_speed = (fast_speed / 4) if fast_speed else None
    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Archival",
            name="Archival",
            encoder_key="slow",
            crf=18,
            sw_preset="slow",
            estimated_output_bytes=estimated_output_bytes_for(18, "slow"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Visually lossless",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Smallest",
            name="Smallest",
            encoder_key="slow",
            crf=28,
            sw_preset="slow",
            estimated_output_bytes=estimated_output_bytes_for(28, "slow"),
            estimated_encode_seconds=_estimate_time(total_media_seconds, slow_speed),
            quality_label="Good",
            is_recommended=False,
            why_choose="",
        )
    )
    idx += 1

    # Built-in intent presets
    _BUILTIN_QUALITY_LABELS = {
        "Fast Batch": "Very good",
        "Archival": "Visually lossless",
        "GPU Offload": "Good",
        "Smallest Acceptable": "Acceptable",
    }
    _BUILTIN_INTENTS = {
        "Fast Batch": "Fast",
        "Archival": "Archival",
        "GPU Offload": "GPU offload",
        "Smallest Acceptable": "Smallest",
    }
    best_hw = hardware_profiles[0][0] if hardware_profiles else None
    best_hw_speed = hardware_profiles[0][1] if hardware_profiles else None

    for bp in get_builtin_profiles():
        # "GPU Offload" uses the best available HW encoder, or falls back to sw
        if bp.name == "GPU Offload":
            encoder_key = best_hw if best_hw else "faster"
            speed = best_hw_speed if best_hw else benchmark_speeds.get("faster")
            sw_preset = None if best_hw else "faster"
        else:
            encoder_key = bp.preset
            speed = benchmark_speeds.get(bp.preset)
            if encoder_key not in _HW_ENCODERS:
                # For slow preset use the slow-speed estimate derived from fast
                if encoder_key == "slow" and benchmark_speeds.get("fast"):
                    speed = (benchmark_speeds["fast"] or 0) / 4
            sw_preset = bp.preset if encoder_key not in _HW_ENCODERS else None

        profiles.append(
            EncoderProfile(
                index=idx,
                intent_label=_BUILTIN_INTENTS.get(bp.name, "Balanced"),
                name=bp.name,
                encoder_key=encoder_key,
                crf=bp.crf,
                sw_preset=sw_preset,
                estimated_output_bytes=estimated_output_bytes_for(bp.crf, encoder_key),
                estimated_encode_seconds=_estimate_time(total_media_seconds, speed),
                quality_label=_BUILTIN_QUALITY_LABELS.get(bp.name, "Good"),
                is_recommended=False,
                why_choose="",
                is_builtin=True,
            )
        )
        idx += 1

    profiles.append(
        EncoderProfile(
            index=idx,
            intent_label="Custom",
            name="Custom",
            encoder_key="",
            crf=20,
            sw_preset=None,
            estimated_output_bytes=0,
            estimated_encode_seconds=0,
            quality_label="",
            is_recommended=False,
            why_choose="",
            is_custom=True,
        )
    )

    failure_rates = {
        profile.encoder_key: estimate_failure_rate(
            active_calibration,
            preset=profile.encoder_key,
            container=(candidate_items[0].source.suffix.lower() if candidate_items else ".mkv"),
        )
        for profile in profiles
        if not profile.is_custom
    }

    if hardware_profiles:
        timed_profiles = [
            profile
            for profile in profiles
            if not profile.is_custom and profile.estimated_encode_seconds > 0
        ]
        fastest_overall = (
            min(timed_profiles, key=lambda profile: profile.estimated_encode_seconds)
            if timed_profiles
            else None
        )
        hw_primary = next(
            (
                profile
                for profile in profiles
                if profile.encoder_key == best_hw and not profile.is_builtin
            ),
            None,
        )
        if hw_primary is not None and hw_primary is fastest_overall:
            hw_primary.name = "Fastest on this device"
        recommended = _select_recommended_profile(
            profiles,
            policy=policy,
            failure_rates=failure_rates,
        )
        if recommended is not None:
            recommended.is_recommended = True
        elif hw_primary is not None:
            hw_primary.is_recommended = True
    else:
        balanced = next((profile for profile in profiles if profile.name == "Balanced"), None)
        if balanced is not None and not _is_profile_dominated(balanced, profiles):
            balanced.is_recommended = True
        else:
            recommended = _select_recommended_profile(
                profiles,
                policy=policy,
                failure_rates=failure_rates,
            )
            if recommended is not None:
                recommended.is_recommended = True

    for profile in profiles:
        if profile.is_custom:
            continue
        grouped: dict[str, int] = {}
        compatible_items = candidate_items or []
        incompatible_items: list[AnalysisItem] = []
        if candidate_items and ffprobe is not None:
            compatible_items = []
            for item in candidate_items:
                observed_reason = (
                    (observed_probe_failures or {})
                    .get((profile.encoder_key, profile.crf), {})
                    .get(item.source)
                )
                if observed_reason:
                    lowered = observed_reason.lower()
                    if "audio codec copy" in lowered:
                        reason = "unsupported copied audio codec"
                    elif "subtitle" in lowered or "mov_text" in lowered:
                        reason = "unsupported copied subtitle codec"
                    elif "attachment" in lowered:
                        reason = "attachment stream incompatibility"
                    elif "data" in lowered or "bin_data" in lowered:
                        reason = "auxiliary data stream incompatibility"
                    elif (
                        "could not open encoder before eof" in lowered or "internal bug" in lowered
                    ):
                        reason = "hardware encoder startup failure"
                    else:
                        reason = "output header failure"
                else:
                    reason = _profile_predicted_incompatibility(
                        item,
                        profile,
                        ffprobe,
                        failure_rates.get(profile.encoder_key, 0.0),
                        container_incompatibility_cache=container_incompatibility_cache,
                    )
                if reason is None:
                    compatible_items.append(item)
                else:
                    incompatible_items.append(item)
                    grouped[reason] = grouped.get(reason, 0) + 1
        profile.compatible_count = len(compatible_items) if candidate_items else 0
        profile.incompatible_count = sum(grouped.values()) if candidate_items else 0
        profile.grouped_incompatibilities = grouped or None
        profile.compatibility_summary = _grouped_incompatibility_summary(grouped)
        if (
            len(incompatible_items) == 1
            and len(candidate_items or []) >= 2
            and incompatible_items[0].source.suffix.lower() in {".mp4", ".m4v"}
            and sum(1 for item in (candidate_items or []) if item.source.suffix.lower() == ".mkv")
            >= len(candidate_items or []) - 1
        ):
            profile.outlier_hint = (
                "All MKV files look compatible; the single MP4 likely needs MKV output."
            )
        else:
            profile.outlier_hint = None
        profile.effective_input_bytes = sum(item.size_bytes for item in compatible_items)
        profile.size_uncertainty = _average_size_error_for_items(
            preset=profile.encoder_key,
            items=compatible_items,
            ffprobe=ffprobe,
            calibration_store=active_calibration,
        )
        if compatible_items:
            speed_key = (
                profile.encoder_key
                if profile.encoder_key in benchmark_speeds
                else (profile.sw_preset or profile.encoder_key)
            )
            profile.estimated_output_bytes = sum(
                estimate_output_size(
                    item.source,
                    ffprobe,
                    codec=item.codec,
                    crf=profile.crf,
                    preset=profile.encoder_key,
                    use_calibration=use_calibration,
                    calibration_store=active_calibration,
                )
                if ffprobe is not None
                else item.estimated_output_bytes
                for item in compatible_items
            )
            calibrated_speed = _adjust_profile_speed_with_calibration(
                base_speed=benchmark_speeds.get(speed_key),
                preset=profile.encoder_key,
                items=compatible_items,
                ffprobe=ffprobe,
                calibration_store=active_calibration,
            )
            profile.estimated_encode_seconds = _estimate_time(
                _sum_item_durations(compatible_items),
                calibrated_speed,
            )
        elif candidate_items:
            profile.estimated_output_bytes = 0
            profile.estimated_encode_seconds = 0.0
        recommended_scope_items = candidate_items or []
        if recommended_scope_items and ffprobe is not None:
            rec_ok, rec_followup = _predict_compatibility_counts_for_items(
                profile,
                recommended_scope_items,
                ffprobe=ffprobe,
                calibration_store=active_calibration,
                observed_probe_failures=observed_probe_failures,
            )
            profile.recommended_compatible_count = rec_ok
            profile.recommended_incompatible_count = rec_followup
    for profile in profiles:
        profile.is_recommended = False
    if candidate_items:
        recommended = _select_recommended_profile(
            profiles,
            policy=policy,
            failure_rates=failure_rates,
        )
        if recommended is not None:
            recommended.is_recommended = True
    elif not hardware_profiles:
        balanced = next((profile for profile in profiles if profile.name == "Balanced"), None)
        if balanced is not None and not _is_profile_dominated(balanced, profiles):
            balanced.is_recommended = True
        recommended = next((profile for profile in profiles if profile.is_recommended), None)
    else:
        recommended = _select_recommended_profile(
            profiles,
            policy=policy,
            failure_rates=failure_rates,
        )
        if recommended is not None:
            recommended.is_recommended = True
    fastest = min(
        (
            profile
            for profile in profiles
            if not profile.is_custom and profile.estimated_encode_seconds > 0
        ),
        key=lambda profile: profile.estimated_encode_seconds,
        default=None,
    )
    for profile in profiles:
        profile.why_choose = _profile_why_choose(profile, recommended, fastest=fastest)

    return profiles


def display_profiles_table(
    profiles: list[EncoderProfile],
    total_input_bytes: int,
    candidate_count: int,
    recommended_count: int,
    device_labels: dict[str, str],
    console: Console,
    time_confidence: str | None = None,
    time_confidence_detail: str | None = None,
    size_confidence: str | None = None,
    size_confidence_detail: str | None = None,
    size_error_by_preset: dict[str, float | None] | None = None,
    bias_note: str | None = None,
    show_all_profiles: bool = False,
    plain_output: bool = False,
) -> tuple[list[EncoderProfile], dict[int, EncoderProfile]]:
    def dedupe_key(profile: EncoderProfile) -> tuple[object, ...]:
        encoder_family = (
            "hardware"
            if profile.encoder_key in _HW_ENCODERS
            else profile.sw_preset or profile.encoder_key
        )
        time_bucket = (
            round(profile.estimated_encode_seconds / 1800)
            if profile.estimated_encode_seconds > 0
            else -1
        )
        output_bucket = (
            round(profile.estimated_output_bytes / (5 * _GB))
            if profile.estimated_output_bytes > 0
            else -1
        )
        return (
            encoder_family,
            profile.compatible_count,
            time_bucket,
            output_bucket,
            profile.intent_label
            if profile.intent_label in {"Fast", "Balanced", "GPU offload"}
            else "",
        )

    render_mode = _render_mode(console, plain_output=plain_output)
    visible_profiles = [
        profile
        for profile in profiles
        if not (
            profile.name == "GPU Offload"
            and profile.encoder_key not in _HW_ENCODERS
            and profile.sw_preset is not None
        )
    ]
    if not show_all_profiles:
        seen: set[tuple[object, ...]] = set()
        filtered: list[EncoderProfile] = []
        dedupe_threshold = 1 if len(visible_profiles) <= 7 else 0
        for profile in visible_profiles:
            if profile.is_custom:
                filtered.append(profile)
                continue
            key = dedupe_key(profile)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(profile)
        if len(visible_profiles) - len(filtered) > dedupe_threshold:
            visible_profiles = filtered

    # Build sequential display indices 1..N so the table never has gaps.
    # display_index_map[display_idx] → profile (for prompt_profile_selection)
    display_index_map: dict[int, EncoderProfile] = {
        display_idx: profile for display_idx, profile in enumerate(visible_profiles, start=1)
    }
    # Reverse map: profile.index → display_idx (for the default/recommended prompt value)
    _profile_to_display: dict[int, int] = {
        profile.index: display_idx for display_idx, profile in display_index_map.items()
    }

    def _est_saving_text(profile: EncoderProfile) -> str:
        saved = total_input_bytes - profile.estimated_output_bytes
        size_error = (size_error_by_preset or {}).get(profile.encoder_key)
        uncertainty = estimate_display_uncertainty(
            size_confidence,
            average_error=size_error,
            widen_by=0.04 if profile.incompatible_count else 0.0,
        )
        offset = uncertainty * total_input_bytes
        saving_low = max(0, int(saved - offset))
        saving_high = min(total_input_bytes, int(saved + offset))
        pct_low = saving_low / total_input_bytes * 100 if total_input_bytes else 0.0
        pct_high = saving_high / total_input_bytes * 100 if total_input_bytes else 0.0
        if (pct_high - pct_low) >= 40:
            return "~highly variable"
        return f"~{_fmt_size(saving_low)}-{_fmt_size(saving_high)} ({pct_low:.0f}-{pct_high:.0f}%)"

    def _est_time_text(profile: EncoderProfile) -> str:
        if profile.estimated_encode_seconds > 0:
            widen = 0.0
            if candidate_count >= _LARGE_BATCH_FILE_THRESHOLD:
                widen += 0.08
            if candidate_count >= 48:
                widen += 0.06
            if profile.encoder_key not in _HW_ENCODERS and candidate_count >= 18:
                widen += 0.05
            low_seconds, high_seconds = estimate_value_range(
                profile.estimated_encode_seconds,
                confidence=time_confidence,
                widen_by=(0.05 if profile.incompatible_count else 0.0) + widen,
            )
            return _format_duration_range(low_seconds, high_seconds)
        if profile.sw_preset in {"slow", "slower", "veryslow"}:
            return "~slower than Balanced"
        return "~unknown"

    if render_mode == "plain":
        console.print()
        console.print("[bold cyan]Available encoding profiles[/bold cyan]")
        for display_idx, profile in display_index_map.items():
            if profile.is_custom:
                console.print(f"{display_idx}. Custom | manual override", highlight=False)
                continue
            recommended_tag = " [recommended]" if profile.is_recommended else ""
            encoder_display = _encoder_display_name(
                profile.encoder_key, device_labels, truncate=True
            )
            fit = (
                f"{profile.compatible_count} likely ok / {profile.incompatible_count} follow-up"
                if candidate_count
                else "-"
            )
            console.print(
                f"{display_idx}. {profile.name}{recommended_tag} | {encoder_display} | {_est_saving_text(profile)} | {_est_time_text(profile)} | {profile.quality_label} | {fit}",
                highlight=False,
            )
            console.print(f"   {profile.why_choose}", highlight=False)
            if recommended_count:
                console.print(
                    f"   Recommended-only: {profile.recommended_compatible_count} now / {profile.recommended_incompatible_count} follow-up",
                    highlight=False,
                )
            if profile.grouped_incompatibilities:
                console.print(
                    f"   Follow-up risk: {_format_grouped_reason_list(profile.grouped_incompatibilities)}",
                    highlight=False,
                )
    else:
        compact = render_mode in {"compact", "narrow"}
        narrow = render_mode == "narrow"
        if narrow:
            console.print()
            console.print("[bold cyan]Available encoding profiles[/bold cyan]")
            for display_idx, profile in display_index_map.items():
                if profile.is_custom:
                    console.print(f"{display_idx}. Custom")
                    console.print("   Manual override.")
                    continue
                recommended_tag = " [recommended]" if profile.is_recommended else ""
                encoder_display = _encoder_display_name(
                    profile.encoder_key, device_labels, truncate=True
                )
                fit = (
                    f"{profile.compatible_count} likely ok / {profile.incompatible_count} follow-up"
                    if candidate_count
                    else "-"
                )
                console.print(f"{display_idx}. {profile.name}{recommended_tag}")
                console.print(
                    f"   {encoder_display} | {_est_saving_text(profile)} | {_est_time_text(profile)} | {profile.quality_label}"
                )
                console.print(f"   Fit: {fit}")
                if recommended_count:
                    console.print(
                        f"   Recommended-only: {profile.recommended_compatible_count} now / {profile.recommended_incompatible_count} follow-up"
                    )
                console.print(f"   Why: {profile.why_choose}")
                if profile.grouped_incompatibilities:
                    console.print(
                        f"   Follow-up risk: {_format_grouped_reason_list(profile.grouped_incompatibilities)}"
                    )
        else:
            table = Table(
                title="Available encoding profiles",
                header_style="bold cyan",
                expand=True,
                show_lines=False,
            )
            table.add_column("#", justify="right", style="bold", no_wrap=True)
            table.add_column("Intent", no_wrap=True)
            table.add_column("Profile", no_wrap=True)
            table.add_column("Encoder", style="dim cyan")
            table.add_column("Est. Saving", justify="right", style="bold green", no_wrap=True)
            table.add_column("Est. Time", justify="right", no_wrap=True)
            table.add_column("Quality", no_wrap=True)
            if not compact:
                table.add_column("Works for", justify="center", no_wrap=True)
                table.add_column("Likely incompatible", justify="center", no_wrap=True)
            table.add_column("Why choose this", no_wrap=False)
            if not compact:
                table.add_column("CRF", justify="center", no_wrap=True)

            for display_idx, profile in display_index_map.items():
                if profile.is_custom:
                    row = [str(display_idx), profile.intent_label, "Custom", "-", "-", "-"]
                    row.extend(
                        ["-", "-", "Manual override."] if not compact else ["-", "Manual override."]
                    )
                    if not compact:
                        row.append("-")
                    table.add_row(*row)
                    continue

                encoder_display = _encoder_display_name(
                    profile.encoder_key, device_labels, truncate=True
                )
                profile_name: str | Text = (
                    Text.assemble(profile.name, " ", ("[recommended]", "bold cyan"))
                    if profile.is_recommended
                    else profile.name
                )
                row = [
                    str(display_idx),
                    profile.intent_label,
                    profile_name,
                    encoder_display,
                    _est_saving_text(profile),
                    _est_time_text(profile),
                ]
                quality_style = {
                    "Visually lossless": "green bold",
                    "Excellent": "green",
                    "Very good": "green",
                    "Good": "yellow",
                }.get(profile.quality_label, "white")
                row.append(Text(profile.quality_label, style=quality_style))
                if not compact:
                    row.extend(
                        [
                            str(profile.compatible_count) if candidate_count else "-",
                            str(profile.incompatible_count) if candidate_count else "-",
                        ]
                    )
                row.append(profile.why_choose)
                if not compact:
                    row.append(str(profile.crf))
                table.add_row(*row)

            console.print()
            console.print(table)

    recommended = next((profile for profile in profiles if profile.is_recommended), None)
    fastest = min(
        (
            profile
            for profile in profiles
            if not profile.is_custom and profile.estimated_encode_seconds > 0
        ),
        key=lambda profile: profile.estimated_encode_seconds,
        default=None,
    )

    console.print(
        f"  [dim]Likely encode candidates: {candidate_count} file(s) / {_fmt_size(total_input_bytes)}[/dim]"
    )
    if recommended_count:
        scope_tone = "high-value batch"
        if recommended_count < max(3, candidate_count // 6):
            scope_tone = "selective starter batch"
        elif candidate_count >= _LARGE_BATCH_FILE_THRESHOLD and recommended_count < candidate_count:
            scope_tone = "cleanup-style batch"
        console.print(
            f"  [dim]Recommended-only default scope: {recommended_count} file(s) ({scope_tone}). Each profile also shows a recommended-only compatibility line.[/dim]"
        )
        if recommended is not None and recommended.compatible_count * 2 <= max(
            recommended_count, 1
        ):
            console.print(
                "  [dim yellow]This default only covers half the recommended set or less, so treat it as a partial-batch starting point rather than a fully compatible batch plan.[/dim yellow]"
            )
    console.print(
        "  [dim]Time and size numbers are approximate estimates for likely encode candidates, not already-skipped files.[/dim]"
    )
    console.print(
        "  [dim]Hardware presets are still full re-encodes; source bitrate, resolution, and runtime dominate total time.[/dim]"
    )
    guidance = _large_batch_guidance(
        selected_count=recommended_count or candidate_count,
        total_candidates=candidate_count,
        estimated_seconds=(
            recommended.estimated_encode_seconds
            if recommended is not None and recommended.estimated_encode_seconds > 0
            else None
        ),
    )
    if guidance:
        console.print(f"  [dim]{guidance}[/dim]")
    if time_confidence is not None:
        console.print(
            f"  [dim]Time confidence: {time_confidence}"
            + (f" ({time_confidence_detail})" if time_confidence_detail else "")
            + ".[/dim]"
        )
    if size_confidence is not None:
        console.print(
            f"  [dim]Size confidence: {size_confidence}"
            + (f" ({size_confidence_detail})" if size_confidence_detail else "")
            + ".[/dim]"
        )
    if fastest is not None:
        console.print(
            f"  [dim]Lowest estimated wait: {fastest.name} (~{_fmt_duration(fastest.estimated_encode_seconds)}), working for {fastest.compatible_count} file(s).[/dim]"
        )
    if recommended is not None:
        mkv_first = _mkv_first_guidance(recommended, recommended_count=recommended_count)
        if (
            recommended.why_choose.startswith("Fastest wait:")
            or "highly variable" in recommended.why_choose
            or "partial-batch" in recommended.why_choose
        ):
            console.print(f"  [dim]Default pick: {recommended.why_choose}[/dim]")
            if mkv_first:
                console.print(f"  [dim]Default workflow hint: {mkv_first}[/dim]")
        else:
            console.print(
                f"  [dim]Default pick: {recommended.name} because it is estimated to work for {recommended.compatible_count} file(s) with {recommended.incompatible_count} likely left for follow-up.[/dim]"
            )
            if recommended.outlier_hint:
                console.print(f"  [dim]{recommended.outlier_hint}[/dim]")
            if fastest is not None and fastest is not recommended:
                console.print(
                    f"  [dim]Fastest wait is still {fastest.name}, but {recommended.name} is the steadier default for size/compatibility on similar files.[/dim]"
                )
    if render_mode in {"compact", "narrow"}:
        console.print(
            "  [dim]Compact view switches to denser profile summaries on smaller terminals.[/dim]"
        )
    if bias_note:
        console.print(f"  [dim]Estimate bias: {bias_note}.[/dim]")
    hidden_rows = len(profiles) - len(visible_profiles)
    if not show_all_profiles and hidden_rows >= 2:
        console.print(
            f"  [dim]Hidden {hidden_rows} near-duplicate profile row(s). Use --show-all-profiles to inspect every profile.[/dim]"
        )
    console.print()
    return visible_profiles, display_index_map


def prompt_profile_selection(
    profiles: list[EncoderProfile],
    display_index_map: dict[int, EncoderProfile],
    console: Console,
) -> EncoderProfile:
    recommended = next((profile for profile in profiles if profile.is_recommended), profiles[0])
    # Find the sequential display index for the recommended profile
    recommended_display_idx = next(
        (didx for didx, p in display_index_map.items() if p.index == recommended.index),
        1,
    )
    display_max = len(display_index_map)

    while True:
        choice = _wizard_prompt(
            f"Select a profile [1-{display_max}, Enter for {recommended_display_idx} ({recommended.name})]",
            default=str(recommended_display_idx),
            show_default=False,
            prompt_id="profile-selection",
            acceptance_label="Profile selection received",
        ).strip()

        try:
            selected_display_idx = int(choice)
        except ValueError:
            selected_display_idx = -1

        profile = display_index_map.get(selected_display_idx)
        if profile is not None:
            _get_wizard_session() and _get_wizard_session().add_event(
                f"Selected profile row {selected_display_idx}: {profile.name}"
            )
            return profile

        console.print(f"[yellow]Please enter a number between 1 and {display_max}.[/yellow]")
        _track_prompt_anomaly("profile-selection: out-of-range selection")


def run_custom_wizard(available_hw: list[str], console: Console) -> tuple[str, int, str | None]:
    """Walk through manual encoder/CRF selection."""
    encoder_choices: list[tuple[str, str]] = []
    for key in ("qsv", "nvenc", "amf"):
        if key in available_hw:
            encoder_choices.append((key, f"{_HARDWARE_DISPLAY_NAMES[key]} (hardware)"))
    encoder_choices.append(("libx265", "libx265 (software)"))

    console.print("\n[bold]Custom encoder:[/bold]")
    for idx, (_, label) in enumerate(encoder_choices, start=1):
        console.print(f"  {idx}. {label}")

    while True:
        try:
            enc_idx = (
                int(
                    _wizard_prompt(
                        f"Choose encoder [1-{len(encoder_choices)}]",
                        default="1",
                        prompt_id="custom-encoder",
                        acceptance_label="Custom encoder selection",
                    )
                )
                - 1
            )
        except ValueError:
            enc_idx = -1
        if 0 <= enc_idx < len(encoder_choices):
            break
        console.print("[yellow]Invalid choice.[/yellow]")

    chosen_key, _ = encoder_choices[enc_idx]

    while True:
        try:
            crf = int(
                _wizard_prompt(
                    "CRF quality value [0-51, lower = better quality]",
                    default="20",
                    prompt_id="custom-crf",
                    acceptance_label="CRF received",
                )
            )
        except ValueError:
            crf = -1
        if 0 <= crf <= 51:
            break
        console.print("[yellow]CRF must be between 0 and 51.[/yellow]")

    if chosen_key != "libx265":
        return chosen_key, crf, None

    sw_presets = ["ultrafast", "faster", "fast", "medium", "slow"]
    console.print("\n[bold]Software preset:[/bold] (slower = better compression)")
    for idx, sw_preset in enumerate(sw_presets, start=1):
        console.print(f"  {idx}. {sw_preset}")

    while True:
        try:
            preset_idx = (
                int(
                    _wizard_prompt(
                        "Choose preset [1-5]",
                        default="3",
                        prompt_id="custom-preset",
                        acceptance_label="Preset selection",
                    )
                )
                - 1
            )
        except ValueError:
            preset_idx = -1
        if 0 <= preset_idx < len(sw_presets):
            selected_preset = sw_presets[preset_idx]
            return selected_preset, crf, selected_preset
        console.print("[yellow]Invalid choice.[/yellow]")


def maybe_save_profile(
    preset: str,
    crf: int,
    display_label: str,
    console: Console,
) -> None:
    if not _wizard_confirm(
        "Save these settings as a named profile?",
        default=False,
        prompt_id="save-profile",
        acceptance_label="Save profile",
    ):
        _get_wizard_session() and _get_wizard_session().add_event("Skipped saving profile")
        return

    while True:
        name = _wizard_prompt(
            "Profile name",
            show_default=False,
            prompt_id="profile-name",
            acceptance_label="Profile name received",
        ).strip()
        if name:
            break
        console.print("[yellow]Profile name cannot be empty.[/yellow]")

    upsert_profile(
        SavedProfile(
            name=name,
            preset=preset,
            crf=crf,
            label=display_label,
            created_from_wizard=True,
        )
    )
    console.print(f"[green]Saved profile[/green] {name}")


def display_candidate_table(title: str, items: list[AnalysisItem], console: Console) -> None:
    table = Table(title=title, header_style="bold cyan", expand=True)
    table.add_column("File")
    table.add_column("Codec", justify="center", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Est. Saving", justify="right", no_wrap=True)
    table.add_column("Reason")

    for item in sorted(
        items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True
    )[:12]:
        saving = (
            "-"
            if item.estimated_savings_bytes <= 0
            else f"~{_fmt_size(item.estimated_savings_bytes)}"
        )
        table.add_row(
            item.source.name,
            item.codec or "?",
            _fmt_size(item.size_bytes),
            saving,
            item.reason_text,
        )

    console.print()
    console.print(table)
    console.print()


def prompt_analysis_action(
    recommended_count: int,
    maybe_count: int,
    console: Console,
    *,
    recommended_label: str = "recommended",
    maybe_label: str = "maybe",
) -> str:
    console.print("[bold]Next step:[/bold]")
    if maybe_count:
        console.print(f"  1. Compress {recommended_label} only ({recommended_count} file(s))")
    else:
        console.print(f"  1. Compress {recommended_label} files ({recommended_count} file(s))")
    if maybe_count:
        console.print(f"  2. Review {maybe_label} files ({maybe_count} file(s))")
        console.print("  3. Export manifest")
        console.print("  4. Cancel")
        max_choice = 4
        export_choice = 3
        cancel_choice = 4
    else:
        console.print("  2. Export manifest")
        console.print("  3. Cancel")
        max_choice = 3
        export_choice = 2
        cancel_choice = 3

    while True:
        try:
            choice = int(
                _wizard_prompt(
                    f"Choose action [1-{max_choice}]",
                    default="1",
                    prompt_id="next-step",
                    acceptance_label="Next-step selection",
                )
            )
        except ValueError:
            choice = -1

        if choice == 1:
            console.print(f"[dim]Next step selected:[/dim] Compress {recommended_label} files")
            return "compress_recommended"
        if maybe_count and choice == 2:
            console.print(f"[dim]Next step selected:[/dim] Review {maybe_label} files")
            return "review_maybe"
        if choice == export_choice:
            console.print("[dim]Next step selected:[/dim] Export manifest")
            return "export"
        if choice == cancel_choice:
            console.print("[dim]Next step selected:[/dim] Cancel")
            return "cancel"

        console.print("[yellow]Invalid choice.[/yellow]")


def review_maybe_items(
    maybe_items: list[AnalysisItem],
    console: Console,
    *,
    title: str = "Maybe files",
    prompt_text: str = "Include maybe files in this run?",
    decision_label: str = "Maybe files decision",
) -> bool:
    display_candidate_table(title, maybe_items, console)
    include = _wizard_confirm(
        prompt_text,
        default=False,
        prompt_id="include-maybe",
        acceptance_label="Include maybe files",
    )
    console.print(
        f"[dim]{decision_label}:[/dim] {'Included in this run' if include else 'Left out for now'}"
    )
    return include


def prompt_manifest_split_mode(console: Console) -> str:
    console.print("[bold]Manifest export:[/bold]")
    console.print("  1. Combined manifest")
    console.print("  2. Split manifests by show")
    console.print("  3. Split manifests by season")
    while True:
        choice = _wizard_prompt(
            "Choose export mode [1-3]",
            default="1",
            prompt_id="manifest-split-mode",
            acceptance_label="Manifest export mode",
        ).strip()
        if choice == "1":
            return "combined"
        if choice == "2":
            return "show"
        if choice == "3":
            return "season"
        console.print("[yellow]Invalid choice.[/yellow]")


def _subtitle_drop_warning(jobs: list[EncodeJob], ffprobe: Path) -> str | None:
    affected: list[str] = []
    for job in jobs:
        if job.skip or not output_drops_subtitles(job.output):
            continue
        if source_has_subtitle_streams(job.source, ffprobe):
            affected.append(job.source.name)
    if not affected:
        return None
    examples = ", ".join(affected[:3])
    return (
        f"{len(affected)} file(s) with subtitle streams will lose subtitles because "
        f"{jobs[0].output.suffix.lower()} outputs use '-sn' for compatibility."
        + (f" Examples: {examples}." if examples else "")
    )


def _preflight_candidates(jobs: list[EncodeJob]) -> list[EncodeJob]:
    non_mkv = [job for job in jobs if job.output.suffix.lower() != ".mkv"]
    if non_mkv:
        return non_mkv
    if not jobs:
        return []
    return [max(jobs, key=lambda job: job.source.stat().st_size)]


def _run_preflight_checks(
    jobs: list[EncodeJob],
    ffmpeg: Path,
    ffprobe: Path,
    *,
    crf: int,
    preset: str,
    console: Console,
) -> tuple[list[EncodeJob], list[tuple[EncodeJob, EncodeResult]]]:
    candidates = _preflight_candidates(jobs)
    compatible: list[EncodeJob] = []
    failed: list[tuple[EncodeJob, EncodeResult]] = []
    with console.status("[dim]Running final compatibility check...[/dim]", spinner="dots"):
        for job in candidates:
            result = preflight_encode_job(
                job.source,
                ffmpeg,
                ffprobe,
                crf=crf,
                preset=preset,
            )
            if result.success:
                compatible.append(job)
            else:
                failed.append((job, result))
    if len(candidates) == 1 and compatible:
        return jobs, failed
    compatible_sources = {job.source for job in compatible}
    passthrough = [job for job in jobs if job not in candidates]
    return passthrough + compatible, failed


def _group_preflight_failures(
    failures: list[tuple[EncodeJob, EncodeResult]],
    ffprobe: Path,
) -> dict[str, list[EncodeJob]]:
    grouped: dict[str, list[EncodeJob]] = {}
    for job, result in failures:
        reason = result.error_message or ""
        lowered = reason.lower()
        label = "unknown compatibility failure"
        if "could not open encoder before eof" in lowered or "internal bug" in lowered:
            label = "hardware encoder startup failure"
        elif "audio codec copy is not supported" in lowered:
            label = "unsupported copied audio codec"
        elif (
            "mov_text" in lowered
            or "subtitle" in lowered
            or "subrip" in lowered
            or "ass" in lowered
        ):
            label = "unsupported copied subtitle codec"
        elif "attachment" in lowered:
            label = "attachment stream incompatibility"
        elif "data" in lowered or "bin_data" in lowered:
            label = "auxiliary data stream incompatibility"
        elif "could not write header" in lowered:
            container_reason = describe_container_incompatibility(job.source, job.output, ffprobe)
            if container_reason and "audio codec copy" in container_reason:
                label = "unsupported copied audio codec"
            elif container_reason and "subtitle" in container_reason:
                label = "unsupported copied subtitle codec"
            elif container_reason and "attachment" in container_reason:
                label = "attachment stream incompatibility"
            elif container_reason and "auxiliary data" in container_reason:
                label = "auxiliary data stream incompatibility"
            elif container_reason:
                label = "output header failure"
            else:
                label = "output header failure"
        elif job.output.suffix.lower() in {".mp4", ".m4v"}:
            container_reason = describe_container_incompatibility(job.source, job.output, ffprobe)
            if container_reason and "audio codec copy" in container_reason:
                label = "unsupported copied audio codec"
            elif container_reason and "subtitle" in container_reason:
                label = "unsupported copied subtitle codec"
            elif container_reason and "attachment" in container_reason:
                label = "attachment stream incompatibility"
            elif container_reason and "auxiliary data" in container_reason:
                label = "auxiliary data stream incompatibility"
            else:
                label = "output header failure"
        grouped.setdefault(label, []).append(job)
    return grouped


def _write_followup_manifest(
    directory: Path,
    recursive: bool,
    preset: str,
    crf: int,
    incompatible_items: list[AnalysisItem],
    *,
    notes: list[str] | None = None,
) -> Path | None:
    if not incompatible_items:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    manifest_path = directory / f"mediashrink_followup_{timestamp}.json"
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
        duplicate_policy=None,
        recommended_only=False,
        notes=notes,
        items=incompatible_items,
    )
    save_manifest(manifest, manifest_path)
    return manifest_path


def _followup_manifest_notes(
    grouped_failures: dict[str, list[EncodeJob]],
    incompatible_items: list[AnalysisItem],
    *,
    ffprobe: Path,
) -> list[str]:
    if not grouped_failures or not incompatible_items:
        return []
    remaining_sources = {item.source for item in incompatible_items}
    notes = ["Automatically generated from files left out by preflight compatibility checks."]
    ranked = sorted(
        grouped_failures.items(),
        key=lambda item: (
            -len([job for job in item[1] if job.source in remaining_sources]),
            item[0],
        ),
    )
    for reason, jobs in ranked:
        matching_jobs = [job for job in jobs if job.source in remaining_sources]
        if not matching_jobs:
            continue
        examples = ", ".join(job.source.name for job in matching_jobs[:3])
        note = f"{len(matching_jobs)} file(s): {reason}"
        if examples:
            note += f" ({examples})"
        notes.append(note)
    for detail in _per_file_followup_details(
        grouped_failures,
        incompatible_items,
        ffprobe=ffprobe,
    ):
        notes.append(detail)
    notes.append("Suggested retry: prefer MKV output first for MP4/container-copy issues.")
    return notes


def _followup_remediation(reason: str) -> str:
    mapping = {
        "unsupported copied audio codec": "Use MKV output or re-encode the copied audio stream.",
        "unsupported copied subtitle codec": "Use MKV output if subtitle preservation matters.",
        "attachment stream incompatibility": "Use MKV output to preserve attachment streams.",
        "auxiliary data stream incompatibility": "Use MKV output or drop auxiliary data streams.",
        "output header failure": "Use MKV output and re-check copied stream compatibility.",
        "hardware encoder startup failure": "Retry with a software profile such as Fast or Balanced.",
    }
    return mapping.get(reason, "Review the file manually before retrying.")


def _per_file_followup_details(
    grouped_failures: dict[str, list[EncodeJob]],
    incompatible_items: list[AnalysisItem],
    *,
    ffprobe: Path,
) -> list[str]:
    if not grouped_failures or not incompatible_items:
        return []
    reason_by_source: dict[Path, str] = {}
    for reason, jobs in grouped_failures.items():
        for job in jobs:
            reason_by_source[job.source] = reason
    details: list[str] = []
    for item in incompatible_items:
        explicit = describe_container_incompatibilities(item.source, item.source, ffprobe)
        if explicit:
            details.append(
                f"{item.source.name}: {', '.join(explicit)} -> Use MKV output or re-check copied stream compatibility."
            )
            continue
        notes = describe_output_container_constraints(item.source, item.source, ffprobe)
        if notes:
            details.append(
                f"{item.source.name}: {', '.join(notes)} -> Use MKV output or review copied streams manually."
            )
            continue
        reason = reason_by_source.get(item.source, "unknown compatibility failure")
        details.append(f"{item.source.name}: {reason} -> {_followup_remediation(reason)}")
    return details


def _followup_next_step_hint(
    *,
    preset: str,
    grouped_failures: dict[str, list[EncodeJob]],
) -> str:
    reasons = set(grouped_failures)
    container_blocking = {
        "unsupported copied audio codec",
        "unsupported copied subtitle codec",
        "attachment stream incompatibility",
        "auxiliary data stream incompatibility",
        "output header failure",
    }
    if reasons and reasons <= container_blocking:
        return "Or re-run the wizard on the same folder with MKV output first, or review copied streams manually."
    if preset in _HW_ENCODERS:
        return "Or re-run the wizard on the same folder with a software profile."
    return (
        "Or re-run the wizard on the same folder with MKV output or review copied streams manually."
    )


def _select_profile_interactively(
    profiles: list[EncoderProfile],
    available_hw: list[str],
    device_labels: dict[str, str],
    auto: bool,
    console: Console,
    display_index_map: dict[int, EncoderProfile] | None = None,
) -> tuple[str, int, str | None, str]:
    if auto:
        selected = next((p for p in profiles if p.is_recommended), profiles[0])
        console.print(f"[dim]Non-interactive mode:[/dim] selected profile {selected.name}")
    else:
        effective_map = display_index_map or {i + 1: p for i, p in enumerate(profiles)}
        selected = prompt_profile_selection(profiles, effective_map, console)

    if selected.is_custom:
        preset, crf, sw_preset = run_custom_wizard(available_hw, console)
        display_label = f"Custom ({preset}, CRF {crf})"
    else:
        preset = selected.encoder_key
        crf = selected.crf
        sw_preset = selected.sw_preset
        display_label = selected.name

    console.print(
        f"[dim]Selected profile:[/dim] {display_label} "
        f"([dim]encoder:[/dim] {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}, "
        f"[dim]CRF:[/dim] {crf})"
    )
    if not selected.is_custom and selected.why_choose:
        console.print(
            f"  [dim]Why choose this for likely encode candidates:[/dim] {selected.why_choose}"
        )
    return preset, crf, sw_preset, display_label


def _predict_compatibility_counts_for_items(
    profile: EncoderProfile,
    items: list[AnalysisItem],
    *,
    ffprobe: Path,
    calibration_store: dict[str, object] | None,
    observed_probe_failures: dict[tuple[str, int], dict[Path, str]] | None = None,
) -> tuple[int, int]:
    if not items:
        return 0, 0
    compatible, incompatible, _ = _predict_profile_compatibility(
        profile=profile,
        items=items,
        ffprobe=ffprobe,
        failure_rate=estimate_failure_rate(
            calibration_store,
            preset=profile.encoder_key,
            container=items[0].source.suffix.lower() if items else ".mkv",
        ),
        observed_probe_failures=observed_probe_failures,
    )
    return compatible, incompatible


def _run_analysis_with_progress(
    files: list[Path],
    ffprobe: Path,
    console: Console,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
) -> list[AnalysisItem]:
    if not files:
        return []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console,
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


def _maybe_run_preview(
    preview_items: list[AnalysisItem],
    ffmpeg: Path,
    ffprobe: Path,
    preset: str,
    crf: int,
    auto: bool,
    console: Console,
    plain_output: bool = False,
) -> bool:
    if auto:
        console.print("[dim]Preview decision:[/dim] Skipped preview in non-interactive mode")
        return True
    if not _wizard_confirm(
        "Test a 2-minute preview clip before the full batch?",
        default=False,
        prompt_id="preview",
        acceptance_label="Preview decision",
    ):
        console.print("[dim]Preview decision:[/dim] Skipped preview")
        return True

    if not preview_items:
        console.print("[dim]Preview decision:[/dim] No preview candidates available")
        return True

    preview_results: list[EncodeResult] = []
    console.print(f"[dim]Preview encoding {len(preview_items)} representative clip(s)...[/dim]")
    for item in preview_items:
        console.print(f"  [dim]Preview:[/dim] {item.source.name}")
        preview_results.append(
            encode_preview(
                source=item.source,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                duration_minutes=2.0,
                crf=crf,
                preset=preset,
            )
        )
    from mediashrink.progress import EncodingDisplay

    EncodingDisplay(
        console,
        render_mode="plain" if plain_output else "auto",
    ).show_summary(preview_results)
    successful_previews = [
        result for result in preview_results if result.success and result.job.output.exists()
    ]
    if successful_previews and all(result.success for result in preview_results):
        for result in successful_previews:
            console.print(f"  [dim]Preview saved:[/dim] {result.job.output}")
        console.print(
            "  [dim]Inspect video quality and verify audio/subtitle playback before continuing.[/dim]"
        )
        console.print(
            "  [dim]Use the preview clip as a quality check, not as a file-size estimate.[/dim]"
        )
        console.print("[dim]Preview decision:[/dim] Preview completed successfully")
        return True

    failed_preview = next((result for result in preview_results if not result.success), None)
    if failed_preview and failed_preview.error_message:
        console.print(f"[red]Preview encode failed:[/red] {failed_preview.error_message}")
    else:
        console.print("[red]Preview encode failed.[/red]")
    console.print(
        "[dim]This is likely an encoder configuration issue, not a problem with your files. "
        "If it persists, try a software profile (e.g. 'Fast').[/dim]"
    )
    continue_anyway = _wizard_confirm(
        "Continue to full batch anyway?",
        default=False,
        prompt_id="preview-continue",
        acceptance_label="Preview failure decision",
    )
    console.print(
        f"[dim]Preview failure handling:[/dim] {'Continue to batch' if continue_anyway else 'Stop wizard'}"
    )
    return continue_anyway


def run_wizard(
    directory: Path,
    ffmpeg: Path,
    ffprobe: Path,
    recursive: bool,
    output_dir: Path | None,
    overwrite: bool,
    no_skip: bool,
    console: Console,
    auto: bool = False,
    policy: str = "fastest-wall-clock",
    on_file_failure: str = "retry",
    use_calibration: bool = True,
    duplicate_policy: str = "prefer-mkv",
    show_all_profiles: bool = False,
    plain_output: bool = False,
    non_interactive_wizard: bool = False,
    debug_session_log: bool = False,
) -> tuple[list[EncodeJob], str, bool, Path | None]:
    """Run the interactive wizard and return (jobs, action, cleanup_after, followup_manifest_path)."""
    global _ACTIVE_WIZARD_SESSION, _LAST_WIZARD_REPORT_CONTEXT
    _LAST_WIZARD_REPORT_CONTEXT = None
    requested_non_interactive = auto or non_interactive_wizard
    session = WizardSessionState(
        console=console,
        directory=directory,
        output_dir=output_dir,
        debug_session_log=debug_session_log,
        input_mode="non-interactive" if requested_non_interactive else "interactive",
        plain_prompts=plain_output or requested_non_interactive,
    )
    prior_session = _ACTIVE_WIZARD_SESSION
    _ACTIVE_WIZARD_SESSION = session
    try:
        console.print("\n[bold cyan]mediashrink wizard[/bold cyan]")
        console.print("[dim]Discovering supported files...[/dim]\n")

        files = scan_directory(directory, recursive=recursive)
        if not files:
            console.print(
                f"[yellow]No supported video files ({supported_formats_label()}) found in[/yellow] {directory}"
            )
            return [], "cancel", False, None

        total_input_bytes = sum(path.stat().st_size for path in files)
        console.print(
            f"Found [bold]{len(files)}[/bold] file(s) "
            f"([yellow]{_fmt_size(total_input_bytes)}[/yellow] total)\n"
        )

        analysis_items = _run_analysis_with_progress(
            files,
            ffprobe,
            console,
            use_calibration=use_calibration,
        )
        filename_hygiene_actions = _maybe_fix_filename_hygiene(
            analysis_items,
            console=console,
            active_auto=requested_non_interactive,
        )
        analysis_items, duplicate_notes = apply_duplicate_policy_to_items(
            analysis_items,
            policy=duplicate_policy,
        )
        early_notes = list(duplicate_notes)
        if filename_hygiene_actions:
            renamed_count = sum(
                1 for action in filename_hygiene_actions if action.get("renamed") == "yes"
            )
            early_notes.append(
                f"Filename hygiene: {len(filename_hygiene_actions)} suspicious name(s) detected"
                + (
                    f", {renamed_count} renamed before encode."
                    if renamed_count
                    else "; no automatic rename was applied."
                )
            )
        compatibility_signals = collect_container_risk_signals(analysis_items, ffprobe)
        container_risk_count, container_risk_reasons, container_risk_examples = (
            summarize_container_risks(
                analysis_items,
                ffprobe,
            )
        )
        if container_risk_count:
            early_notes.append(
                "Early compatibility warning: "
                f"{container_risk_count} likely encode candidate(s) may need MKV output because of copied-stream or container constraints"
                + (f" ({', '.join(container_risk_examples)})" if container_risk_examples else "")
            )
        initial_confidence = estimate_analysis_confidence(analysis_items)
        display_analysis_summary(
            analysis_items,
            None,
            console,
            estimate_confidence=initial_confidence,
            estimate_confidence_detail=describe_estimate_confidence(analysis_items),
            size_confidence=estimate_size_confidence(
                analysis_items,
                use_calibration=use_calibration,
            ),
            size_confidence_detail=describe_size_confidence(
                analysis_items,
                use_calibration=use_calibration,
            ),
            time_confidence=estimate_time_confidence(
                analysis_items,
                benchmarked_files=0,
                use_calibration=use_calibration,
            ),
            time_confidence_detail=describe_time_confidence(
                analysis_items,
                benchmarked_files=0,
                use_calibration=use_calibration,
            ),
            notes=early_notes,
            compatibility_signals=compatibility_signals,
            calibration_store=load_calibration_store() if use_calibration else None,
            plain_output=plain_output or requested_non_interactive,
        )

        recommended_items = [
            item for item in analysis_items if item.recommendation == "recommended"
        ]
        maybe_items = [item for item in analysis_items if item.recommendation == "maybe"]
        if not recommended_items:
            if not maybe_items:
                console.print(
                    "[dim]No recommended files were found for automatic compression.[/dim]"
                )
                return [], "cancel", False, None
            strongest_maybe_items = _strongest_maybe_items(maybe_items)
            strongest_sources = {item.source for item in strongest_maybe_items}
            console.print(
                "[dim]No strong auto-selections were found, but some maybe files still look worthwhile for TV/library cleanup.[/dim]"
            )
            console.print(
                f"[dim]Default shortlist:[/dim] {len(strongest_maybe_items)} strongest maybe file(s); "
                f"{max(len(maybe_items) - len(strongest_maybe_items), 0)} broader maybe file(s) left for later review."
            )
            recommended_items = strongest_maybe_items
            maybe_items = [item for item in maybe_items if item.source not in strongest_sources]

        candidate_items = recommended_items + maybe_items
        mkv_candidate_count, mkv_grouped_reasons, mkv_examples = _summarize_mkv_suitable_candidates(
            candidate_items,
            ffprobe,
        )
        if mkv_candidate_count and output_dir is None:
            console.print(
                f"[bold yellow]MKV-first warning:[/bold yellow] {mkv_candidate_count} likely encode candidate(s) contain streams better suited to MKV sidecar output."
            )
            console.print(
                f"  [dim]{_format_grouped_reason_list(mkv_grouped_reasons)}[/dim]"
                + (f" [dim](examples: {', '.join(mkv_examples)})[/dim]" if mkv_examples else "")
            )
            console.print(
                "  [dim]Container/copied-stream issues look like the main blocker here. The wizard can switch incompatible files into MKV sidecar output instead of leaving them only in a manifest.[/dim]"
            )
            session.add_event(
                f"Early MKV warning: {mkv_candidate_count} candidate(s) likely need MKV sidecar output"
            )
        sample_pool = recommended_items or maybe_items or analysis_items
        representative_pool = select_representative_items(sample_pool, limit=3) or sample_pool
        sample_item = max(
            representative_pool,
            key=lambda item: (
                item.duration_seconds if item.duration_seconds > 0 else 0.0,
                item.size_bytes,
            ),
        )
        sample_file = sample_item.source
        available_hw = detect_available_encoders(
            ffmpeg, console, sample_file=sample_file, ffprobe=ffprobe
        )
        device_labels = detect_device_labels()

        if available_hw:
            console.print("[green]Hardware encoders available:[/green]")
            for key in available_hw:
                console.print(f"  - {_encoder_display_name(key, device_labels)}")
        else:
            console.print("[dim]No hardware encoders detected. Software only.[/dim]")
        console.print(
            f"[dim]Next: benchmark a representative candidate, suggest profiles for likely encode files, and optionally preview "
            f"{sample_file.name} before batch encoding.[/dim]"
        )
        console.print(
            "[dim]Benchmark and profile estimates are sample-based and exclude files already expected to be skipped.[/dim]"
        )

        planning = prepare_profile_planning(
            analysis_items=analysis_items,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            policy=policy,
            use_calibration=use_calibration,
            console=console,
            available_hw=available_hw,
        )
        assert planning is not None
        candidate_items = planning.candidate_items
        candidate_input_bytes = planning.candidate_input_bytes
        candidate_media_seconds = planning.candidate_media_seconds
        sample_item = planning.sample_item
        sample_duration = planning.sample_duration
        preview_items = planning.preview_items

        benchmark_speeds = planning.benchmark_speeds
        active_calibration = planning.active_calibration
        planning_bias_summary = recent_bias_summary(active_calibration)
        profiles = planning.profiles
        size_error_by_preset = planning.size_error_by_preset
        for profile in profiles:
            if profile.is_custom:
                continue
            rec_ok, rec_followup = _predict_compatibility_counts_for_items(
                profile,
                recommended_items,
                ffprobe=ffprobe,
                calibration_store=active_calibration,
                observed_probe_failures=planning.observed_probe_failures,
            )
            profile.recommended_compatible_count = rec_ok
            profile.recommended_incompatible_count = rec_followup
        if benchmark_speeds:
            console.print(
                f"[dim]Benchmarked {len([key for key, speed in benchmark_speeds.items() if speed is not None])} profile candidate(s).[/dim]"
            )
        probe_target_count = len(_iter_probe_targets(profiles))
        if probe_target_count:
            console.print(
                f"[dim]Smoke-probed {probe_target_count} risky profile combination(s).[/dim]"
            )
        display_result = display_profiles_table(
            profiles,
            candidate_input_bytes,
            len(candidate_items),
            len(recommended_items),
            device_labels,
            console,
            time_confidence=estimate_time_confidence(
                candidate_items,
                benchmarked_files=1,
                preset=next(
                    (profile.encoder_key for profile in profiles if profile.is_recommended), "fast"
                ),
                use_calibration=use_calibration,
            ),
            time_confidence_detail=describe_time_confidence(
                candidate_items,
                benchmarked_files=1,
                preset=next(
                    (profile.encoder_key for profile in profiles if profile.is_recommended), "fast"
                ),
                use_calibration=use_calibration,
            ),
            size_confidence=estimate_size_confidence(
                candidate_items,
                preset=next(
                    (profile.encoder_key for profile in profiles if profile.is_recommended), "fast"
                ),
                use_calibration=use_calibration,
            ),
            size_confidence_detail=describe_size_confidence(
                candidate_items,
                preset=next(
                    (profile.encoder_key for profile in profiles if profile.is_recommended), "fast"
                ),
                use_calibration=use_calibration,
            ),
            size_error_by_preset=size_error_by_preset,
            bias_note=(
                planning_bias_summary.get("summary")
                if isinstance(planning_bias_summary, dict)
                else None
            ),
            show_all_profiles=show_all_profiles,
            plain_output=plain_output or requested_non_interactive,
        )
        if isinstance(display_result, tuple):
            visible_profiles, display_index_map = display_result
        else:
            visible_profiles = display_result if isinstance(display_result, list) else profiles
            display_index_map = {i + 1: p for i, p in enumerate(visible_profiles)}

        recommended_profile = next(
            (profile for profile in profiles if profile.is_recommended), None
        )
        closest_history_line = (
            _closest_history_line(
                active_calibration,
                preset=recommended_profile.encoder_key,
                items=candidate_items,
            )
            if recommended_profile is not None
            else None
        )
        if closest_history_line:
            console.print(f"[dim]{closest_history_line}[/dim]")
        trust_line = _calibration_trust_line(active_calibration if use_calibration else None)
        if trust_line:
            console.print(f"[dim]{trust_line}[/dim]")

        for hw_key in available_hw:
            if hw_key in _HW_ENCODER_CAVEATS:
                console.print(
                    f"  [dim yellow]Note:[/dim yellow] [dim]{_HW_ENCODER_CAVEATS[hw_key]}[/dim]"
                )

        action_taken = False
        profile_saved = requested_non_interactive
        selected_items = list(recommended_items)
        estimated_total_encode_seconds: float | None = None
        followup_manifest_path: Path | None = None
        compatibility_prediction_note: str | None = None
        selected_profile_size_error: float | None = None
        ready_time_refinement_note: str | None = None
        selected_queue_strategy = "original"
        selected_queue_rationale = "Original order is fine for this run size."
        mkv_attempt_failed_count = 0
        mkv_attempt_failed_names: list[str] = []
        direct_followup_names: list[str] = []
        followup_file_names: list[str] = []
        original_benchmark_source = planning.sample_item.source
        default_scope_is_strongest_maybes = not any(
            item.recommendation == "recommended" for item in analysis_items
        ) and bool(recommended_items)
        selected_scope_label = _describe_selected_scope(
            selected_items,
            recommended_items=recommended_items,
            maybe_items=maybe_items,
        )
        active_auto = requested_non_interactive
        if active_auto:
            session.add_event("Wizard entered non-interactive mode")
        while True:
            try:
                preset, crf, sw_preset, display_label = _select_profile_interactively(
                    visible_profiles,
                    available_hw,
                    device_labels,
                    active_auto,
                    console,
                    display_index_map=display_index_map,
                )
            except _WizardFallbackRequested as exc:
                if active_auto:
                    raise typer.Abort() from exc
                active_auto = True
                requested_non_interactive = True
                session.input_mode = "non-interactive-fallback"
                session.plain_prompts = True
                session.add_event(f"Auto-fallback triggered: {exc}")
                console.print(
                    "[yellow]Detected unreliable terminal input. Switching to a safer plain non-interactive wizard flow.[/yellow]"
                )
                action_taken = False
                selected_items = list(recommended_items)
                selected_scope_label = _describe_selected_scope(
                    selected_items,
                    recommended_items=recommended_items,
                    maybe_items=maybe_items,
                )
                continue

            if mkv_candidate_count and output_dir is None:
                console.print(
                    f"[yellow]Selected profile expectation:[/yellow] {mkv_candidate_count} file(s) may still need MKV sidecar output after preflight."
                )

            if not active_auto and not profile_saved:
                maybe_save_profile(preset, crf, display_label, console)
                profile_saved = True

            if not _maybe_run_preview(
                preview_items,
                ffmpeg,
                ffprobe,
                preset,
                crf,
                active_auto,
                console,
                plain_output=plain_output or requested_non_interactive,
            ):
                return [], "cancel", False, None

            if not action_taken:
                action = (
                    "compress_recommended"
                    if active_auto
                    else prompt_analysis_action(
                        len(recommended_items),
                        len(maybe_items),
                        console,
                        recommended_label=(
                            "strongest maybe shortlist"
                            if default_scope_is_strongest_maybes
                            else "recommended"
                        ),
                        maybe_label=(
                            "remaining maybe" if default_scope_is_strongest_maybes else "maybe"
                        ),
                    )
                )
                if active_auto:
                    console.print(
                        "[dim]Next step selected:[/dim] "
                        + (
                            "Compress strongest maybe shortlist files"
                            if default_scope_is_strongest_maybes
                            else "Compress recommended files"
                        )
                    )
                if action == "cancel":
                    return [], "cancel", False, None
                if action == "export":
                    export_estimate = estimate_analysis_encode_seconds(
                        items=recommended_items,
                        preset=preset,
                        crf=crf,
                        ffmpeg=ffmpeg,
                        known_speed=benchmark_speeds.get(preset),
                    )
                    manifest = build_manifest(
                        directory=directory,
                        recursive=recursive,
                        preset=preset,
                        crf=crf,
                        profile_name=None,
                        estimated_total_encode_seconds=export_estimate,
                        estimate_confidence=estimate_analysis_confidence(
                            recommended_items, benchmarked_files=1
                        ),
                        size_confidence=estimate_size_confidence(
                            analysis_items,
                            preset=preset,
                            use_calibration=use_calibration,
                        ),
                        size_confidence_detail=describe_size_confidence(
                            analysis_items,
                            preset=preset,
                            use_calibration=use_calibration,
                        ),
                        time_confidence=estimate_time_confidence(
                            analysis_items,
                            benchmarked_files=1,
                            preset=preset,
                            use_calibration=use_calibration,
                        ),
                        time_confidence_detail=describe_time_confidence(
                            analysis_items,
                            benchmarked_files=1,
                            preset=preset,
                            use_calibration=use_calibration,
                        ),
                        duplicate_policy=duplicate_policy,
                        items=analysis_items,
                    )
                    default_manifest_path = directory / "mediashrink-analysis.json"
                    manifest_path = Path(
                        _wizard_prompt(
                            "Manifest path",
                            default=str(default_manifest_path),
                            prompt_id="manifest-path",
                            acceptance_label="Manifest path",
                        )
                    )
                    split_mode = "combined" if active_auto else prompt_manifest_split_mode(console)
                    if split_mode == "combined":
                        save_manifest(manifest, manifest_path)
                    else:
                        write_split_manifests(
                            directory=directory,
                            recursive=recursive,
                            preset=preset,
                            crf=crf,
                            profile_name=None,
                            estimated_total_encode_seconds=export_estimate,
                            estimate_confidence=estimate_analysis_confidence(
                                recommended_items, benchmarked_files=1
                            ),
                            size_confidence=estimate_size_confidence(
                                analysis_items,
                                preset=preset,
                                use_calibration=use_calibration,
                            ),
                            size_confidence_detail=describe_size_confidence(
                                analysis_items,
                                preset=preset,
                                use_calibration=use_calibration,
                            ),
                            time_confidence=estimate_time_confidence(
                                analysis_items,
                                benchmarked_files=1,
                                preset=preset,
                                use_calibration=use_calibration,
                            ),
                            time_confidence_detail=describe_time_confidence(
                                analysis_items,
                                benchmarked_files=1,
                                preset=preset,
                                use_calibration=use_calibration,
                            ),
                            duplicate_policy=duplicate_policy,
                            items=recommended_items,
                            split_by=split_mode,
                            index_path=manifest_path,
                            notes=duplicate_notes,
                        )
                    console.print(f"[green]Wrote manifest[/green] {manifest_path}")
                    return [], "export", False, None

                selected_items = list(recommended_items)
                if (
                    action == "review_maybe"
                    and maybe_items
                    and review_maybe_items(
                        maybe_items,
                        console,
                        title=(
                            "Remaining maybe files"
                            if default_scope_is_strongest_maybes
                            else "Maybe files"
                        ),
                        prompt_text=(
                            "Include the remaining maybe files in this run?"
                            if default_scope_is_strongest_maybes
                            else "Include maybe files in this run?"
                        ),
                        decision_label=(
                            "Remaining maybe files decision"
                            if default_scope_is_strongest_maybes
                            else "Maybe files decision"
                        ),
                    )
                ):
                    selected_items.extend(maybe_items)
                selected_scope_label = _describe_selected_scope(
                    selected_items,
                    recommended_items=recommended_items,
                    maybe_items=maybe_items,
                )
                action_taken = True
                session.add_event(f"Selected run scope: {selected_scope_label}")

            followup_count = 0  # files moved to follow-up manifest due to incompatibility
            mkv_sidecar_count = 0
            mkv_attempt_failed_count = 0
            mkv_attempt_failed_names = []
            direct_followup_names = []
            followup_file_names = []
            selected_queue_strategy, selected_queue_rationale = _auto_queue_strategy_for_items(
                selected_items
            )
            if selected_queue_strategy == "largest-first":
                selected_items = sorted(
                    selected_items,
                    key=lambda item: (-item.size_bytes, item.source.name.lower()),
                )
            elif selected_queue_strategy == "safe-first":
                selected_items = sorted(
                    selected_items,
                    key=lambda item: (
                        1 if item.source.suffix.lower() in {".mp4", ".m4v"} else 0,
                        -item.size_bytes,
                        item.source.name.lower(),
                    ),
                )
            jobs = build_jobs(
                files=[item.source for item in selected_items],
                output_dir=output_dir,
                overwrite=overwrite,
                crf=crf,
                preset=preset,
                dry_run=False,
                ffprobe=ffprobe,
                no_skip=no_skip,
            )
            to_encode = [job for job in jobs if not job.skip]
            if not to_encode:
                console.print("[dim]Nothing to encode after re-checking the selected files.[/dim]")
                return [], "cancel", False, None

            estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                items=selected_items,
                preset=preset,
                crf=crf,
                ffmpeg=ffmpeg,
                known_speed=benchmark_speeds.get(preset),
                calibration_store=active_calibration,
            )
            selected_profile = next(
                (
                    profile
                    for profile in profiles
                    if not profile.is_custom
                    and profile.encoder_key == preset
                    and profile.crf == crf
                ),
                None,
            )
            selected_profile_size_error = _average_size_error_for_items(
                preset=preset,
                items=selected_items,
                ffprobe=ffprobe,
                calibration_store=active_calibration,
            )
            selected_profile_speed_error = _average_speed_error_for_items(
                preset=preset,
                items=selected_items,
                ffprobe=ffprobe,
                calibration_store=active_calibration,
            )
            compatibility_prediction_note = None
            if selected_profile is not None:
                predicted_compatible, predicted_incompatible, _ = _predict_profile_compatibility(
                    profile=selected_profile,
                    items=selected_items,
                    ffprobe=ffprobe,
                    failure_rate=estimate_failure_rate(
                        active_calibration,
                        preset=selected_profile.encoder_key,
                        container=selected_items[0].source.suffix.lower()
                        if selected_items
                        else ".mkv",
                    ),
                    observed_probe_failures=planning.observed_probe_failures,
                )
                if predicted_incompatible > 0:
                    compatibility_prediction_note = (
                        f"Selected run scope: {selected_scope_label}; this profile is estimated to work for "
                        f"{predicted_compatible} selected file(s), with {predicted_incompatible} likely follow-up."
                    )
                else:
                    compatibility_prediction_note = (
                        f"Selected run scope: {selected_scope_label}; this profile currently looks compatible "
                        f"for all {predicted_compatible} selected file(s)."
                    )

            compatible_jobs, preflight_failures = _run_preflight_checks(
                to_encode,
                ffmpeg,
                ffprobe,
                crf=crf,
                preset=preset,
                console=console,
            )
            if not preflight_failures:
                break

            console.print()
            console.print(
                "[red]Selected settings failed a short compatibility check before batch encoding.[/red]"
            )
            console.print(
                f"[red]Profile:[/red] {display_label} "
                f"([red]encoder:[/red] {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'})"
            )
            grouped_failures = _group_preflight_failures(preflight_failures, ffprobe)
            for reason, jobs_for_reason in grouped_failures.items():
                examples = ", ".join(job.source.name for job in jobs_for_reason[:3])
                console.print(
                    f"[yellow]{len(jobs_for_reason)} file(s):[/yellow] {reason}"
                    + (f" [dim]({examples})[/dim]" if examples else "")
                )
            if preset in _HW_ENCODERS:
                output_header_hint = (
                    "FFmpeg could not write the output file header. "
                    "This can be caused by MP4 copied-stream compatibility, and hardware paths can make it show up sooner. "
                    "Try the Balanced (software libx265) profile or write .mkv output."
                )
            else:
                output_header_hint = (
                    "FFmpeg could not write the output file header. "
                    "This usually points to container or copied-stream compatibility for this source. "
                    "Try writing .mkv output or excluding the incompatible copied streams."
                )
            _PREFLIGHT_HINTS: dict[str, str] = {
                "unsupported copied audio codec": (
                    "The audio codec cannot be copied into this output container. "
                    "Try a software profile, or use --output-dir with an .mkv destination."
                ),
                "unsupported copied subtitle codec": (
                    "A copied subtitle stream is not compatible with this output container. "
                    "Use MKV output if you need to preserve it."
                ),
                "attachment stream incompatibility": (
                    "Attachment streams cannot be preserved in this output container. "
                    "Use --output-dir to write .mkv output instead."
                ),
                "auxiliary data stream incompatibility": (
                    "Auxiliary data streams cannot be preserved in this output container. "
                    "Use --output-dir to write .mkv output instead."
                ),
                "output header failure": output_header_hint,
                "hardware encoder startup failure": (
                    "The hardware encoder failed to initialise for this file. Try a software profile."
                ),
            }
            shown_hints: set[str] = set()
            for reason in grouped_failures:
                hint = _PREFLIGHT_HINTS.get(reason)
                if hint and hint not in shown_hints:
                    console.print(f"  [dim]{hint}[/dim]")
                    shown_hints.add(hint)

            if compatible_jobs and len(compatible_jobs) < len(to_encode):
                originally_selected_items = list(selected_items)
                compatible_sources = {job.source for job in compatible_jobs}
                incompatible_items = [
                    item for item in selected_items if item.source not in compatible_sources
                ]
                remaining_incompatible_items = list(incompatible_items)
                retained_jobs = [
                    job for job in jobs if job.skip or job.source in compatible_sources
                ]
                mkv_switched_jobs: list[EncodeJob] = []
                mkv_safe_reasons = {
                    "unsupported copied audio codec",
                    "unsupported copied subtitle codec",
                    "attachment stream incompatibility",
                    "auxiliary data stream incompatibility",
                    "output header failure",
                }
                if grouped_failures and all(
                    reason in mkv_safe_reasons for reason in grouped_failures
                ):
                    mkv_followup_dir = _default_mkv_followup_dir(directory, output_dir)
                    console.print(
                        f"[dim]Auto-routing {len(incompatible_items)} container-risk file(s) to MKV sidecar preflight in {mkv_followup_dir}.[/dim]"
                    )
                    candidate_mkv_jobs = _build_mkv_followup_jobs(
                        incompatible_items,
                        output_dir=mkv_followup_dir,
                        overwrite=False,
                        crf=crf,
                        preset=preset,
                        ffprobe=ffprobe,
                        no_skip=no_skip,
                    )
                    mkv_compatible_jobs, _mkv_failures = _run_preflight_checks(
                        [job for job in candidate_mkv_jobs if not job.skip],
                        ffmpeg,
                        ffprobe,
                        crf=crf,
                        preset=preset,
                        console=console,
                    )
                    mkv_sources = {job.source for job in mkv_compatible_jobs}
                    if mkv_compatible_jobs:
                        mkv_switched_jobs = mkv_compatible_jobs
                        mkv_sidecar_count = len(mkv_switched_jobs)
                        retained_jobs.extend(mkv_compatible_jobs)
                        remaining_incompatible_items = [
                            item for item in incompatible_items if item.source not in mkv_sources
                        ]
                        console.print(
                            f"[green]Switched {len(mkv_compatible_jobs)} file(s) to MKV sidecar output in[/green] {mkv_followup_dir}"
                        )
                        session.add_event(
                            f"Preflight switched {len(mkv_compatible_jobs)} file(s) to MKV sidecar output"
                        )
                    failed_mkv_sources = [
                        job.source
                        for job in candidate_mkv_jobs
                        if job.source not in mkv_sources and not job.skip
                    ]
                    if failed_mkv_sources:
                        mkv_attempt_failed_count = len(failed_mkv_sources)
                        mkv_attempt_failed_names = [path.name for path in failed_mkv_sources[:3]]
                        console.print(
                            f"[yellow]Tried MKV sidecar output for {len(failed_mkv_sources)} file(s), but they still needed follow-up.[/yellow]"
                        )
                followup_manifest = _write_followup_manifest(
                    directory,
                    recursive,
                    preset,
                    crf,
                    remaining_incompatible_items,
                    notes=_followup_manifest_notes(
                        grouped_failures,
                        remaining_incompatible_items,
                        ffprobe=ffprobe,
                    ),
                )
                followup_manifest_path = followup_manifest
                followup_count = len(remaining_incompatible_items)
                followup_file_names = [
                    item.source.name for item in remaining_incompatible_items[:5]
                ]
                direct_followup_names = [
                    item.source.name
                    for item in remaining_incompatible_items
                    if item.source.name not in mkv_attempt_failed_names
                ][:5]
                included_sources = compatible_sources | {job.source for job in mkv_switched_jobs}
                selected_items = [
                    item for item in selected_items if item.source in included_sources
                ]
                jobs = retained_jobs
                to_encode = compatible_jobs + mkv_switched_jobs
                refined_speed = benchmark_speeds.get(preset)
                if selected_items and original_benchmark_source not in {
                    item.source for item in selected_items
                }:
                    replacement_sample = select_representative_items(selected_items, limit=1)
                    if replacement_sample:
                        refined_speed = (
                            benchmark_encoder(
                                encoder_key=preset,
                                sample_file=replacement_sample[0].source,
                                sample_duration=max(replacement_sample[0].duration_seconds, 600.0),
                                crf=crf,
                                ffmpeg=ffmpeg,
                            )
                            or refined_speed
                        )
                        ready_time_refinement_note = "Time estimate was re-benchmarked after the original sample file moved to follow-up."
                estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                    items=selected_items,
                    preset=preset,
                    crf=crf,
                    ffmpeg=ffmpeg,
                    known_speed=refined_speed,
                    calibration_store=active_calibration,
                )
                if selected_profile is not None:
                    predicted_compatible, predicted_incompatible, _ = (
                        _predict_profile_compatibility(
                            profile=selected_profile,
                            items=originally_selected_items,
                            ffprobe=ffprobe,
                            failure_rate=estimate_failure_rate(
                                active_calibration,
                                preset=selected_profile.encoder_key,
                                container=(
                                    selected_items[0].source.suffix.lower()
                                    if selected_items
                                    else ".mkv"
                                ),
                            ),
                            observed_probe_failures=planning.observed_probe_failures,
                        )
                    )
                    compatibility_prediction_note = (
                        "Predicted compatibility for this selection: "
                        f"{predicted_compatible} compatible / {predicted_incompatible} likely follow-up; "
                        + _compact_split_summary(
                            len(originally_selected_items),
                            len(compatible_jobs),
                            len(mkv_switched_jobs),
                            len(remaining_incompatible_items),
                        )
                        + "."
                    )
                else:
                    compatibility_prediction_note = (
                        "Preflight confirmed "
                        + _compact_split_summary(
                            len(originally_selected_items),
                            len(compatible_jobs),
                            len(mkv_switched_jobs),
                            len(remaining_incompatible_items),
                        )
                        + "."
                    )
                console.print(
                    f"[yellow]{len(compatible_jobs)} file(s) can run now with {display_label}. "
                    + (
                        f"{len(mkv_switched_jobs)} file(s) were switched to MKV sidecar output, "
                        if mkv_switched_jobs
                        else ""
                    )
                    + (
                        f"{mkv_attempt_failed_count} still needed follow-up after an MKV retry, "
                        if mkv_attempt_failed_count
                        else ""
                    )
                    + (
                        f"{len(remaining_incompatible_items)} incompatible file(s) were moved to follow-up planning.[/yellow]"
                        if remaining_incompatible_items
                        else "no manual follow-up remains.[/yellow]"
                    )
                )
                if followup_manifest is not None:
                    console.print(f"[dim]Follow-up manifest:[/dim] {followup_manifest}")
                    console.print(
                        f'  [dim]To encode these with a different profile: mediashrink apply "{followup_manifest}"[/dim]'
                    )
                    if grouped_failures and set(grouped_failures).issubset(
                        {
                            "unsupported copied audio codec",
                            "unsupported copied subtitle codec",
                            "attachment stream incompatibility",
                            "auxiliary data stream incompatibility",
                            "output header failure",
                        }
                    ):
                        console.print(
                            f'  [dim]MKV-first retry command: mediashrink apply "{followup_manifest}" --output-dir "{_default_mkv_followup_dir(directory, output_dir)}"[/dim]'
                        )
                    console.print(
                        f"  [dim]{_followup_next_step_hint(preset=preset, grouped_failures=grouped_failures)}[/dim]"
                    )
                    for detail in _per_file_followup_details(
                        grouped_failures,
                        remaining_incompatible_items,
                        ffprobe=ffprobe,
                    )[:5]:
                        console.print(f"  [dim]- {detail}[/dim]")
                break

            fallback_preset, fallback_crf, fallback_label, fallback_sw_preset = (
                _DEFAULT_FALLBACK_PROFILE
            )
            should_try_fallback = on_file_failure != "stop" and (
                active_auto
                or _wizard_confirm(
                    f"Switch to {fallback_label} (libx265, CRF {fallback_crf}) and retry?",
                    default=True,
                    prompt_id="fallback-profile",
                    acceptance_label="Fallback profile decision",
                )
            )
            if preset != fallback_preset and should_try_fallback:
                fallback_jobs = build_jobs(
                    files=[item.source for item in selected_items],
                    output_dir=output_dir,
                    overwrite=overwrite,
                    crf=fallback_crf,
                    preset=fallback_preset,
                    dry_run=False,
                    ffprobe=ffprobe,
                    no_skip=no_skip,
                )
                fallback_to_encode = [job for job in fallback_jobs if not job.skip]
                console.print(f"[dim]Retrying with[/dim] {fallback_label}...")
                fallback_compatible, fallback_failures = _run_preflight_checks(
                    fallback_to_encode,
                    ffmpeg,
                    ffprobe,
                    crf=fallback_crf,
                    preset=fallback_preset,
                    console=console,
                )
                if not fallback_failures:
                    jobs = fallback_jobs
                    to_encode = fallback_compatible
                    preset = fallback_preset
                    crf = fallback_crf
                    sw_preset = fallback_sw_preset
                    display_label = fallback_label
                    estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                        items=selected_items,
                        preset=preset,
                        crf=crf,
                        ffmpeg=ffmpeg,
                        known_speed=benchmark_speeds.get(preset),
                        calibration_store=active_calibration,
                    )
                    selected_profile_size_error = _average_size_error_for_items(
                        preset=preset,
                        items=selected_items,
                        ffprobe=ffprobe,
                        calibration_store=active_calibration,
                    )
                    selected_profile_speed_error = _average_speed_error_for_items(
                        preset=preset,
                        items=selected_items,
                        ffprobe=ffprobe,
                        calibration_store=active_calibration,
                    )
                    break
                if fallback_compatible and len(fallback_compatible) < len(fallback_to_encode):
                    fallback_sources = {job.source for job in fallback_compatible}
                    if (
                        on_file_failure == "skip"
                        or active_auto
                        or _wizard_confirm(
                            f"Skip {len(fallback_failures)} incompatible file(s) and continue with {len(fallback_compatible)} compatible file(s) using {fallback_label}?",
                            default=True,
                            prompt_id="fallback-skip-incompatible",
                            acceptance_label="Fallback skip decision",
                        )
                    ):
                        selected_items = [
                            item for item in selected_items if item.source in fallback_sources
                        ]
                        jobs = [
                            job
                            for job in fallback_jobs
                            if job.skip or job.source in fallback_sources
                        ]
                        to_encode = fallback_compatible
                        preset = fallback_preset
                        crf = fallback_crf
                        sw_preset = fallback_sw_preset
                        display_label = fallback_label
                        estimated_total_encode_seconds = estimate_analysis_encode_seconds(
                            items=selected_items,
                            preset=preset,
                            crf=crf,
                            ffmpeg=ffmpeg,
                            known_speed=benchmark_speeds.get(preset),
                            calibration_store=active_calibration,
                        )
                        selected_profile_size_error = _average_size_error_for_items(
                            preset=preset,
                            items=selected_items,
                            ffprobe=ffprobe,
                            calibration_store=active_calibration,
                        )
                        selected_profile_speed_error = _average_speed_error_for_items(
                            preset=preset,
                            items=selected_items,
                            ffprobe=ffprobe,
                            calibration_store=active_calibration,
                        )
                        console.print(
                            f"[yellow]Skipping {len(fallback_failures)} incompatible file(s) for this run.[/yellow]"
                        )
                        break
                console.print()
                console.print("[red]Fallback profile also failed compatibility checks.[/red]")
                for failed_job, fallback_result in fallback_failures[:5]:
                    if fallback_result.error_message:
                        console.print(
                            f"[red]{failed_job.source.name}:[/red] {fallback_result.error_message}"
                        )

            if on_file_failure == "stop":
                return [], "cancel", False, None

            console.print(
                "[dim]Choose another profile. Software profiles are the safest fallback when hardware encoding is not compatible with the selected files.[/dim]"
            )

        console.print()
        console.print("[bold]Ready to encode[/bold]")
        console.print(f"  Files:    {len(to_encode)}")
        console.print(
            f"  Encoder:  {_encoder_display_name(preset, device_labels) if preset in _HW_ENCODERS else f'libx265 ({sw_preset or preset})'}"
        )
        console.print(f"  CRF:      {crf}")
        selected_input_bytes = sum(item.size_bytes for item in selected_items)
        selected_output_bytes = _estimate_selected_output_bytes(
            selected_items,
            ffprobe=ffprobe,
            preset=preset,
            crf=crf,
            use_calibration=use_calibration,
            calibration_store=active_calibration,
        )
        size_confidence_label, time_confidence_label = _post_split_confidence_labels(
            selected_items,
            original_items=candidate_items,
            preset=preset,
            use_calibration=use_calibration,
            benchmarked_files=1,
            sidecar_count=mkv_sidecar_count,
            followup_count=followup_count,
        )
        time_confidence_scope_note = describe_time_confidence_scope_adjustment(
            selected_items,
            original_items=candidate_items,
            sidecar_count=mkv_sidecar_count,
            followup_count=followup_count,
        )
        console.print(f"  Input:    {_fmt_size(selected_input_bytes)}")
        maybe_left_out = max(
            0,
            sum(1 for item in analysis_items if item.recommendation == "maybe")
            - sum(1 for item in selected_items if item.recommendation == "maybe"),
        )
        skip_left_out = sum(1 for item in analysis_items if item.recommendation == "skip")
        if followup_count > 0:
            console.print(
                f"  [yellow]Moved to follow-up:[/yellow] {followup_count} file(s) failed "
                "compatibility check; see follow-up manifest above."
            )
            if followup_count == 1 and followup_file_names:
                console.print(f"  [dim]Follow-up file:[/dim] {followup_file_names[0]}")
        if mkv_attempt_failed_count and mkv_attempt_failed_names:
            console.print(
                "  [dim]MKV retry still left out:[/dim] " + ", ".join(mkv_attempt_failed_names)
            )
        if compatibility_prediction_note:
            console.print(f"  [dim]{compatibility_prediction_note}[/dim]")
        elif followup_count == 0:
            console.print(
                f"  [dim]Selected run scope:[/dim] {selected_scope_label}; no compatibility split is currently expected for this run."
            )
        if maybe_left_out > 0:
            console.print(
                f"  [dim]Not in this run:[/dim] {maybe_left_out} maybe file(s) were left out by choice."
            )
        if skip_left_out > 0:
            console.print(
                f"  [dim]Skipped before encode:[/dim] {skip_left_out} file(s) were already HEVC or otherwise marked skip."
            )
        left_out_items = [
            item
            for item in analysis_items
            if item not in selected_items and item.recommendation in {"recommended", "maybe"}
        ]
        for line in _cohort_guidance_lines(
            selected_items=selected_items,
            left_out_items=left_out_items,
        ):
            console.print(f"  [dim]{line}[/dim]")
        if selected_output_bytes > 0:
            console.print(
                _format_ready_size_estimate(
                    input_bytes=selected_input_bytes,
                    output_bytes=selected_output_bytes,
                    confidence=size_confidence_label,
                    size_error=selected_profile_size_error,
                )
            )
            if selected_profile_size_error is not None and abs(selected_profile_size_error) >= 0.25:
                console.print(
                    "  [dim]Local history for this encoder/profile is too inconsistent to offer a tighter output-size estimate for the surviving batch.[/dim]"
                )
        if estimated_total_encode_seconds is not None and estimated_total_encode_seconds > 0:
            time_widen = estimate_time_range_widening(
                selected_items,
                preset=preset,
                benchmarked_files=1,
                calibration_store=active_calibration,
                use_calibration=use_calibration,
            )
            time_low, time_high = estimate_value_range(
                estimated_total_encode_seconds,
                confidence=time_confidence_label,
                average_error=selected_profile_speed_error,
                widen_by=(0.04 if (mkv_sidecar_count or followup_count) else 0.0) + time_widen,
            )
            console.print(f"  Est. time: {_format_duration_range(time_low, time_high)}")
            if ready_time_refinement_note:
                console.print(f"  [dim]{ready_time_refinement_note}[/dim]")
            elif (
                selected_profile is not None
                and selected_profile.estimated_encode_seconds > 0
                and abs(selected_profile.estimated_encode_seconds - estimated_total_encode_seconds)
                >= max(60.0, selected_profile.estimated_encode_seconds * 0.10)
            ):
                console.print(
                    "  [dim]Time estimate was refined after selection using the exact run scope and current calibration data.[/dim]"
                )
        ready_guidance = _large_batch_guidance(
            selected_count=len(selected_items),
            total_candidates=len(candidate_items),
            estimated_seconds=estimated_total_encode_seconds,
        )
        if ready_guidance:
            console.print(f"  [dim]{ready_guidance}[/dim]")
        console.print(f"  Size confidence: {size_confidence_label}")
        console.print(f"  Time confidence: {time_confidence_label}")
        if time_confidence_scope_note:
            console.print(f"  [dim]Time confidence note: {time_confidence_scope_note}.[/dim]")
        ready_bias_summary = recent_bias_summary(active_calibration if use_calibration else None)
        if isinstance(ready_bias_summary, dict) and isinstance(
            ready_bias_summary.get("summary"), str
        ):
            console.print(f"  [dim]Estimate bias: {ready_bias_summary['summary']}.[/dim]")
        if preset in _HW_ENCODERS:
            console.print(
                "  [dim]Hardware encoding is faster, but source duration and bitrate still dominate total runtime.[/dim]"
            )
        elif preset in {"faster", "ultrafast"}:
            console.print(
                "  [dim]This favors faster completion over maximum compression efficiency.[/dim]"
            )
        else:
            console.print(
                "  [dim]Slower software presets trade more time for slightly smaller files.[/dim]"
            )

        if not to_encode[0].output.parent.exists():
            console.print(f"  Output:   {to_encode[0].output.parent}")
        subtitle_warning = _subtitle_drop_warning(to_encode, ffprobe)
        if subtitle_warning:
            console.print(f"  [yellow]{subtitle_warning}[/yellow]")
        queue_hint = _queue_strategy_recommendation(selected_items)
        if queue_hint:
            console.print(f"  [dim]Queue hint: {queue_hint}[/dim]")
        console.print(
            f"  [dim]Queue strategy:[/dim] {selected_queue_strategy} - {selected_queue_rationale}"
        )
        console.print("  [dim]Estimates are approximate.[/dim]")
        console.print(
            "  [dim]Safe to stop with Ctrl+C: completed files stay done, the current temp output is discarded, and you can resume unfinished files later.[/dim]"
        )

        cleanup_after = False
        if not overwrite and output_dir is None and not active_auto:
            cleanup_after = _wizard_confirm(
                "Delete originals only after successful side-by-side encodes?",
                default=False,
                prompt_id="cleanup-after",
                acceptance_label="Cleanup decision",
            )
            console.print(
                f"[dim]Cleanup decision:[/dim] {'Delete originals after success' if cleanup_after else 'Keep originals'}"
            )
        for cleanup_line in _cleanup_expectation_lines(to_encode, cleanup_after=cleanup_after):
            console.print(f"  [dim]{cleanup_line}[/dim]")
        console.print()

        if not active_auto:
            if not _wizard_confirm(
                "Start encoding?",
                default=True,
                prompt_id="start-encoding",
                acceptance_label="Start encoding decision",
            ):
                console.print("[dim]Start encoding decision:[/dim] Cancelled")
                return [], "cancel", False, None
            console.print("[dim]Start encoding decision:[/dim] Begin encoding")
        session.add_event(
            f"Follow-up manifest path: {followup_manifest_path}"
            if followup_manifest_path
            else "No follow-up manifest written"
        )
        _LAST_WIZARD_REPORT_CONTEXT = {
            "estimate_context": {
                "initial_scope": "recommended default scope",
                "initial_estimated_seconds": selected_profile.estimated_encode_seconds
                if selected_profile is not None and selected_profile.estimated_encode_seconds > 0
                else None,
                "selected_scope_label": selected_scope_label,
                "selected_estimated_seconds": estimated_total_encode_seconds,
                "rebenchmarked_after_split": bool(ready_time_refinement_note),
                "original_benchmark_source": original_benchmark_source.name,
                "benchmark_sources": [item.source.name for item in planning.benchmark_items],
            },
            "estimate_ranges": {
                "output_bytes": (
                    {
                        "low": int(
                            estimate_value_range(
                                float(selected_output_bytes),
                                confidence=size_confidence_label,
                                average_error=selected_profile_size_error,
                            )[0]
                        ),
                        "high": int(
                            estimate_value_range(
                                float(selected_output_bytes),
                                confidence=size_confidence_label,
                                average_error=selected_profile_size_error,
                            )[1]
                        ),
                    }
                    if selected_output_bytes > 0
                    else None
                ),
                "saved_bytes": (
                    {
                        "low": max(
                            selected_input_bytes
                            - int(
                                estimate_value_range(
                                    float(selected_output_bytes),
                                    confidence=size_confidence_label,
                                    average_error=selected_profile_size_error,
                                )[1]
                            ),
                            0,
                        ),
                        "high": max(
                            selected_input_bytes
                            - int(
                                estimate_value_range(
                                    float(selected_output_bytes),
                                    confidence=size_confidence_label,
                                    average_error=selected_profile_size_error,
                                )[0]
                            ),
                            0,
                        ),
                    }
                    if selected_output_bytes > 0
                    else None
                ),
                "encode_seconds": (
                    {
                        "low": estimate_value_range(
                            estimated_total_encode_seconds,
                            confidence=time_confidence_label,
                            average_error=selected_profile_speed_error,
                            widen_by=0.04 if (mkv_sidecar_count or followup_count) else 0.0,
                        )[0],
                        "high": estimate_value_range(
                            estimated_total_encode_seconds,
                            confidence=time_confidence_label,
                            average_error=selected_profile_speed_error,
                            widen_by=0.04 if (mkv_sidecar_count or followup_count) else 0.0,
                        )[1],
                    }
                    if estimated_total_encode_seconds is not None
                    and estimated_total_encode_seconds > 0
                    else None
                ),
                "bias_note": (
                    ready_bias_summary.get("summary")
                    if isinstance(ready_bias_summary, dict)
                    else None
                ),
            },
            "container_fallback_actions": {
                "mkv_sidecar_outputs": mkv_sidecar_count,
                "mkv_retry_failed_count": mkv_attempt_failed_count,
                "mkv_retry_failed_names": mkv_attempt_failed_names,
                "followup_manifest": str(followup_manifest_path)
                if followup_manifest_path is not None
                else None,
                "followup_count": followup_count,
                "excluded_files": [
                    {
                        "name": name,
                        "reason": "container/copied-stream incompatibility",
                        "next_step": "Use MKV output first.",
                    }
                    for name in followup_file_names
                ],
            },
            "queue_decision": {
                "strategy": selected_queue_strategy,
                "rationale": selected_queue_rationale,
            },
            "filename_hygiene": filename_hygiene_actions,
        }
        return jobs, "encode", cleanup_after, followup_manifest_path
    finally:
        debug_path = _write_debug_session_log()
        if debug_path is not None:
            console.print(f"[dim]Wizard debug log:[/dim] {debug_path}")
        _ACTIVE_WIZARD_SESSION = prior_session
