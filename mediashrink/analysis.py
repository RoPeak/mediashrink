from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mediashrink.calibration import (
    bitrate_bucket,
    describe_calibration_estimate,
    describe_history_slices,
    estimate_display_uncertainty,
    format_family_container_summary,
    load_calibration_store,
    lookup_estimate,
    recent_bias_summary,
    resolution_bucket,
    summarize_calibration_store,
)
from mediashrink.encoder import (
    _HW_ENCODERS,
    describe_output_container_constraints,
    estimate_output_size,
    get_duration_seconds,
    get_video_bitrate_kbps,
)
from mediashrink.models import AnalysisItem, AnalysisManifest
from mediashrink.scanner import (
    apply_duplicate_title_policy,
    is_already_compressed,
    probe_video_codec,
    scan_directory,
)

MANIFEST_VERSION = 1

_GB = 1024**3
_MB = 1024**2
_RECOMMENDED_CODECS = {"h264", "vc1", "mpeg2video"}

_MIN_SKIP_SAVINGS_BYTES = 250 * _MB
_MIN_SKIP_SAVINGS_PCT = 10.0
_MIN_RECOMMENDED_SAVINGS_BYTES = 1 * _GB
_MIN_RECOMMENDED_SAVINGS_PCT = 25.0
_HIGH_SIZE_ERROR_THRESHOLD = 0.18
_VERY_HIGH_SIZE_ERROR_THRESHOLD = 0.25


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


def _average_size_error_for_items(
    items: list[AnalysisItem],
    *,
    preset: str,
    use_calibration: bool,
    calibration_store: dict[str, object] | None = None,
) -> float | None:
    if not use_calibration:
        return None
    candidates = [item for item in items if item.recommendation != "skip"]
    if not candidates:
        return None
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    errors: list[float] = []
    for item in candidates[:5]:
        estimate = lookup_estimate(
            active_store,
            codec=item.codec,
            resolution="unknown",
            bitrate="unknown",
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if estimate is not None and estimate.average_size_error is not None:
            errors.append(estimate.average_size_error)
    if not errors:
        return None
    return sum(errors) / len(errors)


def _candidate_items(items: list[AnalysisItem]) -> list[AnalysisItem]:
    return [item for item in items if item.recommendation != "skip"]


def _selected_batch_complexity_notes(
    items: list[AnalysisItem],
    *,
    original_items: list[AnalysisItem] | None = None,
    sidecar_count: int = 0,
    followup_count: int = 0,
) -> list[str]:
    candidates = _candidate_items(items)
    if not candidates:
        return []
    notes: list[str] = []
    codec_count = len({item.codec or "unknown" for item in candidates})
    container_count = len({item.source.suffix.lower() or ".mkv" for item in candidates})
    if codec_count >= 2:
        notes.append(f"{codec_count} codec groups")
    if container_count >= 2:
        notes.append(f"{container_count} containers")
    if sidecar_count > 0:
        notes.append(f"{sidecar_count} MKV sidecar output{'s' if sidecar_count != 1 else ''}")
    if followup_count > 0:
        notes.append(f"{followup_count} follow-up split{'s' if followup_count != 1 else ''}")
    if original_items is not None and len(candidates) < len(_candidate_items(original_items)):
        notes.append("selected scope differs from the original recommendation set")
    return notes


def adjust_time_confidence_for_scope(
    label: str,
    items: list[AnalysisItem],
    *,
    original_items: list[AnalysisItem] | None = None,
    sidecar_count: int = 0,
    followup_count: int = 0,
) -> str:
    notes = _selected_batch_complexity_notes(
        items,
        original_items=original_items,
        sidecar_count=sidecar_count,
        followup_count=followup_count,
    )
    if not notes:
        return label
    lowered = set(notes)
    downgrade = False
    if any("codec groups" in note for note in lowered):
        downgrade = True
    if any("containers" in note for note in lowered):
        downgrade = True
    if sidecar_count > 0 or followup_count > 0:
        downgrade = True
    if downgrade and label == "High":
        return "Medium"
    return label


def describe_time_confidence_scope_adjustment(
    items: list[AnalysisItem],
    *,
    original_items: list[AnalysisItem] | None = None,
    sidecar_count: int = 0,
    followup_count: int = 0,
) -> str | None:
    notes = _selected_batch_complexity_notes(
        items,
        original_items=original_items,
        sidecar_count=sidecar_count,
        followup_count=followup_count,
    )
    if not notes:
        return None
    return "reduced by " + ", ".join(notes)


def build_analysis_item(
    path: Path,
    ffprobe: Path,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> AnalysisItem:
    codec = probe_video_codec(path, ffprobe)
    skip, skip_reason = is_already_compressed(path, ffprobe, codec=codec)
    size_bytes = path.stat().st_size
    duration_seconds = get_duration_seconds(path, ffprobe)
    bitrate_kbps = get_video_bitrate_kbps(path, ffprobe)
    estimated_output_bytes = (
        0
        if skip
        else estimate_output_size(
            path,
            ffprobe,
            codec=codec,
            crf=crf,
            preset=preset,
            use_calibration=use_calibration,
            calibration_store=calibration_store,
        )
    )
    estimated_savings_bytes = (
        max(size_bytes - estimated_output_bytes, 0) if estimated_output_bytes > 0 else 0
    )

    if skip:
        reason_code = "already_hevc" if codec == "hevc" else "already_marked_compressed"
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="skip",
            reason_code=reason_code,
            reason_text=skip_reason,
        )

    if estimated_output_bytes <= 0:
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=0,
            estimated_savings_bytes=0,
            recommendation="maybe",
            reason_code="estimate_unavailable",
            reason_text="could not estimate compressed size confidently",
        )

    savings_pct = estimated_savings_bytes / size_bytes * 100 if size_bytes else 0.0
    if estimated_savings_bytes < _MIN_SKIP_SAVINGS_BYTES or savings_pct < _MIN_SKIP_SAVINGS_PCT:
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="skip",
            reason_code="savings_too_small",
            reason_text="estimated savings are too small to justify recompression",
        )

    if codec in _RECOMMENDED_CODECS and (
        estimated_savings_bytes >= _MIN_RECOMMENDED_SAVINGS_BYTES
        and savings_pct >= _MIN_RECOMMENDED_SAVINGS_PCT
    ):
        return AnalysisItem(
            source=path,
            codec=codec,
            size_bytes=size_bytes,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            estimated_output_bytes=estimated_output_bytes,
            estimated_savings_bytes=estimated_savings_bytes,
            recommendation="recommended",
            reason_code="strong_savings_candidate",
            reason_text="legacy codec with strong projected space savings",
        )

    return AnalysisItem(
        source=path,
        codec=codec,
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        bitrate_kbps=bitrate_kbps,
        estimated_output_bytes=estimated_output_bytes,
        estimated_savings_bytes=estimated_savings_bytes,
        recommendation="maybe",
        reason_code="borderline_candidate",
        reason_text="projected savings may be worthwhile, but the case is not strong enough for auto-selection",
    )


def analyze_files(
    files: list[Path],
    ffprobe: Path,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
) -> list[AnalysisItem]:
    items: list[AnalysisItem] = []
    total = len(files)
    calibration_store = load_calibration_store() if use_calibration else None
    for index, path in enumerate(files, start=1):
        try:
            item = build_analysis_item(
                path,
                ffprobe,
                preset=preset,
                crf=crf,
                use_calibration=use_calibration,
                calibration_store=calibration_store,
            )
        except TypeError:
            # Some tests patch build_analysis_item with the older 2-arg signature.
            item = build_analysis_item(path, ffprobe)
        items.append(item)
        if progress_callback is not None:
            progress_callback(index, total, path)
    return items


def apply_duplicate_policy_to_items(
    items: list[AnalysisItem],
    *,
    policy: str = "prefer-mkv",
) -> tuple[list[AnalysisItem], list[str]]:
    if policy == "all":
        return items, []
    selected, warnings, deprioritized = apply_duplicate_title_policy(
        [item.source for item in items],
        policy=policy,
    )
    selected_set = {path for path in selected}
    suppressed = {path for paths in deprioritized.values() for path in paths}
    updated: list[AnalysisItem] = []
    for item in items:
        if item.source in selected_set or item.source not in suppressed:
            updated.append(item)
            continue
        updated.append(
            AnalysisItem(
                source=item.source,
                codec=item.codec,
                size_bytes=item.size_bytes,
                duration_seconds=item.duration_seconds,
                bitrate_kbps=item.bitrate_kbps,
                estimated_output_bytes=item.estimated_output_bytes,
                estimated_savings_bytes=item.estimated_savings_bytes,
                recommendation="skip",
                reason_code="duplicate_preferred_format"
                if policy == "prefer-mkv"
                else "duplicate_title_group",
                reason_text=(
                    "same title also exists as a preferred .mkv copy"
                    if policy == "prefer-mkv"
                    else "same title appears in multiple formats and was skipped as a group"
                ),
            )
        )
    return updated, warnings


def analyze_directory(
    directory: Path,
    recursive: bool,
    ffprobe: Path,
    *,
    preset: str = "fast",
    crf: int = 20,
    use_calibration: bool = True,
) -> list[AnalysisItem]:
    files = scan_directory(directory, recursive=recursive)
    return analyze_files(files, ffprobe, preset=preset, crf=crf, use_calibration=use_calibration)


def estimate_analysis_encode_seconds(
    items: list[AnalysisItem],
    preset: str,
    crf: int,
    ffmpeg: Path,
    known_speed: float | None = None,
    *,
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> float | None:
    recommended = [item for item in items if item.recommendation == "recommended"]
    if not recommended:
        return 0.0

    speed: float | None = known_speed if known_speed is not None and known_speed > 0 else None
    if use_calibration and recommended:
        active_store = (
            calibration_store if calibration_store is not None else load_calibration_store()
        )
        weighted_speeds: list[tuple[float, float]] = []
        adjustment_factors: list[tuple[float, float]] = []
        for item in recommended:
            resolution = (
                resolution_bucket(item.width, item.height)
                if getattr(item, "width", 0) and getattr(item, "height", 0)
                else "unknown"
            )
            bitrate = (
                bitrate_bucket(item.bitrate_kbps)
                if item.bitrate_kbps and item.bitrate_kbps > 0
                else "unknown"
            )
            lookup = lookup_estimate(
                active_store,
                codec=item.codec,
                resolution=resolution,
                bitrate=bitrate,
                preset=preset,
                container=item.source.suffix.lower() or ".mkv",
            )
            if lookup is None:
                continue
            weight = max(item.duration_seconds, 1.0)
            if preset not in _HW_ENCODERS and resolution != "unknown":
                weight *= 1.25
            if lookup.speed is not None and lookup.speed > 0:
                calibrated_speed = lookup.speed
                if lookup.average_speed_error is not None:
                    calibrated_speed = max(
                        0.05, calibrated_speed * (1.0 + lookup.average_speed_error)
                    )
                weighted_speeds.append((calibrated_speed, weight))
            if lookup.average_speed_error is not None:
                adjustment_factors.append((max(0.2, 1.0 + lookup.average_speed_error), weight))

        related_speed: float | None = None
        if weighted_speeds:
            total_weight = sum(weight for _, weight in weighted_speeds)
            related_speed = sum(value * weight for value, weight in weighted_speeds) / total_weight

        if speed is not None:
            if adjustment_factors:
                total_factor_weight = sum(weight for _, weight in adjustment_factors)
                average_factor = (
                    sum(value * weight for value, weight in adjustment_factors)
                    / total_factor_weight
                )
                speed = max(0.05, speed * average_factor)
            if related_speed is not None:
                speed = (
                    ((min(speed, related_speed) * 0.75) + (max(speed, related_speed) * 0.25))
                    if preset in _HW_ENCODERS
                    else ((min(speed, related_speed) * 0.70) + (max(speed, related_speed) * 0.30))
                )
        else:
            speed = related_speed
    if speed is None:
        from mediashrink.wizard import benchmark_encoder

        sample = recommended[0]
        if sample.duration_seconds <= 0:
            return None
        speed = benchmark_encoder(
            encoder_key=preset,
            sample_file=sample.source,
            sample_duration=sample.duration_seconds,
            crf=crf,
            ffmpeg=ffmpeg,
        )

    if speed is None or speed <= 0:
        return None

    total_media_seconds = sum(
        item.duration_seconds for item in recommended if item.duration_seconds > 0
    )
    return total_media_seconds / speed if total_media_seconds > 0 else None


def describe_estimate_calibration(
    items: list[AnalysisItem],
    *,
    preset: str,
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> str | None:
    if not use_calibration:
        return "calibration disabled"
    candidates = [item for item in items if item.recommendation == "recommended"] or list(items)
    if not candidates:
        return None
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    notes: list[str] = []
    seen: set[tuple[str | None, str]] = set()
    for item in candidates[:3]:
        key = (item.codec, item.source.suffix.lower() or ".mkv")
        if key in seen:
            continue
        seen.add(key)
        estimate = lookup_estimate(
            active_store,
            codec=item.codec,
            resolution="unknown",
            bitrate="unknown",
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        note = describe_calibration_estimate(estimate)
        if note:
            notes.append(note)
    if not notes:
        return "using heuristic estimates only"
    return notes[0]


def build_manifest(
    directory: Path,
    recursive: bool,
    preset: str,
    crf: int,
    profile_name: str | None,
    estimated_total_encode_seconds: float | None,
    estimate_confidence: str | None,
    *,
    size_confidence: str | None = None,
    size_confidence_detail: str | None = None,
    time_confidence: str | None = None,
    time_confidence_detail: str | None = None,
    duplicate_policy: str | None = None,
    recommended_only: bool = True,
    notes: list[str] | None = None,
    items: list[AnalysisItem],
) -> AnalysisManifest:
    return AnalysisManifest(
        version=MANIFEST_VERSION,
        analyzed_directory=directory,
        recursive=recursive,
        preset=preset,
        crf=crf,
        profile_name=profile_name,
        estimated_total_encode_seconds=estimated_total_encode_seconds,
        estimate_confidence=estimate_confidence,
        size_confidence=size_confidence,
        size_confidence_detail=size_confidence_detail,
        time_confidence=time_confidence,
        time_confidence_detail=time_confidence_detail,
        duplicate_policy=duplicate_policy,
        items=(
            [item for item in items if item.recommendation == "recommended"]
            if recommended_only
            else list(items)
        ),
        notes=notes,
    )


def estimate_analysis_confidence(
    items: list[AnalysisItem],
    *,
    benchmarked_files: int = 0,
) -> str:
    if not items:
        return "Low"

    total = len(items)
    known_durations = sum(1 for item in items if item.duration_seconds > 0)
    known_ratio = known_durations / total
    codec_count = len({item.codec or "unknown" for item in items})
    fallback_estimates = sum(
        1 for item in items if item.recommendation != "skip" and item.estimated_output_bytes <= 0
    )

    score = 0
    if benchmarked_files >= 1:
        score += 1
    if known_ratio >= 0.9:
        score += 2
    elif known_ratio >= 0.6:
        score += 1
    else:
        score -= 1
    if codec_count <= 2:
        score += 1
    elif codec_count >= 4:
        score -= 1
    if fallback_estimates == 0:
        score += 1
    elif fallback_estimates >= max(1, total // 3):
        score -= 1

    # Single-file batches offer too little evidence for "High" — cap at Medium.
    # This prevents post-split confidence inflation when the batch shrinks to one file.
    if total <= 1:
        score = min(score, 3)

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def estimate_size_confidence(
    items: list[AnalysisItem],
    *,
    preset: str = "fast",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> str:
    candidates = _candidate_items(items)
    if not candidates:
        return "Low"
    known_estimates = sum(1 for item in candidates if item.estimated_output_bytes > 0)
    known_ratio = known_estimates / len(candidates)
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    calibration_hits = 0
    for item in candidates[:5]:
        estimate = lookup_estimate(
            active_store if use_calibration else None,
            codec=item.codec,
            resolution="unknown",
            bitrate="unknown",
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if estimate is not None and estimate.output_ratio is not None:
            calibration_hits += 1
    score = 0
    if known_ratio >= 0.95:
        score += 2
    elif known_ratio >= 0.70:
        score += 1
    else:
        score -= 1
    if calibration_hits >= min(3, len(candidates)):
        score += 2
    elif calibration_hits >= 1:
        score += 1

    average_size_error = _average_size_error_for_items(
        candidates,
        preset=preset,
        use_calibration=use_calibration,
        calibration_store=active_store,
    )
    if average_size_error is not None:
        if abs(average_size_error) >= _VERY_HIGH_SIZE_ERROR_THRESHOLD:
            score -= 2
        elif abs(average_size_error) >= _HIGH_SIZE_ERROR_THRESHOLD:
            score -= 1

    # Single-file batches have too little evidence to justify "High" size confidence.
    if len(candidates) <= 1:
        score = min(score, 2)
    if (
        average_size_error is not None
        and abs(average_size_error) >= _VERY_HIGH_SIZE_ERROR_THRESHOLD
    ):
        score = min(score, 1)
    elif average_size_error is not None and abs(average_size_error) >= _HIGH_SIZE_ERROR_THRESHOLD:
        score = min(score, 2)

    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def estimate_time_confidence(
    items: list[AnalysisItem],
    *,
    benchmarked_files: int = 0,
    preset: str = "fast",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> str:
    base = estimate_analysis_confidence(items, benchmarked_files=benchmarked_files)
    candidates = _candidate_items(items)
    if not candidates:
        return "Low"
    if benchmarked_files <= 0 and base == "High":
        base = "Medium"
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    speed_hits = 0
    for item in candidates[:5]:
        estimate = lookup_estimate(
            active_store if use_calibration else None,
            codec=item.codec,
            resolution="unknown",
            bitrate="unknown",
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if estimate is not None and estimate.speed is not None and estimate.speed > 0:
            speed_hits += 1
    # Single-file batches: cap time confidence at Medium regardless of other signals.
    if len(candidates) <= 1 and base == "High":
        base = "Medium"

    if base == "High" or (base == "Medium" and speed_hits >= 2):
        return "High"
    if base == "Low" and speed_hits == 0:
        return "Low"
    return "Medium"


def describe_estimate_confidence(
    items: list[AnalysisItem],
    *,
    benchmarked_files: int = 0,
) -> str:
    if not items:
        return "No estimate inputs were available."

    total = len(items)
    known_durations = sum(1 for item in items if item.duration_seconds > 0)
    codec_count = len({item.codec or "unknown" for item in items})
    fallback_estimates = sum(
        1 for item in items if item.recommendation != "skip" and item.estimated_output_bytes <= 0
    )
    benchmark_note = (
        f"{benchmarked_files} benchmark sample{'s' if benchmarked_files != 1 else ''}"
        if benchmarked_files > 0
        else "no encode benchmark sample"
    )
    detail = (
        f"{benchmark_note}, {known_durations}/{total} file durations known, "
        f"{codec_count} codec group{'s' if codec_count != 1 else ''}"
    )
    if fallback_estimates:
        detail += (
            f", {fallback_estimates} estimate fallback{'s' if fallback_estimates != 1 else ''}"
        )
    return detail


def describe_size_confidence(
    items: list[AnalysisItem],
    *,
    preset: str = "fast",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> str:
    candidates = _candidate_items(items)
    if not candidates:
        return "no encodable files"
    known_estimates = sum(1 for item in candidates if item.estimated_output_bytes > 0)
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    calibration_note = describe_estimate_calibration(
        candidates,
        preset=preset,
        use_calibration=use_calibration,
        calibration_store=active_store,
    )
    detail = f"{known_estimates}/{len(candidates)} output size estimates available"
    if calibration_note:
        detail += f"; local history: {calibration_note}"
    summary = summarize_calibration_store(active_store if use_calibration else None)
    history_slices = describe_history_slices(
        active_store if use_calibration else None,
        preset=preset,
        containers={item.source.suffix.lower() or ".mkv" for item in candidates},
    )
    if history_slices.get("closest_preset_history"):
        detail += f"; closest preset history: {history_slices['closest_preset_history']}"
    if history_slices.get("container_mix_history"):
        detail += f"; current container mix: {history_slices['container_mix_history']}"
    if history_slices.get("overall_history"):
        detail += f"; overall machine history: {history_slices['overall_history']}"
    family_summary = format_family_container_summary(
        summary.get("family_container_summaries") if isinstance(summary, dict) else None
    )
    if family_summary and not history_slices.get("overall_history"):
        detail += f"; history mix: {family_summary}"
    return detail


def describe_time_confidence(
    items: list[AnalysisItem],
    *,
    benchmarked_files: int = 0,
    preset: str = "fast",
    use_calibration: bool = True,
    calibration_store: dict[str, object] | None = None,
) -> str:
    detail = describe_estimate_confidence(items, benchmarked_files=benchmarked_files)
    active_store = calibration_store if calibration_store is not None else load_calibration_store()
    candidates = _candidate_items(items)
    speed_notes = 0
    for item in candidates[:5]:
        estimate = lookup_estimate(
            active_store if use_calibration else None,
            codec=item.codec,
            resolution="unknown",
            bitrate="unknown",
            preset=preset,
            container=item.source.suffix.lower() or ".mkv",
        )
        if estimate is not None and estimate.speed is not None:
            speed_notes += 1
    if speed_notes:
        detail += (
            f"; local speed matches for {speed_notes}/{min(len(candidates), 5)} sample file(s)"
        )
    summary = summarize_calibration_store(active_store if use_calibration else None)
    if isinstance(summary, dict):
        history_slices = describe_history_slices(
            active_store if use_calibration else None,
            preset=preset,
            containers={item.source.suffix.lower() or ".mkv" for item in candidates},
        )
        if history_slices.get("closest_preset_history"):
            detail += f"; closest preset history: {history_slices['closest_preset_history']}"
        if history_slices.get("container_mix_history"):
            detail += f"; current container mix: {history_slices['container_mix_history']}"
        if history_slices.get("overall_history"):
            detail += f"; overall machine history: {history_slices['overall_history']}"
        family_summary = format_family_container_summary(summary.get("family_container_summaries"))
        if family_summary and not history_slices.get("overall_history"):
            detail += f"; history mix: {family_summary}"
        bias_summary = summary.get("bias_summary")
        if isinstance(bias_summary, dict) and isinstance(bias_summary.get("summary"), str):
            detail += f"; {bias_summary['summary']}"
    return detail


def select_representative_items(items: list[AnalysisItem], limit: int = 3) -> list[AnalysisItem]:
    if limit <= 0 or not items:
        return []
    remaining = sorted(
        items,
        key=lambda item: (
            item.duration_seconds if item.duration_seconds > 0 else item.size_bytes / max(_GB, 1),
            item.size_bytes,
        ),
        reverse=True,
    )
    selected: list[AnalysisItem] = []
    seen_clusters: set[tuple[str, str, str]] = set()

    def cluster_key(item: AnalysisItem) -> tuple[str, str, str]:
        container = item.source.suffix.lower() or ".mkv"
        risk = "mkv-risk" if container in {".mp4", ".m4v"} else "normal"
        if (item.codec or "") in {"vc1", "mpeg2video"}:
            risk = "legacy"
        return (item.codec or "unknown", container, risk)

    priority_selectors = (
        lambda item: (item.codec or "") in {"vc1", "mpeg2video"},
        lambda item: (item.codec or "") == "h264",
        lambda item: item.recommendation == "maybe",
    )
    for selector in priority_selectors:
        for item in remaining:
            if selector(item) and item not in selected:
                selected.append(item)
                seen_clusters.add(cluster_key(item))
                break
        if len(selected) >= limit:
            return selected[:limit]

    for item in remaining:
        key = cluster_key(item)
        if key in seen_clusters:
            continue
        seen_clusters.add(key)
        selected.append(item)
        if len(selected) >= limit:
            return selected

    for item in remaining:
        if item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected[:limit]


def summarize_container_risks(
    items: list[AnalysisItem],
    ffprobe: Path,
    *,
    limit: int = 4,
) -> tuple[int, dict[str, int], list[str]]:
    risky: list[tuple[AnalysisItem, list[str]]] = []
    for item in _candidate_items(items):
        constraints = describe_output_container_constraints(item.source, item.source, ffprobe)
        relevant = [
            constraint
            for constraint in constraints
            if "mp4" in constraint.lower()
            or "subtitle" in constraint.lower()
            or "audio codec" in constraint.lower()
            or "attachment" in constraint.lower()
            or "data stream" in constraint.lower()
        ]
        if relevant:
            risky.append((item, relevant))
    grouped: dict[str, int] = {}
    examples: list[str] = []
    for item, reasons in risky[:limit]:
        if len(examples) < 3:
            examples.append(item.source.name)
        for reason in reasons:
            grouped[reason] = grouped.get(reason, 0) + 1
    return len(risky), grouped, examples


def collect_container_risk_signals(
    items: list[AnalysisItem],
    ffprobe: Path,
) -> dict[Path, str]:
    signals: dict[Path, str] = {}
    for item in _candidate_items(items):
        constraints = describe_output_container_constraints(item.source, item.source, ffprobe)
        relevant = [
            constraint.lower()
            for constraint in constraints
            if "mp4" in constraint.lower()
            or "subtitle" in constraint.lower()
            or "audio codec" in constraint.lower()
            or "attachment" in constraint.lower()
            or "data stream" in constraint.lower()
        ]
        if not relevant:
            continue
        if any(
            marker in reason
            for reason in relevant
            for marker in ("subtitle", "attachment", "audio codec", "data stream")
        ):
            signals[item.source] = "MKV first"
        else:
            signals[item.source] = "Follow-up"
    return signals


def estimate_value_range(
    value: float,
    *,
    confidence: str | None,
    average_error: float | None = None,
    widen_by: float = 0.0,
) -> tuple[float, float]:
    uncertainty = estimate_display_uncertainty(
        confidence,
        average_error=average_error,
        widen_by=widen_by,
    )
    low = max(0.0, value * (1.0 - uncertainty))
    high = max(low, value * (1.0 + uncertainty))
    return low, high


def save_manifest(manifest: AnalysisManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> AnalysisManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")
    manifest = AnalysisManifest.from_dict(raw)
    if manifest.version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.version}")
    return manifest


def display_analysis_summary(
    items: list[AnalysisItem],
    estimated_total_encode_seconds: float | None,
    console: Console,
    estimate_confidence: str | None = None,
    estimate_confidence_detail: str | None = None,
    size_confidence: str | None = None,
    size_confidence_detail: str | None = None,
    time_confidence: str | None = None,
    time_confidence_detail: str | None = None,
    notes: list[str] | None = None,
    compatibility_signals: dict[Path, str] | None = None,
    calibration_store: dict[str, object] | None = None,
    plain_output: bool = False,
) -> None:
    recommended = [item for item in items if item.recommendation == "recommended"]
    maybe = [item for item in items if item.recommendation == "maybe"]
    skipped = [item for item in items if item.recommendation == "skip"]
    narrow = plain_output or console.width < 120

    if plain_output:
        console.print()
        console.print("[bold cyan]Compression analysis[/bold cyan]")
        _TABLE_LIMIT = 12
        sorted_items = sorted(
            items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True
        )
        for item in sorted_items[:_TABLE_LIMIT]:
            savings_text = (
                "-"
                if item.estimated_savings_bytes <= 0
                else f"~{_fmt_size(item.estimated_savings_bytes)}"
            )
            batch_fit = (compatibility_signals or {}).get(item.source, "Now")
            console.print(
                f"- {item.source.name} | {item.codec or '?'} | {_fmt_size(item.size_bytes)}"
                f" | {savings_text} | {item.recommendation.upper()} | {batch_fit} | {item.reason_text}",
                highlight=False,
            )
        console.print()
    else:
        table = Table(title="Compression analysis", header_style="bold cyan", expand=True)
        table.add_column("File")
        table.add_column("Codec", justify="center", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Est. Saving", justify="right", no_wrap=True)
        table.add_column("Recommendation", justify="center", no_wrap=True)
        table.add_column("Batch Fit", justify="center", no_wrap=True)
        if not narrow:
            table.add_column("Reason")

        _TABLE_LIMIT = 12
        sorted_items = sorted(
            items, key=lambda candidate: candidate.estimated_savings_bytes, reverse=True
        )
        for item in sorted_items[:_TABLE_LIMIT]:
            savings_text = (
                "-"
                if item.estimated_savings_bytes <= 0
                else f"~{_fmt_size(item.estimated_savings_bytes)}"
            )
            row = [
                item.source.name,
                item.codec or "?",
                _fmt_size(item.size_bytes),
                savings_text,
                item.recommendation.upper(),
                (compatibility_signals or {}).get(item.source, "Now"),
            ]
            if not narrow:
                row.append(item.reason_text)
            table.add_row(*row)

    total_current = sum(item.size_bytes for item in recommended)
    total_estimated = sum(item.estimated_output_bytes for item in recommended)
    total_saved = sum(item.estimated_savings_bytes for item in recommended)
    total_saved_pct = total_saved / total_current * 100 if total_current else 0.0
    maybe_total = sum(item.size_bytes for item in maybe)
    skipped_total = sum(item.size_bytes for item in skipped)

    if not plain_output:
        console.print()
        console.print(table)
    if len(sorted_items) > _TABLE_LIMIT:
        hidden = len(sorted_items) - _TABLE_LIMIT
        console.print(
            f"[dim]Showing top {_TABLE_LIMIT} of {len(sorted_items)} files by estimated saving. "
            f"{hidden} more not shown — use --manifest-out to export the full list.[/dim]",
            highlight=False,
        )
    candidate_groups: dict[tuple[str, str], int] = {}
    for item in recommended + maybe:
        key = (item.source.suffix.lower() or "?", item.codec or "?")
        candidate_groups[key] = candidate_groups.get(key, 0) + 1
    if candidate_groups:
        top_groups = []
        for (container, codec), count in sorted(
            candidate_groups.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]:
            top_groups.append(f"{count} {container} {codec}")
        console.print(
            f"[dim]Candidate mix: {', '.join(top_groups)}.[/dim]",
            highlight=False,
        )
    if narrow and not plain_output:
        console.print(
            "[dim]Compact analysis view hides the longer reason column on narrow terminals.[/dim]"
        )
    console.print(
        f"[bold]{len(items)}[/bold] file(s) scanned - "
        f"[green bold]{len(recommended)}[/green bold] recommended, "
        f"[yellow]{len(maybe)}[/yellow] maybe, "
        f"[dim]{len(skipped)}[/dim] skip",
        highlight=False,
    )
    console.print(
        f"Rollup: "
        f"[green]recommended {len(recommended)} / {_fmt_size(total_current)}[/green], "
        f"[yellow]maybe {len(maybe)} / {_fmt_size(maybe_total)}[/yellow], "
        f"[dim]skip {len(skipped)} / {_fmt_size(skipped_total)}[/dim]",
        highlight=False,
    )
    confidence = estimate_confidence or estimate_analysis_confidence(items)
    resolved_size_confidence = size_confidence or confidence
    resolved_time_confidence = time_confidence or confidence
    if recommended:
        range_low, range_high = estimate_value_range(
            float(total_estimated),
            confidence=resolved_size_confidence,
        )
        saved_low = max(total_current - int(range_high), 0)
        saved_high = max(total_current - int(range_low), 0)
        saved_pct_low = saved_low / total_current * 100 if total_current else 0.0
        saved_pct_high = saved_high / total_current * 100 if total_current else 0.0
        console.print(
            f"Recommended set: [yellow]{_fmt_size(total_current)}[/yellow] -> "
            f"[green]~{_fmt_size(int(range_low))}-{_fmt_size(int(range_high))}[/green] "
            f"([bold green]~{_fmt_size(saved_low)}-{_fmt_size(saved_high)} saved, "
            f"~{saved_pct_low:.0f}-{saved_pct_high:.0f}%[/bold green])",
            highlight=False,
        )
    if estimated_total_encode_seconds is not None and estimated_total_encode_seconds > 0:
        time_low, time_high = estimate_value_range(
            estimated_total_encode_seconds,
            confidence=time_confidence or estimate_confidence,
        )
        console.print(
            f"Rough encode time: [cyan]~{_fmt_duration(time_low)}-{_fmt_duration(time_high)}[/cyan]"
        )
    risky_now = sum(
        1 for item in recommended + maybe if (compatibility_signals or {}).get(item.source)
    )
    if risky_now:
        console.print(
            f"Batch-ready now: [green]{len(recommended) + len(maybe) - risky_now}[/green], "
            f"[yellow]{risky_now} likely need MKV-first routing or follow-up[/yellow]",
            highlight=False,
        )
    console.print(
        f"Size confidence: [cyan]{resolved_size_confidence}[/cyan]"
        + (f" [dim]({size_confidence_detail})[/dim]" if size_confidence_detail else "")
    )
    console.print(
        f"Time confidence: [cyan]{resolved_time_confidence}[/cyan]"
        + (
            f" [dim]({time_confidence_detail or estimate_confidence_detail})[/dim]"
            if (time_confidence_detail or estimate_confidence_detail)
            else ""
        )
    )
    if notes:
        for note in notes:
            console.print(f"[dim yellow]{note}[/dim yellow]")
    bias_summary = recent_bias_summary(calibration_store)
    if isinstance(bias_summary, dict) and isinstance(bias_summary.get("summary"), str):
        console.print(f"[dim]Estimate bias: {bias_summary['summary']}.[/dim]")
    console.print("[dim]Analysis estimates are approximate.[/dim]")
    console.print(
        "[dim]Hardware encoders are faster, but source duration, bitrate, and resolution still dominate total runtime.[/dim]"
    )
    console.print()
