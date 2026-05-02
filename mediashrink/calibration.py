from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


_STORE_VERSION = 2
_MAX_SUCCESS_RECORDS = 2000
_MAX_FAILURE_RECORDS = 1000
_MAX_BATCH_BIAS_RECORDS = 500


@dataclass
class CalibrationRecord:
    codec: str
    container: str
    resolution_bucket: str
    bitrate_bucket: str
    preset: str
    preset_family: str
    crf: int
    input_bytes: int
    output_bytes: int
    duration_seconds: float
    wall_seconds: float
    effective_speed: float
    fallback_used: bool
    retry_used: bool
    codec_family: str = "unknown"
    predicted_output_ratio: float | None = None
    predicted_speed: float | None = None
    accepted_output: bool = True
    safety_rejection_reason: str | None = None


@dataclass
class FailureRecord:
    encoder: str
    container: str
    stage: str
    reason: str


@dataclass
class BatchBiasRecord:
    codec_family: str
    container: str
    resolution_bucket: str
    bitrate_bucket: str
    preset: str
    preset_family: str
    average_size_error: float
    sample_count: int


@dataclass
class CalibrationEstimate:
    output_ratio: float | None
    speed: float | None
    failure_rate: float
    confidence: str
    exact_matches: int
    loose_matches: int
    weighted_samples: float
    source: str
    average_size_error: float | None = None
    average_speed_error: float | None = None
    batch_bias_adjustment: float | None = None


def get_calibration_path() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "mediashrink" / "calibration.json"

    xdg_config = os.getenv("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "mediashrink" / "calibration.json"

    return Path.home() / ".config" / "mediashrink" / "calibration.json"


def _empty_store() -> dict[str, object]:
    return {"version": _STORE_VERSION, "records": [], "failures": [], "batch_biases": []}


def load_calibration_store(path: Path | None = None) -> dict[str, object]:
    calibration_path = path or get_calibration_path()
    if not calibration_path.exists():
        return _empty_store()
    try:
        raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(raw, dict):
        return _empty_store()
    records = raw.get("records", [])
    failures = raw.get("failures", [])
    batch_biases = raw.get("batch_biases", [])
    if not isinstance(records, list):
        records = []
    if not isinstance(failures, list):
        failures = []
    if not isinstance(batch_biases, list):
        batch_biases = []
    return {
        "version": int(raw.get("version", _STORE_VERSION)),
        "records": [item for item in records if isinstance(item, dict)],
        "failures": [item for item in failures if isinstance(item, dict)],
        "batch_biases": [item for item in batch_biases if isinstance(item, dict)],
    }


def save_calibration_store(store: dict[str, object], path: Path | None = None) -> Path:
    calibration_path = path or get_calibration_path()
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    return calibration_path


def _trim_store(store: dict[str, object]) -> None:
    records = store.get("records", [])
    failures = store.get("failures", [])
    batch_biases = store.get("batch_biases", [])
    if isinstance(records, list) and len(records) > _MAX_SUCCESS_RECORDS:
        store["records"] = records[-_MAX_SUCCESS_RECORDS:]
    if isinstance(failures, list) and len(failures) > _MAX_FAILURE_RECORDS:
        store["failures"] = failures[-_MAX_FAILURE_RECORDS:]
    if isinstance(batch_biases, list) and len(batch_biases) > _MAX_BATCH_BIAS_RECORDS:
        store["batch_biases"] = batch_biases[-_MAX_BATCH_BIAS_RECORDS:]


def append_success_record(
    record: CalibrationRecord,
    *,
    store: dict[str, object] | None = None,
    path: Path | None = None,
) -> Path:
    active_store = store if store is not None else load_calibration_store(path)
    records = active_store.setdefault("records", [])
    if not isinstance(records, list):
        records = []
        active_store["records"] = records
    records.append(asdict(record))
    _trim_store(active_store)
    try:
        return save_calibration_store(active_store, path)
    except OSError:
        return path or get_calibration_path()


def append_failure_record(
    record: FailureRecord,
    *,
    store: dict[str, object] | None = None,
    path: Path | None = None,
) -> Path:
    active_store = store if store is not None else load_calibration_store(path)
    failures = active_store.setdefault("failures", [])
    if not isinstance(failures, list):
        failures = []
        active_store["failures"] = failures
    failures.append(asdict(record))
    _trim_store(active_store)
    try:
        return save_calibration_store(active_store, path)
    except OSError:
        return path or get_calibration_path()


def append_batch_bias_record(
    record: BatchBiasRecord,
    *,
    store: dict[str, object] | None = None,
    path: Path | None = None,
) -> Path:
    active_store = store if store is not None else load_calibration_store(path)
    batch_biases = active_store.setdefault("batch_biases", [])
    if not isinstance(batch_biases, list):
        batch_biases = []
        active_store["batch_biases"] = batch_biases
    batch_biases.append(asdict(record))
    _trim_store(active_store)
    try:
        return save_calibration_store(active_store, path)
    except OSError:
        return path or get_calibration_path()


def resolution_bucket(width: int, height: int) -> str:
    larger = max(width, height)
    if larger >= 3000:
        return "2160p+"
    if larger >= 1800:
        return "1440p"
    if larger >= 1200:
        return "1080p"
    if larger >= 700:
        return "720p"
    if larger > 0:
        return "sd"
    return "unknown"


def bitrate_bucket(bitrate_kbps: float) -> str:
    if bitrate_kbps >= 16000:
        return "very_high"
    if bitrate_kbps >= 8000:
        return "high"
    if bitrate_kbps >= 3000:
        return "medium"
    if bitrate_kbps > 0:
        return "low"
    return "unknown"


def preset_family(preset: str) -> str:
    if preset in {"qsv", "nvenc", "amf"}:
        return "hardware"
    if preset in {"ultrafast", "faster", "fast", "medium", "slow"}:
        return "software"
    return preset


def codec_family(codec: str | None) -> str:
    normalized = (codec or "unknown").lower()
    if normalized in {"h264", "avc", "mpeg4"}:
        return "avc-family"
    if normalized in {"hevc", "h265"}:
        return "hevc-family"
    if normalized in {"mpeg2video", "mpeg1video", "vc1"}:
        return "legacy-broadcast"
    if normalized in {"vp8", "vp9", "av1"}:
        return "modern-web"
    return normalized or "unknown"


def _matches(
    raw: dict[str, object],
    *,
    codec: str | None,
    codec_family_name: str,
    resolution: str,
    bitrate: str,
    preset: str,
    container: str,
) -> tuple[bool, bool]:
    raw_codec = raw.get("codec")
    raw_codec_family = raw.get("codec_family")
    raw_resolution = raw.get("resolution_bucket")
    raw_bitrate = raw.get("bitrate_bucket")
    raw_preset = raw.get("preset")
    raw_family = raw.get("preset_family")
    raw_container = raw.get("container")
    codec_match = codec in {None, "unknown"} or raw_codec == codec
    family_match = (
        codec_family_name == "unknown"
        or raw_codec_family in {None, "", "unknown"}
        or raw_codec_family == codec_family_name
    )
    resolution_match = resolution == "unknown" or raw_resolution == resolution
    bitrate_match = bitrate == "unknown" or raw_bitrate == bitrate
    container_match = container == "unknown" or raw_container == container
    preset_match = raw_preset == preset or raw_family == preset_family(preset)
    exact = (
        codec_match
        and family_match
        and resolution_match
        and bitrate_match
        and container_match
        and preset_match
    )
    loose = family_match and resolution_match and preset_match
    return exact, loose


def lookup_estimate(
    store: dict[str, object] | None,
    *,
    codec: str | None,
    resolution: str,
    bitrate: str,
    preset: str,
    container: str,
) -> CalibrationEstimate | None:
    if not store:
        return None
    raw_records = store.get("records", [])
    if not isinstance(raw_records, list) or not raw_records:
        return None

    family_name = codec_family(codec)
    exact_matches: list[dict[str, object]] = []
    loose_matches: list[dict[str, object]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        exact, loose = _matches(
            raw,
            codec=codec,
            codec_family_name=family_name,
            resolution=resolution,
            bitrate=bitrate,
            preset=preset,
            container=container,
        )
        if exact:
            exact_matches.append(raw)
        elif loose:
            loose_matches.append(raw)

    batch_bias_adjustment = _lookup_batch_bias_adjustment(
        store,
        codec_family_name=family_name,
        resolution=resolution,
        bitrate=bitrate,
        preset=preset,
        container=container,
    )

    if not exact_matches and not loose_matches and batch_bias_adjustment is None:
        return None

    weighted_output_total = 0.0
    weighted_output_samples = 0.0
    weighted_speed_total = 0.0
    weighted_speed_samples = 0.0
    weighted_samples = 0.0

    total_candidates = len(exact_matches) + len(loose_matches)

    def _recency_weight(position: int) -> float:
        if total_candidates <= 1:
            return 1.0
        return 0.75 + (position / max(total_candidates - 1, 1)) * 0.5

    ordered_matches = exact_matches + loose_matches
    positions = {id(raw): index for index, raw in enumerate(ordered_matches)}

    for raw in exact_matches:
        weight = 1.0 * _recency_weight(positions[id(raw)])
        if int(raw.get("input_bytes", 0)) > 0 and int(raw.get("output_bytes", 0)) > 0:
            weighted_output_total += (
                float(raw["output_bytes"]) / max(int(raw["input_bytes"]), 1)
            ) * weight
            weighted_output_samples += weight
        if float(raw.get("effective_speed", 0.0)) > 0:
            weighted_speed_total += float(raw["effective_speed"]) * weight
            weighted_speed_samples += weight
        weighted_samples += weight

    for raw in loose_matches:
        weight = 0.35 * _recency_weight(positions[id(raw)])
        if int(raw.get("input_bytes", 0)) > 0 and int(raw.get("output_bytes", 0)) > 0:
            weighted_output_total += (
                float(raw["output_bytes"]) / max(int(raw["input_bytes"]), 1)
            ) * weight
            weighted_output_samples += weight
        if float(raw.get("effective_speed", 0.0)) > 0:
            weighted_speed_total += float(raw["effective_speed"]) * weight
            weighted_speed_samples += weight
        weighted_samples += weight

    failure_rate = estimate_failure_rate(store, preset=preset, container=container)
    exact_count = len(exact_matches)
    loose_count = len(loose_matches)
    confidence = "Low"
    if exact_count >= 2 and weighted_samples >= 3.0:
        confidence = "High"
    elif exact_count >= 1 or weighted_samples >= 1.5:
        confidence = "Medium"

    return CalibrationEstimate(
        output_ratio=(
            weighted_output_total / weighted_output_samples if weighted_output_samples > 0 else None
        ),
        speed=(
            weighted_speed_total / weighted_speed_samples if weighted_speed_samples > 0 else None
        ),
        failure_rate=failure_rate,
        confidence=confidence,
        exact_matches=exact_count,
        loose_matches=loose_count,
        weighted_samples=weighted_samples,
        source="exact" if exact_count > 0 else "related",
        average_size_error=_average_prediction_error(
            exact_matches + loose_matches, "predicted_output_ratio", "input_bytes", "output_bytes"
        ),
        average_speed_error=_average_speed_error(exact_matches + loose_matches),
        batch_bias_adjustment=batch_bias_adjustment,
    )


def _lookup_batch_bias_adjustment(
    store: dict[str, object] | None,
    *,
    codec_family_name: str,
    resolution: str,
    bitrate: str,
    preset: str,
    container: str,
) -> float | None:
    if not store:
        return None
    raw_biases = store.get("batch_biases", [])
    if not isinstance(raw_biases, list) or not raw_biases:
        return None
    matches: list[float] = []
    preset_family_name = preset_family(preset)
    for raw in raw_biases:
        if not isinstance(raw, dict):
            continue
        if raw.get("codec_family") != codec_family_name:
            continue
        if resolution != "unknown" and raw.get("resolution_bucket") != resolution:
            continue
        if bitrate != "unknown" and raw.get("bitrate_bucket") != bitrate:
            continue
        if container != "unknown" and raw.get("container") != container:
            continue
        if raw.get("preset") != preset and raw.get("preset_family") != preset_family_name:
            continue
        adjustment = raw.get("average_size_error")
        if isinstance(adjustment, (int, float)):
            matches.append(float(adjustment))
    if not matches:
        return None
    return sum(matches) / len(matches)


def estimate_failure_rate(
    store: dict[str, object] | None,
    *,
    preset: str,
    container: str,
) -> float:
    if not store:
        return 0.0
    successes = 0
    failures = 0
    raw_records = store.get("records", [])
    raw_failures = store.get("failures", [])
    if isinstance(raw_records, list):
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            if raw.get("container") == container and (
                raw.get("preset") == preset or raw.get("preset_family") == preset_family(preset)
            ):
                if raw.get("accepted_output", True):
                    successes += 1
                else:
                    failures += 1
    if isinstance(raw_failures, list):
        for raw in raw_failures:
            if not isinstance(raw, dict):
                continue
            if raw.get("container") == container and (
                raw.get("encoder") == preset or raw.get("encoder") == preset_family(preset)
            ):
                failures += 1
    total = successes + failures
    if total <= 0:
        return 0.0
    return failures / total


def describe_calibration_estimate(estimate: CalibrationEstimate | None) -> str | None:
    if estimate is None:
        return None
    sample_parts: list[str] = []
    if estimate.exact_matches:
        sample_parts.append(
            f"{estimate.exact_matches} close local match{'es' if estimate.exact_matches != 1 else ''}"
        )
    if estimate.loose_matches:
        sample_parts.append(
            f"{estimate.loose_matches} related match{'es' if estimate.loose_matches != 1 else ''}"
        )
    if not sample_parts:
        sample_parts.append("no local matches")
    note = ", ".join(sample_parts)
    if estimate.failure_rate > 0:
        note += f", {estimate.failure_rate * 100:.0f}% failure history"
    if estimate.average_size_error is not None and abs(estimate.average_size_error) >= 0.08:
        note += ", size estimates tend to " + (
            "underestimate output size"
            if estimate.average_size_error > 0
            else "overestimate output size"
        )
    return note


def recent_bias_summary(
    store: dict[str, object] | None,
    *,
    codec: str | None = None,
    resolution: str | None = None,
    bitrate: str | None = None,
    preset: str | None = None,
    container: str | None = None,
) -> dict[str, object] | None:
    if not store:
        return None
    records = store.get("records", [])
    if not isinstance(records, list):
        return None
    filtered = [raw for raw in records if isinstance(raw, dict)]
    if any(value is not None for value in (codec, resolution, bitrate, preset, container)):
        family_name = codec_family(codec)
        filtered = [
            raw
            for raw in filtered
            if (
                codec is None or raw.get("codec") == codec or raw.get("codec_family") == family_name
            )
            and (resolution is None or raw.get("resolution_bucket") == resolution)
            and (bitrate is None or raw.get("bitrate_bucket") == bitrate)
            and (
                preset is None
                or raw.get("preset") == preset
                or raw.get("preset_family") == preset_family(preset)
            )
            and (container is None or raw.get("container") == container)
        ]
    return _recent_bias_summary(filtered)


def estimate_display_uncertainty(
    confidence: str | None,
    *,
    average_error: float | None = None,
    widen_by: float = 0.0,
) -> float:
    base = {
        "High": 0.10,
        "Medium": 0.18,
        "Low": 0.30,
    }.get(confidence or "", 0.22)
    if average_error is not None:
        base = max(base, min(abs(average_error) * 1.35, 0.45))
    return min(max(base + widen_by, 0.06), 0.55)


def _accepted_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [raw for raw in records if bool(raw.get("accepted_output", True))]


def _recent_bias_summary(records: list[dict[str, object]]) -> dict[str, object] | None:
    accepted = _accepted_records(records)[-25:]
    if not accepted:
        return None

    size_errors: list[float] = []
    speed_errors: list[float] = []
    for raw in accepted:
        predicted_ratio = raw.get("predicted_output_ratio")
        input_bytes = raw.get("input_bytes")
        output_bytes = raw.get("output_bytes")
        if (
            isinstance(predicted_ratio, (int, float))
            and predicted_ratio > 0
            and isinstance(input_bytes, (int, float))
            and input_bytes > 0
            and isinstance(output_bytes, (int, float))
            and output_bytes >= 0
        ):
            actual_ratio = float(output_bytes) / max(float(input_bytes), 1.0)
            size_errors.append(actual_ratio - float(predicted_ratio))

        predicted_speed = raw.get("predicted_speed")
        actual_speed = raw.get("effective_speed")
        if (
            isinstance(predicted_speed, (int, float))
            and predicted_speed > 0
            and isinstance(actual_speed, (int, float))
            and actual_speed > 0
        ):
            speed_errors.append(
                (float(actual_speed) - float(predicted_speed)) / float(predicted_speed)
            )

    size_bias = None
    size_text = None
    if size_errors:
        average_size_error = sum(size_errors) / len(size_errors)
        if average_size_error >= 0.08:
            size_bias = "larger_than_estimated"
            size_text = "recent runs have usually saved less space than forecast"
        elif average_size_error <= -0.08:
            size_bias = "smaller_than_estimated"
            size_text = "recent runs have usually saved more space than forecast"

    speed_bias = None
    speed_text = None
    if speed_errors:
        average_speed_error = sum(speed_errors) / len(speed_errors)
        if average_speed_error >= 0.15:
            speed_bias = "faster_than_estimated"
            speed_text = "recent encodes have usually finished faster than forecast"
        elif average_speed_error <= -0.15:
            speed_bias = "slower_than_estimated"
            speed_text = "recent encodes have usually finished slower than forecast"

    if not size_text and not speed_text:
        return None

    summary_parts = [part for part in (size_text, speed_text) if part]
    return {
        "size_bias": size_bias,
        "speed_bias": speed_bias,
        "summary": "; ".join(summary_parts),
        "samples": len(accepted),
    }


def _family_container_summaries(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for raw in records:
        preset_family_name = str(raw.get("preset_family") or "unknown")
        container = str(raw.get("container") or "unknown")
        entry = grouped.setdefault(
            (preset_family_name, container),
            {
                "preset_family": preset_family_name,
                "container": container,
                "samples": 0,
                "accepted_samples": 0,
                "rejected_samples": 0,
            },
        )
        entry["samples"] = int(entry["samples"]) + 1
        if raw.get("accepted_output", True):
            entry["accepted_samples"] = int(entry["accepted_samples"]) + 1
        else:
            entry["rejected_samples"] = int(entry["rejected_samples"]) + 1

    summaries = list(grouped.values())
    summaries.sort(
        key=lambda item: (
            -int(item["samples"]),
            str(item["preset_family"]),
            str(item["container"]),
        )
    )
    return summaries


def format_family_container_summary(
    summaries: list[dict[str, object]] | None,
    *,
    limit: int = 2,
) -> str | None:
    if not summaries:
        return None
    parts: list[str] = []
    for entry in summaries[:limit]:
        if not isinstance(entry, dict):
            continue
        samples = int(entry.get("samples", 0) or 0)
        if samples <= 0:
            continue
        parts.append(
            f"{samples} {entry.get('preset_family', 'unknown')} {entry.get('container', 'unknown')} match"
            + ("es" if samples != 1 else "")
        )
    return ", ".join(parts) if parts else None


def _preset_history_label(preset: str | None) -> str:
    if not preset:
        return "similar profiles"
    if preset in {"faster", "fast", "medium"}:
        return "Fast-like profiles"
    if preset in {"slow", "slower", "veryslow"}:
        return "slower software profiles"
    family = preset_family(preset)
    if family == "hardware":
        return "hardware profiles"
    if family == "software":
        return "software profiles"
    return f"{preset} profiles"


def describe_history_slices(
    store: dict[str, object] | None,
    *,
    preset: str | None = None,
    containers: set[str] | None = None,
) -> dict[str, str | None]:
    if not store:
        return {
            "closest_preset_history": None,
            "container_mix_history": None,
            "overall_history": None,
        }
    raw_records = store.get("records", [])
    if not isinstance(raw_records, list):
        raw_records = []
    accepted = [
        raw for raw in raw_records if isinstance(raw, dict) and raw.get("accepted_output", True)
    ]
    if not accepted:
        return {
            "closest_preset_history": None,
            "container_mix_history": None,
            "overall_history": None,
        }

    preset_matches: list[dict[str, object]] = []
    if preset:
        family_name = preset_family(preset)
        preset_matches = [
            raw
            for raw in accepted
            if raw.get("preset") == preset or raw.get("preset_family") == family_name
        ]

    normalized_containers = {container.lower() for container in (containers or set()) if container}
    container_matches = [
        raw
        for raw in accepted
        if not normalized_containers
        or str(raw.get("container") or "").lower() in normalized_containers
    ]

    closest_preset_history = None
    if preset_matches:
        dominant_container = None
        if normalized_containers:
            for candidate in normalized_containers:
                count = sum(
                    1
                    for raw in preset_matches
                    if str(raw.get("container") or "").lower() == candidate
                )
                if count:
                    dominant_container = candidate
                    break
        container_text = f" {dominant_container}" if dominant_container else ""
        closest_preset_history = f"{len(preset_matches)} accepted sample(s){container_text} for {_preset_history_label(preset)}"

    container_mix_history = None
    if container_matches and normalized_containers:
        listed = ", ".join(sorted(normalized_containers))
        container_mix_history = (
            f"{len(container_matches)} accepted sample(s) in the current container mix ({listed})"
        )

    overall_history = f"{len(accepted)} accepted sample(s) overall"
    return {
        "closest_preset_history": closest_preset_history,
        "container_mix_history": container_mix_history,
        "overall_history": overall_history,
    }


def _average_prediction_error(
    records: list[dict[str, object]],
    predicted_key: str,
    input_key: str,
    output_key: str,
) -> float | None:
    errors: list[float] = []
    for raw in records:
        predicted = raw.get(predicted_key)
        input_bytes = raw.get(input_key)
        output_bytes = raw.get(output_key)
        if not isinstance(predicted, (int, float)) or predicted <= 0:
            continue
        if not isinstance(input_bytes, (int, float)) or input_bytes <= 0:
            continue
        if not isinstance(output_bytes, (int, float)) or output_bytes < 0:
            continue
        actual_ratio = float(output_bytes) / max(float(input_bytes), 1.0)
        errors.append(actual_ratio - float(predicted))
    if not errors:
        return None
    return sum(errors) / len(errors)


def _average_speed_error(records: list[dict[str, object]]) -> float | None:
    errors: list[float] = []
    for raw in records:
        predicted = raw.get("predicted_speed")
        actual = raw.get("effective_speed")
        if not isinstance(predicted, (int, float)) or predicted <= 0:
            continue
        if not isinstance(actual, (int, float)) or actual <= 0:
            continue
        errors.append((float(actual) - float(predicted)) / float(predicted))
    if not errors:
        return None
    return sum(errors) / len(errors)


def summarize_calibration_store(store: dict[str, object] | None) -> dict[str, object]:
    records = store.get("records", []) if isinstance(store, dict) else []
    failures = store.get("failures", []) if isinstance(store, dict) else []
    if not isinstance(records, list):
        records = []
    if not isinstance(failures, list):
        failures = []

    accepted_records = [
        raw for raw in records if isinstance(raw, dict) and raw.get("accepted_output", True)
    ]
    rejected_records = [
        raw for raw in records if isinstance(raw, dict) and not raw.get("accepted_output", True)
    ]
    rejection_reasons: dict[str, int] = {}
    for raw in rejected_records:
        reason = str(raw.get("safety_rejection_reason") or "unknown")
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    by_preset: dict[tuple[str, str], dict[str, object]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        preset = str(raw.get("preset") or "unknown")
        container = str(raw.get("container") or "unknown")
        entry = by_preset.setdefault(
            (preset, container),
            {
                "preset": preset,
                "container": container,
                "samples": 0,
                "accepted_samples": 0,
                "rejected_samples": 0,
                "output_ratio_total": 0.0,
                "output_ratio_samples": 0,
                "speed_total": 0.0,
                "speed_samples": 0,
            },
        )
        entry["samples"] = int(entry["samples"]) + 1
        if raw.get("accepted_output", True):
            entry["accepted_samples"] = int(entry["accepted_samples"]) + 1
        else:
            entry["rejected_samples"] = int(entry["rejected_samples"]) + 1

        input_bytes = raw.get("input_bytes")
        output_bytes = raw.get("output_bytes")
        if (
            isinstance(input_bytes, (int, float))
            and input_bytes > 0
            and isinstance(output_bytes, (int, float))
            and output_bytes >= 0
        ):
            entry["output_ratio_total"] = float(entry["output_ratio_total"]) + (
                float(output_bytes) / float(input_bytes)
            )
            entry["output_ratio_samples"] = int(entry["output_ratio_samples"]) + 1

        speed = raw.get("effective_speed")
        if isinstance(speed, (int, float)) and speed > 0:
            entry["speed_total"] = float(entry["speed_total"]) + float(speed)
            entry["speed_samples"] = int(entry["speed_samples"]) + 1

    preset_summaries: list[dict[str, object]] = []
    for entry in by_preset.values():
        ratio_samples = int(entry["output_ratio_samples"])
        speed_samples = int(entry["speed_samples"])
        preset_summaries.append(
            {
                "preset": entry["preset"],
                "container": entry["container"],
                "samples": int(entry["samples"]),
                "accepted_samples": int(entry["accepted_samples"]),
                "rejected_samples": int(entry["rejected_samples"]),
                "avg_output_ratio": (
                    float(entry["output_ratio_total"]) / ratio_samples
                    if ratio_samples > 0
                    else None
                ),
                "avg_speed": (
                    float(entry["speed_total"]) / speed_samples if speed_samples > 0 else None
                ),
            }
        )
    preset_summaries.sort(key=lambda item: (-int(item["samples"]), str(item["preset"])))

    recent_records = []
    for raw in records[-5:]:
        if not isinstance(raw, dict):
            continue
        recent_records.append(
            {
                "preset": str(raw.get("preset") or "unknown"),
                "container": str(raw.get("container") or "unknown"),
                "accepted_output": bool(raw.get("accepted_output", True)),
                "input_bytes": int(raw.get("input_bytes", 0) or 0),
                "output_bytes": int(raw.get("output_bytes", 0) or 0),
            }
        )

    family_container_summaries = _family_container_summaries(
        [raw for raw in records if isinstance(raw, dict)]
    )
    bias_summary = _recent_bias_summary([raw for raw in records if isinstance(raw, dict)])

    return {
        "records": len(records),
        "accepted_records": len(accepted_records),
        "rejected_records": len(rejected_records),
        "rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                rejection_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "failures": len([raw for raw in failures if isinstance(raw, dict)]),
        "batch_bias_records": len(
            [raw for raw in store.get("batch_biases", []) if isinstance(raw, dict)]
        )
        if isinstance(store, dict)
        else 0,
        "preset_summaries": preset_summaries,
        "family_container_summaries": family_container_summaries,
        "family_container_summary_text": format_family_container_summary(
            family_container_summaries
        ),
        "bias_summary": bias_summary,
        "recent_records": recent_records,
    }
