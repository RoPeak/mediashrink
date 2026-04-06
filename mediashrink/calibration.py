from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


_STORE_VERSION = 1
_MAX_SUCCESS_RECORDS = 2000
_MAX_FAILURE_RECORDS = 1000


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


@dataclass
class FailureRecord:
    encoder: str
    container: str
    stage: str
    reason: str


@dataclass
class CalibrationEstimate:
    output_ratio: float | None
    speed: float | None
    failure_rate: float
    confidence: str
    exact_matches: int
    loose_matches: int


def get_calibration_path() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "mediashrink" / "calibration.json"

    xdg_config = os.getenv("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "mediashrink" / "calibration.json"

    return Path.home() / ".config" / "mediashrink" / "calibration.json"


def _empty_store() -> dict[str, object]:
    return {"version": _STORE_VERSION, "records": [], "failures": []}


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
    if not isinstance(records, list):
        records = []
    if not isinstance(failures, list):
        failures = []
    return {
        "version": int(raw.get("version", _STORE_VERSION)),
        "records": [item for item in records if isinstance(item, dict)],
        "failures": [item for item in failures if isinstance(item, dict)],
    }


def save_calibration_store(store: dict[str, object], path: Path | None = None) -> Path:
    calibration_path = path or get_calibration_path()
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    return calibration_path


def _trim_store(store: dict[str, object]) -> None:
    records = store.get("records", [])
    failures = store.get("failures", [])
    if isinstance(records, list) and len(records) > _MAX_SUCCESS_RECORDS:
        store["records"] = records[-_MAX_SUCCESS_RECORDS:]
    if isinstance(failures, list) and len(failures) > _MAX_FAILURE_RECORDS:
        store["failures"] = failures[-_MAX_FAILURE_RECORDS:]


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


def _matches(
    raw: dict[str, object],
    *,
    codec: str | None,
    resolution: str,
    bitrate: str,
    preset: str,
    container: str,
) -> tuple[bool, bool]:
    raw_codec = raw.get("codec")
    raw_resolution = raw.get("resolution_bucket")
    raw_bitrate = raw.get("bitrate_bucket")
    raw_preset = raw.get("preset")
    raw_family = raw.get("preset_family")
    raw_container = raw.get("container")
    codec_match = codec in {None, "unknown"} or raw_codec == codec
    resolution_match = resolution == "unknown" or raw_resolution == resolution
    bitrate_match = bitrate == "unknown" or raw_bitrate == bitrate
    container_match = container == "unknown" or raw_container == container
    preset_match = raw_preset == preset or raw_family == preset_family(preset)
    exact = codec_match and resolution_match and bitrate_match and container_match and preset_match
    loose = codec_match and resolution_match and preset_match
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

    exact_matches: list[dict[str, object]] = []
    loose_matches: list[dict[str, object]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        exact, loose = _matches(
            raw,
            codec=codec,
            resolution=resolution,
            bitrate=bitrate,
            preset=preset,
            container=container,
        )
        if exact:
            exact_matches.append(raw)
        elif loose:
            loose_matches.append(raw)

    selected = exact_matches or loose_matches
    if not selected:
        return None

    output_ratios = [
        float(item["output_bytes"]) / max(int(item["input_bytes"]), 1)
        for item in selected
        if int(item.get("input_bytes", 0)) > 0 and int(item.get("output_bytes", 0)) > 0
    ]
    speeds = [
        float(item["effective_speed"])
        for item in selected
        if float(item.get("effective_speed", 0.0)) > 0
    ]
    failure_rate = estimate_failure_rate(store, preset=preset, container=container)
    exact_count = len(exact_matches)
    loose_count = len(loose_matches)
    confidence = "Low"
    if exact_count >= 3:
        confidence = "High"
    elif exact_count >= 1 or loose_count >= 3:
        confidence = "Medium"

    return CalibrationEstimate(
        output_ratio=(sum(output_ratios) / len(output_ratios)) if output_ratios else None,
        speed=(sum(speeds) / len(speeds)) if speeds else None,
        failure_rate=failure_rate,
        confidence=confidence,
        exact_matches=exact_count,
        loose_matches=loose_count,
    )


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
                successes += 1
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
