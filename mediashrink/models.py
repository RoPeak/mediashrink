from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EncodeJob:
    source: Path
    output: Path
    tmp_output: Path
    crf: int
    preset: str
    dry_run: bool
    skip: bool = False
    skip_reason: str | None = None
    source_codec: str | None = None          # e.g. "vc1", "h264", "hevc"
    estimated_output_bytes: int = 0          # 0 means unknown


@dataclass
class EncodeResult:
    job: EncodeJob
    skipped: bool
    skip_reason: str | None
    success: bool
    input_size_bytes: int
    output_size_bytes: int
    duration_seconds: float
    error_message: str | None = field(default=None)
    media_duration_seconds: float = 0.0   # source file's playback duration

    @property
    def size_reduction_bytes(self) -> int:
        return self.input_size_bytes - self.output_size_bytes

    @property
    def size_reduction_pct(self) -> float:
        if self.input_size_bytes == 0:
            return 0.0
        return (self.size_reduction_bytes / self.input_size_bytes) * 100

    @property
    def size_reduction_gb(self) -> float:
        return self.size_reduction_bytes / (1024 ** 3)


@dataclass
class AnalysisItem:
    source: Path
    codec: str | None
    size_bytes: int
    duration_seconds: float
    bitrate_kbps: float
    estimated_output_bytes: int
    estimated_savings_bytes: int
    recommendation: str
    reason_code: str
    reason_text: str

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "source": str(self.source),
            "codec": self.codec,
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "bitrate_kbps": self.bitrate_kbps,
            "estimated_output_bytes": self.estimated_output_bytes,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "recommendation": self.recommendation,
            "reason_code": self.reason_code,
            "reason_text": self.reason_text,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "AnalysisItem":
        source = raw.get("source")
        recommendation = raw.get("recommendation")
        reason_code = raw.get("reason_code")
        reason_text = raw.get("reason_text")
        if not isinstance(source, str):
            raise ValueError("analysis item source must be a string")
        if not isinstance(recommendation, str):
            raise ValueError("analysis item recommendation must be a string")
        if not isinstance(reason_code, str):
            raise ValueError("analysis item reason_code must be a string")
        if not isinstance(reason_text, str):
            raise ValueError("analysis item reason_text must be a string")
        codec = raw.get("codec")
        if codec is not None and not isinstance(codec, str):
            raise ValueError("analysis item codec must be a string or null")
        return cls(
            source=Path(source),
            codec=codec,
            size_bytes=int(raw.get("size_bytes", 0)),
            duration_seconds=float(raw.get("duration_seconds", 0.0)),
            bitrate_kbps=float(raw.get("bitrate_kbps", 0.0)),
            estimated_output_bytes=int(raw.get("estimated_output_bytes", 0)),
            estimated_savings_bytes=int(raw.get("estimated_savings_bytes", 0)),
            recommendation=recommendation,
            reason_code=reason_code,
            reason_text=reason_text,
        )


@dataclass
class AnalysisManifest:
    version: int
    analyzed_directory: Path
    recursive: bool
    preset: str
    crf: int
    profile_name: str | None
    estimated_total_encode_seconds: float | None
    items: list[AnalysisItem]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "analyzed_directory": str(self.analyzed_directory),
            "recursive": self.recursive,
            "preset": self.preset,
            "crf": self.crf,
            "profile_name": self.profile_name,
            "estimated_total_encode_seconds": self.estimated_total_encode_seconds,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "AnalysisManifest":
        analyzed_directory = raw.get("analyzed_directory")
        preset = raw.get("preset")
        if not isinstance(analyzed_directory, str):
            raise ValueError("manifest analyzed_directory must be a string")
        if not isinstance(preset, str):
            raise ValueError("manifest preset must be a string")
        raw_items = raw.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("manifest items must be a list")
        profile_name = raw.get("profile_name")
        if profile_name is not None and not isinstance(profile_name, str):
            raise ValueError("manifest profile_name must be a string or null")
        estimated_total = raw.get("estimated_total_encode_seconds")
        if estimated_total is not None:
            estimated_total = float(estimated_total)
        return cls(
            version=int(raw.get("version", 0)),
            analyzed_directory=Path(analyzed_directory),
            recursive=bool(raw.get("recursive", False)),
            preset=preset,
            crf=int(raw.get("crf", 0)),
            profile_name=profile_name,
            estimated_total_encode_seconds=estimated_total,
            items=[AnalysisItem.from_dict(item) for item in raw_items],
        )
