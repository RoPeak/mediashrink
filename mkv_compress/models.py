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
