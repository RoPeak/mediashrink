from __future__ import annotations

from pathlib import Path

from mkv_compress.cleanup import cleanup_successful_results, eligible_cleanup_results
from mkv_compress.models import EncodeJob, EncodeResult


def _make_result(
    tmp_path: Path,
    *,
    source_name: str = "ep01.mkv",
    output_name: str = "ep01_compressed.mkv",
    success: bool = True,
    skipped: bool = False,
    dry_run: bool = False,
) -> EncodeResult:
    source = tmp_path / source_name
    output = tmp_path / output_name
    source.write_bytes(b"original")
    if success and not dry_run and source != output:
        output.write_bytes(b"compressed")

    job = EncodeJob(
        source=source,
        output=output,
        tmp_output=tmp_path / f".tmp_{Path(output_name).name}",
        crf=20,
        preset="fast",
        dry_run=dry_run,
        skip=False,
        skip_reason=None,
    )
    return EncodeResult(
        job=job,
        skipped=skipped,
        skip_reason=None,
        success=success,
        input_size_bytes=source.stat().st_size,
        output_size_bytes=output.stat().st_size if output.exists() else 0,
        duration_seconds=10.0,
    )


def test_eligible_cleanup_results_filters_to_successful_side_by_side_outputs(tmp_path: Path) -> None:
    good = _make_result(tmp_path, source_name="good.mkv", output_name="good_compressed.mkv")
    skipped = _make_result(tmp_path, source_name="skip.mkv", output_name="skip_compressed.mkv", skipped=True)
    failed = _make_result(tmp_path, source_name="fail.mkv", output_name="fail_compressed.mkv", success=False)
    overwrite = _make_result(tmp_path, source_name="overwrite.mkv", output_name="overwrite.mkv")

    eligible = eligible_cleanup_results([good, skipped, failed, overwrite])

    assert eligible == [good]


def test_cleanup_successful_results_restores_original_name_and_removes_source(tmp_path: Path) -> None:
    result = _make_result(tmp_path, source_name="ep01.mp4", output_name="ep01_compressed.mp4")

    cleaned = cleanup_successful_results([result])

    assert cleaned == [tmp_path / "ep01.mp4"]
    assert (tmp_path / "ep01.mp4").read_bytes() == b"compressed"
    assert not (tmp_path / "ep01_compressed.mp4").exists()
    assert not (tmp_path / ".cleanup_backup_ep01.mp4").exists()
