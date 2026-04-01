from __future__ import annotations

import json
from pathlib import Path

from mediashrink.models import SessionFileEntry, SessionManifest
from mediashrink.profiles import (
    SavedProfile,
    get_builtin_profiles,
    list_all_profiles,
    upsert_profile,
)

_BUILTIN_NAMES = {"TV Batch", "Archival", "Fast GPU Transcode", "Smallest Acceptable"}


def test_builtin_profiles_not_persisted(tmp_path: Path) -> None:
    profiles_file = tmp_path / "profiles.json"
    builtin = SavedProfile(name="TV Batch", preset="faster", crf=22, builtin=True)
    upsert_profile(builtin, profiles_file)
    raw = json.loads(profiles_file.read_text(encoding="utf-8"))
    names = {item["name"] for item in raw}
    assert "TV Batch" not in names


def test_list_all_profiles_includes_builtins(tmp_path: Path) -> None:
    profiles_file = tmp_path / "profiles.json"
    all_profiles = list_all_profiles(profiles_file)
    names = {p.name for p in all_profiles}
    assert _BUILTIN_NAMES.issubset(names)


def test_get_builtin_profiles_returns_four() -> None:
    builtins = get_builtin_profiles()
    assert len(builtins) == 4
    assert all(p.builtin for p in builtins)


def test_session_manifest_round_trip() -> None:
    entry = SessionFileEntry(source="/video/a.mkv", status="success", output="/video/a_compressed.mkv")
    manifest = SessionManifest(
        version=1,
        directory="/video",
        timestamp="2026-04-01T12:00:00",
        preset="fast",
        crf=20,
        overwrite=False,
        output_dir=None,
        entries=[entry],
    )
    restored = SessionManifest.from_dict(manifest.to_dict())
    assert restored.version == manifest.version
    assert restored.directory == manifest.directory
    assert restored.preset == manifest.preset
    assert restored.crf == manifest.crf
    assert restored.overwrite == manifest.overwrite
    assert restored.output_dir is None
    assert len(restored.entries) == 1
    assert restored.entries[0].source == entry.source
    assert restored.entries[0].status == entry.status
    assert restored.entries[0].output == entry.output
