from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SavedProfile:
    name: str
    preset: str
    crf: int
    label: str | None = None
    created_from_wizard: bool = False
    builtin: bool = False


def get_profiles_path() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "mediashrink" / "profiles.json"

    xdg_config = os.getenv("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "mediashrink" / "profiles.json"

    return Path.home() / ".config" / "mediashrink" / "profiles.json"


def load_profiles(path: Path | None = None) -> list[SavedProfile]:
    profiles_path = path or get_profiles_path()
    if not profiles_path.exists():
        return []

    try:
        raw = json.loads(profiles_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(raw, list):
        return []

    profiles: list[SavedProfile] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        preset = item.get("preset")
        crf = item.get("crf")
        if not isinstance(name, str) or not isinstance(preset, str) or not isinstance(crf, int):
            continue
        label = item.get("label")
        created_from_wizard = item.get("created_from_wizard", False)
        builtin = item.get("builtin", False)
        profiles.append(
            SavedProfile(
                name=name,
                preset=preset,
                crf=crf,
                label=label if isinstance(label, str) else None,
                created_from_wizard=bool(created_from_wizard),
                builtin=bool(builtin),
            )
        )
    return profiles


def get_builtin_profiles() -> list[SavedProfile]:
    """Return the fixed built-in intent presets. These are never persisted."""
    return [
        SavedProfile(
            name="Fast Batch", preset="faster", crf=22, label="TV shows / fast batch", builtin=True
        ),
        SavedProfile(name="Archival", preset="slow", crf=16, label="Maximum quality", builtin=True),
        SavedProfile(
            name="GPU Offload", preset="nvenc", crf=22, label="Hardware speed (GPU)", builtin=True
        ),
        SavedProfile(
            name="Smallest Acceptable",
            preset="slow",
            crf=28,
            label="Maximum compression",
            builtin=True,
        ),
    ]


def list_all_profiles(path: Path | None = None) -> list[SavedProfile]:
    """Return built-in profiles followed by user-saved profiles."""
    return get_builtin_profiles() + load_profiles(path)


def save_profiles(profiles: list[SavedProfile], path: Path | None = None) -> Path:
    profiles_path = path or get_profiles_path()
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    # Never persist built-in profiles
    payload = [asdict(profile) for profile in profiles if not profile.builtin]
    profiles_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return profiles_path


def get_profile(name: str, path: Path | None = None) -> SavedProfile | None:
    for profile in load_profiles(path):
        if profile.name == name:
            return profile
    return None


def upsert_profile(profile: SavedProfile, path: Path | None = None) -> Path:
    profiles = load_profiles(path)
    updated = False
    for idx, existing in enumerate(profiles):
        if existing.name == profile.name:
            profiles[idx] = profile
            updated = True
            break
    if not updated:
        profiles.append(profile)
    profiles.sort(key=lambda item: item.name.lower())
    return save_profiles(profiles, path)


def delete_profile(name: str, path: Path | None = None) -> bool:
    profiles = load_profiles(path)
    remaining = [profile for profile in profiles if profile.name != name]
    if len(remaining) == len(profiles):
        return False
    save_profiles(remaining, path)
    return True
