# mkv-compress

A CLI tool to re-encode `.mkv` files to H.265/HEVC, keeping all audio and subtitle streams while cutting file sizes substantially.

For Blu-ray-era H.264 or VC-1 sources, CRF 20 often lands around a 50-70% reduction with little visible loss in normal TV/movie viewing. Files already using HEVC are skipped by default.

## Requirements

- Python 3.10+
- FFmpeg with `libx265` support
- `ffmpeg` and `ffprobe` available on `PATH`

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux
source .venv/bin/activate

pip install -e .[dev]
```

## Usage

```bash
# Encode in-place with a _compressed suffix
mkvcompress /path/to/mkvs

# Analyze a library and write a candidate manifest
mkvcompress analyze /path/to/library --recursive --manifest-out candidates.json

# Apply a previously generated manifest
mkvcompress apply candidates.json

# Dry run only
mkvcompress /path/to/mkvs --dry-run

# Send output to another directory
mkvcompress /path/to/mkvs --output-dir /path/to/output

# Overwrite originals after successful encode
mkvcompress /path/to/mkvs --overwrite

# Override defaults manually
mkvcompress /path/to/mkvs --crf 22 --preset slow

# Reuse a saved wizard profile
mkvcompress /path/to/mkvs --profile tv-batch
```

## Wizard

The wizard detects usable encoders with FFmpeg probe runs, benchmarks a short sample, and then offers generated presets plus a custom path.

```bash
mkvcompress wizard /path/to/mkvs
```

The wizard:

1. Scans the batch and totals the input size.
2. Detects working hardware encoders such as `qsv`, `nvenc`, and `amf`.
3. Shows best-effort device labels when they can be derived safely.
4. Benchmarks a short clip to estimate relative speed.
5. Presents generated profiles with approximate output-size, time, and quality trade-offs.
6. Lets you select a generated profile or customize encoder, CRF, and software preset.
7. Optionally saves the chosen settings as a named profile before encoding starts.

Important: wizard time and size numbers are approximate estimates, not guarantees.

## Library Analysis

`mkvcompress analyze` adds a conservative recommendation pass before encoding:

```bash
mkvcompress analyze /path/to/library --recursive --manifest-out candidates.json
```

The analysis flow:

1. Scans `.mkv` files in the target directory tree.
2. Probes codec, duration, bitrate, and projected output size.
3. Classifies each file as `recommended`, `maybe`, or `skip`.
4. Prints a summary of likely savings and rough encode time using the chosen profile/settings.
5. Optionally writes a JSON manifest containing only recommended files.

Defaults are conservative:

- already-HEVC files are skipped
- files already marked `_compressed` are skipped
- small projected wins are skipped
- borderline cases are shown as `maybe` in the console but excluded from the default manifest

Analysis supports the same settings precedence as encoding:

```bash
mkvcompress analyze /path/to/library --profile tv-batch
mkvcompress analyze /path/to/library --profile tv-batch --crf 18 --preset slow
```

Explicit `--crf` and `--preset` still override profile values.

## Manifest Apply

Use `apply` to run the existing encode pipeline against a manifest:

```bash
mkvcompress apply candidates.json
mkvcompress apply candidates.json --output-dir /path/to/output
mkvcompress apply candidates.json --profile tv-batch
```

`apply`:

- reads the manifest settings by default
- allows `--profile`, `--crf`, and `--preset` overrides
- skips missing files from stale manifests with a warning
- re-checks skip conditions at execution time instead of trusting the manifest blindly

## Saved Profiles

Profiles store only encoder settings:

- `preset`
- `crf`
- optional display label and wizard metadata

Profiles are saved per-user in a local JSON config file:

- Windows: `%APPDATA%\mkvcompress\profiles.json`
- Linux: `$XDG_CONFIG_HOME/mkvcompress/profiles.json`
- Fallback: `~/.config/mkvcompress/profiles.json`

Use them with:

```bash
mkvcompress /path/to/mkvs --profile tv-batch
mkvcompress profiles list
mkvcompress profiles delete tv-batch
```

If `--profile` is supplied together with `--crf` or `--preset`, the explicit CLI flags win.

## Main Options

| Flag | Default | Description |
| --- | --- | --- |
| `directory` | required | Directory containing `.mkv` files |
| `--output-dir`, `-o` | `None` | Write output files here instead of alongside originals |
| `--overwrite` | `False` | Replace original files after successful encoding |
| `--crf` | `20` | H.265 CRF quality value (`0-51`, lower = better quality) |
| `--preset` | `fast` | FFmpeg preset or hardware encoder key |
| `--profile` | `None` | Load saved `preset` and `crf` defaults |
| `--recursive`, `-r` | `False` | Scan subdirectories |
| `--dry-run` | `False` | Preview jobs without encoding |
| `--no-skip` | `False` | Encode files even if they already appear to be HEVC |
| `--yes`, `-y` | `False` | Skip the confirmation prompt |

## Quality Guide

| CRF | Quality | Typical use |
| --- | --- | --- |
| `18` | Near-lossless | Archival or maximum quality |
| `20` | Visually lossless | Good default |
| `22` | High quality | Balanced manual choice |
| `28` | Acceptable | Smaller files, more compromise |

## Preset Guide

Software presets trade time for compression efficiency:

| Preset | Relative speed | Typical use |
| --- | --- | --- |
| `slow` | Slowest | Best compression efficiency |
| `medium` | Moderate | Balanced quality/time |
| `fast` | Faster | Default batch choice |
| `faster` | Faster again | Lower wait time |
| `ultrafast` | Fastest software | Lowest compression efficiency |

Hardware presets are much faster when supported:

| Preset | Hardware |
| --- | --- |
| `qsv` | Intel Quick Sync |
| `nvenc` | Nvidia NVENC |
| `amf` | AMD AMF |

FFmpeg probe success determines whether a hardware encoder is offered. If a hardware option is unavailable or unstable on a system, use a software preset such as `fast`.

## How It Works

1. Scan the target directory for `.mkv` files.
2. Probe the source codec with `ffprobe`.
3. Skip already-compressed HEVC files unless `--no-skip` is set.
4. Encode to a temporary `.tmp_<stem>.mkv` file first.
5. Rename the temp file only after a successful encode.
6. Delete temp files on failure or interruption.
7. Never replace originals unless `--overwrite` is explicitly requested.

The optional analyze/apply flow sits in front of this to help you choose which files are worth compressing before any encode begins.

## Testing

```bash
python -m pytest -q
```

## FFmpeg Install Notes

Windows:

- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add the `bin` directory to `PATH`

Linux (Debian/Ubuntu):

```bash
sudo apt install ffmpeg
```

Linux (Fedora):

```bash
sudo dnf install ffmpeg
```
