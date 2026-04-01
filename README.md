# mediashrink

A CLI tool to re-encode supported video files (`.mkv`, `.mp4`, `.m4v`) to H.265/HEVC, keeping all audio and subtitle streams while cutting file sizes substantially.

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

For most users, start with the wizard:

```bash
mediashrink wizard /path/to/library
```

It now handles:

- encoder/profile selection
- automatic library analysis
- recommended/maybe/skip review
- manifest export if wanted
- compressing the recommended set by default

Advanced/manual flows are still available below.

## Advanced Usage

```bash
# Encode in-place with a _compressed suffix
mediashrink /path/to/mkvs

# Encode and then clean up originals after success
mediashrink /path/to/mkvs --cleanup

# Analyze a library and write a candidate manifest
mediashrink analyze /path/to/library --recursive --manifest-out candidates.json

# Apply a previously generated manifest
mediashrink apply candidates.json

# Dry run only
mediashrink /path/to/mkvs --dry-run

# Send output to another directory
mediashrink /path/to/mkvs --output-dir /path/to/output

# Overwrite originals after successful encode
mediashrink /path/to/mkvs --overwrite

# Override defaults manually
mediashrink /path/to/mkvs --crf 22 --preset slow

# Reuse a saved wizard profile
mediashrink /path/to/mkvs --profile tv-batch
```

## Wizard

The wizard is the main guided workflow. It detects usable encoders with FFmpeg probe runs, benchmarks a short sample, analyzes the target library with the chosen settings, and then walks the user to a simple next step.

```bash
mediashrink wizard /path/to/mkvs
```

The wizard scans subdirectories by default, so it works out of the box for libraries organized into per-movie or per-show folders.
Use `--no-recursive` if you only want the top-level directory.

The wizard:

1. Scans the batch and totals the input size.
2. Detects working hardware encoders such as `qsv`, `nvenc`, and `amf`.
3. Shows best-effort device labels when they can be derived safely.
4. Benchmarks a short clip to estimate relative speed.
5. Presents generated profiles with approximate output-size, time, and quality trade-offs.
6. Lets you select a generated profile or customize encoder, CRF, and software preset.
7. Analyzes the library with those settings and classifies files as `recommended`, `maybe`, or `skip`.
8. Defaults to compressing the recommended files only.
9. Can optionally review `maybe` files or export a manifest instead of encoding immediately.

Important: wizard time and size numbers are approximate estimates, not guarantees.

## Library Analysis

`mediashrink analyze` is the advanced/manual version of the same recommendation pass used by the wizard:

```bash
mediashrink analyze /path/to/library --recursive --manifest-out candidates.json
```

The analysis flow:

1. Scans supported video files (`.mkv`, `.mp4`, `.m4v`) in the target directory tree.
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
mediashrink analyze /path/to/library --profile tv-batch
mediashrink analyze /path/to/library --profile tv-batch --crf 18 --preset slow
```

Explicit `--crf` and `--preset` still override profile values.

## Manifest Apply

Use `apply` to run the existing encode pipeline against a manifest when you want a scriptable two-step workflow:

```bash
mediashrink apply candidates.json
mediashrink apply candidates.json --output-dir /path/to/output
mediashrink apply candidates.json --profile tv-batch
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

- Windows: `%APPDATA%\mediashrink\profiles.json`
- Linux: `$XDG_CONFIG_HOME/mediashrink/profiles.json`
- Fallback: `~/.config/mediashrink/profiles.json`

Use them with:

```bash
mediashrink /path/to/mkvs --profile tv-batch
mediashrink profiles list
mediashrink profiles delete tv-batch
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
| `--cleanup` | `False` | After successful side-by-side encodes, delete originals and rename outputs back to the original filename |

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

Even with hardware acceleration, this is still a full video re-encode. Source duration, bitrate, and resolution dominate runtime, so multi-hour runs for large TV seasons are normal.

## How It Works

1. Scan the target directory for supported video files (`.mkv`, `.mp4`, `.m4v`).
2. Probe the source codec with `ffprobe`.
3. Skip already-compressed HEVC files unless `--no-skip` is set.
4. Encode to a temporary `.tmp_<stem><suffix>` file first.
5. Rename the temp file only after a successful encode.
6. Delete temp files on failure or interruption.
7. Never replace originals unless `--overwrite` is explicitly requested.

Re-runs are conservative by default:

- files whose names already contain `_compressed` are skipped
- files whose first video stream is already HEVC/H.265 are skipped
- this means re-running on a half-finished season folder should normally target only the newly added, still-uncompressed files

If you encode side-by-side instead of using `--overwrite`, the interactive flows can offer a cleanup step at the end to delete the originals and restore the compressed outputs to the original filenames.

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
