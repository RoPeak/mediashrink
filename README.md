# mkv-compress

A CLI tool to re-encode MKV files using H.265/HEVC, reducing Blu-ray rips (~9 GB) to
manageable sizes (~1–4 GB) while preserving all audio and subtitle streams.

At CRF 20 (the default), expect roughly **60–75% file size reduction** — a 9 GB Blu-ray
episode typically comes out around 1.5–3 GB with no perceptible quality loss at normal
viewing distance. Files whose video stream is already H.265 are skipped automatically.

Companion tool to [plexify](../plexify).

## Requirements

- Python 3.10+
- FFmpeg with libx265 support (must be on PATH)

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
# Dry run (show what would happen, no encoding)
mkvcompress /path/to/mkvs --dry-run

# Encode in-place with _compressed suffix (default)
mkvcompress /path/to/mkvs

# Encode to a separate output directory
mkvcompress /path/to/mkvs --output-dir /path/to/output

# Overwrite originals after successful encoding
mkvcompress /path/to/mkvs --overwrite

# Custom CRF value (default: 20)
mkvcompress /path/to/mkvs --crf 22

# Skip the confirmation prompt
mkvcompress /path/to/mkvs --yes
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `directory` | required | Directory containing .mkv files |
| `--output-dir` / `-o` | None | Write output files here instead of alongside originals |
| `--overwrite` | False | Replace original files after successful encoding |
| `--crf` | 20 | H.265 CRF quality value (0–51, lower = better quality) |
| `--preset` | slow | FFmpeg encoding preset (slower = better compression) |
| `--recursive` / `-r` | False | Scan subdirectories for .mkv files |
| `--dry-run` | False | Preview what would be encoded without actually encoding |
| `--no-skip` | False | Encode files even if they appear already H.265 |
| `--yes` / `-y` | False | Skip the confirmation prompt |

## Quality guide

| CRF | Quality | Typical use |
|-----|---------|-------------|
| 18 | Near-lossless | Archival, very large output |
| 20 | Visually lossless | **Recommended default** |
| 22 | High quality | Good balance of size and quality |
| 28 | Acceptable | Smallest practical size |

## How it works

1. Scans the directory for `.mkv` files.
2. Probes each file with `ffprobe` to detect if the video stream is already H.265 (skips if so).
3. Displays a summary table and prompts for confirmation.
4. For each file: encodes to a `.tmp_<stem>.mkv` file, then renames to the final output on success.
5. On failure or interruption, the temporary file is deleted — originals are never modified unless `--overwrite` is passed.
6. Displays per-file stats (input size, output size, reduction %, time elapsed).
7. Shows a final summary table across all files.

## Testing

```bash
python -m pytest -q
```

## Platform support

Windows and Linux. Uses `pathlib.Path` throughout and `platform.system()` for OS detection.
FFmpeg must be installed separately and available on `PATH`.

### Installing FFmpeg

**Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your `PATH`.

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

**Linux (Fedora):**
```bash
sudo dnf install ffmpeg
```
