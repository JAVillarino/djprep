# djprep

High-performance audio analysis for DJs. Extracts BPM, musical key, and stems from your music library, outputting Rekordbox-compatible XML for seamless import into Pioneer DJ software.

**djprep** is a free, open-source alternative to Mixed In Key with stem separation capabilities.

## Features

- **BPM Detection** - Accurate tempo analysis with confidence scoring and double-tempo correction
- **Key Detection** - Musical key in Camelot, Open Key, and standard notation
- **Stem Separation** - Split tracks into vocals, drums, bass, and other (HTDemucs via ONNX)
- **Rekordbox XML Export** - Direct import into Pioneer DJ software
- **JSON Export** - Machine-readable output for custom workflows
- **Batch Processing** - Analyze entire libraries with parallel processing
- **Multiple Formats** - MP3, WAV, FLAC, and AIFF support

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) 1.75 or later

### Build from Source

```bash
git clone https://github.com/JAVillarino/djprep.git
cd djprep
cargo build --release
```

The binary will be at `target/release/djprep`.

**With stem separation support:**

```bash
cargo build --release --features stems
```

This enables the `--stems` flag and adds ONNX Runtime dependencies (~50MB larger binary).

### Install to PATH

```bash
cargo install --path .
```

## Usage

### Basic Analysis

Analyze a folder of music:

```bash
djprep -i ~/Music/DJ -o ~/Music/analyzed
```

Analyze a single file:

```bash
djprep -i track.mp3 -o ./output
```

### CLI Options

```
djprep - High-performance audio analysis for DJs

USAGE:
    djprep [OPTIONS] --input <PATH> --output <DIR>

OPTIONS:
    -i, --input <PATH>      Input path (file or directory)
    -o, --output <DIR>      Output directory for XML/JSON files

    --stems                 Enable stem separation (GPU recommended)
    --stems-output <DIR>    Directory for stem files [default: <output>/stems]
    --genre <GENRE>         Genre hint for BPM detection (resolves double-tempo)
                            [values: house, techno, trance, dnb, dubstep,
                             hiphop, pop, rock]

    -j, --threads <N>       Number of worker threads [default: CPU count - 2]
    -r, --recursive         Scan subdirectories [default: true]
    --force                 Re-analyze all files (by default, skips already-analyzed)
    --json                  Output JSON in addition to XML [default: true]
    --dry-run               Show files that would be analyzed without processing

    -v, --verbose           Increase verbosity (-v, -vv, -vvv)
    -q, --quiet             Suppress progress bars
    -h, --help              Print help
    -V, --version           Print version
```

### Examples

```bash
# Analyze with genre hint (improves BPM accuracy)
djprep -i ./music --genre dnb -o ./analyzed

# Enable stem separation
djprep -i ./music --stems -o ./analyzed

# Preview what would be analyzed (no processing)
djprep -i ./music -o ./out --dry-run

# Force re-analysis of all files
djprep -i ./music -o ./out --force

# Verbose output for debugging
djprep -i ./music -o ./out -vv
```

## Output Files

djprep generates two output files in the specified output directory:

### rekordbox.xml

Rekordbox-compatible XML containing:
- Track metadata (name, artist, album)
- BPM with 2 decimal precision
- Musical key in Camelot notation
- File location as properly encoded URI
- Analysis timestamp

```xml
<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="djprep" Version="0.1.0"/>
  <COLLECTION Entries="2">
    <TRACK TrackID="123456" Name="Track Name" Artist="Artist"
           Location="file://localhost/path/to/track.mp3"
           TotalTime="300" AverageBpm="128.00" Tonality="11B"
           DateAdded="2024-01-15"/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT">
      <NODE Type="1" Name="djprep_import_20240115_120000" Entries="2">
        <TRACK Key="123456"/>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
```

### djprep.json

JSON sidecar with detailed analysis data:

```json
{
  "version": "1.0",
  "metadata": {
    "generator_version": "0.1.0",
    "exported_at": "2024-01-15T12:00:00Z",
    "track_count": 2
  },
  "tracks": [
    {
      "track_id": 123456,
      "path": "/path/to/track.mp3",
      "bpm": {
        "value": 128.0,
        "confidence": 0.95,
        "candidates": [
          { "value": 64.0, "confidence": 0.3 }
        ]
      },
      "key": {
        "standard": "Am",
        "camelot": "8A",
        "open_key": "8m",
        "confidence": 0.88
      },
      "duration_seconds": 300.5
    }
  ]
}
```

## Importing into Rekordbox

### Important: The Playlist Workaround

Rekordbox has a known bug where directly importing XML does **not** update metadata for tracks that already exist in your collection. To ensure BPM and key values are properly updated:

1. In Rekordbox: **File > Import Collection** and select `rekordbox.xml`
2. In the tree view, find the `djprep_import_*` playlist
3. Right-click the playlist and select **Import Playlist**

This playlist-based import forces Rekordbox to update all metadata correctly.

### Why This Workaround?

When you drag tracks directly from an imported XML collection, Rekordbox only imports tracks that don't already exist. For existing tracks, it silently ignores the new BPM/key values. The playlist import method forces Rekordbox to reconcile the metadata.

## Key Notation

djprep outputs musical keys in multiple notations:

| Standard | Camelot | Open Key | Description |
|----------|---------|----------|-------------|
| Am       | 8A      | 8m       | A minor     |
| C        | 8B      | 8d       | C major     |
| Fm       | 4A      | 4m       | F minor     |
| G#       | 4B      | 4d       | G# major    |

**Harmonic Mixing Tip:** Keys that are adjacent on the Camelot wheel (e.g., 8A to 7A or 9A) or share the same number (8A to 8B) mix harmonically.

## Stem Separation

When `--stems` is enabled, djprep uses HTDemucs (via ONNX Runtime) to separate each track into four stems:

- **vocals.wav** - Isolated vocals
- **drums.wav** - Drums and percussion
- **bass.wav** - Bass frequencies
- **other.wav** - Everything else (synths, guitars, etc.)

### Hardware Acceleration

djprep automatically selects the best available execution provider:
1. **CUDA** (NVIDIA GPUs) - Fastest
2. **CoreML** (Apple Silicon) - Fast on M1/M2/M3
3. **DirectML** (Windows) - AMD/Intel GPUs
4. **CPU** - Fallback (slow but always works)

### Model Setup

Stem separation requires an HTDemucs ONNX model (~170MB). djprep automatically searches for the model in these locations:

1. `DJPREP_MODEL_PATH` environment variable
2. `~/.cache/djprep/models/htdemucs_v4.onnx` (Linux) or `~/Library/Caches/com.djprep.djprep/models/` (macOS)
3. `~/.local/share/djprep/models/` (Linux) or `~/Library/Application Support/com.djprep.djprep/models/` (macOS)
4. `./models/htdemucs_v4.onnx` (current directory)
5. `~/djprep/models/htdemucs_v4.onnx`

**Option 1: Place the model in a standard location**
```bash
mkdir -p ~/.cache/djprep/models
# Copy your htdemucs_v4.onnx to ~/.cache/djprep/models/
```

**Option 2: Set the environment variable**
```bash
export DJPREP_MODEL_PATH=/path/to/htdemucs_v4.onnx
djprep -i ./music --stems -o ./output
```

**Where to get the model:**
- [Intel OpenVINO Audacity Plugin](https://github.com/intel/openvino-plugins-ai-audacity/releases) - Extract `htdemucs_v4.onnx` from the release zip
- [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) - Convert from PyTorch (requires Python)

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| CLI | Complete | Full argument parsing with --dry-run, --force |
| File Discovery | Complete | Recursive scanning, extension filtering |
| Audio Decoding | Complete | MP3, WAV, FLAC, AIFF via symphonia |
| BPM Detection | Complete | stratum-dsp 1.0, genre-aware tempo correction |
| Key Detection | Complete | stratum-dsp 1.0, Camelot/Open Key notation |
| Stem Separation | Complete | ort 2.0 + HTDemucs, GPU acceleration |
| Metadata Extraction | Complete | ID3v2, Vorbis comments, AIFF tags via lofty |
| Rekordbox XML | Complete | Streaming writer, URI encoding |
| JSON Export | Complete | Full analysis data |
| Progress Bars | Complete | Multi-progress with indicatif |
| Error Handling | Complete | Actionable error messages with suggestions |
| Incremental Analysis | Complete | Skips already-analyzed files (--force to override) |

### Roadmap

**v0.1.0 (Current):**
- CLI and file discovery
- Audio decoding via symphonia
- BPM/Key detection via stratum-dsp
- Stem separation via ONNX Runtime (HTDemucs)
- GPU acceleration (CUDA, CoreML, DirectML)
- Metadata extraction from file tags
- Incremental analysis (skip unchanged files)
- Dry-run mode for previewing analysis
- Comprehensive error messages

**Future:**
- Additional export formats (Serato, Traktor)
- Automatic model download
- Waveform generation

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| MP3 | `.mp3` | All bitrates |
| WAV | `.wav` | PCM 16/24/32-bit |
| FLAC | `.flac` | Lossless |
| AIFF | `.aiff`, `.aif` | Common on macOS |

## Technical Details

### Analysis Engine
- **BPM/Key Detection:** stratum-dsp 1.0 (pure Rust, no FFI dependencies)
- **Sample Rate:** Audio resampled to 22,050 Hz mono for BPM/key analysis
- **Thread Pool:** Uses N-2 threads by default (reserves capacity for stems and export)

### Stem Separation
- **Model:** HTDemucs v4 via ONNX Runtime 2.0
- **Sample Rate:** 44,100 Hz stereo (full fidelity for stems)
- **Chunking:** 7.8-second segments with 1-second overlap
- **Reconstruction:** Overlap-add with linear crossfade
- **Concurrency:** Dedicated worker thread with bounded channel (backpressure)

### Output
- **Track IDs:** Deterministic FNV-1a hash of normalized path (consistent across runs)
- **Memory:** Streaming XML output handles large libraries without memory issues

## Troubleshooting

### "HTDemucs model not found"

The stem separation model couldn't be located. djprep checks these locations:
- `DJPREP_MODEL_PATH` environment variable
- `~/.cache/djprep/models/htdemucs_v4.onnx`
- `./models/htdemucs_v4.onnx`

**Solution:** Download the model from [Intel OpenVINO Audacity releases](https://github.com/intel/openvino-plugins-ai-audacity/releases) and place it in one of the above locations.

### "Failed to decode audio file"

The audio file couldn't be read. This can happen with:
- Corrupted files
- Unsupported codecs (e.g., some AAC variants)
- DRM-protected files

**Solution:**
- Try playing the file in another app to verify it's valid
- Convert to a standard format: `ffmpeg -i input.m4a -acodec pcm_s16le output.wav`
- Supported formats: MP3, WAV, FLAC, AIFF

### "Permission denied" errors

Can't write to the output directory.

**Solution:**
- Check you have write access: `ls -la /path/to/output`
- Create the directory first: `mkdir -p /path/to/output`
- Use a different output location

### Analysis seems slow

BPM/Key analysis is CPU-bound (~3 seconds per track). Stem separation is much slower without GPU acceleration.

**Solutions:**
- Use `--dry-run` first to preview file count
- For stems, ensure GPU acceleration is active (check verbose output for "CoreML", "CUDA", or "DirectML")
- Reduce thread count if system is overloaded: `-j 2`

## Development

```bash
# Run tests
cargo test

# Run tests with stems feature
cargo test --features stems

# Run with debug logging
RUST_LOG=debug cargo run -- -i ./test -o ./out

# Check for issues
cargo clippy
cargo clippy --features stems

# Format code
cargo fmt
```

## Contributing

Contributions welcome! Areas that could use help:

- **Export formats:** Serato, Traktor support
- **Performance:** Benchmarking and optimization
- **Platform testing:** Windows and Linux validation
- **Documentation:** More usage examples

**Report bugs or request features:** [GitHub Issues](https://github.com/JAVillarino/djprep/issues)

### How to Contribute

1. Fork the [repository](https://github.com/JAVillarino/djprep)
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and ensure tests pass: `cargo test`
4. Check for issues: `cargo clippy`
5. Format code: `cargo fmt`
6. Submit a [pull request](https://github.com/JAVillarino/djprep/pulls)

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests (takes ~20 seconds)
cargo test --test integration_tests

# All tests with stems feature
cargo test --features stems
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note:** This project is not affiliated with Pioneer DJ, Rekordbox, or Mixed In Key. Rekordbox is a trademark of AlphaTheta Corporation.
