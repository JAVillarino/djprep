# djprep - Claude Code Implementation Guide

**Repository:** https://github.com/JAVillarino/djprep

## Project Overview

djprep is a CLI tool that analyzes audio files (MP3, WAV, FLAC, AIFF) to extract BPM, musical key, and optionally separate stems. It outputs Rekordbox-compatible XML for import into Pioneer DJ software, plus a JSON sidecar for interoperability.

**Core value proposition:** Free, open-source alternative to Mixed In Key with stem separation capabilities.

---

## Current Project State (January 2026)

The project is **feature-complete** and ready for v0.1.0 release:

| Component | Status | Notes |
|-----------|--------|-------|
| CLI (clap) | ✅ Complete | `--input`, `--output`, `--verbose`, `--stems`, `--dry-run`, `--force` |
| File Discovery | ✅ Complete | walkdir, filters by extension |
| Audio Decoding | ✅ Complete | symphonia, 22050Hz mono for analysis, 44100Hz stereo for stems |
| BPM Detection | ✅ Complete | stratum-dsp 1.0, genre-aware tempo correction |
| Key Detection | ✅ Complete | stratum-dsp 1.0, Camelot/Open Key notation |
| Stem Separation | ✅ Complete | ort 2.0 + HTDemucs, STFT preprocessing, overlap-add |
| Metadata Extraction | ✅ Complete | lofty 0.18, ID3v2/Vorbis/AIFF tags |
| Rekordbox XML | ✅ Complete | Streaming writer, URI encoding |
| JSON Export | ✅ Complete | Sidecar file with full analysis data |
| Progress Bars | ✅ Complete | indicatif multi-progress |
| Error Handling | ✅ Complete | Actionable error messages with suggestions |
| Concurrency | ✅ Complete | Bounded channel queue for stem backpressure |
| Incremental Analysis | ✅ Complete | Skips already-analyzed files (--force to override) |
| Tests | ✅ Complete | 25 unit + 9 integration + 2 doc tests |

**All Phase 5 (Polish) items complete:**
- ✅ Metadata extraction (ID3/Vorbis tags)
- ✅ Integration tests with sample audio
- ✅ Improved error messages with actionable guidance
- ✅ Model auto-discovery (checks multiple locations)
- ✅ Dry-run mode
- ✅ Incremental analysis (--force flag)

---

## How to Use Claude Code on This Project

### Starting a Session
```bash
cd /path/to/djprep
claude
```

### Effective Prompts for Claude Code

**To add a new export format (e.g., Serato):**
```
Add Serato export format support.

Requirements:
- Research Serato crate file format
- Create src/export/serato.rs module
- Add --format serato CLI option
- Test with real Serato installation if possible
```

**To add automatic model download:**
```
Implement automatic HTDemucs model download.

Requirements:
- Download from Intel OpenVINO releases
- Show progress bar during download
- Verify SHA256 hash after download
- Store in ~/.cache/djprep/models/
- Handle network errors gracefully
```

**To optimize performance:**
```
Profile and optimize the analysis pipeline for large libraries.

Steps:
1. Add benchmarks using criterion
2. Profile memory usage with valgrind/heaptrack
3. Identify bottlenecks
4. Implement optimizations
```

**To debug/understand code:**
```
Explain how the pipeline orchestrator works and trace the data flow from
file discovery through to XML export.
```

**To run tests:**
```
Run cargo test and fix any failures. Then run cargo clippy and address warnings.
```

---

## Technical Constraints

### Core Dependencies
- **Rust 1.75+** - Memory safety, single binary distribution
- **symphonia** - Pure Rust audio decoding (no FFmpeg dependency)
- **stratum-dsp 1.0** - BPM and key detection (pure Rust, zero C dependencies)
- **lofty 0.18** - Metadata extraction from ID3v2, Vorbis, AIFF tags
- **ort 2.0** - ONNX Runtime for HTDemucs stem separation (feature-gated)
- **rustfft 6.2** - STFT preprocessing for stem separation
- **rayon** - Data parallelism for batch processing
- **crossbeam-channel** - Bounded channel for stem backpressure
- **quick-xml** - Streaming XML writer for large libraries
- **clap** - CLI argument parsing

### Architecture Principles
1. **Trait abstractions** for analysis backends (swap implementations later)
2. **Staged pipeline** with backpressure between CPU analysis and GPU inference
3. **Streaming output** - don't hold entire XML in memory
4. **Graceful degradation** - if stems fail, still output BPM/key

---

## Data Types

```rust
// Core analysis results
pub struct AnalysisResult {
    pub track_id: i32,          // Deterministic, derived from path hash
    pub bpm: f64,
    pub bpm_confidence: f64,
    pub key: KeyResult,
    pub duration_seconds: f64,
    pub stems: Option<StemPaths>,
}

pub struct KeyResult {
    pub pitch_class: PitchClass,  // C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B
    pub mode: Mode,               // Major, Minor
    pub camelot: String,          // "1A" through "12B"
}

pub enum PitchClass { C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B }
pub enum Mode { Major, Minor }

pub struct StemPaths {
    pub vocals: PathBuf,
    pub drums: PathBuf,
    pub bass: PathBuf,
    pub other: PathBuf,
}
```

---

## Module Structure

Current directory structure:

```
src/
├── main.rs                 # Entry point, CLI setup
├── lib.rs                  # Public API, re-exports
├── error.rs                # Error types (thiserror)
├── types.rs                # Core data types (AnalyzedTrack, BpmResult, KeyResult, etc.)
│
├── config/
│   ├── mod.rs
│   └── settings.rs         # Settings struct, CLI parsing
│
├── discovery/
│   └── mod.rs              # walkdir file discovery, track ID generation
│
├── audio/
│   ├── mod.rs
│   └── decoder.rs          # symphonia decoding (22050Hz mono + 44100Hz stereo)
│
├── analysis/
│   ├── mod.rs
│   ├── traits.rs           # BpmDetector, KeyDetector, StemSeparator traits
│   ├── stratum.rs          # stratum-dsp BPM/Key implementation
│   ├── camelot.rs          # Key to Camelot notation mapping
│   ├── metadata.rs         # ID3/Vorbis tag extraction via lofty
│   └── stems/
│       ├── mod.rs
│       ├── model.rs        # ONNX model download/cache
│       ├── separator.rs    # OrtStemSeparator - ort session management
│       ├── stft.rs         # STFT/ISTFT preprocessing (rustfft)
│       └── chunking.rs     # Audio chunking with overlap-add
│
├── pipeline/
│   ├── mod.rs
│   └── orchestrator.rs     # Batch processing with bounded channel queue
│
└── export/
    ├── mod.rs
    ├── rekordbox.rs        # XML generation (streaming)
    └── json.rs             # JSON sidecar export
```

---

## Implementation Phases

### Phase 1: Walking Skeleton ✅ COMPLETE
**Goal:** End-to-end data flow with hardcoded values

- CLI with clap, file discovery with walkdir
- symphonia decoding, basic XML output

### Phase 2: Real Analysis ✅ COMPLETE
**Goal:** Accurate BPM and key detection

- stratum-dsp 1.0 integration (replaced bliss-audio due to FFI issues)
- Camelot mapping, TrackID generation (FNV-1a)
- Genre-aware tempo correction

### Phase 3: Batch Processing ✅ COMPLETE
**Goal:** Efficient parallel processing

- rayon parallel iteration
- indicatif progress bars
- JSON sidecar export

### Phase 4: Stem Separation ✅ COMPLETE
**Goal:** HTDemucs integration

- ort 2.0 session with EP fallback (CoreML → DirectML → CPU)
- STFT preprocessing with rustfft (nfft=4096, hop_length=1024)
- 7.8-second chunking with 1-second overlap
- Overlap-add reconstruction with linear crossfade
- Bounded channel queue (capacity=4) with dedicated stem worker thread
- WAV output via hound crate

### Phase 5: Polish ✅ COMPLETE
**Goal:** Production ready

1. ✅ The "djprep_import_" playlist workaround for Rekordbox bug
2. ✅ Metadata extraction (ID3/Vorbis tags via lofty)
3. ✅ Integration tests with sample audio (9 tests)
4. ✅ Improved error messages with actionable suggestions
5. ✅ Model auto-discovery (checks 5 common locations)
6. ✅ Dry-run mode (--dry-run flag)
7. ✅ Incremental analysis (--force flag to re-analyze)

---

## Critical Implementation Details

### TrackID Generation

Rekordbox uses signed 32-bit integers. Generate deterministically:

```rust
use std::hash::{Hash, Hasher};
use hash32::{FnvHasher, Hasher as Hash32Hasher};

fn generate_track_id(path: &Path) -> i32 {
    let normalized = normalize_path(path);
    let mut hasher = FnvHasher::default();
    hasher.write(normalized.as_bytes());
    let hash = hasher.finish32();
    (hash & 0x7FFFFFFF) as i32  // Ensure positive
}

fn normalize_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    let s = s.replace('\\', "/");
    let s = s.to_lowercase();
    // Strip trailing slashes
    s.trim_end_matches('/').to_string()
}
```

### URI Encoding for Rekordbox

```rust
use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};

const ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'`')
    .add(b'?')
    .add(b'{')
    .add(b'}')
    .add(b'[')
    .add(b']');

fn path_to_rekordbox_uri(path: &Path) -> String {
    let path_str = path.to_string_lossy();
    let normalized = path_str.replace('\\', "/");
    
    // Add leading slash for Windows drive letters
    let normalized = if normalized.chars().nth(1) == Some(':') {
        format!("/{}", normalized)
    } else {
        normalized.to_string()
    };
    
    // Encode each path segment separately (preserve slashes)
    let encoded: String = normalized
        .split('/')
        .map(|seg| utf8_percent_encode(seg, ENCODE_SET).to_string())
        .collect::<Vec<_>>()
        .join("/");
    
    format!("file://localhost{}", encoded)
}
```

### Camelot Wheel Mapping

```rust
fn to_camelot(pitch_class: PitchClass, mode: Mode) -> String {
    use PitchClass::*;
    use Mode::*;
    
    match (pitch_class, mode) {
        (C, Major) => "8B",   (C, Minor) => "5A",
        (Cs, Major) => "3B",  (Cs, Minor) => "12A",
        (D, Major) => "10B",  (D, Minor) => "7A",
        (Ds, Major) => "5B",  (Ds, Minor) => "2A",
        (E, Major) => "12B",  (E, Minor) => "9A",
        (F, Major) => "7B",   (F, Minor) => "4A",
        (Fs, Major) => "2B",  (Fs, Minor) => "11A",
        (G, Major) => "9B",   (G, Minor) => "6A",
        (Gs, Major) => "4B",  (Gs, Minor) => "1A",
        (A, Major) => "11B",  (A, Minor) => "8A",
        (As, Major) => "6B",  (As, Minor) => "3A",
        (B, Major) => "1B",   (B, Minor) => "10A",
    }.to_string()
}
```

### Rekordbox XML Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="djprep" Version="0.1.0" Company=""/>
  <COLLECTION Entries="2">
    <TRACK TrackID="123456" Name="Track Name" Artist="Artist" 
           Album="" Kind="MP3 File" Size="10485760"
           TotalTime="300" AverageBpm="128.00" 
           Tonality="11B" Location="file://localhost/path/to/track.mp3"
           DateAdded="2024-01-15"/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Type="1" Name="djprep_import_20240115" KeyType="0" Entries="2">
        <TRACK Key="123456"/>
        <TRACK Key="789012"/>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
```

**IMPORTANT:** Always generate the `djprep_import_*` playlist. This is a workaround for a Rekordbox bug where direct XML import doesn't update existing tracks. Users must import via the playlist.

### Concurrency Model

```
Discovery (single thread)
    │
    ▼
Analysis Pool (N-2 threads, rayon)
    │
    ├──► [BPM/Key only tracks] ──► Results Channel ──► Export Thread
    │
    └──► [Stems requested] ──► Bounded Channel (cap=2) ──► Stem Worker (1 thread)
                                                              │
                                                              ▼
                                                         Results Channel ──► Export Thread
```

The bounded channel between Analysis and Stems provides backpressure. When GPU is busy, analysis naturally slows.

### ort Execution Provider Selection

Try in order: CUDA → CoreML → DirectML → CPU

```rust
fn create_stem_session(model_path: &Path) -> Result<ort::Session> {
    let builder = ort::Session::builder()?;
    
    // Try CUDA first (NVIDIA)
    #[cfg(feature = "cuda")]
    if let Ok(session) = builder.clone()
        .with_execution_providers([ort::CUDAExecutionProvider::default().build()])?
        .commit_from_file(model_path) {
        info!("Using CUDA for stem separation");
        return Ok(session);
    }
    
    // Try CoreML (Apple Silicon)
    #[cfg(target_os = "macos")]
    if let Ok(session) = builder.clone()
        .with_execution_providers([ort::CoreMLExecutionProvider::default().build()])?
        .commit_from_file(model_path) {
        info!("Using CoreML for stem separation");
        return Ok(session);
    }
    
    // Fallback to CPU
    info!("Using CPU for stem separation (this will be slow)");
    builder.commit_from_file(model_path)
}
```

---

## Cargo.toml

```toml
[package]
name = "djprep"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
description = "High-performance audio analysis for DJs - BPM, Key, and Stem separation"
license = "MIT"

[features]
default = []
stems = ["dep:ort", "dep:reqwest", "dep:sha2", "dep:hex", "dep:rustfft", "dep:ndarray"]

[dependencies]
# CLI & UX
clap = { version = "4", features = ["derive"] }
indicatif = { version = "0.17", features = ["rayon"] }

# Audio decoding
symphonia = { version = "0.5", features = ["mp3", "flac", "wav", "aiff", "pcm"] }

# Analysis - stratum-dsp for BPM/key detection (pure Rust, no FFI)
stratum-dsp = "1.0"

# Metadata extraction
lofty = "0.18"

# Stem separation (optional, feature-gated)
ort = { version = "2.0.0-rc.10", optional = true }
rustfft = { version = "6.2", optional = true }
ndarray = { version = "0.16", optional = true }

# Concurrency
rayon = "1.10"
crossbeam-channel = "0.5"
num_cpus = "1.16"

# Serialization
quick-xml = { version = "0.37", features = ["serialize"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utilities
thiserror = "2"
anyhow = "1"
walkdir = "2"
percent-encoding = "2"
chrono = { version = "0.4", features = ["serde"] }
directories = "5"
hash32 = "0.3"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Audio writing (for stems)
hound = "3.5"

# Model download (optional, for stems)
reqwest = { version = "0.12", features = ["blocking", "rustls-tls"], default-features = false, optional = true }
sha2 = { version = "0.10", optional = true }
hex = { version = "0.4", optional = true }

[dev-dependencies]
tempfile = "3"

[profile.release]
lto = true
strip = true
panic = "abort"
codegen-units = 1
```

---

## CLI Interface

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

EXAMPLES:
    djprep -i ./music -o ./analyzed
    djprep -i ./music --stems --genre dnb -o ./analyzed
    djprep -i ./music -o ./out --dry-run
    djprep -i ./music -o ./out --force
```

---

## Error Handling Strategy

Use `thiserror` for library errors, `anyhow` in main for ergonomics:

```rust
// src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum DjprepError {
    #[error("Failed to decode audio: {path}")]
    DecodeError {
        path: PathBuf,
        #[source]
        source: symphonia::core::errors::Error,
    },
    
    #[error("Analysis failed for {path}: {reason}")]
    AnalysisError {
        path: PathBuf,
        reason: String,
    },
    
    #[error("Stem separation failed: {0}")]
    StemError(String),
    
    #[error("Failed to write output: {0}")]
    OutputError(#[from] std::io::Error),
    
    #[error("Model not found and download failed: {0}")]
    ModelError(String),
}

// Per-file errors are collected, not fatal
pub struct BatchResult {
    pub succeeded: Vec<AnalysisResult>,
    pub failed: Vec<(PathBuf, DjprepError)>,
}
```

---

## Testing Approach

### Unit tests (no audio files needed)
- URI encoding edge cases
- Camelot mapping correctness
- TrackID generation determinism
- XML escaping

### Integration tests (need small fixtures)
- Create 1-second test WAV files programmatically
- Verify pipeline produces valid XML

### Manual testing
- Use real tracks with known BPM/key
- Compare against Rekordbox/Mixed In Key output

---

## Next Steps (Post v0.1.0)

All core functionality is complete. Future enhancements:

### Priority 1: Additional Export Formats

```
Add support for Serato and Traktor export formats.

Research needed:
- Serato DJ crate format (.crate files)
- Traktor NML format
- What metadata each software expects

Implementation:
- New export modules: src/export/serato.rs, src/export/traktor.rs
- CLI flag: --format rekordbox|serato|traktor|all
```

### Priority 2: Automatic Model Download

```
Automatically download HTDemucs model when needed.

Considerations:
- Model is ~170MB, need progress bar
- Should verify SHA256 hash
- Store in user cache directory
- Handle download failures gracefully
```

### Priority 3: Waveform Generation

```
Generate waveform data for visual display.

Options:
- Peak amplitude per segment
- RMS levels
- Frequency bands (low/mid/high)
- Output as JSON or binary format
```

### Priority 4: Performance Optimization

```
Benchmark and optimize for large libraries (10,000+ tracks).

Areas to investigate:
- Memory usage during batch processing
- Parallel I/O for file reading
- Caching intermediate results
```

---

## Notes for Claude Code

### Do's
1. **Read CLAUDE.md first** - This file contains all architectural decisions
2. **Test incrementally** - `cargo run` and `cargo check` after each change
3. **Use existing patterns** - Look at StratumBpmDetector in src/analysis/stratum.rs
4. **Check compilation** - Run `cargo check` before claiming code is complete
5. **Handle errors** - Use `?` operator, don't unwrap() in library code
6. **Feature-gate stem code** - Use `#[cfg(feature = "stems")]` for ort/rustfft code

### Don'ts
1. **Don't refactor working code** unless asked - Focus on the task at hand
2. **Don't add dependencies** without checking if they're needed
3. **Don't ignore the trait abstractions** - They exist for swappability
4. **Don't block rayon threads** with stem separation - Use the bounded channel queue

### Debugging Commands
```bash
# Check compilation (default features)
cargo check

# Check with stems feature
cargo check --features stems

# Run with verbose logging
RUST_LOG=debug cargo run -- -i ./test -o ./out

# Run with stems enabled
DJPREP_MODEL_PATH=/path/to/model.onnx cargo run --features stems -- -i ./test -o ./out --stems

# Run all tests
cargo test

# Run tests with stems feature
cargo test --features stems

# Check for issues
cargo clippy
cargo clippy --features stems

# Format code
cargo fmt
```

### Key Files to Understand
```
src/analysis/traits.rs       # Trait definitions (BpmDetector, KeyDetector, StemSeparator)
src/analysis/stratum.rs      # BPM/Key detection implementation
src/analysis/stems/separator.rs  # Stem separation with ort
src/pipeline/orchestrator.rs # Main pipeline with bounded channel queue
src/types.rs                 # Data structures (AnalyzedTrack, BpmResult, KeyResult)
src/error.rs                 # Error types (DjprepError enum)
```

When stuck on Rust specifics (ownership, lifetimes, traits), ask for help. When stuck on architecture decisions, refer back to this document.
