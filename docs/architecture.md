Intial research
Technical Architecture Report: djprep – High-Performance Audio Analysis & Metadata Interchange System
1. Executive Summary
This report presents a comprehensive architectural specification for djprep, a command-line interface (CLI) utility engineered to serve as a high-performance, open-source alternative to proprietary audio analysis tools such as "Mixed In Key." The primary objective of djprep is to perform batch analysis of digital audio files—specifically MP3, WAV, FLAC, and AIFF formats—to extract critical musical metadata including Beats Per Minute (BPM), Musical Key (Tonality), and Stem separation. A crucial requirement of this system is the capability to export this analyzed data into a standardized rekordbox.xml format, facilitating seamless integration with Pioneer DJ’s Rekordbox ecosystem, as well as a generic JSON format for interoperability with other systems.
The proposed architecture leverages the Rust programming language to prioritize memory safety, thread-level parallelism, and zero-cost abstractions. Central to the design is the utilization of symphonia for pure-Rust audio decoding, rayon for data parallelism, and ort (ONNX Runtime) for deep learning inference required by stem separation. The report conducts a rigorous analysis of the Rekordbox XML schema, identifying critical constraints regarding 32-bit signed integer limits for Track IDs and the idiosyncrasies of URI encoding across Windows and POSIX systems. Furthermore, it addresses the significant challenge of "Octave Errors" in BPM detection through heuristic signal processing strategies and evaluates the trade-offs between Hybrid Transformer Demucs (HTDemucs) and MDX-Net architectures for stem separation. By resolving resource contention between CPU-bound DSP tasks and tensor-based inference, djprep aims to deliver professional-grade analysis speed and accuracy in a portable, single-binary distribution.
2. Introduction to the Problem Space
The professional DJ software market is characterized by a dichotomy between standardized hardware interfaces and fragmented library management utilities. While Rekordbox serves as the de facto standard for library management due to its integration with Pioneer DJ hardware (CDJs and XDJs), its internal analysis algorithms—specifically for musical key detection and downbeat analysis—are often viewed by professionals as inferior to specialized third-party tools like Mixed In Key.
2.1 The Legacy of "Mixed In Key"
Mixed In Key established its dominance by introducing the "Camelot Wheel" notation system, a user-friendly abstraction of the Circle of Fifths that simplifies harmonic mixing for DJs without formal music theory training. However, relying on proprietary, closed-source software introduces several friction points:
 * Cost and Licensing: The software is paid and tied to specific licensing servers.
 * Opacity: The analysis algorithms are black boxes, offering users no control over parameters such as BPM range priors or stem separation model selection.
 * Platform Dependency: Reliance on specific OS frameworks often delays support for new operating systems or hardware architectures (e.g., Apple Silicon).
2.2 The Open Source Mandate
The djprep initiative seeks to disrupt this monopoly by providing a free, open-source alternative built on the robust Rust ecosystem. The choice of Rust is strategic. Its ownership model ensures memory safety without a garbage collector, which is critical for real-time or near-real-time audio processing where inconsistent latency (jitter) is unacceptable. Furthermore, Rust’s rich ecosystem of crates, such as symphonia for decoding and rayon for parallelism, allows for the construction of a high-throughput ETL (Extract, Transform, Load) pipeline that can process thousands of high-fidelity audio tracks efficiently.
2.3 The Integration Challenge
The core engineering challenge for djprep is not merely the signal processing, but the reliable injection of this data into the Rekordbox database. Rekordbox utilizes a dual-database architecture: an encrypted, obfuscated SQLite database (master.db) for internal operations, and an XML interchange format (rekordbox.xml) for importing and exporting data. Since direct manipulation of master.db poses a high risk of database corruption and is liable to break with any minor software update from AlphaTheta (Pioneer DJ’s parent company), djprep must act as a "Bridge" application, generating compliant XML that Rekordbox can ingest.
3. Systems Architecture and Rust Engineering
The architectural foundation of djprep is built upon a modular, concurrent pipeline designed to maximize resource utilization on modern multi-core processors. The system creates a separation of concerns between audio decoding, digital signal processing (DSP), deep learning inference, and metadata serialization.
3.1 Core Crate Selection and Justification
The selection of crates (libraries) dictates the performance characteristics and portability of the final binary. The following table summarizes the core dependencies and the engineering justification for their selection.
Table 1: Core Crate Selection Matrix
| Functional Domain | Selected Crate | Alternative Considered | Justification for Selection |
|---|---|---|---|
| Audio Decoding | symphonia | ffmpeg-next, rodio | symphonia is a pure-Rust implementation. It eliminates the complexity of linking against system C-libraries (libavcodec, libavformat) which varies wildly across macOS, Linux, and Windows ("DLL Hell"). It provides safe, zero-copy access to audio frames, essential for high-throughput analysis. |
| BPM/Key Analysis | stratum-dsp 1.0 | bliss-audio, aubio-rs | stratum-dsp is a pure-Rust library purpose-built for DJ audio analysis. Unlike bliss-audio (which requires aubio FFI bindings), stratum-dsp has zero C dependencies, ensuring single-binary distribution. It provides accurate BPM detection with genre-aware tempo correction and Camelot key notation. |
| Parallelism | rayon | tokio, std::thread | djprep is primarily CPU-bound (DSP) and compute-bound (Inference), not I/O bound. rayon's work-stealing thread pool is optimized for data parallelism (processing a list of files) rather than tokio's async model which is optimized for waiting on network/disk I/O. rayon minimizes thread management overhead. |
| Progress/UX | indicatif | termion | indicatif provides thread-safe multi-bar rendering. This allows djprep to display a global progress bar (Total Files) alongside thread-local spinners (e.g., "Demixing Track A..."), improving user experience during long batch operations. |
| AI Inference | ort 2.0.0-rc.10 | tch-rs, tensorflow | ort provides Rust bindings for the ONNX Runtime. It allows for the use of pre-trained models (like HTDemucs) without requiring a Python environment. It supports multiple Execution Providers (CUDA, CoreML, DirectML) dynamically, adapting to the user's hardware. Version 2.0 provides improved API ergonomics and better tensor handling. |
| STFT Processing | rustfft 6.2 | realfft | HTDemucs requires STFT preprocessing outside the model. rustfft provides efficient FFT computation in pure Rust, used for computing spectrograms and inverse STFT for audio reconstruction. |
| Serialization | quick-xml | serde-xml-rs | rekordbox.xml files can exceed hundreds of megabytes. quick-xml offers a streaming writer that is significantly more performant and memory-efficient than DOM-based XML writers, preventing OOM errors on large libraries. |
| Hashing | hash32 | std::collections::hash_map | Used for generating stable, deterministic Track IDs. We require a 32-bit hash (FNV or Murmur3) to match Rekordbox's integer constraints, rather than the 64-bit SipHash used by std. |
3.2 The Concurrency Model: Resolving Resource Contention
A critical architectural risk in djprep is resource contention between the DSP pipeline and the Deep Learning pipeline.
 * The Scenario: The user initiates a batch process for 1,000 tracks with stem separation enabled.
 * The Conflict:
   * rayon defaults to a global thread pool size equal to the number of logical cores (N). It attempts to saturate these cores with MP3 decoding and FFT calculations.
   * ort (ONNX Runtime) manages its own intra-operator thread pool for tensor mathematics, also attempting to utilize multiple cores per inference session.
   * Result: If both run unchecked, the system creates N \times M threads, leading to massive context switching and cache thrashing, significantly degrading performance.
Proposed Solution: Thread Pool Isolation and Pipelining
djprep will implement a segmented pipeline architecture:
 * Discovery Stage: A single thread scans directories using walkdir, filtering for valid extensions and producing a Vec<PathBuf>.
 * Analysis Stage (Rayon Pool): This pool is configured to use N - K threads (where K is reserved for inference). It handles light tasks: Decoding, Key Detection (FFT), and BPM analysis.
 * Inference Stage (Sequential Queue): Deep learning inference is heavily memory bandwidth and VRAM constrained. Running 16 concurrent Stem separations will crash most consumer GPUs (VRAM OOM). Therefore, files flagged for stem separation are pushed to a Receiver channel. A dedicated consumer loop (running on the reserved K threads) processes stems sequentially or with very low parallelism (e.g., 2 concurrent jobs).
This architecture ensures that the CPU-bound FFT analysis does not starve the tensor operations, and the tensor operations do not exhaust system memory.
4. Reverse-Engineering the Rekordbox Ecosystem
To effectively replace Mixed In Key, djprep must generate XML files that are fully compliant with the undocumented and often idiosyncratic validation rules of Rekordbox. Failure to adhere to these rules results in "Import Failed" errors or, worse, data corruption where tracks are duplicated or metadata is lost.
4.1 The Rekordbox XML Schema Specification
The rekordbox.xml file is a hierarchical dataset rooted in DJ_PLAYLISTS. The most critical section is the COLLECTION node, which contains a list of TRACK elements. Each TRACK element possesses specific attributes that djprep must populate.
Table 2: Critical TRACK Attributes and Constraints
| Attribute | Type | Constraint Description |
|---|---|---|
| TrackID | int32 | Critical. A signed 32-bit integer. Must be unique within the XML. Rekordbox uses this as a primary key. Negative values are theoretically allowed by the type but practically reserved for internal use; djprep must generate positive integers. |
| Location | string | URI Encoded. Must start with file://localhost/. Windows paths require drive letter normalization. |
| Tonality | string | The musical key. While Rekordbox's analyzer outputs "Fm", the field accepts arbitrary UTF-8 strings, enabling Camelot notation injection (e.g., "4A"). |
| AverageBpm | float64 | The global tempo. Used to set the initial beat grid. |
| TotalTime | float64 | Duration in seconds. If this deviates significantly from the file's actual duration, Rekordbox may flag the track as corrupt. |
| DateAdded | string | Format: yyyy-mm-dd. Essential for sorting in the "Bridge" view. |
4.2 The "TrackID" Generation Problem
Rekordbox identifies tracks by TrackID. Unlike modern systems that use UUIDs (128-bit), Rekordbox relies on a signed 32-bit integer. This limited namespace (2^{31}-1 \approx 2.14 billion positive integers) presents a challenge for djprep.
The Issue: djprep is stateless. It does not have access to the user's internal master.db to check which IDs are already taken. If djprep generates an XML with TrackID="1001" and the user already has a track with ID 1001 in their collection, Rekordbox’s import behavior is complex:
 * If the Location matches, it updates the metadata (subject to the "Import Bug" discussed in 4.4).
 * If the Location differs, it treats it as a conflict or ignores the import.
Algorithmic Solution:
We must generate deterministic IDs based on invariant file properties to ensure that analyzing the same file twice produces the same ID.
 * Method: Stable Hashing of the Canonical Path.
 * Algorithm: We will use FNV-1a (Fowler-Noll-Vo) or Murmur3 via the hash32 crate. These algorithms are non-cryptographic but offer excellent distribution and performance for short strings (file paths).
 * Implementation Detail: To support cross-platform consistency (e.g., a library on an external drive moving between macOS and Windows), djprep must normalize the path string before hashing:
   * Convert all backslashes (\) to forward slashes (/).
   * Convert Windows drive letters to uppercase (or lowercase, provided consistency is maintained).
   * Hash the string.
   * Cast the resulting u32 to i32. If the result is negative, absolute it or shift it to the positive range to ensure compatibility with Rekordbox’s UI expectations.
 * Collision Probability: With a 32-bit space, the Birthday Paradox suggests a 50% probability of collision after \approx 77,000 items. While typical libraries are smaller (10k-50k), the risk is non-zero. djprep should implement a hash-probe strategy: if a collision is detected within the current batch of files, increment the ID. External collisions are handled by Rekordbox's import logic.
4.3 URI Encoding and OS Specifics
The Location attribute requires strict URI encoding that differs from standard web URLs.
 * Protocol: Must be file://localhost/.
 * Windows: C:\Music\Track.mp3 \rightarrow file://localhost/C:/Music/Track.mp3.
 * macOS: /Users/DJ/Music/Track.mp3 \rightarrow file://localhost/Users/DJ/Music/Track.mp3.
 * Escape Sequences:
   * Space \rightarrow %20
   * Ampersand (&) \rightarrow %26 (XML requires & be escaped as &amp; after URI encoding if passing it as text, but quick-xml handles the XML entity escaping).
   * Brackets `` and Parentheses () \rightarrow Must be URI encoded in some Rekordbox versions, though modern versions are more lenient.
 * Validation: djprep must ensure the path exists before writing to XML. A broken link in the XML results in an unplayable track upon import.
4.4 The "XML Import Bug" (Rekordbox 6 & 7)
Research indicates a persistent bug in Rekordbox versions 6 and 7 regarding XML imports.
 * Symptom: When a user selects "Update Collection" or drags tracks from the XML tree to the main Collection, Rekordbox ignores metadata updates (like Key or Comments) if the track already exists in the master.db. It only imports new tracks.
 * Implication: This renders djprep useless for updating existing libraries with new Camelot Keys.
 * The Workaround: The bug does not affect Playlist Imports. If the tracks are listed inside a <PLAYLIST> node in the XML, and the user right-clicks that playlist and selects "Import Playlist," Rekordbox forces a reconciliation of the metadata for all tracks in that playlist.
 * Architectural Requirement: djprep must not only generate a COLLECTION of tracks but also automatically generate a "Batch Import" playlist (e.g., named djprep_import_) containing all analyzed tracks. This guides the user toward the working import path.
5. Digital Signal Processing (DSP) Architecture
The core competency of djprep is the accuracy of its musical analysis. We prioritize a "Pure Rust" implementation to avoid the distribution complexity of FFI-bound libraries like aubio or librosa (Python).
5.1 Audio Decoding Pipeline
The decoding pipeline must convert various input formats into a uniform representation for analysis.
 * Engine: symphonia crate.
 * Target Format:
   * Sample Rate: Downsample to 22,050 Hz.
   * Justification: Key and BPM detection rely primarily on frequencies below 11kHz. Downsampling reduces the FFT size and memory bandwidth required by 50% without compromising analysis accuracy.
   * Channels: Mono summation (L+R). Stereo imaging is irrelevant for harmonic analysis.
5.2 Musical Key Detection (Tonality)
Key detection involves identifying the harmonic center (Tonic) and the scale (Major/Minor).
5.2.1 Algorithm: Krumhansl-Schmuckler
We will implement the standard Krumhansl-Schmuckler algorithm using Chroma Features.
 * Short-Time Fourier Transform (STFT):
   * Use rustfft or realfft.
   * Window Size: Large windows are required for frequency resolution. At 22,050 Hz, a window of 4096 samples provides a frequency resolution of \approx 5.3 Hz. This is sufficient to distinguish low bass notes (e.g., Low E1 is ~41Hz, F1 is ~43Hz).
 * Chromagram Generation:
   * Map magnitude spectrum bins to the 12 semi-tones (C, C#, D...).
   * Sum energy across octaves.
   * Apply a logarithmic frequency mapping to align linear FFT bins with the logarithmic musical scale.
 * Profile Correlation:
   * Compute the Pearson correlation coefficient between the track's summed chromagram and the 24 Krumhansl-Kessler reference profiles (idealized distributions of notes in each key).
   * The profile with the maximum correlation is the detected key.
5.2.2 Implementation: stratum-dsp
 * Selected Solution: stratum-dsp 1.0, a pure-Rust library purpose-built for DJ audio analysis.
 * Why stratum-dsp:
   * Zero C dependencies - ensures single-binary distribution without "DLL Hell"
   * Purpose-built for DJ workflows with Camelot wheel notation
   * Genre-aware BPM detection with double-tempo correction
   * Optimized for electronic music characteristics (bass-heavy content)
 * Alternatives Evaluated:
   * bliss-audio: Requires aubio FFI bindings, causing build complexity on Windows/macOS
   * aubio-rs: Requires system libaubio installation, violating single-binary goal
   * Custom implementation: Would require significant DSP expertise and testing
5.3 BPM Detection and "Double Tempo" Correction
BPM detection is notoriously prone to "Octave Errors," where a 140 BPM Dubstep track is detected as 70 BPM, or a 70 BPM Hip-Hop track is detected as 140 BPM.
5.3.1 Detection Algorithm
 * Onset Detection Function (ODF): We calculate the Spectral Flux—the positive change in magnitude spectrum between successive frames. This captures percussive transients (kick drums, snares).
 * Periodicity Estimation: We perform an Autocorrelation of the ODF. This reveals the dominant repeating intervals (lags) in the signal.
 * Peak Picking: The lags with the highest correlation coefficients correspond to the beat period (\tau). BPM = \frac{60 \cdot SR}{\tau}.
5.3.2 The "Double Tempo" Heuristic
Mathematically, a 140 BPM signal is periodic at 70 BPM (every second beat). To resolve this ambiguity, djprep will implement a multi-stage heuristic:
 * Genre Priors (Bayesian Approach):
   * The user can supply a hint via CLI: djprep --genre dnb.
   * This applies a Log-Gaussian Weighting to the autocorrelation. For dnb, the Gaussian is centered at 174 BPM. For house, 124 BPM. This penalizes valid mathematical peaks that fall outside the genre's stylistic norms.
 * Event Density Analysis:
   * If no genre is supplied, we analyze the density of onsets between beats.
   * Logic: A 70 BPM track typically has fewer 1/16th note subdivisions than a 140 BPM track. If we detect high spectral energy at frequencies corresponding to hi-hats (usually >5kHz) occurring on the 1/2 or 1/4 subdivisions of the estimated beat, we favor the higher tempo.
 * Three-Band Analysis:
   * We split the ODF into Low (Kick), Mid (Snare), and High (Hats).
   * Low frequency periodicity usually defines the "half-time" feel (70 BPM).
   * High frequency periodicity usually defines the "double-time" feel (140 BPM).
   * djprep will prioritize the "Mid" band (Snare) for the most stable tempo estimation, as snares typically land on beats 2 and 4 in 4/4 time.
6. AI & Stem Separation Architecture
A key differentiator for djprep is the integration of Deep Neural Networks (DNN) to separate tracks into Stems (Vocals, Drums, Bass, Other). This allows DJs to perform live remixes.
6.1 Model Architecture: HTDemucs vs. MDX-Net
 * HTDemucs (Hybrid Transformer Demucs): Uses a hybrid architecture of Convolutional Neural Networks (CNNs) and Transformers in the time domain.
   * Pros: Excellent phase coherence (essential for remixing) and minimal leakage.
   * Cons: Computationally expensive, larger model size.
 * MDX-Net: Operates in the frequency domain.
   * Pros: Often cleaner vocal separation.
   * Cons: Can introduce high-frequency "musical noise" artifacts.
Decision: djprep will utilize HTDemucs (v4) exported to ONNX format. The time-domain approach ensures that the sum of the stems equals the original track (perfect reconstruction), which is critical for DJ use cases where audio quality is paramount.
6.2 Inference Engine: ONNX Runtime (ort)
We use the ort crate to run the model.
 * Execution Providers (EP): ort allows djprep to be hardware-agnostic.
   * NVIDIA Users: djprep will try to load the CUDA EP.
   * macOS Users: The CoreML EP allows the model to run on the Apple Neural Engine (ANE), providing massive speedups over CPU inference.
   * Windows/Generic: DirectML or OpenVINO.
 * Distribution Strategy:
   * We cannot bundle the ~200MB model in the binary.
   * Dynamic Download: On the first run with --stems, djprep will download the quantized ONNX model from a repository (e.g., HuggingFace) to ~/.cache/djprep/models/.
   * Verification: The download is verified against a hardcoded SHA-256 hash to ensure integrity.
7. Output Formats and Interoperability
7.1 XML Serialization
We utilize quick-xml for writing the rekordbox.xml.
 * Streaming Writer: Unlike DOM-based parsers that build the entire tree in RAM, quick-xml allows us to stream TRACK events to disk. This is essential when exporting libraries with >50,000 tracks, which would otherwise consume gigabytes of RAM.
 * Camelot Mapping:
   * djprep calculates Key index (0-11) and Scale (Maj/Min).
   * Mapping Table:
     * 0 Major (C) -> "8B"
     * 0 Minor (Am) -> "8A"
     * ...
   * This string is written to the Tonality attribute.
7.2 JSON Interchange Format
To satisfy the project requirement for "Standardized JSON," djprep will export a djprep.json sidecar.
 * Schema:
{
  "version": "1.0",
  "tracks":
}

This JSON allows other tools (e.g., streaming overlays, set planners) to consume djprep data without parsing the complex Rekordbox XML.
8. Binary Size and Distribution
Rust binaries can become large, especially when linking against C++ runtimes like ort.
8.1 Linking Strategy
 * Linux: We should target x86_64-unknown-linux-musl for static linking of standard libraries, but ort often requires dynamic linking to glibc. We will likely need to distribute a tar.gz containing the binary and the libonnxruntime.so if static linking proves unstable.
 * Windows/macOS: We can utilize ort's download-binaries feature during the CI build process to fetch the correct dynamic libraries and bundle them in the release archive.
 * Static Linking ort: The ort crate supports a static feature. If enabled, it compiles the ONNX Runtime from source. This massively increases build time (30+ minutes) but results in a truly portable single binary. For the official release, we recommend Static Linking to prioritize user convenience ("It just works") over build server time.
8.2 Optimization Profiles
To minimize bloat:
 * LTO (Link Time Optimization): Enabled in Cargo.toml.
 * Strip Symbols: strip = true.
 * Panic Strategy: panic = "abort".
   These settings can reduce the binary size of a standard Rust app from ~50MB to ~5-10MB (excluding the ONNX model).
9. Implementation Status (January 2026)

The following components have been implemented as specified in this architecture:

9.1 Completed Components
 * Audio Decoding: symphonia with MP3, WAV, FLAC, AIFF support
 * BPM/Key Detection: stratum-dsp 1.0 integration (StratumBpmDetector, StratumKeyDetector)
 * Stem Separation: Full HTDemucs implementation via ort 2.0
   * STFT preprocessing with rustfft (nfft=4096, hop_length=1024)
   * 7.8-second chunking with 1-second overlap
   * Overlap-add reconstruction with linear crossfade
   * Execution provider fallback (CoreML → DirectML → CPU)
 * Concurrency: Bounded channel queue (capacity=4) with dedicated stem worker thread
 * Export: Rekordbox XML with streaming writer, JSON sidecar
 * CLI: Full clap integration with all specified options

9.2 Implementation Files
 * src/analysis/stratum.rs - BPM/Key detection wrappers
 * src/analysis/stems/separator.rs - ORT session management and inference
 * src/analysis/stems/stft.rs - STFT/ISTFT implementation
 * src/analysis/stems/chunking.rs - Audio chunking and overlap-add
 * src/pipeline/orchestrator.rs - Concurrent pipeline with bounded channels

9.3 Remaining Work
 * Metadata Extraction: ID3/Vorbis tag reading (currently stubbed)
 * Integration Tests: End-to-end tests with sample audio
 * Model Distribution: Automatic download (infrastructure exists, needs activation)

10. Conclusion
djprep is architected to fill a significant void in the DJ software ecosystem. By combining the safety and concurrency of Rust with advanced DSP and Deep Learning, it offers a robust, free alternative to closed-source legacy tools. The system is designed to be resilient to the quirks of the Rekordbox ecosystem—specifically addressing the ID generation limits, URI encoding rules, and import bugs—while providing professional-grade analysis features like "Double Tempo" correction and Stem separation. This report provides the complete blueprint for implementation, ensuring that the final product is not just a technical curiosity, but a viable production tool for working DJs.
