//! Core data types for djprep
//!
//! These types represent the domain model and flow through the pipeline.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// =============================================================================
// Musical primitives
// =============================================================================

/// The 12 pitch classes in Western music
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PitchClass {
    C,
    Cs, // C#/Db
    D,
    Ds, // D#/Eb
    E,
    F,
    Fs, // F#/Gb
    G,
    Gs, // G#/Ab
    A,
    As, // A#/Bb
    B,
}

impl PitchClass {
    /// Convert from numeric index (0 = C, 1 = C#, ..., 11 = B)
    pub fn from_index(index: u8) -> Option<Self> {
        match index % 12 {
            0 => Some(PitchClass::C),
            1 => Some(PitchClass::Cs),
            2 => Some(PitchClass::D),
            3 => Some(PitchClass::Ds),
            4 => Some(PitchClass::E),
            5 => Some(PitchClass::F),
            6 => Some(PitchClass::Fs),
            7 => Some(PitchClass::G),
            8 => Some(PitchClass::Gs),
            9 => Some(PitchClass::A),
            10 => Some(PitchClass::As),
            11 => Some(PitchClass::B),
            _ => None,
        }
    }

    /// Convert to numeric index (0 = C, 1 = C#, ..., 11 = B)
    pub fn to_index(self) -> u8 {
        match self {
            PitchClass::C => 0,
            PitchClass::Cs => 1,
            PitchClass::D => 2,
            PitchClass::Ds => 3,
            PitchClass::E => 4,
            PitchClass::F => 5,
            PitchClass::Fs => 6,
            PitchClass::G => 7,
            PitchClass::Gs => 8,
            PitchClass::A => 9,
            PitchClass::As => 10,
            PitchClass::B => 11,
        }
    }

    /// Standard notation (e.g., "C", "F#", "Bb")
    pub fn to_standard_notation(self) -> &'static str {
        match self {
            PitchClass::C => "C",
            PitchClass::Cs => "C#",
            PitchClass::D => "D",
            PitchClass::Ds => "D#",
            PitchClass::E => "E",
            PitchClass::F => "F",
            PitchClass::Fs => "F#",
            PitchClass::G => "G",
            PitchClass::Gs => "G#",
            PitchClass::A => "A",
            PitchClass::As => "A#",
            PitchClass::B => "B",
        }
    }
}

/// Major or Minor scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    Major,
    Minor,
}

// =============================================================================
// Analysis results
// =============================================================================

/// BPM analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpmResult {
    /// Primary detected BPM
    pub value: f64,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Alternative candidate tempos (value, confidence)
    pub candidates: Vec<(f64, f64)>,
}

impl Default for BpmResult {
    fn default() -> Self {
        Self {
            value: 120.0,
            confidence: 0.0,
            candidates: vec![],
        }
    }
}

/// Musical key analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyResult {
    /// Detected pitch class (C, C#, D, etc.)
    pub pitch_class: PitchClass,
    /// Major or Minor
    pub mode: Mode,
    /// Camelot notation ("1A" - "12B")
    pub camelot: String,
    /// Open Key notation ("1m" - "12d")
    pub open_key: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl Default for KeyResult {
    fn default() -> Self {
        Self {
            pitch_class: PitchClass::C,
            mode: Mode::Major,
            camelot: "8B".to_string(),
            open_key: "1d".to_string(),
            confidence: 0.0,
        }
    }
}

/// Paths to separated stem files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemPaths {
    pub vocals: PathBuf,
    pub drums: PathBuf,
    pub bass: PathBuf,
    pub other: PathBuf,
}

// =============================================================================
// Track representation
// =============================================================================

/// Metadata extracted from audio file tags
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub year: Option<i32>,
}

/// Complete analysis result for a single track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzedTrack {
    /// Deterministic ID derived from path (for Rekordbox)
    pub track_id: i32,
    /// Original file path
    pub path: PathBuf,
    /// File metadata from tags
    pub metadata: TrackMetadata,
    /// BPM analysis
    pub bpm: BpmResult,
    /// Key analysis
    pub key: KeyResult,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Sample rate of source file
    pub sample_rate: u32,
    /// Stem separation output paths (if enabled)
    pub stems: Option<StemPaths>,
    /// Timestamp of analysis
    pub analyzed_at: chrono::DateTime<chrono::Utc>,
}

impl AnalyzedTrack {
    /// Create a new AnalyzedTrack with default/placeholder values
    pub fn new(path: PathBuf, track_id: i32) -> Self {
        Self {
            track_id,
            path,
            metadata: TrackMetadata::default(),
            bpm: BpmResult::default(),
            key: KeyResult::default(),
            duration_seconds: 0.0,
            sample_rate: 44100,
            stems: None,
            analyzed_at: chrono::Utc::now(),
        }
    }
}

// =============================================================================
// Audio buffer types
// =============================================================================

/// Decoded audio samples ready for analysis
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Mono samples normalized to [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        // Guard against division by zero - use 0 duration for invalid sample rate
        let duration = if sample_rate > 0 {
            samples.len() as f64 / sample_rate as f64
        } else {
            0.0
        };
        Self {
            samples,
            sample_rate,
            duration,
        }
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Stereo audio buffer for stem separation (full fidelity)
#[derive(Debug, Clone)]
pub struct StereoBuffer {
    /// Left channel samples normalized to [-1.0, 1.0]
    pub left: Vec<f32>,
    /// Right channel samples normalized to [-1.0, 1.0]
    pub right: Vec<f32>,
    /// Sample rate in Hz (typically 44100)
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
}

impl StereoBuffer {
    pub fn new(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        let num_samples = left.len().min(right.len());
        // Guard against division by zero - use 0 duration for invalid sample rate
        let duration = if sample_rate > 0 {
            num_samples as f64 / sample_rate as f64
        } else {
            0.0
        };
        Self {
            left,
            right,
            sample_rate,
            duration,
        }
    }

    /// Number of samples per channel
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// Get interleaved samples [L, R, L, R, ...]
    pub fn interleaved(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.left.len() * 2);
        for (l, r) in self.left.iter().zip(self.right.iter()) {
            result.push(*l);
            result.push(*r);
        }
        result
    }

    /// Create from interleaved samples
    pub fn from_interleaved(samples: &[f32], sample_rate: u32) -> Self {
        let num_frames = samples.len() / 2;
        let mut left = Vec::with_capacity(num_frames);
        let mut right = Vec::with_capacity(num_frames);

        for chunk in samples.chunks(2) {
            if chunk.len() == 2 {
                left.push(chunk[0]);
                right.push(chunk[1]);
            }
        }

        Self::new(left, right, sample_rate)
    }
}

// =============================================================================
// Supported formats
// =============================================================================

/// Audio formats supported by djprep
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Aiff,
}

impl AudioFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(AudioFormat::Mp3),
            "wav" => Some(AudioFormat::Wav),
            "flac" => Some(AudioFormat::Flac),
            "aiff" | "aif" => Some(AudioFormat::Aiff),
            _ => None,
        }
    }

    /// Check if a path has a supported extension
    pub fn is_supported_path(path: &std::path::Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(Self::from_extension)
            .is_some()
    }
}
