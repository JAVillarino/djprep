//! Core data types for djprep.
//!
//! Analysis-domain primitives are provided by `djprep-core` and re-exported
//! here to minimize churn in CLI modules.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use djprep_core::{AudioBuffer, BpmResult, KeyResult, Mode, PitchClass};

// =============================================================================
// Track representation (CLI-side)
// =============================================================================

/// Paths to separated stem files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StemPaths {
    pub vocals: PathBuf,
    pub drums: PathBuf,
    pub bass: PathBuf,
    pub other: PathBuf,
}

/// Metadata extracted from audio file tags.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub genre: Option<String>,
    pub year: Option<i32>,
}

/// Complete analysis result for a single track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzedTrack {
    /// Deterministic ID derived from path (for Rekordbox).
    pub track_id: i32,
    /// Original file path.
    pub path: PathBuf,
    /// File metadata from tags.
    pub metadata: TrackMetadata,
    /// BPM analysis.
    pub bpm: BpmResult,
    /// Key analysis.
    pub key: KeyResult,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Sample rate of source file.
    pub sample_rate: u32,
    /// Stem separation output paths (if enabled).
    pub stems: Option<StemPaths>,
    /// Timestamp of analysis.
    pub analyzed_at: chrono::DateTime<chrono::Utc>,
}

impl AnalyzedTrack {
    /// Create a new AnalyzedTrack with default/placeholder values.
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
// Audio buffer types (CLI-side)
// =============================================================================

/// Stereo audio buffer for stem separation (full fidelity).
#[derive(Debug, Clone)]
pub struct StereoBuffer {
    /// Left channel samples normalized to [-1.0, 1.0].
    pub left: Vec<f32>,
    /// Right channel samples normalized to [-1.0, 1.0].
    pub right: Vec<f32>,
    /// Sample rate in Hz (typically 44100).
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration: f64,
}

impl StereoBuffer {
    pub fn new(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        let num_samples = left.len().min(right.len());
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

    /// Number of samples per channel.
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// Get interleaved samples [L, R, L, R, ...].
    pub fn interleaved(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.left.len() * 2);
        for (l, r) in self.left.iter().zip(self.right.iter()) {
            result.push(*l);
            result.push(*r);
        }
        result
    }

    /// Create from interleaved samples.
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

/// Audio formats supported by djprep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Aiff,
}

impl AudioFormat {
    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp3" => Some(AudioFormat::Mp3),
            "wav" => Some(AudioFormat::Wav),
            "flac" => Some(AudioFormat::Flac),
            "aiff" | "aif" => Some(AudioFormat::Aiff),
            _ => None,
        }
    }

    /// Check if a path has a supported extension.
    pub fn is_supported_path(path: &std::path::Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(Self::from_extension)
            .is_some()
    }
}
