//! Analysis trait abstractions
//!
//! These traits define the interface for swappable analysis backends.
//! Current implementation uses stratum-dsp for BPM/key detection.

use crate::error::Result;
use crate::types::{AudioBuffer, BpmResult, KeyResult, StemPaths};
use std::path::Path;

/// BPM detection backend
pub trait BpmDetector: Send + Sync {
    /// Detect BPM from audio samples
    fn detect(&self, buffer: &AudioBuffer) -> Result<BpmResult>;

    /// Get the name of this detector (for logging)
    fn name(&self) -> &'static str;
}

/// Musical key detection backend
pub trait KeyDetector: Send + Sync {
    /// Detect musical key from audio samples
    fn detect(&self, buffer: &AudioBuffer) -> Result<KeyResult>;

    /// Get the name of this detector (for logging)
    fn name(&self) -> &'static str;
}

/// Stem separation backend
pub trait StemSeparator: Send + Sync {
    /// Separate audio into stems (vocals, drums, bass, other)
    ///
    /// # Arguments
    /// * `input_path` - Path to the source audio file
    /// * `output_dir` - Directory to write stem files
    ///
    /// # Returns
    /// Paths to the generated stem files
    fn separate(&self, input_path: &Path, output_dir: &Path) -> Result<StemPaths>;

    /// Check if the separator is available (model loaded, GPU ready, etc.)
    fn is_available(&self) -> bool;

    /// Get the name of this separator (for logging)
    fn name(&self) -> &'static str;
}
