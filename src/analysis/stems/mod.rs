//! Stem separation module
//!
//! Provides audio stem separation using HTDemucs via ONNX Runtime.
//! Separates tracks into: vocals, drums, bass, and other.

pub mod chunking;
pub mod model;
pub mod separator;
pub mod stft;

use crate::analysis::traits::StemSeparator;
use crate::error::{DjprepError, Result};
use crate::types::StemPaths;
use std::path::Path;

// Re-export the ORT-based separator
pub use separator::OrtStemSeparator;

/// Placeholder stem separator (fallback when stems feature is disabled)
pub struct PlaceholderStemSeparator {
    available: bool,
}

impl PlaceholderStemSeparator {
    pub fn new() -> Self {
        Self { available: false }
    }
}

impl Default for PlaceholderStemSeparator {
    fn default() -> Self {
        Self::new()
    }
}

impl StemSeparator for PlaceholderStemSeparator {
    fn separate(&self, input_path: &Path, _output_dir: &Path) -> Result<StemPaths> {
        Err(DjprepError::StemUnavailable {
            reason: format!(
                "Stem separation not available for '{}'. Build with --features stems",
                input_path.display()
            ),
        })
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "placeholder"
    }
}
