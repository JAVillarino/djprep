//! Stratum-DSP based audio analysis.
//!
//! This module now delegates analysis math to `djprep-core` and keeps the
//! CLI-facing trait implementations/path-aware errors.

use crate::analysis::traits::{BpmDetector, KeyDetector};
use crate::error::{DjprepError, Result};
use crate::types::{AudioBuffer, BpmResult, KeyResult};
use tracing::debug;

/// BPM detector using djprep-core (stratum-dsp backend).
pub struct StratumBpmDetector {
    /// Genre hint for double-tempo correction (not yet used by stratum-dsp).
    #[allow(dead_code)]
    genre_hint: Option<String>,
}

impl StratumBpmDetector {
    pub fn new(genre_hint: Option<String>) -> Self {
        Self { genre_hint }
    }
}

impl Default for StratumBpmDetector {
    fn default() -> Self {
        Self::new(None)
    }
}

impl BpmDetector for StratumBpmDetector {
    fn detect(&self, buffer: &AudioBuffer) -> Result<BpmResult> {
        debug!(
            "Analyzing BPM with djprep-core/stratum-dsp ({} samples, {}Hz)",
            buffer.len(),
            buffer.sample_rate
        );

        djprep_core::detect_bpm(buffer).map_err(|e| DjprepError::AnalysisError {
            path: std::path::PathBuf::new(),
            reason: e.to_string(),
        })
    }

    fn name(&self) -> &'static str {
        "stratum-dsp"
    }
}

/// Key detector using djprep-core (stratum-dsp backend).
pub struct StratumKeyDetector;

impl StratumKeyDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StratumKeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyDetector for StratumKeyDetector {
    fn detect(&self, buffer: &AudioBuffer) -> Result<KeyResult> {
        debug!(
            "Analyzing key with djprep-core/stratum-dsp ({} samples, {}Hz)",
            buffer.len(),
            buffer.sample_rate
        );

        djprep_core::detect_key(buffer).map_err(|e| DjprepError::AnalysisError {
            path: std::path::PathBuf::new(),
            reason: e.to_string(),
        })
    }

    fn name(&self) -> &'static str {
        "stratum-dsp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratum_bpm_detector_name() {
        let detector = StratumBpmDetector::default();
        assert_eq!(detector.name(), "stratum-dsp");
    }

    #[test]
    fn test_stratum_key_detector_name() {
        let detector = StratumKeyDetector;
        assert_eq!(detector.name(), "stratum-dsp");
    }
}
