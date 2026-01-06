//! Stratum-DSP based audio analysis
//!
//! This module provides BPM and key detection using the stratum-dsp library,
//! a pure-Rust implementation designed specifically for DJ applications.

use crate::analysis::key::camelot;
use crate::analysis::traits::{BpmDetector, KeyDetector};
use crate::error::{DjprepError, Result};
use crate::types::{AudioBuffer, BpmResult, KeyResult, Mode, PitchClass};
use stratum_dsp::{analyze_audio, AnalysisConfig, Key};
use tracing::debug;

/// BPM detector using stratum-dsp
///
/// Uses autocorrelation and comb filterbank analysis for accurate tempo detection.
pub struct StratumBpmDetector {
    /// Genre hint for double-tempo correction (not yet used by stratum-dsp)
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
            "Analyzing BPM with stratum-dsp ({} samples, {}Hz)",
            buffer.len(),
            buffer.sample_rate
        );

        let config = AnalysisConfig::default();

        let result = analyze_audio(&buffer.samples, buffer.sample_rate, config).map_err(|e| {
            DjprepError::AnalysisError {
                path: std::path::PathBuf::new(),
                reason: format!("BPM analysis failed: {}", e),
            }
        })?;

        let bpm = result.bpm as f64;
        let confidence = result.bpm_confidence as f64;

        debug!("Detected BPM: {:.2} (confidence: {:.2})", bpm, confidence);

        // Generate candidate tempos (half and double time)
        let candidates = vec![
            (bpm / 2.0, confidence * 0.5),
            (bpm * 2.0, confidence * 0.5),
        ];

        Ok(BpmResult {
            value: bpm,
            confidence,
            candidates,
        })
    }

    fn name(&self) -> &'static str {
        "stratum-dsp"
    }
}

/// Key detector using stratum-dsp
///
/// Uses chroma-based analysis with template matching for key detection.
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
            "Analyzing key with stratum-dsp ({} samples, {}Hz)",
            buffer.len(),
            buffer.sample_rate
        );

        let config = AnalysisConfig::default();

        let result = analyze_audio(&buffer.samples, buffer.sample_rate, config).map_err(|e| {
            DjprepError::AnalysisError {
                path: std::path::PathBuf::new(),
                reason: format!("Key analysis failed: {}", e),
            }
        })?;

        // Convert stratum-dsp Key to our types
        let (pitch_class, mode) = match result.key {
            Key::Major(pitch_idx) => {
                let pitch = PitchClass::from_index(pitch_idx as u8).unwrap_or(PitchClass::C);
                (pitch, Mode::Major)
            }
            Key::Minor(pitch_idx) => {
                let pitch = PitchClass::from_index(pitch_idx as u8).unwrap_or(PitchClass::C);
                (pitch, Mode::Minor)
            }
        };

        let camelot_str = camelot::to_camelot(pitch_class, mode).to_string();
        let open_key_str = camelot::to_open_key(pitch_class, mode).to_string();
        let confidence = result.key_confidence as f64;

        debug!(
            "Detected key: {:?} {:?} ({}) (confidence: {:.2})",
            pitch_class, mode, camelot_str, confidence
        );

        Ok(KeyResult {
            pitch_class,
            mode,
            camelot: camelot_str,
            open_key: open_key_str,
            confidence,
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
        let detector = StratumKeyDetector::default();
        assert_eq!(detector.name(), "stratum-dsp");
    }
}
