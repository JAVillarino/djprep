//! BPM detection module

use crate::analysis::traits::BpmDetector;
use crate::error::Result;
use crate::types::{AudioBuffer, BpmResult};

/// Placeholder BPM detector (fallback only)
///
/// The actual implementation uses StratumBpmDetector in stratum.rs
pub struct PlaceholderBpmDetector {
    /// Genre hint for resolving double-tempo ambiguity
    #[allow(dead_code)]
    genre_hint: Option<String>,
}

impl PlaceholderBpmDetector {
    pub fn new(genre_hint: Option<String>) -> Self {
        Self { genre_hint }
    }
}

impl Default for PlaceholderBpmDetector {
    fn default() -> Self {
        Self::new(None)
    }
}

impl BpmDetector for PlaceholderBpmDetector {
    fn detect(&self, buffer: &AudioBuffer) -> Result<BpmResult> {
        // Placeholder: Return a deterministic BPM based on audio characteristics
        // This will be replaced with actual onset detection + autocorrelation

        // Use buffer duration to generate a pseudo-random but deterministic BPM
        // Real music typically ranges from 70-180 BPM
        let duration_hash = (buffer.duration * 1000.0) as usize;
        let bpm_offset = (duration_hash % 110) as f64; // 0-109
        let bpm = 70.0 + bpm_offset;

        Ok(BpmResult {
            value: bpm,
            confidence: 0.0, // Zero confidence indicates placeholder
            candidates: vec![
                (bpm / 2.0, 0.3),  // Half-time candidate
                (bpm * 2.0, 0.3),  // Double-time candidate
            ],
        })
    }

    fn name(&self) -> &'static str {
        "placeholder"
    }
}
