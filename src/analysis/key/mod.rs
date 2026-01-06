//! Key detection module

pub mod camelot;

use crate::analysis::traits::KeyDetector;
use crate::error::Result;
use crate::types::{AudioBuffer, KeyResult, Mode, PitchClass};

/// Placeholder key detector (fallback only)
///
/// The actual implementation uses StratumKeyDetector in stratum.rs
pub struct PlaceholderKeyDetector;

impl PlaceholderKeyDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PlaceholderKeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyDetector for PlaceholderKeyDetector {
    fn detect(&self, buffer: &AudioBuffer) -> Result<KeyResult> {
        // Placeholder: Return a deterministic key based on audio characteristics
        // This will be replaced with actual chromagram analysis
        
        // Use buffer length to generate a pseudo-random but deterministic result
        let hash = buffer.len() % 24;
        let pitch_idx = (hash % 12) as u8;
        let is_minor = hash >= 12;

        let pitch_class = PitchClass::from_index(pitch_idx).unwrap_or(PitchClass::C);
        let mode = if is_minor { Mode::Minor } else { Mode::Major };

        let camelot = camelot::to_camelot(pitch_class, mode).to_string();
        let open_key = camelot::to_open_key(pitch_class, mode).to_string();

        Ok(KeyResult {
            pitch_class,
            mode,
            camelot,
            open_key,
            confidence: 0.0, // Zero confidence indicates placeholder
        })
    }

    fn name(&self) -> &'static str {
        "placeholder"
    }
}
