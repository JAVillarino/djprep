//! Camelot Wheel and Open Key notation mapping.
//!
//! Delegates to `djprep-core` so CLI and core stay aligned.

use crate::types::{Mode, PitchClass};

/// Mapping from (PitchClass, Mode) to Camelot notation.
pub fn to_camelot(pitch: PitchClass, mode: Mode) -> &'static str {
    djprep_core::camelot::to_camelot(pitch, mode)
}

/// Mapping from (PitchClass, Mode) to Open Key notation.
pub fn to_open_key(pitch: PitchClass, mode: Mode) -> &'static str {
    djprep_core::camelot::to_open_key(pitch, mode)
}

/// Get harmonically compatible keys (for mixing suggestions).
pub fn compatible_keys(camelot: &str) -> Vec<&'static str> {
    djprep_core::camelot::compatible_keys(camelot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camelot_mapping_covers_all_keys() {
        let mut codes = std::collections::HashSet::new();

        for pitch_idx in 0..12 {
            let pitch = PitchClass::from_index(pitch_idx).unwrap();
            for mode in [Mode::Major, Mode::Minor] {
                let code = to_camelot(pitch, mode);
                assert!(!code.is_empty(), "Empty code for {:?} {:?}", pitch, mode);
                assert!(codes.insert(code), "Duplicate code: {}", code);
            }
        }

        assert_eq!(codes.len(), 24);
    }

    #[test]
    fn test_camelot_examples() {
        assert_eq!(to_camelot(PitchClass::A, Mode::Minor), "8A");
        assert_eq!(to_camelot(PitchClass::C, Mode::Major), "8B");
        assert_eq!(to_camelot(PitchClass::G, Mode::Minor), "6A");
    }

    #[test]
    fn test_compatible_keys() {
        let compatible = compatible_keys("8A");
        assert!(compatible.contains(&"8A"));
        assert!(compatible.contains(&"7A"));
        assert!(compatible.contains(&"9A"));
        assert!(compatible.contains(&"8B"));
    }

    #[test]
    fn test_compatible_keys_wrap() {
        let compatible = compatible_keys("12A");
        assert!(compatible.contains(&"1A"));
        assert!(compatible.contains(&"11A"));

        let compatible = compatible_keys("1B");
        assert!(compatible.contains(&"12B"));
        assert!(compatible.contains(&"2B"));
    }
}
