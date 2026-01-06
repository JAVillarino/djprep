//! Camelot Wheel and Open Key notation mapping
//!
//! The Camelot Wheel is a visual representation of musical keys that
//! makes harmonic mixing intuitive for DJs.
//!
//! - Numbers 1-12 represent positions on the wheel
//! - 'A' suffix = minor key, 'B' suffix = major key
//! - Adjacent numbers are harmonically compatible (perfect fifth)
//! - Same number, different letter = relative major/minor

use crate::types::{Mode, PitchClass};

/// Mapping from (PitchClass, Mode) to Camelot notation
///
/// Layout:
/// ```text
///      5A      5B
///    /    \  /    \
///  4A      4B      6B  
///  |       |       |
///  3A      3B      7B
///    \    /  \    /
///      2A      8B
///       ...
/// ```
pub fn to_camelot(pitch: PitchClass, mode: Mode) -> &'static str {
    match (pitch, mode) {
        // Minor keys (A)
        (PitchClass::A, Mode::Minor) => "8A",   // Am
        (PitchClass::As, Mode::Minor) => "3A",  // A#m / Bbm
        (PitchClass::B, Mode::Minor) => "10A",  // Bm
        (PitchClass::C, Mode::Minor) => "5A",   // Cm
        (PitchClass::Cs, Mode::Minor) => "12A", // C#m / Dbm
        (PitchClass::D, Mode::Minor) => "7A",   // Dm
        (PitchClass::Ds, Mode::Minor) => "2A",  // D#m / Ebm
        (PitchClass::E, Mode::Minor) => "9A",   // Em
        (PitchClass::F, Mode::Minor) => "4A",   // Fm
        (PitchClass::Fs, Mode::Minor) => "11A", // F#m / Gbm
        (PitchClass::G, Mode::Minor) => "6A",   // Gm
        (PitchClass::Gs, Mode::Minor) => "1A",  // G#m / Abm

        // Major keys (B)
        (PitchClass::A, Mode::Major) => "11B",  // A
        (PitchClass::As, Mode::Major) => "6B",  // A# / Bb
        (PitchClass::B, Mode::Major) => "1B",   // B
        (PitchClass::C, Mode::Major) => "8B",   // C
        (PitchClass::Cs, Mode::Major) => "3B",  // C# / Db
        (PitchClass::D, Mode::Major) => "10B",  // D
        (PitchClass::Ds, Mode::Major) => "5B",  // D# / Eb
        (PitchClass::E, Mode::Major) => "12B",  // E
        (PitchClass::F, Mode::Major) => "7B",   // F
        (PitchClass::Fs, Mode::Major) => "2B",  // F# / Gb
        (PitchClass::G, Mode::Major) => "9B",   // G
        (PitchClass::Gs, Mode::Major) => "4B",  // G# / Ab
    }
}

/// Mapping from (PitchClass, Mode) to Open Key notation
///
/// Open Key uses:
/// - Numbers 1-12 (same positions as Camelot)
/// - 'd' suffix = major (dur), 'm' suffix = minor (moll)
pub fn to_open_key(pitch: PitchClass, mode: Mode) -> &'static str {
    match (pitch, mode) {
        // Minor keys (m)
        (PitchClass::A, Mode::Minor) => "8m",
        (PitchClass::As, Mode::Minor) => "3m",
        (PitchClass::B, Mode::Minor) => "10m",
        (PitchClass::C, Mode::Minor) => "5m",
        (PitchClass::Cs, Mode::Minor) => "12m",
        (PitchClass::D, Mode::Minor) => "7m",
        (PitchClass::Ds, Mode::Minor) => "2m",
        (PitchClass::E, Mode::Minor) => "9m",
        (PitchClass::F, Mode::Minor) => "4m",
        (PitchClass::Fs, Mode::Minor) => "11m",
        (PitchClass::G, Mode::Minor) => "6m",
        (PitchClass::Gs, Mode::Minor) => "1m",

        // Major keys (d)
        (PitchClass::A, Mode::Major) => "11d",
        (PitchClass::As, Mode::Major) => "6d",
        (PitchClass::B, Mode::Major) => "1d",
        (PitchClass::C, Mode::Major) => "8d",
        (PitchClass::Cs, Mode::Major) => "3d",
        (PitchClass::D, Mode::Major) => "10d",
        (PitchClass::Ds, Mode::Major) => "5d",
        (PitchClass::E, Mode::Major) => "12d",
        (PitchClass::F, Mode::Major) => "7d",
        (PitchClass::Fs, Mode::Major) => "2d",
        (PitchClass::G, Mode::Major) => "9d",
        (PitchClass::Gs, Mode::Major) => "4d",
    }
}

/// Get harmonically compatible keys (for mixing suggestions)
///
/// Returns keys that are safe to mix with the given key:
/// - Same key
/// - +1/-1 on the wheel (perfect fifth relationship)
/// - Same number, opposite letter (relative major/minor)
pub fn compatible_keys(camelot: &str) -> Vec<&'static str> {
    // Parse the Camelot code
    let (num, letter) = parse_camelot(camelot);
    if num == 0 {
        return vec![];
    }

    let mut compatible = Vec::with_capacity(4);

    // Same key (using static lookup instead of input reference)
    compatible.push(camelot_string(num, letter));

    // +1 on wheel (wraps 12 -> 1)
    let plus_one = if num == 12 { 1 } else { num + 1 };
    // -1 on wheel (wraps 1 -> 12)
    let minus_one = if num == 1 { 12 } else { num - 1 };

    // Relative major/minor (same number, different letter)
    let relative_letter = if letter == 'A' { 'B' } else { 'A' };

    compatible.push(camelot_string(plus_one, letter));
    compatible.push(camelot_string(minus_one, letter));
    compatible.push(camelot_string(num, relative_letter));

    compatible
}

fn parse_camelot(code: &str) -> (u8, char) {
    if code.len() < 2 {
        return (0, ' ');
    }
    let letter = code.chars().last().unwrap_or(' ');
    let num_str: String = code.chars().take_while(|c| c.is_ascii_digit()).collect();
    let num: u8 = num_str.parse().unwrap_or(0);
    (num, letter)
}

fn camelot_string(num: u8, letter: char) -> &'static str {
    match (num, letter) {
        (1, 'A') => "1A", (1, 'B') => "1B",
        (2, 'A') => "2A", (2, 'B') => "2B",
        (3, 'A') => "3A", (3, 'B') => "3B",
        (4, 'A') => "4A", (4, 'B') => "4B",
        (5, 'A') => "5A", (5, 'B') => "5B",
        (6, 'A') => "6A", (6, 'B') => "6B",
        (7, 'A') => "7A", (7, 'B') => "7B",
        (8, 'A') => "8A", (8, 'B') => "8B",
        (9, 'A') => "9A", (9, 'B') => "9B",
        (10, 'A') => "10A", (10, 'B') => "10B",
        (11, 'A') => "11A", (11, 'B') => "11B",
        (12, 'A') => "12A", (12, 'B') => "12B",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camelot_mapping_covers_all_keys() {
        // Ensure all 24 key combinations map to unique Camelot codes
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
        // Common DJ reference points
        assert_eq!(to_camelot(PitchClass::A, Mode::Minor), "8A");
        assert_eq!(to_camelot(PitchClass::C, Mode::Major), "8B");
        assert_eq!(to_camelot(PitchClass::G, Mode::Minor), "6A");
    }

    #[test]
    fn test_compatible_keys() {
        let compatible = compatible_keys("8A"); // Am
        assert!(compatible.contains(&"8A"));  // Same
        assert!(compatible.contains(&"7A"));  // -1
        assert!(compatible.contains(&"9A"));  // +1
        assert!(compatible.contains(&"8B"));  // Relative major (C)
    }

    #[test]
    fn test_compatible_keys_wrap() {
        let compatible = compatible_keys("12A");
        assert!(compatible.contains(&"1A"));  // Wraps 12 -> 1
        assert!(compatible.contains(&"11A")); // -1

        let compatible = compatible_keys("1B");
        assert!(compatible.contains(&"12B")); // Wraps 1 -> 12
        assert!(compatible.contains(&"2B"));  // +1
    }
}
