//! Camelot Wheel and Open Key notation mapping.

use crate::types::{Mode, PitchClass};

/// Mapping from (PitchClass, Mode) to Camelot notation.
pub fn to_camelot(pitch: PitchClass, mode: Mode) -> &'static str {
    match (pitch, mode) {
        (PitchClass::A, Mode::Minor) => "8A",
        (PitchClass::As, Mode::Minor) => "3A",
        (PitchClass::B, Mode::Minor) => "10A",
        (PitchClass::C, Mode::Minor) => "5A",
        (PitchClass::Cs, Mode::Minor) => "12A",
        (PitchClass::D, Mode::Minor) => "7A",
        (PitchClass::Ds, Mode::Minor) => "2A",
        (PitchClass::E, Mode::Minor) => "9A",
        (PitchClass::F, Mode::Minor) => "4A",
        (PitchClass::Fs, Mode::Minor) => "11A",
        (PitchClass::G, Mode::Minor) => "6A",
        (PitchClass::Gs, Mode::Minor) => "1A",

        (PitchClass::A, Mode::Major) => "11B",
        (PitchClass::As, Mode::Major) => "6B",
        (PitchClass::B, Mode::Major) => "1B",
        (PitchClass::C, Mode::Major) => "8B",
        (PitchClass::Cs, Mode::Major) => "3B",
        (PitchClass::D, Mode::Major) => "10B",
        (PitchClass::Ds, Mode::Major) => "5B",
        (PitchClass::E, Mode::Major) => "12B",
        (PitchClass::F, Mode::Major) => "7B",
        (PitchClass::Fs, Mode::Major) => "2B",
        (PitchClass::G, Mode::Major) => "9B",
        (PitchClass::Gs, Mode::Major) => "4B",
    }
}

/// Mapping from (PitchClass, Mode) to Open Key notation.
pub fn to_open_key(pitch: PitchClass, mode: Mode) -> &'static str {
    match (pitch, mode) {
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

/// Get harmonically compatible keys.
pub fn compatible_keys(camelot: &str) -> Vec<&'static str> {
    let (num, letter) = parse_camelot(camelot);
    if num == 0 {
        return vec![];
    }

    let mut compatible = Vec::with_capacity(4);
    compatible.push(camelot_string(num, letter));

    let plus_one = if num == 12 { 1 } else { num + 1 };
    let minus_one = if num == 1 { 12 } else { num - 1 };
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
        (1, 'A') => "1A",
        (1, 'B') => "1B",
        (2, 'A') => "2A",
        (2, 'B') => "2B",
        (3, 'A') => "3A",
        (3, 'B') => "3B",
        (4, 'A') => "4A",
        (4, 'B') => "4B",
        (5, 'A') => "5A",
        (5, 'B') => "5B",
        (6, 'A') => "6A",
        (6, 'B') => "6B",
        (7, 'A') => "7A",
        (7, 'B') => "7B",
        (8, 'A') => "8A",
        (8, 'B') => "8B",
        (9, 'A') => "9A",
        (9, 'B') => "9B",
        (10, 'A') => "10A",
        (10, 'B') => "10B",
        (11, 'A') => "11A",
        (11, 'B') => "11B",
        (12, 'A') => "12A",
        (12, 'B') => "12B",
        _ => "",
    }
}
