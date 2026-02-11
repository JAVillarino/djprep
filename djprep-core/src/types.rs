//! Shared analysis domain types.

use serde::{Deserialize, Serialize};

/// The 12 pitch classes in Western music.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PitchClass {
    C,
    Cs,
    D,
    Ds,
    E,
    F,
    Fs,
    G,
    Gs,
    A,
    As,
    B,
}

impl PitchClass {
    /// Convert from numeric index (0 = C, ..., 11 = B).
    pub fn from_index(index: u8) -> Option<Self> {
        match index % 12 {
            0 => Some(PitchClass::C),
            1 => Some(PitchClass::Cs),
            2 => Some(PitchClass::D),
            3 => Some(PitchClass::Ds),
            4 => Some(PitchClass::E),
            5 => Some(PitchClass::F),
            6 => Some(PitchClass::Fs),
            7 => Some(PitchClass::G),
            8 => Some(PitchClass::Gs),
            9 => Some(PitchClass::A),
            10 => Some(PitchClass::As),
            11 => Some(PitchClass::B),
            _ => None,
        }
    }

    /// Convert to numeric index (0 = C, ..., 11 = B).
    pub fn to_index(self) -> u8 {
        match self {
            PitchClass::C => 0,
            PitchClass::Cs => 1,
            PitchClass::D => 2,
            PitchClass::Ds => 3,
            PitchClass::E => 4,
            PitchClass::F => 5,
            PitchClass::Fs => 6,
            PitchClass::G => 7,
            PitchClass::Gs => 8,
            PitchClass::A => 9,
            PitchClass::As => 10,
            PitchClass::B => 11,
        }
    }

    /// Standard notation (e.g., "C", "F#", "Bb").
    pub fn to_standard_notation(self) -> &'static str {
        match self {
            PitchClass::C => "C",
            PitchClass::Cs => "C#",
            PitchClass::D => "D",
            PitchClass::Ds => "D#",
            PitchClass::E => "E",
            PitchClass::F => "F",
            PitchClass::Fs => "F#",
            PitchClass::G => "G",
            PitchClass::Gs => "G#",
            PitchClass::A => "A",
            PitchClass::As => "A#",
            PitchClass::B => "B",
        }
    }
}

/// Major or minor scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    Major,
    Minor,
}

/// BPM analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpmResult {
    /// Primary detected BPM.
    pub value: f64,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Alternative candidate tempos as (value, confidence).
    pub candidates: Vec<(f64, f64)>,
}

impl Default for BpmResult {
    fn default() -> Self {
        Self {
            value: 120.0,
            confidence: 0.0,
            candidates: vec![],
        }
    }
}

/// Musical key analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyResult {
    /// Detected pitch class.
    pub pitch_class: PitchClass,
    /// Major or minor mode.
    pub mode: Mode,
    /// Camelot notation ("1A" - "12B").
    pub camelot: String,
    /// Open Key notation ("1m" - "12d").
    pub open_key: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
}

impl Default for KeyResult {
    fn default() -> Self {
        Self {
            pitch_class: PitchClass::C,
            mode: Mode::Major,
            camelot: "8B".to_string(),
            open_key: "1d".to_string(),
            confidence: 0.0,
        }
    }
}

/// Decoded mono audio samples ready for analysis.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Mono samples normalized to [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration: f64,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        let duration = samples.len() as f64 / sample_rate as f64;
        Self {
            samples,
            sample_rate,
            duration,
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
