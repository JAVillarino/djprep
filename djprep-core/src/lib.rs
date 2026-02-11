//! djprep-core
//!
//! Shared, platform-neutral audio analysis primitives used by the djprep CLI
//! and WebAssembly bindings.

pub mod analysis;
pub mod camelot;
pub mod error;
pub mod types;

pub use analysis::{
    analyze_audio_buffer, analyze_audio_bytes, analyze_audio_owned, detect_bpm, detect_key,
    AnalysisSummary,
};
pub use error::{CoreError, Result};
pub use types::{AudioBuffer, BpmResult, KeyResult, Mode, PitchClass};

#[cfg(feature = "decode")]
pub mod audio;

#[cfg(feature = "decode")]
pub use audio::{decode_bytes, decode_owned, decode_reader, TARGET_SAMPLE_RATE};

#[cfg(feature = "wasm-bindings")]
pub mod wasm;
