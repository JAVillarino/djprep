//! Audio decoding and buffer handling

mod decoder;

pub use decoder::{decode, decode_stereo, STEM_SAMPLE_RATE, TARGET_SAMPLE_RATE};
