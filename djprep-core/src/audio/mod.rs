//! Audio decoding for analysis.

mod decoder;

pub use decoder::{decode_bytes, decode_owned, decode_reader, TARGET_SAMPLE_RATE};
