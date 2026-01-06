//! Audio analysis modules
//!
//! This module provides traits for analysis backends and concrete implementations.
//! The trait abstraction allows swapping backends without changing pipeline code.

pub mod bpm;
pub mod key;
pub mod metadata;
pub mod stems;
pub mod stratum;
pub mod traits;

pub use traits::{BpmDetector, KeyDetector, StemSeparator};

// Placeholder implementations (for testing/fallback)
pub use bpm::PlaceholderBpmDetector;
pub use key::PlaceholderKeyDetector;
pub use stems::PlaceholderStemSeparator;

// Stem separator (requires 'stems' feature)
pub use stems::OrtStemSeparator;

// Real implementations using stratum-dsp
pub use stratum::{StratumBpmDetector, StratumKeyDetector};
