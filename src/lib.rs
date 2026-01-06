//! djprep - High-Performance Audio Analysis & Metadata Interchange System
//!
//! A command-line utility for batch analysis of audio files to extract
//! BPM, musical key, and stems. Outputs Rekordbox-compatible XML and JSON.
//!
//! # Architecture
//!
//! The library is organized into several key modules:
//!
//! - `config`: CLI argument parsing and runtime settings
//! - `discovery`: File scanning and track ID generation  
//! - `audio`: Audio decoding using symphonia
//! - `analysis`: BPM, key, and stem separation (with swappable backends)
//! - `pipeline`: Parallel processing orchestration
//! - `export`: Rekordbox XML and JSON output
//!
//! # Example
//!
//! ```no_run
//! use djprep::{config::Settings, pipeline};
//!
//! let settings = Settings::default();
//! let result = pipeline::run(&settings).expect("Analysis failed");
//! println!("Processed {} tracks", result.successful);
//! ```

pub mod analysis;
pub mod audio;
pub mod config;
pub mod discovery;
pub mod error;
pub mod export;
pub mod pipeline;
pub mod types;

// Re-export key types at crate root
pub use error::{DjprepError, Result};
pub use types::{AnalyzedTrack, AudioBuffer, BpmResult, KeyResult};
