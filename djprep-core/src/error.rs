//! Core error types for platform-neutral analysis.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Audio decode failed: {0}")]
    DecodeError(String),

    #[error("Audio analysis failed: {0}")]
    AnalysisError(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;
