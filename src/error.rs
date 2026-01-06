//! Unified error types for djprep
//!
//! Error strategy:
//! - Per-file errors (decode, analysis): Recoverable, skip and continue
//! - System errors (output, model download): Fatal, abort batch
//!
//! All errors include actionable suggestions where possible.

use std::path::PathBuf;
use thiserror::Error;

/// Supported audio formats for helpful error messages
pub const SUPPORTED_FORMATS: &str = "MP3, WAV, FLAC, AIFF";

/// Top-level error type for djprep operations
#[derive(Debug, Error)]
pub enum DjprepError {
    // =========================================================================
    // Recoverable errors - skip file, continue batch
    // =========================================================================
    #[error("Failed to decode audio file '{path}': {reason}\n  Supported formats: {SUPPORTED_FORMATS}\n  Tip: If the file plays in other apps, it may be corrupted or use an unsupported codec")]
    DecodeError { path: PathBuf, reason: String },

    #[error("Unsupported audio format for '{path}': {format}\n  Supported formats: {SUPPORTED_FORMATS}")]
    UnsupportedFormat { path: PathBuf, format: String },

    #[error("Analysis failed for '{path}': {reason}")]
    AnalysisError { path: PathBuf, reason: String },

    #[error("File not found: '{0}'\n  Tip: Check the path exists and is accessible")]
    FileNotFound(PathBuf),

    // =========================================================================
    // Stem-specific errors - may disable stems and continue
    // =========================================================================
    #[error("Stem separation unavailable: {reason}\n\n  To enable stem separation:\n  1. Download HTDemucs model from:\n     https://github.com/intel/openvino-plugins-ai-audacity/releases\n  2. Set environment variable:\n     export DJPREP_MODEL_PATH=/path/to/htdemucs_v4.onnx\n  3. Build with stems feature:\n     cargo build --release --features stems")]
    StemUnavailable { reason: String },

    #[error("Model inference failed: {reason}\n  Tip: This may indicate insufficient memory or an incompatible model file")]
    InferenceError { reason: String },

    // =========================================================================
    // Fatal errors - abort entire batch
    // =========================================================================
    #[error("Cannot write output to '{path}': {reason}\n  Tip: Check write permissions for the output directory")]
    OutputError { path: PathBuf, reason: String },

    #[error("Model download failed: {reason}\n\n  Alternative: Download manually from:\n  https://github.com/intel/openvino-plugins-ai-audacity/releases\n  Then set: export DJPREP_MODEL_PATH=/path/to/htdemucs_v4.onnx")]
    ModelDownloadError { reason: String },

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for djprep operations
pub type Result<T> = std::result::Result<T, DjprepError>;

/// Represents the outcome of processing a single file
#[derive(Debug)]
pub enum FileOutcome {
    /// Successfully analyzed
    Success,
    /// Skipped due to recoverable error
    Skipped { reason: String },
    /// Partial success (e.g., BPM/Key ok, stems failed)
    Partial { warning: String },
}

impl DjprepError {
    /// Returns true if this error is recoverable (should skip file, continue batch)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DjprepError::DecodeError { .. }
                | DjprepError::UnsupportedFormat { .. }
                | DjprepError::AnalysisError { .. }
                | DjprepError::FileNotFound(_)
        )
    }

    /// Returns true if this error should disable stems but continue processing
    pub fn is_stem_error(&self) -> bool {
        matches!(
            self,
            DjprepError::StemUnavailable { .. } | DjprepError::InferenceError { .. }
        )
    }

    /// Create a decode error with context about the issue
    pub fn decode_error(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        DjprepError::DecodeError {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Create an output error, checking for common issues
    pub fn output_error(path: impl Into<PathBuf>, err: std::io::Error) -> Self {
        let path = path.into();
        let reason = match err.kind() {
            std::io::ErrorKind::PermissionDenied => {
                format!("Permission denied. Check that you have write access to {}", path.display())
            }
            std::io::ErrorKind::NotFound => {
                format!("Directory does not exist: {}", path.parent().map(|p| p.display().to_string()).unwrap_or_default())
            }
            std::io::ErrorKind::AlreadyExists => {
                format!("File already exists: {}", path.display())
            }
            _ => err.to_string(),
        };
        DjprepError::OutputError { path, reason }
    }

    /// Create a stem unavailable error for missing model
    pub fn stem_model_not_found() -> Self {
        DjprepError::StemUnavailable {
            reason: "HTDemucs model not found".to_string(),
        }
    }

    /// Create a stem unavailable error for invalid/corrupt model
    pub fn stem_model_invalid(details: impl Into<String>) -> Self {
        DjprepError::StemUnavailable {
            reason: format!(
                "Model file appears invalid or corrupt: {}\n  Tip: Try re-downloading the model",
                details.into()
            ),
        }
    }

    /// Create a stem unavailable error when the feature is not enabled
    pub fn stem_feature_disabled() -> Self {
        DjprepError::StemUnavailable {
            reason: "Stem separation feature not compiled in".to_string(),
        }
    }
}

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error about which file was being processed
    fn with_file_context(self, path: &std::path::Path) -> Result<T>;
}

impl<T, E: std::fmt::Display> ErrorContext<T> for std::result::Result<T, E> {
    fn with_file_context(self, path: &std::path::Path) -> Result<T> {
        self.map_err(|e| DjprepError::AnalysisError {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })
    }
}
