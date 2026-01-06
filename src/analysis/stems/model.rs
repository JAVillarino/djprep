//! HTDemucs model management
//!
//! Downloads and caches the ONNX model for stem separation.
//! Model resolution checks multiple common locations automatically.

use crate::error::{DjprepError, Result};
use directories::ProjectDirs;
use std::fs;
use std::path::PathBuf;
#[cfg(feature = "stems")]
use tracing::{debug, info, warn};

/// HTDemucs model configuration
pub struct ModelConfig {
    /// URL to download the model from
    pub url: &'static str,
    /// Expected SHA-256 hash of the model file
    pub sha256: &'static str,
    /// Model filename
    pub filename: &'static str,
    /// Model size in bytes (for progress reporting)
    pub size_bytes: u64,
}

/// Default HTDemucs ONNX model
///
/// Model source options:
/// 1. Intel OpenVINO Audacity Plugin (pre-converted)
/// 2. Convert using sevagh/demucs.onnx
/// 3. User-provided local path via DJPREP_MODEL_PATH env var
///
/// Note: The Intel model requires extraction from their release zip.
/// See README for setup instructions.
pub const HTDEMUCS_MODEL: ModelConfig = ModelConfig {
    // Note: This URL is for reference. The actual model needs manual download
    // because Intel packages it in a zip archive.
    url: "https://github.com/intel/openvino-plugins-ai-audacity/releases/download/v3.4.2-R1/openvino-models.zip",
    sha256: "placeholder_hash_replace_with_actual",
    filename: "htdemucs_v4.onnx",
    size_bytes: 170_000_000, // ~170MB
};

/// Check for user-provided model path via environment variable
pub fn get_user_model_path() -> Option<PathBuf> {
    std::env::var("DJPREP_MODEL_PATH").ok().map(PathBuf::from)
}

/// Find the model file by checking multiple common locations
///
/// Search order:
/// 1. DJPREP_MODEL_PATH environment variable
/// 2. ProjectDirs cache: ~/.cache/djprep/models/htdemucs_v4.onnx (Linux)
///    or ~/Library/Caches/com.djprep.djprep/models/ (macOS)
/// 3. ProjectDirs data: ~/.local/share/djprep/models/ (Linux XDG)
///    or ~/Library/Application Support/com.djprep.djprep/models/ (macOS)
/// 4. Current directory: ./models/htdemucs_v4.onnx
/// 5. Home directory: ~/djprep/models/htdemucs_v4.onnx
///
/// Returns the first existing model path found, or an error listing all checked locations.
pub fn find_model_path() -> Result<PathBuf> {
    let filename = HTDEMUCS_MODEL.filename;
    let mut checked_locations: Vec<String> = Vec::new();

    // 1. Check environment variable first
    if let Some(env_path) = get_user_model_path() {
        if env_path.exists() {
            return Ok(env_path);
        }
        checked_locations.push(format!("DJPREP_MODEL_PATH={}", env_path.display()));
    }

    // 2. Check ProjectDirs cache directory
    if let Some(proj_dirs) = ProjectDirs::from("com", "djprep", "djprep") {
        let cache_path = proj_dirs.cache_dir().join("models").join(filename);
        if cache_path.exists() {
            return Ok(cache_path);
        }
        checked_locations.push(cache_path.display().to_string());

        // 3. Check ProjectDirs data directory
        let data_path = proj_dirs.data_dir().join("models").join(filename);
        if data_path.exists() {
            return Ok(data_path);
        }
        checked_locations.push(data_path.display().to_string());
    }

    // 4. Check current directory
    let cwd_path = PathBuf::from("./models").join(filename);
    if cwd_path.exists() {
        return Ok(cwd_path.canonicalize().unwrap_or(cwd_path));
    }
    checked_locations.push(cwd_path.display().to_string());

    // 5. Check home directory
    if let Some(base_dirs) = directories::BaseDirs::new() {
        let home_path = base_dirs.home_dir().join("djprep").join("models").join(filename);
        if home_path.exists() {
            return Ok(home_path);
        }
        checked_locations.push(home_path.display().to_string());
    }

    // Model not found - generate helpful error
    let locations_list = checked_locations
        .iter()
        .map(|loc| format!("  - {}", loc))
        .collect::<Vec<_>>()
        .join("\n");

    Err(DjprepError::StemUnavailable {
        reason: format!(
            "HTDemucs model not found.\n\n\
             Locations checked:\n{}\n\n\
             To fix this, either:\n\
             1. Set the environment variable:\n\
                export DJPREP_MODEL_PATH=/path/to/htdemucs_v4.onnx\n\n\
             2. Or place the model in one of the above locations.\n\n\
             Download the model from:\n\
             https://github.com/intel/openvino-plugins-ai-audacity/releases",
            locations_list
        ),
    })
}

/// Get the model cache directory
pub fn get_cache_dir() -> Result<PathBuf> {
    let proj_dirs = ProjectDirs::from("com", "djprep", "djprep").ok_or_else(|| {
        DjprepError::ConfigError("Could not determine cache directory".to_string())
    })?;

    let cache_dir = proj_dirs.cache_dir().join("models");
    fs::create_dir_all(&cache_dir).map_err(|e| DjprepError::OutputError {
        path: cache_dir.clone(),
        reason: format!("Failed to create cache directory: {}", e),
    })?;

    Ok(cache_dir)
}

/// Get the path to the model file
pub fn get_model_path(config: &ModelConfig) -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    Ok(cache_dir.join(config.filename))
}

/// Check if the model is already cached
pub fn is_model_cached(config: &ModelConfig) -> bool {
    match get_model_path(config) {
        Ok(path) => path.exists(),
        Err(_) => false,
    }
}

/// Download the model if not already cached
#[cfg(feature = "stems")]
pub fn ensure_model(config: &ModelConfig) -> Result<PathBuf> {
    let model_path = get_model_path(config)?;

    if model_path.exists() {
        debug!("Model already cached at {}", model_path.display());

        // Optionally verify hash
        if verify_model_hash(&model_path, config.sha256)? {
            return Ok(model_path);
        } else {
            warn!("Cached model hash mismatch, re-downloading...");
            fs::remove_file(&model_path).ok();
        }
    }

    info!("Downloading HTDemucs model (~200MB)...");
    download_model(config, &model_path)?;

    Ok(model_path)
}

/// Download the model from URL
#[cfg(feature = "stems")]
fn download_model(config: &ModelConfig, dest_path: &PathBuf) -> Result<()> {
    use std::io::Write;

    let response = reqwest::blocking::get(config.url).map_err(|e| {
        DjprepError::ModelDownloadError {
            reason: format!("Failed to download model: {}", e),
        }
    })?;

    if !response.status().is_success() {
        return Err(DjprepError::ModelDownloadError {
            reason: format!("Model download failed with status: {}", response.status()),
        });
    }

    let bytes = response.bytes().map_err(|e| {
        DjprepError::ModelDownloadError {
            reason: format!("Failed to read model data: {}", e),
        }
    })?;

    let mut file = fs::File::create(dest_path).map_err(|e| DjprepError::OutputError {
        path: dest_path.clone(),
        reason: format!("Failed to create model file: {}", e),
    })?;

    file.write_all(&bytes).map_err(|e| DjprepError::OutputError {
        path: dest_path.clone(),
        reason: format!("Failed to write model file: {}", e),
    })?;

    info!("Model downloaded to {}", dest_path.display());

    // Verify hash after download
    if !verify_model_hash(dest_path, config.sha256)? {
        fs::remove_file(dest_path).ok();
        return Err(DjprepError::ModelDownloadError {
            reason: "Downloaded model hash verification failed".to_string(),
        });
    }

    Ok(())
}

/// Verify the SHA-256 hash of a model file
#[cfg(feature = "stems")]
fn verify_model_hash(path: &PathBuf, expected_hash: &str) -> Result<bool> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    // Skip verification if hash is placeholder
    if expected_hash.starts_with("placeholder") {
        debug!("Skipping hash verification (placeholder hash)");
        return Ok(true);
    }

    let mut file = fs::File::open(path).map_err(|e| DjprepError::OutputError {
        path: path.clone(),
        reason: format!("Failed to open model for verification: {}", e),
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| DjprepError::OutputError {
            path: path.clone(),
            reason: format!("Failed to read model for verification: {}", e),
        })?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    let actual_hash = hex::encode(hasher.finalize());
    let matches = actual_hash == expected_hash;

    if !matches {
        warn!(
            "Model hash mismatch: expected {}, got {}",
            expected_hash, actual_hash
        );
    }

    Ok(matches)
}

#[cfg(not(feature = "stems"))]
pub fn ensure_model(_config: &ModelConfig) -> Result<PathBuf> {
    Err(DjprepError::StemUnavailable {
        reason: "Stem separation requires the 'stems' feature".to_string(),
    })
}

#[cfg(not(feature = "stems"))]
#[allow(dead_code)]
fn verify_model_hash(_path: &PathBuf, _expected_hash: &str) -> Result<bool> {
    Ok(false)
}
