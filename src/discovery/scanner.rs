//! File discovery and scanning

use crate::error::{DjprepError, Result};
use crate::types::AudioFormat;
use hash32::FnvHasher;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

/// Discovered audio file with basic metadata
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub format: AudioFormat,
    pub size_bytes: u64,
}

/// Scan a path (file or directory) for audio files
pub fn scan(input: &Path, recursive: bool) -> Result<Vec<DiscoveredFile>> {
    if !input.exists() {
        return Err(DjprepError::FileNotFound(input.to_path_buf()));
    }

    let mut files = Vec::new();

    if input.is_file() {
        // Single file mode
        if let Some(file) = try_discover_file(input) {
            files.push(file);
        } else {
            return Err(DjprepError::UnsupportedFormat {
                path: input.to_path_buf(),
                format: input
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
    } else if input.is_dir() {
        // Directory mode
        let walker = if recursive {
            WalkDir::new(input)
        } else {
            WalkDir::new(input).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                if let Some(file) = try_discover_file(path) {
                    debug!("Discovered: {}", file.path.display());
                    files.push(file);
                }
            }
        }
    }

    info!("Discovered {} audio files", files.len());

    if files.is_empty() {
        warn!("No supported audio files found in {}", input.display());
    }

    Ok(files)
}

/// Try to create a DiscoveredFile if the path is a supported audio format
fn try_discover_file(path: &Path) -> Option<DiscoveredFile> {
    let ext = path.extension()?.to_str()?;
    let format = AudioFormat::from_extension(ext)?;

    let metadata = std::fs::metadata(path).ok()?;
    let size_bytes = metadata.len();

    Some(DiscoveredFile {
        path: path.to_path_buf(),
        format,
        size_bytes,
    })
}

/// Generate a deterministic track ID from a file path
///
/// Uses FNV-1a hash, masked to positive i32 range for Rekordbox compatibility
pub fn generate_track_id(path: &Path) -> i32 {
    use hash32::Hasher as Hash32Hasher;

    // Normalize path for cross-platform consistency
    let normalized = normalize_path_for_hash(path);

    let mut hasher = FnvHasher::default();
    hasher.write(normalized.as_bytes());
    let hash = hasher.finish32();

    // Mask off sign bit to ensure positive value
    (hash & 0x7FFFFFFF) as i32
}

/// Normalize a path string for consistent hashing across platforms
fn normalize_path_for_hash(path: &Path) -> String {
    let path_str = path.to_string_lossy();

    // Convert backslashes to forward slashes
    let normalized = path_str.replace('\\', "/");

    // Lowercase for case-insensitive filesystems
    normalized.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_id_deterministic() {
        let path = Path::new("/Users/dj/music/track.mp3");
        let id1 = generate_track_id(path);
        let id2 = generate_track_id(path);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_track_id_positive() {
        let paths = [
            "/a.mp3",
            "/very/long/path/to/some/deeply/nested/file.flac",
            "C:\\Music\\Track.wav",
        ];

        for path_str in paths {
            let id = generate_track_id(Path::new(path_str));
            assert!(id > 0, "Track ID should be positive: {}", id);
        }
    }

    #[test]
    fn test_path_normalization() {
        // Windows and Unix paths should hash the same
        let win = normalize_path_for_hash(Path::new("C:\\Music\\Track.mp3"));
        let unix = normalize_path_for_hash(Path::new("c:/music/track.mp3"));
        assert_eq!(win, unix);
    }
}
