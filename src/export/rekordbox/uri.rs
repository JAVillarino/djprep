//! URI encoding for Rekordbox Location attribute
//!
//! Rekordbox requires paths in a specific URI format:
//! - Protocol: file://localhost/
//! - Windows: C:\Music\Track.mp3 → file://localhost/C:/Music/Track.mp3
//! - macOS/Linux: /Users/DJ/Music/Track.mp3 → file://localhost/Users/DJ/Music/Track.mp3

use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use std::path::Path;

/// Characters that must be percent-encoded in path segments
/// Based on RFC 3986, but more conservative for Rekordbox compatibility
const PATH_SEGMENT_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'&')
    .add(b'\'')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'[')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

/// Convert a filesystem path to Rekordbox URI format
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use djprep::export::rekordbox::uri::path_to_rekordbox_uri;
///
/// // macOS/Linux
/// let path = Path::new("/Users/DJ/Music/Track.mp3");
/// assert!(path_to_rekordbox_uri(path).starts_with("file://localhost/"));
///
/// // Windows
/// let path = Path::new("C:\\Music\\Track.mp3");
/// let uri = path_to_rekordbox_uri(path);
/// assert!(uri.starts_with("file://localhost/"));
/// ```
pub fn path_to_rekordbox_uri(path: &Path) -> String {
    // Get canonical path if possible, otherwise use as-is
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let path_str = path.to_string_lossy();

    // Normalize separators (Windows backslashes to forward slashes)
    let normalized = path_str.replace('\\', "/");

    // Handle Windows drive letters: C:/... → /C:/...
    let normalized = if is_windows_path(&normalized) {
        format!("/{}", normalized)
    } else {
        normalized.to_string()
    };

    // Encode each path segment separately (preserve slashes)
    let encoded = normalized
        .split('/')
        .map(|segment| {
            utf8_percent_encode(segment, PATH_SEGMENT_ENCODE_SET).to_string()
        })
        .collect::<Vec<_>>()
        .join("/");

    format!("file://localhost{}", encoded)
}

/// Check if a path string looks like a Windows path (has drive letter)
fn is_windows_path(path: &str) -> bool {
    let chars: Vec<char> = path.chars().take(3).collect();
    chars.len() >= 2
        && chars[0].is_ascii_alphabetic()
        && chars[1] == ':'
}

/// Validate that a path exists and is accessible
pub fn validate_path(path: &Path) -> bool {
    path.exists() && path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unix_path() {
        let path = Path::new("/Users/DJ/Music/Track.mp3");
        let uri = path_to_rekordbox_uri(path);
        // Note: canonicalize may fail in tests, so check the format
        assert!(uri.starts_with("file://localhost/"));
        assert!(uri.ends_with("Track.mp3"));
    }

    #[test]
    fn test_space_encoding() {
        let path = Path::new("/Users/DJ/My Music/Track Name.mp3");
        let uri = path_to_rekordbox_uri(path);
        assert!(uri.contains("%20") || uri.contains("My%20Music"));
    }

    #[test]
    fn test_special_chars() {
        let path = Path::new("/Music/[2024] Album & More/Track.mp3");
        let uri = path_to_rekordbox_uri(path);
        // Should encode brackets and ampersand
        assert!(uri.contains("%5B") || !uri.contains('['));
        assert!(uri.contains("%26") || !uri.contains('&'));
    }

    #[test]
    fn test_is_windows_path() {
        assert!(is_windows_path("C:/Music/Track.mp3"));
        assert!(is_windows_path("D:/"));
        assert!(!is_windows_path("/Users/DJ"));
        assert!(!is_windows_path("relative/path"));
    }
}
