//! Metadata extraction from audio file tags
//!
//! Uses lofty to read ID3v2 (MP3), Vorbis comments (FLAC), and AIFF tags.

use crate::types::TrackMetadata;
use lofty::{Accessor, Probe, TaggedFileExt};
use std::path::Path;
use tracing::{debug, warn};

/// Extract metadata from an audio file's tags
///
/// Returns TrackMetadata with whatever fields are available.
/// On error (corrupt tags, missing file), returns default empty metadata.
pub fn extract_metadata(path: &Path) -> TrackMetadata {
    match extract_metadata_inner(path) {
        Ok(metadata) => metadata,
        Err(e) => {
            warn!("Failed to read metadata from {}: {}", path.display(), e);
            TrackMetadata::default()
        }
    }
}

fn extract_metadata_inner(path: &Path) -> Result<TrackMetadata, lofty::error::LoftyError> {
    let tagged_file = Probe::open(path)?.read()?;
    let tag = tagged_file.primary_tag().or_else(|| tagged_file.first_tag());

    let metadata = match tag {
        Some(tag) => TrackMetadata {
            title: tag.title().map(|s| s.to_string()),
            artist: tag.artist().map(|s| s.to_string()),
            album: tag.album().map(|s| s.to_string()),
            genre: tag.genre().map(|s| s.to_string()),
            year: tag.year().map(|y| y as i32),
        },
        None => {
            debug!("No tags found in {}", path.display());
            TrackMetadata::default()
        }
    };

    Ok(metadata)
}
