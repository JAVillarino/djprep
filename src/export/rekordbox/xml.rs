//! Rekordbox XML writer
//!
//! Generates rekordbox.xml files using streaming XML output to handle
//! large libraries without memory issues.

use crate::error::{DjprepError, Result};
use crate::types::AnalyzedTrack;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
use quick_xml::Writer;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tracing::info;

use super::schema::{self, attrs, node_types};
use super::uri::path_to_rekordbox_uri;

/// Write analyzed tracks to a Rekordbox XML file
///
/// Uses atomic write pattern: writes to a temp file first, then renames.
/// This prevents data corruption if the write is interrupted.
pub fn write_rekordbox_xml(tracks: &[AnalyzedTrack], output_path: &Path) -> Result<()> {
    // Write to temp file in same directory (ensures same filesystem for atomic rename)
    let temp_path = output_path.with_extension("xml.tmp");

    // Helper to clean up temp file on error
    let cleanup_and_error = |reason: String| -> DjprepError {
        let _ = std::fs::remove_file(&temp_path);
        DjprepError::OutputError {
            path: output_path.to_path_buf(),
            reason,
        }
    };

    let file = File::create(&temp_path).map_err(|e| DjprepError::OutputError {
        path: output_path.to_path_buf(),
        reason: format!("Failed to create temp file: {}", e),
    })?;

    let writer = BufWriter::new(file);
    let mut xml = Writer::new_with_indent(writer, b' ', 2);

    // XML declaration
    xml.write_event(Event::Decl(BytesDecl::new(
        schema::XML_VERSION,
        Some(schema::XML_ENCODING),
        None,
    )))
    .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // Root element: DJ_PLAYLISTS
    let mut root = BytesStart::new("DJ_PLAYLISTS");
    root.push_attribute(("Version", schema::PLAYLISTS_VERSION));
    xml.write_event(Event::Start(root))
        .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // PRODUCT element
    let mut product = BytesStart::new("PRODUCT");
    product.push_attribute(("Name", schema::PRODUCT_NAME));
    product.push_attribute(("Version", schema::PRODUCT_VERSION));
    xml.write_event(Event::Empty(product))
        .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // COLLECTION element
    let mut collection = BytesStart::new("COLLECTION");
    collection.push_attribute(("Entries", tracks.len().to_string().as_str()));
    xml.write_event(Event::Start(collection))
        .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // Write each track
    for track in tracks {
        write_track(&mut xml, track, &temp_path).map_err(|e| {
            let _ = std::fs::remove_file(&temp_path);
            e
        })?;
    }

    // Close COLLECTION
    xml.write_event(Event::End(BytesEnd::new("COLLECTION")))
        .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // PLAYLISTS section (with import workaround playlist)
    write_playlists(&mut xml, tracks, &temp_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp_path);
        e
    })?;

    // Close DJ_PLAYLISTS
    xml.write_event(Event::End(BytesEnd::new("DJ_PLAYLISTS")))
        .map_err(|e| cleanup_and_error(format!("XML write error: {}", e)))?;

    // Flush and drop the writer before rename
    drop(xml);

    // Atomic rename: either succeeds completely or fails without modifying target
    std::fs::rename(&temp_path, output_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp_path);
        DjprepError::OutputError {
            path: output_path.to_path_buf(),
            reason: format!("Failed to finalize file: {}", e),
        }
    })?;

    info!("Wrote {} tracks to {}", tracks.len(), output_path.display());

    Ok(())
}

/// Write a single TRACK element
fn write_track<W: std::io::Write>(
    xml: &mut Writer<W>,
    track: &AnalyzedTrack,
    output_path: &Path,
) -> Result<()> {
    let mut elem = BytesStart::new("TRACK");

    // Required attributes
    elem.push_attribute((attrs::TRACK_ID, track.track_id.to_string().as_str()));

    // Name (from metadata or filename)
    // Use as_deref() to avoid cloning when title exists
    let name: std::borrow::Cow<str> = match track.metadata.title.as_deref() {
        Some(title) => std::borrow::Cow::Borrowed(title),
        None => std::borrow::Cow::Owned(
            track
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Unknown")
                .to_string(),
        ),
    };
    elem.push_attribute((attrs::NAME, name.as_ref()));

    // Artist
    if let Some(ref artist) = track.metadata.artist {
        elem.push_attribute((attrs::ARTIST, artist.as_str()));
    }

    // Album
    if let Some(ref album) = track.metadata.album {
        elem.push_attribute((attrs::ALBUM, album.as_str()));
    }

    // Genre
    if let Some(ref genre) = track.metadata.genre {
        elem.push_attribute((attrs::GENRE, genre.as_str()));
    }

    // Location (URI encoded path)
    let location = path_to_rekordbox_uri(&track.path);
    elem.push_attribute((attrs::LOCATION, location.as_str()));

    // Duration - guard against NaN/Inf which would cause undefined behavior in cast
    let total_time = if track.duration_seconds.is_finite() && track.duration_seconds >= 0.0 {
        track.duration_seconds.round().min(i64::MAX as f64) as i64
    } else {
        0 // Use 0 for invalid duration
    };
    elem.push_attribute((attrs::TOTAL_TIME, total_time.to_string().as_str()));

    // BPM (2 decimal places) - guard against NaN/Inf for valid XML
    let bpm = if track.bpm.value.is_finite() && track.bpm.value > 0.0 {
        format!("{:.2}", track.bpm.value.clamp(1.0, 999.99))
    } else {
        "120.00".to_string() // Reasonable default for invalid BPM
    };
    elem.push_attribute((attrs::AVERAGE_BPM, bpm.as_str()));

    // Key (Camelot notation)
    elem.push_attribute((attrs::TONALITY, track.key.camelot.as_str()));

    // Date added
    let date_added = track.analyzed_at.format("%Y-%m-%d").to_string();
    elem.push_attribute((attrs::DATE_ADDED, date_added.as_str()));

    // Sample rate
    elem.push_attribute((attrs::SAMPLE_RATE, track.sample_rate.to_string().as_str()));

    // Comments (include analysis confidence info) - clamp confidence to valid range
    let bpm_conf = if track.bpm.confidence.is_finite() {
        (track.bpm.confidence * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let key_conf = if track.key.confidence.is_finite() {
        (track.key.confidence * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let comment = format!(
        "djprep: BPM conf={:.0}%, Key conf={:.0}%",
        bpm_conf, key_conf
    );
    elem.push_attribute((attrs::COMMENTS, comment.as_str()));

    // Write as empty element (no children)
    xml.write_event(Event::Empty(elem))
        .map_err(|e| write_error(output_path, e))?;

    Ok(())
}

/// Write PLAYLISTS section with the import workaround playlist
fn write_playlists<W: std::io::Write>(
    xml: &mut Writer<W>,
    tracks: &[AnalyzedTrack],
    output_path: &Path,
) -> Result<()> {
    // PLAYLISTS root
    xml.write_event(Event::Start(BytesStart::new("PLAYLISTS")))
        .map_err(|e| write_error(output_path, e))?;

    // ROOT node (required by Rekordbox)
    let mut root_node = BytesStart::new("NODE");
    root_node.push_attribute(("Type", node_types::ROOT));
    root_node.push_attribute(("Name", "ROOT"));
    xml.write_event(Event::Start(root_node))
        .map_err(|e| write_error(output_path, e))?;

    // Import workaround playlist
    // Per Section 4.4 of spec: Rekordbox only updates metadata via playlist import
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let playlist_name = format!("djprep_import_{}", timestamp);

    let mut playlist_node = BytesStart::new("NODE");
    playlist_node.push_attribute(("Type", node_types::PLAYLIST));
    playlist_node.push_attribute(("Name", playlist_name.as_str()));
    playlist_node.push_attribute(("KeyType", "0"));
    playlist_node.push_attribute(("Entries", tracks.len().to_string().as_str()));

    xml.write_event(Event::Start(playlist_node))
        .map_err(|e| write_error(output_path, e))?;

    // Add track references to playlist
    for track in tracks {
        let mut track_ref = BytesStart::new("TRACK");
        track_ref.push_attribute(("Key", track.track_id.to_string().as_str()));
        xml.write_event(Event::Empty(track_ref))
            .map_err(|e| write_error(output_path, e))?;
    }

    // Close playlist node
    xml.write_event(Event::End(BytesEnd::new("NODE")))
        .map_err(|e| write_error(output_path, e))?;

    // Close ROOT node
    xml.write_event(Event::End(BytesEnd::new("NODE")))
        .map_err(|e| write_error(output_path, e))?;

    // Close PLAYLISTS
    xml.write_event(Event::End(BytesEnd::new("PLAYLISTS")))
        .map_err(|e| write_error(output_path, e))?;

    Ok(())
}

/// Convert I/O errors during XML writing to DjprepError
///
/// Note: `Writer::write_event` returns `io::Result<()>` when the underlying
/// writer is `BufWriter<File>`, so we receive `std::io::Error` here rather
/// than `quick_xml::Error`. This is intentional - the quick_xml Writer
/// propagates I/O errors directly from the underlying writer.
fn write_error(path: &Path, e: std::io::Error) -> DjprepError {
    DjprepError::OutputError {
        path: path.to_path_buf(),
        reason: format!("XML write error: {}", e),
    }
}
