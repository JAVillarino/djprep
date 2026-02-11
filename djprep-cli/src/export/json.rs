//! JSON export for interoperability with other tools

use crate::error::{DjprepError, Result};
use crate::types::AnalyzedTrack;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tracing::{debug, info};

/// JSON output schema version
const SCHEMA_VERSION: &str = "1.0";

/// Top-level JSON output structure
#[derive(Debug, Serialize, Deserialize)]
pub struct DjprepJson {
    /// Schema version for forward compatibility
    pub version: String,
    /// Analysis metadata
    pub metadata: ExportMetadata,
    /// Analyzed tracks
    pub tracks: Vec<TrackJson>,
}

/// Export metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// djprep version that generated this file
    pub generator_version: String,
    /// Timestamp of export
    pub exported_at: String,
    /// Number of tracks
    pub track_count: usize,
}

/// JSON representation of an analyzed track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackJson {
    /// Unique track ID (same as Rekordbox TrackID)
    pub track_id: i32,
    /// File path
    pub path: String,
    /// Track metadata
    pub metadata: TrackMetadataJson,
    /// BPM analysis results
    pub bpm: BpmJson,
    /// Key analysis results
    pub key: KeyJson,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Stem file paths (if separated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stems: Option<StemsJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadataJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artist: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub album: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genre: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpmJson {
    pub value: f64,
    pub confidence: f64,
    /// Alternative tempo candidates
    #[serde(default)]
    pub candidates: Vec<BpmCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpmCandidate {
    pub value: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyJson {
    /// Standard notation (e.g., "Am", "C#")
    pub standard: String,
    /// Camelot wheel notation (e.g., "8A")
    pub camelot: String,
    /// Open Key notation (e.g., "8m")
    pub open_key: String,
    /// Confidence score
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemsJson {
    pub vocals: String,
    pub drums: String,
    pub bass: String,
    pub other: String,
}

/// Write analyzed tracks to a JSON file
///
/// Uses atomic write pattern: writes to a temp file first, then renames.
/// This prevents data corruption if the write is interrupted.
pub fn write_json(tracks: &[AnalyzedTrack], output_path: &Path) -> Result<()> {
    // Write to temp file in same directory (ensures same filesystem for atomic rename)
    let temp_path = output_path.with_extension("json.tmp");

    let file = File::create(&temp_path).map_err(|e| DjprepError::OutputError {
        path: output_path.to_path_buf(),
        reason: format!("Failed to create temp file: {}", e),
    })?;

    let writer = BufWriter::new(file);

    let output = DjprepJson {
        version: SCHEMA_VERSION.to_string(),
        metadata: ExportMetadata {
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            track_count: tracks.len(),
        },
        tracks: tracks.iter().map(track_to_json).collect(),
    };

    serde_json::to_writer_pretty(writer, &output).map_err(|e| {
        // Clean up temp file on error
        let _ = std::fs::remove_file(&temp_path);
        DjprepError::OutputError {
            path: output_path.to_path_buf(),
            reason: e.to_string(),
        }
    })?;

    // Atomic rename: either succeeds completely or fails without modifying target
    std::fs::rename(&temp_path, output_path).map_err(|e| {
        // Clean up temp file on error
        let _ = std::fs::remove_file(&temp_path);
        DjprepError::OutputError {
            path: output_path.to_path_buf(),
            reason: format!("Failed to finalize file: {}", e),
        }
    })?;

    info!("Wrote {} tracks to {}", tracks.len(), output_path.display());

    Ok(())
}

fn track_to_json(track: &AnalyzedTrack) -> TrackJson {
    let standard_key = format!(
        "{}{}",
        track.key.pitch_class.to_standard_notation(),
        match track.key.mode {
            crate::types::Mode::Major => "",
            crate::types::Mode::Minor => "m",
        }
    );

    TrackJson {
        track_id: track.track_id,
        path: track.path.to_string_lossy().to_string(),
        metadata: TrackMetadataJson {
            title: track.metadata.title.clone(),
            artist: track.metadata.artist.clone(),
            album: track.metadata.album.clone(),
            genre: track.metadata.genre.clone(),
        },
        bpm: BpmJson {
            value: track.bpm.value,
            confidence: track.bpm.confidence,
            candidates: track
                .bpm
                .candidates
                .iter()
                .map(|(v, c)| BpmCandidate {
                    value: *v,
                    confidence: *c,
                })
                .collect(),
        },
        key: KeyJson {
            standard: standard_key,
            camelot: track.key.camelot.clone(),
            open_key: track.key.open_key.clone(),
            confidence: track.key.confidence,
        },
        duration_seconds: track.duration_seconds,
        stems: track.stems.as_ref().map(|s| StemsJson {
            vocals: s.vocals.to_string_lossy().to_string(),
            drums: s.drums.to_string_lossy().to_string(),
            bass: s.bass.to_string_lossy().to_string(),
            other: s.other.to_string_lossy().to_string(),
        }),
    }
}

/// Read existing analysis from JSON file
///
/// Returns the set of file paths that have already been analyzed.
/// If the file doesn't exist or can't be parsed, returns an empty set.
pub fn read_existing_analysis(json_path: &Path) -> HashSet<String> {
    if !json_path.exists() {
        debug!("No existing analysis file at {}", json_path.display());
        return HashSet::new();
    }

    let file = match File::open(json_path) {
        Ok(f) => f,
        Err(e) => {
            debug!("Could not open existing analysis: {}", e);
            return HashSet::new();
        }
    };

    let reader = BufReader::new(file);
    let json: DjprepJson = match serde_json::from_reader(reader) {
        Ok(j) => j,
        Err(e) => {
            debug!("Could not parse existing analysis: {}", e);
            return HashSet::new();
        }
    };

    let paths: HashSet<String> = json.tracks.iter().map(|t| t.path.clone()).collect();

    debug!(
        "Loaded {} previously analyzed tracks from {}",
        paths.len(),
        json_path.display()
    );

    paths
}

/// Read existing analysis and return the full track data
///
/// Returns existing tracks that should be preserved when not re-analyzing.
pub fn read_existing_tracks(json_path: &Path) -> Vec<TrackJson> {
    if !json_path.exists() {
        return Vec::new();
    }

    let file = match File::open(json_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = BufReader::new(file);
    match serde_json::from_reader::<_, DjprepJson>(reader) {
        Ok(json) => json.tracks,
        Err(_) => Vec::new(),
    }
}
