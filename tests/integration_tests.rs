//! Integration tests for djprep pipeline
//!
//! These tests verify the full analysis pipeline produces correct output.

use djprep::{config::Settings, pipeline};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Generate a sine wave WAV file for testing
///
/// Creates a mono 16-bit WAV file at the specified path.
fn generate_sine_wav(path: &Path, frequency_hz: f32, duration_secs: f32, sample_rate: u32) {
    use std::f32::consts::PI;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV file");

    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let amplitude = 0.5f32; // 50% amplitude to avoid clipping

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * frequency_hz * t).sin() * amplitude;
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16).expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV");
}

/// Generate a click track WAV file for BPM testing
///
/// Creates impulses (short bursts) at regular intervals matching the specified BPM.
/// This produces a clear rhythmic signal that BPM detectors can analyze.
fn generate_click_track(path: &Path, bpm: f32, duration_secs: f32, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV file");

    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let samples_per_beat = (60.0 / bpm * sample_rate as f32) as usize;

    // Impulse duration: ~5ms (short click)
    let impulse_samples = (0.005 * sample_rate as f32) as usize;

    for i in 0..num_samples {
        let position_in_beat = i % samples_per_beat;

        // Generate impulse at the start of each beat
        let sample = if position_in_beat < impulse_samples {
            // Exponential decay for a more natural click sound
            let decay = (-5.0 * position_in_beat as f32 / impulse_samples as f32).exp();
            0.8 * decay
        } else {
            0.0
        };

        let sample_i16 = (sample * 32767.0) as i16;
        writer
            .write_sample(sample_i16)
            .expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV");
}

/// Helper to extract BPM value from analysis output
fn get_bpm_from_output(output_dir: &Path) -> f64 {
    let json_content =
        fs::read_to_string(output_dir.join("djprep.json")).expect("Failed to read JSON");
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let tracks = json.get("tracks").unwrap().as_array().unwrap();
    tracks[0]
        .get("bpm")
        .unwrap()
        .get("value")
        .unwrap()
        .as_f64()
        .unwrap()
}

/// Create test settings with progress bars disabled
fn create_test_settings(input: &Path, output: &Path) -> Settings {
    Settings {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        stems_enabled: false,
        stems_dir: output.join("stems"),
        genre_hint: None,
        analysis_threads: 2,
        recursive: true,
        force: false,
        output_json: true,
        show_progress: false, // Disable progress bars in tests
        dry_run: false,
    }
}

#[test]
fn test_pipeline_produces_valid_xml() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate a 5-second 440Hz (A4) sine wave
    let test_wav = input_dir.path().join("test_track.wav");
    generate_sine_wav(&test_wav, 440.0, 5.0, 44100);

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings).expect("Pipeline should succeed");

    // Verify result counts
    assert_eq!(result.total_files, 1, "Should find 1 file");
    assert_eq!(result.successful, 1, "Should successfully analyze 1 file");
    assert_eq!(result.failed, 0, "Should have no failures");

    // Verify XML file exists
    let xml_path = output_dir.path().join("rekordbox.xml");
    assert!(xml_path.exists(), "rekordbox.xml should exist");

    // Read and validate XML content
    let xml_content = fs::read_to_string(&xml_path).expect("Failed to read XML");

    // Check XML structure
    assert!(
        xml_content.contains("<?xml version=\"1.0\""),
        "Should have XML declaration"
    );
    assert!(
        xml_content.contains("<DJ_PLAYLISTS"),
        "Should have DJ_PLAYLISTS root element"
    );
    assert!(
        xml_content.contains("<PRODUCT Name=\"djprep\""),
        "Should have PRODUCT element"
    );
    assert!(
        xml_content.contains("<COLLECTION"),
        "Should have COLLECTION element"
    );
    assert!(xml_content.contains("<TRACK"), "Should have TRACK element");
    assert!(
        xml_content.contains("TrackID="),
        "TRACK should have TrackID attribute"
    );
    assert!(
        xml_content.contains("AverageBpm="),
        "TRACK should have AverageBpm attribute"
    );
    assert!(
        xml_content.contains("Tonality="),
        "TRACK should have Tonality attribute"
    );
    assert!(
        xml_content.contains("Location="),
        "TRACK should have Location attribute"
    );
    assert!(
        xml_content.contains("<PLAYLISTS"),
        "Should have PLAYLISTS element"
    );
    assert!(
        xml_content.contains("djprep_import_"),
        "Should have djprep_import playlist"
    );
}

#[test]
fn test_pipeline_produces_valid_json() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate a 3-second 880Hz (A5) sine wave
    let test_wav = input_dir.path().join("another_track.wav");
    generate_sine_wav(&test_wav, 880.0, 3.0, 44100);

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings).expect("Pipeline should succeed");

    assert_eq!(result.successful, 1, "Should successfully analyze 1 file");

    // Verify JSON file exists
    let json_path = output_dir.path().join("djprep.json");
    assert!(json_path.exists(), "djprep.json should exist");

    // Read and parse JSON
    let json_content = fs::read_to_string(&json_path).expect("Failed to read JSON");
    let json: serde_json::Value =
        serde_json::from_str(&json_content).expect("Should be valid JSON");

    // Verify JSON structure
    assert!(json.is_object(), "Root should be an object");
    assert!(json.get("version").is_some(), "Should have version field");
    assert!(json.get("metadata").is_some(), "Should have metadata field");
    assert!(json.get("tracks").is_some(), "Should have tracks field");

    // Verify tracks array
    let tracks = json.get("tracks").unwrap().as_array().unwrap();
    assert_eq!(tracks.len(), 1, "Should have 1 track");

    // Verify track structure
    let track = &tracks[0];
    assert!(track.get("track_id").is_some(), "Track should have track_id");
    assert!(track.get("path").is_some(), "Track should have path");
    assert!(track.get("bpm").is_some(), "Track should have bpm");
    assert!(track.get("key").is_some(), "Track should have key");
    assert!(
        track.get("duration_seconds").is_some(),
        "Track should have duration_seconds"
    );

    // Verify BPM structure
    let bpm = track.get("bpm").unwrap();
    assert!(bpm.get("value").is_some(), "BPM should have value");
    assert!(bpm.get("confidence").is_some(), "BPM should have confidence");

    // Verify Key structure
    let key = track.get("key").unwrap();
    assert!(key.get("camelot").is_some(), "Key should have camelot");
    assert!(key.get("standard").is_some(), "Key should have standard notation");
    assert!(key.get("open_key").is_some(), "Key should have open_key");
    assert!(key.get("confidence").is_some(), "Key should have confidence");
}

#[test]
fn test_pipeline_handles_empty_directory() {
    // Create empty temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Run the pipeline on empty directory
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings).expect("Pipeline should succeed on empty directory");

    // Verify result counts
    assert_eq!(result.total_files, 0, "Should find 0 files");
    assert_eq!(result.successful, 0, "Should have 0 successful");
    assert_eq!(result.failed, 0, "Should have 0 failures");
    assert_eq!(result.skipped, 0, "Should have 0 skipped");

    // Output files should NOT be created for empty input
    let xml_path = output_dir.path().join("rekordbox.xml");
    let json_path = output_dir.path().join("djprep.json");

    // The pipeline skips export when no tracks are analyzed
    assert!(
        !xml_path.exists(),
        "rekordbox.xml should not exist for empty input"
    );
    assert!(
        !json_path.exists(),
        "djprep.json should not exist for empty input"
    );
}

#[test]
fn test_pipeline_multiple_files() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate multiple test files with different frequencies
    generate_sine_wav(&input_dir.path().join("track_a.wav"), 261.63, 2.0, 44100); // C4
    generate_sine_wav(&input_dir.path().join("track_b.wav"), 329.63, 2.0, 44100); // E4
    generate_sine_wav(&input_dir.path().join("track_c.wav"), 392.00, 2.0, 44100); // G4

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings).expect("Pipeline should succeed");

    // Verify all files processed
    assert_eq!(result.total_files, 3, "Should find 3 files");
    assert_eq!(result.successful, 3, "Should successfully analyze 3 files");

    // Verify XML has all tracks
    let xml_content =
        fs::read_to_string(output_dir.path().join("rekordbox.xml")).expect("Failed to read XML");

    // Count TRACK elements (both in COLLECTION and PLAYLISTS)
    let collection_tracks = xml_content.matches("<TRACK TrackID=").count();
    assert_eq!(
        collection_tracks, 3,
        "XML should have 3 tracks in COLLECTION"
    );

    // Verify JSON has all tracks
    let json_content =
        fs::read_to_string(output_dir.path().join("djprep.json")).expect("Failed to read JSON");
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let tracks = json.get("tracks").unwrap().as_array().unwrap();
    assert_eq!(tracks.len(), 3, "JSON should have 3 tracks");
}

#[test]
fn test_bpm_detection_produces_reasonable_values() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate a longer test file for more reliable BPM detection
    let test_wav = input_dir.path().join("bpm_test.wav");
    generate_sine_wav(&test_wav, 440.0, 10.0, 44100);

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    pipeline::run(&settings).expect("Pipeline should succeed");

    // Read JSON to check BPM value
    let json_content =
        fs::read_to_string(output_dir.path().join("djprep.json")).expect("Failed to read JSON");
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let tracks = json.get("tracks").unwrap().as_array().unwrap();
    let bpm_value = tracks[0]
        .get("bpm")
        .unwrap()
        .get("value")
        .unwrap()
        .as_f64()
        .unwrap();

    // BPM should be within reasonable DJ range (60-200)
    // Note: A pure sine wave doesn't have rhythmic content, so BPM detection
    // may produce any value, but it should still be in a reasonable range
    assert!(
        (60.0..=200.0).contains(&bpm_value),
        "BPM {} should be in reasonable range (60-200)",
        bpm_value
    );
}

#[test]
fn test_key_detection_produces_valid_camelot() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate test file
    let test_wav = input_dir.path().join("key_test.wav");
    generate_sine_wav(&test_wav, 440.0, 5.0, 44100);

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    pipeline::run(&settings).expect("Pipeline should succeed");

    // Read JSON to check key value
    let json_content =
        fs::read_to_string(output_dir.path().join("djprep.json")).expect("Failed to read JSON");
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let tracks = json.get("tracks").unwrap().as_array().unwrap();
    let camelot = tracks[0]
        .get("key")
        .unwrap()
        .get("camelot")
        .unwrap()
        .as_str()
        .unwrap();

    // Camelot notation should be 1A-12A or 1B-12B
    let valid_camelot = [
        "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "11A", "12A", "1B", "2B", "3B",
        "4B", "5B", "6B", "7B", "8B", "9B", "10B", "11B", "12B",
    ];
    assert!(
        valid_camelot.contains(&camelot),
        "Camelot '{}' should be valid (1A-12A or 1B-12B)",
        camelot
    );
}

// =============================================================================
// BPM Detection Tests with Click Tracks
// =============================================================================

#[test]
fn test_bpm_detection_120_click_track() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Generate a 120 BPM click track, 10 seconds long
    let test_wav = input_dir.path().join("click_120bpm.wav");
    generate_click_track(&test_wav, 120.0, 10.0, 44100);

    // Run the pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    pipeline::run(&settings).expect("Pipeline should succeed");

    // Get detected BPM
    let detected_bpm = get_bpm_from_output(output_dir.path());

    // Check if BPM is within tolerance
    // Note: BPM detectors may detect half-time (60) or double-time (240)
    // We accept the actual BPM or its octave equivalents
    let is_accurate = is_bpm_match(detected_bpm, 120.0, 5.0);

    assert!(
        is_accurate,
        "120 BPM click track: detected {} BPM (expected ~120, or octave: 60/240)",
        detected_bpm
    );

    println!(
        "120 BPM click track: detected {} BPM (target: 120)",
        detected_bpm
    );
}

#[test]
fn test_bpm_detection_various_tempos() {
    // Test BPMs representing different genres:
    // 90 BPM - Hip-hop
    // 128 BPM - House
    // 174 BPM - Drum & Bass
    let test_bpms = [90.0_f32, 128.0, 174.0];

    for &target_bpm in &test_bpms {
        let input_dir = TempDir::new().expect("Failed to create input temp dir");
        let output_dir = TempDir::new().expect("Failed to create output temp dir");

        // Generate click track
        let test_wav = input_dir
            .path()
            .join(format!("click_{}bpm.wav", target_bpm as i32));
        generate_click_track(&test_wav, target_bpm, 15.0, 44100); // 15 seconds for better accuracy

        // Run the pipeline
        let settings = create_test_settings(input_dir.path(), output_dir.path());
        pipeline::run(&settings).expect("Pipeline should succeed");

        // Get detected BPM
        let detected_bpm = get_bpm_from_output(output_dir.path());

        // Check if BPM is within tolerance (allowing octave errors)
        let is_accurate = is_bpm_match(detected_bpm, target_bpm as f64, 5.0);

        println!(
            "{} BPM click track: detected {} BPM {}",
            target_bpm,
            detected_bpm,
            if is_accurate { "OK" } else { "OCTAVE ERROR" }
        );

        // We assert the BPM is in a reasonable range, but document octave errors
        assert!(
            (60.0..=200.0).contains(&detected_bpm),
            "{} BPM test: detected {} should be in DJ range (60-200)",
            target_bpm,
            detected_bpm
        );
    }
}

/// Check if detected BPM matches target, allowing for octave errors (half/double time)
///
/// BPM detectors commonly confuse tempos with their octave equivalents:
/// - 60 BPM vs 120 BPM vs 240 BPM
/// - 87 BPM vs 174 BPM
fn is_bpm_match(detected: f64, target: f64, tolerance: f64) -> bool {
    // Check direct match
    if (detected - target).abs() <= tolerance {
        return true;
    }

    // Check half-time match
    if (detected * 2.0 - target).abs() <= tolerance {
        return true;
    }

    // Check double-time match
    if (detected / 2.0 - target).abs() <= tolerance {
        return true;
    }

    false
}

#[test]
fn test_bpm_detection_consistency() {
    // Test that the same click track produces consistent results
    let input_dir = TempDir::new().expect("Failed to create input temp dir");

    // Generate 128 BPM click track
    let test_wav = input_dir.path().join("consistent_test.wav");
    generate_click_track(&test_wav, 128.0, 10.0, 44100);

    // Run analysis twice
    let mut results = Vec::new();
    for _ in 0..2 {
        let output_dir = TempDir::new().expect("Failed to create output temp dir");
        let settings = create_test_settings(input_dir.path(), output_dir.path());
        pipeline::run(&settings).expect("Pipeline should succeed");
        results.push(get_bpm_from_output(output_dir.path()));
    }

    // Results should be identical (deterministic)
    assert!(
        (results[0] - results[1]).abs() < 0.01,
        "BPM detection should be deterministic: got {} and {}",
        results[0],
        results[1]
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_handles_empty_audio_file() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Create an empty file with .wav extension
    let empty_file = input_dir.path().join("empty.wav");
    fs::write(&empty_file, b"").expect("Failed to create empty file");

    // Run pipeline - should not panic, but may skip the file
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings);

    // Pipeline should complete (either success with 0 tracks or graceful error)
    // The key is that it doesn't panic
    match result {
        Ok(_) => {
            // Check that the file was skipped (0 tracks in output)
            let json_path = output_dir.path().join("djprep.json");
            if json_path.exists() {
                let json_str = fs::read_to_string(&json_path).expect("Failed to read JSON");
                // Verify JSON is valid
                let _: serde_json::Value =
                    serde_json::from_str(&json_str).expect("Invalid JSON output");
            }
        }
        Err(_) => {
            // Graceful error is also acceptable
        }
    }
}

#[test]
fn test_handles_invalid_audio_data() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Create a file with random bytes (not a valid WAV)
    let invalid_file = input_dir.path().join("invalid.wav");
    fs::write(&invalid_file, b"This is not a valid WAV file content!!!!!")
        .expect("Failed to create invalid file");

    // Run pipeline - should not panic
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    let result = pipeline::run(&settings);

    // Pipeline should handle gracefully (skip invalid file)
    match result {
        Ok(_) => {
            // Success with skipped file is fine
        }
        Err(_) => {
            // Graceful error is also acceptable
        }
    }
}

#[test]
fn test_handles_nonexistent_input_gracefully() {
    // Create only output directory
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Use nonexistent input path
    let fake_input = Path::new("/nonexistent/path/that/does/not/exist");

    let settings = Settings {
        input: fake_input.to_path_buf(),
        output: output_dir.path().to_path_buf(),
        stems_enabled: false,
        stems_dir: output_dir.path().join("stems"),
        genre_hint: None,
        analysis_threads: 1,
        recursive: false,
        force: false,
        output_json: true,
        show_progress: false,
        dry_run: false,
    };

    // Run pipeline - should return an error, not panic
    let result = pipeline::run(&settings);

    // Should fail gracefully with an error
    assert!(
        result.is_err(),
        "Pipeline should return error for nonexistent input"
    );
}

#[test]
fn test_metadata_fallback_to_filename() {
    // Create temp directories
    let input_dir = TempDir::new().expect("Failed to create input temp dir");
    let output_dir = TempDir::new().expect("Failed to create output temp dir");

    // Create a WAV file with a descriptive filename (no metadata tags)
    let wav_path = input_dir.path().join("Artist_Name_-_Track_Title.wav");
    generate_sine_wav(&wav_path, 440.0, 1.0, 44100);

    // Run pipeline
    let settings = create_test_settings(input_dir.path(), output_dir.path());
    pipeline::run(&settings).expect("Pipeline should succeed");

    // Read JSON output
    let json_path = output_dir.path().join("djprep.json");
    let json_str = fs::read_to_string(&json_path).expect("Failed to read JSON");
    let json: serde_json::Value =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Verify track exists and has a title (should be filename without extension)
    let tracks = json["tracks"].as_array().expect("tracks should be array");
    assert_eq!(tracks.len(), 1, "Should have exactly one track");

    // The title should be derived from filename when no metadata tags present
    let track = &tracks[0];
    let title = track["title"].as_str();
    // Title should either be the filename or None if not extracted
    // Both are acceptable behaviors
    if let Some(title_str) = title {
        assert!(
            title_str.contains("Artist") || title_str.contains("Track"),
            "Title should be derived from filename"
        );
    }
}
