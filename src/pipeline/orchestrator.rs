//! Pipeline orchestration
//!
//! Coordinates file discovery, parallel analysis, and export.
//! Stem separation runs on a separate thread with bounded channel for backpressure.

use crate::analysis::{
    BpmDetector, KeyDetector, OrtStemSeparator, StemSeparator, StratumBpmDetector,
    StratumKeyDetector,
};
use crate::audio;
use crate::config::Settings;
use crate::discovery::{self, DiscoveredFile};
use crate::error::{DjprepError, Result};
use crate::export;
use crate::types::{AnalyzedTrack, StemPaths};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{debug, error, info, warn};

/// Pipeline result summary
#[derive(Debug)]
pub struct PipelineResult {
    pub total_files: usize,
    pub successful: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// Run the full analysis pipeline
pub fn run(settings: &Settings) -> Result<PipelineResult> {
    use std::time::Instant;

    let pipeline_start = Instant::now();

    // Configure thread pool
    configure_thread_pool(settings.analysis_threads)?;

    // Phase 1: Discovery
    let discovery_start = Instant::now();
    info!("Scanning for audio files...");
    let files = discovery::scan(&settings.input, settings.recursive)?;

    if files.is_empty() {
        return Ok(PipelineResult {
            total_files: 0,
            successful: 0,
            failed: 0,
            skipped: 0,
        });
    }

    let discovery_elapsed = discovery_start.elapsed();
    info!(
        "Found {} audio files in {:.2}s",
        files.len(),
        discovery_elapsed.as_secs_f64()
    );

    // Dry run mode - show files and exit
    if settings.dry_run {
        return run_dry_run(&files, settings);
    }

    // Check for existing analysis (for skip logic)
    let json_path = settings.output.join("djprep.json");
    let existing_paths = if settings.force {
        debug!("Force mode enabled, will re-analyze all files");
        std::collections::HashSet::new()
    } else {
        export::read_existing_analysis(&json_path)
    };

    // Filter files that need analysis
    let (files_to_analyze, skipped_existing): (Vec<_>, Vec<_>) = files
        .into_iter()
        .partition(|f| {
            let path_str = f.path.to_string_lossy().to_string();
            if existing_paths.contains(&path_str) {
                debug!("Skipping {} (already analyzed)", f.path.display());
                false
            } else {
                true
            }
        });

    let skipped_existing_count = skipped_existing.len();
    if skipped_existing_count > 0 {
        info!(
            "Skipping {} already-analyzed files (use --force to re-analyze)",
            skipped_existing_count
        );
    }

    let total_files = files_to_analyze.len() + skipped_existing_count;

    if files_to_analyze.is_empty() {
        info!("All files already analyzed, nothing to do");
        return Ok(PipelineResult {
            total_files,
            successful: 0,
            failed: 0,
            skipped: skipped_existing_count,
        });
    }

    info!("Analyzing {} files", files_to_analyze.len());

    // Phase 2: Analysis
    let analysis_start = std::time::Instant::now();
    let (tracks, stats) = analyze_files(&files_to_analyze, settings)?;
    let analysis_elapsed = analysis_start.elapsed();
    let tracks_per_sec = if analysis_elapsed.as_secs_f64() > 0.0 {
        files_to_analyze.len() as f64 / analysis_elapsed.as_secs_f64()
    } else {
        0.0
    };
    info!(
        "Analysis completed in {:.2}s ({:.1} tracks/sec)",
        analysis_elapsed.as_secs_f64(),
        tracks_per_sec
    );

    // Phase 3: Export
    if !tracks.is_empty() {
        let export_start = std::time::Instant::now();
        export_results(&tracks, settings)?;
        info!(
            "Export completed in {:.2}s",
            export_start.elapsed().as_secs_f64()
        );
    }

    info!(
        "Total pipeline time: {:.2}s",
        pipeline_start.elapsed().as_secs_f64()
    );

    Ok(PipelineResult {
        total_files,
        successful: stats.successful,
        failed: stats.failed,
        skipped: stats.skipped + skipped_existing_count,
    })
}

/// Dry run mode - show files that would be analyzed without processing
fn run_dry_run(files: &[DiscoveredFile], settings: &Settings) -> Result<PipelineResult> {
    use std::collections::HashMap;

    println!();
    println!("=== DRY RUN MODE ===");
    println!();

    // Group files by directory
    let mut by_directory: HashMap<PathBuf, Vec<&DiscoveredFile>> = HashMap::new();
    for file in files {
        let dir = file.path.parent().unwrap_or(&file.path).to_path_buf();
        by_directory.entry(dir).or_default().push(file);
    }

    // Sort directories
    let mut directories: Vec<_> = by_directory.keys().cloned().collect();
    directories.sort();

    // Group files by format for summary
    let mut by_format: HashMap<String, usize> = HashMap::new();
    for file in files {
        let ext = file
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_uppercase();
        *by_format.entry(ext).or_default() += 1;
    }

    // Print files grouped by directory
    for dir in &directories {
        let dir_files = &by_directory[dir];
        println!("{}/ ({} files)", dir.display(), dir_files.len());
        for file in dir_files {
            let filename = file
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            println!("  {}", filename);
        }
        println!();
    }

    // Print summary
    println!("─────────────────────────────────────────");
    println!();
    println!("Would analyze {} files:", files.len());

    // Print format breakdown
    let mut formats: Vec<_> = by_format.iter().collect();
    formats.sort_by(|a, b| b.1.cmp(a.1));
    for (format, count) in formats {
        println!("  {} {} files", count, format);
    }
    println!();

    // Estimate processing time (rough: ~3 seconds per file for BPM/key analysis)
    let estimate_secs = files.len() * 3;
    let estimate_mins = estimate_secs / 60;
    let remaining_secs = estimate_secs % 60;

    print!("Estimated time: ");
    if estimate_mins > 0 {
        print!("{}m ", estimate_mins);
    }
    println!("{}s", remaining_secs);

    // Show what outputs would be created
    println!();
    println!("Would create:");
    println!("  {}/rekordbox.xml", settings.output.display());
    if settings.output_json {
        println!("  {}/djprep.json", settings.output.display());
    }
    if settings.stems_enabled {
        println!("  {}/stems/*.wav (4 stems per track)", settings.output.display());
    }
    println!();

    Ok(PipelineResult {
        total_files: files.len(),
        successful: 0,
        failed: 0,
        skipped: files.len(), // All "skipped" in dry run mode
    })
}

/// Configure the Rayon thread pool
fn configure_thread_pool(num_threads: usize) -> Result<()> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(()) => {
            debug!("Configured thread pool with {} threads", num_threads);
        }
        Err(e) => {
            // If the pool is already initialized (e.g., in tests), that's OK
            if e.to_string().contains("already been initialized") {
                debug!("Thread pool already initialized, using existing pool");
            } else {
                return Err(DjprepError::ConfigError(format!(
                    "Failed to configure thread pool: {}",
                    e
                )));
            }
        }
    }
    Ok(())
}

/// Analysis statistics
struct AnalysisStats {
    successful: usize,
    failed: usize,
    skipped: usize,
}

/// Job for stem separation worker thread
struct StemJob {
    track_id: i32,
    input_path: PathBuf,
    output_dir: PathBuf,
}

/// Result from stem separation worker
struct StemResult {
    track_id: i32,
    stems: Option<StemPaths>,
}

/// Analyze files in parallel with optional stem separation
fn analyze_files(
    files: &[DiscoveredFile],
    settings: &Settings,
) -> Result<(Vec<AnalyzedTrack>, AnalysisStats)> {
    // Create analyzers using stratum-dsp
    let bpm_detector: Arc<dyn BpmDetector> =
        Arc::new(StratumBpmDetector::new(settings.genre_hint.clone()));
    let key_detector: Arc<dyn KeyDetector> = Arc::new(StratumKeyDetector::new());

    // Initialize stem separator if enabled
    let stem_separator: Option<Arc<dyn StemSeparator>> = if settings.stems_enabled {
        let separator = OrtStemSeparator::new();
        if separator.is_available() {
            info!("Stem separation enabled using {}", separator.name());
            Some(Arc::new(separator))
        } else {
            warn!("Stem separation requested but not available. Set DJPREP_MODEL_PATH to your .onnx model file.");
            None
        }
    } else {
        None
    };

    // Set up stem worker thread if stem separation is enabled
    // Channel capacity of 4 provides backpressure: when the GPU stem worker is busy,
    // analysis threads will block on send(), naturally throttling CPU work to match
    // GPU throughput. Capacity chosen to allow some buffering without excessive memory
    // use (~4 chunks × ~7.8s × 44.1kHz × 2ch × 4bytes ≈ 11MB peak buffer).
    const STEM_CHANNEL_CAPACITY: usize = 4;
    let (stem_tx, stem_rx): (Option<Sender<StemJob>>, Option<Receiver<StemJob>>) =
        if stem_separator.is_some() {
            let (tx, rx) = bounded::<StemJob>(STEM_CHANNEL_CAPACITY);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

    // Result channel is UNBOUNDED to prevent deadlock:
    // The stem worker sends results while main thread waits on join().
    // If result channel were bounded, the worker could block on send() while
    // main thread blocks on join(), causing deadlock. Unbounded channel allows
    // worker to complete all sends before main thread drains results.
    let (result_tx, result_rx): (Option<Sender<StemResult>>, Option<Receiver<StemResult>>) =
        if stem_separator.is_some() {
            let (tx, rx) = unbounded::<StemResult>();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

    // Spawn stem worker thread
    let stem_handle = if let (Some(rx), Some(result_tx), Some(separator)) =
        (stem_rx, result_tx.clone(), stem_separator.clone())
    {
        Some(thread::spawn(move || {
            stem_worker(rx, result_tx, separator);
        }))
    } else {
        None
    };

    // Progress tracking
    let multi_progress = MultiProgress::new();
    let progress_bar = if settings.show_progress {
        let pb = multi_progress.add(ProgressBar::new(files.len() as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Counters
    let successful = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    // Stems output directory
    let stems_dir = settings.output.join("stems");

    // Process files in parallel
    let tracks: Vec<AnalyzedTrack> = files
        .par_iter()
        .filter_map(|file| {
            let result = analyze_single_file(file, &bpm_detector, &key_detector);

            match result {
                Ok(track) => {
                    // Submit stem job if enabled
                    if let Some(ref tx) = stem_tx {
                        let job = StemJob {
                            track_id: track.track_id,
                            input_path: file.path.clone(),
                            output_dir: stems_dir.clone(),
                        };
                        // Use send_timeout to prevent deadlock if stem worker crashes.
                        // Timeout of 30 seconds is generous - if channel is blocked that long,
                        // the stem worker is likely dead. This allows rayon threads to continue
                        // rather than blocking forever.
                        use std::time::Duration;
                        match tx.send_timeout(job, Duration::from_secs(30)) {
                            Ok(()) => {}
                            Err(crossbeam_channel::SendTimeoutError::Timeout(_)) => {
                                warn!(
                                    "Stem job queue blocked for 30s, skipping stems for {}. \
                                     Stem worker may have crashed.",
                                    file.path.display()
                                );
                            }
                            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                                // Stem worker has shut down, channel closed
                                debug!("Stem channel closed, skipping stems for {}", file.path.display());
                            }
                        }
                    }

                    successful.fetch_add(1, Ordering::Relaxed);
                    if let Some(ref pb) = progress_bar {
                        pb.inc(1);
                        pb.set_message(format!(
                            "{}",
                            file.path.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    }
                    Some(track)
                }
                Err(e) => {
                    if e.is_recoverable() {
                        warn!("Skipping {}: {}", file.path.display(), e);
                        skipped.fetch_add(1, Ordering::Relaxed);
                    } else {
                        error!("Failed {}: {}", file.path.display(), e);
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Some(ref pb) = progress_bar {
                        pb.inc(1);
                    }
                    None
                }
            }
        })
        .collect();

    if let Some(pb) = progress_bar {
        pb.finish_with_message("Analysis complete");
    }

    // Close stem job channel to signal worker to finish
    drop(stem_tx);

    // Wait for stem worker to complete
    if let Some(handle) = stem_handle {
        match handle.join() {
            Ok(()) => {
                debug!("Stem worker thread completed successfully");
            }
            Err(panic_info) => {
                // Extract panic message if possible
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                error!(
                    "Stem worker thread panicked: {}. Some stems may not have been generated.",
                    panic_msg
                );
            }
        }
    }

    // Collect stem results and update tracks
    let mut tracks = tracks;
    if let Some(result_rx) = result_rx {
        // Drop the sender to allow receiver to drain
        drop(result_tx);

        for result in result_rx {
            if let Some(track) = tracks.iter_mut().find(|t| t.track_id == result.track_id) {
                track.stems = result.stems;
            }
        }
    }

    let stats = AnalysisStats {
        successful: successful.load(Ordering::Relaxed),
        failed: failed.load(Ordering::Relaxed),
        skipped: skipped.load(Ordering::Relaxed),
    };

    Ok((tracks, stats))
}

/// Worker thread for stem separation
fn stem_worker(
    rx: Receiver<StemJob>,
    tx: Sender<StemResult>,
    separator: Arc<dyn StemSeparator>,
) {
    for job in rx {
        debug!("Processing stems for track {}", job.track_id);

        let stems = match separator.separate(&job.input_path, &job.output_dir) {
            Ok(paths) => {
                info!(
                    "Stems created for {}",
                    job.input_path.file_name().unwrap_or_default().to_string_lossy()
                );
                Some(paths)
            }
            Err(e) => {
                warn!(
                    "Stem separation failed for {}: {}",
                    job.input_path.display(),
                    e
                );
                None
            }
        };

        let result = StemResult {
            track_id: job.track_id,
            stems,
        };

        if tx.send(result).is_err() {
            // Receiver dropped, we're shutting down
            break;
        }
    }
}

/// Minimum audio duration in seconds required for reliable analysis
/// stratum-dsp needs at least 3-5 seconds for BPM/key detection
const MIN_AUDIO_DURATION_SECS: f64 = 3.0;

/// Analyze a single file
fn analyze_single_file(
    file: &DiscoveredFile,
    bpm_detector: &Arc<dyn BpmDetector>,
    key_detector: &Arc<dyn KeyDetector>,
) -> Result<AnalyzedTrack> {
    debug!("Analyzing: {}", file.path.display());

    // Generate deterministic track ID
    let track_id = discovery::generate_track_id(&file.path);

    // Decode audio
    let buffer = audio::decode(&file.path)?;

    // Validate minimum duration for reliable analysis
    if buffer.duration < MIN_AUDIO_DURATION_SECS {
        return Err(DjprepError::AnalysisError {
            path: file.path.clone(),
            reason: format!(
                "Audio too short ({:.1}s). Minimum {:.0}s required for reliable BPM/key detection.",
                buffer.duration, MIN_AUDIO_DURATION_SECS
            ),
        });
    }

    // Run BPM detection
    let bpm = bpm_detector.detect(&buffer).map_err(|e| {
        // Add file context to analysis errors
        match e {
            DjprepError::AnalysisError { reason, .. } => DjprepError::AnalysisError {
                path: file.path.clone(),
                reason,
            },
            other => other,
        }
    })?;

    // Run key detection
    let key = key_detector.detect(&buffer).map_err(|e| {
        // Add file context to analysis errors
        match e {
            DjprepError::AnalysisError { reason, .. } => DjprepError::AnalysisError {
                path: file.path.clone(),
                reason,
            },
            other => other,
        }
    })?;

    // Create analyzed track
    let mut track = AnalyzedTrack::new(file.path.clone(), track_id);
    track.bpm = bpm;
    track.key = key;
    track.duration_seconds = buffer.duration;
    track.sample_rate = buffer.sample_rate;

    // Extract metadata from file tags
    track.metadata = crate::analysis::metadata::extract_metadata(&file.path);

    // Use filename as fallback if no title in tags
    if track.metadata.title.is_none() {
        track.metadata.title = file.path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string());
    }

    debug!(
        "Analyzed {}: BPM={:.1}, Key={}",
        file.path.file_name().unwrap_or_default().to_string_lossy(),
        track.bpm.value,
        track.key.camelot
    );

    Ok(track)
}

/// Export analysis results
fn export_results(tracks: &[AnalyzedTrack], settings: &Settings) -> Result<()> {
    // Ensure output directory exists
    std::fs::create_dir_all(&settings.output).map_err(|e| DjprepError::OutputError {
        path: settings.output.clone(),
        reason: e.to_string(),
    })?;

    // Write Rekordbox XML
    let xml_path = settings.output.join("rekordbox.xml");
    export::write_rekordbox_xml(tracks, &xml_path)?;

    // Write JSON if enabled
    if settings.output_json {
        let json_path = settings.output.join("djprep.json");
        export::write_json(tracks, &json_path)?;
    }

    // Print import instructions
    print_import_instructions(&xml_path, tracks.len());

    Ok(())
}

/// Print instructions for importing into Rekordbox
fn print_import_instructions(xml_path: &std::path::Path, track_count: usize) {
    println!();
    println!("✓ Wrote {} tracks to {}", track_count, xml_path.display());
    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ IMPORTANT: Rekordbox Import Instructions                        │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ To correctly import metadata (including updated keys):          │");
    println!("│                                                                 │");
    println!("│ 1. In Rekordbox: File → Import Collection → Select rekordbox.xml│");
    println!("│ 2. In the tree view, find the \"djprep_import_*\" playlist       │");
    println!("│ 3. Right-click the playlist → \"Import Playlist\"                │");
    println!("│                                                                 │");
    println!("│ ⚠ Direct drag-and-drop will NOT update existing track metadata │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}
