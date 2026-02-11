//! CLI argument parsing and configuration

use clap::Parser;
use std::path::PathBuf;

/// djprep - High-performance audio analysis for DJs
///
/// Analyzes audio files to extract BPM, musical key, and optionally separates
/// stems. Outputs Rekordbox-compatible XML and JSON formats.
#[derive(Parser, Debug)]
#[command(name = "djprep")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Input path (file or directory)
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output directory for XML/JSON files
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Enable stem separation (vocals, drums, bass, other)
    #[arg(long, default_value = "false")]
    pub stems: bool,

    /// Directory to write separated stems (defaults to output/stems)
    #[arg(long, value_name = "DIR")]
    pub stems_output: Option<PathBuf>,

    /// Genre hint for BPM detection (helps resolve double-tempo ambiguity)
    #[arg(long, value_name = "GENRE")]
    #[arg(value_parser = ["house", "techno", "trance", "dnb", "dubstep", "hiphop", "pop", "rock"])]
    pub genre: Option<String>,

    /// Number of worker threads (defaults to CPU count - 2)
    #[arg(short = 'j', long, value_name = "N")]
    pub threads: Option<usize>,

    /// Scan subdirectories recursively
    #[arg(short, long, default_value = "true")]
    pub recursive: bool,

    /// Overwrite existing analysis (by default, skips already-analyzed files)
    #[arg(long, default_value = "false")]
    pub force: bool,

    /// Output JSON in addition to Rekordbox XML
    #[arg(long, default_value = "true")]
    pub json: bool,

    /// Verbose output (can be repeated: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (suppress progress bars)
    #[arg(short, long, default_value = "false")]
    pub quiet: bool,

    /// Dry run - show files that would be analyzed without processing
    #[arg(long, default_value = "false")]
    pub dry_run: bool,
}

impl Cli {
    /// Get the effective stems output directory
    pub fn stems_dir(&self) -> PathBuf {
        self.stems_output
            .clone()
            .unwrap_or_else(|| self.output.join("stems"))
    }

    /// Get the log level based on verbosity flags
    pub fn log_level(&self) -> tracing::Level {
        match self.verbose {
            0 => tracing::Level::WARN,
            1 => tracing::Level::INFO,
            2 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        }
    }
}
