//! Runtime configuration settings

use std::path::PathBuf;

/// Runtime settings for the analysis pipeline
#[derive(Debug, Clone)]
pub struct Settings {
    /// Input path (file or directory)
    pub input: PathBuf,
    /// Output directory
    pub output: PathBuf,
    /// Enable stem separation
    pub stems_enabled: bool,
    /// Stems output directory
    pub stems_dir: PathBuf,
    /// Genre hint for BPM detection
    pub genre_hint: Option<String>,
    /// Number of analysis worker threads
    pub analysis_threads: usize,
    /// Scan recursively
    pub recursive: bool,
    /// Overwrite existing analysis
    pub force: bool,
    /// Output JSON
    pub output_json: bool,
    /// Show progress bars
    pub show_progress: bool,
    /// Dry run mode - show files without processing
    pub dry_run: bool,
}

impl Settings {
    /// Create settings from CLI arguments
    pub fn from_cli(cli: &super::cli::Cli) -> Self {
        let total_cores = num_cpus::get();

        // Reserve threads for stems (1) and export (1) if stems enabled
        let reserved = if cli.stems { 2 } else { 1 };
        let default_threads = total_cores.saturating_sub(reserved).max(1);

        let analysis_threads = cli.threads.unwrap_or(default_threads);

        Self {
            input: cli.input.clone(),
            output: cli.output.clone(),
            stems_enabled: cli.stems,
            stems_dir: cli.stems_dir(),
            genre_hint: cli.genre.clone(),
            analysis_threads,
            recursive: cli.recursive,
            force: cli.force,
            output_json: cli.json,
            show_progress: !cli.quiet,
            dry_run: cli.dry_run,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input: PathBuf::from("."),
            output: PathBuf::from("./output"),
            stems_enabled: false,
            stems_dir: PathBuf::from("./output/stems"),
            genre_hint: None,
            analysis_threads: num_cpus::get().saturating_sub(1).max(1),
            recursive: true,
            force: false,
            output_json: true,
            show_progress: true,
            dry_run: false,
        }
    }
}
