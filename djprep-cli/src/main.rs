//! djprep CLI entry point

use clap::Parser;
use djprep::config::{Cli, Settings};
use djprep::pipeline;
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    init_logging(&cli);

    // Build settings from CLI
    let settings = Settings::from_cli(&cli);

    // Validate inputs
    if let Err(e) = validate_inputs(&cli) {
        eprintln!("Error: {}", e);
        return ExitCode::FAILURE;
    }

    // Run the pipeline
    match pipeline::run(&settings) {
        Ok(result) => {
            println!();
            println!(
                "Summary: {} successful, {} failed, {} skipped (of {} total)",
                result.successful, result.failed, result.skipped, result.total_files
            );

            if result.failed > 0 {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("Fatal error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn init_logging(cli: &Cli) {
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let filter = if cli.quiet { "error" } else { filter };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();
}

fn validate_inputs(cli: &Cli) -> Result<(), String> {
    // Check input exists
    if !cli.input.exists() {
        return Err(format!(
            "Input path does not exist: {}\n\n  Tip: Check the path is correct and accessible.\n  Examples:\n    djprep -i ~/Music/DJ -o ./analyzed\n    djprep -i ./track.mp3 -o ./output",
            cli.input.display()
        ));
    }

    // Check output parent directory exists (we'll create the output dir itself)
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(format!(
                "Output parent directory does not exist: {}\n\n  Tip: The output directory will be created automatically,\n  but its parent directory must exist.\n  Example: mkdir -p {}",
                parent.display(),
                parent.display()
            ));
        }
    }

    Ok(())
}
