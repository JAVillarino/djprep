//! ONNX Runtime based stem separator
//!
//! Uses HTDemucs to separate audio into vocals, drums, bass, and other stems.

use crate::analysis::traits::StemSeparator;
use crate::error::{DjprepError, Result};
use crate::types::StemPaths;
#[cfg(feature = "stems")]
use crate::types::StereoBuffer;
use std::path::{Path, PathBuf};
#[cfg(feature = "stems")]
use std::sync::Mutex;
#[allow(unused_imports)]
use tracing::{debug, info, warn};

#[cfg(feature = "stems")]
use super::chunking::{chunk_audio, overlap_add, ChunkConfig, StemChunk};
#[cfg(feature = "stems")]
use super::model::find_model_path;
#[cfg(feature = "stems")]
use ort::session::Session;

/// ONNX Runtime based stem separator using HTDemucs
pub struct OrtStemSeparator {
    /// Path to the ONNX model
    #[allow(dead_code)]
    model_path: Option<PathBuf>,
    /// ORT session (wrapped in Mutex for interior mutability)
    #[cfg(feature = "stems")]
    session: Option<Mutex<Session>>,
    #[cfg(not(feature = "stems"))]
    #[allow(dead_code)]
    session: Option<()>,
    /// Whether the separator is available
    available: bool,
    /// Execution provider being used
    #[allow(dead_code)]
    execution_provider: String,
}

impl OrtStemSeparator {
    /// Create a new stem separator
    ///
    /// This will attempt to:
    /// 1. Load or download the HTDemucs model
    /// 2. Initialize ONNX Runtime with the best available execution provider
    #[cfg(feature = "stems")]
    pub fn new() -> Self {
        match Self::initialize() {
            Ok(separator) => separator,
            Err(e) => {
                warn!("Stem separator initialization failed: {}", e);
                Self {
                    model_path: None,
                    session: None,
                    available: false,
                    execution_provider: "none".to_string(),
                }
            }
        }
    }

    #[cfg(not(feature = "stems"))]
    pub fn new() -> Self {
        Self {
            model_path: None,
            session: None,
            available: false,
            execution_provider: "none".to_string(),
        }
    }

    #[cfg(feature = "stems")]
    fn initialize() -> Result<Self> {
        // Find model using multi-location search
        let model_path = find_model_path()?;

        // Determine best execution provider
        let execution_provider = Self::detect_best_provider();

        // Create ORT session
        let session = Self::create_session(&model_path, &execution_provider)?;

        info!(
            "Stem separator initialized with {} provider, model: {}",
            execution_provider,
            model_path.display()
        );

        Ok(Self {
            model_path: Some(model_path),
            session: Some(Mutex::new(session)),
            available: true,
            execution_provider,
        })
    }

    /// Create ORT session with the specified execution provider
    #[cfg(feature = "stems")]
    fn create_session(model_path: &Path, ep: &str) -> Result<Session> {
        use ort::execution_providers::CPUExecutionProvider;

        let builder = Session::builder().map_err(|e| DjprepError::StemUnavailable {
            reason: format!("Failed to create ORT session builder: {}", e),
        })?;

        // Configure execution providers based on detected provider
        let session = match ep {
            #[cfg(target_os = "macos")]
            "CoreML" => {
                use ort::execution_providers::CoreMLExecutionProvider;
                builder
                    .with_execution_providers([
                        CoreMLExecutionProvider::default().build(),
                        CPUExecutionProvider::default().build(),
                    ])
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to configure CoreML: {}", e),
                    })?
                    .commit_from_file(model_path)
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to load model with CoreML: {}", e),
                    })?
            }
            #[cfg(target_os = "windows")]
            "DirectML" => {
                use ort::execution_providers::DirectMLExecutionProvider;
                builder
                    .with_execution_providers([
                        DirectMLExecutionProvider::default().build(),
                        CPUExecutionProvider::default().build(),
                    ])
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to configure DirectML: {}", e),
                    })?
                    .commit_from_file(model_path)
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to load model with DirectML: {}", e),
                    })?
            }
            _ => {
                // CPU fallback
                builder
                    .with_execution_providers([CPUExecutionProvider::default().build()])
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to configure CPU provider: {}", e),
                    })?
                    .commit_from_file(model_path)
                    .map_err(|e| DjprepError::StemUnavailable {
                        reason: format!("Failed to load model: {}", e),
                    })?
            }
        };

        Ok(session)
    }

    /// Detect the best available execution provider
    #[cfg(feature = "stems")]
    #[allow(clippy::needless_return)] // Returns needed due to cfg conditional compilation
    fn detect_best_provider() -> String {
        // CoreML (Apple Silicon)
        #[cfg(target_os = "macos")]
        {
            return "CoreML".to_string();
        }

        // DirectML (Windows)
        #[cfg(target_os = "windows")]
        {
            return "DirectML".to_string();
        }

        // Fallback to CPU
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        "CPU".to_string()
    }
}

impl Default for OrtStemSeparator {
    fn default() -> Self {
        Self::new()
    }
}

impl StemSeparator for OrtStemSeparator {
    #[allow(clippy::needless_return)] // Return needed due to cfg conditional compilation
    fn separate(&self, input_path: &Path, output_dir: &Path) -> Result<StemPaths> {
        if !self.available {
            return Err(DjprepError::StemUnavailable {
                reason: "Stem separator not initialized. Set DJPREP_MODEL_PATH or build with --features stems".to_string(),
            });
        }

        debug!(
            "Separating stems for {} -> {}",
            input_path.display(),
            output_dir.display()
        );

        // Create output directory
        std::fs::create_dir_all(output_dir).map_err(|e| DjprepError::OutputError {
            path: output_dir.to_path_buf(),
            reason: format!("Failed to create stems output directory: {}", e),
        })?;

        // Get the base filename
        let stem_base = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("track");

        // Generate output paths
        let stems = StemPaths {
            vocals: output_dir.join(format!("{}_vocals.wav", stem_base)),
            drums: output_dir.join(format!("{}_drums.wav", stem_base)),
            bass: output_dir.join(format!("{}_bass.wav", stem_base)),
            other: output_dir.join(format!("{}_other.wav", stem_base)),
        };

        // Perform stem separation
        #[cfg(feature = "stems")]
        {
            self.run_inference(input_path, &stems)?;
            return Ok(stems);
        }

        #[cfg(not(feature = "stems"))]
        {
            let _ = stems;
            Err(DjprepError::StemUnavailable {
                reason: "Stem separation requires the 'stems' feature".to_string(),
            })
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "htdemucs-ort"
    }
}

#[cfg(feature = "stems")]
impl OrtStemSeparator {
    /// Run ONNX inference to separate stems
    fn run_inference(&self, input_path: &Path, stem_paths: &StemPaths) -> Result<()> {
        use crate::audio;
        use ndarray::Array3;
        use ort::value::Tensor;

        // Lock the session mutex for mutable access
        let session_mutex = self.session.as_ref().ok_or_else(|| DjprepError::StemUnavailable {
            reason: "ORT session not initialized".to_string(),
        })?;

        let mut session = session_mutex.lock().map_err(|_| DjprepError::StemUnavailable {
            reason: "Failed to acquire session lock".to_string(),
        })?;

        // Decode input audio at full fidelity (44.1kHz stereo)
        info!("Loading audio for stem separation...");
        let audio = audio::decode_stereo(input_path)?;
        let total_samples = audio.len();

        info!(
            "Processing {:.2}s of audio ({} samples at {}Hz)",
            audio.duration, total_samples, audio.sample_rate
        );

        // Chunk audio for processing
        let config = ChunkConfig::htdemucs();
        let chunks = chunk_audio(&audio, &config);

        info!("Processing {} chunks...", chunks.len());

        // Process each chunk
        let mut stem_chunks: Vec<StemChunk> = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            debug!("Processing chunk {}/{}", i + 1, chunks.len());

            // Prepare input tensor: shape (batch=1, channels=2, samples)
            let chunk_len = chunk.audio.len();
            let mut input_data = Array3::<f32>::zeros((1, 2, chunk_len));

            for j in 0..chunk_len {
                input_data[[0, 0, j]] = chunk.audio.left[j];
                input_data[[0, 1, j]] = chunk.audio.right[j];
            }

            // Create input tensor (ort 2.0 needs owned array)
            let input_tensor = Tensor::from_array(input_data)
                .map_err(|e| DjprepError::StemUnavailable {
                    reason: format!("Failed to create input tensor: {}", e),
                })?;

            // Get input name from session
            let input_name = session
                .inputs
                .first()
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "input".to_string());

            // Run inference
            let inputs = ort::inputs![input_name.as_str() => input_tensor];
            let outputs = session
                .run(inputs)
                .map_err(|e| DjprepError::StemUnavailable {
                    reason: format!("Inference failed: {}", e),
                })?;

            // Extract output: shape (batch=1, stems=4, channels=2, samples)
            // Stems order: vocals, drums, bass, other
            let output = outputs
                .iter()
                .next()
                .map(|(_, v)| v)
                .ok_or_else(|| DjprepError::StemUnavailable {
                    reason: "No output tensor from model".to_string(),
                })?;

            let (output_shape, output_data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| DjprepError::StemUnavailable {
                    reason: format!("Failed to extract output tensor: {}", e),
                })?;

            let shape: Vec<i64> = output_shape.iter().copied().collect();

            // Validate output shape
            if shape.len() != 4 || shape[1] != 4 || shape[2] != 2 {
                return Err(DjprepError::StemUnavailable {
                    reason: format!(
                        "Unexpected output shape {:?}, expected (1, 4, 2, samples)",
                        shape
                    ),
                });
            }

            let output_samples = shape[3] as usize;
            let num_channels = shape[2] as usize;

            // Extract stems from flat tensor data
            // Layout: [batch, stems, channels, samples] in row-major order
            let extract_stem = |stem_idx: usize| -> StereoBuffer {
                let mut left = Vec::with_capacity(output_samples);
                let mut right = Vec::with_capacity(output_samples);

                for s in 0..output_samples {
                    // Calculate flat index for row-major 4D array
                    // Layout: [stem_idx][channel][sample] where channel 0=left, 1=right
                    let stem_offset = stem_idx * num_channels * output_samples;
                    let left_idx = stem_offset + s;
                    let right_idx = stem_offset + output_samples + s;

                    left.push(output_data[left_idx]);
                    right.push(output_data[right_idx]);
                }

                StereoBuffer::new(left, right, chunk.audio.sample_rate)
            };

            stem_chunks.push(StemChunk {
                index: chunk.index,
                start_sample: chunk.start_sample,
                end_sample: chunk.end_sample,
                vocals: extract_stem(0),
                drums: extract_stem(1),
                bass: extract_stem(2),
                other: extract_stem(3),
            });
        }

        // Reassemble stems using overlap-add
        info!("Reassembling stems...");
        let stems = overlap_add(&stem_chunks, &config, total_samples);

        // Write output WAV files
        info!("Writing stem files...");
        self.write_stereo_wav(&stem_paths.vocals, &stems.vocals)?;
        self.write_stereo_wav(&stem_paths.drums, &stems.drums)?;
        self.write_stereo_wav(&stem_paths.bass, &stems.bass)?;
        self.write_stereo_wav(&stem_paths.other, &stems.other)?;

        info!("Stem separation complete");
        Ok(())
    }

    /// Write stereo audio to a WAV file
    fn write_stereo_wav(&self, path: &Path, audio: &StereoBuffer) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: audio.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer =
            hound::WavWriter::create(path, spec).map_err(|e| DjprepError::OutputError {
                path: path.to_path_buf(),
                reason: format!("Failed to create WAV file: {}", e),
            })?;

        // Write interleaved stereo samples
        for (l, r) in audio.left.iter().zip(audio.right.iter()) {
            let l_i16 = (*l * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let r_i16 = (*r * 32767.0).clamp(-32768.0, 32767.0) as i16;

            writer
                .write_sample(l_i16)
                .map_err(|e| DjprepError::OutputError {
                    path: path.to_path_buf(),
                    reason: format!("Failed to write sample: {}", e),
                })?;
            writer
                .write_sample(r_i16)
                .map_err(|e| DjprepError::OutputError {
                    path: path.to_path_buf(),
                    reason: format!("Failed to write sample: {}", e),
                })?;
        }

        writer
            .finalize()
            .map_err(|e| DjprepError::OutputError {
                path: path.to_path_buf(),
                reason: format!("Failed to finalize WAV: {}", e),
            })?;

        debug!("Wrote stem to {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separator_name() {
        let separator = OrtStemSeparator::new();
        assert_eq!(separator.name(), "htdemucs-ort");
    }
}
