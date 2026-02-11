//! Audio decoding using symphonia.
//!
//! Mono analysis decoding is delegated to `djprep-core`.
//! Stereo decoding for stem separation remains CLI-local.

use crate::error::{DjprepError, Result};
use crate::types::{AudioBuffer, StereoBuffer};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::{debug, trace};

/// Target sample rate for analysis (22050 Hz).
pub const TARGET_SAMPLE_RATE: u32 = djprep_core::TARGET_SAMPLE_RATE;

/// Maximum file size we'll attempt to decode (2GB).
/// Prevents OOM on extremely large files.
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Default sample rate if not specified in codec metadata.
const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// Default channel count if not specified in codec metadata.
const DEFAULT_CHANNELS: usize = 2;

/// Decode an audio file to a mono AudioBuffer.
pub fn decode(path: &Path) -> Result<AudioBuffer> {
    let metadata = std::fs::metadata(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to read file metadata: {e}"),
    })?;

    if metadata.len() > MAX_FILE_SIZE {
        return Err(DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!(
                "File too large ({:.1} GB). Maximum supported size is 2 GB.",
                metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
        });
    }

    let file = std::fs::File::open(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to open file: {e}"),
    })?;

    let ext = path.extension().and_then(|e| e.to_str());
    djprep_core::decode_reader(file, ext).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

/// Target sample rate for stem separation (44100 Hz - standard CD quality).
pub const STEM_SAMPLE_RATE: u32 = 44100;

/// Decode an audio file to stereo at full fidelity for stem separation.
///
/// Unlike `decode()`, this preserves stereo channels and uses 44.1kHz
/// which is required by HTDemucs and other stem separation models.
pub fn decode_stereo(path: &Path) -> Result<StereoBuffer> {
    let file = std::fs::File::open(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to open file: {e}"),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to probe format: {e}"),
        })?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: "No audio tracks found".to_string(),
        })?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_sample_rate = codec_params.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE);
    let channels = codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(DEFAULT_CHANNELS);

    debug!(
        "Decoding stereo: {} @ {}Hz, {} channels",
        path.display(),
        source_sample_rate,
        channels
    );

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to create decoder: {e}"),
        })?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Failed to read packet: {e}"),
                });
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                trace!("Skipping corrupted frame: {e}");
                continue;
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Decode error: {e}"),
                });
            }
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend(sample_buf.samples());
    }

    let stereo = if channels == 1 {
        StereoBuffer::new(all_samples.clone(), all_samples, source_sample_rate)
    } else if channels == 2 {
        StereoBuffer::from_interleaved(&all_samples, source_sample_rate)
    } else {
        let stereo_samples = downmix_to_stereo(&all_samples, channels);
        StereoBuffer::from_interleaved(&stereo_samples, source_sample_rate)
    };

    let final_stereo = if source_sample_rate != STEM_SAMPLE_RATE {
        let left = resample(&stereo.left, source_sample_rate, STEM_SAMPLE_RATE);
        let right = resample(&stereo.right, source_sample_rate, STEM_SAMPLE_RATE);
        StereoBuffer::new(left, right, STEM_SAMPLE_RATE)
    } else {
        stereo
    };

    debug!(
        "Decoded stereo {} samples ({:.2}s)",
        final_stereo.len(),
        final_stereo.duration
    );

    Ok(final_stereo)
}

/// Downmix multi-channel audio to stereo.
fn downmix_to_stereo(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 2 {
        return samples.to_vec();
    }

    let num_frames = samples.len() / channels;
    let mut stereo = Vec::with_capacity(num_frames * 2);

    for frame in samples.chunks(channels) {
        let left = if channels >= 1 { frame[0] } else { 0.0 };
        let right = if channels >= 2 { frame[1] } else { left };

        stereo.push(left);
        stereo.push(right);
    }

    stereo
}

/// Simple linear interpolation resampler.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == 0 || to_rate == 0 {
        debug!(
            "Invalid sample rate for resample (from={}, to={}), skipping",
            from_rate, to_rate
        );
        return samples.to_vec();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] * (1.0 - frac as f32) + samples[src_idx + 1] * frac as f32
        } else {
            samples[src_idx.min(samples.len() - 1)]
        };

        output.push(sample);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = resample(&samples, 44100, 44100);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let result = resample(&samples, 44100, 22050);
        assert!((result.len() as f64 - 500.0).abs() < 2.0);
    }
}
