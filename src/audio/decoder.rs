//! Audio decoding using symphonia
//!
//! Decodes audio files to mono f32 samples at the target sample rate.

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

/// Target sample rate for analysis (22050 Hz)
/// 
/// This is sufficient for BPM and key detection (frequencies < 11kHz)
/// while reducing computation by 50% compared to 44.1kHz
pub const TARGET_SAMPLE_RATE: u32 = 22050;

/// Decode an audio file to a mono AudioBuffer
pub fn decode(path: &Path) -> Result<AudioBuffer> {
    let file = std::fs::File::open(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to open file: {}", e),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Provide a hint based on file extension
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to probe format: {}", e),
        })?;

    let mut format = probed.format;

    // Find the first audio track
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

    let source_sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

    debug!(
        "Decoding: {} @ {}Hz, {} channels",
        path.display(),
        source_sample_rate,
        channels
    );

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to create decoder: {}", e),
        })?;

    // Collect all samples
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break; // End of stream
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Failed to read packet: {}", e),
                });
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode packet
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                // Skip corrupted frames
                trace!("Skipping corrupted frame: {}", e);
                continue;
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Decode error: {}", e),
                });
            }
        };

        // Convert to f32 samples
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Convert to mono by averaging channels
        let mono_samples = to_mono(samples, channels);
        all_samples.extend(mono_samples);
    }

    // Resample to target rate if needed
    let final_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        resample(&all_samples, source_sample_rate, TARGET_SAMPLE_RATE)
    } else {
        all_samples
    };

    debug!(
        "Decoded {} samples ({:.2}s)",
        final_samples.len(),
        final_samples.len() as f64 / TARGET_SAMPLE_RATE as f64
    );

    Ok(AudioBuffer::new(final_samples, TARGET_SAMPLE_RATE))
}

/// Convert interleaved multi-channel audio to mono
fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Simple linear interpolation resampler
///
/// For production, consider using rubato crate for higher quality
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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
            // Linear interpolation
            samples[src_idx] * (1.0 - frac as f32) + samples[src_idx + 1] * frac as f32
        } else {
            samples[src_idx.min(samples.len() - 1)]
        };

        output.push(sample);
    }

    output
}

/// Target sample rate for stem separation (44100 Hz - standard CD quality)
pub const STEM_SAMPLE_RATE: u32 = 44100;

/// Decode an audio file to stereo at full fidelity for stem separation
///
/// Unlike `decode()`, this preserves stereo channels and uses 44.1kHz
/// which is required by HTDemucs and other stem separation models.
pub fn decode_stereo(path: &Path) -> Result<StereoBuffer> {
    let file = std::fs::File::open(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to open file: {}", e),
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Provide a hint based on file extension
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to probe format: {}", e),
        })?;

    let mut format = probed.format;

    // Find the first audio track
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

    let source_sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

    debug!(
        "Decoding stereo: {} @ {}Hz, {} channels",
        path.display(),
        source_sample_rate,
        channels
    );

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| DjprepError::DecodeError {
            path: path.to_path_buf(),
            reason: format!("Failed to create decoder: {}", e),
        })?;

    // Collect all interleaved samples
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break; // End of stream
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Failed to read packet: {}", e),
                });
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode packet
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                // Skip corrupted frames
                trace!("Skipping corrupted frame: {}", e);
                continue;
            }
            Err(e) => {
                return Err(DjprepError::DecodeError {
                    path: path.to_path_buf(),
                    reason: format!("Decode error: {}", e),
                });
            }
        };

        // Convert to f32 samples
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend(sample_buf.samples());
    }

    // Convert to stereo buffer
    let stereo = if channels == 1 {
        // Mono: duplicate to stereo
        StereoBuffer::new(all_samples.clone(), all_samples, source_sample_rate)
    } else if channels == 2 {
        // Already stereo
        StereoBuffer::from_interleaved(&all_samples, source_sample_rate)
    } else {
        // Multi-channel: downmix to stereo
        let stereo_samples = downmix_to_stereo(&all_samples, channels);
        StereoBuffer::from_interleaved(&stereo_samples, source_sample_rate)
    };

    // Resample to 44.1kHz if needed
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

/// Downmix multi-channel audio to stereo
fn downmix_to_stereo(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 2 {
        return samples.to_vec();
    }

    // Simple downmix: average left channels (0, 2, 4...) and right channels (1, 3, 5...)
    let num_frames = samples.len() / channels;
    let mut stereo = Vec::with_capacity(num_frames * 2);

    for frame in samples.chunks(channels) {
        // Average odd indices for left, even for right (or vice versa)
        // For 5.1 surround: FL, FR, FC, LFE, BL, BR
        // Simple approach: mix front channels
        let left = if channels >= 1 { frame[0] } else { 0.0 };
        let right = if channels >= 2 { frame[1] } else { left };

        stereo.push(left);
        stereo.push(right);
    }

    stereo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono_stereo() {
        let stereo = vec![0.5, 0.3, 0.8, 0.2, 1.0, 0.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.4).abs() < 0.001); // (0.5 + 0.3) / 2
        assert!((mono[1] - 0.5).abs() < 0.001); // (0.8 + 0.2) / 2
        assert!((mono[2] - 0.5).abs() < 0.001); // (1.0 + 0.0) / 2
    }

    #[test]
    fn test_to_mono_already_mono() {
        let mono = vec![0.5, 0.8, 1.0];
        let result = to_mono(&mono, 1);
        assert_eq!(result, mono);
    }

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
        // Should be approximately half the length
        assert!((result.len() as f64 - 500.0).abs() < 2.0);
    }
}
