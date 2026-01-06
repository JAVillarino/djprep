//! Audio decoding using symphonia
//!
//! Decodes audio files to mono f32 samples at the target sample rate.
//! Uses rubato for high-quality resampling with proper anti-aliasing.

use crate::error::{DjprepError, Result};
use crate::types::{AudioBuffer, StereoBuffer};
use rubato::{FftFixedInOut, Resampler};
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

/// Maximum file size we'll attempt to decode (2GB)
/// Prevents OOM on extremely large files
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Decode an audio file to a mono AudioBuffer
pub fn decode(path: &Path) -> Result<AudioBuffer> {
    // Check file size before attempting to decode
    let metadata = std::fs::metadata(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to read file metadata: {}", e),
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
    // Handle edge cases that would cause division by zero or incorrect behavior
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|frame| {
            // Use actual frame length to avoid division by zero on incomplete frames
            if frame.is_empty() {
                0.0
            } else {
                frame.iter().sum::<f32>() / frame.len() as f32
            }
        })
        .collect()
}

/// High-quality audio resampling using rubato
///
/// Uses FFT-based resampling with proper anti-aliasing filter to prevent
/// aliasing artifacts when downsampling. This is critical for accurate
/// frequency analysis in BPM and key detection.
///
/// The resampler uses a sinc interpolation kernel with configurable quality.
/// For analysis (mono, 22kHz), we use moderate quality settings that balance
/// accuracy with performance.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    // Handle edge cases that would cause division by zero or incorrect behavior
    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == 0 || to_rate == 0 {
        // Invalid sample rates - return original samples rather than panic
        debug!("Invalid sample rate (from={}, to={}), skipping resample", from_rate, to_rate);
        return samples.to_vec();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    // Calculate chunk size - rubato works on fixed-size chunks
    // Use a reasonable chunk size for efficiency
    const CHUNK_SIZE: usize = 1024;

    // Create resampler with FFT-based algorithm
    // FftFixedInOut provides good quality with reasonable performance
    let mut resampler = match FftFixedInOut::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        CHUNK_SIZE,
        1, // mono channel
    ) {
        Ok(r) => r,
        Err(e) => {
            // Fallback to simple resampling if rubato fails to initialize
            debug!("Rubato initialization failed ({}), using fallback", e);
            return resample_linear_fallback(samples, from_rate, to_rate);
        }
    };

    let input_frames_per_chunk = resampler.input_frames_next();
    let output_frames_per_chunk = resampler.output_frames_next();

    // Validate rubato returned valid frame counts
    if input_frames_per_chunk == 0 || output_frames_per_chunk == 0 {
        debug!("Rubato returned zero frame count, using fallback");
        return resample_linear_fallback(samples, from_rate, to_rate);
    }

    // Estimate output size with overflow protection
    let ratio = to_rate as f64 / from_rate as f64;
    let estimated_output_len = ((samples.len() as f64 * ratio).ceil() as usize)
        .min(samples.len().saturating_mul(4)); // Cap at 4x input length as sanity check
    let mut output = Vec::with_capacity(estimated_output_len);

    // Process in chunks using saturating arithmetic to prevent overflow
    let mut pos = 0;
    while pos < samples.len() {
        let end = pos.saturating_add(input_frames_per_chunk).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();

        // Pad last chunk if needed
        if chunk.len() < input_frames_per_chunk {
            chunk.resize(input_frames_per_chunk, 0.0);
        }

        // Rubato expects Vec<Vec<f32>> for multi-channel, we have mono
        let input_channels = vec![chunk];

        match resampler.process(&input_channels, None) {
            Ok(resampled) => {
                if let Some(channel) = resampled.first() {
                    // Only take valid samples (not padding)
                    let valid_samples = if pos.saturating_add(input_frames_per_chunk) > samples.len() {
                        // Last chunk - calculate how many output samples are valid
                        let input_valid = samples.len().saturating_sub(pos);
                        let output_valid = (input_valid as f64 * ratio).ceil() as usize;
                        output_valid.min(output_frames_per_chunk)
                    } else {
                        output_frames_per_chunk
                    };
                    // Guard against floating-point rounding causing out-of-bounds
                    let safe_samples = valid_samples.min(channel.len());
                    output.extend_from_slice(&channel[..safe_samples]);
                }
            }
            Err(e) => {
                debug!("Rubato processing error ({}), using fallback for remaining", e);
                // Fallback for remaining samples
                let remaining = resample_linear_fallback(&samples[pos..], from_rate, to_rate);
                output.extend(remaining);
                break;
            }
        }

        pos = pos.saturating_add(input_frames_per_chunk);
    }

    output
}

/// Fallback linear interpolation resampler
///
/// Used only when rubato fails to initialize or process. This is a simple
/// linear interpolation that may introduce aliasing artifacts.
fn resample_linear_fallback(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    // Handle edge cases that would cause division by zero or out-of-bounds access
    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == 0 || to_rate == 0 {
        // Invalid sample rates - return original samples rather than panic
        return samples.to_vec();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    // Protect against ratio producing zero or huge output lengths
    if ratio <= 0.0 || !ratio.is_finite() {
        return samples.to_vec();
    }

    let output_len = (samples.len() as f64 / ratio) as usize;
    // Sanity check: output shouldn't be more than 10x input (very generous)
    let output_len = output_len.min(samples.len().saturating_mul(10));
    if output_len == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        // Safe indexing with bounds checking
        let sample = if src_idx >= samples.len() {
            // Past end of samples - use last sample
            *samples.last().unwrap() // Safe: we checked is_empty() above
        } else if src_idx + 1 < samples.len() {
            // Normal case: interpolate between two samples
            samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac
        } else {
            // At last sample - no interpolation
            samples[src_idx]
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
    // Check file size before attempting to decode (same as decode())
    // Stereo decoding uses 2x memory, so this check is even more important
    let metadata = std::fs::metadata(path).map_err(|e| DjprepError::DecodeError {
        path: path.to_path_buf(),
        reason: format!("Failed to read file metadata: {}", e),
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
    if channels == 0 {
        return Vec::new();
    }
    if channels <= 2 {
        return samples.to_vec();
    }

    // Simple downmix: average left channels (0, 2, 4...) and right channels (1, 3, 5...)
    let num_frames = samples.len() / channels;
    let mut stereo = Vec::with_capacity(num_frames * 2);

    for frame in samples.chunks(channels) {
        // Use safe accessors to avoid out-of-bounds on incomplete final frames
        // For 5.1 surround: FL, FR, FC, LFE, BL, BR
        // Simple approach: mix front channels
        let left = frame.first().copied().unwrap_or(0.0);
        let right = frame.get(1).copied().unwrap_or(left);

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

    #[test]
    fn test_resample_upsample() {
        // Test upsampling from 22050 to 44100
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let result = resample(&samples, 22050, 44100);
        // Should be approximately double the length
        assert!((result.len() as f64 - 2000.0).abs() < 10.0);
    }

    #[test]
    fn test_resample_sine_wave_integrity() {
        // Generate a 440Hz sine wave at 44100Hz (100 samples = ~2.27 cycles)
        use std::f32::consts::PI;
        let sample_rate = 44100.0;
        let freq = 440.0;
        let num_samples = 2000;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        // Downsample to 22050Hz
        let result = resample(&samples, 44100, 22050);

        // The resampled signal should still oscillate between -1 and 1
        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);

        // High-quality resampler should preserve amplitude reasonably well
        assert!(max_val > 0.9, "Max value {} should be > 0.9", max_val);
        assert!(min_val < -0.9, "Min value {} should be < -0.9", min_val);
    }

    #[test]
    fn test_resample_fallback_works() {
        // Test the linear fallback directly
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let result = resample_linear_fallback(&samples, 44100, 22050);
        assert!((result.len() as f64 - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_to_mono_zero_channels() {
        // Zero channels should return empty vec (not panic)
        let samples = vec![0.5, 0.8, 1.0];
        let result = to_mono(&samples, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_to_mono_incomplete_frame() {
        // Incomplete frame at end should be handled gracefully
        let samples = vec![0.5, 0.3, 0.8, 0.2, 0.7]; // 5 samples, 2 channels = incomplete last frame
        let result = to_mono(&samples, 2);
        assert_eq!(result.len(), 3); // 2 complete frames + 1 incomplete
        assert!((result[0] - 0.4).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 0.7).abs() < 0.001); // Single sample in last frame
    }

    #[test]
    fn test_to_mono_empty_input() {
        let result = to_mono(&[], 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downmix_to_stereo_zero_channels() {
        // Zero channels should return empty vec (not panic)
        let samples = vec![0.5, 0.8, 1.0];
        let result = downmix_to_stereo(&samples, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downmix_to_stereo_incomplete_frame() {
        // 6-channel audio with incomplete last frame
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 8 samples, 6 channels
        let result = downmix_to_stereo(&samples, 6);
        // First complete frame: left=0.1, right=0.2
        assert_eq!(result.len(), 4); // 1 complete frame + 1 incomplete
        assert!((result[0] - 0.1).abs() < 0.001);
        assert!((result[1] - 0.2).abs() < 0.001);
        // Incomplete frame: only 2 samples [0.7, 0.8]
        assert!((result[2] - 0.7).abs() < 0.001);
        assert!((result[3] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_downmix_to_stereo_surround() {
        // 6-channel (5.1) surround: FL, FR, FC, LFE, BL, BR
        let frame = vec![0.5, 0.3, 0.1, 0.0, 0.2, 0.4];
        let result = downmix_to_stereo(&frame, 6);
        assert_eq!(result.len(), 2);
        // Should extract FL and FR
        assert!((result[0] - 0.5).abs() < 0.001);
        assert!((result[1] - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_downmix_to_stereo_passthrough() {
        // 2 channels should pass through unchanged
        let samples = vec![0.5, 0.3, 0.8, 0.2];
        let result = downmix_to_stereo(&samples, 2);
        assert_eq!(result, samples);
    }
}
