//! Audio decoding using symphonia.

use crate::error::{CoreError, Result};
use crate::types::AudioBuffer;
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Target sample rate for analysis (22050 Hz).
pub const TARGET_SAMPLE_RATE: u32 = 22050;

/// Decode borrowed bytes to a mono analysis buffer.
///
/// This API copies into an owned `Vec<u8>` internally for compatibility with
/// the symphonia cursor-based media source.
pub fn decode_bytes(bytes: &[u8], hint_extension: Option<&str>) -> Result<AudioBuffer> {
    decode_owned(bytes.to_vec(), hint_extension)
}

/// Decode owned bytes to a mono analysis buffer without an additional copy.
pub fn decode_owned(bytes: Vec<u8>, hint_extension: Option<&str>) -> Result<AudioBuffer> {
    decode_reader(Cursor::new(bytes), hint_extension)
}

/// Decode from an arbitrary media reader.
pub fn decode_reader<R>(reader: R, hint_extension: Option<&str>) -> Result<AudioBuffer>
where
    R: MediaSource + 'static,
{
    let mss = MediaSourceStream::new(Box::new(reader), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = hint_extension {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| CoreError::DecodeError(format!("Failed to probe format: {e}")))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| CoreError::DecodeError("No audio tracks found".to_string()))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_sample_rate = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| CoreError::DecodeError(format!("Failed to create decoder: {e}")))?;

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
                return Err(CoreError::DecodeError(format!(
                    "Failed to read packet: {e}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                continue;
            }
            Err(e) => {
                return Err(CoreError::DecodeError(format!("Decode error: {e}")));
            }
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        let mono_samples = to_mono(samples, channels);
        all_samples.extend(mono_samples);
    }

    if all_samples.is_empty() {
        return Err(CoreError::DecodeError(
            "Decoded audio contained no samples".to_string(),
        ));
    }

    let final_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        resample(&all_samples, source_sample_rate, TARGET_SAMPLE_RATE)
    } else {
        all_samples
    };

    Ok(AudioBuffer::new(final_samples, TARGET_SAMPLE_RATE))
}

fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

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
    fn test_to_mono_stereo() {
        let stereo = vec![0.5, 0.3, 0.8, 0.2, 1.0, 0.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.4).abs() < 0.001);
        assert!((mono[1] - 0.5).abs() < 0.001);
        assert!((mono[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_resample_identity() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = resample(&samples, 44100, 44100);
        assert_eq!(result, samples);
    }
}
