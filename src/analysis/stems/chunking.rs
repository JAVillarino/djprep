//! Audio chunking with overlap-add for stem separation
//!
//! HTDemucs v4 has a maximum segment length of ~7.8 seconds due to memory constraints
//! and the model's receptive field. This module handles splitting audio into chunks
//! and reassembling stems using overlap-add with linear crossfade.
//!
//! # Chunking Strategy
//!
//! ```text
//! Input audio:  [===============================================]
//! Chunk 1:      [=======]
//! Chunk 2:          [=======]     (overlaps with chunk 1)
//! Chunk 3:              [=======] (overlaps with chunk 2)
//!                   ^^^
//!                   1-second overlap zone with linear crossfade
//! ```
//!
//! # Why 7.8 seconds?
//!
//! - HTDemucs v4 was trained on 7.8-second segments (343,980 samples at 44.1kHz)
//! - Longer segments would exceed GPU memory on typical hardware
//! - Shorter segments lose musical context, hurting separation quality
//!
//! # Why 1-second overlap?
//!
//! - Provides enough audio for smooth crossfade transitions
//! - Minimizes audible artifacts at chunk boundaries
//! - Balances quality vs computational overhead (more overlap = more processing)
//!
//! # Crossfade Formula
//!
//! In the overlap region, samples are blended using linear interpolation:
//! ```text
//! output[i] = chunk_a[i] * (1 - t) + chunk_b[i] * t
//! ```
//! where `t` ranges from 0 to 1 across the overlap zone.

use crate::error::{DjprepError, Result};
use crate::types::StereoBuffer;

/// HTDemucs v4 maximum segment length in seconds
/// This matches the model's training configuration (343,980 samples at 44.1kHz)
pub const MAX_SEGMENT_SECONDS: f32 = 7.8;

/// Overlap between segments in seconds for smooth crossfade reconstruction
/// 1 second provides ~44,100 samples for blending at 44.1kHz
pub const OVERLAP_SECONDS: f32 = 1.0;

/// Configuration for audio chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum samples per chunk
    pub chunk_samples: usize,
    /// Overlap samples between chunks
    pub overlap_samples: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl ChunkConfig {
    /// Create config for HTDemucs at 44.1kHz
    pub fn htdemucs() -> Self {
        let sample_rate = 44100;
        Self {
            chunk_samples: (MAX_SEGMENT_SECONDS * sample_rate as f32) as usize, // ~344,190
            overlap_samples: (OVERLAP_SECONDS * sample_rate as f32) as usize,   // ~44,100
            sample_rate,
        }
    }

    /// Create config with custom parameters
    pub fn new(max_seconds: f32, overlap_seconds: f32, sample_rate: u32) -> Self {
        Self {
            chunk_samples: (max_seconds * sample_rate as f32) as usize,
            overlap_samples: (overlap_seconds * sample_rate as f32) as usize,
            sample_rate,
        }
    }

    /// Calculate stride (hop) between chunk starts
    pub fn stride(&self) -> usize {
        self.chunk_samples.saturating_sub(self.overlap_samples)
    }
}

/// A single audio chunk ready for processing
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Chunk index (0-based)
    pub index: usize,
    /// Start sample in original audio
    pub start_sample: usize,
    /// End sample in original audio
    pub end_sample: usize,
    /// Audio data for this chunk
    pub audio: StereoBuffer,
}

/// Separated stems for a single chunk
#[derive(Debug, Clone)]
pub struct StemChunk {
    /// Chunk index (0-based)
    pub index: usize,
    /// Start sample in original audio
    pub start_sample: usize,
    /// End sample in original audio
    pub end_sample: usize,
    /// Vocals stem
    pub vocals: StereoBuffer,
    /// Drums stem
    pub drums: StereoBuffer,
    /// Bass stem
    pub bass: StereoBuffer,
    /// Other/melody stem
    pub other: StereoBuffer,
}

/// Four stems for the full track
#[derive(Debug, Clone)]
pub struct FourStems {
    pub vocals: StereoBuffer,
    pub drums: StereoBuffer,
    pub bass: StereoBuffer,
    pub other: StereoBuffer,
}

/// Split audio into overlapping chunks
pub fn chunk_audio(audio: &StereoBuffer, config: &ChunkConfig) -> Vec<AudioChunk> {
    let total_samples = audio.len();
    let stride = config.stride();

    if total_samples <= config.chunk_samples {
        // Short audio: single chunk
        return vec![AudioChunk {
            index: 0,
            start_sample: 0,
            end_sample: total_samples,
            audio: audio.clone(),
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut index = 0;

    while start < total_samples {
        let end = (start + config.chunk_samples).min(total_samples);

        // Extract chunk samples
        let left: Vec<f32> = audio.left[start..end].to_vec();
        let right: Vec<f32> = audio.right[start..end].to_vec();

        chunks.push(AudioChunk {
            index,
            start_sample: start,
            end_sample: end,
            audio: StereoBuffer::new(left, right, config.sample_rate),
        });

        // Move to next chunk
        start += stride;
        index += 1;

        // Avoid tiny final chunk
        if total_samples - start < config.overlap_samples {
            break;
        }
    }

    chunks
}

/// Interleaved stereo sample for a single time point across all 4 stems
/// Using struct-of-arrays would require 8 separate Vec accesses per sample.
/// This array-of-structs layout keeps all data for one sample in a single cache line.
#[derive(Clone, Copy, Default)]
struct StemSample {
    vocals_l: f32,
    vocals_r: f32,
    drums_l: f32,
    drums_r: f32,
    bass_l: f32,
    bass_r: f32,
    other_l: f32,
    other_r: f32,
}

/// Reassemble separated stem chunks using overlap-add with linear crossfade
///
/// Uses array-of-structs layout for better cache locality: all stem data for a single
/// sample position is stored contiguously, reducing cache misses during the inner loop.
///
/// # Errors
///
/// Returns an error if `chunks` is empty (no stems to reassemble).
pub fn overlap_add(chunks: &[StemChunk], config: &ChunkConfig, total_samples: usize) -> Result<FourStems> {
    // Require at least one chunk to determine sample rate
    let sample_rate = chunks
        .first()
        .map(|c| c.vocals.sample_rate)
        .ok_or_else(|| DjprepError::StemUnavailable {
            reason: "No stem chunks to reassemble (overlap_add requires at least one chunk)".to_string(),
        })?;

    // Use array-of-structs for cache locality: all stems for sample i are adjacent
    let mut output = vec![StemSample::default(); total_samples];
    let mut weight_sum = vec![0.0f32; total_samples];

    let num_chunks = chunks.len();
    for chunk in chunks {
        let chunk_len = chunk.vocals.len();
        let is_first = chunk.index == 0;
        let is_last = chunk.index == num_chunks - 1;

        // Generate crossfade weights for this chunk
        let weights = generate_crossfade_weights(chunk_len, config.overlap_samples, is_first, is_last);

        // Process samples with good cache locality
        // The cache benefit comes from StemSample struct keeping all stem data for
        // one position adjacent in memory. Using indexed loop is clearer than deeply
        // nested zip iterators and equally efficient with bounds check elision.
        // We deliberately use indexed loop here rather than iterator-based approach
        // because we index into 10 different arrays (weights, 8 stem channels, output).
        #[allow(clippy::needless_range_loop)]
        for i in 0..chunk_len {
            let out_idx = chunk.start_sample + i;
            if out_idx < total_samples {
                let w = weights[i];
                let s = &mut output[out_idx];

                // All stem reads for position i are close in the source data,
                // and all writes go to the same StemSample struct (one cache line)
                s.vocals_l += chunk.vocals.left[i] * w;
                s.vocals_r += chunk.vocals.right[i] * w;
                s.drums_l += chunk.drums.left[i] * w;
                s.drums_r += chunk.drums.right[i] * w;
                s.bass_l += chunk.bass.left[i] * w;
                s.bass_r += chunk.bass.right[i] * w;
                s.other_l += chunk.other.left[i] * w;
                s.other_r += chunk.other.right[i] * w;
                weight_sum[out_idx] += w;
            }
        }
    }

    // Normalize by weight sum and extract to separate buffers
    let mut vocals_left = Vec::with_capacity(total_samples);
    let mut vocals_right = Vec::with_capacity(total_samples);
    let mut drums_left = Vec::with_capacity(total_samples);
    let mut drums_right = Vec::with_capacity(total_samples);
    let mut bass_left = Vec::with_capacity(total_samples);
    let mut bass_right = Vec::with_capacity(total_samples);
    let mut other_left = Vec::with_capacity(total_samples);
    let mut other_right = Vec::with_capacity(total_samples);

    for (s, &ws) in output.iter().zip(&weight_sum) {
        let inv_w = if ws > 1e-8 { 1.0 / ws } else { 1.0 };
        vocals_left.push(s.vocals_l * inv_w);
        vocals_right.push(s.vocals_r * inv_w);
        drums_left.push(s.drums_l * inv_w);
        drums_right.push(s.drums_r * inv_w);
        bass_left.push(s.bass_l * inv_w);
        bass_right.push(s.bass_r * inv_w);
        other_left.push(s.other_l * inv_w);
        other_right.push(s.other_r * inv_w);
    }

    Ok(FourStems {
        vocals: StereoBuffer::new(vocals_left, vocals_right, sample_rate),
        drums: StereoBuffer::new(drums_left, drums_right, sample_rate),
        bass: StereoBuffer::new(bass_left, bass_right, sample_rate),
        other: StereoBuffer::new(other_left, other_right, sample_rate),
    })
}

/// Generate crossfade weights for a chunk
///
/// Uses linear ramps at the start and end of each chunk for smooth blending
fn generate_crossfade_weights(
    chunk_len: usize,
    overlap: usize,
    is_first: bool,
    is_last: bool,
) -> Vec<f32> {
    let mut weights = vec![1.0f32; chunk_len];

    // Fade in at start (unless first chunk)
    if !is_first {
        let fade_len = overlap.min(chunk_len);
        for (i, weight) in weights.iter_mut().take(fade_len).enumerate() {
            *weight = i as f32 / fade_len as f32;
        }
    }

    // Fade out at end (unless last chunk)
    if !is_last {
        let fade_len = overlap.min(chunk_len);
        let start = chunk_len.saturating_sub(fade_len);
        for (i, weight) in weights[start..].iter_mut().enumerate() {
            *weight *= (fade_len - i) as f32 / fade_len as f32;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_htdemucs() {
        let config = ChunkConfig::htdemucs();
        assert_eq!(config.sample_rate, 44100);
        // ~7.8 seconds
        assert!(config.chunk_samples > 340000 && config.chunk_samples < 350000);
        // ~1 second overlap
        assert!(config.overlap_samples > 43000 && config.overlap_samples < 45000);
    }

    #[test]
    fn test_chunk_config_stride() {
        let config = ChunkConfig::htdemucs();
        let stride = config.stride();
        // Stride should be chunk_samples - overlap_samples
        assert_eq!(stride, config.chunk_samples - config.overlap_samples);
    }

    #[test]
    fn test_chunk_short_audio() {
        let config = ChunkConfig::htdemucs();
        let short_audio = StereoBuffer::new(
            vec![0.0; 1000],
            vec![0.0; 1000],
            44100,
        );

        let chunks = chunk_audio(&short_audio, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].audio.len(), 1000);
    }

    #[test]
    fn test_crossfade_weights() {
        let weights = generate_crossfade_weights(100, 20, false, false);
        assert_eq!(weights.len(), 100);
        // Should fade in at start
        assert!(weights[0] < 0.1);
        assert!(weights[10] > 0.4 && weights[10] < 0.6);
        // Should be ~1 in middle
        assert!(weights[50] > 0.9);
        // Should fade out at end
        assert!(weights[99] < 0.1);
    }

    #[test]
    fn test_crossfade_first_chunk() {
        let weights = generate_crossfade_weights(100, 20, true, false);
        // First chunk: no fade in, only fade out
        assert!(weights[0] > 0.99);
        assert!(weights[99] < 0.1);
    }

    #[test]
    fn test_crossfade_last_chunk() {
        let weights = generate_crossfade_weights(100, 20, false, true);
        // Last chunk: fade in, no fade out
        assert!(weights[0] < 0.1);
        assert!(weights[99] > 0.99);
    }

    #[test]
    fn test_overlap_add_single_chunk() {
        // Test overlap_add with a single chunk (no overlap needed)
        let config = ChunkConfig::new(1.0, 0.1, 44100);
        let chunk_len = 1000;

        let chunk = StemChunk {
            index: 0,
            start_sample: 0,
            end_sample: chunk_len,
            vocals: StereoBuffer::new(vec![1.0; chunk_len], vec![0.5; chunk_len], 44100),
            drums: StereoBuffer::new(vec![0.8; chunk_len], vec![0.4; chunk_len], 44100),
            bass: StereoBuffer::new(vec![0.6; chunk_len], vec![0.3; chunk_len], 44100),
            other: StereoBuffer::new(vec![0.2; chunk_len], vec![0.1; chunk_len], 44100),
        };

        let result = overlap_add(&[chunk], &config, chunk_len).unwrap();

        // With single chunk, values should be unchanged
        assert_eq!(result.vocals.left.len(), chunk_len);
        assert!((result.vocals.left[500] - 1.0).abs() < 0.01);
        assert!((result.vocals.right[500] - 0.5).abs() < 0.01);
        assert!((result.drums.left[500] - 0.8).abs() < 0.01);
        assert!((result.bass.left[500] - 0.6).abs() < 0.01);
        assert!((result.other.left[500] - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_overlap_add_preserves_sample_rate() {
        let config = ChunkConfig::new(1.0, 0.1, 48000);
        let chunk_len = 100;

        let chunk = StemChunk {
            index: 0,
            start_sample: 0,
            end_sample: chunk_len,
            vocals: StereoBuffer::new(vec![0.0; chunk_len], vec![0.0; chunk_len], 48000),
            drums: StereoBuffer::new(vec![0.0; chunk_len], vec![0.0; chunk_len], 48000),
            bass: StereoBuffer::new(vec![0.0; chunk_len], vec![0.0; chunk_len], 48000),
            other: StereoBuffer::new(vec![0.0; chunk_len], vec![0.0; chunk_len], 48000),
        };

        let result = overlap_add(&[chunk], &config, chunk_len).unwrap();

        // Sample rate should be preserved from chunks
        assert_eq!(result.vocals.sample_rate, 48000);
        assert_eq!(result.drums.sample_rate, 48000);
    }

    #[test]
    fn test_overlap_add_empty_chunks_returns_error() {
        let config = ChunkConfig::htdemucs();
        let result = overlap_add(&[], &config, 1000);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("No stem chunks"));
    }
}
