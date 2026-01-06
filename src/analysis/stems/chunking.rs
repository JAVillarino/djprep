//! Audio chunking with overlap-add for stem separation
//!
//! HTDemucs v4 has a maximum segment length of ~7.8 seconds.
//! This module handles splitting audio into chunks and reassembling stems.

use crate::types::StereoBuffer;

/// HTDemucs v4 maximum segment length in seconds
pub const MAX_SEGMENT_SECONDS: f32 = 7.8;

/// Overlap between segments in seconds (for smooth crossfade)
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

/// Reassemble separated stem chunks using overlap-add with linear crossfade
pub fn overlap_add(chunks: &[StemChunk], config: &ChunkConfig, total_samples: usize) -> FourStems {
    // Initialize output buffers
    let mut vocals_left = vec![0.0f32; total_samples];
    let mut vocals_right = vec![0.0f32; total_samples];
    let mut drums_left = vec![0.0f32; total_samples];
    let mut drums_right = vec![0.0f32; total_samples];
    let mut bass_left = vec![0.0f32; total_samples];
    let mut bass_right = vec![0.0f32; total_samples];
    let mut other_left = vec![0.0f32; total_samples];
    let mut other_right = vec![0.0f32; total_samples];

    // Weight accumulator for normalization
    let mut weight_sum = vec![0.0f32; total_samples];

    for chunk in chunks {
        let chunk_len = chunk.vocals.len();

        // Generate crossfade weights for this chunk
        let weights = generate_crossfade_weights(chunk_len, config.overlap_samples, chunk.index == 0, chunk.index == chunks.len() - 1);

        // Add weighted samples to output
        // Note: i is used to index multiple arrays (weights, vocals, drums, etc.)
        #[allow(clippy::needless_range_loop)]
        for i in 0..chunk_len {
            let out_idx = chunk.start_sample + i;
            if out_idx < total_samples {
                let w = weights[i];

                vocals_left[out_idx] += chunk.vocals.left[i] * w;
                vocals_right[out_idx] += chunk.vocals.right[i] * w;
                drums_left[out_idx] += chunk.drums.left[i] * w;
                drums_right[out_idx] += chunk.drums.right[i] * w;
                bass_left[out_idx] += chunk.bass.left[i] * w;
                bass_right[out_idx] += chunk.bass.right[i] * w;
                other_left[out_idx] += chunk.other.left[i] * w;
                other_right[out_idx] += chunk.other.right[i] * w;

                weight_sum[out_idx] += w;
            }
        }
    }

    // Normalize by weight sum
    for i in 0..total_samples {
        if weight_sum[i] > 1e-8 {
            let inv_w = 1.0 / weight_sum[i];
            vocals_left[i] *= inv_w;
            vocals_right[i] *= inv_w;
            drums_left[i] *= inv_w;
            drums_right[i] *= inv_w;
            bass_left[i] *= inv_w;
            bass_right[i] *= inv_w;
            other_left[i] *= inv_w;
            other_right[i] *= inv_w;
        }
    }

    let sample_rate = chunks.first().map(|c| c.vocals.sample_rate).unwrap_or(44100);

    FourStems {
        vocals: StereoBuffer::new(vocals_left, vocals_right, sample_rate),
        drums: StereoBuffer::new(drums_left, drums_right, sample_rate),
        bass: StereoBuffer::new(bass_left, bass_right, sample_rate),
        other: StereoBuffer::new(other_left, other_right, sample_rate),
    }
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
}
