//! STFT (Short-Time Fourier Transform) for HTDemucs preprocessing
//!
//! HTDemucs requires STFT-preprocessed audio as input.
//! Parameters match the model training: nfft=4096, hop_length=1024
//!
//! # STFT Parameter Choices
//!
//! - **NFFT = 4096**: Window size that balances frequency resolution vs time resolution.
//!   At 44.1kHz, this gives ~93ms windows with 2049 frequency bins (~10.7 Hz resolution).
//!   This matches the HTDemucs model training parameters from Facebook Research.
//!
//! - **HOP_LENGTH = 1024**: Step size between consecutive frames (75% overlap).
//!   Provides good temporal resolution while maintaining COLA (Constant Overlap-Add)
//!   property needed for perfect reconstruction in ISTFT.
//!
//! - **Hann window**: Smooth tapering reduces spectral leakage. Combined with 75%
//!   overlap, satisfies the COLA condition for artifact-free reconstruction.
//!
//! # Performance
//!
//! This implementation uses a cached `StftProcessor` to avoid per-frame allocations:
//! - FFT planner and plans are created once and reused
//! - Hann window is computed once and cached
//! - Work buffers are pre-allocated and reused across frames

#[cfg(feature = "stems")]
use rustfft::{num_complex::Complex, Fft, FftPlanner};
#[cfg(feature = "stems")]
use std::sync::Arc;

use crate::types::StereoBuffer;

/// FFT window size - matches HTDemucs training (4096 samples = ~93ms at 44.1kHz)
pub const NFFT: usize = 4096;

/// Hop length between frames - 75% overlap for COLA compliance
pub const HOP_LENGTH: usize = 1024;

/// Number of frequency bins in positive-frequency half of spectrum
pub const NUM_FREQ_BINS: usize = NFFT / 2 + 1; // 2049

/// Cached STFT processor to avoid repeated allocations
///
/// Reuses FFT plans, window, and work buffers across multiple STFT/ISTFT calls.
#[cfg(feature = "stems")]
pub struct StftProcessor {
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// Pre-allocated work buffer for FFT operations
    work_buffer: Vec<Complex<f32>>,
}

#[cfg(feature = "stems")]
impl StftProcessor {
    /// Create a new STFT processor with cached FFT plans
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(NFFT);
        let fft_inverse = planner.plan_fft_inverse(NFFT);
        let window = hann_window(NFFT);
        let work_buffer = vec![Complex::new(0.0, 0.0); NFFT];

        Self {
            fft_forward,
            fft_inverse,
            window,
            work_buffer,
        }
    }

    /// Compute STFT for stereo audio using cached resources
    pub fn compute_stft(&mut self, audio: &StereoBuffer) -> StereoSpectrogram {
        let left_spec = self.stft_channel(&audio.left);
        let right_spec = self.stft_channel(&audio.right);

        let num_frames = left_spec.len();

        StereoSpectrogram {
            left: left_spec,
            right: right_spec,
            num_frames,
            sample_rate: audio.sample_rate,
        }
    }

    /// Compute STFT for a single channel, reusing work buffer
    fn stft_channel(&mut self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let num_frames = (samples.len().saturating_sub(NFFT)) / HOP_LENGTH + 1;

        // Pre-allocate entire spectrogram at once to reduce allocations
        let mut spectrogram: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            let end = (start + NFFT).min(samples.len());

            // Reset work buffer and apply window
            for (i, buf) in self.work_buffer.iter_mut().enumerate() {
                if start + i < end {
                    *buf = Complex::new(samples[start + i] * self.window[i], 0.0);
                } else {
                    *buf = Complex::new(0.0, 0.0);
                }
            }

            // Compute FFT in-place
            self.fft_forward.process(&mut self.work_buffer);

            // Copy only positive frequencies to output
            spectrogram.push(self.work_buffer[..NUM_FREQ_BINS].to_vec());
        }

        spectrogram
    }

    /// Compute inverse STFT to reconstruct audio
    pub fn compute_istft(&mut self, spectrogram: &StereoSpectrogram, output_length: usize) -> StereoBuffer {
        let left = self.istft_channel(&spectrogram.left, output_length);
        let right = self.istft_channel(&spectrogram.right, output_length);

        StereoBuffer::new(left, right, spectrogram.sample_rate)
    }

    /// Compute inverse STFT for a single channel
    fn istft_channel(&mut self, spectrogram: &[Vec<Complex<f32>>], output_length: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; output_length];
        let mut window_sum = vec![0.0f32; output_length];

        for (frame_idx, frame) in spectrogram.iter().enumerate() {
            let start = frame_idx * HOP_LENGTH;

            // Reset work buffer
            for buf in self.work_buffer.iter_mut() {
                *buf = Complex::new(0.0, 0.0);
            }

            // Copy positive frequencies
            self.work_buffer[..frame.len()].copy_from_slice(frame);

            // Mirror negative frequencies (conjugate symmetric)
            for i in 1..NUM_FREQ_BINS - 1 {
                self.work_buffer[NFFT - i] = frame[i].conj();
            }

            // Compute inverse FFT in-place
            self.fft_inverse.process(&mut self.work_buffer);

            // Normalize and apply window, add to output
            let scale = 1.0 / NFFT as f32;
            for (i, &w) in self.window.iter().enumerate() {
                if start + i < output_length {
                    output[start + i] += self.work_buffer[i].re * scale * w;
                    window_sum[start + i] += w * w;
                }
            }
        }

        // Normalize by window sum (COLA normalization)
        for (i, &ws) in window_sum.iter().enumerate() {
            if ws > 1e-8 {
                output[i] /= ws;
            }
        }

        output
    }
}

#[cfg(feature = "stems")]
impl Default for StftProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Complex spectrogram for a stereo signal
#[cfg(feature = "stems")]
#[derive(Debug, Clone)]
pub struct StereoSpectrogram {
    /// Left channel spectrogram: [time_frames][freq_bins]
    pub left: Vec<Vec<Complex<f32>>>,
    /// Right channel spectrogram: [time_frames][freq_bins]
    pub right: Vec<Vec<Complex<f32>>>,
    /// Number of time frames
    pub num_frames: usize,
    /// Original sample rate
    pub sample_rate: u32,
}

/// Placeholder spectrogram when stems feature is disabled
#[cfg(not(feature = "stems"))]
#[derive(Debug, Clone)]
pub struct StereoSpectrogram {
    /// Number of time frames
    pub num_frames: usize,
    /// Original sample rate
    pub sample_rate: u32,
}

#[cfg(feature = "stems")]
impl StereoSpectrogram {
    /// Get the shape as (channels, freq_bins, time_frames)
    pub fn shape(&self) -> (usize, usize, usize) {
        (2, NUM_FREQ_BINS, self.num_frames)
    }
}

#[cfg(not(feature = "stems"))]
impl StereoSpectrogram {
    /// Get the shape as (channels, freq_bins, time_frames)
    pub fn shape(&self) -> (usize, usize, usize) {
        (2, NUM_FREQ_BINS, self.num_frames)
    }
}

/// Compute STFT for stereo audio
///
/// Returns complex spectrograms for both channels.
/// Note: For repeated use, prefer `StftProcessor` to avoid reallocating FFT plans.
#[cfg(feature = "stems")]
pub fn compute_stft(audio: &StereoBuffer) -> StereoSpectrogram {
    let mut processor = StftProcessor::new();
    processor.compute_stft(audio)
}

/// Compute inverse STFT to reconstruct audio
///
/// Note: For repeated use, prefer `StftProcessor` to avoid reallocating FFT plans.
#[cfg(feature = "stems")]
pub fn compute_istft(spectrogram: &StereoSpectrogram, output_length: usize) -> StereoBuffer {
    let mut processor = StftProcessor::new();
    processor.compute_istft(spectrogram, output_length)
}

/// Generate Hann window of given size
#[cfg(feature = "stems")]
fn hann_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    // Handle edge case: zero size would cause division by zero
    if size == 0 {
        return Vec::new();
    }
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Convert spectrogram to magnitude and phase
/// Returns (left_magnitude, left_phase, right_magnitude, right_phase)
#[cfg(feature = "stems")]
#[allow(clippy::type_complexity)]
pub fn to_magnitude_phase(
    spectrogram: &StereoSpectrogram,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut left_mag = Vec::with_capacity(spectrogram.num_frames);
    let mut left_phase = Vec::with_capacity(spectrogram.num_frames);
    let mut right_mag = Vec::with_capacity(spectrogram.num_frames);
    let mut right_phase = Vec::with_capacity(spectrogram.num_frames);

    for (l_frame, r_frame) in spectrogram.left.iter().zip(spectrogram.right.iter()) {
        let l_m: Vec<f32> = l_frame.iter().map(|c| c.norm()).collect();
        let l_p: Vec<f32> = l_frame.iter().map(|c| c.arg()).collect();
        let r_m: Vec<f32> = r_frame.iter().map(|c| c.norm()).collect();
        let r_p: Vec<f32> = r_frame.iter().map(|c| c.arg()).collect();

        left_mag.push(l_m);
        left_phase.push(l_p);
        right_mag.push(r_m);
        right_phase.push(r_p);
    }

    (left_mag, left_phase, right_mag, right_phase)
}

/// Reconstruct complex spectrogram from magnitude and phase
#[cfg(feature = "stems")]
pub fn from_magnitude_phase(
    left_mag: &[Vec<f32>],
    left_phase: &[Vec<f32>],
    right_mag: &[Vec<f32>],
    right_phase: &[Vec<f32>],
    sample_rate: u32,
) -> StereoSpectrogram {
    let num_frames = left_mag.len();
    let mut left = Vec::with_capacity(num_frames);
    let mut right = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let l_frame: Vec<Complex<f32>> = left_mag[i]
            .iter()
            .zip(left_phase[i].iter())
            .map(|(&m, &p)| Complex::from_polar(m, p))
            .collect();
        let r_frame: Vec<Complex<f32>> = right_mag[i]
            .iter()
            .zip(right_phase[i].iter())
            .map(|(&m, &p)| Complex::from_polar(m, p))
            .collect();

        left.push(l_frame);
        right.push(r_frame);
    }

    StereoSpectrogram {
        left,
        right,
        num_frames,
        sample_rate,
    }
}

#[cfg(not(feature = "stems"))]
pub fn compute_stft(_audio: &StereoBuffer) -> StereoSpectrogram {
    StereoSpectrogram {
        num_frames: 0,
        sample_rate: 44100,
    }
}

#[cfg(not(feature = "stems"))]
pub fn compute_istft(_spectrogram: &StereoSpectrogram, _output_length: usize) -> StereoBuffer {
    StereoBuffer::new(vec![], vec![], 44100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "stems")]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);
        // Hann window should be 0 at endpoints, max at center
        assert!(window[0] < 0.01);
        assert!(window[2] > 0.9);
    }

    #[test]
    fn test_stft_constants() {
        assert_eq!(NFFT, 4096);
        assert_eq!(HOP_LENGTH, 1024);
        assert_eq!(NUM_FREQ_BINS, 2049);
    }
}
