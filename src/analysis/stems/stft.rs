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

#[cfg(feature = "stems")]
use rustfft::{num_complex::Complex, FftPlanner};

use crate::types::StereoBuffer;

/// FFT window size - matches HTDemucs training (4096 samples = ~93ms at 44.1kHz)
pub const NFFT: usize = 4096;

/// Hop length between frames - 75% overlap for COLA compliance
pub const HOP_LENGTH: usize = 1024;

/// Number of frequency bins in positive-frequency half of spectrum
pub const NUM_FREQ_BINS: usize = NFFT / 2 + 1; // 2049

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
#[cfg(feature = "stems")]
pub fn compute_stft(audio: &StereoBuffer) -> StereoSpectrogram {
    let left_spec = stft_channel(&audio.left);
    let right_spec = stft_channel(&audio.right);

    let num_frames = left_spec.len();

    StereoSpectrogram {
        left: left_spec,
        right: right_spec,
        num_frames,
        sample_rate: audio.sample_rate,
    }
}

/// Compute STFT for a single channel
#[cfg(feature = "stems")]
fn stft_channel(samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(NFFT);

    // Create Hann window
    let window = hann_window(NFFT);

    // Calculate number of frames
    let num_frames = (samples.len().saturating_sub(NFFT)) / HOP_LENGTH + 1;
    let mut spectrogram = Vec::with_capacity(num_frames);

    // Process each frame
    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        let end = (start + NFFT).min(samples.len());

        // Prepare input buffer with windowing
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); NFFT];
        for (i, &w) in window.iter().enumerate() {
            if start + i < end {
                buffer[i] = Complex::new(samples[start + i] * w, 0.0);
            }
        }

        // Compute FFT
        fft.process(&mut buffer);

        // Keep only positive frequencies (first half + DC + Nyquist)
        let frame: Vec<Complex<f32>> = buffer[..NUM_FREQ_BINS].to_vec();
        spectrogram.push(frame);
    }

    spectrogram
}

/// Compute inverse STFT to reconstruct audio
#[cfg(feature = "stems")]
pub fn compute_istft(spectrogram: &StereoSpectrogram, output_length: usize) -> StereoBuffer {
    let left = istft_channel(&spectrogram.left, output_length);
    let right = istft_channel(&spectrogram.right, output_length);

    StereoBuffer::new(left, right, spectrogram.sample_rate)
}

/// Compute inverse STFT for a single channel
#[cfg(feature = "stems")]
fn istft_channel(spectrogram: &[Vec<Complex<f32>>], output_length: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(NFFT);

    // Create Hann window for synthesis
    let window = hann_window(NFFT);

    // Output buffer with overlap-add
    let mut output = vec![0.0f32; output_length];
    let mut window_sum = vec![0.0f32; output_length];

    for (frame_idx, frame) in spectrogram.iter().enumerate() {
        let start = frame_idx * HOP_LENGTH;

        // Reconstruct full spectrum (add conjugate symmetric part)
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); NFFT];

        // Copy positive frequencies
        for (i, &c) in frame.iter().enumerate() {
            buffer[i] = c;
        }

        // Mirror negative frequencies (conjugate symmetric)
        for i in 1..NUM_FREQ_BINS - 1 {
            buffer[NFFT - i] = frame[i].conj();
        }

        // Compute inverse FFT
        ifft.process(&mut buffer);

        // Normalize and apply window, add to output
        let scale = 1.0 / NFFT as f32;
        for (i, &w) in window.iter().enumerate() {
            if start + i < output_length {
                output[start + i] += buffer[i].re * scale * w;
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

/// Generate Hann window of given size
#[cfg(feature = "stems")]
fn hann_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
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
