//! BPM/key analysis.

use crate::camelot;
use crate::error::{CoreError, Result};
use crate::types::{AudioBuffer, BpmResult, KeyResult, Mode, PitchClass};
use serde::{Deserialize, Serialize};

#[cfg(feature = "analysis-stratum")]
use stratum_dsp::{analyze_audio, AnalysisConfig, Key};

/// Combined analysis output for convenience APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub bpm: BpmResult,
    pub key: KeyResult,
    pub duration_seconds: f64,
    pub sample_rate: u32,
}

/// Detect BPM from decoded audio.
pub fn detect_bpm(buffer: &AudioBuffer) -> Result<BpmResult> {
    #[cfg(feature = "analysis-stratum")]
    {
        let config = AnalysisConfig::default();

        let result = analyze_audio(&buffer.samples, buffer.sample_rate, config)
            .map_err(|e| CoreError::AnalysisError(format!("BPM analysis failed: {e}")))?;

        let bpm = result.bpm as f64;
        let confidence = result.bpm_confidence as f64;

        let candidates = vec![(bpm / 2.0, confidence * 0.5), (bpm * 2.0, confidence * 0.5)];

        return Ok(BpmResult {
            value: bpm,
            confidence,
            candidates,
        });
    }

    #[cfg(all(not(feature = "analysis-stratum"), feature = "analysis-lite"))]
    {
        return detect_bpm_lite(buffer);
    }

    #[cfg(not(any(feature = "analysis-stratum", feature = "analysis-lite")))]
    {
        let _ = buffer;
        Err(CoreError::AnalysisError(
            "No analysis backend enabled (enable `analysis-stratum` or `analysis-lite`)"
                .to_string(),
        ))
    }
}

/// Detect key from decoded audio.
pub fn detect_key(buffer: &AudioBuffer) -> Result<KeyResult> {
    #[cfg(feature = "analysis-stratum")]
    {
        let config = AnalysisConfig::default();

        let result = analyze_audio(&buffer.samples, buffer.sample_rate, config)
            .map_err(|e| CoreError::AnalysisError(format!("Key analysis failed: {e}")))?;

        let (pitch_class, mode) = match result.key {
            Key::Major(pitch_idx) => {
                let pitch = PitchClass::from_index(pitch_idx as u8).unwrap_or(PitchClass::C);
                (pitch, Mode::Major)
            }
            Key::Minor(pitch_idx) => {
                let pitch = PitchClass::from_index(pitch_idx as u8).unwrap_or(PitchClass::C);
                (pitch, Mode::Minor)
            }
        };

        let camelot_str = camelot::to_camelot(pitch_class, mode).to_string();
        let open_key_str = camelot::to_open_key(pitch_class, mode).to_string();
        let confidence = result.key_confidence as f64;

        return Ok(KeyResult {
            pitch_class,
            mode,
            camelot: camelot_str,
            open_key: open_key_str,
            confidence,
        });
    }

    #[cfg(all(not(feature = "analysis-stratum"), feature = "analysis-lite"))]
    {
        return detect_key_lite(buffer);
    }

    #[cfg(not(any(feature = "analysis-stratum", feature = "analysis-lite")))]
    {
        let _ = buffer;
        Err(CoreError::AnalysisError(
            "No analysis backend enabled (enable `analysis-stratum` or `analysis-lite`)"
                .to_string(),
        ))
    }
}

/// Analyze pre-decoded audio and return combined results.
pub fn analyze_audio_buffer(buffer: &AudioBuffer) -> Result<AnalysisSummary> {
    Ok(AnalysisSummary {
        bpm: detect_bpm(buffer)?,
        key: detect_key(buffer)?,
        duration_seconds: buffer.duration,
        sample_rate: buffer.sample_rate,
    })
}

/// Decode and analyze borrowed bytes.
pub fn analyze_audio_bytes(
    audio_bytes: &[u8],
    hint_extension: Option<&str>,
) -> Result<AnalysisSummary> {
    #[cfg(feature = "decode")]
    {
        let buffer = crate::audio::decode_bytes(audio_bytes, hint_extension)?;
        return analyze_audio_buffer(&buffer);
    }

    #[cfg(not(feature = "decode"))]
    {
        let _ = (audio_bytes, hint_extension);
        Err(CoreError::UnsupportedFormat(
            "Byte decoding is disabled in this build. Enable `decode-*` feature or call analyze_audio_buffer().".to_string(),
        ))
    }
}

/// Decode and analyze owned bytes.
pub fn analyze_audio_owned(
    audio_bytes: Vec<u8>,
    hint_extension: Option<&str>,
) -> Result<AnalysisSummary> {
    #[cfg(feature = "decode")]
    {
        let buffer = crate::audio::decode_owned(audio_bytes, hint_extension)?;
        return analyze_audio_buffer(&buffer);
    }

    #[cfg(not(feature = "decode"))]
    {
        let _ = (audio_bytes, hint_extension);
        Err(CoreError::UnsupportedFormat(
            "Byte decoding is disabled in this build. Enable `decode-*` feature or call analyze_audio_buffer().".to_string(),
        ))
    }
}

#[cfg(feature = "analysis-lite")]
fn detect_bpm_lite(buffer: &AudioBuffer) -> Result<BpmResult> {
    if buffer.samples.is_empty() || buffer.sample_rate == 0 {
        return Err(CoreError::InvalidInput(
            "Empty samples or invalid sample rate".to_string(),
        ));
    }

    // Lightweight tempo estimator: onset envelope + bounded autocorrelation.
    let hop = (buffer.sample_rate as usize / 200).max(1);
    let mut envelope = Vec::with_capacity(buffer.samples.len() / hop + 1);
    for chunk in buffer.samples.chunks(hop) {
        let energy = chunk.iter().map(|s| s.abs()).sum::<f32>() / chunk.len() as f32;
        envelope.push(energy);
    }

    if envelope.len() < 8 {
        return Ok(BpmResult::default());
    }

    let mut flux = Vec::with_capacity(envelope.len() - 1);
    for i in 1..envelope.len() {
        flux.push((envelope[i] - envelope[i - 1]).max(0.0));
    }

    let env_rate = buffer.sample_rate as f64 / hop as f64;
    let min_lag = (env_rate * 60.0 / 200.0).round().max(1.0) as usize;
    let max_lag = (env_rate * 60.0 / 70.0).round().max(min_lag as f64 + 1.0) as usize;

    let mut best_lag = min_lag;
    let mut best_score = 0.0_f64;

    for lag in min_lag..=max_lag.min(flux.len().saturating_sub(1)) {
        let mut score = 0.0_f64;
        for i in lag..flux.len() {
            score += (flux[i] as f64) * (flux[i - lag] as f64);
        }
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    let bpm = (60.0 * env_rate / best_lag as f64).clamp(60.0, 200.0);
    let norm = flux
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    let confidence = if norm > 0.0 {
        (best_score / (norm * norm + 1e-9)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(BpmResult {
        value: bpm,
        confidence,
        candidates: vec![(bpm / 2.0, confidence * 0.5), (bpm * 2.0, confidence * 0.5)],
    })
}

#[cfg(feature = "analysis-lite")]
fn detect_key_lite(buffer: &AudioBuffer) -> Result<KeyResult> {
    if buffer.samples.len() < 2 || buffer.sample_rate == 0 {
        return Err(CoreError::InvalidInput(
            "Insufficient samples or invalid sample rate".to_string(),
        ));
    }

    // Lightweight key proxy: dominant periodicity via zero-crossing estimate.
    let mut crossings = 0usize;
    for i in 1..buffer.samples.len() {
        let a = buffer.samples[i - 1];
        let b = buffer.samples[i];
        if (a <= 0.0 && b > 0.0) || (a >= 0.0 && b < 0.0) {
            crossings += 1;
        }
    }

    let duration = buffer.duration.max(1e-6);
    let freq = ((crossings as f64 / 2.0) / duration).clamp(20.0, 2000.0);
    let midi = (69.0 + 12.0 * (freq / 440.0).log2()).round() as i32;
    let pitch_idx = ((midi % 12) + 12) % 12;

    let pitch_class = PitchClass::from_index(pitch_idx as u8).unwrap_or(PitchClass::C);

    // Use a simple low/high energy heuristic for major/minor tendency.
    let mid = buffer.samples.len() / 2;
    let low = buffer.samples[..mid]
        .iter()
        .map(|s| s.abs() as f64)
        .sum::<f64>();
    let high = buffer.samples[mid..]
        .iter()
        .map(|s| s.abs() as f64)
        .sum::<f64>();
    let mode = if high > low { Mode::Major } else { Mode::Minor };

    let camelot_str = camelot::to_camelot(pitch_class, mode).to_string();
    let open_key_str = camelot::to_open_key(pitch_class, mode).to_string();

    let confidence = (0.2 + (crossings as f64 / buffer.samples.len() as f64)).clamp(0.0, 0.6);

    Ok(KeyResult {
        pitch_class,
        mode,
        camelot: camelot_str,
        open_key: open_key_str,
        confidence,
    })
}
