//! wasm-bindgen bindings.

use wasm_bindgen::prelude::*;

/// Analyze already-decoded mono PCM samples.
#[wasm_bindgen(js_name = analyzePcm)]
pub fn analyze_pcm(samples: Vec<f32>, sample_rate: u32) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    let buffer = crate::types::AudioBuffer::new(samples, sample_rate);
    let result = crate::analysis::analyze_audio_buffer(&buffer)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Analyze encoded audio bytes (full build with decode features enabled).
#[cfg(feature = "decode")]
#[wasm_bindgen]
pub fn analyze_audio(
    audio_bytes: &[u8],
    hint_extension: Option<String>,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    let result = crate::analysis::analyze_audio_bytes(audio_bytes, hint_extension.as_deref())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}
