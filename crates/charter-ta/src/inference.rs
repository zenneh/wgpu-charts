//! ONNX model inference for ML predictions.
//!
//! This module provides inference capabilities for trained ML models.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

use crate::ml::{MlFeatures, MlPrediction};

/// Scaler parameters for feature normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalerParams {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

impl ScalerParams {
    /// Load scaler parameters from JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let params: ScalerParams = serde_json::from_str(&content)?;
        Ok(params)
    }

    /// Apply normalization to features.
    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let mean = self.mean.get(i).copied().unwrap_or(0.0);
                let scale = self.scale.get(i).copied().unwrap_or(1.0);
                if scale.abs() > f32::EPSILON {
                    (f - mean) / scale
                } else {
                    f - mean
                }
            })
            .collect()
    }
}

/// ML model inference engine.
pub struct MlInference {
    session: Mutex<Session>,
    scaler: Option<ScalerParams>,
}

impl MlInference {
    /// Load a model from an ONNX file.
    ///
    /// Optionally loads scaler parameters from a .scaler.json file with the same base name.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = model_path.as_ref();

        // Initialize ONNX Runtime
        ort::init()
            .with_name("charter-ml")
            .commit();

        // Load the model file
        let model_bytes = std::fs::read(model_path)?;

        // Load the session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(&model_bytes)?;

        // Try to load scaler
        let scaler_path = model_path.with_extension("scaler.json");
        let scaler = if scaler_path.exists() {
            match ScalerParams::load(&scaler_path) {
                Ok(s) => {
                    eprintln!("Loaded scaler with {} mean values, {} scale values", s.mean.len(), s.scale.len());
                    Some(s)
                }
                Err(e) => {
                    eprintln!("Failed to load scaler: {}", e);
                    None
                }
            }
        } else {
            eprintln!("No scaler file found at {:?}", scaler_path);
            None
        };

        Ok(Self {
            session: Mutex::new(session),
            scaler,
        })
    }

    /// Run inference on features.
    pub fn predict(&self, features: &MlFeatures) -> Result<MlPrediction, Box<dyn std::error::Error>> {
        let raw_feature_vec = features.to_vec();

        // Apply scaling if available
        let feature_vec = if let Some(ref scaler) = self.scaler {
            scaler.transform(&raw_feature_vec)
        } else {
            raw_feature_vec
        };

        let feature_dim = feature_vec.len();

        // Create input tensor
        let input_tensor = Tensor::from_array(([1usize, feature_dim], feature_vec))?;

        // Run inference
        let mut session = self.session.lock().map_err(|e| format!("Lock error: {}", e))?;
        let outputs = session.run(ort::inputs!["features" => input_tensor])?;

        // Try different output names (XGBoost uses "probabilities" or "output_probability")
        let direction_up_prob = if let Some(output) = outputs.get("probabilities") {
            // XGBoost ONNX format: probabilities tensor with shape [batch, 2]
            let (_, data_slice) = output.try_extract_tensor::<f32>()?;
            // Index 1 is probability of class 1 (up)
            data_slice.get(1).copied().unwrap_or(0.5)
        } else if let Some(output) = outputs.get("output_probability") {
            let (_, data_slice) = output.try_extract_tensor::<f32>()?;
            data_slice.get(1).copied().unwrap_or(0.5)
        } else if let Some(output) = outputs.get("predictions") {
            // Simple model format: single probability output
            let (_, data_slice) = output.try_extract_tensor::<f32>()?;
            data_slice.first().copied().unwrap_or(0.5)
        } else {
            // Fallback: try first output
            let (_, output) = outputs.iter().next().ok_or("No output tensors")?;
            let (_, data_slice) = output.try_extract_tensor::<f32>()?;
            // If 2 values, take second (prob up); if 1 value, use it directly
            if data_slice.len() >= 2 {
                data_slice.get(1).copied().unwrap_or(0.5)
            } else {
                data_slice.first().copied().unwrap_or(0.5)
            }
        };

        // Confidence = how far from 0.5 the prediction is, scaled to 0-1
        let confidence = (direction_up_prob - 0.5).abs() * 2.0;

        let prediction = MlPrediction {
            level_break_prob: 0.0, // Not predicted
            direction_up_prob,
            confidence,
        };

        Ok(prediction)
    }

    /// Batch inference for multiple feature sets.
    pub fn predict_batch(
        &self,
        features_batch: &[MlFeatures],
    ) -> Result<Vec<MlPrediction>, Box<dyn std::error::Error>> {
        if features_batch.is_empty() {
            return Ok(Vec::new());
        }

        let feature_dim = features_batch[0].feature_count();
        let batch_size = features_batch.len();

        // Flatten all features
        let mut flat_features: Vec<f32> = Vec::with_capacity(batch_size * feature_dim);
        for features in features_batch {
            let mut vec = features.to_vec();
            if let Some(ref scaler) = self.scaler {
                vec = scaler.transform(&vec);
            }
            flat_features.extend(vec);
        }

        // Create input tensor
        let input_tensor = Tensor::from_array(([batch_size, feature_dim], flat_features))?;

        // Run inference
        let mut session = self.session.lock().map_err(|e| format!("Lock error: {}", e))?;
        let outputs = session.run(ort::inputs!["features" => input_tensor])?;

        // Extract predictions
        let output = &outputs["predictions"];
        let (_, data_slice) = output.try_extract_tensor::<f32>()?;

        // Parse batch outputs
        let mut predictions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let offset = i * 3;
            predictions.push(MlPrediction {
                level_break_prob: data_slice.get(offset).copied().unwrap_or(0.5),
                direction_up_prob: data_slice.get(offset + 1).copied().unwrap_or(0.5),
                confidence: data_slice.get(offset + 2).copied().unwrap_or(0.5),
            });
        }

        Ok(predictions)
    }
}

/// Thread-safe wrapper for MlInference.
#[derive(Clone)]
pub struct MlInferenceHandle {
    inner: Arc<MlInference>,
}

impl MlInferenceHandle {
    /// Create a new handle from an inference engine.
    pub fn new(inference: MlInference) -> Self {
        Self {
            inner: Arc::new(inference),
        }
    }

    /// Load a model and create a handle.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let inference = MlInference::load(model_path)?;
        Ok(Self::new(inference))
    }

    /// Run inference.
    pub fn predict(&self, features: &MlFeatures) -> Result<MlPrediction, Box<dyn std::error::Error>> {
        self.inner.predict(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler_transform() {
        let scaler = ScalerParams {
            mean: vec![10.0, 20.0, 30.0],
            scale: vec![2.0, 4.0, 5.0],
        };

        let features = vec![12.0, 24.0, 35.0];
        let normalized = scaler.transform(&features);

        assert!((normalized[0] - 1.0).abs() < 0.001); // (12 - 10) / 2 = 1
        assert!((normalized[1] - 1.0).abs() < 0.001); // (24 - 20) / 4 = 1
        assert!((normalized[2] - 1.0).abs() < 0.001); // (35 - 30) / 5 = 1
    }
}
