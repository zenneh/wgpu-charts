//! Machine Learning feature extraction for technical analysis.
//!
//! This module provides types and utilities for extracting ML features from TA data
//! and exporting training samples for model training.

use serde::{Deserialize, Serialize};

use crate::types::{CandleDirection, Level, LevelState, LevelType, Trend, TrendState};

/// Number of levels to extract per category (hold/greedy × bullish/bearish).
pub const LEVELS_PER_CATEGORY: usize = 3;

/// Number of timeframes in the system.
pub const NUM_TIMEFRAMES: usize = 12;

/// Normalized level features for ML.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LevelFeatures {
    /// Normalized price distance from current price (positive = above, negative = below).
    /// Value is (level_price - current_price) / current_price.
    pub price_distance: f32,
    /// Number of hits this level has received.
    pub hit_count: u8,
    /// Number of respected hits (wick only).
    pub respected_hits: u8,
    /// Whether level is active (1.0) or broken (0.0).
    pub is_active: f32,
    /// Age of level in candles (normalized by dividing by 1000).
    pub age_normalized: f32,
}

impl LevelFeatures {
    /// Extract features from a level.
    pub fn from_level(level: &Level, current_price: f32, current_candle_index: usize) -> Self {
        let price_distance = if current_price > 0.0 {
            (level.price - current_price) / current_price
        } else {
            0.0
        };

        let respected = level.hits.iter().filter(|h| h.respected).count() as u8;
        let age = current_candle_index.saturating_sub(level.created_at_index);

        Self {
            price_distance,
            hit_count: level.hits.len().min(255) as u8,
            respected_hits: respected,
            is_active: if level.state != LevelState::Broken { 1.0 } else { 0.0 },
            age_normalized: (age as f32) / 1000.0,
        }
    }

    /// Create empty/padding features.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Convert to feature vector.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.price_distance,
            self.hit_count as f32,
            self.respected_hits as f32,
            self.is_active,
            self.age_normalized,
        ]
    }
}

/// Normalized trend features for ML.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TrendFeatures {
    /// Slope of trendline (price change per candle, normalized).
    pub slope_normalized: f32,
    /// Current distance from trendline (positive = above, negative = below).
    pub distance_normalized: f32,
    /// Number of hits this trend has received.
    pub hit_count: u8,
    /// Whether trend is active (1.0) or broken (0.0).
    pub is_active: f32,
    /// Age of trend in candles (normalized).
    pub age_normalized: f32,
    /// Whether trend exists (1.0) or is padding (0.0).
    pub exists: f32,
}

impl TrendFeatures {
    /// Extract features from a trend.
    pub fn from_trend(trend: &Trend, current_price: f32, current_candle_index: usize) -> Self {
        let trend_price = trend.price_at(current_candle_index);

        // Calculate slope (price change per candle)
        let dx = trend.end.candle_index as f32 - trend.start.candle_index as f32;
        let slope = if dx.abs() > f32::EPSILON {
            (trend.end.price - trend.start.price) / dx
        } else {
            0.0
        };

        // Normalize slope by current price
        let slope_normalized = if current_price > 0.0 {
            slope / current_price
        } else {
            0.0
        };

        let distance_normalized = if current_price > 0.0 {
            (current_price - trend_price) / current_price
        } else {
            0.0
        };

        let age = current_candle_index.saturating_sub(trend.created_at_index);

        Self {
            slope_normalized,
            distance_normalized,
            hit_count: trend.hits.len().min(255) as u8,
            is_active: if trend.state != TrendState::Broken { 1.0 } else { 0.0 },
            age_normalized: (age as f32) / 1000.0,
            exists: 1.0,
        }
    }

    /// Create empty/padding features.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Convert to feature vector.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.slope_normalized,
            self.distance_normalized,
            self.hit_count as f32,
            self.is_active,
            self.age_normalized,
            self.exists,
        ]
    }
}

/// Features for a single timeframe.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeframeFeatures {
    /// Timeframe index (0-11).
    pub timeframe_index: usize,
    /// Top 3 bullish hold levels (support).
    pub bullish_hold_levels: [LevelFeatures; LEVELS_PER_CATEGORY],
    /// Top 3 bearish hold levels (resistance).
    pub bearish_hold_levels: [LevelFeatures; LEVELS_PER_CATEGORY],
    /// Top 3 bullish greedy levels.
    pub bullish_greedy_levels: [LevelFeatures; LEVELS_PER_CATEGORY],
    /// Top 3 bearish greedy levels.
    pub bearish_greedy_levels: [LevelFeatures; LEVELS_PER_CATEGORY],
    /// Latest active bullish trend.
    pub bullish_trend: TrendFeatures,
    /// Latest active bearish trend.
    pub bearish_trend: TrendFeatures,
    /// Number of active levels.
    pub active_level_count: u16,
    /// Number of active trends.
    pub active_trend_count: u16,
}

impl TimeframeFeatures {
    /// Extract features from levels and trends for a timeframe.
    pub fn extract(
        timeframe_index: usize,
        levels: &[Level],
        trends: &[Trend],
        current_price: f32,
        current_candle_index: usize,
    ) -> Self {
        let mut features = Self {
            timeframe_index,
            ..Default::default()
        };

        // Filter and sort levels by proximity to current price
        let mut bullish_hold: Vec<_> = levels
            .iter()
            .filter(|l| {
                l.direction == CandleDirection::Bullish
                    && l.level_type == LevelType::Hold
                    && l.state != LevelState::Broken
            })
            .collect();

        let mut bearish_hold: Vec<_> = levels
            .iter()
            .filter(|l| {
                l.direction == CandleDirection::Bearish
                    && l.level_type == LevelType::Hold
                    && l.state != LevelState::Broken
            })
            .collect();

        let mut bullish_greedy: Vec<_> = levels
            .iter()
            .filter(|l| {
                l.direction == CandleDirection::Bullish
                    && l.level_type == LevelType::GreedyHold
                    && l.state != LevelState::Broken
            })
            .collect();

        let mut bearish_greedy: Vec<_> = levels
            .iter()
            .filter(|l| {
                l.direction == CandleDirection::Bearish
                    && l.level_type == LevelType::GreedyHold
                    && l.state != LevelState::Broken
            })
            .collect();

        // Sort by proximity to current price
        let sort_by_proximity = |a: &&Level, b: &&Level| {
            let dist_a = (a.price - current_price).abs();
            let dist_b = (b.price - current_price).abs();
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        };

        bullish_hold.sort_by(sort_by_proximity);
        bearish_hold.sort_by(sort_by_proximity);
        bullish_greedy.sort_by(sort_by_proximity);
        bearish_greedy.sort_by(sort_by_proximity);

        // Extract top 3 for each category
        for (i, level) in bullish_hold.iter().take(LEVELS_PER_CATEGORY).enumerate() {
            features.bullish_hold_levels[i] =
                LevelFeatures::from_level(level, current_price, current_candle_index);
        }
        for (i, level) in bearish_hold.iter().take(LEVELS_PER_CATEGORY).enumerate() {
            features.bearish_hold_levels[i] =
                LevelFeatures::from_level(level, current_price, current_candle_index);
        }
        for (i, level) in bullish_greedy.iter().take(LEVELS_PER_CATEGORY).enumerate() {
            features.bullish_greedy_levels[i] =
                LevelFeatures::from_level(level, current_price, current_candle_index);
        }
        for (i, level) in bearish_greedy.iter().take(LEVELS_PER_CATEGORY).enumerate() {
            features.bearish_greedy_levels[i] =
                LevelFeatures::from_level(level, current_price, current_candle_index);
        }

        // Find latest active trends
        let latest_bullish_trend = trends
            .iter()
            .filter(|t| t.direction == CandleDirection::Bullish && t.state != TrendState::Broken)
            .max_by_key(|t| t.created_at_index);

        let latest_bearish_trend = trends
            .iter()
            .filter(|t| t.direction == CandleDirection::Bearish && t.state != TrendState::Broken)
            .max_by_key(|t| t.created_at_index);

        if let Some(trend) = latest_bullish_trend {
            features.bullish_trend =
                TrendFeatures::from_trend(trend, current_price, current_candle_index);
        }
        if let Some(trend) = latest_bearish_trend {
            features.bearish_trend =
                TrendFeatures::from_trend(trend, current_price, current_candle_index);
        }

        // Counts
        features.active_level_count = levels
            .iter()
            .filter(|l| l.state != LevelState::Broken)
            .count() as u16;
        features.active_trend_count = trends
            .iter()
            .filter(|t| t.state != TrendState::Broken)
            .count() as u16;

        features
    }

    /// Convert to flat feature vector.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(80); // Approximate size

        // Levels: 4 categories × 3 levels × 5 features = 60
        for level in &self.bullish_hold_levels {
            v.extend(level.to_vec());
        }
        for level in &self.bearish_hold_levels {
            v.extend(level.to_vec());
        }
        for level in &self.bullish_greedy_levels {
            v.extend(level.to_vec());
        }
        for level in &self.bearish_greedy_levels {
            v.extend(level.to_vec());
        }

        // Trends: 2 × 6 features = 12
        v.extend(self.bullish_trend.to_vec());
        v.extend(self.bearish_trend.to_vec());

        // Counts: 2
        v.push(self.active_level_count as f32);
        v.push(self.active_trend_count as f32);

        v
    }
}

/// Multi-timeframe ML features.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MlFeatures {
    /// Features per timeframe.
    pub timeframes: Vec<TimeframeFeatures>,
    /// Current price (for reference).
    pub current_price: f32,
    /// Current volume (normalized).
    pub current_volume_normalized: f32,
    /// Price change from previous candle (normalized).
    pub price_change_normalized: f32,
    /// Candle body ratio (body size / total range).
    pub body_ratio: f32,
    /// Whether current candle is bullish (1.0) or bearish (0.0).
    pub is_bullish: f32,
    /// RSI (14-period) normalized to 0-1 range.
    pub rsi_14: f32,
}

impl MlFeatures {
    /// Get total feature count.
    pub fn feature_count(&self) -> usize {
        // Per timeframe: 60 (levels) + 12 (trends) + 2 (counts) = 74
        // Global: 6 (added RSI)
        self.timeframes.len() * 74 + 6
    }

    /// Convert to flat feature vector for model input.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.feature_count());

        // Global features first
        v.push(self.current_volume_normalized);
        v.push(self.price_change_normalized);
        v.push(self.body_ratio);
        v.push(self.is_bullish);
        v.push(self.timeframes.len() as f32); // Number of timeframes
        v.push(self.rsi_14); // RSI feature

        // Then per-timeframe features
        for tf in &self.timeframes {
            v.extend(tf.to_vec());
        }

        v
    }
}

/// Training sample with features and labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input features.
    pub features: MlFeatures,
    /// Candle index when this sample was taken.
    pub candle_index: usize,
    /// Timestamp of the sample.
    pub timestamp: f64,

    // Labels (filled in after observing future price action)
    /// Did the nearest level break? (binary: 0 or 1)
    pub nearest_level_broke: Option<u8>,
    /// Price direction after N candles (1 = up, 0 = down).
    pub direction_after_n: Option<u8>,
    /// Price change percentage after N candles.
    pub price_change_pct: Option<f32>,
    /// Number of candles for lookahead.
    pub lookahead_candles: usize,
}

impl TrainingSample {
    /// Create a new training sample (labels will be filled later).
    pub fn new(features: MlFeatures, candle_index: usize, timestamp: f64, lookahead: usize) -> Self {
        Self {
            features,
            candle_index,
            timestamp,
            nearest_level_broke: None,
            direction_after_n: None,
            price_change_pct: None,
            lookahead_candles: lookahead,
        }
    }

    /// Fill in labels based on future price.
    pub fn fill_labels(&mut self, future_price: f32, level_broke: bool) {
        let current = self.features.current_price;
        if current > 0.0 {
            let change_pct = (future_price - current) / current * 100.0;
            self.price_change_pct = Some(change_pct);
            self.direction_after_n = Some(if future_price > current { 1 } else { 0 });
        }
        self.nearest_level_broke = Some(if level_broke { 1 } else { 0 });
    }

    /// Check if labels are complete.
    pub fn has_labels(&self) -> bool {
        self.nearest_level_broke.is_some()
            && self.direction_after_n.is_some()
            && self.price_change_pct.is_some()
    }
}

/// Model predictions output.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MlPrediction {
    /// Probability that the nearest level will break (0.0 to 1.0).
    pub level_break_prob: f32,
    /// Probability of upward price movement (0.0 to 1.0).
    pub direction_up_prob: f32,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

impl MlPrediction {
    /// Returns true if the model predicts the level will likely break.
    pub fn predicts_level_break(&self) -> bool {
        self.level_break_prob > 0.5
    }

    /// Returns true if the model predicts upward price movement.
    pub fn predicts_up(&self) -> bool {
        self.direction_up_prob > 0.5
    }

    /// Get the predicted direction as a string.
    pub fn direction_str(&self) -> &'static str {
        if self.direction_up_prob > 0.5 {
            "UP"
        } else {
            "DOWN"
        }
    }

    /// Get a simple signal strength (0-100).
    pub fn signal_strength(&self) -> u8 {
        let dir_strength = (self.direction_up_prob - 0.5).abs() * 2.0; // 0-1
        let conf = self.confidence;
        ((dir_strength * conf) * 100.0).min(100.0) as u8
    }
}

/// Training data collection for export.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// All training samples.
    pub samples: Vec<TrainingSample>,
    /// Feature dimension per sample.
    pub feature_dim: usize,
    /// Number of timeframes used.
    pub num_timeframes: usize,
    /// Lookahead candles used for labels.
    pub lookahead_candles: usize,
}

impl TrainingDataset {
    /// Create a new dataset.
    pub fn new(num_timeframes: usize, lookahead_candles: usize) -> Self {
        // Per timeframe: 74 features, plus 6 global (added RSI)
        let feature_dim = num_timeframes * 74 + 6;
        Self {
            samples: Vec::new(),
            feature_dim,
            num_timeframes,
            lookahead_candles,
        }
    }

    /// Add a sample.
    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    /// Get samples with complete labels.
    pub fn labeled_samples(&self) -> impl Iterator<Item = &TrainingSample> {
        self.samples.iter().filter(|s| s.has_labels())
    }

    /// Export to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export feature matrix and labels for Python.
    pub fn to_numpy_format(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for sample in self.labeled_samples() {
            features.push(sample.features.to_vec());
            labels.push(vec![
                sample.nearest_level_broke.unwrap_or(0) as f32,
                sample.direction_after_n.unwrap_or(0) as f32,
                sample.price_change_pct.unwrap_or(0.0),
            ]);
        }

        (features, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_features() {
        let features = LevelFeatures {
            price_distance: 0.05,
            hit_count: 3,
            respected_hits: 2,
            is_active: 1.0,
            age_normalized: 0.1,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec[0], 0.05);
        assert_eq!(vec[1], 3.0);
    }

    #[test]
    fn test_timeframe_features_vec_size() {
        let tf = TimeframeFeatures::default();
        let vec = tf.to_vec();
        // 4 categories × 3 levels × 5 features + 2 trends × 6 features + 2 counts
        // = 60 + 12 + 2 = 74
        assert_eq!(vec.len(), 74);
    }

    #[test]
    fn test_ml_features_vec_size() {
        let features = MlFeatures {
            timeframes: vec![TimeframeFeatures::default(); 3],
            current_price: 100.0,
            current_volume_normalized: 0.5,
            price_change_normalized: 0.01,
            body_ratio: 0.6,
            is_bullish: 1.0,
            rsi_14: 0.5,
        };

        let vec = features.to_vec();
        // 6 global + 3 × 74 = 6 + 222 = 228
        assert_eq!(vec.len(), 228);
    }
}
