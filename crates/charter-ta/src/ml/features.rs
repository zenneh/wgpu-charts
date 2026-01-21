//! ML feature structures and extraction trait.

use std::collections::HashMap;

use charter_core::Candle;

use crate::analyzer::AnalyzerState;
use crate::types::LevelState;

/// Number of closest levels to track per category.
pub const N_LEVELS: usize = 3;

/// Errors that can occur during feature extraction.
#[derive(Debug, Clone)]
pub enum ExtractionError {
    /// Not enough data to extract features.
    InsufficientData(String),
    /// State is in an invalid condition.
    InvalidState(String),
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractionError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            ExtractionError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for ExtractionError {}

/// Requirements for feature extraction.
#[derive(Debug, Clone, Default)]
pub struct ExtractionRequirements {
    /// Minimum candles required per timeframe.
    pub min_candles_per_timeframe: HashMap<u8, usize>,
    /// Minimum ranges required per timeframe.
    pub min_ranges_per_timeframe: HashMap<u8, usize>,
    /// Minimum total active levels across all timeframes.
    pub min_active_levels: usize,
}

impl ExtractionRequirements {
    /// Create new requirements with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum candles for a specific timeframe.
    pub fn with_min_candles(mut self, timeframe_idx: u8, count: usize) -> Self {
        self.min_candles_per_timeframe.insert(timeframe_idx, count);
        self
    }

    /// Set minimum ranges for a specific timeframe.
    pub fn with_min_ranges(mut self, timeframe_idx: u8, count: usize) -> Self {
        self.min_ranges_per_timeframe.insert(timeframe_idx, count);
        self
    }

    /// Set minimum active levels.
    pub fn with_min_active_levels(mut self, count: usize) -> Self {
        self.min_active_levels = count;
        self
    }

    /// Check if requirements are met by the given state.
    pub fn is_satisfied(&self, state: &AnalyzerState) -> bool {
        // Check minimum active levels
        if state.total_active_levels() < self.min_active_levels {
            return false;
        }

        // Check per-timeframe requirements
        for (&tf_idx, &min_ranges) in &self.min_ranges_per_timeframe {
            if let Some(tf_state) = state.get_timeframe(tf_idx) {
                if tf_state.range_count() < min_ranges {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// Trait for extracting ML features from analyzer state.
pub trait FeatureExtractor {
    /// Extract features from current analyzer state.
    ///
    /// Returns `Ok(None)` if there is insufficient data for feature extraction.
    /// Returns `Ok(Some(features))` if extraction was successful.
    /// Returns `Err(error)` if there was an error during extraction.
    fn extract_features(&self) -> Result<Option<MlFeatures>, ExtractionError>;

    /// Get the requirements for feature extraction.
    fn extraction_requirements(&self) -> ExtractionRequirements;

    /// Check if extraction requirements are currently met.
    fn can_extract_features(&self) -> bool;
}

/// Features extracted from a single level.
#[derive(Debug, Clone, Copy, Default)]
pub struct LevelFeatures {
    /// 1.0 if this level exists, 0.0 for padding.
    pub exists: f32,
    /// Normalized price distance: (level_price - current_price) / current_price.
    pub price_distance: f32,
    /// Number of times this level has been hit.
    pub hit_count: u8,
    /// Ratio of respected hits to total hits.
    pub respected_ratio: f32,
    /// 1.0 if level is active, 0.0 if inactive.
    pub is_active: f32,
    /// Normalized age: candles_since_creation / normalization_factor.
    pub age_normalized: f32,
}

impl LevelFeatures {
    /// Create an empty (padding) level feature.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create features from level data.
    pub fn from_level(
        price: f32,
        current_price: f32,
        hit_count: usize,
        respected_count: usize,
        is_active: bool,
        created_at_index: usize,
        current_index: usize,
        age_normalization: f32,
    ) -> Self {
        let total_hits = hit_count.max(1) as f32;
        Self {
            exists: 1.0,
            price_distance: (price - current_price) / current_price,
            hit_count: hit_count.min(255) as u8,
            respected_ratio: respected_count as f32 / total_hits,
            is_active: if is_active { 1.0 } else { 0.0 },
            age_normalized: (current_index.saturating_sub(created_at_index) as f32)
                / age_normalization,
        }
    }
}

/// Features for a single timeframe.
#[derive(Debug, Clone)]
pub struct TimeframeFeatures {
    /// Index of this timeframe.
    pub timeframe_index: u8,

    /// Closest N support levels (sorted by distance to current price).
    pub support_levels: [LevelFeatures; N_LEVELS],

    /// Closest N resistance levels (sorted by distance to current price).
    pub resistance_levels: [LevelFeatures; N_LEVELS],

    /// Number of active levels in this timeframe.
    pub active_level_count: u16,

    /// Total number of levels in this timeframe.
    pub total_level_count: u16,

    /// Number of ranges detected.
    pub range_count: u16,
}

impl Default for TimeframeFeatures {
    fn default() -> Self {
        Self {
            timeframe_index: 0,
            support_levels: [LevelFeatures::empty(); N_LEVELS],
            resistance_levels: [LevelFeatures::empty(); N_LEVELS],
            active_level_count: 0,
            total_level_count: 0,
            range_count: 0,
        }
    }
}

/// Complete ML features extracted from analyzer state.
#[derive(Debug, Clone, Default)]
pub struct MlFeatures {
    /// Features for each timeframe.
    pub timeframes: Vec<TimeframeFeatures>,

    /// Current price.
    pub current_price: f32,

    /// Price at the start of analysis (for normalization).
    pub reference_price: f32,

    /// Total active levels across all timeframes.
    pub total_active_levels: u16,

    /// Total levels across all timeframes.
    pub total_levels: u16,

    /// Whether we have resistance above current price.
    pub has_resistance_above: bool,

    /// Whether we have support below current price.
    pub has_support_below: bool,

    /// Distance to closest resistance (normalized).
    pub closest_resistance_distance: Option<f32>,

    /// Distance to closest support (normalized).
    pub closest_support_distance: Option<f32>,

    // === Price Action Features ===
    /// Candle body ratio: |close - open| / (high - low), 0 if doji/no range.
    pub body_ratio: f32,

    /// Upper wick ratio: (high - max(open, close)) / (high - low).
    pub upper_wick_ratio: f32,

    /// Lower wick ratio: (min(open, close) - low) / (high - low).
    pub lower_wick_ratio: f32,

    /// 1.0 if close > open (bullish), 0.0 otherwise.
    pub is_bullish: f32,

    // === Momentum Features ===
    /// Price change over 1 candle: (close - prev_close) / prev_close.
    pub price_change_1: f32,

    /// Price change over 3 candles: (close - close_3_ago) / close_3_ago.
    pub price_change_3: f32,

    /// Price change over 5 candles: (close - close_5_ago) / close_5_ago.
    pub price_change_5: f32,
}

impl MlFeatures {
    /// Create new MlFeatures.
    pub fn new() -> Self {
        Self::default()
    }

    /// Flatten features into a fixed-size f32 vector for ML input.
    ///
    /// The output format matches the CSV export format:
    /// [current_price, reference_price, total_active_levels, total_levels,
    ///  has_resistance, has_support, closest_resistance_dist, closest_support_dist,
    ///  body_ratio, upper_wick_ratio, lower_wick_ratio, is_bullish,
    ///  price_change_1, price_change_3, price_change_5,
    ///  ...per-timeframe features...]
    ///
    /// Note: timeframe_index is NOT included (it's redundant since ordering is fixed).
    pub fn flatten(&self) -> Vec<f32> {
        let mut output = Vec::new();

        // Global features (15 total)
        // Original 8 features
        output.push(self.current_price);
        output.push(self.reference_price);
        output.push(self.total_active_levels as f32);
        output.push(self.total_levels as f32);
        output.push(if self.has_resistance_above { 1.0 } else { 0.0 });
        output.push(if self.has_support_below { 1.0 } else { 0.0 });
        output.push(self.closest_resistance_distance.unwrap_or(0.0));
        output.push(self.closest_support_distance.unwrap_or(0.0));

        // Price action features (4)
        output.push(self.body_ratio);
        output.push(self.upper_wick_ratio);
        output.push(self.lower_wick_ratio);
        output.push(self.is_bullish);

        // Momentum features (3)
        output.push(self.price_change_1);
        output.push(self.price_change_3);
        output.push(self.price_change_5);

        // Per-timeframe features (39 each: 3 base + 18 support + 18 resistance)
        for tf in &self.timeframes {
            // Base features (3) - NOT including timeframe_index
            output.push(tf.active_level_count as f32);
            output.push(tf.total_level_count as f32);
            output.push(tf.range_count as f32);

            // Support levels (6 features × N_LEVELS)
            for level in &tf.support_levels {
                output.push(level.exists);
                output.push(level.price_distance);
                output.push(level.hit_count as f32);
                output.push(level.respected_ratio);
                output.push(level.is_active);
                output.push(level.age_normalized);
            }

            // Resistance levels (6 features × N_LEVELS)
            for level in &tf.resistance_levels {
                output.push(level.exists);
                output.push(level.price_distance);
                output.push(level.hit_count as f32);
                output.push(level.respected_ratio);
                output.push(level.is_active);
                output.push(level.age_normalized);
            }
        }

        output
    }

    /// Get the expected size of flattened features for a given number of timeframes.
    pub fn flattened_size(num_timeframes: usize) -> usize {
        // Global features: 15 (8 original + 4 price action + 3 momentum)
        // Per-timeframe: 3 + (N_LEVELS * 6) * 2 = 3 + 36 = 39
        let per_timeframe = 3 + (N_LEVELS * 6) * 2;
        15 + num_timeframes * per_timeframe
    }

    /// Convert features to a vector (alias for flatten()).
    pub fn to_vec(&self) -> Vec<f32> {
        self.flatten()
    }

    /// Get the number of features.
    pub fn feature_count(&self) -> usize {
        Self::flattened_size(self.timeframes.len())
    }
}

/// ML model prediction result.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlPrediction {
    /// Probability that a level will break.
    pub level_break_prob: f32,
    /// Probability that price will go up.
    pub direction_up_prob: f32,
    /// Confidence in the prediction (0.0 - 1.0).
    pub confidence: f32,
}

/// Extract features from analyzer state.
pub fn extract_features_from_state(
    state: &AnalyzerState,
    current_index: usize,
    age_normalization: f32,
) -> MlFeatures {
    let mut features = MlFeatures::new();
    features.current_price = state.current_price;
    features.reference_price = state.current_price; // Can be set differently
    features.total_active_levels = state.total_active_levels() as u16;
    features.total_levels = state.total_levels() as u16;

    // Closest levels
    if let Some((price, _)) = state.closest_unbroken_resistance {
        features.has_resistance_above = true;
        features.closest_resistance_distance =
            Some((price - state.current_price) / state.current_price);
    }

    if let Some((price, _)) = state.closest_unbroken_support {
        features.has_support_below = true;
        // Use consistent formula: (level_price - current_price) / current_price
        // Support below = negative distance, Resistance above = positive distance
        features.closest_support_distance =
            Some((price - state.current_price) / state.current_price);
    }

    // Per-timeframe features
    for (&tf_idx, tf_state) in &state.timeframe_states {
        let mut tf_features = TimeframeFeatures {
            timeframe_index: tf_idx,
            active_level_count: tf_state.active_level_count() as u16,
            total_level_count: tf_state.level_count() as u16,
            range_count: tf_state.range_count() as u16,
            ..Default::default()
        };

        // Get closest support levels
        let support_levels = tf_state
            .level_index
            .closest_n_support_below(state.current_price, N_LEVELS);
        for (i, level) in support_levels.into_iter().enumerate() {
            tf_features.support_levels[i] = LevelFeatures::from_level(
                level.price,
                state.current_price,
                level.hit_count(),
                level.respected_hit_count(),
                level.state == LevelState::Active,
                level.created_at_index,
                current_index,
                age_normalization,
            );
        }

        // Get closest resistance levels
        let resistance_levels = tf_state
            .level_index
            .closest_n_resistance_above(state.current_price, N_LEVELS);
        for (i, level) in resistance_levels.into_iter().enumerate() {
            tf_features.resistance_levels[i] = LevelFeatures::from_level(
                level.price,
                state.current_price,
                level.hit_count(),
                level.respected_hit_count(),
                level.state == LevelState::Active,
                level.created_at_index,
                current_index,
                age_normalization,
            );
        }

        features.timeframes.push(tf_features);
    }

    // Sort timeframes by index
    features.timeframes.sort_by_key(|tf| tf.timeframe_index);

    features
}

/// Extract features from analyzer state with price action and momentum from candles.
///
/// This is the preferred extraction function as it includes all feature types:
/// - Level-based features from analyzer state
/// - Price action features from current candle
/// - Momentum features from candle history
///
/// # Arguments
/// * `state` - The analyzer state containing level information
/// * `candles` - Slice of candles for momentum calculation (most recent at end)
/// * `current_index` - Index of the current candle in the analysis
/// * `age_normalization` - Factor for normalizing level ages
pub fn extract_features_with_candles(
    state: &AnalyzerState,
    candles: &[Candle],
    current_index: usize,
    age_normalization: f32,
) -> MlFeatures {
    // Start with basic state-based features
    let mut features = extract_features_from_state(state, current_index, age_normalization);

    // Add price action and momentum if we have candle data
    if let Some(current_candle) = candles.last() {
        // Price action features from current candle
        let range = current_candle.high - current_candle.low;
        if range > f32::EPSILON {
            let body = (current_candle.close - current_candle.open).abs();
            let upper_wick = current_candle.high - current_candle.open.max(current_candle.close);
            let lower_wick = current_candle.open.min(current_candle.close) - current_candle.low;

            features.body_ratio = body / range;
            features.upper_wick_ratio = upper_wick / range;
            features.lower_wick_ratio = lower_wick / range;
        }
        features.is_bullish = if current_candle.close > current_candle.open {
            1.0
        } else {
            0.0
        };

        // Momentum features from candle history
        let n = candles.len();
        if n > 1 {
            let prev_close = candles[n - 2].close;
            if prev_close > f32::EPSILON {
                features.price_change_1 = (current_candle.close - prev_close) / prev_close;
            }
        }
        if n > 3 {
            let close_3_ago = candles[n - 4].close;
            if close_3_ago > f32::EPSILON {
                features.price_change_3 = (current_candle.close - close_3_ago) / close_3_ago;
            }
        }
        if n > 5 {
            let close_5_ago = candles[n - 6].close;
            if close_5_ago > f32::EPSILON {
                features.price_change_5 = (current_candle.close - close_5_ago) / close_5_ago;
            }
        }
    }

    features
}

/// Compute price action features from a single candle.
/// Returns (body_ratio, upper_wick_ratio, lower_wick_ratio, is_bullish).
pub fn compute_price_action(candle: &Candle) -> (f32, f32, f32, f32) {
    let range = candle.high - candle.low;
    if range <= f32::EPSILON {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let body = (candle.close - candle.open).abs();
    let upper_wick = candle.high - candle.open.max(candle.close);
    let lower_wick = candle.open.min(candle.close) - candle.low;
    let is_bullish = if candle.close > candle.open { 1.0 } else { 0.0 };

    (body / range, upper_wick / range, lower_wick / range, is_bullish)
}

/// Compute momentum features from candle history.
/// Returns (price_change_1, price_change_3, price_change_5).
pub fn compute_momentum(candles: &[Candle]) -> (f32, f32, f32) {
    let n = candles.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let current_close = candles[n - 1].close;
    let mut pc1 = 0.0;
    let mut pc3 = 0.0;
    let mut pc5 = 0.0;

    if n > 1 {
        let prev_close = candles[n - 2].close;
        if prev_close > f32::EPSILON {
            pc1 = (current_close - prev_close) / prev_close;
        }
    }
    if n > 3 {
        let close_3_ago = candles[n - 4].close;
        if close_3_ago > f32::EPSILON {
            pc3 = (current_close - close_3_ago) / close_3_ago;
        }
    }
    if n > 5 {
        let close_5_ago = candles[n - 6].close;
        if close_5_ago > f32::EPSILON {
            pc5 = (current_close - close_5_ago) / close_5_ago;
        }
    }

    (pc1, pc3, pc5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_features_empty() {
        let f = LevelFeatures::empty();
        assert_eq!(f.exists, 0.0);
        assert_eq!(f.price_distance, 0.0);
    }

    #[test]
    fn test_level_features_from_level() {
        let f = LevelFeatures::from_level(
            110.0,  // price
            100.0,  // current_price
            5,      // hit_count
            3,      // respected_count
            true,   // is_active
            10,     // created_at_index
            50,     // current_index
            100.0,  // age_normalization
        );

        assert_eq!(f.exists, 1.0);
        assert!((f.price_distance - 0.1).abs() < 0.001); // (110-100)/100 = 0.1
        assert_eq!(f.hit_count, 5);
        assert!((f.respected_ratio - 0.6).abs() < 0.001); // 3/5 = 0.6
        assert_eq!(f.is_active, 1.0);
        assert!((f.age_normalized - 0.4).abs() < 0.001); // (50-10)/100 = 0.4
    }

    #[test]
    fn test_ml_features_flatten_size() {
        // 15 global + (3 + 36) per timeframe
        assert_eq!(MlFeatures::flattened_size(1), 15 + (3 + 36));  // = 54
        assert_eq!(MlFeatures::flattened_size(2), 15 + 2 * (3 + 36));  // = 93
        assert_eq!(MlFeatures::flattened_size(7), 15 + 7 * 39);  // = 288
    }

    #[test]
    fn test_ml_features_flatten() {
        let mut features = MlFeatures::new();
        features.current_price = 100.0;
        features.reference_price = 95.0;
        features.total_active_levels = 5;
        features.total_levels = 10;
        features.has_resistance_above = true;
        features.has_support_below = true;
        features.closest_resistance_distance = Some(0.05);
        features.closest_support_distance = Some(-0.03); // Negative for support below
        // Price action features
        features.body_ratio = 0.6;
        features.upper_wick_ratio = 0.2;
        features.lower_wick_ratio = 0.2;
        features.is_bullish = 1.0;
        // Momentum features
        features.price_change_1 = 0.01;
        features.price_change_3 = 0.02;
        features.price_change_5 = 0.03;

        let flattened = features.flatten();

        // Check first 15 values (global features)
        assert_eq!(flattened[0], 100.0); // current_price
        assert_eq!(flattened[1], 95.0);  // reference_price
        assert_eq!(flattened[2], 5.0);   // total_active_levels
        assert_eq!(flattened[3], 10.0);  // total_levels
        assert_eq!(flattened[4], 1.0);   // has_resistance_above
        assert_eq!(flattened[5], 1.0);   // has_support_below
        assert!((flattened[6] - 0.05).abs() < 0.001); // closest_resistance_distance
        assert!((flattened[7] - -0.03).abs() < 0.001); // closest_support_distance
        // Price action
        assert!((flattened[8] - 0.6).abs() < 0.001);  // body_ratio
        assert!((flattened[9] - 0.2).abs() < 0.001);  // upper_wick_ratio
        assert!((flattened[10] - 0.2).abs() < 0.001); // lower_wick_ratio
        assert_eq!(flattened[11], 1.0);               // is_bullish
        // Momentum
        assert!((flattened[12] - 0.01).abs() < 0.001); // price_change_1
        assert!((flattened[13] - 0.02).abs() < 0.001); // price_change_3
        assert!((flattened[14] - 0.03).abs() < 0.001); // price_change_5
    }

    #[test]
    fn test_extraction_requirements() {
        let reqs = ExtractionRequirements::new()
            .with_min_active_levels(3)
            .with_min_ranges(0, 2);

        assert_eq!(reqs.min_active_levels, 3);
        assert_eq!(reqs.min_ranges_per_timeframe.get(&0), Some(&2));
    }
}
