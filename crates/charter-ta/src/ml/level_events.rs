//! Level interaction event detection and feature extraction for ML.
//!
//! This module provides:
//! - Detection of "level approach" events (when price gets close to a level)
//! - Feature extraction for predicting hold vs break
//! - Export functionality for training data

use charter_core::Candle;
use serde::{Deserialize, Serialize};

use crate::types::{Level, LevelDirection, LevelState};

/// Proximity threshold to consider price "approaching" a level (as fraction of price).
pub const APPROACH_THRESHOLD: f32 = 0.0015; // 0.15%

/// Threshold for determining if level broke (price went through).
pub const BREAK_THRESHOLD: f32 = 0.002; // 0.2% through level

/// Threshold for determining if level held (price bounced).
pub const HOLD_THRESHOLD: f32 = 0.002; // 0.2% bounce (symmetric with break)

/// Minimum candles a level must exist before we consider events.
pub const MIN_LEVEL_AGE: usize = 10;

/// Result of a level interaction - did it hold or break?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LevelOutcome {
    /// Level held - price bounced
    Held,
    /// Level broke - price went through
    Broke,
    /// Still pending - not enough data yet
    Pending,
}

/// A level approach event - when price gets close to a level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelApproachEvent {
    /// Candle index when approach started
    pub start_index: usize,
    /// Candle index when outcome was determined
    pub end_index: Option<usize>,
    /// The level price
    pub level_price: f32,
    /// Level direction (support or resistance)
    pub level_direction: LevelDirection,
    /// Timeframe that created this level
    pub level_timeframe: u8,
    /// Price when approach started
    pub approach_price: f32,
    /// The outcome (held, broke, or pending)
    pub outcome: LevelOutcome,
    /// Features extracted at approach time
    pub features: LevelEventFeatures,
}

/// Features for a single level approach event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelEventFeatures {
    // === Level Properties ===
    /// Distance to level as fraction of price (negative = below, positive = above)
    pub distance_to_level: f32,
    /// Total times this level has been hit before
    pub prior_hit_count: u8,
    /// Ratio of respected hits (bounces) to total hits
    pub respected_ratio: f32,
    /// Age of level in candles
    pub level_age: u16,
    /// Timeframe of level (higher = stronger) normalized 0-1
    pub level_timeframe_strength: f32,
    /// 1.0 if support, 0.0 if resistance
    pub is_support: f32,

    // === Approach Dynamics ===
    /// Momentum into level (price change over last 3 candles)
    pub approach_momentum: f32,
    /// Average body ratio of last 3 candles (trend strength)
    pub approach_body_ratio: f32,
    /// 1.0 if approaching with trend (bearish into support, bullish into resistance)
    pub with_trend_approach: f32,

    // === Current Candle ===
    /// Body ratio of current candle
    pub current_body_ratio: f32,
    /// Upper wick ratio
    pub current_upper_wick: f32,
    /// Lower wick ratio
    pub current_lower_wick: f32,
    /// 1.0 if bullish, 0.0 if bearish
    pub current_is_bullish: f32,

    // === Context ===
    /// Number of other active levels nearby (within 1%)
    pub nearby_level_count: u8,
    /// Distance to next level on other side
    pub opposite_level_distance: f32,
}

impl LevelEventFeatures {
    /// Flatten features into a vector for ML.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.distance_to_level,
            self.prior_hit_count as f32,
            self.respected_ratio,
            self.level_age as f32 / 1000.0, // Normalize age
            self.level_timeframe_strength,
            self.is_support,
            self.approach_momentum,
            self.approach_body_ratio,
            self.with_trend_approach,
            self.current_body_ratio,
            self.current_upper_wick,
            self.current_lower_wick,
            self.current_is_bullish,
            self.nearby_level_count as f32 / 10.0, // Normalize
            self.opposite_level_distance,
        ]
    }

    /// Number of features.
    pub const fn feature_count() -> usize {
        15
    }

    /// Feature names for CSV header.
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "distance_to_level",
            "prior_hit_count",
            "respected_ratio",
            "level_age",
            "level_timeframe_strength",
            "is_support",
            "approach_momentum",
            "approach_body_ratio",
            "with_trend_approach",
            "current_body_ratio",
            "current_upper_wick",
            "current_lower_wick",
            "current_is_bullish",
            "nearby_level_count",
            "opposite_level_distance",
        ]
    }
}

/// Extract features for a level approach.
pub fn extract_level_features(
    level: &Level,
    current_price: f32,
    current_candle: &Candle,
    recent_candles: &[Candle], // Last 3-5 candles for momentum
    current_index: usize,
    nearby_levels: usize,
    opposite_distance: Option<f32>,
) -> LevelEventFeatures {
    let mut features = LevelEventFeatures::default();

    // Level properties
    features.distance_to_level = (level.price - current_price) / current_price;
    features.prior_hit_count = level.hits.len().min(255) as u8;
    features.respected_ratio = if level.hits.is_empty() {
        0.5 // No data, assume neutral
    } else {
        level.hits.iter().filter(|h| h.respected).count() as f32 / level.hits.len() as f32
    };
    features.level_age = current_index.saturating_sub(level.created_at_index).min(65535) as u16;

    // Timeframe strength (higher TF = stronger)
    // 0=1m, 1=3m, 2=5m, 3=30m, 4=1h, 5=1d, 6=1w
    features.level_timeframe_strength = (level.source_timeframe as f32) / 6.0;

    features.is_support = match level.level_direction {
        LevelDirection::Support => 1.0,
        LevelDirection::Resistance => 0.0,
    };

    // Approach dynamics from recent candles
    if recent_candles.len() >= 3 {
        let n = recent_candles.len();
        let oldest = &recent_candles[0];
        let newest = &recent_candles[n - 1];

        // Momentum: price change over the period
        features.approach_momentum = (newest.close - oldest.close) / oldest.close;

        // Average body ratio
        let total_body_ratio: f32 = recent_candles
            .iter()
            .map(|c| {
                let range = c.high - c.low;
                if range > f32::EPSILON {
                    (c.close - c.open).abs() / range
                } else {
                    0.0
                }
            })
            .sum();
        features.approach_body_ratio = total_body_ratio / recent_candles.len() as f32;

        // With trend approach
        let approaching_down = newest.close < oldest.close;
        let is_support = level.level_direction == LevelDirection::Support;
        features.with_trend_approach = if (approaching_down && is_support)
            || (!approaching_down && !is_support)
        {
            1.0
        } else {
            0.0
        };
    }

    // Current candle features
    let range = current_candle.high - current_candle.low;
    if range > f32::EPSILON {
        let body = (current_candle.close - current_candle.open).abs();
        let upper = current_candle.high - current_candle.open.max(current_candle.close);
        let lower = current_candle.open.min(current_candle.close) - current_candle.low;

        features.current_body_ratio = body / range;
        features.current_upper_wick = upper / range;
        features.current_lower_wick = lower / range;
    }
    features.current_is_bullish = if current_candle.close > current_candle.open {
        1.0
    } else {
        0.0
    };

    // Context
    features.nearby_level_count = nearby_levels.min(255) as u8;
    features.opposite_level_distance = opposite_distance.unwrap_or(0.1); // Default 10% if none

    features
}

/// Check if price is approaching a level.
pub fn is_approaching_level(level: &Level, current_price: f32) -> bool {
    if level.state != LevelState::Active {
        return false;
    }

    let distance = (level.price - current_price).abs() / current_price;
    distance <= APPROACH_THRESHOLD
}

/// Determine outcome of a level approach by looking at subsequent candles.
///
/// Returns (outcome, candles_to_resolution)
pub fn determine_outcome(
    level: &Level,
    approach_candle: &Candle,
    subsequent_candles: &[Candle],
    max_lookahead: usize,
) -> (LevelOutcome, usize) {
    let lookahead = subsequent_candles.len().min(max_lookahead);

    // Track the extreme prices during lookahead
    let mut max_bounce = 0.0f32; // How far price bounced away from level
    let mut max_break = 0.0f32;  // How far price went through level

    for (i, candle) in subsequent_candles.iter().take(lookahead).enumerate() {
        match level.level_direction {
            LevelDirection::Support => {
                // For support: bounce = price going up, break = price going down
                let bounce = (candle.high - level.price) / level.price;
                let through = (level.price - candle.low) / level.price;

                max_bounce = max_bounce.max(bounce);
                max_break = max_break.max(through);

                // Check for clear break (closed below)
                if candle.close < level.price * (1.0 - BREAK_THRESHOLD) {
                    return (LevelOutcome::Broke, i + 1);
                }
                // Check for clear hold (bounced significantly)
                if bounce >= HOLD_THRESHOLD && candle.close > level.price {
                    return (LevelOutcome::Held, i + 1);
                }
            }
            LevelDirection::Resistance => {
                // For resistance: bounce = price going down, break = price going up
                let bounce = (level.price - candle.low) / level.price;
                let through = (candle.high - level.price) / level.price;

                max_bounce = max_bounce.max(bounce);
                max_break = max_break.max(through);

                // Check for clear break (closed above)
                if candle.close > level.price * (1.0 + BREAK_THRESHOLD) {
                    return (LevelOutcome::Broke, i + 1);
                }
                // Check for clear hold (bounced significantly)
                if bounce >= HOLD_THRESHOLD && candle.close < level.price {
                    return (LevelOutcome::Held, i + 1);
                }
            }
        }
    }

    // Check the approach candle itself for immediate resolution
    let body_top = approach_candle.open.max(approach_candle.close);
    let body_bottom = approach_candle.open.min(approach_candle.close);

    match level.level_direction {
        LevelDirection::Support => {
            // Wick touched but body stayed above = held (if wick rejection is significant)
            let wick_rejection = (body_bottom - approach_candle.low) / level.price;
            if approach_candle.low <= level.price * (1.0 + APPROACH_THRESHOLD)
                && body_bottom > level.price
                && wick_rejection >= HOLD_THRESHOLD * 0.5
            {
                return (LevelOutcome::Held, 0);
            }
        }
        LevelDirection::Resistance => {
            // Wick touched but body stayed below = held (if wick rejection is significant)
            let wick_rejection = (approach_candle.high - body_top) / level.price;
            if approach_candle.high >= level.price * (1.0 - APPROACH_THRESHOLD)
                && body_top < level.price
                && wick_rejection >= HOLD_THRESHOLD * 0.5
            {
                return (LevelOutcome::Held, 0);
            }
        }
    }

    // If lookahead exhausted, classify based on which was stronger
    if lookahead >= max_lookahead {
        if max_bounce > max_break && max_bounce >= HOLD_THRESHOLD * 0.5 {
            return (LevelOutcome::Held, lookahead);
        }
        if max_break > max_bounce && max_break >= BREAK_THRESHOLD * 0.5 {
            return (LevelOutcome::Broke, lookahead);
        }
    }

    (LevelOutcome::Pending, lookahead)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        let features = LevelEventFeatures::default();
        assert_eq!(features.to_vec().len(), LevelEventFeatures::feature_count());
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(
            LevelEventFeatures::feature_names().len(),
            LevelEventFeatures::feature_count()
        );
    }
}
