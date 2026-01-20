//! Candle direction classification and metadata.

use charter_core::Candle;
use serde::{Deserialize, Serialize};

/// Classification of candle movement direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandleDirection {
    /// Close > Open
    Bullish,
    /// Close < Open
    Bearish,
    /// |Close - Open| / (High - Low) < threshold
    Doji,
}

impl CandleDirection {
    /// Classify a candle's direction with a given doji threshold.
    ///
    /// The threshold is the ratio of body size to total range below which
    /// a candle is considered a doji.
    #[inline]
    pub fn from_candle(candle: &Candle, doji_threshold: f32) -> Self {
        let body = candle.close - candle.open;
        let range = candle.high - candle.low;

        if range <= 0.0 {
            return CandleDirection::Doji;
        }

        let body_ratio = body.abs() / range;

        if body_ratio <= doji_threshold {
            CandleDirection::Doji
        } else if body > 0.0 {
            CandleDirection::Bullish
        } else {
            CandleDirection::Bearish
        }
    }

    /// Returns the opposite direction (Bullish <-> Bearish, Doji stays Doji).
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            CandleDirection::Bullish => CandleDirection::Bearish,
            CandleDirection::Bearish => CandleDirection::Bullish,
            CandleDirection::Doji => CandleDirection::Doji,
        }
    }

    /// Returns true if this is a directional candle (not a doji).
    #[inline]
    pub fn is_directional(self) -> bool {
        !matches!(self, CandleDirection::Doji)
    }
}

/// Pre-computed metadata for a candle.
///
/// Computing these values once and storing them avoids redundant calculations
/// during analysis.
#[derive(Debug, Clone, Copy)]
pub struct CandleMetadata {
    /// Classified direction
    pub direction: CandleDirection,
    /// Signed body size (close - open)
    pub body: f32,
    /// Absolute body size
    pub body_abs: f32,
    /// Total range (high - low)
    pub range: f32,
    /// Upper wick size
    pub wick_upper: f32,
    /// Lower wick size
    pub wick_lower: f32,
    /// Body ratio (body_abs / range), 0.0 if range is 0
    pub body_ratio: f32,
}

impl CandleMetadata {
    /// Compute metadata for a candle with a given doji threshold.
    pub fn from_candle(candle: &Candle, doji_threshold: f32) -> Self {
        let body = candle.close - candle.open;
        let body_abs = body.abs();
        let range = candle.high - candle.low;

        let body_ratio = if range > 0.0 { body_abs / range } else { 0.0 };

        let direction = if range <= 0.0 || body_ratio <= doji_threshold {
            CandleDirection::Doji
        } else if body > 0.0 {
            CandleDirection::Bullish
        } else {
            CandleDirection::Bearish
        };

        // Calculate wicks based on direction
        let (body_top, body_bottom) = if body >= 0.0 {
            (candle.close, candle.open)
        } else {
            (candle.open, candle.close)
        };

        Self {
            direction,
            body,
            body_abs,
            range,
            wick_upper: candle.high - body_top,
            wick_lower: body_bottom - candle.low,
            body_ratio,
        }
    }

    /// Get the top of the candle body.
    #[inline]
    pub fn body_top(&self, candle: &Candle) -> f32 {
        candle.open.max(candle.close)
    }

    /// Get the bottom of the candle body.
    #[inline]
    pub fn body_bottom(&self, candle: &Candle) -> f32 {
        candle.open.min(candle.close)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 0.0)
    }

    #[test]
    fn test_bullish_direction() {
        let candle = make_candle(100.0, 110.0, 95.0, 108.0);
        let dir = CandleDirection::from_candle(&candle, 0.1);
        assert_eq!(dir, CandleDirection::Bullish);
    }

    #[test]
    fn test_bearish_direction() {
        let candle = make_candle(108.0, 110.0, 95.0, 100.0);
        let dir = CandleDirection::from_candle(&candle, 0.1);
        assert_eq!(dir, CandleDirection::Bearish);
    }

    #[test]
    fn test_doji_direction() {
        let candle = make_candle(100.0, 110.0, 90.0, 100.5);
        let dir = CandleDirection::from_candle(&candle, 0.1);
        assert_eq!(dir, CandleDirection::Doji);
    }

    #[test]
    fn test_metadata_calculation() {
        let candle = make_candle(100.0, 115.0, 95.0, 110.0);
        let meta = CandleMetadata::from_candle(&candle, 0.1);

        assert_eq!(meta.direction, CandleDirection::Bullish);
        assert_eq!(meta.body, 10.0);
        assert_eq!(meta.body_abs, 10.0);
        assert_eq!(meta.range, 20.0);
        assert_eq!(meta.wick_upper, 5.0); // 115 - 110
        assert_eq!(meta.wick_lower, 5.0); // 100 - 95
        assert!((meta.body_ratio - 0.5).abs() < 0.001);
    }
}
