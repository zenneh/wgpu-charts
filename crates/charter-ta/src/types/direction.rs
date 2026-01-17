//! Candle direction types and helpers.

use charter_core::Candle;

/// The direction of a candle based on open/close relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleDirection {
    /// Close > Open (price went up)
    Bullish,
    /// Close < Open (price went down)
    Bearish,
    /// Close == Open (within tolerance)
    Doji,
}

impl CandleDirection {
    /// Determine direction from a candle with default doji threshold.
    #[inline]
    pub fn from_candle(candle: &Candle) -> Self {
        Self::from_candle_with_threshold(candle, 0.0)
    }

    /// Determine direction from a candle with custom doji threshold.
    ///
    /// The threshold is a ratio of body size to total range.
    /// If body_ratio <= threshold, the candle is considered a Doji.
    /// When threshold is 0.0, only exact open == close is considered Doji.
    #[inline]
    pub fn from_candle_with_threshold(candle: &Candle, doji_threshold: f32) -> Self {
        let body = candle.close - candle.open;
        let range = candle.high - candle.low;

        if range == 0.0 {
            return CandleDirection::Doji;
        }

        // Exact doji: open == close
        if body == 0.0 {
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

    /// Returns true if this is a bullish direction.
    #[inline]
    pub fn is_bullish(self) -> bool {
        matches!(self, CandleDirection::Bullish)
    }

    /// Returns true if this is a bearish direction.
    #[inline]
    pub fn is_bearish(self) -> bool {
        matches!(self, CandleDirection::Bearish)
    }

    /// Returns true if this is a doji.
    #[inline]
    pub fn is_doji(self) -> bool {
        matches!(self, CandleDirection::Doji)
    }

    /// Returns the opposite direction.
    /// Doji returns Doji.
    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            CandleDirection::Bullish => CandleDirection::Bearish,
            CandleDirection::Bearish => CandleDirection::Bullish,
            CandleDirection::Doji => CandleDirection::Doji,
        }
    }
}

impl std::fmt::Display for CandleDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandleDirection::Bullish => write!(f, "Bullish"),
            CandleDirection::Bearish => write!(f, "Bearish"),
            CandleDirection::Doji => write!(f, "Doji"),
        }
    }
}

/// Pre-computed metadata for a candle to avoid repeated calculations.
#[derive(Debug, Clone, Copy)]
pub struct CandleMetadata {
    /// The direction of this candle.
    pub direction: CandleDirection,
    /// Size of the body (close - open), can be negative.
    pub body: f32,
    /// Absolute size of the body.
    pub body_abs: f32,
    /// Total range (high - low).
    pub range: f32,
    /// Upper wick size.
    pub wick_upper: f32,
    /// Lower wick size.
    pub wick_lower: f32,
    /// Body ratio (body_abs / range), 0.0 if range is 0.
    pub body_ratio: f32,
}

impl CandleMetadata {
    /// Compute metadata for a candle.
    #[inline]
    pub fn from_candle(candle: &Candle, doji_threshold: f32) -> Self {
        let body = candle.close - candle.open;
        let body_abs = body.abs();
        let range = candle.high - candle.low;

        let body_ratio = if range > 0.0 { body_abs / range } else { 0.0 };

        // Use the same logic as CandleDirection::from_candle_with_threshold
        let direction = CandleDirection::from_candle_with_threshold(candle, doji_threshold);

        // Calculate wicks based on direction
        let (wick_upper, wick_lower) = if body >= 0.0 {
            // Bullish or Doji (close >= open)
            (candle.high - candle.close, candle.open - candle.low)
        } else {
            // Bearish (close < open)
            (candle.high - candle.open, candle.close - candle.low)
        };

        Self {
            direction,
            body,
            body_abs,
            range,
            wick_upper,
            wick_lower,
            body_ratio,
        }
    }

    /// Returns the body top price (max of open/close).
    #[inline]
    pub fn body_top(&self, candle: &Candle) -> f32 {
        candle.open.max(candle.close)
    }

    /// Returns the body bottom price (min of open/close).
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
        let candle = make_candle(100.0, 110.0, 95.0, 105.0);
        assert_eq!(CandleDirection::from_candle(&candle), CandleDirection::Bullish);
    }

    #[test]
    fn test_bearish_direction() {
        let candle = make_candle(100.0, 105.0, 90.0, 95.0);
        assert_eq!(CandleDirection::from_candle(&candle), CandleDirection::Bearish);
    }

    #[test]
    fn test_doji_direction() {
        let candle = make_candle(100.0, 105.0, 95.0, 100.0);
        assert_eq!(CandleDirection::from_candle(&candle), CandleDirection::Doji);
    }

    #[test]
    fn test_doji_with_threshold() {
        // Body is 2, range is 20, body_ratio = 0.1
        let candle = make_candle(100.0, 110.0, 90.0, 102.0);
        assert_eq!(
            CandleDirection::from_candle_with_threshold(&candle, 0.15),
            CandleDirection::Doji
        );
        assert_eq!(
            CandleDirection::from_candle_with_threshold(&candle, 0.05),
            CandleDirection::Bullish
        );
    }

    #[test]
    fn test_metadata() {
        let candle = make_candle(100.0, 110.0, 95.0, 105.0);
        let meta = CandleMetadata::from_candle(&candle, 0.0);

        assert_eq!(meta.direction, CandleDirection::Bullish);
        assert_eq!(meta.body, 5.0);
        assert_eq!(meta.body_abs, 5.0);
        assert_eq!(meta.range, 15.0);
        assert_eq!(meta.wick_upper, 5.0);  // 110 - 105
        assert_eq!(meta.wick_lower, 5.0);  // 100 - 95
    }
}
