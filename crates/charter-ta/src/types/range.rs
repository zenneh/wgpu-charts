//! Range types - consecutive candles of the same direction.

use charter_core::Candle;

use super::direction::CandleDirection;

/// Unique identifier for a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeId(pub u64);

impl RangeId {
    /// Create a new range ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// A consecutive sequence of candles with the same direction.
///
/// Ranges are the building blocks for levels and trends.
/// A range is formed when multiple candles of the same direction
/// appear consecutively.
#[derive(Debug, Clone)]
pub struct Range {
    /// Unique identifier for this range.
    pub id: RangeId,
    /// Direction of all candles in this range.
    pub direction: CandleDirection,
    /// Start index in the candle array (inclusive).
    pub start_index: usize,
    /// End index in the candle array (inclusive).
    pub end_index: usize,
    /// Number of candles in this range.
    pub candle_count: usize,
    /// Highest high in the range.
    pub high: f32,
    /// Lowest low in the range.
    pub low: f32,
    /// Open price of the first candle.
    pub open: f32,
    /// Close price of the last candle.
    pub close: f32,
    /// Total volume across all candles.
    pub total_volume: f32,
    /// High of the first candle (for level calculation).
    pub first_high: f32,
    /// Low of the first candle (for level calculation).
    pub first_low: f32,
    /// High of the last candle (for level calculation).
    pub last_high: f32,
    /// Low of the last candle (for level calculation).
    pub last_low: f32,
}

impl Range {
    /// Total price range (high - low).
    #[inline]
    pub fn price_range(&self) -> f32 {
        self.high - self.low
    }

    /// Body of the range (close - open).
    #[inline]
    pub fn body(&self) -> f32 {
        self.close - self.open
    }

    /// Absolute body size.
    #[inline]
    pub fn body_abs(&self) -> f32 {
        (self.close - self.open).abs()
    }

    /// Calculate the hold level price for this range.
    ///
    /// For bearish ranges: min(first_low, last_low)
    /// For bullish ranges: max(first_high, last_high)
    #[inline]
    pub fn hold_level_price(&self) -> f32 {
        match self.direction {
            CandleDirection::Bearish => self.first_low.min(self.last_low),
            CandleDirection::Bullish => self.first_high.max(self.last_high),
            CandleDirection::Doji => (self.first_low + self.first_high) / 2.0,
        }
    }

    /// Calculate the greedy hold level price for this range.
    ///
    /// For bearish ranges: max(first_low, last_low) - the "greedier" level
    /// For bullish ranges: min(first_high, last_high) - the "greedier" level
    #[inline]
    pub fn greedy_hold_level_price(&self) -> f32 {
        match self.direction {
            CandleDirection::Bearish => self.first_low.max(self.last_low),
            CandleDirection::Bullish => self.first_high.min(self.last_high),
            CandleDirection::Doji => (self.first_low + self.first_high) / 2.0,
        }
    }

    /// Returns true if this range is complete (no longer being built).
    /// A range is complete when the next candle has a different direction.
    pub fn is_complete(&self) -> bool {
        // This is typically managed by the RangeBuilder
        // A range on its own doesn't know if it's complete
        self.candle_count > 0
    }
}

/// Builder for constructing ranges incrementally.
///
/// Use this to build ranges candle by candle in a streaming fashion.
#[derive(Debug)]
pub struct RangeBuilder {
    next_id: u64,
    doji_threshold: f32,
    current: Option<BuildingRange>,
}

#[derive(Debug)]
struct BuildingRange {
    id: RangeId,
    direction: CandleDirection,
    start_index: usize,
    end_index: usize,
    candle_count: usize,
    high: f32,
    low: f32,
    open: f32,
    close: f32,
    total_volume: f32,
    first_high: f32,
    first_low: f32,
    last_high: f32,
    last_low: f32,
}

impl RangeBuilder {
    /// Create a new range builder.
    pub fn new(doji_threshold: f32) -> Self {
        Self {
            next_id: 0,
            doji_threshold,
            current: None,
        }
    }

    /// Process a candle and return a completed range if the direction changed.
    ///
    /// Returns `Some(Range)` when a range is completed (direction change detected).
    /// Returns `None` if the range is still being built or if this is the first candle.
    pub fn process(&mut self, index: usize, candle: &Candle) -> Option<Range> {
        let direction = CandleDirection::from_candle_with_threshold(candle, self.doji_threshold);

        // Skip doji candles - they don't affect ranges
        if direction.is_doji() {
            // Update the current range's last values if we have one
            if let Some(ref mut current) = self.current {
                current.end_index = index;
                current.high = current.high.max(candle.high);
                current.low = current.low.min(candle.low);
                current.close = candle.close;
                current.total_volume += candle.volume;
                current.last_high = candle.high;
                current.last_low = candle.low;
                current.candle_count += 1;
            }
            return None;
        }

        match &mut self.current {
            None => {
                // Start a new range
                self.current = Some(BuildingRange {
                    id: RangeId::new(self.next_id),
                    direction,
                    start_index: index,
                    end_index: index,
                    candle_count: 1,
                    high: candle.high,
                    low: candle.low,
                    open: candle.open,
                    close: candle.close,
                    total_volume: candle.volume,
                    first_high: candle.high,
                    first_low: candle.low,
                    last_high: candle.high,
                    last_low: candle.low,
                });
                self.next_id += 1;
                None
            }
            Some(current) if current.direction == direction => {
                // Continue the current range
                current.end_index = index;
                current.candle_count += 1;
                current.high = current.high.max(candle.high);
                current.low = current.low.min(candle.low);
                current.close = candle.close;
                current.total_volume += candle.volume;
                current.last_high = candle.high;
                current.last_low = candle.low;
                None
            }
            Some(_) => {
                // Direction changed - complete current range and start new one
                let completed = self.current.take().map(|b| Range {
                    id: b.id,
                    direction: b.direction,
                    start_index: b.start_index,
                    end_index: b.end_index,
                    candle_count: b.candle_count,
                    high: b.high,
                    low: b.low,
                    open: b.open,
                    close: b.close,
                    total_volume: b.total_volume,
                    first_high: b.first_high,
                    first_low: b.first_low,
                    last_high: b.last_high,
                    last_low: b.last_low,
                });

                // Start new range
                self.current = Some(BuildingRange {
                    id: RangeId::new(self.next_id),
                    direction,
                    start_index: index,
                    end_index: index,
                    candle_count: 1,
                    high: candle.high,
                    low: candle.low,
                    open: candle.open,
                    close: candle.close,
                    total_volume: candle.volume,
                    first_high: candle.high,
                    first_low: candle.low,
                    last_high: candle.high,
                    last_low: candle.low,
                });
                self.next_id += 1;

                completed
            }
        }
    }

    /// Finalize and return the current range if one is being built.
    ///
    /// Call this at the end of processing to get the final incomplete range.
    pub fn finalize(&mut self) -> Option<Range> {
        self.current.take().map(|b| Range {
            id: b.id,
            direction: b.direction,
            start_index: b.start_index,
            end_index: b.end_index,
            candle_count: b.candle_count,
            high: b.high,
            low: b.low,
            open: b.open,
            close: b.close,
            total_volume: b.total_volume,
            first_high: b.first_high,
            first_low: b.first_low,
            last_high: b.last_high,
            last_low: b.last_low,
        })
    }

    /// Check if a range is currently being built.
    pub fn is_building(&self) -> bool {
        self.current.is_some()
    }

    /// Get the direction of the currently building range, if any.
    pub fn current_direction(&self) -> Option<CandleDirection> {
        self.current.as_ref().map(|r| r.direction)
    }

    /// Get the candle count of the currently building range, if any.
    pub fn current_candle_count(&self) -> Option<usize> {
        self.current.as_ref().map(|r| r.candle_count)
    }

    /// Reset the builder, discarding any current range.
    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Process a slice of candles and return all ranges.
///
/// This is a convenience function for batch processing.
pub fn detect_ranges(candles: &[Candle], doji_threshold: f32) -> Vec<Range> {
    let mut builder = RangeBuilder::new(doji_threshold);
    let mut ranges = Vec::new();

    for (i, candle) in candles.iter().enumerate() {
        if let Some(range) = builder.process(i, candle) {
            ranges.push(range);
        }
    }

    // Don't forget the final range
    if let Some(range) = builder.finalize() {
        ranges.push(range);
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 100.0)
    }

    #[test]
    fn test_single_bullish_range() {
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0),
            make_candle(105.0, 115.0, 100.0, 110.0),
            make_candle(110.0, 120.0, 105.0, 115.0),
        ];

        let ranges = detect_ranges(&candles, 0.0);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].direction, CandleDirection::Bullish);
        assert_eq!(ranges[0].candle_count, 3);
    }

    #[test]
    fn test_bullish_then_bearish() {
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0),   // Bullish
            make_candle(105.0, 115.0, 100.0, 110.0),  // Bullish
            make_candle(110.0, 115.0, 100.0, 105.0),  // Bearish
            make_candle(105.0, 110.0, 95.0, 100.0),   // Bearish
        ];

        let ranges = detect_ranges(&candles, 0.0);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].direction, CandleDirection::Bullish);
        assert_eq!(ranges[0].candle_count, 2);
        assert_eq!(ranges[1].direction, CandleDirection::Bearish);
        assert_eq!(ranges[1].candle_count, 2);
    }

    #[test]
    fn test_hold_level_bearish() {
        let candles = vec![
            make_candle(110.0, 115.0, 100.0, 105.0),  // Bearish, low=100
            make_candle(105.0, 110.0, 95.0, 100.0),   // Bearish, low=95
        ];

        let ranges = detect_ranges(&candles, 0.0);
        assert_eq!(ranges.len(), 1);

        // Hold level = min(first_low, last_low) = min(100, 95) = 95
        assert_eq!(ranges[0].hold_level_price(), 95.0);
        // Greedy hold = max(first_low, last_low) = max(100, 95) = 100
        assert_eq!(ranges[0].greedy_hold_level_price(), 100.0);
    }

    #[test]
    fn test_hold_level_bullish() {
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0),   // Bullish, high=110
            make_candle(105.0, 120.0, 100.0, 115.0),  // Bullish, high=120
        ];

        let ranges = detect_ranges(&candles, 0.0);
        assert_eq!(ranges.len(), 1);

        // Hold level = max(first_high, last_high) = max(110, 120) = 120
        assert_eq!(ranges[0].hold_level_price(), 120.0);
        // Greedy hold = min(first_high, last_high) = min(110, 120) = 110
        assert_eq!(ranges[0].greedy_hold_level_price(), 110.0);
    }
}
