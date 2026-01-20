//! Range detection and representation.
//!
//! A Range is a consecutive sequence of candles with the same dominant direction.

use charter_core::Candle;
use serde::{Deserialize, Serialize};

use super::candle::{CandleDirection, CandleMetadata};

/// Unique identifier for a Range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RangeId(pub u64);

impl RangeId {
    /// Create a new RangeId from raw components.
    #[inline]
    pub fn new(timeframe_idx: u8, sequence: u32) -> Self {
        // Pack timeframe in upper 8 bits, sequence in lower 32 bits
        let id = ((timeframe_idx as u64) << 32) | (sequence as u64);
        Self(id)
    }

    /// Extract the timeframe index from the ID.
    #[inline]
    pub fn timeframe_idx(self) -> u8 {
        (self.0 >> 32) as u8
    }
}

/// A consecutive sequence of candles with the same dominant direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    /// Unique identifier
    pub id: RangeId,
    /// Direction of this range
    pub direction: CandleDirection,

    /// Start index in the candle array
    pub start_index: usize,
    /// End index in the candle array (inclusive)
    pub end_index: usize,
    /// Number of candles in this range
    pub candle_count: usize,

    /// Maximum high price in the range
    pub high: f32,
    /// Minimum low price in the range
    pub low: f32,
    /// First candle's open price
    pub open: f32,
    /// Last candle's close price
    pub close: f32,

    /// Total volume accumulated
    pub total_volume: f32,

    /// First candle's (high, low) for level calculation
    pub first_candle_hl: (f32, f32),
    /// Last candle's (high, low) for level calculation
    pub last_candle_hl: (f32, f32),
}

impl Range {
    /// Calculate the Hold level price for this range.
    ///
    /// - Bullish (Support): min(first_low, last_low)
    /// - Bearish (Resistance): max(first_high, last_high)
    #[inline]
    pub fn hold_level_price(&self) -> f32 {
        match self.direction {
            CandleDirection::Bullish => self.first_candle_hl.1.min(self.last_candle_hl.1),
            CandleDirection::Bearish => self.first_candle_hl.0.max(self.last_candle_hl.0),
            CandleDirection::Doji => self.low, // Fallback
        }
    }

    /// Calculate the Greedy Hold level price for this range.
    ///
    /// - Bullish (Support): max(first_low, last_low)
    /// - Bearish (Resistance): min(first_high, last_high)
    #[inline]
    pub fn greedy_hold_level_price(&self) -> f32 {
        match self.direction {
            CandleDirection::Bullish => self.first_candle_hl.1.max(self.last_candle_hl.1),
            CandleDirection::Bearish => self.first_candle_hl.0.min(self.last_candle_hl.0),
            CandleDirection::Doji => self.high, // Fallback
        }
    }

    /// Returns the candle index that defines the Hold level.
    ///
    /// This is the candle whose wick defines the level price.
    #[inline]
    pub fn hold_level_candle_index(&self) -> usize {
        match self.direction {
            CandleDirection::Bullish => {
                if self.first_candle_hl.1 <= self.last_candle_hl.1 {
                    self.start_index
                } else {
                    self.end_index
                }
            }
            CandleDirection::Bearish => {
                if self.first_candle_hl.0 >= self.last_candle_hl.0 {
                    self.start_index
                } else {
                    self.end_index
                }
            }
            CandleDirection::Doji => self.start_index,
        }
    }

    /// Returns the candle index that defines the Greedy Hold level.
    #[inline]
    pub fn greedy_hold_level_candle_index(&self) -> usize {
        match self.direction {
            CandleDirection::Bullish => {
                if self.first_candle_hl.1 >= self.last_candle_hl.1 {
                    self.start_index
                } else {
                    self.end_index
                }
            }
            CandleDirection::Bearish => {
                if self.first_candle_hl.0 <= self.last_candle_hl.0 {
                    self.start_index
                } else {
                    self.end_index
                }
            }
            CandleDirection::Doji => self.end_index,
        }
    }

    /// Check if this range meets the minimum candle count requirement.
    #[inline]
    pub fn is_valid(&self, min_candles: usize) -> bool {
        self.candle_count >= min_candles && self.direction.is_directional()
    }

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
}

/// Builder for detecting ranges incrementally.
///
/// Processes candles one at a time and emits completed ranges when
/// the direction changes.
pub struct RangeBuilder {
    /// Current direction being tracked
    current_direction: Option<CandleDirection>,
    /// Start index of current range
    start_index: usize,
    /// Current end index
    end_index: usize,
    /// Candle count in current range
    candle_count: usize,

    /// Price tracking
    high: f32,
    low: f32,
    open: f32,
    close: f32,
    total_volume: f32,

    /// First and last candle data
    first_candle_hl: (f32, f32),
    last_candle_hl: (f32, f32),

    /// ID generation
    timeframe_idx: u8,
    next_sequence: u32,

    /// Configuration
    doji_threshold: f32,
}

impl RangeBuilder {
    /// Create a new RangeBuilder for a specific timeframe.
    pub fn new(timeframe_idx: u8, doji_threshold: f32) -> Self {
        Self {
            current_direction: None,
            start_index: 0,
            end_index: 0,
            candle_count: 0,
            high: f32::NEG_INFINITY,
            low: f32::INFINITY,
            open: 0.0,
            close: 0.0,
            total_volume: 0.0,
            first_candle_hl: (0.0, 0.0),
            last_candle_hl: (0.0, 0.0),
            timeframe_idx,
            next_sequence: 0,
            doji_threshold,
        }
    }

    /// Process a candle and potentially return a completed range.
    ///
    /// Returns `Some(Range)` when the direction changes, indicating the
    /// previous range is complete. Returns `None` if the range continues.
    pub fn process(&mut self, index: usize, candle: &Candle) -> Option<Range> {
        let meta = CandleMetadata::from_candle(candle, self.doji_threshold);
        let direction = meta.direction;

        // Doji candles don't break a range
        if direction == CandleDirection::Doji {
            if self.current_direction.is_some() {
                // Update range stats but don't change direction
                self.update_stats(index, candle);
            }
            return None;
        }

        match self.current_direction {
            None => {
                // Start a new range
                self.start_new_range(index, candle, direction);
                None
            }
            Some(current_dir) if current_dir == direction => {
                // Continue current range
                self.update_stats(index, candle);
                None
            }
            Some(_) => {
                // Direction changed - complete current range and start new
                let completed = self.build_range();
                self.start_new_range(index, candle, direction);
                Some(completed)
            }
        }
    }

    /// Process a candle in reverse order (for reverse pass).
    ///
    /// Similar to `process` but handles indexing correctly for reverse iteration.
    pub fn process_reverse(&mut self, index: usize, candle: &Candle) -> Option<Range> {
        let meta = CandleMetadata::from_candle(candle, self.doji_threshold);
        let direction = meta.direction;

        // Doji candles don't break a range
        if direction == CandleDirection::Doji {
            if self.current_direction.is_some() {
                self.update_stats_reverse(index, candle);
            }
            return None;
        }

        match self.current_direction {
            None => {
                self.start_new_range_reverse(index, candle, direction);
                None
            }
            Some(current_dir) if current_dir == direction => {
                self.update_stats_reverse(index, candle);
                None
            }
            Some(_) => {
                let completed = self.build_range();
                self.start_new_range_reverse(index, candle, direction);
                Some(completed)
            }
        }
    }

    /// Finalize and return the current range (even if incomplete).
    pub fn finalize(&mut self) -> Option<Range> {
        if self.current_direction.is_some() && self.candle_count > 0 {
            let range = self.build_range();
            self.current_direction = None;
            self.candle_count = 0;
            Some(range)
        } else {
            None
        }
    }

    /// Get the current direction being built.
    pub fn current_direction(&self) -> Option<CandleDirection> {
        self.current_direction
    }

    /// Get the current candle count.
    pub fn current_candle_count(&self) -> usize {
        self.candle_count
    }

    /// Reset the builder state.
    pub fn reset(&mut self) {
        self.current_direction = None;
        self.candle_count = 0;
        self.high = f32::NEG_INFINITY;
        self.low = f32::INFINITY;
    }

    // Private helpers

    fn start_new_range(&mut self, index: usize, candle: &Candle, direction: CandleDirection) {
        self.current_direction = Some(direction);
        self.start_index = index;
        self.end_index = index;
        self.candle_count = 1;
        self.high = candle.high;
        self.low = candle.low;
        self.open = candle.open;
        self.close = candle.close;
        self.total_volume = candle.volume;
        self.first_candle_hl = (candle.high, candle.low);
        self.last_candle_hl = (candle.high, candle.low);
    }

    fn start_new_range_reverse(&mut self, index: usize, candle: &Candle, direction: CandleDirection) {
        self.current_direction = Some(direction);
        self.start_index = index; // In reverse, this is actually the "end" chronologically
        self.end_index = index;
        self.candle_count = 1;
        self.high = candle.high;
        self.low = candle.low;
        self.open = candle.open;
        self.close = candle.close;
        self.total_volume = candle.volume;
        // In reverse, first encountered is chronologically last
        self.first_candle_hl = (candle.high, candle.low);
        self.last_candle_hl = (candle.high, candle.low);
    }

    fn update_stats(&mut self, index: usize, candle: &Candle) {
        self.end_index = index;
        self.candle_count += 1;
        self.high = self.high.max(candle.high);
        self.low = self.low.min(candle.low);
        self.close = candle.close;
        self.total_volume += candle.volume;
        self.last_candle_hl = (candle.high, candle.low);
    }

    fn update_stats_reverse(&mut self, index: usize, candle: &Candle) {
        // In reverse iteration, start_index becomes the older (smaller) index
        self.start_index = index;
        self.candle_count += 1;
        self.high = self.high.max(candle.high);
        self.low = self.low.min(candle.low);
        self.open = candle.open; // Earlier candle's open
        self.total_volume += candle.volume;
        // In reverse, this is chronologically the first candle
        self.last_candle_hl = (candle.high, candle.low);
    }

    fn build_range(&mut self) -> Range {
        let id = RangeId::new(self.timeframe_idx, self.next_sequence);
        self.next_sequence += 1;

        // Ensure start_index < end_index
        let (start, end) = if self.start_index <= self.end_index {
            (self.start_index, self.end_index)
        } else {
            (self.end_index, self.start_index)
        };

        // Swap first/last if we were processing in reverse
        let (first_hl, last_hl) = if self.start_index <= self.end_index {
            (self.first_candle_hl, self.last_candle_hl)
        } else {
            (self.last_candle_hl, self.first_candle_hl)
        };

        Range {
            id,
            direction: self.current_direction.unwrap_or(CandleDirection::Doji),
            start_index: start,
            end_index: end,
            candle_count: self.candle_count,
            high: self.high,
            low: self.low,
            open: self.open,
            close: self.close,
            total_volume: self.total_volume,
            first_candle_hl: first_hl,
            last_candle_hl: last_hl,
        }
    }
}

/// Detect all ranges in a slice of candles.
pub fn detect_ranges(
    candles: &[Candle],
    timeframe_idx: u8,
    doji_threshold: f32,
    min_candles: usize,
) -> Vec<Range> {
    let mut builder = RangeBuilder::new(timeframe_idx, doji_threshold);
    let mut ranges = Vec::new();

    for (index, candle) in candles.iter().enumerate() {
        if let Some(range) = builder.process(index, candle) {
            if range.is_valid(min_candles) {
                ranges.push(range);
            }
        }
    }

    // Don't forget the final range
    if let Some(range) = builder.finalize() {
        if range.is_valid(min_candles) {
            ranges.push(range);
        }
    }

    ranges
}

/// Detect ranges in reverse order (newest to oldest).
///
/// Returns ranges in chronological order (oldest first) despite reverse processing.
pub fn detect_ranges_reverse(
    candles: &[Candle],
    timeframe_idx: u8,
    doji_threshold: f32,
    min_candles: usize,
) -> Vec<Range> {
    if candles.is_empty() {
        return Vec::new();
    }

    let mut builder = RangeBuilder::new(timeframe_idx, doji_threshold);
    let mut ranges = Vec::new();

    // Process in reverse (newest to oldest)
    for (i, candle) in candles.iter().enumerate().rev() {
        if let Some(range) = builder.process_reverse(i, candle) {
            if range.is_valid(min_candles) {
                ranges.push(range);
            }
        }
    }

    if let Some(range) = builder.finalize() {
        if range.is_valid(min_candles) {
            ranges.push(range);
        }
    }

    // Reverse to get chronological order
    ranges.reverse();
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 1.0)
    }

    fn bullish_candle(base: f32, size: f32) -> Candle {
        make_candle(base, base + size * 1.2, base - size * 0.1, base + size)
    }

    fn bearish_candle(base: f32, size: f32) -> Candle {
        make_candle(base + size, base + size * 1.1, base - size * 0.1, base)
    }

    #[test]
    fn test_range_builder_basic() {
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bullish_candle(110.0, 5.0),
            bearish_candle(115.0, 5.0), // Direction change
        ];

        let mut builder = RangeBuilder::new(0, 0.1);
        let mut ranges = Vec::new();

        for (i, c) in candles.iter().enumerate() {
            if let Some(r) = builder.process(i, c) {
                ranges.push(r);
            }
        }
        if let Some(r) = builder.finalize() {
            ranges.push(r);
        }

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].direction, CandleDirection::Bullish);
        assert_eq!(ranges[0].candle_count, 3);
        assert_eq!(ranges[1].direction, CandleDirection::Bearish);
        assert_eq!(ranges[1].candle_count, 1);
    }

    #[test]
    fn test_hold_level_prices() {
        // Bullish range: first_low=95, last_low=105
        // Hold = min(95, 105) = 95
        // Greedy = max(95, 105) = 105
        let range = Range {
            id: RangeId::new(0, 0),
            direction: CandleDirection::Bullish,
            start_index: 0,
            end_index: 2,
            candle_count: 3,
            high: 120.0,
            low: 95.0,
            open: 100.0,
            close: 115.0,
            total_volume: 3.0,
            first_candle_hl: (106.0, 95.0),
            last_candle_hl: (120.0, 105.0),
        };

        assert_eq!(range.hold_level_price(), 95.0);
        assert_eq!(range.greedy_hold_level_price(), 105.0);
    }

    #[test]
    fn test_bearish_hold_level_prices() {
        // Bearish range: first_high=120, last_high=110
        // Hold = max(120, 110) = 120
        // Greedy = min(120, 110) = 110
        let range = Range {
            id: RangeId::new(0, 0),
            direction: CandleDirection::Bearish,
            start_index: 0,
            end_index: 2,
            candle_count: 3,
            high: 120.0,
            low: 95.0,
            open: 115.0,
            close: 100.0,
            total_volume: 3.0,
            first_candle_hl: (120.0, 100.0),
            last_candle_hl: (110.0, 95.0),
        };

        assert_eq!(range.hold_level_price(), 120.0);
        assert_eq!(range.greedy_hold_level_price(), 110.0);
    }

    #[test]
    fn test_detect_ranges_reverse_matches_forward() {
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bullish_candle(110.0, 5.0),
            bearish_candle(115.0, 5.0),
            bearish_candle(110.0, 5.0),
            bullish_candle(105.0, 5.0),
        ];

        let forward = detect_ranges(&candles, 0, 0.1, 1);
        let reverse = detect_ranges_reverse(&candles, 0, 0.1, 1);

        assert_eq!(forward.len(), reverse.len());
        for (f, r) in forward.iter().zip(reverse.iter()) {
            assert_eq!(f.direction, r.direction);
            assert_eq!(f.start_index, r.start_index);
            assert_eq!(f.end_index, r.end_index);
            assert_eq!(f.candle_count, r.candle_count);
        }
    }

    #[test]
    fn test_min_candles_filter() {
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bearish_candle(110.0, 5.0), // Only 1 bearish candle
            bullish_candle(105.0, 5.0),
            bullish_candle(110.0, 5.0),
            bullish_candle(115.0, 5.0),
        ];

        // With min_candles=2, the single bearish should be filtered
        let ranges = detect_ranges(&candles, 0, 0.1, 2);

        // Should only have bullish ranges that meet the minimum
        for r in &ranges {
            assert!(r.candle_count >= 2);
        }
    }
}
