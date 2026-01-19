//! Trend types - trendlines derived from consecutive ranges with interaction tracking.

use rayon::prelude::*;

use super::direction::CandleDirection;
use super::range::Range;
use charter_core::Candle;

/// Unique identifier for a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrendId(pub u64);

impl TrendId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// The current state of a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendState {
    /// Trend is active and being tracked.
    Active,
    /// Trend has been hit (wick touched but body held).
    Hit,
    /// Trend has been broken (body closed through).
    Broken,
}

/// A point on a trendline.
#[derive(Debug, Clone, Copy)]
pub struct TrendPoint {
    /// Candle index of this point.
    pub candle_index: usize,
    /// Price at this point.
    pub price: f32,
}

/// Record of a trend being hit.
#[derive(Debug, Clone, Copy)]
pub struct TrendHit {
    /// Index of the candle that hit the trend.
    pub candle_index: usize,
    /// Price where the wick touched the trend.
    pub touch_price: f32,
    /// The trendline price at this candle index.
    pub trend_price_at_candle: f32,
    /// How far past the trend the wick went.
    pub penetration: f32,
}

/// Record of a trend being broken.
#[derive(Debug, Clone, Copy)]
pub struct TrendBreak {
    /// Index of the candle that broke the trend.
    pub candle_index: usize,
    /// Close price of the breaking candle.
    pub close_price: f32,
}

/// A trendline connecting two ranges.
///
/// For bearish trends: connects the lowest wicks of two bearish ranges.
/// For bullish trends: connects the highest wicks of two bullish ranges.
#[derive(Debug, Clone)]
pub struct Trend {
    /// Unique identifier.
    pub id: TrendId,
    /// Direction of the trend (bearish = down, bullish = up).
    pub direction: CandleDirection,
    /// Starting point of the trendline.
    pub start: TrendPoint,
    /// Ending point of the trendline (used to calculate slope).
    pub end: TrendPoint,
    /// Index of the candle where this trend was created.
    pub created_at_index: usize,
    /// Current state of the trend.
    pub state: TrendState,
    /// History of hits on this trend.
    pub hits: Vec<TrendHit>,
    /// The break event, if the trend was broken.
    pub break_event: Option<TrendBreak>,
    /// Tolerance for trend interactions (in price units).
    pub tolerance: f32,
}

impl Trend {
    /// Create a new trend from two consecutive ranges.
    ///
    /// For bearish ranges: uses the lowest wicks.
    /// For bullish ranges: uses the highest wicks.
    pub fn from_ranges(
        id: TrendId,
        first_range: &Range,
        second_range: &Range,
        created_at_index: usize,
        tolerance: f32,
    ) -> Option<Self> {
        // Both ranges must have the same direction
        if first_range.direction != second_range.direction {
            return None;
        }

        // Skip doji ranges
        if first_range.direction == CandleDirection::Doji {
            return None;
        }

        let (start, end) = match first_range.direction {
            CandleDirection::Bearish => {
                // Bearish trend: connect lowest wicks
                let first_low = first_range.first_low.min(first_range.last_low);
                let first_idx = if first_range.first_low <= first_range.last_low {
                    first_range.start_index
                } else {
                    first_range.end_index
                };

                let second_low = second_range.first_low.min(second_range.last_low);
                let second_idx = if second_range.first_low <= second_range.last_low {
                    second_range.start_index
                } else {
                    second_range.end_index
                };

                (
                    TrendPoint { candle_index: first_idx, price: first_low },
                    TrendPoint { candle_index: second_idx, price: second_low },
                )
            }
            CandleDirection::Bullish => {
                // Bullish trend: connect highest wicks
                let first_high = first_range.first_high.max(first_range.last_high);
                let first_idx = if first_range.first_high >= first_range.last_high {
                    first_range.start_index
                } else {
                    first_range.end_index
                };

                let second_high = second_range.first_high.max(second_range.last_high);
                let second_idx = if second_range.first_high >= second_range.last_high {
                    second_range.start_index
                } else {
                    second_range.end_index
                };

                (
                    TrendPoint { candle_index: first_idx, price: first_high },
                    TrendPoint { candle_index: second_idx, price: second_high },
                )
            }
            CandleDirection::Doji => return None,
        };

        Some(Self {
            id,
            direction: first_range.direction,
            start,
            end,
            created_at_index,
            state: TrendState::Active,
            hits: Vec::new(),
            break_event: None,
            tolerance,
        })
    }

    /// Calculate the trendline price at a given candle index.
    ///
    /// Uses linear interpolation/extrapolation from the two defining points.
    #[inline]
    pub fn price_at(&self, candle_index: usize) -> f32 {
        let dx = self.end.candle_index as f32 - self.start.candle_index as f32;
        if dx.abs() < f32::EPSILON {
            return self.start.price;
        }

        let slope = (self.end.price - self.start.price) / dx;
        let x = candle_index as f32 - self.start.candle_index as f32;
        self.start.price + slope * x
    }

    /// Number of times this trend has been hit.
    #[inline]
    pub fn hit_count(&self) -> usize {
        self.hits.len()
    }

    /// Returns true if this trend is still active (not broken).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state != TrendState::Broken
    }

    /// Returns true if this is a bearish trend (connecting lows of bearish ranges).
    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.direction == CandleDirection::Bearish
    }

    /// Returns true if this is a bullish trend (connecting highs of bullish ranges).
    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.direction == CandleDirection::Bullish
    }

    /// Check if a candle interacts with this trend and update state.
    ///
    /// Returns the type of interaction that occurred.
    pub fn check_interaction(&mut self, candle_index: usize, candle: &Candle) -> TrendInteraction {
        if self.state == TrendState::Broken {
            return TrendInteraction::None;
        }

        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);
        let trend_price = self.price_at(candle_index);

        match self.direction {
            CandleDirection::Bearish => {
                // Bearish trend (at lows) - price approaches from ABOVE
                // Hit when wick goes below, body stays above
                // Broken when body closes below
                self.check_bearish_trend_interaction(candle_index, candle, body_top, body_bottom, trend_price)
            }
            CandleDirection::Bullish => {
                // Bullish trend (at highs) - price approaches from BELOW
                // Hit when wick goes above, body stays below
                // Broken when body closes above
                self.check_bullish_trend_interaction(candle_index, candle, body_top, body_bottom, trend_price)
            }
            CandleDirection::Doji => TrendInteraction::None,
        }
    }

    /// Check bearish trend interaction.
    ///
    /// Bearish trends connect the lows of bearish ranges.
    /// Hit when the wick touches or goes BELOW the trendline.
    /// Broken when the full body closes below the trendline.
    fn check_bearish_trend_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
        trend_price: f32,
    ) -> TrendInteraction {
        let trend_upper = trend_price + self.tolerance;

        // Check if the body closed below the trend (broken)
        if body_top < trend_price - self.tolerance {
            // Full body is below the trend - BROKEN
            self.state = TrendState::Broken;
            self.break_event = Some(TrendBreak {
                candle_index,
                close_price: candle.close,
            });
            return TrendInteraction::Broken;
        }

        // Check if the wick touched or went below the trend but body stayed above
        if candle.low <= trend_upper && body_bottom >= trend_price - self.tolerance {
            // Wick touched or went below, but body closed above - HIT
            let penetration = (trend_price - candle.low).max(0.0);
            let hit = TrendHit {
                candle_index,
                touch_price: candle.low,
                trend_price_at_candle: trend_price,
                penetration,
            };
            self.hits.push(hit);
            self.state = TrendState::Hit;
            return TrendInteraction::Hit(hit);
        }

        TrendInteraction::None
    }

    /// Check bullish trend interaction.
    ///
    /// Bullish trends connect the highs of bullish ranges.
    /// Hit when the wick touches or goes ABOVE the trendline.
    /// Broken when the full body closes above the trendline.
    fn check_bullish_trend_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
        trend_price: f32,
    ) -> TrendInteraction {
        let trend_lower = trend_price - self.tolerance;

        // Check if the body closed above the trend (broken)
        if body_bottom > trend_price + self.tolerance {
            // Full body is above the trend - BROKEN
            self.state = TrendState::Broken;
            self.break_event = Some(TrendBreak {
                candle_index,
                close_price: candle.close,
            });
            return TrendInteraction::Broken;
        }

        // Check if the wick touched or went above the trend but body stayed below
        if candle.high >= trend_lower && body_top <= trend_price + self.tolerance {
            // Wick touched or went above, but body closed below - HIT
            let penetration = (candle.high - trend_price).max(0.0);
            let hit = TrendHit {
                candle_index,
                touch_price: candle.high,
                trend_price_at_candle: trend_price,
                penetration,
            };
            self.hits.push(hit);
            self.state = TrendState::Hit;
            return TrendInteraction::Hit(hit);
        }

        TrendInteraction::None
    }
}

/// The result of checking a trend interaction.
#[derive(Debug, Clone, Copy)]
pub enum TrendInteraction {
    /// No interaction occurred.
    None,
    /// The trend was hit (wick touched, body held).
    Hit(TrendHit),
    /// The trend was broken (body closed through).
    Broken,
}

/// Events that can occur on trends.
#[derive(Debug, Clone, Copy)]
pub enum TrendEvent {
    /// A new trend was created.
    Created { trend_id: TrendId },
    /// A trend was hit.
    Hit { trend_id: TrendId, hit: TrendHit },
    /// A trend was broken.
    Broken { trend_id: TrendId, break_event: TrendBreak },
}

/// Tracker for multiple trends.
#[derive(Debug)]
pub struct TrendTracker {
    next_id: u64,
    /// All trends being tracked.
    pub trends: Vec<Trend>,
    /// Default tolerance for trend interactions.
    pub default_tolerance: f32,
    /// Last bearish range seen (to connect bearish trends even if not consecutive).
    last_bearish_range: Option<Range>,
    /// Last bullish range seen (to connect bullish trends even if not consecutive).
    last_bullish_range: Option<Range>,
}

impl TrendTracker {
    /// Create a new trend tracker.
    pub fn new(default_tolerance: f32) -> Self {
        Self {
            next_id: 0,
            trends: Vec::new(),
            default_tolerance,
            last_bearish_range: None,
            last_bullish_range: None,
        }
    }

    /// Process a completed range, potentially creating a new trend.
    ///
    /// A trend is created when we see a new range of the same direction as a previous one,
    /// even if they're not immediately consecutive (e.g., bearish → bullish → bearish creates a bearish trend).
    pub fn process_range(&mut self, range: &Range, created_at_index: usize) -> Option<TrendEvent> {
        // Skip doji ranges
        if range.direction == CandleDirection::Doji {
            return None;
        }

        let result = match range.direction {
            CandleDirection::Bearish => {
                let event = if let Some(ref last) = self.last_bearish_range {
                    // Create a trend from the last bearish range to this one
                    if let Some(trend) = Trend::from_ranges(
                        TrendId::new(self.next_id),
                        last,
                        range,
                        created_at_index,
                        self.default_tolerance,
                    ) {
                        let trend_id = trend.id;
                        self.next_id += 1;
                        self.trends.push(trend);
                        Some(TrendEvent::Created { trend_id })
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Remember this bearish range for next time
                self.last_bearish_range = Some(range.clone());
                event
            }
            CandleDirection::Bullish => {
                let event = if let Some(ref last) = self.last_bullish_range {
                    // Create a trend from the last bullish range to this one
                    if let Some(trend) = Trend::from_ranges(
                        TrendId::new(self.next_id),
                        last,
                        range,
                        created_at_index,
                        self.default_tolerance,
                    ) {
                        let trend_id = trend.id;
                        self.next_id += 1;
                        self.trends.push(trend);
                        Some(TrendEvent::Created { trend_id })
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Remember this bullish range for next time
                self.last_bullish_range = Some(range.clone());
                event
            }
            CandleDirection::Doji => None,
        };

        result
    }

    /// Check all active trends for interactions with a candle.
    ///
    /// Returns a list of events that occurred.
    /// Uses parallel processing for improved performance with many trends.
    pub fn check_interactions(&mut self, candle_index: usize, candle: &Candle) -> Vec<TrendEvent> {
        // Parallel phase: check all trends concurrently
        // Each trend check is independent, making this embarrassingly parallel
        self.trends
            .par_iter_mut()
            .filter_map(|trend| {
                // Skip if trend was created on or after this candle (not yet active)
                if trend.created_at_index >= candle_index {
                    return None;
                }
                // Skip already broken trends
                if trend.state == TrendState::Broken {
                    return None;
                }

                match trend.check_interaction(candle_index, candle) {
                    TrendInteraction::Hit(hit) => Some(TrendEvent::Hit {
                        trend_id: trend.id,
                        hit,
                    }),
                    TrendInteraction::Broken => Some(TrendEvent::Broken {
                        trend_id: trend.id,
                        break_event: trend.break_event.unwrap(),
                    }),
                    TrendInteraction::None => None,
                }
            })
            .collect()
    }

    /// Get all active (non-broken) trends.
    pub fn active_trends(&self) -> impl Iterator<Item = &Trend> {
        self.trends.iter().filter(|t| t.is_active())
    }

    /// Get all broken trends.
    pub fn broken_trends(&self) -> impl Iterator<Item = &Trend> {
        self.trends.iter().filter(|t| !t.is_active())
    }

    /// Remove all broken trends from tracking.
    pub fn prune_broken(&mut self) {
        self.trends.retain(|t| t.is_active());
    }

    /// Clear all trends.
    pub fn clear(&mut self) {
        self.trends.clear();
        self.last_bearish_range = None;
        self.last_bullish_range = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 1000.0)
    }

    fn make_bearish_range(id: u64, start: usize, end: usize, first_low: f32, last_low: f32) -> Range {
        Range {
            id: super::super::range::RangeId::new(id),
            direction: CandleDirection::Bearish,
            start_index: start,
            end_index: end,
            candle_count: end - start + 1,
            high: 125.0,
            low: first_low.min(last_low),
            open: 120.0,
            close: 110.0,
            total_volume: 1000.0,
            first_high: 125.0,
            first_low,
            last_high: 115.0,
            last_low,
        }
    }

    #[allow(dead_code)]
    fn make_bullish_range(id: u64, start: usize, end: usize, first_high: f32, last_high: f32) -> Range {
        Range {
            id: super::super::range::RangeId::new(id),
            direction: CandleDirection::Bullish,
            start_index: start,
            end_index: end,
            candle_count: end - start + 1,
            high: first_high.max(last_high),
            low: 95.0,
            open: 100.0,
            close: 110.0,
            total_volume: 1000.0,
            first_high,
            first_low: 95.0,
            last_high,
            last_low: 105.0,
        }
    }

    #[test]
    fn test_bearish_trend_creation() {
        // Two bearish ranges with lows at 100 (index 2) and 95 (index 5)
        let range1 = make_bearish_range(1, 0, 2, 102.0, 100.0);
        let range2 = make_bearish_range(2, 3, 5, 97.0, 95.0);

        let trend = Trend::from_ranges(TrendId::new(1), &range1, &range2, 5, 0.5).unwrap();

        assert!(trend.is_bearish());
        assert_eq!(trend.start.candle_index, 2); // last_low is lower
        assert_eq!(trend.start.price, 100.0);
        assert_eq!(trend.end.candle_index, 5); // last_low is lower
        assert_eq!(trend.end.price, 95.0);
    }

    #[test]
    fn test_bearish_trend_price_at() {
        let range1 = make_bearish_range(1, 0, 2, 102.0, 100.0);
        let range2 = make_bearish_range(2, 3, 5, 97.0, 95.0);

        let trend = Trend::from_ranges(TrendId::new(1), &range1, &range2, 5, 0.5).unwrap();

        // Slope: (95 - 100) / (5 - 2) = -5/3 ≈ -1.667
        // At index 2: 100
        // At index 5: 95
        // At index 8: 100 + (-5/3) * 6 = 100 - 10 = 90

        assert!((trend.price_at(2) - 100.0).abs() < 0.01);
        assert!((trend.price_at(5) - 95.0).abs() < 0.01);
        assert!((trend.price_at(8) - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_bearish_trend_hit() {
        let range1 = make_bearish_range(1, 0, 2, 102.0, 100.0);
        let range2 = make_bearish_range(2, 3, 5, 97.0, 95.0);

        let mut trend = Trend::from_ranges(TrendId::new(1), &range1, &range2, 5, 0.5).unwrap();

        // At index 8, trend price is 90
        // Candle with wick going below but body staying above
        let candle = make_candle(92.0, 95.0, 89.5, 93.0);
        let interaction = trend.check_interaction(8, &candle);

        match interaction {
            TrendInteraction::Hit(hit) => {
                assert_eq!(hit.candle_index, 8);
                assert_eq!(hit.touch_price, 89.5);
                assert!((hit.trend_price_at_candle - 90.0).abs() < 0.01);
            }
            _ => panic!("Expected Hit"),
        }

        assert_eq!(trend.state, TrendState::Hit);
        assert_eq!(trend.hit_count(), 1);
    }

    #[test]
    fn test_bearish_trend_broken() {
        let range1 = make_bearish_range(1, 0, 2, 102.0, 100.0);
        let range2 = make_bearish_range(2, 3, 5, 97.0, 95.0);

        let mut trend = Trend::from_ranges(TrendId::new(1), &range1, &range2, 5, 0.5).unwrap();

        // At index 8, trend price is 90
        // Candle with full body below the trend
        let candle = make_candle(89.0, 89.5, 85.0, 87.0);
        let interaction = trend.check_interaction(8, &candle);

        assert!(matches!(interaction, TrendInteraction::Broken));
        assert_eq!(trend.state, TrendState::Broken);
    }

    #[test]
    fn test_trend_tracker() {
        let mut tracker = TrendTracker::new(0.5);

        // First bearish range - no trend yet
        let range1 = make_bearish_range(1, 0, 2, 102.0, 100.0);
        let event1 = tracker.process_range(&range1, 2);
        assert!(event1.is_none());

        // Second bearish range - trend created!
        let range2 = make_bearish_range(2, 3, 5, 97.0, 95.0);
        let event2 = tracker.process_range(&range2, 5);
        assert!(matches!(event2, Some(TrendEvent::Created { .. })));

        assert_eq!(tracker.trends.len(), 1);
    }
}
