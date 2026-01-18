//! Level types - price levels derived from ranges with interaction tracking.

use std::collections::HashMap;

use super::direction::CandleDirection;
use super::range::{Range, RangeId};
use charter_core::Candle;

/// A bucket key for price-based spatial indexing.
///
/// Used to partition levels into buckets based on their price for O(1) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BucketKey(pub i64);

impl BucketKey {
    /// Create a bucket key from a price and bucket size.
    #[inline]
    pub fn from_price(price: f32, bucket_size: f32) -> Self {
        Self((price / bucket_size).floor() as i64)
    }
}

/// Unique identifier for a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LevelId(pub u64);

impl LevelId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// The type of level derived from a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LevelType {
    /// Primary hold level.
    Hold,
    /// Secondary "greedy" hold level.
    GreedyHold,
}

/// The current state of a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelState {
    /// Level is active and has not been interacted with.
    Active,
    /// Level is active and has been hit (wick touched, body held).
    Hit,
    /// Level has been broken (body closed through).
    Broken,
}

/// A record of a level being hit.
#[derive(Debug, Clone, Copy)]
pub struct LevelHit {
    /// Index of the candle that caused the hit.
    pub candle_index: usize,
    /// The wick price that touched the level.
    pub touch_price: f32,
    /// Distance from the level (how deep the wick went).
    pub penetration: f32,
}

/// A record of a level being broken.
#[derive(Debug, Clone, Copy)]
pub struct LevelBreak {
    /// Index of the candle that broke the level.
    pub candle_index: usize,
    /// Close price of the breaking candle.
    pub close_price: f32,
}

/// A price level derived from a range.
///
/// Levels track their interaction history (hits and breaks).
#[derive(Debug, Clone)]
pub struct Level {
    /// Unique identifier.
    pub id: LevelId,
    /// The price of this level.
    pub price: f32,
    /// Type of level (Hold or GreedyHold).
    pub level_type: LevelType,
    /// Direction of the source range (determines if this is support or resistance).
    pub direction: CandleDirection,
    /// ID of the range that created this level.
    pub source_range_id: RangeId,
    /// Index of the candle where this level was created (when range completed).
    pub created_at_index: usize,
    /// Index of the candle whose wick defines this level's price.
    pub source_candle_index: usize,
    /// Current state of the level.
    pub state: LevelState,
    /// History of hits on this level.
    pub hits: Vec<LevelHit>,
    /// The break event, if the level was broken.
    pub break_event: Option<LevelBreak>,
    /// Tolerance for level interactions (in price units).
    pub tolerance: f32,
}

impl Level {
    /// Create a new level from a range.
    pub fn from_range(
        id: LevelId,
        range: &Range,
        level_type: LevelType,
        created_at_index: usize,
        tolerance: f32,
    ) -> Self {
        let (price, source_candle_index) = match level_type {
            LevelType::Hold => (range.hold_level_price(), range.hold_level_candle_index()),
            LevelType::GreedyHold => (range.greedy_hold_level_price(), range.greedy_hold_level_candle_index()),
        };

        Self {
            id,
            price,
            level_type,
            direction: range.direction,
            source_range_id: range.id,
            created_at_index,
            source_candle_index,
            state: LevelState::Active,
            hits: Vec::new(),
            break_event: None,
            tolerance,
        }
    }

    /// Number of times this level has been hit.
    #[inline]
    pub fn hit_count(&self) -> usize {
        self.hits.len()
    }

    /// Returns true if this level is still active (not broken).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state != LevelState::Broken
    }

    /// Returns true if this is a bearish level (from bearish range, at highs).
    ///
    /// Bearish levels are hit when the wick goes below them.
    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.direction == CandleDirection::Bearish
    }

    /// Returns true if this is a bullish level (from bullish range, at lows).
    ///
    /// Bullish levels are hit when the wick goes above them.
    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.direction == CandleDirection::Bullish
    }

    /// Check if a candle interacts with this level and update state.
    ///
    /// Returns the type of interaction that occurred.
    pub fn check_interaction(&mut self, candle_index: usize, candle: &Candle) -> LevelInteraction {
        if self.state == LevelState::Broken {
            return LevelInteraction::None;
        }

        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);

        match self.direction {
            CandleDirection::Bearish => {
                // Bearish level (at highs) - hit when wick goes BELOW the level
                // Broken when body closes BELOW the level
                self.check_bearish_level_interaction(candle_index, candle, body_top, body_bottom)
            }
            CandleDirection::Bullish => {
                // Bullish level (at lows) - hit when wick goes ABOVE the level
                // Broken when body closes ABOVE the level
                self.check_bullish_level_interaction(candle_index, candle, body_top, body_bottom)
            }
            CandleDirection::Doji => LevelInteraction::None,
        }
    }

    /// Check bearish level interaction.
    ///
    /// Bearish levels are at the highs of bearish ranges.
    /// Hit when the wick touches or goes BELOW the level.
    /// Broken when the full body closes below the level.
    fn check_bearish_level_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        let level_upper = self.price + self.tolerance;

        // Check if the body closed below the level (broken)
        if body_top < self.price - self.tolerance {
            // Full body is below the level - BROKEN
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Check if the wick touched or went below the level but body stayed above
        if candle.low <= level_upper && body_bottom >= self.price - self.tolerance {
            // Wick touched or went below, but body closed above - HIT
            let penetration = (self.price - candle.low).max(0.0);
            let hit = LevelHit {
                candle_index,
                touch_price: candle.low,
                penetration,
            };
            self.hits.push(hit);
            self.state = LevelState::Hit;
            return LevelInteraction::Hit(hit);
        }

        LevelInteraction::None
    }

    /// Check bullish level interaction.
    ///
    /// Bullish levels are at the lows of bullish ranges.
    /// Hit when the wick touches or goes ABOVE the level.
    /// Broken when the full body closes above the level.
    fn check_bullish_level_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        let level_lower = self.price - self.tolerance;

        // Check if the body closed above the level (broken)
        if body_bottom > self.price + self.tolerance {
            // Full body is above the level - BROKEN
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Check if the wick touched or went above the level but body stayed below
        if candle.high >= level_lower && body_top <= self.price + self.tolerance {
            // Wick touched or went above, but body closed below - HIT
            let penetration = (candle.high - self.price).max(0.0);
            let hit = LevelHit {
                candle_index,
                touch_price: candle.high,
                penetration,
            };
            self.hits.push(hit);
            self.state = LevelState::Hit;
            return LevelInteraction::Hit(hit);
        }

        LevelInteraction::None
    }
}

/// The result of checking a level interaction.
#[derive(Debug, Clone, Copy)]
pub enum LevelInteraction {
    /// No interaction occurred.
    None,
    /// The level was hit (wick touched, body held).
    Hit(LevelHit),
    /// The level was broken (body closed through).
    Broken,
}

/// Tracker for multiple levels.
///
/// Manages the lifecycle of levels and checks interactions.
#[derive(Debug)]
pub struct LevelTracker {
    next_id: u64,
    /// Active levels being tracked.
    pub levels: Vec<Level>,
    /// Default tolerance for level interactions.
    pub default_tolerance: f32,
    /// Whether to create both Hold and GreedyHold levels.
    pub create_greedy_levels: bool,
}

impl LevelTracker {
    /// Create a new level tracker.
    pub fn new(default_tolerance: f32, create_greedy_levels: bool) -> Self {
        Self {
            next_id: 0,
            levels: Vec::new(),
            default_tolerance,
            create_greedy_levels,
        }
    }

    /// Create levels from a completed range.
    pub fn create_levels_from_range(&mut self, range: &Range, created_at_index: usize) {
        // Skip doji ranges
        if range.direction == CandleDirection::Doji {
            return;
        }

        // Create hold level
        let hold_level = Level::from_range(
            LevelId::new(self.next_id),
            range,
            LevelType::Hold,
            created_at_index,
            self.default_tolerance,
        );
        self.next_id += 1;
        self.levels.push(hold_level);

        // Optionally create greedy hold level
        if self.create_greedy_levels {
            let greedy_level = Level::from_range(
                LevelId::new(self.next_id),
                range,
                LevelType::GreedyHold,
                created_at_index,
                self.default_tolerance,
            );
            self.next_id += 1;
            self.levels.push(greedy_level);
        }
    }

    /// Check all active levels for interactions with a candle.
    ///
    /// Returns a list of events that occurred.
    pub fn check_interactions(
        &mut self,
        candle_index: usize,
        candle: &Candle,
    ) -> Vec<LevelEvent> {
        let mut events = Vec::new();

        for level in &mut self.levels {
            match level.check_interaction(candle_index, candle) {
                LevelInteraction::Hit(hit) => {
                    events.push(LevelEvent::Hit {
                        level_id: level.id,
                        hit,
                    });
                }
                LevelInteraction::Broken => {
                    events.push(LevelEvent::Broken {
                        level_id: level.id,
                        break_event: level.break_event.unwrap(),
                    });
                }
                LevelInteraction::None => {}
            }
        }

        events
    }

    /// Get all active (non-broken) levels.
    pub fn active_levels(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| l.is_active())
    }

    /// Get all broken levels.
    pub fn broken_levels(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| !l.is_active())
    }

    /// Remove all broken levels from tracking.
    pub fn prune_broken(&mut self) {
        self.levels.retain(|l| l.is_active());
    }

    /// Clear all levels.
    pub fn clear(&mut self) {
        self.levels.clear();
    }
}

/// Optimized level tracker with O(1) bucket-based lookup.
///
/// Uses a HashMap where keys are price buckets and values are level indices.
/// Given a candle's price range, we compute which buckets to check in O(1).
///
/// This reduces `check_interactions()` from O(N) to O(B) per candle,
/// where B is the number of buckets spanned (typically 1-10, constant).
#[derive(Debug)]
pub struct OptimizedLevelTracker {
    next_id: u64,
    /// All levels (storage).
    pub levels: Vec<Level>,
    /// Active level indices per bucket.
    price_buckets: HashMap<BucketKey, Vec<usize>>,
    /// Size of each price bucket (e.g., 0.1% of reference price).
    bucket_size: f32,
    /// Default tolerance for level interactions.
    pub default_tolerance: f32,
    /// Whether to create both Hold and GreedyHold levels.
    pub create_greedy_levels: bool,
    /// Whether bucket_size has been initialized.
    initialized: bool,
}

impl OptimizedLevelTracker {
    /// Create a new optimized level tracker.
    pub fn new(default_tolerance: f32, create_greedy_levels: bool) -> Self {
        Self {
            next_id: 0,
            levels: Vec::new(),
            price_buckets: HashMap::new(),
            bucket_size: 1.0, // Will be initialized on first candle
            default_tolerance,
            create_greedy_levels,
            initialized: false,
        }
    }

    /// Initialize bucket size based on a reference price (typically first candle).
    ///
    /// Sets bucket_size to 0.1% of the reference price.
    pub fn initialize_bucket_size(&mut self, reference_price: f32) {
        if !self.initialized && reference_price > 0.0 {
            // 0.1% of reference price as bucket size
            self.bucket_size = reference_price * 0.001;
            self.initialized = true;
        }
    }

    /// Get the bucket key for a price.
    #[inline]
    fn bucket_key(&self, price: f32) -> BucketKey {
        BucketKey::from_price(price, self.bucket_size)
    }

    /// Register a level in its price bucket.
    fn register_level(&mut self, level_index: usize, price: f32) {
        let key = self.bucket_key(price);
        self.price_buckets
            .entry(key)
            .or_insert_with(Vec::new)
            .push(level_index);
    }

    /// Remove a level from its price bucket.
    fn unregister_level(&mut self, level_index: usize, price: f32) {
        let key = self.bucket_key(price);
        if let Some(indices) = self.price_buckets.get_mut(&key) {
            indices.retain(|&idx| idx != level_index);
            // Clean up empty buckets
            if indices.is_empty() {
                self.price_buckets.remove(&key);
            }
        }
    }

    /// Create levels from a completed range.
    pub fn create_levels_from_range(&mut self, range: &Range, created_at_index: usize) {
        // Skip doji ranges
        if range.direction == CandleDirection::Doji {
            return;
        }

        // Create hold level
        let hold_level = Level::from_range(
            LevelId::new(self.next_id),
            range,
            LevelType::Hold,
            created_at_index,
            self.default_tolerance,
        );
        let hold_price = hold_level.price;
        let hold_index = self.levels.len();
        self.next_id += 1;
        self.levels.push(hold_level);
        self.register_level(hold_index, hold_price);

        // Optionally create greedy hold level
        if self.create_greedy_levels {
            let greedy_level = Level::from_range(
                LevelId::new(self.next_id),
                range,
                LevelType::GreedyHold,
                created_at_index,
                self.default_tolerance,
            );
            let greedy_price = greedy_level.price;
            let greedy_index = self.levels.len();
            self.next_id += 1;
            self.levels.push(greedy_level);
            self.register_level(greedy_index, greedy_price);
        }
    }

    /// Check all active levels for interactions with a candle.
    ///
    /// Only checks levels in buckets spanned by the candle's price range.
    /// Returns a list of events that occurred.
    pub fn check_interactions(
        &mut self,
        candle_index: usize,
        candle: &Candle,
    ) -> Vec<LevelEvent> {
        let mut events = Vec::new();

        // Compute bucket range spanned by this candle
        let low_bucket = self.bucket_key(candle.low).0;
        let high_bucket = self.bucket_key(candle.high).0;

        // Collect level indices to check (avoid borrowing issues)
        let mut levels_to_check = Vec::new();
        for bucket_idx in low_bucket..=high_bucket {
            if let Some(level_indices) = self.price_buckets.get(&BucketKey(bucket_idx)) {
                for &level_idx in level_indices {
                    // Avoid duplicates (a level might span multiple buckets conceptually)
                    if !levels_to_check.contains(&level_idx) {
                        levels_to_check.push(level_idx);
                    }
                }
            }
        }

        // Track levels to unregister (broken levels)
        let mut broken_levels = Vec::new();

        // Check each level
        for level_idx in levels_to_check {
            let level = &mut self.levels[level_idx];

            // Skip if level was created on or after this candle (not yet active)
            if level.created_at_index >= candle_index {
                continue;
            }

            // Skip already broken levels
            if level.state == LevelState::Broken {
                continue;
            }

            match level.check_interaction(candle_index, candle) {
                LevelInteraction::Hit(hit) => {
                    events.push(LevelEvent::Hit {
                        level_id: level.id,
                        hit,
                    });
                }
                LevelInteraction::Broken => {
                    events.push(LevelEvent::Broken {
                        level_id: level.id,
                        break_event: level.break_event.unwrap(),
                    });
                    // Mark for removal from bucket
                    broken_levels.push((level_idx, level.price));
                }
                LevelInteraction::None => {}
            }
        }

        // Remove broken levels from buckets (lazy cleanup)
        for (level_idx, price) in broken_levels {
            self.unregister_level(level_idx, price);
        }

        events
    }

    /// Get all active (non-broken) levels.
    pub fn active_levels(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| l.is_active())
    }

    /// Get all broken levels.
    pub fn broken_levels(&self) -> impl Iterator<Item = &Level> {
        self.levels.iter().filter(|l| !l.is_active())
    }

    /// Remove all broken levels from tracking.
    ///
    /// Note: This also removes them from the levels Vec, which invalidates
    /// bucket indices. After calling this, the buckets are rebuilt.
    pub fn prune_broken(&mut self) {
        // Remove broken levels and rebuild buckets
        self.levels.retain(|l| l.is_active());
        self.rebuild_buckets();
    }

    /// Rebuild the bucket index from scratch.
    fn rebuild_buckets(&mut self) {
        self.price_buckets.clear();
        for (idx, level) in self.levels.iter().enumerate() {
            if level.is_active() {
                let key = self.bucket_key(level.price);
                self.price_buckets
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }
    }

    /// Clear all levels.
    pub fn clear(&mut self) {
        self.levels.clear();
        self.price_buckets.clear();
        self.initialized = false;
    }
}

/// Events that can occur on levels.
#[derive(Debug, Clone, Copy)]
pub enum LevelEvent {
    /// A new level was created.
    Created { level_id: LevelId },
    /// A level was hit.
    Hit { level_id: LevelId, hit: LevelHit },
    /// A level was broken.
    Broken {
        level_id: LevelId,
        break_event: LevelBreak,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 100.0)
    }

    fn make_bearish_range() -> Range {
        // Bearish range creates RESISTANCE at highs
        Range {
            id: RangeId::new(1),
            direction: CandleDirection::Bearish,
            start_index: 0,
            end_index: 1,
            candle_count: 2,
            high: 115.0,
            low: 95.0,
            open: 110.0,
            close: 100.0,
            total_volume: 200.0,
            first_high: 115.0,  // First candle high
            first_low: 100.0,
            last_high: 112.0,   // Last candle high
            last_low: 95.0,
        }
    }

    #[test]
    fn test_bearish_level_hit() {
        let range = make_bearish_range();
        // Bearish creates level at highs
        // Hold level = max(first_high, last_high) = max(115, 112) = 115
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0.5);

        assert_eq!(level.price, 115.0);
        assert!(level.is_bearish());

        // Candle with wick touching/going below bearish level but body staying above
        // Low = 114.5 (touches level at 115), but body closes above (close = 118)
        let candle = make_candle(117.0, 120.0, 114.5, 118.0);
        let interaction = level.check_interaction(3, &candle);

        match interaction {
            LevelInteraction::Hit(hit) => {
                assert_eq!(hit.candle_index, 3);
                assert_eq!(hit.touch_price, 114.5);
            }
            _ => panic!("Expected Hit"),
        }

        assert_eq!(level.state, LevelState::Hit);
        assert_eq!(level.hit_count(), 1);
    }

    #[test]
    fn test_bearish_level_broken() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0.5);

        // Bearish level is broken when body closes BELOW the level
        // Level is at 115, body is fully below (open=114, close=112)
        let candle = make_candle(114.0, 114.5, 110.0, 112.0);
        let interaction = level.check_interaction(3, &candle);

        assert!(matches!(interaction, LevelInteraction::Broken));
        assert_eq!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_level_tracker() {
        let mut tracker = LevelTracker::new(0.5, true);
        let range = make_bearish_range();

        tracker.create_levels_from_range(&range, 2);

        assert_eq!(tracker.levels.len(), 2);
        assert_eq!(tracker.levels[0].level_type, LevelType::Hold);
        assert_eq!(tracker.levels[1].level_type, LevelType::GreedyHold);
    }
}
