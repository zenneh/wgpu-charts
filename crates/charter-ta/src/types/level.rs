//! Level types - price levels derived from ranges with interaction tracking.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

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
    /// Level exists but price hasn't closed on the other side yet.
    /// For bearish levels: waiting for full candle body to close ABOVE the level.
    /// For bullish levels: waiting for full candle body to close BELOW the level.
    Inactive,
    /// Level is active (price closed on other side). Can receive hits.
    /// Hits are recorded but don't change the state.
    Active,
    /// Level has been broken (same-direction candle closed fully through).
    /// No more hits are recorded once broken.
    Broken,
}

/// A record of a level being hit.
#[derive(Debug, Clone, Copy)]
pub struct LevelHit {
    /// Index of the candle that caused the hit.
    pub candle_index: usize,
    /// The timeframe index that caused the hit.
    /// This allows distinguishing between e.g. 1min hits vs 15min hits.
    pub timeframe: usize,
    /// The wick price that touched the level.
    pub touch_price: f32,
    /// Distance from the level (how deep the wick went).
    pub penetration: f32,
    /// Whether the level was respected (only wick touched, body stayed on correct side).
    /// If false, the body also touched or crossed the level.
    pub respected: bool,
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
    /// The timeframe index this level was created from.
    /// A level can only be broken by candles from the same timeframe.
    pub source_timeframe: usize,
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
    ///
    /// `source_timeframe` is the timeframe index this level belongs to.
    /// A level can only be broken by candles from the same timeframe.
    pub fn from_range(
        id: LevelId,
        range: &Range,
        level_type: LevelType,
        created_at_index: usize,
        source_timeframe: usize,
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
            source_timeframe,
            state: LevelState::Inactive, // Start as Pending until price crosses to other side
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
    /// `candle_timeframe` is the timeframe index of the candle being checked.
    /// A level can only be BROKEN by candles from the same timeframe it was created in.
    /// Activations and hits can occur from any timeframe.
    ///
    /// Level interaction rules:
    /// 1. A level starts as Pending until price crosses to the "other side"
    ///    - Bearish level: activated when price goes ABOVE the level
    ///    - Bullish level: activated when price goes BELOW the level
    /// 2. Once active, a level can be "hit" when:
    ///    - Bearish level: the LOW wick touches or goes below the level
    ///    - Bullish level: the HIGH wick touches or goes above the level
    /// 3. A hit is "respected" if only the wick touched (body stayed on correct side)
    ///    Not respected if the body also touched or crossed the level
    /// 4. A level is "broken" only when a candle of the SAME direction AND SAME TIMEFRAME:
    ///    - Bearish level: a BEARISH candle opens AND closes fully BELOW the level
    ///    - Bullish level: a BULLISH candle opens AND closes fully ABOVE the level
    pub fn check_interaction(&mut self, candle_index: usize, candle: &Candle, candle_timeframe: usize) -> LevelInteraction {
        if self.state == LevelState::Broken {
            return LevelInteraction::None;
        }

        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);
        let candle_direction = if candle.close > candle.open {
            CandleDirection::Bullish
        } else if candle.close < candle.open {
            CandleDirection::Bearish
        } else {
            CandleDirection::Doji
        };

        // A level can only be broken by candles from the same timeframe
        let can_break = candle_timeframe == self.source_timeframe;

        match self.direction {
            CandleDirection::Bearish => {
                self.check_bearish_level_interaction(candle_index, candle, candle_timeframe, body_top, body_bottom, candle_direction, can_break)
            }
            CandleDirection::Bullish => {
                self.check_bullish_level_interaction(candle_index, candle, candle_timeframe, body_top, body_bottom, candle_direction, can_break)
            }
            CandleDirection::Doji => LevelInteraction::None,
        }
    }

    /// Check bearish level interaction.
    ///
    /// Bearish levels are at the highs of bearish ranges (resistance).
    /// - Activated when full candle body is ABOVE the level (both open and close above)
    /// - Hit when LOW wick touches or goes BELOW the level
    /// - Broken when a BEARISH candle opens AND closes fully BELOW the level (same timeframe only)
    fn check_bearish_level_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        candle_timeframe: usize,
        body_top: f32,
        body_bottom: f32,
        candle_direction: CandleDirection,
        can_break: bool,
    ) -> LevelInteraction {
        // Step 1: Check for activation (full candle body must be ABOVE the level)
        if self.state == LevelState::Inactive {
            // For bearish level, activate when full candle body is above the level
            // (both open and close must be above)
            if body_bottom > self.price + self.tolerance {
                self.state = LevelState::Active;
                return LevelInteraction::Activated;
            }
            // Not yet activated
            return LevelInteraction::None;
        }

        // Step 2: Check for break (ONLY bearish candles from SAME TIMEFRAME can break bearish levels)
        // Bearish candle must open AND close fully BELOW the level (not below level - tolerance)
        // Can only break from Active state (not from Hit state)
        if self.state == LevelState::Active && can_break && candle_direction == CandleDirection::Bearish {
            if body_top < self.price && body_bottom < self.price {
                // Bearish candle with both open and close below level - BROKEN
                self.state = LevelState::Broken;
                self.break_event = Some(LevelBreak {
                    candle_index,
                    close_price: candle.close,
                });
                return LevelInteraction::Broken;
            }
        }

        // Step 3: Check for hit (LOW wick touches or goes below level)
        if candle.low <= self.price + self.tolerance {
            // Wick touched or went below the level
            let penetration = (self.price - candle.low).max(0.0);

            // Check if respected: body must stay ABOVE the level
            // Respected = only wick touched, body didn't cross
            let respected = body_bottom > self.price - self.tolerance;

            let hit = LevelHit {
                candle_index,
                timeframe: candle_timeframe,
                touch_price: candle.low,
                penetration,
                respected,
            };
            self.hits.push(hit);
            // State stays Active - hits don't change the state
            return LevelInteraction::Hit(hit);
        }

        LevelInteraction::None
    }

    /// Check bullish level interaction.
    ///
    /// Bullish levels are at the lows of bullish ranges (support).
    /// - Activated when full candle body is BELOW the level (both open and close below)
    /// - Hit when HIGH wick touches or goes ABOVE the level
    /// - Broken when a BULLISH candle opens AND closes fully ABOVE the level (same timeframe only)
    fn check_bullish_level_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        candle_timeframe: usize,
        body_top: f32,
        body_bottom: f32,
        candle_direction: CandleDirection,
        can_break: bool,
    ) -> LevelInteraction {
        // Step 1: Check for activation (full candle body must be BELOW the level)
        if self.state == LevelState::Inactive {
            // For bullish level, activate when full candle body is below the level
            // (both open and close must be below)
            if body_top < self.price - self.tolerance {
                self.state = LevelState::Active;
                return LevelInteraction::Activated;
            }
            // Not yet activated
            return LevelInteraction::None;
        }

        // Step 2: Check for break (ONLY bullish candles from SAME TIMEFRAME can break bullish levels)
        // Bullish candle must open AND close fully ABOVE the level (not above level + tolerance)
        // Can only break from Active state (not from Hit state)
        if self.state == LevelState::Active && can_break && candle_direction == CandleDirection::Bullish {
            if body_bottom > self.price && body_top > self.price {
                // Bullish candle with both open and close above level - BROKEN
                self.state = LevelState::Broken;
                self.break_event = Some(LevelBreak {
                    candle_index,
                    close_price: candle.close,
                });
                return LevelInteraction::Broken;
            }
        }

        // Step 3: Check for hit (HIGH wick touches or goes above level)
        if candle.high >= self.price - self.tolerance {
            // Wick touched or went above the level
            let penetration = (candle.high - self.price).max(0.0);

            // Check if respected: body must stay BELOW the level
            // Respected = only wick touched, body didn't cross
            let respected = body_top < self.price + self.tolerance;

            let hit = LevelHit {
                candle_index,
                timeframe: candle_timeframe,
                touch_price: candle.high,
                penetration,
                respected,
            };
            self.hits.push(hit);
            // State stays Active - hits don't change the state
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
    /// The level was activated (price crossed to the other side).
    Activated,
    /// The level was hit (wick touched level).
    /// Check `hit.respected` to see if body stayed on correct side.
    Hit(LevelHit),
    /// The level was broken (same-direction candle closed fully through).
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
    /// The timeframe index this tracker is for.
    pub timeframe: usize,
}

impl LevelTracker {
    /// Create a new level tracker for a specific timeframe.
    pub fn new(default_tolerance: f32, create_greedy_levels: bool, timeframe: usize) -> Self {
        Self {
            next_id: 0,
            levels: Vec::new(),
            default_tolerance,
            create_greedy_levels,
            timeframe,
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
            self.timeframe,
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
                self.timeframe,
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
            match level.check_interaction(candle_index, candle, self.timeframe) {
                LevelInteraction::Activated => {
                    events.push(LevelEvent::Activated {
                        level_id: level.id,
                    });
                }
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
    /// The timeframe index this tracker is for.
    pub timeframe: usize,
}

impl OptimizedLevelTracker {
    /// Create a new optimized level tracker for a specific timeframe.
    pub fn new(default_tolerance: f32, create_greedy_levels: bool, timeframe: usize) -> Self {
        Self {
            next_id: 0,
            levels: Vec::new(),
            price_buckets: HashMap::new(),
            bucket_size: 1.0, // Will be initialized on first candle
            default_tolerance,
            create_greedy_levels,
            initialized: false,
            timeframe,
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
            self.timeframe,
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
                self.timeframe,
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
        // Using HashSet for O(1) duplicate detection instead of O(N) Vec::contains
        let mut levels_to_check = HashSet::new();
        for bucket_idx in low_bucket..=high_bucket {
            if let Some(level_indices) = self.price_buckets.get(&BucketKey(bucket_idx)) {
                for &level_idx in level_indices {
                    levels_to_check.insert(level_idx);
                }
            }
        }

        let timeframe = self.timeframe;

        // Parallel phase: check all relevant levels concurrently
        // Each level check is independent, making this embarrassingly parallel
        let results: Vec<(usize, LevelId, f32, LevelInteraction)> = self
            .levels
            .par_iter_mut()
            .enumerate()
            .filter(|(idx, _)| levels_to_check.contains(idx))
            .filter_map(|(idx, level)| {
                // Skip if level was created on or after this candle (not yet active)
                if level.created_at_index >= candle_index {
                    return None;
                }
                // Skip already broken levels
                if level.state == LevelState::Broken {
                    return None;
                }
                let interaction = level.check_interaction(candle_index, candle, timeframe);
                if matches!(interaction, LevelInteraction::None) {
                    None
                } else {
                    Some((idx, level.id, level.price, interaction))
                }
            })
            .collect();

        // Sequential phase: collect events and track broken levels
        let mut broken_levels = Vec::new();
        for (idx, level_id, price, interaction) in results {
            match interaction {
                LevelInteraction::Activated => {
                    events.push(LevelEvent::Activated { level_id });
                }
                LevelInteraction::Hit(hit) => {
                    events.push(LevelEvent::Hit { level_id, hit });
                }
                LevelInteraction::Broken => {
                    // Need to get break_event from the level
                    let break_event = self.levels[idx].break_event.unwrap();
                    events.push(LevelEvent::Broken {
                        level_id,
                        break_event,
                    });
                    broken_levels.push((idx, price));
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
    /// A level was activated (price crossed to the other side).
    Activated { level_id: LevelId },
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

    fn make_bullish_range() -> Range {
        // Bullish range creates SUPPORT at lows
        Range {
            id: RangeId::new(2),
            direction: CandleDirection::Bullish,
            start_index: 0,
            end_index: 1,
            candle_count: 2,
            high: 120.0,
            low: 100.0,
            open: 105.0,
            close: 115.0,
            total_volume: 200.0,
            first_high: 110.0,
            first_low: 100.0,  // First candle low
            last_high: 120.0,
            last_low: 105.0,   // Last candle low
        }
    }

    #[test]
    fn test_bearish_level_activation() {
        let range = make_bearish_range();
        // Level at 115.0 (max of first_high and last_high)
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        assert_eq!(level.price, 115.0);
        assert_eq!(level.state, LevelState::Inactive);

        // Candle below level - should NOT activate (need full body ABOVE first)
        let candle_below = make_candle(110.0, 112.0, 108.0, 111.0);
        let interaction = level.check_interaction(3, &candle_below, 0);
        assert!(matches!(interaction, LevelInteraction::None));
        assert_eq!(level.state, LevelState::Inactive);

        // Candle with only wick above - should NOT activate (need full body above)
        let candle_wick_above = make_candle(114.0, 118.0, 113.0, 115.0);
        let interaction = level.check_interaction(4, &candle_wick_above, 0);
        assert!(matches!(interaction, LevelInteraction::None));
        assert_eq!(level.state, LevelState::Inactive);

        // Candle with full body above level (both open and close > 115.5) - should activate
        let candle_body_above = make_candle(116.0, 120.0, 115.8, 118.0);
        let interaction = level.check_interaction(5, &candle_body_above, 0);
        assert!(matches!(interaction, LevelInteraction::Activated));
        assert_eq!(level.state, LevelState::Active);
    }

    #[test]
    fn test_bearish_level_hit_respected() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // First activate the level (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);
        assert_eq!(level.state, LevelState::Active);

        // Now test hit: wick touches level but body stays above (respected)
        // Level at 115, low=114.5 touches, but body (open=117, close=118) stays above
        let candle = make_candle(117.0, 120.0, 114.5, 118.0);
        let interaction = level.check_interaction(4, &candle, 0);

        match interaction {
            LevelInteraction::Hit(hit) => {
                assert_eq!(hit.candle_index, 4);
                assert_eq!(hit.timeframe, 0);
                assert_eq!(hit.touch_price, 114.5);
                assert!(hit.respected, "Body stayed above, should be respected");
            }
            _ => panic!("Expected Hit"),
        }

        assert_eq!(level.state, LevelState::Active); // State stays Active after hit
        assert_eq!(level.hit_count(), 1);
    }

    #[test]
    fn test_bearish_level_hit_not_respected() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);

        // Hit where body also crosses the level (not respected)
        // Level at 115, body bottom = min(114.5, 116) = 114.5 which is below level
        let candle = make_candle(114.5, 120.0, 112.0, 116.0);
        let interaction = level.check_interaction(4, &candle, 0);

        match interaction {
            LevelInteraction::Hit(hit) => {
                assert!(!hit.respected, "Body crossed level, should NOT be respected");
            }
            _ => panic!("Expected Hit"),
        }
    }

    #[test]
    fn test_bearish_level_broken_by_bearish_candle() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);

        // Bearish candle (close < open) with body fully below level = BROKEN
        // Level at 115, open=114 close=112 both below 115
        let candle = make_candle(114.0, 114.5, 110.0, 112.0);
        let interaction = level.check_interaction(4, &candle, 0);

        assert!(matches!(interaction, LevelInteraction::Broken));
        assert_eq!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_bearish_level_not_broken_by_bullish_candle() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);

        // Bullish candle (close > open) with body fully below level = NOT broken (just a hit)
        // Only BEARISH candles can break bearish levels
        let candle = make_candle(112.0, 114.5, 110.0, 114.0);
        let interaction = level.check_interaction(4, &candle, 0);

        // Should be a hit (wick touched) but NOT broken
        assert!(matches!(interaction, LevelInteraction::Hit(_)));
        assert_eq!(level.state, LevelState::Active); // State stays Active after hit
    }

    #[test]
    fn test_bullish_level_activation() {
        let range = make_bullish_range();
        // Level at 100.0 (min of first_low and last_low)
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        assert_eq!(level.price, 100.0);
        assert_eq!(level.state, LevelState::Inactive);

        // Candle above level - should NOT activate (need full body BELOW first)
        let candle_above = make_candle(105.0, 110.0, 102.0, 108.0);
        let interaction = level.check_interaction(3, &candle_above, 0);
        assert!(matches!(interaction, LevelInteraction::None));
        assert_eq!(level.state, LevelState::Inactive);

        // Candle with only wick below - should NOT activate (need full body below)
        let candle_wick_below = make_candle(101.0, 102.0, 98.0, 100.5);
        let interaction = level.check_interaction(4, &candle_wick_below, 0);
        assert!(matches!(interaction, LevelInteraction::None));
        assert_eq!(level.state, LevelState::Inactive);

        // Candle with full body below level (both open and close < 99.5) - should activate
        let candle_body_below = make_candle(99.0, 99.3, 97.0, 98.0);
        let interaction = level.check_interaction(5, &candle_body_below, 0);
        assert!(matches!(interaction, LevelInteraction::Activated));
        assert_eq!(level.state, LevelState::Active);
    }

    #[test]
    fn test_bullish_level_broken_by_bullish_candle() {
        let range = make_bullish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body below level)
        let candle_below = make_candle(99.0, 99.3, 97.0, 98.0);
        level.check_interaction(3, &candle_below, 0);

        // Bullish candle (close > open) with body fully above level = BROKEN
        // Level at 100, open=101 close=105 both above 100
        let candle = make_candle(101.0, 106.0, 100.5, 105.0);
        let interaction = level.check_interaction(4, &candle, 0);

        assert!(matches!(interaction, LevelInteraction::Broken));
        assert_eq!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_level_tracker() {
        let mut tracker = LevelTracker::new(0.5, true, 0);
        let range = make_bearish_range();

        tracker.create_levels_from_range(&range, 2);

        assert_eq!(tracker.levels.len(), 2);
        assert_eq!(tracker.levels[0].level_type, LevelType::Hold);
        assert_eq!(tracker.levels[1].level_type, LevelType::GreedyHold);
    }

    #[test]
    fn test_level_not_broken_by_different_timeframe() {
        let range = make_bearish_range();
        // Create level in timeframe 0
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level) from any timeframe
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 1); // Different timeframe
        assert_eq!(level.state, LevelState::Active);

        // Bearish candle that would break the level IF same timeframe
        // But we pass timeframe 1 (different from level's timeframe 0)
        let candle = make_candle(114.0, 114.5, 110.0, 112.0);
        let interaction = level.check_interaction(4, &candle, 1); // Different timeframe

        // Should NOT be broken (different timeframe), but could still be a hit
        assert!(!matches!(interaction, LevelInteraction::Broken));
        assert_ne!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_level_broken_by_same_timeframe() {
        let range = make_bearish_range();
        // Create level in timeframe 0
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);
        assert_eq!(level.state, LevelState::Active);

        // Bearish candle with same timeframe should break it
        let candle = make_candle(114.0, 114.5, 110.0, 112.0);
        let interaction = level.check_interaction(4, &candle, 0); // Same timeframe

        assert!(matches!(interaction, LevelInteraction::Broken));
        assert_eq!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_hit_tracks_timeframe() {
        let range = make_bearish_range();
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0, 0.5);

        // Activate (full body above level)
        let candle_above = make_candle(116.0, 120.0, 115.8, 118.0);
        level.check_interaction(3, &candle_above, 0);

        // Hit from timeframe 1
        let candle1 = make_candle(117.0, 120.0, 114.5, 118.0);
        let interaction1 = level.check_interaction(4, &candle1, 1);
        match interaction1 {
            LevelInteraction::Hit(hit) => {
                assert_eq!(hit.timeframe, 1, "Hit should record timeframe 1");
            }
            _ => panic!("Expected Hit"),
        }

        // Hit from timeframe 2
        let candle2 = make_candle(117.0, 120.0, 114.0, 118.0);
        let interaction2 = level.check_interaction(5, &candle2, 2);
        match interaction2 {
            LevelInteraction::Hit(hit) => {
                assert_eq!(hit.timeframe, 2, "Hit should record timeframe 2");
            }
            _ => panic!("Expected Hit"),
        }

        // Verify both hits are recorded with their timeframes
        assert_eq!(level.hits.len(), 2);
        assert_eq!(level.hits[0].timeframe, 1);
        assert_eq!(level.hits[1].timeframe, 2);
    }
}
