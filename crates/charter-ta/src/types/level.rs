//! Level types - price levels derived from ranges with interaction tracking.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::candle::CandleDirection;
use super::range::{Range, RangeId};
use charter_core::Candle;

/// Unique identifier for a Level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LevelId(pub u64);

impl LevelId {
    /// Create a new LevelId from raw components.
    #[inline]
    pub fn new(timeframe_idx: u8, sequence: u32) -> Self {
        let id = ((timeframe_idx as u64) << 32) | (sequence as u64);
        Self(id)
    }

    /// Extract the timeframe index from the ID.
    #[inline]
    pub fn timeframe_idx(self) -> u8 {
        (self.0 >> 32) as u8
    }
}

/// The type of level derived from a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LevelType {
    /// Primary hold level (more conservative).
    Hold,
    /// Secondary "greedy" hold level (more aggressive).
    GreedyHold,
}

/// The direction a level acts as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LevelDirection {
    /// Support level (from bullish range, below current price).
    Support,
    /// Resistance level (from bearish range, above current price).
    Resistance,
}

/// The current state of a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LevelState {
    /// Level exists but hasn't been activated yet.
    /// For resistance: waiting for full candle body to close ABOVE.
    /// For support: waiting for full candle body to close BELOW.
    Inactive,
    /// Level is active and can receive hits.
    Active,
    /// Level has been broken by a same-timeframe candle.
    Broken,
}

/// A record of a level being hit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LevelHit {
    /// Index of the candle that caused the hit.
    pub candle_index: usize,
    /// The timeframe that caused the hit.
    pub timeframe_idx: u8,
    /// The wick price that touched the level.
    pub touch_price: f32,
    /// Distance from the level (how deep the wick went).
    pub penetration: f32,
    /// Whether the level was respected (only wick touched, body stayed correct side).
    pub respected: bool,
}

/// A record of a level being broken.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LevelBreak {
    /// Index of the candle that broke the level.
    pub candle_index: usize,
    /// Close price of the breaking candle.
    pub close_price: f32,
}

/// A price level derived from a range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    /// Unique identifier.
    pub id: LevelId,
    /// The price of this level.
    pub price: f32,
    /// Type of level (Hold or GreedyHold).
    pub level_type: LevelType,
    /// Direction (Support or Resistance).
    pub level_direction: LevelDirection,
    /// Direction of the source range.
    pub source_direction: CandleDirection,
    /// ID of the range that created this level.
    pub source_range_id: RangeId,
    /// Timeframe that created this level.
    pub source_timeframe: u8,
    /// Index of the candle where this level was created.
    pub created_at_index: usize,
    /// Index of the candle whose wick defines the price.
    pub source_candle_index: usize,
    /// Current state.
    pub state: LevelState,
    /// History of hits.
    pub hits: Vec<LevelHit>,
    /// Break event if broken.
    pub break_event: Option<LevelBreak>,
}

impl Level {
    /// Create a new level from a range.
    pub fn from_range(
        range: &Range,
        level_type: LevelType,
        id: LevelId,
        created_at_index: usize,
    ) -> Self {
        let (price, source_candle_index) = match level_type {
            LevelType::Hold => (range.hold_level_price(), range.hold_level_candle_index()),
            LevelType::GreedyHold => (
                range.greedy_hold_level_price(),
                range.greedy_hold_level_candle_index(),
            ),
        };

        let level_direction = match range.direction {
            CandleDirection::Bullish => LevelDirection::Support,
            CandleDirection::Bearish => LevelDirection::Resistance,
            CandleDirection::Doji => LevelDirection::Support, // Fallback
        };

        Self {
            id,
            price,
            level_type,
            level_direction,
            source_direction: range.direction,
            source_range_id: range.id,
            source_timeframe: range.id.timeframe_idx(),
            created_at_index,
            source_candle_index,
            state: LevelState::Inactive,
            hits: Vec::new(),
            break_event: None,
        }
    }

    /// Check if this level is still active (not broken).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state == LevelState::Active
    }

    /// Check if this level is broken.
    #[inline]
    pub fn is_broken(&self) -> bool {
        self.state == LevelState::Broken
    }

    /// Get the number of hits.
    #[inline]
    pub fn hit_count(&self) -> usize {
        self.hits.len()
    }

    /// Get the number of respected hits.
    #[inline]
    pub fn respected_hit_count(&self) -> usize {
        self.hits.iter().filter(|h| h.respected).count()
    }

    /// Check interaction with a candle and update state.
    ///
    /// Returns the type of interaction that occurred.
    pub fn check_interaction(
        &mut self,
        candle: &Candle,
        candle_direction: CandleDirection,
        candle_index: usize,
        timeframe_idx: u8,
    ) -> LevelInteraction {
        if self.state == LevelState::Broken {
            return LevelInteraction::None;
        }

        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);

        match self.level_direction {
            LevelDirection::Resistance => {
                self.check_resistance_interaction(
                    candle,
                    candle_direction,
                    candle_index,
                    timeframe_idx,
                    body_top,
                    body_bottom,
                )
            }
            LevelDirection::Support => {
                self.check_support_interaction(
                    candle,
                    candle_direction,
                    candle_index,
                    timeframe_idx,
                    body_top,
                    body_bottom,
                )
            }
        }
    }

    fn check_resistance_interaction(
        &mut self,
        candle: &Candle,
        candle_direction: CandleDirection,
        candle_index: usize,
        timeframe_idx: u8,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        // Activation: full body closes ABOVE level
        if self.state == LevelState::Inactive {
            if body_bottom > self.price {
                self.state = LevelState::Active;
                return LevelInteraction::Activated;
            }
            return LevelInteraction::None;
        }

        // Break check: bearish candle with full body BELOW level (same timeframe only)
        if timeframe_idx == self.source_timeframe
            && candle_direction == CandleDirection::Bearish
            && body_top < self.price
        {
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Hit check: wick touches or goes below level, body stays above
        if candle.low <= self.price && body_bottom > self.price {
            let hit = LevelHit {
                candle_index,
                timeframe_idx,
                touch_price: candle.low,
                penetration: self.price - candle.low,
                respected: true,
            };
            self.hits.push(hit);
            return LevelInteraction::Hit(hit);
        }

        // Non-respected hit: wick touches and body also crossed
        if candle.low <= self.price && body_bottom <= self.price {
            let hit = LevelHit {
                candle_index,
                timeframe_idx,
                touch_price: candle.low,
                penetration: self.price - candle.low,
                respected: false,
            };
            self.hits.push(hit);
            return LevelInteraction::Hit(hit);
        }

        LevelInteraction::None
    }

    fn check_support_interaction(
        &mut self,
        candle: &Candle,
        candle_direction: CandleDirection,
        candle_index: usize,
        timeframe_idx: u8,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        // Activation: full body closes BELOW level
        if self.state == LevelState::Inactive {
            if body_top < self.price {
                self.state = LevelState::Active;
                return LevelInteraction::Activated;
            }
            return LevelInteraction::None;
        }

        // Break check: bullish candle with full body ABOVE level (same timeframe only)
        if timeframe_idx == self.source_timeframe
            && candle_direction == CandleDirection::Bullish
            && body_bottom > self.price
        {
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Hit check: wick touches or goes above level, body stays below
        if candle.high >= self.price && body_top < self.price {
            let hit = LevelHit {
                candle_index,
                timeframe_idx,
                touch_price: candle.high,
                penetration: candle.high - self.price,
                respected: true,
            };
            self.hits.push(hit);
            return LevelInteraction::Hit(hit);
        }

        // Non-respected hit
        if candle.high >= self.price && body_top >= self.price {
            let hit = LevelHit {
                candle_index,
                timeframe_idx,
                touch_price: candle.high,
                penetration: candle.high - self.price,
                respected: false,
            };
            self.hits.push(hit);
            return LevelInteraction::Hit(hit);
        }

        LevelInteraction::None
    }

    /// Check if this level would be broken by a price traversal.
    ///
    /// Used during reverse pass to determine if old levels are already broken.
    pub fn is_broken_by_close(&self, close_price: f32, candle_direction: CandleDirection) -> bool {
        match self.level_direction {
            LevelDirection::Resistance => {
                // Broken if bearish candle closed below
                candle_direction == CandleDirection::Bearish && close_price < self.price
            }
            LevelDirection::Support => {
                // Broken if bullish candle closed above
                candle_direction == CandleDirection::Bullish && close_price > self.price
            }
        }
    }
}

/// The type of interaction that occurred with a level.
#[derive(Debug, Clone, Copy)]
pub enum LevelInteraction {
    /// No interaction.
    None,
    /// Level was activated.
    Activated,
    /// Level was hit.
    Hit(LevelHit),
    /// Level was broken.
    Broken,
}

/// Events emitted by level tracking.
#[derive(Debug, Clone)]
pub enum LevelEvent {
    Created { level_id: LevelId },
    Activated { level_id: LevelId },
    Hit { level_id: LevelId, hit: LevelHit },
    Broken { level_id: LevelId, break_event: LevelBreak },
}

/// Efficient index for levels with O(log n) price-based queries.
#[derive(Debug, Default)]
pub struct LevelIndex {
    /// All levels by ID.
    by_id: HashMap<LevelId, Level>,

    /// Active resistance levels ordered by price (ascending for closest above).
    active_resistance: BTreeSet<(OrderedFloat<f32>, LevelId)>,

    /// Active support levels ordered by price descending (for closest below).
    active_support: BTreeSet<(Reverse<OrderedFloat<f32>>, LevelId)>,

    /// All levels by price for range queries.
    by_price: BTreeMap<OrderedFloat<f32>, Vec<LevelId>>,

    /// ID generation
    next_sequence: u32,
    timeframe_idx: u8,
}

impl LevelIndex {
    /// Create a new LevelIndex for a specific timeframe.
    pub fn new(timeframe_idx: u8) -> Self {
        Self {
            by_id: HashMap::new(),
            active_resistance: BTreeSet::new(),
            active_support: BTreeSet::new(),
            by_price: BTreeMap::new(),
            next_sequence: 0,
            timeframe_idx,
        }
    }

    /// Generate a new unique LevelId.
    pub fn next_id(&mut self) -> LevelId {
        let id = LevelId::new(self.timeframe_idx, self.next_sequence);
        self.next_sequence += 1;
        id
    }

    /// Insert a level into the index.
    pub fn insert(&mut self, level: Level) {
        let id = level.id;
        let price = OrderedFloat(level.price);
        let direction = level.level_direction;

        // Add to price index
        self.by_price.entry(price).or_default().push(id);

        // Add to active sets if not broken
        if level.state != LevelState::Broken {
            match direction {
                LevelDirection::Resistance => {
                    self.active_resistance.insert((price, id));
                }
                LevelDirection::Support => {
                    self.active_support.insert((Reverse(price), id));
                }
            }
        }

        self.by_id.insert(id, level);
    }

    /// Get a level by ID.
    pub fn get(&self, id: LevelId) -> Option<&Level> {
        self.by_id.get(&id)
    }

    /// Get a mutable reference to a level by ID.
    pub fn get_mut(&mut self, id: LevelId) -> Option<&mut Level> {
        self.by_id.get_mut(&id)
    }

    /// Remove a level from the active sets (when broken).
    pub fn mark_broken(&mut self, id: LevelId) {
        if let Some(level) = self.by_id.get_mut(&id) {
            let price = OrderedFloat(level.price);
            match level.level_direction {
                LevelDirection::Resistance => {
                    self.active_resistance.remove(&(price, id));
                }
                LevelDirection::Support => {
                    self.active_support.remove(&(Reverse(price), id));
                }
            }
        }
    }

    /// Find the closest resistance level above a price.
    ///
    /// O(log n) complexity.
    pub fn closest_resistance_above(&self, price: f32) -> Option<&Level> {
        let price_key = OrderedFloat(price);
        self.active_resistance
            .range((price_key, LevelId(0))..)
            .next()
            .and_then(|(_, id)| self.by_id.get(id))
    }

    /// Find the closest support level below a price.
    ///
    /// O(log n) complexity.
    pub fn closest_support_below(&self, price: f32) -> Option<&Level> {
        let price_key = Reverse(OrderedFloat(price));
        self.active_support
            .range((price_key, LevelId(0))..)
            .next()
            .and_then(|(_, id)| self.by_id.get(id))
    }

    /// Find all levels within a price range.
    ///
    /// O(log n + k) where k is the number of levels in range.
    pub fn levels_in_range(&self, low: f32, high: f32) -> Vec<&Level> {
        let low_key = OrderedFloat(low);
        let high_key = OrderedFloat(high);

        self.by_price
            .range(low_key..=high_key)
            .flat_map(|(_, ids)| ids.iter().filter_map(|id| self.by_id.get(id)))
            .collect()
    }

    /// Get all active levels.
    pub fn active_levels(&self) -> impl Iterator<Item = &Level> {
        self.by_id
            .values()
            .filter(|l| l.state == LevelState::Active)
    }

    /// Get all unbroken levels (active or inactive).
    pub fn unbroken_levels(&self) -> impl Iterator<Item = &Level> {
        self.by_id.values().filter(|l| l.state != LevelState::Broken)
    }

    /// Get the number of active levels.
    pub fn active_count(&self) -> usize {
        self.active_resistance.len() + self.active_support.len()
    }

    /// Get the total number of levels.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Iterate over all levels.
    pub fn iter(&self) -> impl Iterator<Item = &Level> {
        self.by_id.values()
    }

    /// Get closest N resistance levels above a price.
    pub fn closest_n_resistance_above(&self, price: f32, n: usize) -> Vec<&Level> {
        let price_key = OrderedFloat(price);
        self.active_resistance
            .range((price_key, LevelId(0))..)
            .take(n)
            .filter_map(|(_, id)| self.by_id.get(id))
            .collect()
    }

    /// Get closest N support levels below a price.
    pub fn closest_n_support_below(&self, price: f32, n: usize) -> Vec<&Level> {
        let price_key = Reverse(OrderedFloat(price));
        self.active_support
            .range((price_key, LevelId(0))..)
            .take(n)
            .filter_map(|(_, id)| self.by_id.get(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::range::RangeId;

    fn make_resistance_level(price: f32, id: u32) -> Level {
        Level {
            id: LevelId::new(0, id),
            price,
            level_type: LevelType::Hold,
            level_direction: LevelDirection::Resistance,
            source_direction: CandleDirection::Bearish,
            source_range_id: RangeId::new(0, 0),
            source_timeframe: 0,
            created_at_index: 0,
            source_candle_index: 0,
            state: LevelState::Active,
            hits: Vec::new(),
            break_event: None,
        }
    }

    fn make_support_level(price: f32, id: u32) -> Level {
        Level {
            id: LevelId::new(0, id),
            price,
            level_type: LevelType::Hold,
            level_direction: LevelDirection::Support,
            source_direction: CandleDirection::Bullish,
            source_range_id: RangeId::new(0, 0),
            source_timeframe: 0,
            created_at_index: 0,
            source_candle_index: 0,
            state: LevelState::Active,
            hits: Vec::new(),
            break_event: None,
        }
    }

    #[test]
    fn test_level_index_closest_resistance() {
        let mut index = LevelIndex::new(0);

        index.insert(make_resistance_level(110.0, 1));
        index.insert(make_resistance_level(105.0, 2));
        index.insert(make_resistance_level(120.0, 3));

        let closest = index.closest_resistance_above(100.0);
        assert!(closest.is_some());
        assert_eq!(closest.unwrap().price, 105.0);

        let closest = index.closest_resistance_above(106.0);
        assert!(closest.is_some());
        assert_eq!(closest.unwrap().price, 110.0);
    }

    #[test]
    fn test_level_index_closest_support() {
        let mut index = LevelIndex::new(0);

        index.insert(make_support_level(90.0, 1));
        index.insert(make_support_level(95.0, 2));
        index.insert(make_support_level(80.0, 3));

        let closest = index.closest_support_below(100.0);
        assert!(closest.is_some());
        assert_eq!(closest.unwrap().price, 95.0);

        let closest = index.closest_support_below(94.0);
        assert!(closest.is_some());
        assert_eq!(closest.unwrap().price, 90.0);
    }

    #[test]
    fn test_level_index_n_closest() {
        let mut index = LevelIndex::new(0);

        index.insert(make_resistance_level(105.0, 1));
        index.insert(make_resistance_level(110.0, 2));
        index.insert(make_resistance_level(115.0, 3));
        index.insert(make_resistance_level(120.0, 4));

        let closest = index.closest_n_resistance_above(100.0, 3);
        assert_eq!(closest.len(), 3);
        assert_eq!(closest[0].price, 105.0);
        assert_eq!(closest[1].price, 110.0);
        assert_eq!(closest[2].price, 115.0);
    }

    #[test]
    fn test_level_interaction_resistance() {
        let mut level = make_resistance_level(100.0, 1);
        level.state = LevelState::Active;

        // Hit: wick goes below, body stays above
        let candle = Candle::new(0.0, 102.0, 105.0, 99.0, 103.0, 1.0);
        let interaction = level.check_interaction(&candle, CandleDirection::Bullish, 0, 0);
        assert!(matches!(interaction, LevelInteraction::Hit(_)));
        assert_eq!(level.hit_count(), 1);

        // Break: bearish candle closes fully below
        let candle = Candle::new(0.0, 99.0, 100.0, 95.0, 96.0, 1.0);
        let interaction = level.check_interaction(&candle, CandleDirection::Bearish, 1, 0);
        assert!(matches!(interaction, LevelInteraction::Broken));
        assert!(level.is_broken());
    }

    #[test]
    fn test_level_interaction_support() {
        let mut level = make_support_level(100.0, 1);
        level.state = LevelState::Active;

        // Hit: wick goes above, body stays below
        let candle = Candle::new(0.0, 98.0, 101.0, 96.0, 97.0, 1.0);
        let interaction = level.check_interaction(&candle, CandleDirection::Bearish, 0, 0);
        assert!(matches!(interaction, LevelInteraction::Hit(_)));
        assert_eq!(level.hit_count(), 1);

        // Break: bullish candle closes fully above
        let candle = Candle::new(0.0, 101.0, 105.0, 100.0, 104.0, 1.0);
        let interaction = level.check_interaction(&candle, CandleDirection::Bullish, 1, 0);
        assert!(matches!(interaction, LevelInteraction::Broken));
        assert!(level.is_broken());
    }
}
