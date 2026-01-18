//! Level types - price levels derived from ranges with interaction tracking.

use super::direction::CandleDirection;
use super::range::{Range, RangeId};
use charter_core::Candle;

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

    /// Returns true if this level acts as resistance (bearish source range).
    #[inline]
    pub fn is_resistance(&self) -> bool {
        self.direction == CandleDirection::Bearish
    }

    /// Returns true if this level acts as support (bullish source range).
    #[inline]
    pub fn is_support(&self) -> bool {
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
                // This is a RESISTANCE level (at highs) - price should stay BELOW
                // Broken when body closes ABOVE the level
                self.check_resistance_interaction(candle_index, candle, body_top, body_bottom)
            }
            CandleDirection::Bullish => {
                // This is a SUPPORT level (at lows) - price should stay ABOVE
                // Broken when body closes BELOW the level
                self.check_support_interaction(candle_index, candle, body_top, body_bottom)
            }
            CandleDirection::Doji => LevelInteraction::None,
        }
    }

    fn check_resistance_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        let level_lower = self.price - self.tolerance;

        // Check if the body closed above the level (broken resistance)
        if body_bottom > self.price + self.tolerance {
            // Full body is above the level - BROKEN
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Check if the wick touched the level but body stayed below
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

    fn check_support_interaction(
        &mut self,
        candle_index: usize,
        candle: &Candle,
        body_top: f32,
        body_bottom: f32,
    ) -> LevelInteraction {
        let level_upper = self.price + self.tolerance;

        // Check if the body closed below the level (broken support)
        if body_top < self.price - self.tolerance {
            // Full body is below the level - BROKEN
            self.state = LevelState::Broken;
            self.break_event = Some(LevelBreak {
                candle_index,
                close_price: candle.close,
            });
            return LevelInteraction::Broken;
        }

        // Check if the wick touched the level but body stayed above
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
        // Bearish creates resistance at highs
        // Hold level = max(first_high, last_high) = max(115, 112) = 115
        let mut level = Level::from_range(LevelId::new(1), &range, LevelType::Hold, 2, 0.5);

        assert_eq!(level.price, 115.0);
        assert!(level.is_resistance());

        // Candle with wick touching resistance level but body below
        // High = 116, but close = 112 (below level)
        let candle = make_candle(110.0, 116.0, 108.0, 112.0);
        let interaction = level.check_interaction(3, &candle);

        match interaction {
            LevelInteraction::Hit(hit) => {
                assert_eq!(hit.candle_index, 3);
                assert_eq!(hit.touch_price, 116.0);
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

        // Resistance is broken when body closes ABOVE the level
        // Level is at 115, body is fully above (open=116, close=118)
        let candle = make_candle(116.0, 120.0, 115.5, 118.0);
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
