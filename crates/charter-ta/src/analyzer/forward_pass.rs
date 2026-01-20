//! Forward pass algorithm for level interaction evaluation.
//!
//! The forward pass processes candles from oldest to newest, checking
//! for level activations, hits, and breaks.

use charter_core::Candle;

use crate::types::{CandleMetadata, LevelEvent, LevelId, LevelIndex, LevelInteraction};

/// Result of the forward pass.
#[derive(Debug, Default)]
pub struct ForwardPassResult {
    /// Level events that occurred during the pass.
    pub events: Vec<LevelEvent>,
    /// IDs of levels that were activated.
    pub activated: Vec<LevelId>,
    /// IDs of levels that were hit.
    pub hit: Vec<LevelId>,
    /// IDs of levels that were broken.
    pub broken: Vec<LevelId>,
}

/// Perform forward pass on candle data to check level interactions.
///
/// This function processes candles from oldest to newest, checking each
/// candle against all relevant levels for activations, hits, and breaks.
///
/// # Arguments
///
/// * `candles` - The candle data (oldest first)
/// * `level_index` - The level index to check against
/// * `timeframe_idx` - The timeframe being processed
/// * `doji_threshold` - Threshold for doji detection
/// * `start_index` - Index to start processing from (for incremental updates)
pub fn forward_pass(
    candles: &[Candle],
    level_index: &mut LevelIndex,
    timeframe_idx: u8,
    doji_threshold: f32,
    start_index: usize,
) -> ForwardPassResult {
    let mut result = ForwardPassResult::default();

    if candles.is_empty() || start_index >= candles.len() {
        return result;
    }

    for (i, candle) in candles.iter().enumerate().skip(start_index) {
        let meta = CandleMetadata::from_candle(candle, doji_threshold);

        // Get all unbroken levels - we need to check activations (level can be outside candle range)
        // and breaks (level can also be outside candle range)
        let level_ids: Vec<LevelId> = level_index.unbroken_levels().map(|l| l.id).collect();

        for level_id in level_ids {
            if let Some(level) = level_index.get_mut(level_id) {
                let interaction =
                    level.check_interaction(candle, meta.direction, i, timeframe_idx);

                match interaction {
                    LevelInteraction::Activated => {
                        result.events.push(LevelEvent::Activated { level_id });
                        result.activated.push(level_id);
                    }
                    LevelInteraction::Hit(hit) => {
                        result.events.push(LevelEvent::Hit { level_id, hit });
                        result.hit.push(level_id);
                    }
                    LevelInteraction::Broken => {
                        if let Some(break_event) = level.break_event {
                            result.events.push(LevelEvent::Broken {
                                level_id,
                                break_event,
                            });
                            result.broken.push(level_id);
                            // Mark as broken in the index
                            level_index.mark_broken(level_id);
                        }
                    }
                    LevelInteraction::None => {}
                }
            }
        }
    }

    result
}

/// Check a single candle against all levels in the index.
///
/// This is useful for streaming/incremental processing where candles
/// arrive one at a time.
pub fn check_candle_interactions(
    candle: &Candle,
    candle_index: usize,
    level_index: &mut LevelIndex,
    timeframe_idx: u8,
    doji_threshold: f32,
) -> Vec<LevelEvent> {
    let mut events = Vec::new();
    let meta = CandleMetadata::from_candle(candle, doji_threshold);

    // Get all unbroken levels - activations/breaks can involve levels outside candle range
    let level_ids: Vec<LevelId> = level_index.unbroken_levels().map(|l| l.id).collect();

    for level_id in level_ids {
        if let Some(level) = level_index.get_mut(level_id) {
            let interaction =
                level.check_interaction(candle, meta.direction, candle_index, timeframe_idx);

            match interaction {
                LevelInteraction::Activated => {
                    events.push(LevelEvent::Activated { level_id });
                }
                LevelInteraction::Hit(hit) => {
                    events.push(LevelEvent::Hit { level_id, hit });
                }
                LevelInteraction::Broken => {
                    if let Some(break_event) = level.break_event {
                        events.push(LevelEvent::Broken {
                            level_id,
                            break_event,
                        });
                        level_index.mark_broken(level_id);
                    }
                }
                LevelInteraction::None => {}
            }
        }
    }

    events
}

/// Batch check multiple candles for interactions.
///
/// More efficient than calling `check_candle_interactions` repeatedly
/// as it can potentially use SIMD or parallel processing.
pub fn batch_check_interactions(
    candles: &[Candle],
    start_index: usize,
    level_index: &mut LevelIndex,
    timeframe_idx: u8,
    doji_threshold: f32,
) -> Vec<LevelEvent> {
    let mut all_events = Vec::new();

    for (offset, candle) in candles.iter().enumerate() {
        let candle_index = start_index + offset;
        let events =
            check_candle_interactions(candle, candle_index, level_index, timeframe_idx, doji_threshold);
        all_events.extend(events);
    }

    all_events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CandleDirection, Level, LevelDirection, LevelState, LevelType, RangeId,
    };

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 1.0)
    }

    fn create_test_level_index() -> LevelIndex {
        let mut index = LevelIndex::new(0);

        // Create a resistance level at 110
        let resistance = Level {
            id: LevelId::new(0, 0),
            price: 110.0,
            level_type: LevelType::Hold,
            level_direction: LevelDirection::Resistance,
            source_direction: CandleDirection::Bearish,
            source_range_id: RangeId::new(0, 0),
            source_timeframe: 0,
            created_at_index: 0,
            source_candle_index: 0,
            state: LevelState::Inactive,
            hits: Vec::new(),
            break_event: None,
        };
        index.insert(resistance);

        // Create a support level at 90
        let support = Level {
            id: LevelId::new(0, 1),
            price: 90.0,
            level_type: LevelType::Hold,
            level_direction: LevelDirection::Support,
            source_direction: CandleDirection::Bullish,
            source_range_id: RangeId::new(0, 1),
            source_timeframe: 0,
            created_at_index: 0,
            source_candle_index: 0,
            state: LevelState::Inactive,
            hits: Vec::new(),
            break_event: None,
        };
        index.insert(support);

        index
    }

    #[test]
    fn test_forward_pass_activation() {
        let mut index = create_test_level_index();

        // Candle that activates resistance (body fully above 110)
        let candles = vec![
            make_candle(111.0, 115.0, 110.5, 114.0), // Body: 111-114, above 110
        ];

        let result = forward_pass(&candles, &mut index, 0, 0.1, 0);

        assert!(!result.activated.is_empty());
        assert!(result.activated.contains(&LevelId::new(0, 0)));
    }

    #[test]
    fn test_forward_pass_hit() {
        let mut index = create_test_level_index();

        // First activate the resistance
        let activation_candle = make_candle(111.0, 115.0, 110.5, 114.0);
        check_candle_interactions(&activation_candle, 0, &mut index, 0, 0.1);

        // Now hit the resistance (wick touches 110, body stays above)
        let hit_candle = make_candle(112.0, 115.0, 109.0, 113.0); // Low=109, body: 112-113

        let events = check_candle_interactions(&hit_candle, 1, &mut index, 0, 0.1);

        let hit_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, LevelEvent::Hit { .. }))
            .collect();
        assert!(!hit_events.is_empty());
    }

    #[test]
    fn test_forward_pass_break() {
        let mut index = create_test_level_index();

        // First activate the resistance
        let activation_candle = make_candle(111.0, 115.0, 110.5, 114.0);
        check_candle_interactions(&activation_candle, 0, &mut index, 0, 0.1);

        // Break the resistance (bearish candle with body fully below 110)
        let break_candle = make_candle(109.0, 110.0, 105.0, 106.0); // Bearish, body: 106-109

        let events = check_candle_interactions(&break_candle, 1, &mut index, 0, 0.1);

        let break_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, LevelEvent::Broken { .. }))
            .collect();
        assert!(!break_events.is_empty());

        // Verify the level is now broken in the index
        let level = index.get(LevelId::new(0, 0)).unwrap();
        assert_eq!(level.state, LevelState::Broken);
    }

    #[test]
    fn test_batch_check_interactions() {
        let mut index = create_test_level_index();

        let candles = vec![
            make_candle(111.0, 115.0, 110.5, 114.0), // Activates resistance
            make_candle(112.0, 115.0, 109.0, 113.0), // Hits resistance
            make_candle(109.0, 110.0, 105.0, 106.0), // Breaks resistance
        ];

        let events = batch_check_interactions(&candles, 0, &mut index, 0, 0.1);

        // Should have activation, hit, and break events
        let has_activation = events.iter().any(|e| matches!(e, LevelEvent::Activated { .. }));
        let has_hit = events.iter().any(|e| matches!(e, LevelEvent::Hit { .. }));
        let has_break = events.iter().any(|e| matches!(e, LevelEvent::Broken { .. }));

        assert!(has_activation, "Should have activation event");
        assert!(has_hit, "Should have hit event");
        assert!(has_break, "Should have break event");
    }
}
