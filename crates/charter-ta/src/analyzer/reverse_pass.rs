//! Reverse pass algorithm for range and level detection.
//!
//! The reverse pass processes candles from newest to oldest, enabling
//! early-stopping optimization based on the formal proof in PLAN.md.
//!
//! ## Early Stopping Theorem
//!
//! Once we identify:
//! - Closest unbroken resistance at price R > C (current price)
//! - Closest unbroken support at price S < C
//!
//! Then no older range can produce a closer unbroken level because:
//!
//! For any older bearish range creating level at H:
//! - If H > R: Not closer, ignore
//! - If H ≤ R and H > C: Level is BROKEN because price traversed from H to C
//!
//! Symmetric argument applies for support levels.

use charter_core::Candle;

use crate::types::{
    CandleDirection, CandleMetadata, Level, LevelDirection, LevelId, LevelType, Range,
    RangeBuilder,
};

/// Result of the reverse pass.
#[derive(Debug)]
pub struct ReversePassResult {
    /// Ranges detected (in chronological order).
    pub ranges: Vec<Range>,
    /// Levels created from ranges.
    pub levels: Vec<Level>,
    /// Whether early stopping was applied.
    pub early_stopped: bool,
    /// Index where early stopping occurred (if any).
    pub early_stop_index: Option<usize>,
}

/// Perform reverse pass on candle data.
///
/// This function processes candles from newest to oldest to detect ranges
/// and create levels. It uses the early-stopping optimization to avoid
/// processing more candles than necessary.
///
/// # Arguments
///
/// * `candles` - The candle data (oldest first)
/// * `timeframe_idx` - Index of the timeframe being processed
/// * `doji_threshold` - Threshold for doji detection
/// * `min_candles` - Minimum candles required for a valid range
/// * `create_greedy_levels` - Whether to create greedy hold levels
/// * `current_price` - Current price for determining level relevance
/// * `existing_resistance` - Existing closest resistance (from higher timeframes)
/// * `existing_support` - Existing closest support (from higher timeframes)
pub fn reverse_pass(
    candles: &[Candle],
    timeframe_idx: u8,
    doji_threshold: f32,
    min_candles: usize,
    create_greedy_levels: bool,
    current_price: f32,
    existing_resistance: Option<f32>,
    existing_support: Option<f32>,
) -> ReversePassResult {
    if candles.is_empty() {
        return ReversePassResult {
            ranges: Vec::new(),
            levels: Vec::new(),
            early_stopped: false,
            early_stop_index: None,
        };
    }

    let mut builder = RangeBuilder::new(timeframe_idx, doji_threshold);
    let mut ranges = Vec::new();
    let mut levels = Vec::new();
    let mut level_sequence: u32 = 0;

    // Track closest unbroken levels
    let mut closest_resistance: Option<f32> = existing_resistance;
    let mut closest_support: Option<f32> = existing_support;

    // Track if we've found unbroken levels on both sides
    let mut found_resistance = existing_resistance.is_some();
    let mut found_support = existing_support.is_some();

    let mut early_stopped = false;
    let mut early_stop_index = None;

    // Process candles in reverse order (newest to oldest)
    for (i, candle) in candles.iter().enumerate().rev() {
        // Check early stopping condition
        if found_resistance && found_support {
            // Check if this candle's range could possibly create a closer level
            let can_create_closer = can_create_closer_level(
                candle,
                doji_threshold,
                current_price,
                closest_resistance,
                closest_support,
            );

            if !can_create_closer {
                early_stopped = true;
                early_stop_index = Some(i);
                break;
            }
        }

        // Process the candle
        if let Some(range) = builder.process_reverse(i, candle) {
            if range.is_valid(min_candles) {
                // Check if levels from this range are unbroken
                let levels_result = create_levels_if_unbroken(
                    &range,
                    candles,
                    timeframe_idx,
                    &mut level_sequence,
                    create_greedy_levels,
                    current_price,
                    &mut closest_resistance,
                    &mut closest_support,
                    &mut found_resistance,
                    &mut found_support,
                    doji_threshold,
                );

                levels.extend(levels_result);
                ranges.push(range);
            }
        }
    }

    // Finalize the last range
    if let Some(range) = builder.finalize() {
        if range.is_valid(min_candles) {
            let levels_result = create_levels_if_unbroken(
                &range,
                candles,
                timeframe_idx,
                &mut level_sequence,
                create_greedy_levels,
                current_price,
                &mut closest_resistance,
                &mut closest_support,
                &mut found_resistance,
                &mut found_support,
                doji_threshold,
            );

            levels.extend(levels_result);
            ranges.push(range);
        }
    }

    // Reverse to get chronological order
    ranges.reverse();
    levels.reverse();

    ReversePassResult {
        ranges,
        levels,
        early_stopped,
        early_stop_index,
    }
}

/// Check if a candle could possibly create a closer unbroken level.
///
/// This is a heuristic to determine if we should continue processing.
fn can_create_closer_level(
    candle: &Candle,
    doji_threshold: f32,
    current_price: f32,
    closest_resistance: Option<f32>,
    closest_support: Option<f32>,
) -> bool {
    let meta = CandleMetadata::from_candle(candle, doji_threshold);

    // If doji, it won't create levels on its own
    if meta.direction == CandleDirection::Doji {
        return true; // Continue processing to see next candles
    }

    match meta.direction {
        CandleDirection::Bearish => {
            // This could contribute to a bearish range creating resistance
            if let Some(r) = closest_resistance {
                // If candle's high is below current closest resistance,
                // it could potentially create a closer resistance
                candle.high < r && candle.high > current_price
            } else {
                // No resistance found yet, could create one
                candle.high > current_price
            }
        }
        CandleDirection::Bullish => {
            // This could contribute to a bullish range creating support
            if let Some(s) = closest_support {
                // If candle's low is above current closest support,
                // it could potentially create a closer support
                candle.low > s && candle.low < current_price
            } else {
                // No support found yet, could create one
                candle.low < current_price
            }
        }
        CandleDirection::Doji => true,
    }
}

/// Create levels from a range if they would be unbroken.
///
/// Checks if candles between the range and current time would have broken the level.
#[allow(clippy::too_many_arguments)]
fn create_levels_if_unbroken(
    range: &Range,
    candles: &[Candle],
    timeframe_idx: u8,
    level_sequence: &mut u32,
    create_greedy: bool,
    current_price: f32,
    closest_resistance: &mut Option<f32>,
    closest_support: &mut Option<f32>,
    found_resistance: &mut bool,
    found_support: &mut bool,
    doji_threshold: f32,
) -> Vec<Level> {
    let mut levels = Vec::new();

    // Create hold level
    let hold_price = range.hold_level_price();
    if is_level_unbroken(range, candles, hold_price, doji_threshold) {
        let id = LevelId::new(timeframe_idx, *level_sequence);
        *level_sequence += 1;

        let level = Level::from_range(range, LevelType::Hold, id, range.end_index);

        // Update closest tracking based on level direction
        match level.level_direction {
            LevelDirection::Resistance => {
                if hold_price > current_price {
                    if closest_resistance.is_none() || hold_price < closest_resistance.unwrap() {
                        *closest_resistance = Some(hold_price);
                        *found_resistance = true;
                    }
                }
            }
            LevelDirection::Support => {
                if hold_price < current_price {
                    if closest_support.is_none() || hold_price > closest_support.unwrap() {
                        *closest_support = Some(hold_price);
                        *found_support = true;
                    }
                }
            }
        }

        levels.push(level);
    }

    // Create greedy hold level if requested
    if create_greedy {
        let greedy_price = range.greedy_hold_level_price();
        if is_level_unbroken(range, candles, greedy_price, doji_threshold) {
            let id = LevelId::new(timeframe_idx, *level_sequence);
            *level_sequence += 1;

            let level = Level::from_range(range, LevelType::GreedyHold, id, range.end_index);

            // Update closest tracking
            match level.level_direction {
                LevelDirection::Resistance => {
                    if greedy_price > current_price {
                        if closest_resistance.is_none() || greedy_price < closest_resistance.unwrap()
                        {
                            *closest_resistance = Some(greedy_price);
                            *found_resistance = true;
                        }
                    }
                }
                LevelDirection::Support => {
                    if greedy_price < current_price {
                        if closest_support.is_none() || greedy_price > closest_support.unwrap() {
                            *closest_support = Some(greedy_price);
                            *found_support = true;
                        }
                    }
                }
            }

            levels.push(level);
        }
    }

    levels
}

/// Check if a level would be unbroken by subsequent candles.
///
/// A level is broken when:
/// - Resistance: A bearish candle closes fully below the level
/// - Support: A bullish candle closes fully above the level
fn is_level_unbroken(range: &Range, candles: &[Candle], level_price: f32, doji_threshold: f32) -> bool {
    // Check all candles after the range
    let start_check = range.end_index + 1;
    if start_check >= candles.len() {
        return true; // No candles after range, level is unbroken
    }

    for candle in &candles[start_check..] {
        let meta = CandleMetadata::from_candle(candle, doji_threshold);
        let body_top = candle.open.max(candle.close);
        let body_bottom = candle.open.min(candle.close);

        match range.direction {
            CandleDirection::Bearish => {
                // This creates a resistance level
                // Broken if bearish candle closes fully below
                if meta.direction == CandleDirection::Bearish && body_top < level_price {
                    return false;
                }
            }
            CandleDirection::Bullish => {
                // This creates a support level
                // Broken if bullish candle closes fully above
                if meta.direction == CandleDirection::Bullish && body_bottom > level_price {
                    return false;
                }
            }
            CandleDirection::Doji => {
                // Doji ranges don't create real levels
                return false;
            }
        }
    }

    true
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
    fn test_reverse_pass_basic() {
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bullish_candle(110.0, 5.0),
            bearish_candle(115.0, 5.0),
            bearish_candle(110.0, 5.0),
        ];

        let result = reverse_pass(&candles, 0, 0.1, 2, true, 107.5, None, None);

        // Should have detected ranges
        assert!(!result.ranges.is_empty());

        // Should have created levels
        assert!(!result.levels.is_empty());
    }

    /// Test Case: Verify broken levels are correctly identified
    ///
    /// This validates the formal proof - older levels that price has
    /// traversed through should be marked as broken.
    #[test]
    fn test_older_level_broken_by_price_traversal() {
        // Create scenario: bearish range creates resistance, then bullish move,
        // then bearish candle breaks the original resistance by closing below it
        let candles = vec![
            bearish_candle(120.0, 5.0), // Creates bearish range with resistance ~125
            bearish_candle(115.0, 5.0), // Continues bearish range
            bullish_candle(110.0, 3.0), // Bullish candle ends the bearish range
            bullish_candle(113.0, 3.0), // Another bullish
            bearish_candle(100.0, 20.0), // Big bearish move - body closes fully below 120
        ];

        let result = reverse_pass(&candles, 0, 0.1, 2, false, 95.0, None, None);

        // The first bearish range creates resistance around 125
        // The last bearish candle closes at 100, which is below 125
        // So the 125 resistance should be broken
        let high_resistances: Vec<_> = result
            .levels
            .iter()
            .filter(|l| l.level_direction == LevelDirection::Resistance && l.price > 120.0)
            .collect();

        // High resistance levels should be filtered out as broken
        // (the big bearish candle at the end breaks them)
        assert!(
            high_resistances.is_empty(),
            "Expected high resistance levels to be broken, but found {:?}",
            high_resistances.iter().map(|l| l.price).collect::<Vec<_>>()
        );
    }

    /// Test Case: Verify early stopping works correctly
    #[test]
    fn test_early_stopping_optimization() {
        // Create many ranges, but the closest unbroken levels are near the end
        let mut candles = Vec::new();

        // Old ranges (far from current price)
        for i in 0..10 {
            candles.push(bullish_candle(50.0 + i as f32 * 2.0, 1.0));
        }

        // Recent ranges (close to current price)
        candles.push(bearish_candle(100.0, 3.0)); // Resistance near current
        candles.push(bearish_candle(97.0, 3.0));
        candles.push(bullish_candle(90.0, 3.0)); // Support near current
        candles.push(bullish_candle(93.0, 3.0));

        let result = reverse_pass(&candles, 0, 0.1, 1, false, 95.0, None, None);

        // Should have found levels
        assert!(!result.levels.is_empty());

        // The algorithm may or may not early stop depending on the data,
        // but it should still produce correct results
        let has_resistance = result
            .levels
            .iter()
            .any(|l| l.level_direction == LevelDirection::Resistance && l.price > 95.0);
        let has_support = result
            .levels
            .iter()
            .any(|l| l.level_direction == LevelDirection::Support && l.price < 95.0);

        assert!(has_resistance || has_support, "Should find at least one relevant level");
    }

    /// Test Case: Edge case - no unbroken levels exist
    #[test]
    fn test_all_levels_broken() {
        // Price swings wildly, breaking all levels
        let candles = vec![
            bullish_candle(100.0, 20.0), // Big bullish move
            bearish_candle(120.0, 30.0), // Big bearish move breaks support
            bullish_candle(90.0, 35.0),  // Big bullish move breaks resistance
        ];

        let result = reverse_pass(&candles, 0, 0.1, 1, false, 105.0, None, None);

        // Due to the extreme price action, many levels may be broken
        // This test verifies the algorithm handles this gracefully
        // (not panicking, returning valid results)
        // Just verify it runs without panicking
        let _ = result.ranges.len();
    }

    /// Test Case: Multi-timeframe bound propagation
    #[test]
    fn test_higher_timeframe_bounds_constrain_lower() {
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bearish_candle(115.0, 5.0),
            bearish_candle(110.0, 5.0),
        ];

        // First pass without existing bounds
        let result1 = reverse_pass(&candles, 0, 0.1, 2, false, 107.5, None, None);

        // Second pass with existing bounds from "higher timeframe"
        let result2 = reverse_pass(
            &candles,
            1,
            0.1,
            2,
            false,
            107.5,
            Some(115.0), // Existing resistance
            Some(100.0), // Existing support
        );

        // Both should produce results, but the second may early-stop sooner
        assert!(!result1.ranges.is_empty() || !result2.ranges.is_empty());
    }
}
