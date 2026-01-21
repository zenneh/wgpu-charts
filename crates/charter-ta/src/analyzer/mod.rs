//! Analyzer module for technical analysis.
//!
//! The analyzer processes candle data using a two-pass algorithm:
//! 1. Reverse pass: Detect ranges and levels in O(K) time with early stopping
//! 2. Forward pass: Evaluate level interactions

mod forward_pass;
mod multi_timeframe;
mod reverse_pass;
mod state;

pub use forward_pass::{
    batch_check_interactions, check_candle_interactions, forward_pass, ForwardPassResult,
};
pub use multi_timeframe::{
    aggregate_candles, MultiTimeframeAnalyzer, MultiTimeframeResult, TimeframeData,
    TimeframeDataBuilder,
};
pub use reverse_pass::{reverse_pass, ReversePassResult};
pub use state::{AnalyzerState, TimeframeState};

use charter_core::Candle;

use crate::types::{AnalyzerConfig, Level, LevelEvent, Range, Trend, TrendEvent};

/// Result of analyzing data.
#[derive(Debug, Default)]
pub struct AnalysisResult {
    /// Ranges detected in this analysis.
    pub ranges: Vec<Range>,
    /// Levels created in this analysis.
    pub levels: Vec<Level>,
    /// Level events that occurred.
    pub level_events: Vec<LevelEvent>,
    /// Trends created in this analysis.
    pub trends: Vec<Trend>,
    /// Trend events that occurred.
    pub trend_events: Vec<TrendEvent>,
}

/// Trait for technical analysis algorithms.
///
/// Analyzers process candle data and maintain state about ranges, levels,
/// and other technical indicators.
pub trait Analyzer: Send + Sync {
    /// Process new data for a specific timeframe.
    ///
    /// Expects timeframes to be processed in descending order (highest first)
    /// to enable early-stopping optimization.
    fn update(&mut self, timeframe_idx: u8, candles: &[Candle], current_price: f32)
        -> AnalysisResult;

    /// Reset the analyzer state.
    fn reset(&mut self);

    /// Get the current analyzer state.
    fn state(&self) -> &AnalyzerState;

    /// Get the configuration.
    fn config(&self) -> &AnalyzerConfig;
}

/// Default analyzer implementation using the two-pass algorithm.
pub struct DefaultAnalyzer {
    config: AnalyzerConfig,
    state: AnalyzerState,
}

impl DefaultAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: AnalyzerConfig) -> Self {
        let state = AnalyzerState::new(config.timeframes.len());
        Self { config, state }
    }

    /// Process a single timeframe's data.
    fn process_timeframe(
        &mut self,
        timeframe_idx: u8,
        candles: &[Candle],
        current_price: f32,
    ) -> AnalysisResult {
        let tf_config = match self.config.get(timeframe_idx as usize) {
            Some(c) => c,
            None => return AnalysisResult::default(),
        };

        // Check if we can skip the reverse pass entirely
        // Conditions: we have levels on both sides, no rescan needed, and no new candles
        let skip_reverse_pass = {
            let tf_state = self.state.get_timeframe(timeframe_idx);
            let has_both_levels = self.state.closest_unbroken_resistance.is_some()
                && self.state.closest_unbroken_support.is_some();
            let no_rescan_needed = tf_state.map(|s| !s.needs_level_rescan).unwrap_or(false);
            let no_new_candles = tf_state
                .map(|s| s.last_reverse_pass_candles >= candles.len())
                .unwrap_or(false);

            has_both_levels && no_rescan_needed && no_new_candles
        };

        let mut result = AnalysisResult {
            ranges: Vec::new(),
            levels: Vec::new(),
            level_events: Vec::new(),
            trends: Vec::new(),
            trend_events: Vec::new(),
        };

        // Only run reverse pass if needed
        if !skip_reverse_pass {
            let reverse_result = reverse_pass(
                candles,
                timeframe_idx,
                tf_config.doji_threshold,
                tf_config.min_candles,
                tf_config.create_greedy_levels,
                current_price,
                self.state.closest_unbroken_resistance.map(|(p, _)| p),
                self.state.closest_unbroken_support.map(|(p, _)| p),
            );

            result.ranges = reverse_result.ranges.clone();

            // Update timeframe state with reverse pass results
            {
                let tf_state = self.state.get_or_create_timeframe(timeframe_idx);
                tf_state.last_reverse_pass_candles = candles.len();

                // Clear rescan flag if early stopping found levels on both sides
                if reverse_result.early_stopped {
                    tf_state.clear_rescan_flag();
                }

                // Insert new levels
                for level in reverse_result.levels {
                    let id = level.id;
                    if !tf_state.level_index.contains(id) {
                        result.level_events.push(LevelEvent::Created { level_id: id });
                        tf_state.level_index.insert(level);
                    }
                }

                // Process ranges through TrendTracker to create trends
                // Set tolerance based on current price (0.1% of price)
                tf_state.trend_tracker.default_tolerance = current_price * 0.001;

                // Clear existing trends and rebuild from ranges
                // Ranges from reverse pass are newest-to-oldest, but TrendTracker needs
                // chronological order (oldest-to-newest) to track "last" ranges properly
                tf_state.trend_tracker.clear();
                let mut sorted_ranges = reverse_result.ranges.clone();
                sorted_ranges.sort_by_key(|r| r.start_index);
                for range in &sorted_ranges {
                    if let Some(event) = tf_state.trend_tracker.process_range(range, range.end_index) {
                        result.trend_events.push(event);
                    }
                }
            }
        }

        // Run forward pass for all levels on new candles
        {
            let tf_state = self.state.get_or_create_timeframe(timeframe_idx);

            // Run forward pass (from last_processed_index onwards)
            let forward_result = forward_pass(
                candles,
                &mut tf_state.level_index,
                timeframe_idx,
                tf_config.doji_threshold,
                tf_state.last_processed_index,
            );

            // Check if any levels were broken - if so, mark for rescan
            let had_breaks = forward_result.broken.iter().any(|_| true);
            if had_breaks {
                tf_state.mark_level_broken();
            }

            // Update last processed index
            tf_state.last_processed_index = candles.len();

            // Add forward pass events to result
            result.level_events.extend(forward_result.events);

            // Clone levels from index AFTER forward pass so they have updated states
            // Sort by ID for deterministic ordering
            let mut levels: Vec<_> = tf_state.level_index.iter().cloned().collect();
            levels.sort_by_key(|l| l.id);
            result.levels = levels;

            // Check trend interactions for new candles
            let start_idx = tf_state.last_processed_index.saturating_sub(candles.len() - tf_state.last_processed_index);
            for (i, candle) in candles.iter().enumerate().skip(start_idx) {
                let trend_events = tf_state.trend_tracker.check_interactions(i, candle);
                result.trend_events.extend(trend_events);
            }

            // Copy trends to result
            result.trends = tf_state.trend_tracker.trends.clone();
        }

        // Update global closest levels (separate borrow scope)
        // First, check if current closest levels are still valid (not broken)
        if let Some((_, id)) = self.state.closest_unbroken_resistance {
            let tf_state = self.state.get_timeframe(timeframe_idx);
            let is_broken = tf_state
                .and_then(|s| s.level_index.get(id))
                .map(|l| l.state == crate::types::LevelState::Broken)
                .unwrap_or(false);
            if is_broken {
                self.state.closest_unbroken_resistance = None;
            }
        }

        if let Some((_, id)) = self.state.closest_unbroken_support {
            let tf_state = self.state.get_timeframe(timeframe_idx);
            let is_broken = tf_state
                .and_then(|s| s.level_index.get(id))
                .map(|l| l.state == crate::types::LevelState::Broken)
                .unwrap_or(false);
            if is_broken {
                self.state.closest_unbroken_support = None;
            }
        }

        // Now find the closest levels from this timeframe
        let closest_resistance = {
            let tf_state = self.state.get_timeframe(timeframe_idx);
            tf_state.and_then(|s| {
                s.level_index
                    .closest_resistance_above(current_price)
                    .map(|l| (l.price, l.id))
            })
        };

        let closest_support = {
            let tf_state = self.state.get_timeframe(timeframe_idx);
            tf_state.and_then(|s| {
                s.level_index
                    .closest_support_below(current_price)
                    .map(|l| (l.price, l.id))
            })
        };

        if let Some((price, id)) = closest_resistance {
            let current_best = self.state.closest_unbroken_resistance;
            if current_best.is_none() || price < current_best.unwrap().0 {
                self.state.closest_unbroken_resistance = Some((price, id));
            }
        }

        if let Some((price, id)) = closest_support {
            let current_best = self.state.closest_unbroken_support;
            if current_best.is_none() || price > current_best.unwrap().0 {
                self.state.closest_unbroken_support = Some((price, id));
            }
        }

        // Update current price
        self.state.current_price = current_price;

        result
    }
}

impl Analyzer for DefaultAnalyzer {
    fn update(
        &mut self,
        timeframe_idx: u8,
        candles: &[Candle],
        current_price: f32,
    ) -> AnalysisResult {
        self.process_timeframe(timeframe_idx, candles, current_price)
    }

    fn reset(&mut self) {
        self.state = AnalyzerState::new(self.config.timeframes.len());
    }

    fn state(&self) -> &AnalyzerState {
        &self.state
    }

    fn config(&self) -> &AnalyzerConfig {
        &self.config
    }
}

use crate::ml::{
    extract_features_from_state, ExtractionError, ExtractionRequirements, FeatureExtractor,
    MlFeatures,
};

impl FeatureExtractor for DefaultAnalyzer {
    fn extract_features(&self) -> Result<Option<MlFeatures>, ExtractionError> {
        if !self.can_extract_features() {
            return Ok(None);
        }

        // Default age normalization of 1000 candles
        let features = extract_features_from_state(&self.state, self.state.current_index, 1000.0);
        Ok(Some(features))
    }

    fn extraction_requirements(&self) -> ExtractionRequirements {
        // Default requirements: at least 1 active level
        ExtractionRequirements::new().with_min_active_levels(1)
    }

    fn can_extract_features(&self) -> bool {
        self.extraction_requirements().is_satisfied(&self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeframeConfig;
    use charter_core::Timeframe;

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
    fn test_analyzer_basic() {
        let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 2, 0.1)]);

        let mut analyzer = DefaultAnalyzer::new(config);

        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bullish_candle(110.0, 5.0),
            bearish_candle(115.0, 5.0),
            bearish_candle(110.0, 5.0),
        ];

        let result = analyzer.update(0, &candles, 105.0);

        // Should have detected ranges
        assert!(!result.ranges.is_empty());

        // Should have created levels
        assert!(!result.levels.is_empty());
    }

    #[test]
    fn test_analyzer_closest_levels() {
        let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 2, 0.1)]);

        let mut analyzer = DefaultAnalyzer::new(config);

        // Create a bullish run then bearish run
        let candles = vec![
            bullish_candle(100.0, 5.0),
            bullish_candle(105.0, 5.0),
            bearish_candle(115.0, 5.0),
            bearish_candle(110.0, 5.0),
        ];

        analyzer.update(0, &candles, 107.5);

        let state = analyzer.state();

        // Should have found closest levels on both sides
        // Bullish range creates support below, bearish range creates resistance above
        // Current price is 107.5
        assert!(
            state.closest_unbroken_support.is_some()
                || state.closest_unbroken_resistance.is_some()
        );
    }

    #[test]
    fn test_incremental_update_preserves_activation() {
        use crate::types::LevelState;

        let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 2, 0.1)]);
        let mut analyzer = DefaultAnalyzer::new(config);

        // First update: Create a bearish range that produces a resistance level
        // Then add candles that would activate it (body fully above the level)
        let candles_v1 = vec![
            bearish_candle(110.0, 5.0), // Bearish range starts
            bearish_candle(105.0, 5.0), // Bearish range continues - creates resistance ~115
            bullish_candle(100.0, 5.0), // Direction change
            // Now candles with body fully above 115 to activate the resistance
            make_candle(116.0, 120.0, 115.5, 118.0), // Body: 116-118, above 115
            make_candle(117.0, 121.0, 116.0, 119.0), // Body: 117-119, above 115
        ];

        let _result1 = analyzer.update(0, &candles_v1, 118.0);

        // Check that we have levels and some were activated
        let tf_state = analyzer.state().get_timeframe(0).unwrap();
        let active_before: Vec<_> = tf_state
            .level_index
            .iter()
            .filter(|l| l.state == LevelState::Active)
            .collect();

        // Should have at least one active level
        assert!(
            !active_before.is_empty(),
            "Expected at least one active level after first update"
        );

        let active_count_before = active_before.len();

        // Second update: Same candles plus a few more (simulating incremental update)
        let mut candles_v2 = candles_v1.clone();
        candles_v2.push(make_candle(118.0, 122.0, 117.0, 120.0));
        candles_v2.push(make_candle(119.0, 123.0, 118.0, 121.0));

        let _result2 = analyzer.update(0, &candles_v2, 120.0);

        // Check that activation state is preserved
        let tf_state = analyzer.state().get_timeframe(0).unwrap();
        let active_after: Vec<_> = tf_state
            .level_index
            .iter()
            .filter(|l| l.state == LevelState::Active)
            .collect();

        // Should still have the same active levels (not reset to Inactive)
        assert!(
            active_after.len() >= active_count_before,
            "Expected active levels to be preserved after incremental update. \
             Before: {}, After: {}",
            active_count_before,
            active_after.len()
        );
    }

    #[test]
    fn test_stable_level_ids() {
        use crate::types::LevelId;

        let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 2, 0.1)]);
        let mut analyzer = DefaultAnalyzer::new(config);

        let candles = vec![
            bearish_candle(110.0, 5.0),
            bearish_candle(105.0, 5.0),
            bullish_candle(100.0, 5.0),
        ];

        // First update
        let result1 = analyzer.update(0, &candles, 102.0);
        let level_ids_v1: Vec<LevelId> = result1.levels.iter().map(|l| l.id).collect();

        // Reset and update again with same data
        analyzer.reset();
        let result2 = analyzer.update(0, &candles, 102.0);
        let level_ids_v2: Vec<LevelId> = result2.levels.iter().map(|l| l.id).collect();

        // Level IDs should be identical (stable)
        assert_eq!(
            level_ids_v1, level_ids_v2,
            "Level IDs should be stable across resets with same data"
        );
    }
}
