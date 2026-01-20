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

use crate::types::{AnalyzerConfig, Level, LevelEvent, Range};

/// Result of analyzing data.
#[derive(Debug, Default)]
pub struct AnalysisResult {
    /// Ranges detected in this analysis.
    pub ranges: Vec<Range>,
    /// Levels created in this analysis.
    pub levels: Vec<Level>,
    /// Level events that occurred.
    pub level_events: Vec<LevelEvent>,
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

        // Perform reverse pass to detect ranges and create levels
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

        let mut result = AnalysisResult {
            ranges: reverse_result.ranges,
            levels: Vec::new(),
            level_events: Vec::new(),
        };

        // Insert levels into the index and run forward pass
        {
            let tf_state = self.state.get_or_create_timeframe(timeframe_idx);
            for level in reverse_result.levels {
                let id = level.id;
                result.level_events.push(LevelEvent::Created { level_id: id });
                result.levels.push(level.clone());
                tf_state.level_index.insert(level);
            }

            // Run forward pass to check level interactions
            let forward_result = forward_pass(
                candles,
                &mut tf_state.level_index,
                timeframe_idx,
                tf_config.doji_threshold,
                tf_state.last_processed_index,
            );

            // Update last processed index
            tf_state.last_processed_index = candles.len();

            // Add forward pass events to result
            result.level_events.extend(forward_result.events);
        }

        // Update global closest levels (separate borrow scope)
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
}
