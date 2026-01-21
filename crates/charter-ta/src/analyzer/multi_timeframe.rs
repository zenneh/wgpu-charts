//! Multi-timeframe analysis orchestration.
//!
//! This module provides tools for analyzing multiple timeframes with proper
//! bound propagation from higher to lower timeframes.

use charter_core::{Candle, Timeframe};

use super::{AnalysisResult, Analyzer, DefaultAnalyzer};
use crate::ml::{extract_features_from_state, TimeframeFeatures};
use crate::types::{AnalyzerConfig, LevelEvent};

// Re-export charter_core's aggregate_candles for convenience
pub use charter_core::aggregate_candles;

/// Data for a single timeframe update.
#[derive(Debug, Clone)]
pub struct TimeframeData<'a> {
    /// Index of this timeframe in the config.
    pub timeframe_idx: u8,
    /// The candles for this timeframe.
    pub candles: &'a [Candle],
    /// Current price.
    pub current_price: f32,
}

/// Result of multi-timeframe analysis.
#[derive(Debug, Default)]
pub struct MultiTimeframeResult {
    /// Results per timeframe (indexed by timeframe_idx).
    pub results: Vec<(u8, AnalysisResult)>,
    /// All level events across timeframes (in processing order).
    pub all_events: Vec<LevelEvent>,
}

impl MultiTimeframeResult {
    /// Get result for a specific timeframe.
    pub fn get(&self, timeframe_idx: u8) -> Option<&AnalysisResult> {
        self.results
            .iter()
            .find(|(idx, _)| *idx == timeframe_idx)
            .map(|(_, r)| r)
    }
}

/// Multi-timeframe analyzer that orchestrates analysis across timeframes.
///
/// Processes timeframes from highest to lowest, allowing bound propagation
/// to enable early-stopping optimization in lower timeframes.
pub struct MultiTimeframeAnalyzer {
    analyzer: DefaultAnalyzer,
    /// Configured timeframes for this analyzer.
    timeframes: Vec<Timeframe>,
    /// Accumulated 1m candles for aggregation.
    candles_1m: Vec<Candle>,
    /// Aggregated candles per timeframe.
    aggregated_candles: Vec<Vec<Candle>>,
    /// Current candle index for feature extraction.
    current_index: usize,
}

impl MultiTimeframeAnalyzer {
    /// Create a new multi-timeframe analyzer with the given timeframes.
    pub fn with_timeframes(timeframes: Vec<Timeframe>, config: AnalyzerConfig) -> Self {
        let num_tf = timeframes.len();
        Self {
            analyzer: DefaultAnalyzer::new(config),
            timeframes,
            candles_1m: Vec::new(),
            aggregated_candles: vec![Vec::new(); num_tf],
            current_index: 0,
        }
    }

    /// Create a new multi-timeframe analyzer from config.
    pub fn new(config: AnalyzerConfig) -> Self {
        let timeframes: Vec<Timeframe> = config.timeframes.iter().map(|tc| tc.timeframe).collect();
        let num_tf = timeframes.len();
        Self {
            analyzer: DefaultAnalyzer::new(config),
            timeframes,
            candles_1m: Vec::new(),
            aggregated_candles: vec![Vec::new(); num_tf],
            current_index: 0,
        }
    }

    /// Process a single 1-minute candle and update all timeframes.
    ///
    /// This method:
    /// 1. Adds the candle to the 1m buffer
    /// 2. Re-aggregates all higher timeframes
    /// 3. Updates the analyzer with all timeframes
    pub fn process_1m_candle(&mut self, candle: &Candle) {
        self.candles_1m.push(candle.clone());
        self.current_index += 1;

        let current_price = candle.close;

        // Re-aggregate candles for each timeframe
        for (tf_idx, &tf) in self.timeframes.iter().enumerate() {
            self.aggregated_candles[tf_idx] = if tf == Timeframe::Min1 {
                self.candles_1m.clone()
            } else {
                aggregate_candles(&self.candles_1m, tf)
            };
        }

        // Update analyzer for each timeframe (highest first for early stopping)
        for (tf_idx, _tf) in self.timeframes.iter().enumerate().rev() {
            let candles = &self.aggregated_candles[tf_idx];
            if !candles.is_empty() {
                self.analyzer.update(tf_idx as u8, candles, current_price);
            }
        }
    }

    /// Extract features for all timeframes at the current state.
    ///
    /// Returns features for all configured timeframes (always the same count),
    /// padding with empty features for timeframes without data.
    pub fn extract_all_features(&self, _current_price: f32) -> Vec<TimeframeFeatures> {
        let state = self.analyzer.state();
        let extracted = extract_features_from_state(state, self.current_index, 1000.0);

        // Create a map of extracted features by timeframe index
        let mut tf_map: std::collections::HashMap<u8, TimeframeFeatures> =
            extracted.timeframes.into_iter()
                .map(|tf| (tf.timeframe_index, tf))
                .collect();

        // Ensure all configured timeframes have features (in order)
        self.timeframes
            .iter()
            .enumerate()
            .map(|(idx, _tf)| {
                tf_map.remove(&(idx as u8)).unwrap_or_else(|| {
                    TimeframeFeatures {
                        timeframe_index: idx as u8,
                        ..Default::default()
                    }
                })
            })
            .collect()
    }

    /// Get TA statistics: (timeframe, level_count, range_count) for each timeframe.
    pub fn ta_counts(&self) -> Vec<(Timeframe, usize, usize)> {
        let state = self.analyzer.state();
        self.timeframes
            .iter()
            .enumerate()
            .map(|(tf_idx, &tf)| {
                if let Some(tf_state) = state.get_timeframe(tf_idx as u8) {
                    (tf, tf_state.level_count(), tf_state.range_count())
                } else {
                    (tf, 0, 0)
                }
            })
            .collect()
    }

    /// Process multiple timeframes in the correct order.
    ///
    /// Timeframes should be provided in descending order (highest first).
    /// This allows bound propagation from higher to lower timeframes.
    pub fn update(&mut self, data: &[TimeframeData<'_>]) -> MultiTimeframeResult {
        let mut result = MultiTimeframeResult::default();

        // Process each timeframe (expects descending order)
        for tf_data in data {
            let tf_result =
                self.analyzer
                    .update(tf_data.timeframe_idx, tf_data.candles, tf_data.current_price);

            // Collect events
            result.all_events.extend(tf_result.level_events.clone());
            result.results.push((tf_data.timeframe_idx, tf_result));
        }

        result
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.analyzer.reset();
    }

    /// Get the underlying analyzer's state.
    pub fn state(&self) -> &super::AnalyzerState {
        self.analyzer.state()
    }

    /// Get the configuration.
    pub fn config(&self) -> &AnalyzerConfig {
        self.analyzer.config()
    }
}

/// Builder for creating aggregated candle data for multiple timeframes.
///
/// This builder takes candles from a base timeframe and aggregates them
/// into higher timeframes for multi-timeframe analysis.
pub struct TimeframeDataBuilder {
    timeframes: Vec<Timeframe>,
}

impl TimeframeDataBuilder {
    /// Create a new builder with the given timeframes (should be in descending order).
    pub fn new(timeframes: Vec<Timeframe>) -> Self {
        Self { timeframes }
    }

    /// Build aggregated candle data for each configured timeframe.
    ///
    /// # Arguments
    ///
    /// * `base_candles` - Candles from the smallest timeframe
    /// * `current_price` - Current price
    ///
    /// # Returns
    ///
    /// Vector of (timeframe_idx, candles, current_price) for each configured timeframe.
    pub fn build(&self, base_candles: &[Candle], current_price: f32) -> Vec<(u8, Vec<Candle>, f32)> {
        self.timeframes
            .iter()
            .enumerate()
            .map(|(idx, tf)| {
                let candles = aggregate_candles(base_candles, *tf);
                (idx as u8, candles, current_price)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeframeConfig;

    fn make_candle(time: f64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Candle {
        Candle::new(time, open, high, low, close, volume)
    }

    fn bullish_candle(time: f64, base: f32, size: f32) -> Candle {
        make_candle(
            time,
            base,
            base + size * 1.2,
            base - size * 0.1,
            base + size,
            100.0,
        )
    }

    fn bearish_candle(time: f64, base: f32, size: f32) -> Candle {
        make_candle(
            time,
            base + size,
            base + size * 1.1,
            base - size * 0.1,
            base,
            100.0,
        )
    }

    #[test]
    fn test_aggregate_candles() {
        // Create 6 one-minute candles
        let base_candles = vec![
            make_candle(0.0, 100.0, 102.0, 99.0, 101.0, 10.0),
            make_candle(60.0, 101.0, 103.0, 100.0, 102.0, 10.0),
            make_candle(120.0, 102.0, 104.0, 101.0, 103.0, 10.0),
            make_candle(180.0, 103.0, 105.0, 102.0, 104.0, 10.0),
            make_candle(240.0, 104.0, 106.0, 103.0, 105.0, 10.0),
            make_candle(300.0, 105.0, 107.0, 104.0, 106.0, 10.0),
        ];

        // Aggregate to 3-minute candles
        let aggregated = aggregate_candles(&base_candles, Timeframe::Min3);

        assert_eq!(aggregated.len(), 2);

        // First 3-min candle should span first 3 candles
        assert_eq!(aggregated[0].open, 100.0);
        assert_eq!(aggregated[0].close, 103.0);
        assert_eq!(aggregated[0].high, 104.0); // Max of 102, 103, 104
        assert_eq!(aggregated[0].low, 99.0); // Min of 99, 100, 101
        assert_eq!(aggregated[0].volume, 30.0); // Sum of volumes
    }

    #[test]
    fn test_multi_timeframe_analyzer() {
        let config = AnalyzerConfig::new(vec![
            TimeframeConfig::new(Timeframe::Hour1, 2, 0.1),
            TimeframeConfig::new(Timeframe::Min30, 2, 0.1),
        ]);

        let mut analyzer = MultiTimeframeAnalyzer::new(config);

        // Create some candles for each timeframe
        let h1_candles = vec![
            bullish_candle(0.0, 100.0, 5.0),
            bullish_candle(3600.0, 105.0, 5.0),
            bearish_candle(7200.0, 115.0, 5.0),
            bearish_candle(10800.0, 110.0, 5.0),
        ];

        let m30_candles = vec![
            bullish_candle(0.0, 100.0, 1.0),
            bullish_candle(1800.0, 101.0, 1.0),
            bullish_candle(3600.0, 102.0, 1.0),
            bullish_candle(5400.0, 103.0, 1.0),
        ];

        let data = vec![
            TimeframeData {
                timeframe_idx: 0,
                candles: &h1_candles,
                current_price: 107.5,
            },
            TimeframeData {
                timeframe_idx: 1,
                candles: &m30_candles,
                current_price: 107.5,
            },
        ];

        let result = analyzer.update(&data);

        // Should have processed both timeframes
        assert_eq!(result.results.len(), 2);

        // Should have some events
        assert!(!result.all_events.is_empty());
    }

    #[test]
    fn test_timeframe_data_builder() {
        let builder = TimeframeDataBuilder::new(vec![Timeframe::Hour1, Timeframe::Min30]);

        // Create minute candles spanning an hour
        let base_candles: Vec<Candle> = (0..60)
            .map(|i| make_candle(i as f64 * 60.0, 100.0, 102.0, 99.0, 101.0, 10.0))
            .collect();

        let data = builder.build(&base_candles, 100.0);

        assert_eq!(data.len(), 2);
        // Hour1 aggregation should produce 1 candle
        assert_eq!(data[0].1.len(), 1);
        // Min30 aggregation should produce 2 candles
        assert_eq!(data[1].1.len(), 2);
    }
}
