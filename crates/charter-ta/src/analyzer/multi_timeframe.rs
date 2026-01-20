//! Multi-timeframe analysis orchestration.
//!
//! This module provides tools for analyzing multiple timeframes with proper
//! bound propagation from higher to lower timeframes.

use charter_core::{Candle, Timeframe};

use super::{AnalysisResult, Analyzer, DefaultAnalyzer};
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
}

impl MultiTimeframeAnalyzer {
    /// Create a new multi-timeframe analyzer.
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            analyzer: DefaultAnalyzer::new(config),
        }
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
