//! Fast incremental multi-timeframe analysis.
//!
//! This module provides efficient incremental technical analysis across multiple
//! timeframes. Instead of processing all data upfront, it processes 1-minute
//! candles one at a time and aggregates them to higher timeframes on-the-fly.

use charter_core::{Candle, Timeframe};

use crate::analyzer::{Analyzer, AnalyzerConfig};
use crate::ml::TimeframeFeatures;
use crate::types::{Level, Trend};

/// Incrementally aggregates 1-minute candles into a higher timeframe.
#[derive(Debug)]
pub struct IncrementalAggregator {
    timeframe: Timeframe,
    interval_secs: f64,
    current_bucket: Option<Candle>,
    current_bucket_start: f64,
    completed_candles: usize,
}

impl IncrementalAggregator {
    /// Create a new aggregator for the given timeframe.
    pub fn new(timeframe: Timeframe) -> Self {
        Self {
            timeframe,
            interval_secs: timeframe.seconds(),
            current_bucket: None,
            current_bucket_start: 0.0,
            completed_candles: 0,
        }
    }

    /// Process a 1-minute candle.
    /// Returns Some(candle) if a higher-timeframe candle was completed.
    #[inline]
    pub fn process(&mut self, candle: &Candle) -> Option<Candle> {
        let bucket_start = (candle.timestamp / self.interval_secs).floor() * self.interval_secs;

        if let Some(ref mut agg) = self.current_bucket {
            if bucket_start == self.current_bucket_start {
                // Same bucket - update OHLCV
                agg.high = agg.high.max(candle.high);
                agg.low = agg.low.min(candle.low);
                agg.close = candle.close;
                agg.volume += candle.volume;
                None
            } else {
                // New bucket - emit completed candle and start new
                let completed = *agg;
                self.completed_candles += 1;
                self.current_bucket = Some(Candle::new(
                    bucket_start,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ));
                self.current_bucket_start = bucket_start;
                Some(completed)
            }
        } else {
            // First candle
            self.current_bucket = Some(Candle::new(
                bucket_start,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ));
            self.current_bucket_start = bucket_start;
            None
        }
    }

    /// Get the current incomplete candle if any.
    pub fn current_candle(&self) -> Option<&Candle> {
        self.current_bucket.as_ref()
    }

    /// Number of completed candles.
    pub fn completed_count(&self) -> usize {
        self.completed_candles
    }

    /// Get the timeframe.
    pub fn timeframe(&self) -> Timeframe {
        self.timeframe
    }
}

/// Single timeframe analyzer with incremental aggregation.
pub struct TimeframeAnalyzerState {
    /// The timeframe this analyzer handles.
    pub timeframe: Timeframe,
    /// Candle aggregator (None for 1m since no aggregation needed).
    aggregator: Option<IncrementalAggregator>,
    /// The technical analyzer.
    analyzer: Analyzer,
    /// Number of candles processed at this timeframe.
    pub candle_count: usize,
}

impl TimeframeAnalyzerState {
    /// Create a new state for the given timeframe.
    pub fn new(timeframe: Timeframe, config: AnalyzerConfig) -> Self {
        let aggregator = if timeframe == Timeframe::Min1 {
            None // No aggregation needed for 1m
        } else {
            Some(IncrementalAggregator::new(timeframe))
        };

        Self {
            timeframe,
            aggregator,
            analyzer: Analyzer::with_config(config),
            candle_count: 0,
        }
    }

    /// Process a 1-minute candle.
    /// For 1m timeframe, processes directly.
    /// For higher timeframes, aggregates and processes when bucket completes.
    #[inline]
    pub fn process_1m_candle(&mut self, candle: &Candle) {
        if let Some(ref mut aggregator) = self.aggregator {
            // Higher timeframe - aggregate first
            if let Some(agg_candle) = aggregator.process(candle) {
                self.analyzer.process_candle(agg_candle);
                self.candle_count += 1;
            }
        } else {
            // 1m timeframe - process directly
            self.analyzer.process_candle(*candle);
            self.candle_count += 1;
        }
    }

    /// Get all active levels.
    pub fn active_levels(&self) -> impl Iterator<Item = &Level> {
        self.analyzer.active_levels()
    }

    /// Get all levels (including broken).
    pub fn all_levels(&self) -> &[Level] {
        self.analyzer.all_levels()
    }

    /// Get all active trends.
    pub fn active_trends(&self) -> impl Iterator<Item = &Trend> {
        self.analyzer.active_trends()
    }

    /// Get all trends (including broken).
    pub fn all_trends(&self) -> &[Trend] {
        self.analyzer.all_trends()
    }

    /// Extract features for ML at the current state.
    /// Uses fast single-pass extraction over active levels/trends only.
    pub fn extract_features(&self, tf_idx: usize, current_price: f32) -> TimeframeFeatures {
        TimeframeFeatures::extract_fast(
            tf_idx,
            self.analyzer.active_levels(),
            self.analyzer.active_trends(),
            current_price,
            self.candle_count.saturating_sub(1),
        )
    }

    /// Get the analyzer's candle count.
    pub fn analyzer_candle_count(&self) -> usize {
        self.analyzer.candle_count()
    }
}

/// Multi-timeframe analyzer that processes 1m candles and maintains
/// incremental TA state for all timeframes.
pub struct MultiTimeframeAnalyzer {
    /// Analyzers for each timeframe, indexed by their position.
    states: Vec<TimeframeAnalyzerState>,
    /// The timeframes being analyzed.
    timeframes: Vec<Timeframe>,
    /// Total 1m candles processed.
    total_1m_candles: usize,
}

impl MultiTimeframeAnalyzer {
    /// Create a new multi-timeframe analyzer with the given timeframes.
    ///
    /// All timeframes will use the same analyzer config.
    pub fn new(timeframes: Vec<Timeframe>, config: AnalyzerConfig) -> Self {
        let states = timeframes
            .iter()
            .map(|&tf| TimeframeAnalyzerState::new(tf, config.clone()))
            .collect();

        Self {
            states,
            timeframes,
            total_1m_candles: 0,
        }
    }

    /// Create with default config for all timeframes.
    pub fn with_timeframes(timeframes: Vec<Timeframe>) -> Self {
        Self::new(timeframes, AnalyzerConfig::default())
    }

    /// Process a single 1-minute candle through all timeframes.
    #[inline]
    pub fn process_1m_candle(&mut self, candle: &Candle) {
        for state in &mut self.states {
            state.process_1m_candle(candle);
        }
        self.total_1m_candles += 1;
    }

    /// Process multiple 1-minute candles.
    pub fn process_1m_candles(&mut self, candles: &[Candle]) {
        for candle in candles {
            self.process_1m_candle(candle);
        }
    }

    /// Get the number of 1m candles processed.
    pub fn total_1m_candles(&self) -> usize {
        self.total_1m_candles
    }

    /// Get the timeframes being analyzed.
    pub fn timeframes(&self) -> &[Timeframe] {
        &self.timeframes
    }

    /// Get the state for a specific timeframe by index.
    pub fn state(&self, idx: usize) -> Option<&TimeframeAnalyzerState> {
        self.states.get(idx)
    }

    /// Get mutable state for a specific timeframe by index.
    pub fn state_mut(&mut self, idx: usize) -> Option<&mut TimeframeAnalyzerState> {
        self.states.get_mut(idx)
    }

    /// Get all states.
    pub fn states(&self) -> &[TimeframeAnalyzerState] {
        &self.states
    }

    /// Extract ML features for all timeframes at current state.
    pub fn extract_all_features(&self, current_price: f32) -> Vec<TimeframeFeatures> {
        self.states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.candle_count >= 10) // Skip timeframes without enough data
            .map(|(idx, state)| state.extract_features(idx, current_price))
            .collect()
    }

    /// Get candle counts for all timeframes (for diagnostics).
    pub fn candle_counts(&self) -> Vec<(Timeframe, usize)> {
        self.states
            .iter()
            .map(|s| (s.timeframe, s.candle_count))
            .collect()
    }

    /// Get level/trend counts for all timeframes (for diagnostics).
    pub fn ta_counts(&self) -> Vec<(Timeframe, usize, usize)> {
        self.states
            .iter()
            .map(|s| {
                (
                    s.timeframe,
                    s.all_levels().len(),
                    s.all_trends().len(),
                )
            })
            .collect()
    }
}

/// Default ML export timeframes: 1m, 5m, 30m, 1h, 1d
pub fn ml_export_timeframes() -> Vec<Timeframe> {
    vec![
        Timeframe::Min1,
        Timeframe::Min5,
        Timeframe::Min30,
        Timeframe::Hour1,
        Timeframe::Day1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(timestamp: f64, open: f32, close: f32) -> Candle {
        let (high, low) = if close > open {
            (close + 1.0, open - 1.0)
        } else {
            (open + 1.0, close - 1.0)
        };
        Candle::new(timestamp, open, high, low, close, 100.0)
    }

    #[test]
    fn test_incremental_aggregator() {
        let mut agg = IncrementalAggregator::new(Timeframe::Min5);

        // Process 5 one-minute candles (300 seconds = 5 minutes)
        for i in 0..5 {
            let ts = i as f64 * 60.0; // 0, 60, 120, 180, 240
            let result = agg.process(&make_candle(ts, 100.0 + i as f32, 101.0 + i as f32));

            // No completion until we hit the next bucket
            if i < 4 {
                assert!(result.is_none());
            }
        }

        // Process candle in next bucket (at 300 seconds)
        let result = agg.process(&make_candle(300.0, 105.0, 106.0));
        assert!(result.is_some());

        let completed = result.unwrap();
        assert_eq!(completed.timestamp, 0.0); // First bucket starts at 0
        assert_eq!(completed.open, 100.0); // First candle's open
        assert_eq!(completed.close, 104.0); // Last candle's close
    }

    #[test]
    fn test_multi_timeframe_analyzer() {
        let timeframes = vec![Timeframe::Min1, Timeframe::Min5];
        let mut mta = MultiTimeframeAnalyzer::with_timeframes(timeframes);

        // Process 10 one-minute candles
        for i in 0..10 {
            let ts = i as f64 * 60.0;
            mta.process_1m_candle(&make_candle(ts, 100.0, 101.0));
        }

        assert_eq!(mta.total_1m_candles(), 10);

        // 1m should have 10 candles
        assert_eq!(mta.state(0).unwrap().candle_count, 10);

        // 5m should have 1 complete candle (candles 0-4) + in-progress
        // Actually 2 because at ts=300 (candle 5), bucket 0 completes
        // Wait, let me think about this:
        // Candles at ts: 0, 60, 120, 180, 240 -> bucket 0 (0-299)
        // Candles at ts: 300, 360, 420, 480, 540 -> bucket 1 (300-599)
        // When we process ts=300, bucket 0 completes
        assert_eq!(mta.state(1).unwrap().candle_count, 1);
    }
}
