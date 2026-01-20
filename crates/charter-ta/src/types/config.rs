//! Configuration types for the analyzer.

use charter_core::Timeframe;

/// Configuration for a specific timeframe.
#[derive(Debug, Clone)]
pub struct TimeframeConfig {
    /// The timeframe.
    pub timeframe: Timeframe,
    /// Minimum candles required to form a valid range.
    pub min_candles: usize,
    /// Body ratio threshold below which a candle is considered a doji.
    pub doji_threshold: f32,
    /// Whether to create greedy hold levels in addition to regular hold levels.
    pub create_greedy_levels: bool,
}

impl TimeframeConfig {
    /// Create a new timeframe configuration.
    pub fn new(timeframe: Timeframe, min_candles: usize, doji_threshold: f32) -> Self {
        Self {
            timeframe,
            min_candles,
            doji_threshold,
            create_greedy_levels: true,
        }
    }

    /// Create configuration with greedy levels disabled.
    pub fn without_greedy_levels(mut self) -> Self {
        self.create_greedy_levels = false;
        self
    }
}

impl Default for TimeframeConfig {
    fn default() -> Self {
        Self {
            timeframe: Timeframe::Hour1,
            min_candles: 3,
            doji_threshold: 0.001,
            create_greedy_levels: true,
        }
    }
}

/// Global analyzer configuration.
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Per-timeframe configurations.
    /// Should be ordered from highest to lowest timeframe.
    pub timeframes: Vec<TimeframeConfig>,
}

impl AnalyzerConfig {
    /// Create a new analyzer configuration.
    pub fn new(timeframes: Vec<TimeframeConfig>) -> Self {
        Self { timeframes }
    }

    /// Get the configuration for a specific timeframe index.
    pub fn get(&self, idx: usize) -> Option<&TimeframeConfig> {
        self.timeframes.get(idx)
    }

    /// Get the number of configured timeframes.
    pub fn len(&self) -> usize {
        self.timeframes.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.timeframes.is_empty()
    }
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            timeframes: vec![
                TimeframeConfig::new(Timeframe::Day1, 3, 0.001),
                TimeframeConfig::new(Timeframe::Hour3, 3, 0.001),
                TimeframeConfig::new(Timeframe::Hour1, 3, 0.001),
                TimeframeConfig::new(Timeframe::Min30, 3, 0.001),
                TimeframeConfig::new(Timeframe::Min5, 3, 0.001),
            ],
        }
    }
}
