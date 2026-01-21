//! Analyzer state types.

use std::collections::HashMap;

use crate::types::{LevelId, LevelIndex, Range, TrendTracker};

/// State for a specific timeframe within the analyzer.
#[derive(Debug)]
pub struct TimeframeState {
    /// Ranges detected in this timeframe.
    pub ranges: Vec<Range>,
    /// Level index for efficient lookups.
    pub level_index: LevelIndex,
    /// Trend tracker for trendline detection and interaction.
    pub trend_tracker: TrendTracker,
    /// Last processed candle index (for forward pass).
    pub last_processed_index: usize,
    /// Number of candles when reverse pass last ran.
    pub last_reverse_pass_candles: usize,
    /// Flag to force a full reverse pass rescan (set when levels break).
    pub needs_level_rescan: bool,
}

impl TimeframeState {
    /// Create a new timeframe state.
    pub fn new(timeframe_idx: u8) -> Self {
        Self {
            ranges: Vec::new(),
            level_index: LevelIndex::new(timeframe_idx),
            trend_tracker: TrendTracker::new(0.0), // Tolerance set during processing
            last_processed_index: 0,
            last_reverse_pass_candles: 0,
            needs_level_rescan: true, // Start with rescan needed
        }
    }

    /// Mark that a level was broken and rescan may be needed.
    pub fn mark_level_broken(&mut self) {
        self.needs_level_rescan = true;
    }

    /// Clear the rescan flag after a successful reverse pass.
    pub fn clear_rescan_flag(&mut self) {
        self.needs_level_rescan = false;
    }

    /// Get the number of ranges.
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Get the number of levels.
    pub fn level_count(&self) -> usize {
        self.level_index.len()
    }

    /// Get the number of active levels.
    pub fn active_level_count(&self) -> usize {
        self.level_index.active_count()
    }
}

/// Global analyzer state across all timeframes.
#[derive(Debug)]
pub struct AnalyzerState {
    /// Current price.
    pub current_price: f32,
    /// Current candle index.
    pub current_index: usize,
    /// Per-timeframe state.
    pub timeframe_states: HashMap<u8, TimeframeState>,
    /// Closest unbroken resistance (price, level_id).
    pub closest_unbroken_resistance: Option<(f32, LevelId)>,
    /// Closest unbroken support (price, level_id).
    pub closest_unbroken_support: Option<(f32, LevelId)>,
    /// Proven resistance bound - no unbroken resistance closer than this.
    pub resistance_bound: Option<f32>,
    /// Proven support bound - no unbroken support closer than this.
    pub support_bound: Option<f32>,
}

impl AnalyzerState {
    /// Create a new analyzer state for the given number of timeframes.
    pub fn new(num_timeframes: usize) -> Self {
        Self {
            current_price: 0.0,
            current_index: 0,
            timeframe_states: HashMap::with_capacity(num_timeframes),
            closest_unbroken_resistance: None,
            closest_unbroken_support: None,
            resistance_bound: None,
            support_bound: None,
        }
    }

    /// Get the state for a specific timeframe.
    pub fn get_timeframe(&self, timeframe_idx: u8) -> Option<&TimeframeState> {
        self.timeframe_states.get(&timeframe_idx)
    }

    /// Get the state for a specific timeframe, creating it if it doesn't exist.
    pub fn get_or_create_timeframe(&mut self, timeframe_idx: u8) -> &mut TimeframeState {
        self.timeframe_states
            .entry(timeframe_idx)
            .or_insert_with(|| TimeframeState::new(timeframe_idx))
    }

    /// Get the total number of ranges across all timeframes.
    pub fn total_ranges(&self) -> usize {
        self.timeframe_states.values().map(|s| s.range_count()).sum()
    }

    /// Get the total number of levels across all timeframes.
    pub fn total_levels(&self) -> usize {
        self.timeframe_states.values().map(|s| s.level_count()).sum()
    }

    /// Get the total number of active levels across all timeframes.
    pub fn total_active_levels(&self) -> usize {
        self.timeframe_states
            .values()
            .map(|s| s.active_level_count())
            .sum()
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.current_price = 0.0;
        self.current_index = 0;
        self.timeframe_states.clear();
        self.closest_unbroken_resistance = None;
        self.closest_unbroken_support = None;
        self.resistance_bound = None;
        self.support_bound = None;
    }
}

impl Default for AnalyzerState {
    fn default() -> Self {
        Self::new(0)
    }
}
