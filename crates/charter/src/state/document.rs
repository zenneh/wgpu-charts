//! Persistent document data.
//!
//! This module contains the state that represents the "document" being viewed:
//! candle data for all timeframes, technical analysis results, and indicators.

use crate::indicators::IndicatorRegistry;
use crate::replay::TimeframeTaData;
use charter_render::TimeframeData;

/// Settings for TA display filtering.
///
/// Controls which TA elements are visible on the chart.
#[derive(Debug, Clone)]
pub struct TaDisplaySettings {
    /// Master toggle for all TA display.
    pub show_ta: bool,
    /// Show range markers.
    pub show_ranges: bool,
    /// Show hold-type levels.
    pub show_hold_levels: bool,
    /// Show greedy-type levels.
    pub show_greedy_levels: bool,
    /// Show levels in active state.
    pub show_active_levels: bool,
    /// Show levels that have been hit.
    pub show_hit_levels: bool,
    /// Show broken levels.
    pub show_broken_levels: bool,
    /// Show trend lines.
    pub show_trends: bool,
    /// Show active trends.
    pub show_active_trends: bool,
    /// Show hit trends.
    pub show_hit_trends: bool,
    /// Show broken trends.
    pub show_broken_trends: bool,
}

impl Default for TaDisplaySettings {
    fn default() -> Self {
        Self {
            show_ta: false,
            show_ranges: true,
            show_hold_levels: true,
            show_greedy_levels: false,
            show_active_levels: true,
            show_hit_levels: true,
            show_broken_levels: false,
            show_trends: true,
            show_active_trends: true,
            show_hit_trends: true,
            show_broken_trends: false,
        }
    }
}

impl TaDisplaySettings {
    /// Create settings from application config.
    pub fn from_config(config: &charter_config::TaDisplayConfig) -> Self {
        Self {
            show_ta: config.show_ta,
            show_ranges: config.show_ranges,
            show_hold_levels: config.show_hold_levels,
            show_greedy_levels: config.show_greedy_levels,
            show_active_levels: config.show_active_levels,
            show_hit_levels: config.show_hit_levels,
            show_broken_levels: config.show_broken_levels,
            show_trends: config.show_trends,
            show_active_trends: config.show_active_trends,
            show_hit_trends: config.show_hit_trends,
            show_broken_trends: config.show_broken_trends,
        }
    }
}

/// Document state containing all persistent chart data.
///
/// This includes:
/// - Candle data for all timeframes (with GPU buffers)
/// - Technical analysis results
/// - Indicator instances
/// - Currently selected timeframe
pub struct DocumentState {
    /// Timeframe data for all supported timeframes.
    ///
    /// Each entry contains candle data and associated GPU buffers.
    /// Index corresponds to Timeframe::all() order.
    pub timeframes: Vec<TimeframeData>,

    /// Currently selected timeframe index.
    pub current_timeframe: usize,

    /// Technical analysis data for each timeframe.
    ///
    /// Contains ranges, levels, and trends computed by the TA analyzer.
    pub ta_data: Vec<TimeframeTaData>,

    /// TA display settings (filtering).
    pub ta_settings: TaDisplaySettings,

    /// Registry of technical indicators (MACD, etc.).
    pub indicators: IndicatorRegistry,

    /// Index of currently hovered range (if any).
    pub hovered_range: Option<usize>,

    /// Index of currently hovered level (if any).
    pub hovered_level: Option<usize>,
}

impl DocumentState {
    /// Create a new document state with empty data.
    pub fn new(num_timeframes: usize) -> Self {
        let ta_data: Vec<TimeframeTaData> = (0..num_timeframes)
            .map(|_| TimeframeTaData::default())
            .collect();

        Self {
            timeframes: Vec::new(),
            current_timeframe: 0,
            ta_data,
            ta_settings: TaDisplaySettings::default(),
            indicators: IndicatorRegistry::new(),
            hovered_range: None,
            hovered_level: None,
        }
    }

    /// Get the current timeframe's candle data.
    pub fn current_candles(&self) -> &[charter_core::Candle] {
        self.timeframes
            .get(self.current_timeframe)
            .map(|tf| tf.candles.as_slice())
            .unwrap_or(&[])
    }

    /// Get the current timeframe's TA data.
    pub fn current_ta(&self) -> Option<&TimeframeTaData> {
        self.ta_data.get(self.current_timeframe)
    }

    /// Get mutable reference to current timeframe's TA data.
    pub fn current_ta_mut(&mut self) -> Option<&mut TimeframeTaData> {
        self.ta_data.get_mut(self.current_timeframe)
    }

    /// Get timeframe data by index.
    pub fn get_timeframe(&self, index: usize) -> Option<&TimeframeData> {
        self.timeframes.get(index)
    }

    /// Get mutable timeframe data by index.
    pub fn get_timeframe_mut(&mut self, index: usize) -> Option<&mut TimeframeData> {
        self.timeframes.get_mut(index)
    }

    /// Set timeframe data at the given index.
    ///
    /// Returns an error if the index is out of bounds.
    pub fn set_timeframe(&mut self, index: usize, data: TimeframeData) -> Result<(), String> {
        if index >= self.timeframes.len() {
            return Err(format!(
                "Invalid timeframe index: {} (max: {})",
                index,
                self.timeframes.len().saturating_sub(1)
            ));
        }
        self.timeframes[index] = data;
        Ok(())
    }

    /// Clear TA data for all timeframes.
    pub fn clear_ta_data(&mut self) {
        for ta in &mut self.ta_data {
            ta.ranges.clear();
            ta.levels.clear();
            ta.computed = false;
        }
    }

    /// Check if TA has been computed for the current timeframe.
    pub fn is_ta_computed(&self) -> bool {
        self.ta_data
            .get(self.current_timeframe)
            .map(|ta| ta.computed)
            .unwrap_or(false)
    }

    /// Get the number of candles in the current timeframe.
    pub fn candle_count(&self) -> usize {
        self.current_candles().len()
    }

    /// Check if there's any data loaded.
    pub fn has_data(&self) -> bool {
        self.timeframes
            .get(0)
            .map(|tf| !tf.candles.is_empty())
            .unwrap_or(false)
    }
}
