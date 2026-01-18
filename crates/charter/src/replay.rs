//! Replay mode management for historical chart playback.
//!
//! The ReplayManager handles stepping through historical candle data,
//! allowing users to replay market history at configurable step sizes.

use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_render::TimeframeData;
use charter_ta::{Level, Range, Trend};

/// TA data computed for a single timeframe.
#[derive(Debug, Clone)]
pub struct TimeframeTaData {
    pub ranges: Vec<Range>,
    pub levels: Vec<Level>,
    pub trends: Vec<Trend>,
    pub computed: bool,
}

impl TimeframeTaData {
    /// Creates a new empty TimeframeTaData.
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
            levels: Vec::new(),
            trends: Vec::new(),
            computed: false,
        }
    }

    /// Creates a new computed TimeframeTaData with the given data.
    pub fn with_data(ranges: Vec<Range>, levels: Vec<Level>, trends: Vec<Trend>) -> Self {
        Self {
            ranges,
            levels,
            trends,
            computed: true,
        }
    }
}

impl Default for TimeframeTaData {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages replay mode state and operations.
///
/// Replay mode allows users to step through historical candle data,
/// viewing the chart as it would have appeared at any point in time.
pub struct ReplayManager {
    /// Whether replay mode is currently enabled.
    pub enabled: bool,
    /// Current candle index in replay (None = cursor following, Some = locked to index).
    pub index: Option<usize>,
    /// Current replay position as a timestamp.
    pub timestamp: Option<f64>,
    /// Step size for replay navigation (can be finer than view timeframe).
    pub step_timeframe: Timeframe,
    /// Cached partial candles for replay (re-aggregated from base data).
    pub candles: Option<Vec<Candle>>,
    /// TA computed for the replay range.
    pub ta_data: Option<TimeframeTaData>,
    /// GPU data for replay candles.
    pub timeframe_data: Option<TimeframeData>,
}

impl ReplayManager {
    /// Creates a new ReplayManager in disabled state.
    pub fn new() -> Self {
        Self {
            enabled: false,
            index: None,
            timestamp: None,
            step_timeframe: Timeframe::Min1,
            candles: None,
            ta_data: None,
            timeframe_data: None,
        }
    }

    /// Toggles replay mode on/off.
    ///
    /// When enabling, initializes to cursor-following mode.
    /// When disabling, clears all replay state.
    ///
    /// Returns the new enabled state.
    pub fn toggle(&mut self, current_timeframe_idx: usize) -> bool {
        self.enabled = !self.enabled;
        if self.enabled {
            // Entering replay mode - cursor following until click
            self.clear_state();
            // Default step size to current timeframe
            self.step_timeframe = Timeframe::all()[current_timeframe_idx];
        } else {
            // Exiting replay mode - clear replay state
            self.clear_state();
        }
        self.enabled
    }

    /// Clears all replay state (index, timestamp, candles, TA data).
    fn clear_state(&mut self) {
        self.index = None;
        self.ta_data = None;
        self.timestamp = None;
        self.candles = None;
        self.timeframe_data = None;
    }

    /// Steps forward by one step_timeframe interval.
    ///
    /// Returns true if the step was successful and data needs to be recomputed.
    pub fn step_forward(&mut self, base_candles: &[Candle]) -> bool {
        if !self.enabled || self.timestamp.is_none() {
            return false;
        }

        if base_candles.is_empty() {
            return false;
        }

        let max_timestamp = base_candles.last().map(|c| c.timestamp).unwrap_or(0.0);
        let step_seconds = self.step_timeframe.seconds();

        if let Some(ts) = self.timestamp {
            let new_ts = (ts + step_seconds).min(max_timestamp);
            if new_ts > ts {
                self.timestamp = Some(new_ts);
                return true;
            }
        }
        false
    }

    /// Steps backward by one step_timeframe interval.
    ///
    /// Returns true if the step was successful and data needs to be recomputed.
    pub fn step_backward(&mut self, base_candles: &[Candle]) -> bool {
        if !self.enabled || self.timestamp.is_none() {
            return false;
        }

        if base_candles.is_empty() {
            return false;
        }

        let min_timestamp = base_candles.first().map(|c| c.timestamp).unwrap_or(0.0);
        let step_seconds = self.step_timeframe.seconds();

        if let Some(ts) = self.timestamp {
            let new_ts = (ts - step_seconds).max(min_timestamp);
            if new_ts < ts {
                self.timestamp = Some(new_ts);
                return true;
            }
        }
        false
    }

    /// Increases the step size to the next larger timeframe.
    ///
    /// The step size is capped at the current view timeframe.
    /// Returns true if the step size was changed.
    pub fn increase_step_size(&mut self, current_timeframe_idx: usize) -> bool {
        if !self.enabled {
            return false;
        }

        let timeframes = Timeframe::all();
        let current_idx = timeframes
            .iter()
            .position(|&t| t == self.step_timeframe)
            .unwrap_or(0);

        // Increase step size (up to current view timeframe)
        let max_idx = current_timeframe_idx;
        if current_idx < max_idx {
            self.step_timeframe = timeframes[current_idx + 1];
            return true;
        }
        false
    }

    /// Decreases the step size to the next smaller timeframe.
    ///
    /// The minimum step size is 1 minute.
    /// Returns true if the step size was changed.
    pub fn decrease_step_size(&mut self) -> bool {
        if !self.enabled {
            return false;
        }

        let timeframes = Timeframe::all();
        let current_idx = timeframes
            .iter()
            .position(|&t| t == self.step_timeframe)
            .unwrap_or(0);

        // Decrease step size (down to 1min)
        if current_idx > 0 {
            self.step_timeframe = timeframes[current_idx - 1];
            return true;
        }
        false
    }

    /// Sets the replay position to a specific candle index.
    ///
    /// Converts the candle index to a timestamp and stores both.
    pub fn set_index(&mut self, index: usize, candles: &[Candle]) {
        if candles.is_empty() {
            return;
        }

        let clamped_idx = index.min(candles.len().saturating_sub(1));
        let timestamp = candles[clamped_idx].timestamp;

        self.index = Some(clamped_idx);
        self.timestamp = Some(timestamp);
    }

    /// Recomputes the replay candles from base 1min data.
    ///
    /// Re-aggregates candles up to replay_timestamp for accurate partial candle display.
    /// This should be called after any timestamp change.
    ///
    /// # Arguments
    /// * `base_candles` - The base 1-minute candle data
    /// * `current_timeframe_idx` - Index of the current view timeframe
    /// * `create_timeframe_data` - Closure to create GPU TimeframeData from candles
    pub fn recompute_candles<F>(
        &mut self,
        base_candles: &[Candle],
        current_timeframe_idx: usize,
        create_timeframe_data: F,
    ) where
        F: FnOnce(Vec<Candle>, &str) -> TimeframeData,
    {
        let Some(replay_ts) = self.timestamp else {
            self.candles = None;
            self.index = None;
            self.timeframe_data = None;
            return;
        };

        if base_candles.is_empty() {
            self.index = Some(0);
            self.candles = None;
            self.timeframe_data = None;
            return;
        }

        // Binary search to find the last base candle at or before replay_ts
        let base_end_idx = base_candles
            .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap())
            .unwrap_or_else(|i| i.saturating_sub(1))
            .min(base_candles.len().saturating_sub(1));

        // Get the current view timeframe
        let current_tf = Timeframe::all()[current_timeframe_idx];

        // If we're on 1min timeframe, just use the index directly
        if current_tf == Timeframe::Min1 {
            self.index = Some(base_end_idx);
            self.candles = None;
            self.timeframe_data = None;
            return;
        }

        // Re-aggregate base candles up to replay_ts into the current timeframe
        let filtered_base = &base_candles[..=base_end_idx];
        let aggregated = aggregate_candles(filtered_base, current_tf);

        if aggregated.is_empty() {
            self.index = Some(0);
            self.candles = None;
            self.timeframe_data = None;
            return;
        }

        // Update index to the last aggregated candle
        self.index = Some(aggregated.len().saturating_sub(1));

        // Create GPU buffers for the re-aggregated candles
        let tf_label = current_tf.label();
        let tf_data = create_timeframe_data(aggregated.clone(), tf_label);

        self.candles = Some(aggregated);
        self.timeframe_data = Some(tf_data);
    }

    /// Returns the visible candles for rendering.
    ///
    /// If replay has custom candles (re-aggregated), returns those.
    /// Otherwise returns a reference to the normal candles.
    #[allow(dead_code)]
    pub fn get_visible_candles<'a>(&'a self, normal_candles: &'a [Candle]) -> &'a [Candle] {
        if let Some(ref candles) = self.candles {
            candles
        } else {
            normal_candles
        }
    }

    /// Returns whether replay mode is active with a locked position.
    pub fn is_locked(&self) -> bool {
        self.enabled && self.index.is_some()
    }

    /// Returns whether this replay has custom timeframe data (re-aggregated candles).
    pub fn has_custom_timeframe_data(&self) -> bool {
        self.timeframe_data.is_some()
    }
}

impl Default for ReplayManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_manager_new() {
        let manager = ReplayManager::new();
        assert!(!manager.enabled);
        assert!(manager.index.is_none());
        assert!(manager.timestamp.is_none());
        assert_eq!(manager.step_timeframe, Timeframe::Min1);
    }

    #[test]
    fn test_toggle() {
        let mut manager = ReplayManager::new();

        // Toggle on
        let enabled = manager.toggle(2); // 5min timeframe
        assert!(enabled);
        assert!(manager.enabled);
        assert_eq!(manager.step_timeframe, Timeframe::all()[2]);

        // Toggle off
        let enabled = manager.toggle(2);
        assert!(!enabled);
        assert!(!manager.enabled);
    }

    #[test]
    fn test_step_forward_not_enabled() {
        let mut manager = ReplayManager::new();
        let candles = vec![Candle {
            timestamp: 1000.0,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
        }];

        assert!(!manager.step_forward(&candles));
    }

    #[test]
    fn test_step_forward() {
        let mut manager = ReplayManager::new();
        manager.enabled = true;
        manager.timestamp = Some(1000.0);
        manager.step_timeframe = Timeframe::Min1;

        let candles = vec![
            Candle {
                timestamp: 1000.0,
                open: 100.0,
                high: 110.0,
                low: 90.0,
                close: 105.0,
                volume: 1000.0,
            },
            Candle {
                timestamp: 1060.0,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1100.0,
            },
        ];

        assert!(manager.step_forward(&candles));
        assert_eq!(manager.timestamp, Some(1060.0));
    }

    #[test]
    fn test_step_backward() {
        let mut manager = ReplayManager::new();
        manager.enabled = true;
        manager.timestamp = Some(1060.0);
        manager.step_timeframe = Timeframe::Min1;

        let candles = vec![
            Candle {
                timestamp: 1000.0,
                open: 100.0,
                high: 110.0,
                low: 90.0,
                close: 105.0,
                volume: 1000.0,
            },
            Candle {
                timestamp: 1060.0,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1100.0,
            },
        ];

        assert!(manager.step_backward(&candles));
        assert_eq!(manager.timestamp, Some(1000.0));
    }

    #[test]
    fn test_increase_step_size() {
        let mut manager = ReplayManager::new();
        manager.enabled = true;
        manager.step_timeframe = Timeframe::Min1;

        // Can increase up to current timeframe
        assert!(manager.increase_step_size(3));
        assert_eq!(manager.step_timeframe, Timeframe::all()[1]);

        // Cannot increase beyond current timeframe
        manager.step_timeframe = Timeframe::all()[3];
        assert!(!manager.increase_step_size(3));
    }

    #[test]
    fn test_decrease_step_size() {
        let mut manager = ReplayManager::new();
        manager.enabled = true;
        manager.step_timeframe = Timeframe::all()[2];

        assert!(manager.decrease_step_size());
        assert_eq!(manager.step_timeframe, Timeframe::all()[1]);

        // Decrease to min
        assert!(manager.decrease_step_size());
        assert_eq!(manager.step_timeframe, Timeframe::Min1);

        // Cannot decrease below 1min
        assert!(!manager.decrease_step_size());
    }

    #[test]
    fn test_set_index() {
        let mut manager = ReplayManager::new();
        let candles = vec![
            Candle {
                timestamp: 1000.0,
                open: 100.0,
                high: 110.0,
                low: 90.0,
                close: 105.0,
                volume: 1000.0,
            },
            Candle {
                timestamp: 1060.0,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1100.0,
            },
        ];

        manager.set_index(1, &candles);
        assert_eq!(manager.index, Some(1));
        assert_eq!(manager.timestamp, Some(1060.0));
    }

    #[test]
    fn test_set_index_clamped() {
        let mut manager = ReplayManager::new();
        let candles = vec![Candle {
            timestamp: 1000.0,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
        }];

        manager.set_index(100, &candles);
        assert_eq!(manager.index, Some(0));
        assert_eq!(manager.timestamp, Some(1000.0));
    }

    #[test]
    fn test_timeframe_ta_data_default() {
        let ta = TimeframeTaData::default();
        assert!(ta.ranges.is_empty());
        assert!(ta.levels.is_empty());
        assert!(ta.trends.is_empty());
        assert!(!ta.computed);
    }
}
