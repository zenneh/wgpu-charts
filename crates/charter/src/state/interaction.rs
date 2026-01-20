//! Interaction state for current user operations.
//!
//! This module contains state related to the current interaction mode:
//! loading state, replay mode, live data connection, and current symbol.

use crate::replay::ReplayManager;

/// Current loading state of the application.
#[derive(Debug, Clone)]
pub enum LoadingState {
    /// No loading in progress.
    Idle,
    /// Fetching candle data from MEXC API.
    FetchingMexcData { symbol: String },
    /// Aggregating candles into higher timeframes.
    AggregatingTimeframes { current: usize, total: usize },
    /// Creating GPU buffers for timeframe data.
    CreatingBuffers { current: usize, total: usize },
    /// Computing technical analysis for a timeframe.
    ComputingTa { timeframe: usize },
}

impl LoadingState {
    /// Check if currently in a loading state.
    pub fn is_loading(&self) -> bool {
        !matches!(self, LoadingState::Idle)
    }

    /// Get a human-readable message for the current state.
    pub fn message(&self) -> String {
        match self {
            LoadingState::Idle => String::new(),
            LoadingState::FetchingMexcData { symbol } => format!("Loading {} data...", symbol),
            LoadingState::AggregatingTimeframes { current, total } => {
                format!("Aggregating timeframes ({}/{})", current, total)
            }
            LoadingState::CreatingBuffers { current, total } => {
                format!("Creating buffers ({}/{})", current, total)
            }
            LoadingState::ComputingTa { timeframe } => {
                let tf_labels = ["1m", "3m", "5m", "30m", "1h", "3h", "5h", "10h", "1d", "1w", "3w", "1M"];
                let label = tf_labels.get(*timeframe).unwrap_or(&"?");
                format!("Computing TA for {}...", label)
            }
        }
    }
}

impl Default for LoadingState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Interaction state for the current user session.
///
/// This struct tracks:
/// - Current loading operations
/// - Replay mode state
/// - Live data connection status
/// - Current trading symbol
pub struct InteractionState {
    /// Current loading operation state.
    pub loading_state: LoadingState,

    /// Replay mode manager.
    pub replay: ReplayManager,

    /// Current trading symbol (e.g., "BTCUSDT").
    pub current_symbol: String,

    /// Whether WebSocket is connected for live data.
    pub ws_connected: bool,

    /// Receiver for live data events.
    pub live_event_rx: Option<tokio::sync::mpsc::Receiver<charter_data::LiveDataEvent>>,
}

impl InteractionState {
    /// Create new interaction state with the given symbol.
    pub fn new(default_symbol: String) -> Self {
        Self {
            loading_state: LoadingState::Idle,
            replay: ReplayManager::new(),
            current_symbol: default_symbol,
            ws_connected: false,
            live_event_rx: None,
        }
    }

    /// Check if currently loading data.
    pub fn is_loading(&self) -> bool {
        self.loading_state.is_loading()
    }

    /// Check if replay mode is active.
    pub fn is_replay_active(&self) -> bool {
        self.replay.enabled
    }

    /// Check if replay is locked to a specific position.
    pub fn is_replay_locked(&self) -> bool {
        self.replay.is_locked()
    }

    /// Check if live data connection is active.
    pub fn is_connected(&self) -> bool {
        self.ws_connected
    }

    /// Set the loading state.
    pub fn set_loading_state(&mut self, state: LoadingState) {
        self.loading_state = state;
    }

    /// Set connection status.
    pub fn set_connected(&mut self, connected: bool) {
        self.ws_connected = connected;
    }

    /// Change the current symbol.
    pub fn set_symbol(&mut self, symbol: String) {
        self.current_symbol = symbol;
    }

    /// Toggle replay mode.
    pub fn toggle_replay(&mut self, current_timeframe_idx: usize) -> bool {
        self.replay.toggle(current_timeframe_idx)
    }
}

impl Default for InteractionState {
    fn default() -> Self {
        Self::new("BTCUSDT".to_string())
    }
}
