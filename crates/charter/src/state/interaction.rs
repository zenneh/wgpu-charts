//! Interaction state for current user operations.
//!
//! This module contains state related to the current interaction mode:
//! loading state, replay mode, live data connection, and current symbol.

use crate::replay::ReplayManager;
use super::LoadingState;
use charter_ta::MlInferenceHandle;

/// Interaction state for the current user session.
///
/// This struct tracks:
/// - Current loading operations
/// - Replay mode state
/// - Live data connection status
/// - Current trading symbol
/// - ML inference handle
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

    /// ML inference handle (if model is loaded).
    pub ml_inference: Option<MlInferenceHandle>,
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
            ml_inference: None,
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

    /// Check if ML inference is available.
    pub fn has_ml_inference(&self) -> bool {
        self.ml_inference.is_some()
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

    /// Load ML inference model from path.
    pub fn load_ml_model(&mut self, path: &str) -> Result<(), String> {
        match MlInferenceHandle::load(path) {
            Ok(handle) => {
                self.ml_inference = Some(handle);
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Default for InteractionState {
    fn default() -> Self {
        Self::new("BTCUSDT".to_string())
    }
}
