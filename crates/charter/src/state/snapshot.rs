//! Render snapshot for decoupled rendering.
//!
//! This module provides [`RenderSnapshot`], an immutable snapshot of state
//! needed for rendering. This decouples the render pass from mutable state,
//! avoiding borrow checker issues and enabling cleaner rendering code.

use charter_render::CANDLE_SPACING;

use super::document::TaDisplaySettings;

/// Immutable snapshot of state needed for rendering.
///
/// This struct captures all the data needed to render a single frame,
/// avoiding mutable borrows during the render pass. It can be created
/// from `AppState` at the start of each frame.
#[derive(Clone)]
pub struct RenderSnapshot {
    /// Camera position in world coordinates.
    pub camera_position: [f32; 2],

    /// Camera scale (zoom level).
    pub camera_scale: [f32; 2],

    /// Surface dimensions.
    pub surface_width: u32,
    pub surface_height: u32,

    /// Visible candle range.
    pub visible_start: u32,
    pub visible_count: u32,

    /// Current LOD factor.
    pub current_lod_factor: u32,

    /// Current timeframe index.
    pub current_timeframe: usize,

    /// Number of candles in the current timeframe.
    pub candle_count: usize,

    /// TA display settings.
    pub ta_settings: TaDisplaySettings,

    /// Whether replay mode is active.
    pub replay_enabled: bool,

    /// Replay candle index (if locked).
    pub replay_index: Option<usize>,

    /// Replay timestamp (if set).
    pub replay_timestamp: Option<f64>,

    /// Whether WebSocket is connected.
    pub ws_connected: bool,

    /// Current symbol.
    pub current_symbol: String,

    /// Whether MACD panel is visible.
    pub show_macd_panel: bool,

    /// Whether symbol picker is visible.
    pub show_symbol_picker: bool,

    /// Whether sync is enabled.
    pub sync_enabled: bool,

    /// Current FPS.
    pub fps: f32,

    /// Guideline values for price axis.
    pub guideline_values: Vec<f32>,
}

impl RenderSnapshot {
    /// Create a new render snapshot with default values.
    pub fn new() -> Self {
        Self {
            camera_position: [0.0, 0.0],
            camera_scale: [100.0, 100.0],
            surface_width: 800,
            surface_height: 600,
            visible_start: 0,
            visible_count: 0,
            current_lod_factor: 1,
            current_timeframe: 0,
            candle_count: 0,
            ta_settings: TaDisplaySettings::default(),
            replay_enabled: false,
            replay_index: None,
            replay_timestamp: None,
            ws_connected: false,
            current_symbol: String::new(),
            show_macd_panel: false,
            show_symbol_picker: false,
            sync_enabled: false,
            fps: 0.0,
            guideline_values: Vec::new(),
        }
    }

    /// Get the chart area dimensions.
    pub fn chart_dimensions(&self, stats_panel_width: f32, volume_height_ratio: f32) -> (f32, f32) {
        let total_width = self.surface_width as f32;
        let total_height = self.surface_height as f32;
        let chart_width = (total_width - stats_panel_width).max(1.0);
        let chart_height = (total_height * (1.0 - volume_height_ratio)).max(1.0);
        (chart_width, chart_height)
    }

    /// Get the aspect ratio of the chart area.
    pub fn chart_aspect(&self, stats_panel_width: f32, volume_height_ratio: f32) -> f32 {
        let (width, height) = self.chart_dimensions(stats_panel_width, volume_height_ratio);
        width / height
    }

    /// Get the visible X range in world coordinates.
    pub fn visible_x_range(&self, aspect: f32) -> (f32, f32) {
        let half_width = self.camera_scale[0] * aspect;
        let x_min = self.camera_position[0] - half_width;
        let x_max = self.camera_position[0] + half_width;
        (x_min, x_max)
    }

    /// Get the visible Y range in world coordinates.
    pub fn visible_y_range(&self) -> (f32, f32) {
        let y_min = self.camera_position[1] - self.camera_scale[1];
        let y_max = self.camera_position[1] + self.camera_scale[1];
        (y_min, y_max)
    }

    /// Get the effective visible count for rendering.
    ///
    /// This accounts for replay mode, which may limit how many candles are shown.
    pub fn effective_visible_count(&self, has_replay_timeframe_data: bool) -> u32 {
        if has_replay_timeframe_data {
            // When using replay data, show all the re-aggregated candles
            // This would need to be passed in from the actual replay data
            self.visible_count
        } else if self.replay_enabled {
            if let Some(replay_idx) = self.replay_index {
                let visible_start = self.visible_start as usize;
                let visible_count = self.visible_count as usize;
                if replay_idx < visible_start {
                    0
                } else if replay_idx >= visible_start + visible_count {
                    visible_count as u32
                } else {
                    (replay_idx - visible_start + 1) as u32
                }
            } else {
                self.visible_count
            }
        } else {
            self.visible_count
        }
    }

    /// Check if TA should be rendered.
    pub fn should_render_ta(&self) -> bool {
        self.ta_settings.show_ta
    }

    /// Get the X coordinate for the right edge of the data.
    pub fn data_x_max(&self) -> f32 {
        self.candle_count as f32 * CANDLE_SPACING
    }
}

impl Default for RenderSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating RenderSnapshot from AppState components.
///
/// This provides a fluent API for constructing snapshots, making it
/// easier to gather data from different parts of the state.
pub struct RenderSnapshotBuilder {
    snapshot: RenderSnapshot,
}

impl RenderSnapshotBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            snapshot: RenderSnapshot::new(),
        }
    }

    /// Set camera state.
    pub fn camera(mut self, position: [f32; 2], scale: [f32; 2]) -> Self {
        self.snapshot.camera_position = position;
        self.snapshot.camera_scale = scale;
        self
    }

    /// Set surface dimensions.
    pub fn surface_size(mut self, width: u32, height: u32) -> Self {
        self.snapshot.surface_width = width;
        self.snapshot.surface_height = height;
        self
    }

    /// Set visible range.
    pub fn visible_range(mut self, start: u32, count: u32, lod_factor: u32) -> Self {
        self.snapshot.visible_start = start;
        self.snapshot.visible_count = count;
        self.snapshot.current_lod_factor = lod_factor;
        self
    }

    /// Set current timeframe and candle count.
    pub fn timeframe(mut self, index: usize, candle_count: usize) -> Self {
        self.snapshot.current_timeframe = index;
        self.snapshot.candle_count = candle_count;
        self
    }

    /// Set TA settings.
    pub fn ta_settings(mut self, settings: TaDisplaySettings) -> Self {
        self.snapshot.ta_settings = settings;
        self
    }

    /// Set replay state.
    pub fn replay(mut self, enabled: bool, index: Option<usize>, timestamp: Option<f64>) -> Self {
        self.snapshot.replay_enabled = enabled;
        self.snapshot.replay_index = index;
        self.snapshot.replay_timestamp = timestamp;
        self
    }

    /// Set connection and symbol state.
    pub fn connection(mut self, ws_connected: bool, symbol: String) -> Self {
        self.snapshot.ws_connected = ws_connected;
        self.snapshot.current_symbol = symbol;
        self
    }

    /// Set UI panel visibility.
    pub fn panels(mut self, show_macd: bool, show_symbol_picker: bool, sync_enabled: bool) -> Self {
        self.snapshot.show_macd_panel = show_macd;
        self.snapshot.show_symbol_picker = show_symbol_picker;
        self.snapshot.sync_enabled = sync_enabled;
        self
    }

    /// Set FPS value.
    pub fn fps(mut self, fps: f32) -> Self {
        self.snapshot.fps = fps;
        self
    }

    /// Set guideline values.
    pub fn guidelines(mut self, values: Vec<f32>) -> Self {
        self.snapshot.guideline_values = values;
        self
    }

    /// Build the final snapshot.
    pub fn build(self) -> RenderSnapshot {
        self.snapshot
    }
}

impl Default for RenderSnapshotBuilder {
    fn default() -> Self {
        Self::new()
    }
}
