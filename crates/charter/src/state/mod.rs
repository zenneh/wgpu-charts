//! Application state management.
//!
//! This module decomposes the monolithic state into focused sub-states:
//! - [`GraphicsState`]: GPU resources (surface, device, queue, renderer)
//! - [`DocumentState`]: Persistent chart data (timeframes, drawings, TA, indicators)
//! - [`ViewState`]: Camera and visible range
//! - [`InteractionState`]: Current interaction mode (drawing, replay, loading)
//! - [`UiState`]: UI framework state (egui context, panels)
//!
//! The main [`AppState`] struct composes these together and provides
//! methods for common operations.

pub mod document;
pub mod graphics;
pub mod interaction;
pub mod snapshot;
pub mod ui;
pub mod view;

pub use document::DocumentState;
pub use document::TaDisplaySettings;
pub use graphics::GraphicsState;
pub use interaction::InteractionState;
pub use snapshot::RenderSnapshot;
pub use snapshot::RenderSnapshotBuilder;
pub use ui::UiState;
pub use view::CoordinateSystem;
pub use view::ScreenPos;
pub use view::ViewState;
pub use view::WorldPos;

use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use winit::window::Window;

use crate::drawing::DrawingManager;
use crate::input::InputHandler;
use charter_config::Config as AppConfig;
use charter_render::DrawingPipeline;
use charter_sync::SyncManager;

/// Messages sent from background threads to the main thread.
pub enum BackgroundMessage {
    /// Data loaded from file or API.
    DataLoaded(Vec<charter_core::Candle>),
    /// A timeframe has been aggregated.
    TimeframeAggregated {
        index: usize,
        candles: Vec<charter_core::Candle>,
    },
    /// TA computation complete for a timeframe.
    TaComputed {
        timeframe: usize,
        ranges: Vec<charter_ta::Range>,
        levels: Vec<charter_ta::Level>,
        trends: Vec<charter_ta::Trend>,
    },
    /// Progress update for batch ML TA computation.
    MlTaProgress { completed: usize, total: usize },
    /// Loading state update.
    LoadingStateChanged(LoadingState),
    /// Live candle update from WebSocket.
    LiveCandleUpdate {
        candle: charter_core::Candle,
        is_closed: bool,
    },
    /// WebSocket connection status changed.
    ConnectionStatus(bool),
    /// Error occurred.
    Error(String),
    /// Sync progress update.
    SyncProgressUpdate(charter_sync::SyncProgress),
    /// Sync completed successfully.
    SyncComplete { total_candles: u64 },
    /// Sync encountered an error.
    SyncError(String),
}

/// Loading state for background operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingState {
    /// No loading in progress.
    Idle,
    /// Loading data from file.
    LoadingData,
    /// Fetching data from MEXC API.
    FetchingMexcData { symbol: String },
    /// Aggregating timeframes.
    AggregatingTimeframes { current: usize, total: usize },
    /// Creating GPU buffers.
    CreatingBuffers { current: usize, total: usize },
    /// Computing technical analysis.
    ComputingTa { timeframe: usize },
    /// Computing TA for multiple timeframes (for ML inference).
    ComputingMlTa { completed: usize, total: usize },
}

impl LoadingState {
    /// Check if loading is in progress.
    pub fn is_loading(&self) -> bool {
        !matches!(self, LoadingState::Idle)
    }

    /// Get a human-readable message for the current state.
    pub fn message(&self) -> String {
        match self {
            LoadingState::Idle => String::new(),
            LoadingState::LoadingData => "Loading data...".to_string(),
            LoadingState::FetchingMexcData { symbol } => {
                format!("Fetching {} from MEXC...", symbol)
            }
            LoadingState::AggregatingTimeframes { current, total } => {
                format!("Aggregating timeframes ({}/{})", current, total)
            }
            LoadingState::CreatingBuffers { current, total } => {
                format!("Creating GPU buffers ({}/{})", current, total)
            }
            LoadingState::ComputingTa { timeframe } => {
                format!("Computing TA for timeframe {}", timeframe)
            }
            LoadingState::ComputingMlTa { completed, total } => {
                format!("Computing TA for ML ({}/{})", completed, total)
            }
        }
    }
}

/// Main application state, composing all sub-states.
///
/// This struct is the central state container for the Charter application.
/// It owns all GPU resources, document data, and interaction state.
pub struct AppState {
    /// GPU and rendering resources.
    pub graphics: GraphicsState,

    /// Persistent document data (candles, drawings, TA).
    pub document: DocumentState,

    /// Camera and view state.
    pub view: ViewState,

    /// Current interaction mode and state.
    pub interaction: InteractionState,

    /// UI framework state.
    pub ui: UiState,

    // --- Window and platform ---
    /// The main window handle.
    pub window: Arc<Window>,

    // --- Background communication ---
    /// Receiver for background thread messages.
    pub bg_receiver: Receiver<BackgroundMessage>,
    /// Sender for background thread messages.
    pub bg_sender: Sender<BackgroundMessage>,
    /// Candles waiting to be converted to GPU buffers (index, candles).
    pub pending_timeframes: Vec<(usize, Vec<charter_core::Candle>)>,

    // --- Input handling ---
    /// Input handler for keyboard and mouse.
    pub input: InputHandler,

    // --- FPS tracking ---
    /// Last frame timestamp for FPS calculation.
    pub last_frame_time: Instant,
    /// Frame counter for FPS calculation.
    pub frame_count: u32,
    /// Current frames per second.
    pub fps: f32,

    // --- Async runtime ---
    /// Tokio runtime for async operations.
    pub(crate) tokio_runtime: Option<tokio::runtime::Runtime>,

    // --- Application config ---
    /// Application configuration.
    pub app_config: AppConfig,

    // --- Historical data sync ---
    /// Whether sync is enabled.
    pub sync_enabled: bool,
    /// Current sync state.
    pub sync_state: charter_sync::SyncState,
    /// Sync manager instance.
    pub sync_manager: Option<SyncManager>,

    // --- Drawing tools ---
    /// Drawing tool state manager.
    pub drawing: DrawingManager,
    /// Drawing pipeline for GPU rendering.
    pub drawing_pipeline: DrawingPipeline,
    /// GPU buffer for horizontal rays.
    pub drawing_hray_buffer: wgpu::Buffer,
    /// GPU buffer for rays/trendlines.
    pub drawing_ray_buffer: wgpu::Buffer,
    /// GPU buffer for rectangles.
    pub drawing_rect_buffer: wgpu::Buffer,
    /// GPU buffer for anchor handles.
    pub drawing_anchor_buffer: wgpu::Buffer,
    /// GPU buffer for drawing render params.
    pub drawing_params_buffer: wgpu::Buffer,
    /// Bind group for drawing pipeline.
    pub drawing_bind_group: wgpu::BindGroup,
}

impl AppState {
    /// Check if the application is ready to render.
    pub fn is_ready(&self) -> bool {
        self.graphics.is_surface_configured && !self.interaction.loading_state.is_loading()
    }

    /// Request a redraw of the window.
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Get the current timeframe index.
    pub fn current_timeframe(&self) -> usize {
        self.document.current_timeframe
    }

    /// Get the current symbol being displayed.
    pub fn current_symbol(&self) -> &str {
        &self.interaction.current_symbol
    }

    /// Create a render snapshot from the current state.
    pub fn snapshot_for_render(&self) -> RenderSnapshot {
        RenderSnapshotBuilder::new()
            .camera(
                self.graphics.renderer.camera.position,
                self.graphics.renderer.camera.scale,
            )
            .surface_size(self.graphics.config.width, self.graphics.config.height)
            .visible_range(
                self.graphics.renderer.visible_start,
                self.graphics.renderer.visible_count,
                self.graphics.renderer.current_lod_factor as u32,
            )
            .timeframe(
                self.document.current_timeframe,
                self.document.candle_count(),
            )
            .ta_settings(self.document.ta_settings.clone())
            .replay(
                self.interaction.replay.enabled,
                self.interaction.replay.index,
                self.interaction.replay.timestamp,
            )
            .connection(
                self.interaction.ws_connected,
                self.interaction.current_symbol.clone(),
            )
            .panels(
                self.ui.panels.show_macd_panel,
                self.ui.panels.show_symbol_picker,
                self.sync_enabled,
            )
            .fps(self.fps)
            .guidelines(self.graphics.renderer.guideline_values.clone())
            .build()
    }
}
