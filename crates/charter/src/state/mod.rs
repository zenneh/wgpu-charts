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
pub use interaction::LoadingState;
pub use snapshot::RenderSnapshot;
pub use snapshot::RenderSnapshotBuilder;
pub use ui::UiState;
pub use view::CoordinateSystem;
pub use view::ScreenPos;
pub use view::ViewState;
pub use view::WorldPos;

use std::cell::RefCell;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use crate::drawing::{prepare_drawing_render_data, DrawingManager, DrawingTool};
use crate::indicators::{DynMacd, IndicatorGpuBuffers, IndicatorRegistry, MacdGpuBuffers};
use crate::input::{InputAction, InputHandler};
use crate::replay::TimeframeTaData;
use crate::ui::{
    show_drawing_toolbar, show_loading_overlay, show_macd_panel, show_ta_panel,
    DrawingToolbarResponse, MacdPanelResponse, SymbolPickerState, TaHoveredInfo,
};
use charter_config::Config as AppConfig;
use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_data::{LiveDataEvent, LiveDataManager, MexcSource};
use charter_indicators::{Indicator, Macd, MacdConfig};
use charter_render::{
    ChartRenderer, DrawingPipeline, DrawingRenderParams, IndicatorParams, IndicatorPointGpu,
    LevelGpu, RangeGpu, RenderParams, TaRenderParams, TimeframeData, TrendGpu, VolumeRenderParams,
    CANDLE_SPACING, INDICES_PER_CANDLE, MAX_TA_LEVELS, MAX_TA_RANGES, MAX_TA_TRENDS,
    STATS_PANEL_WIDTH, VOLUME_HEIGHT_RATIO,
};
use charter_sync::SyncManager;
use charter_ta::{
    Analyzer, AnalyzerConfig, CandleDirection, Level, LevelState, LevelType, MlFeatures,
    MlInferenceHandle, Range, TimeframeFeatures, Trend, TrendState,
};
use charter_ui::TopBar;

/// MACD conversion result with points and start indices.
struct MacdConversionResult {
    macd_points: Vec<IndicatorPointGpu>,
    signal_points: Vec<IndicatorPointGpu>,
    histogram_points: Vec<IndicatorPointGpu>,
    macd_start_index: usize,
    signal_start_index: usize,
}

impl MacdConversionResult {
    fn empty() -> Self {
        Self {
            macd_points: vec![IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 }],
            signal_points: vec![IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 }],
            histogram_points: vec![IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 }],
            macd_start_index: 0,
            signal_start_index: 0,
        }
    }
}

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

    // --- View state tracking ---
    /// Whether we've performed the initial view fit.
    /// Prevents destroying user's zoom/pan position on data sync.
    pub has_initial_view: bool,
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

    // =========================================================================
    // Constructor
    // =========================================================================

    /// Create a new AppState with all subsystems initialized.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // Create background channel
        let (bg_sender, bg_receiver) = mpsc::channel();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Create tokio runtime for async operations
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        // Load application config
        let app_config = AppConfig::load_default();

        // Default symbol from config
        let default_symbol = app_config.general.default_symbol.clone();

        // Start background data loading
        let sender = bg_sender.clone();
        let symbol = default_symbol.clone();
        let db_path = app_config.sync.get_db_path();
        tokio_runtime.spawn(async move {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                LoadingState::FetchingMexcData {
                    symbol: symbol.clone(),
                },
            ));

            // First, try to load from DuckDB if it has data
            let db_candles = if let Ok(db) = charter_sync::CandleDb::open(&db_path) {
                match db.load_candles(&symbol) {
                    Ok(candles) if !candles.is_empty() => {
                        log::info!(
                            "Loaded {} candles from DuckDB for {}",
                            candles.len(),
                            symbol
                        );
                        Some(candles)
                    }
                    _ => None,
                }
            } else {
                None
            };

            // If we have DB data, use it; otherwise fetch from API
            if let Some(candles) = db_candles {
                let _ = sender.send(BackgroundMessage::DataLoaded(candles));
            } else {
                let source = MexcSource::new(&symbol);
                match source.load().await {
                    Ok(base_candles) => {
                        let _ = sender.send(BackgroundMessage::DataLoaded(base_candles));
                    }
                    Err(e) => {
                        let _ = sender.send(BackgroundMessage::Error(format!(
                            "Failed to load data: {}",
                            e
                        )));
                    }
                }
            }
        });

        // Create renderer with empty candle data
        let empty_candles: Vec<Candle> = vec![];
        let renderer = ChartRenderer::new(
            &device,
            &queue,
            surface_format,
            size.width,
            size.height,
            &empty_candles,
        );

        // Create placeholder timeframes
        let timeframe_types = Timeframe::all();
        let mut timeframes = Vec::new();
        let mut ta_data = Vec::new();

        for tf in timeframe_types {
            ta_data.push(TimeframeTaData {
                ranges: Vec::new(),
                levels: Vec::new(),
                trends: Vec::new(),
                computed: false,
                prediction: None,
            });

            let tf_data = renderer.create_timeframe_data(&device, Vec::new(), tf.label());
            timeframes.push(tf_data);
        }

        // Create graphics state
        let graphics = GraphicsState::new(surface, device.clone(), queue.clone(), config, renderer);

        // Create document state
        let mut document = DocumentState::new(timeframe_types.len());
        document.timeframes = timeframes;
        document.ta_data = ta_data;
        document.ta_settings = TaDisplaySettings::from_config(&app_config.ta.display);

        // Create interaction state
        let mut interaction = InteractionState::new(default_symbol.clone());
        interaction.loading_state = LoadingState::FetchingMexcData {
            symbol: default_symbol,
        };

        // Try to load ML model if it exists
        if let Err(e) = interaction.load_ml_model("data/charter_model.onnx") {
            log::debug!("ML model not available (this is OK): {}", e);
        }

        // Create UI state
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, graphics.surface_format(), None, 1, false);
        let ui = UiState::new(egui_ctx, egui_state, egui_renderer);

        // Create drawing pipeline and buffers
        let drawing_pipeline = DrawingPipeline::new(
            &device,
            graphics.surface_format(),
            &graphics.renderer.candle_pipeline.camera_bind_group_layout,
        );
        let drawing_hray_buffer = drawing_pipeline.create_hray_buffer(&device);
        let drawing_ray_buffer = drawing_pipeline.create_ray_buffer(&device);
        let drawing_rect_buffer = drawing_pipeline.create_rect_buffer(&device);
        let drawing_anchor_buffer = drawing_pipeline.create_anchor_buffer(&device);
        let drawing_params_buffer = drawing_pipeline.create_params_buffer(&device);
        let drawing_bind_group = drawing_pipeline.create_bind_group(
            &device,
            &drawing_hray_buffer,
            &drawing_ray_buffer,
            &drawing_rect_buffer,
            &drawing_anchor_buffer,
            &drawing_params_buffer,
        );

        Ok(Self {
            graphics,
            document,
            view: ViewState::new(),
            interaction,
            ui,
            window,
            bg_receiver,
            bg_sender,
            pending_timeframes: Vec::new(),
            input: InputHandler::new(),
            last_frame_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            tokio_runtime: Some(tokio_runtime),
            app_config: app_config.clone(),
            sync_enabled: app_config.sync.enabled,
            sync_state: charter_sync::SyncState::default(),
            sync_manager: Some(SyncManager::new(
                app_config.sync.get_db_path(),
                app_config.sync.batch_delay_ms,
                app_config.sync.sync_days,
            )),
            drawing: DrawingManager::new(),
            drawing_pipeline,
            drawing_hray_buffer,
            drawing_ray_buffer,
            drawing_rect_buffer,
            drawing_anchor_buffer,
            drawing_params_buffer,
            drawing_bind_group,
            has_initial_view: false,
        })
    }

    // =========================================================================
    // Core Methods
    // =========================================================================

    /// Resize the surface and update the renderer.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.graphics.resize(width, height);
            self.update_visible_range();
        }
    }

    /// Update loop - called each frame.
    pub fn update(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Process any pending messages from background threads.
    /// Returns true if data was updated and view should refresh.
    pub fn process_background_messages(&mut self) -> bool {
        let mut updated = false;

        // Process all available messages (non-blocking)
        while let Ok(msg) = self.bg_receiver.try_recv() {
            match msg {
                BackgroundMessage::LoadingStateChanged(state) => {
                    self.interaction.loading_state = state;
                }
                BackgroundMessage::DataLoaded(base_candles) => {
                    // Clear existing TA data
                    self.document.clear_ta_data();

                    // Start aggregating timeframes in background
                    self.interaction.loading_state = LoadingState::AggregatingTimeframes {
                        current: 0,
                        total: Timeframe::all().len(),
                    };

                    let sender = self.bg_sender.clone();
                    thread::spawn(move || {
                        let timeframe_types = Timeframe::all();
                        let total = timeframe_types.len();

                        for (i, tf) in timeframe_types.iter().enumerate() {
                            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                                LoadingState::AggregatingTimeframes {
                                    current: i + 1,
                                    total,
                                },
                            ));

                            let candles = if *tf == Timeframe::Min1 {
                                base_candles.clone()
                            } else {
                                aggregate_candles(&base_candles, *tf)
                            };

                            let _ = sender.send(BackgroundMessage::TimeframeAggregated {
                                index: i,
                                candles,
                            });
                        }

                        let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                            LoadingState::CreatingBuffers { current: 0, total },
                        ));
                    });
                }
                BackgroundMessage::TimeframeAggregated { index, candles } => {
                    self.pending_timeframes.push((index, candles));
                }
                BackgroundMessage::TaComputed {
                    timeframe,
                    ranges,
                    levels,
                    trends,
                } => {
                    if let Some(ta) = self.document.ta_data.get_mut(timeframe) {
                        ta.ranges = ranges;
                        ta.levels = levels;
                        ta.trends = trends;
                        ta.computed = true;
                        ta.prediction = None;
                    }
                    self.interaction.loading_state = LoadingState::Idle;

                    if timeframe == self.document.current_timeframe
                        && self.document.ta_settings.show_ta
                    {
                        self.update_ta_buffers();
                    }
                    updated = true;
                }
                BackgroundMessage::MlTaProgress { completed, total } => {
                    if completed >= total {
                        self.interaction.loading_state = LoadingState::Idle;
                        if self.interaction.replay.is_locked() {
                            self.recompute_replay_ta();
                        }
                    } else {
                        self.interaction.loading_state =
                            LoadingState::ComputingMlTa { completed, total };
                    }
                    updated = true;
                }
                BackgroundMessage::Error(err) => {
                    eprintln!("Background error: {}", err);
                    self.interaction.loading_state = LoadingState::Idle;
                }
                BackgroundMessage::LiveCandleUpdate { candle, is_closed } => {
                    if self.interaction.loading_state.is_loading() {
                        continue;
                    }
                    if self.document.timeframes.first().map(|tf| tf.candles.is_empty()).unwrap_or(true) {
                        continue;
                    }
                    self.update_live_candle(candle, is_closed);
                    updated = true;
                }
                BackgroundMessage::ConnectionStatus(connected) => {
                    self.interaction.ws_connected = connected;
                    if connected {
                        if let Some(last_candle) = self.document.timeframes.first().and_then(|tf| tf.candles.last()) {
                            self.update_current_price_line(Some(last_candle.close), last_candle.open);
                        }
                    } else {
                        self.update_current_price_line(None, 0.0);
                    }
                    updated = true;
                }
                BackgroundMessage::SyncProgressUpdate(progress) => {
                    self.sync_state = charter_sync::SyncState::Syncing {
                        fetched: progress.fetched,
                        estimated_total: progress.estimated_total,
                        candles_per_sec: progress.candles_per_sec,
                    };
                    updated = true;
                }
                BackgroundMessage::SyncComplete { total_candles } => {
                    self.sync_state = charter_sync::SyncState::Complete { total_candles };
                    self.sync_enabled = false;

                    // Reload data from DuckDB
                    if let Some(runtime) = &self.tokio_runtime {
                        let sender = self.bg_sender.clone();
                        let symbol = self.interaction.current_symbol.clone();
                        let db_path = self.app_config.sync.get_db_path();
                        runtime.spawn(async move {
                            if let Ok(db) = charter_sync::CandleDb::open(&db_path) {
                                if let Ok(candles) = db.load_candles(&symbol) {
                                    if !candles.is_empty() {
                                        log::info!(
                                            "Reloading {} candles from DuckDB after sync",
                                            candles.len()
                                        );
                                        let _ = sender.send(BackgroundMessage::DataLoaded(candles));
                                    }
                                }
                            }
                        });
                    }
                    updated = true;
                }
                BackgroundMessage::SyncError(err) => {
                    eprintln!("Sync error: {}", err);
                    self.sync_state = charter_sync::SyncState::Error(err);
                    self.sync_enabled = false;
                    updated = true;
                }
            }
        }

        // Process pending timeframes (GPU buffer creation on main thread)
        if !self.pending_timeframes.is_empty() {
            let total = Timeframe::all().len();
            let pending: Vec<_> = self.pending_timeframes.drain(..).collect();

            for (index, candles) in pending {
                self.interaction.loading_state = LoadingState::CreatingBuffers {
                    current: index + 1,
                    total,
                };

                let label = Timeframe::all()[index].label();
                let tf_data = self.graphics.renderer.create_timeframe_data(
                    &self.graphics.device,
                    candles,
                    label,
                );
                if let Some(tf) = self.document.timeframes.get_mut(index) {
                    *tf = tf_data;
                }
            }

            self.interaction.loading_state = LoadingState::Idle;

            if !self.has_initial_view {
                self.fit_view();
                self.has_initial_view = true;
            } else {
                self.update_visible_range();
            }

            if !self.interaction.ws_connected {
                self.start_live_updates();
            }

            if self.sync_enabled && !matches!(self.sync_state, charter_sync::SyncState::Syncing { .. }) {
                self.start_sync();
            }

            updated = true;
        }

        updated
    }

    // =========================================================================
    // View Methods
    // =========================================================================

    fn update_visible_range(&mut self) {
        if let Some(tf) = self.document.timeframes.get(self.document.current_timeframe) {
            self.graphics.renderer.update_visible_range(&self.graphics.queue, tf);
            if self.document.ta_settings.show_ta {
                self.update_ta_buffers();
            }
            if !self.document.indicators.is_empty() {
                self.update_macd_params();
            }
            self.graphics.renderer.update_current_price_line(&self.graphics.queue);
        }
    }

    /// Fit all candles in view.
    pub fn fit_view(&mut self) {
        let candles = self.document.current_candles();
        self.graphics.renderer.fit_view(&self.graphics.queue, candles);
        self.update_visible_range();
        self.window.request_redraw();
    }

    /// Switch to a different timeframe.
    pub fn switch_timeframe(&mut self, index: usize) {
        if index >= self.document.timeframes.len() || index == self.document.current_timeframe {
            return;
        }

        let old_candles = &self.document.timeframes[self.document.current_timeframe].candles;
        let new_candles = &self.document.timeframes[index].candles;

        if old_candles.is_empty() || new_candles.is_empty() {
            self.document.current_timeframe = index;
            self.fit_view();
            return;
        }

        let old_candle_idx =
            (self.graphics.renderer.camera.position[0] / CANDLE_SPACING).round() as usize;
        let old_candle_idx = old_candle_idx.min(old_candles.len().saturating_sub(1));
        let target_timestamp = old_candles[old_candle_idx].timestamp;

        let new_candle_idx = match new_candles
            .binary_search_by(|c| c.timestamp.partial_cmp(&target_timestamp).unwrap())
        {
            Ok(idx) => idx,
            Err(idx) => idx.min(new_candles.len().saturating_sub(1)),
        };

        let new_x = (new_candle_idx as f32) * CANDLE_SPACING;
        let ratio = old_candles.len() as f32 / new_candles.len() as f32;
        let new_scale_x = self.graphics.renderer.camera.scale[0] / ratio;

        self.graphics.renderer.camera.position[0] = new_x;
        self.graphics.renderer.camera.scale[0] = new_scale_x.max(5.0);

        self.document.current_timeframe = index;
        self.graphics.renderer.update_camera(&self.graphics.queue);

        if self.interaction.replay.enabled && self.interaction.replay.is_locked() {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
        }

        if self.document.ta_settings.show_ta {
            self.ensure_ta_computed();
        }
        if !self.document.indicators.is_empty() {
            self.recompute_all_macd();
        }
        self.update_visible_range();
        self.window.request_redraw();
    }

    // =========================================================================
    // Input Handling
    // =========================================================================

    /// Convert screen coordinates to world (chart) coordinates.
    fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        // Guard against zero height (E5 fix)
        if chart_height < 1.0 || chart_width < 1.0 {
            return (0.0, 0.0);
        }
        let aspect = chart_width / chart_height;

        let ndc_x = (screen_x / chart_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / chart_height) * 2.0;

        let world_x =
            self.graphics.renderer.camera.position[0] + ndc_x * self.graphics.renderer.camera.scale[0] * aspect;
        let world_y =
            self.graphics.renderer.camera.position[1] + ndc_y * self.graphics.renderer.camera.scale[1];

        (world_x, world_y)
    }

    /// Handle keyboard input.
    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if self.ui.panels.show_symbol_picker && code != KeyCode::Escape {
            return;
        }

        if let Some(action) = self.input.handle_key(code, is_pressed) {
            match action {
                InputAction::DrawingCancel => {
                    if self.ui.panels.show_symbol_picker {
                        self.ui.panels.show_symbol_picker = false;
                        self.window.request_redraw();
                    } else if self.drawing.is_interacting() || self.drawing.tool != DrawingTool::None
                    {
                        self.drawing.cancel();
                        self.drawing.set_tool(DrawingTool::None);
                        self.window.request_redraw();
                    } else {
                        event_loop.exit()
                    }
                }
                InputAction::FitView => self.fit_view(),
                InputAction::ToggleTa => self.toggle_ta(),
                InputAction::ToggleMacdPanel => self.toggle_macd_panel(),
                InputAction::ToggleReplayMode => self.toggle_replay_mode(),
                InputAction::ToggleSymbolPicker => self.toggle_symbol_picker(),
                InputAction::StartLiveUpdates => self.start_live_updates(),
                InputAction::ReplayStepForward => self.replay_step_forward(),
                InputAction::ReplayStepBackward => self.replay_step_backward(),
                InputAction::ReplayDecreaseStep => self.replay_decrease_step_size(),
                InputAction::ReplayIncreaseStep => self.replay_increase_step_size(),
                InputAction::SwitchTimeframe(idx) => self.switch_timeframe(idx),
                InputAction::SelectDrawingTool(tool) => {
                    self.drawing.set_tool(tool);
                    self.window.request_redraw();
                }
                InputAction::ToggleSnap => {
                    self.drawing.toggle_snap();
                    self.window.request_redraw();
                }
                InputAction::DrawingDelete => {
                    self.drawing.delete_selected();
                    self.window.request_redraw();
                }
                InputAction::Exit => event_loop.exit(),
                InputAction::Pan { .. }
                | InputAction::Zoom { .. }
                | InputAction::StartDrag
                | InputAction::EndDrag
                | InputAction::CursorMoved { .. }
                | InputAction::SetReplayIndex => {}
            }
        }
    }

    /// Handle mouse button input.
    pub fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        // Handle drawing interactions - allow when a tool is active OR when there are drawings to select
        if button == MouseButton::Left {
            let has_drawings = !self.drawing.drawings.is_empty();
            let tool_active = self.drawing.tool != DrawingTool::None;

            if tool_active || has_drawings {
                if let Some(last_pos) = self.input.last_mouse_pos {
                    let (world_x, world_y) = self.screen_to_world(last_pos[0], last_pos[1]);
                    let candles = self.document.current_candles();

                    if state == ElementState::Pressed {
                        // For selection when no tool is active, temporarily set to Select mode
                        let original_tool = self.drawing.tool;
                        if !tool_active && has_drawings {
                            self.drawing.tool = DrawingTool::Select;
                        }

                        if self.drawing.handle_press(world_x, world_y, candles, CANDLE_SPACING) {
                            self.window.request_redraw();
                            // Keep Select mode if we selected something, otherwise restore
                            if self.drawing.selected.is_none() && !tool_active {
                                self.drawing.tool = original_tool;
                            }
                            return;
                        }

                        // Restore original tool if nothing was selected
                        if !tool_active {
                            self.drawing.tool = original_tool;
                        }
                    } else {
                        self.drawing.handle_release();
                    }
                }
            }
        }

        let replay_index_set = self.interaction.replay.index.is_some();
        if let Some(action) = self.input.handle_mouse_input(
            state,
            button,
            self.interaction.replay.enabled,
            replay_index_set,
        ) {
            match action {
                InputAction::SetReplayIndex => {
                    let candle_idx = self.get_cursor_candle_index();
                    self.set_replay_index(candle_idx);
                }
                InputAction::StartDrag | InputAction::EndDrag => {}
                _ => {}
            }
        }
    }

    /// Handle cursor movement.
    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let pos = (position.x as f32, position.y as f32);

        // Update drawing cursor and handle dragging when tool is active, interacting, or when drawings exist
        let has_drawings = !self.drawing.drawings.is_empty();
        let needs_drawing_update = self.drawing.tool != DrawingTool::None
            || self.drawing.is_interacting()
            || has_drawings;

        if needs_drawing_update {
            let (world_x, world_y) = self.screen_to_world(pos.0, pos.1);
            let candles = self.document.current_candles();

            // Calculate scale factors for hit detection
            let aspect = self.graphics.config.width as f32 / self.graphics.config.height as f32;
            let x_world_per_pixel = (self.graphics.renderer.camera.scale[0] * aspect * 2.0)
                / self.graphics.config.width as f32;
            let y_world_per_pixel = (self.graphics.renderer.camera.scale[1] * 2.0)
                / self.graphics.config.height as f32;

            self.drawing.update_cursor_with_scale(
                world_x,
                world_y,
                candles,
                CANDLE_SPACING,
                x_world_per_pixel,
                y_world_per_pixel,
            );

            if self.input.mouse_pressed && self.drawing.is_interacting() {
                if self.drawing.handle_drag(world_x, world_y, candles, CANDLE_SPACING) {
                    self.window.request_redraw();
                }
            }
        }

        if let Some(action) = self.input.handle_cursor_moved(pos) {
            match action {
                InputAction::Pan { dx, dy } => {
                    if self.drawing.is_interacting() {
                        return;
                    }

                    let aspect =
                        self.graphics.config.width as f32 / self.graphics.config.height as f32;
                    let world_dx = -dx
                        * (self.graphics.renderer.camera.scale[0] * aspect * 2.0)
                        / self.graphics.config.width as f32;
                    let world_dy = dy * (self.graphics.renderer.camera.scale[1] * 2.0)
                        / self.graphics.config.height as f32;

                    self.graphics.renderer.camera.position[0] += world_dx;
                    self.graphics.renderer.camera.position[1] += world_dy;

                    self.graphics.renderer.update_camera(&self.graphics.queue);
                    self.update_visible_range();
                    self.window.request_redraw();
                }
                InputAction::CursorMoved { x: _, y: _ } => {
                    if self.document.ta_settings.show_ta {
                        self.update_hover_state();
                    }

                    if self.drawing.tool.is_drawing_tool() {
                        self.window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    /// Handle mouse wheel events.
    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        if let Some(InputAction::Zoom {
            delta_x,
            delta_y,
            cursor_x,
            cursor_y,
        }) = self.input.handle_mouse_wheel(delta)
        {
            self.apply_zoom(delta_x, delta_y, cursor_x, cursor_y);
        }
    }

    /// Apply zoom based on scroll deltas and cursor position.
    fn apply_zoom(&mut self, scroll_x: f32, scroll_y: f32, cursor_x: f32, cursor_y: f32) {
        let candles = self.document.current_candles();

        if candles.is_empty() {
            return;
        }

        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        // Guard against zero height (E5 fix)
        if chart_height < 1.0 || chart_width < 1.0 {
            return;
        }
        let aspect = chart_width / chart_height;

        let cursor_ndc = if cursor_x < chart_width && cursor_y < chart_height {
            [
                (cursor_x / chart_width) * 2.0 - 1.0,
                1.0 - (cursor_y / chart_height) * 2.0,
            ]
        } else {
            [0.0, 0.0]
        };

        let world_x = self.graphics.renderer.camera.position[0]
            + cursor_ndc[0] * self.graphics.renderer.camera.scale[0] * aspect;
        let world_y = self.graphics.renderer.camera.position[1]
            + cursor_ndc[1] * self.graphics.renderer.camera.scale[1];

        let data_width = (candles.len() as f32) * CANDLE_SPACING;
        let max_x_zoom = (data_width / 2.0 / aspect).max(10.0) * 1.2;

        let (min_price, max_price) =
            candles
                .iter()
                .fold((f32::MAX, f32::MIN), |(min, max), c| {
                    (min.min(c.low), max.max(c.high))
                });
        let price_range = (max_price - min_price).max(1.0);
        let max_y_zoom = (price_range / 2.0).max(10.0) * 1.5;

        if scroll_x.abs() > 0.001 {
            let zoom_factor = 1.0 - scroll_x * 0.1;
            let old_scale = self.graphics.renderer.camera.scale[0];
            self.graphics.renderer.camera.scale[0] = (old_scale * zoom_factor).clamp(5.0, max_x_zoom);
            let new_world_x = self.graphics.renderer.camera.position[0]
                + cursor_ndc[0] * self.graphics.renderer.camera.scale[0] * aspect;
            self.graphics.renderer.camera.position[0] += world_x - new_world_x;
        }

        if scroll_y.abs() > 0.001 {
            let zoom_factor = 1.0 + scroll_y * 0.1;
            let old_scale = self.graphics.renderer.camera.scale[1];
            self.graphics.renderer.camera.scale[1] = (old_scale * zoom_factor).clamp(1.0, max_y_zoom);
            let new_world_y = self.graphics.renderer.camera.position[1]
                + cursor_ndc[1] * self.graphics.renderer.camera.scale[1];
            self.graphics.renderer.camera.position[1] += world_y - new_world_y;
        }

        self.graphics.renderer.update_camera(&self.graphics.queue);
        self.update_visible_range();
        self.window.request_redraw();
    }

    // =========================================================================
    // UI Toggle Methods
    // =========================================================================

    fn toggle_macd_panel(&mut self) {
        self.ui.panels.show_macd_panel = !self.ui.panels.show_macd_panel;
        self.window.request_redraw();
    }

    fn toggle_symbol_picker(&mut self) {
        self.ui.panels.show_symbol_picker = !self.ui.panels.show_symbol_picker;
        self.window.request_redraw();
    }

    fn toggle_ta(&mut self) {
        self.document.ta_settings.show_ta = !self.document.ta_settings.show_ta;
        if self.document.ta_settings.show_ta {
            self.ensure_ta_computed();
            self.update_ta_buffers();
            if self.interaction.replay.enabled && self.interaction.ml_inference.is_some() {
                self.precompute_ml_ta();
            }
        }
        self.window.request_redraw();
    }

    // =========================================================================
    // Replay Methods
    // =========================================================================

    fn toggle_replay_mode(&mut self) {
        let was_enabled = self.interaction.replay.enabled;
        self.interaction.replay.toggle(self.document.current_timeframe);

        if !was_enabled && self.interaction.replay.enabled {
            if self.interaction.ml_inference.is_some() && self.document.ta_settings.show_ta {
                self.precompute_ml_ta();
            }
        } else if was_enabled && !self.interaction.replay.enabled {
            if self.document.ta_settings.show_ta {
                self.update_ta_buffers();
            }
        }
        self.window.request_redraw();
    }

    fn replay_step_forward(&mut self) {
        let base_candles = self.document.timeframes.first().map(|tf| &tf.candles[..]).unwrap_or(&[]);
        if self.interaction.replay.step_forward(base_candles) {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
            self.window.request_redraw();
        }
    }

    fn replay_step_backward(&mut self) {
        let base_candles = self.document.timeframes.first().map(|tf| &tf.candles[..]).unwrap_or(&[]);
        if self.interaction.replay.step_backward(base_candles) {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
            self.window.request_redraw();
        }
    }

    fn replay_increase_step_size(&mut self) {
        if self.interaction.replay.increase_step_size(self.document.current_timeframe) {
            self.window.request_redraw();
        }
    }

    fn replay_decrease_step_size(&mut self) {
        if self.interaction.replay.decrease_step_size() {
            self.window.request_redraw();
        }
    }

    fn set_replay_index(&mut self, index: usize) {
        let candles = self.document.current_candles();
        self.interaction.replay.set_index(index, candles);
        self.recompute_replay_candles();
        self.recompute_replay_ta();
        self.window.request_redraw();
    }

    fn recompute_replay_candles(&mut self) {
        let base_candles = self.document.timeframes.first().map(|tf| &tf.candles[..]).unwrap_or(&[]);
        let current_timeframe_idx = self.document.current_timeframe;
        let device = &self.graphics.device;
        let renderer = &self.graphics.renderer;

        self.interaction.replay.recompute_candles(
            base_candles,
            current_timeframe_idx,
            |candles, tf_label| renderer.create_timeframe_data(device, candles, tf_label),
        );
    }

    fn recompute_replay_ta(&mut self) {
        if !self.document.ta_settings.show_ta {
            self.interaction.replay.ta_data = None;
            return;
        }

        let replay_ts = match self.interaction.replay.timestamp {
            Some(ts) => ts,
            None => {
                self.interaction.replay.ta_data = None;
                return;
            }
        };

        let current_tf_candles: Vec<Candle> = if let Some(ref replay_candles) = self.interaction.replay.candles {
            replay_candles.clone()
        } else if let Some(replay_idx) = self.interaction.replay.index {
            let tf_candles = self.document.current_candles();
            if tf_candles.is_empty() || replay_idx == 0 {
                Vec::new()
            } else {
                tf_candles[..=replay_idx.min(tf_candles.len() - 1)].to_vec()
            }
        } else {
            self.interaction.replay.ta_data = None;
            return;
        };

        if current_tf_candles.is_empty() {
            self.interaction.replay.ta_data = Some(TimeframeTaData::with_data(Vec::new(), Vec::new(), Vec::new()));
            self.update_ta_buffers();
            return;
        }

        let current_tf_idx = self.document.current_timeframe;
        let replay_candle_idx = current_tf_candles.len().saturating_sub(1);

        let mut ta_data = if let Some(ta) = self.document.ta_data.get(current_tf_idx) {
            if ta.computed {
                let ranges: Vec<_> = ta.ranges.iter().filter(|r| r.end_index <= replay_candle_idx).cloned().collect();
                let levels: Vec<_> = ta.levels.iter().filter(|l| {
                    if l.created_at_index > replay_candle_idx { return false; }
                    if l.state == LevelState::Broken {
                        if let Some(ref break_event) = l.break_event {
                            return break_event.candle_index > replay_candle_idx;
                        }
                    }
                    true
                }).cloned().collect();
                let trends: Vec<_> = ta.trends.iter().filter(|t| {
                    if t.created_at_index > replay_candle_idx { return false; }
                    if t.state == TrendState::Broken {
                        if let Some(ref break_event) = t.break_event {
                            return break_event.candle_index > replay_candle_idx;
                        }
                    }
                    true
                }).cloned().collect();
                TimeframeTaData::with_data(ranges, levels, trends)
            } else {
                let ta_config = self.analyzer_config_for_timeframe(current_tf_idx);
                let mut analyzer = Analyzer::with_config(ta_config);
                for candle in &current_tf_candles {
                    analyzer.process_candle(*candle);
                }
                TimeframeTaData::with_data(analyzer.ranges().to_vec(), analyzer.all_levels().to_vec(), analyzer.all_trends().to_vec())
            }
        } else {
            TimeframeTaData::with_data(Vec::new(), Vec::new(), Vec::new())
        };

        if let Some(ref ml_inference) = self.interaction.ml_inference {
            let prediction = self.compute_ml_prediction(ml_inference, replay_ts, &current_tf_candles);
            if let Some(pred) = prediction {
                ta_data.set_prediction(pred);
            }
        }

        self.interaction.replay.ta_data = Some(ta_data);
        self.update_ta_buffers();
    }

    // =========================================================================
    // Data Loading Methods
    // =========================================================================

    /// Update the current candle with live data from WebSocket.
    fn update_live_candle(&mut self, candle: Candle, is_closed: bool) {
        let stored_open = if is_closed {
            if let Some(tf) = self.document.timeframes.first_mut() {
                tf.candles.push(candle);
            }
            self.reaggregate_timeframes();

            if let Some(candles) = self.document.timeframes.first().map(|tf| tf.candles.clone()) {
                let new_tf_data = self.graphics.renderer.create_timeframe_data(&self.graphics.device, candles, "1m");
                if let Some(tf) = self.document.timeframes.first_mut() {
                    *tf = new_tf_data;
                }
            }
            candle.open
        } else {
            let open_price = if let Some(tf) = self.document.timeframes.first_mut() {
                if let Some(last) = tf.candles.last_mut() {
                    last.high = last.high.max(candle.high);
                    last.low = last.low.min(candle.low);
                    last.close = candle.close;
                    last.volume = candle.volume;
                    last.open
                } else {
                    candle.open
                }
            } else {
                candle.open
            };

            if let Some(candles) = self.document.timeframes.first().map(|tf| tf.candles.clone()) {
                let new_tf_data = self.graphics.renderer.create_timeframe_data(&self.graphics.device, candles, "1m");
                if let Some(tf) = self.document.timeframes.first_mut() {
                    *tf = new_tf_data;
                }
            }
            open_price
        };

        self.update_visible_range();

        if self.interaction.ws_connected {
            self.update_current_price_line(Some(candle.close), stored_open);
        }

        self.window.request_redraw();
    }

    fn update_current_price_line(&mut self, close_price: Option<f32>, open_price: f32) {
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        if chart_height < 1.0 || chart_width < 1.0 {
            return;
        }
        let aspect = chart_width / chart_height;
        let (x_min, x_max) = self.graphics.renderer.camera.visible_x_range(aspect);
        self.graphics.renderer.set_current_price(&self.graphics.queue, close_price, open_price, x_min, x_max);
    }

    fn reaggregate_timeframes(&mut self) {
        let base_candles = self.document.timeframes.first().map(|tf| tf.candles.clone()).unwrap_or_default();
        let timeframe_types = Timeframe::all();

        for (i, tf) in timeframe_types.iter().enumerate().skip(1) {
            let candles = aggregate_candles(&base_candles, *tf);
            let new_tf_data = self.graphics.renderer.create_timeframe_data(&self.graphics.device, candles, tf.label());
            if let Some(tf_data) = self.document.timeframes.get_mut(i) {
                *tf_data = new_tf_data;
            }
        }
    }

    /// Switch to a different trading symbol.
    pub fn switch_symbol(&mut self, symbol: &str) {
        if symbol.to_uppercase() == self.interaction.current_symbol {
            return;
        }

        let symbol = symbol.to_uppercase();
        self.interaction.current_symbol = symbol.clone();
        self.interaction.loading_state = LoadingState::FetchingMexcData { symbol: symbol.clone() };

        let timeframe_types = Timeframe::all();
        for (i, tf) in timeframe_types.iter().enumerate() {
            let empty_tf = self.graphics.renderer.create_timeframe_data(&self.graphics.device, Vec::new(), tf.label());
            if let Some(tf_data) = self.document.timeframes.get_mut(i) {
                *tf_data = empty_tf;
            }
        }

        self.document.clear_ta_data();
        self.interaction.live_event_rx = None;
        self.interaction.ws_connected = false;

        if let Some(runtime) = &self.tokio_runtime {
            let sender = self.bg_sender.clone();
            let sym = symbol.clone();
            let db_path = self.app_config.sync.get_db_path();
            runtime.spawn(async move {
                let db_candles = if let Ok(db) = charter_sync::CandleDb::open(&db_path) {
                    match db.load_candles(&sym) {
                        Ok(candles) if !candles.is_empty() => {
                            log::info!("Loaded {} candles from DuckDB for {}", candles.len(), sym);
                            Some(candles)
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                if let Some(candles) = db_candles {
                    let _ = sender.send(BackgroundMessage::DataLoaded(candles));
                } else {
                    let source = MexcSource::new(&sym);
                    match source.load().await {
                        Ok(base_candles) => {
                            let _ = sender.send(BackgroundMessage::DataLoaded(base_candles));
                        }
                        Err(e) => {
                            let _ = sender.send(BackgroundMessage::Error(format!("Failed to load {}: {}", sym, e)));
                        }
                    }
                }
            });
        }

        self.window.request_redraw();
    }

    /// Start live data subscription for the current symbol.
    pub fn start_live_updates(&mut self) {
        if let Some(runtime) = &self.tokio_runtime {
            let sender = self.bg_sender.clone();
            let symbol = self.interaction.current_symbol.clone();

            runtime.spawn(async move {
                let mut manager = LiveDataManager::new();
                match manager.subscribe(&symbol).await {
                    Ok(mut rx) => {
                        let _ = sender.send(BackgroundMessage::ConnectionStatus(true));

                        while let Some(event) = rx.recv().await {
                            match event {
                                LiveDataEvent::CandleUpdate { candle, is_closed } => {
                                    let _ = sender.send(BackgroundMessage::LiveCandleUpdate { candle, is_closed });
                                }
                                LiveDataEvent::Connected => {
                                    let _ = sender.send(BackgroundMessage::ConnectionStatus(true));
                                }
                                LiveDataEvent::Disconnected => {
                                    let _ = sender.send(BackgroundMessage::ConnectionStatus(false));
                                }
                                LiveDataEvent::Error(e) => {
                                    eprintln!("Live data error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to subscribe to live updates: {}", e);
                        let _ = sender.send(BackgroundMessage::ConnectionStatus(false));
                    }
                }
            });
        }
    }

    /// Toggle historical data sync on/off.
    pub fn toggle_sync(&mut self) {
        self.sync_enabled = !self.sync_enabled;

        if self.sync_enabled {
            self.start_sync();
        } else {
            if let Some(manager) = &self.sync_manager {
                manager.cancel();
            }
            self.sync_state = charter_sync::SyncState::Idle;
        }
    }

    fn start_sync(&mut self) {
        if matches!(self.sync_state, charter_sync::SyncState::Syncing { .. }) {
            return;
        }

        self.sync_state = charter_sync::SyncState::Syncing {
            fetched: 0,
            estimated_total: 0,
            candles_per_sec: 0.0,
        };

        if let (Some(runtime), Some(manager)) = (&self.tokio_runtime, &self.sync_manager) {
            let symbol = self.interaction.current_symbol.clone();
            let (mut progress_rx, handle) = manager.start_sync(symbol, runtime.handle());
            let sender = self.bg_sender.clone();

            runtime.spawn(async move {
                while let Some(progress) = progress_rx.recv().await {
                    let _ = sender.send(BackgroundMessage::SyncProgressUpdate(progress));
                }

                match handle.await {
                    Ok(Ok(total)) => {
                        let _ = sender.send(BackgroundMessage::SyncComplete { total_candles: total });
                    }
                    Ok(Err(e)) => {
                        let _ = sender.send(BackgroundMessage::SyncError(e.to_string()));
                    }
                    Err(e) => {
                        let _ = sender.send(BackgroundMessage::SyncError(format!("Task panicked: {}", e)));
                    }
                }
            });
        }
    }

    // =========================================================================
    // TA Methods
    // =========================================================================

    fn analyzer_config_for_timeframe(&self, tf_idx: usize) -> AnalyzerConfig {
        let tf_label = Timeframe::all()[tf_idx].label();
        let ta_config = self.app_config.ta_analysis_for_timeframe(tf_label);
        AnalyzerConfig::default()
            .doji_threshold(ta_config.doji_threshold)
            .min_range_candles(ta_config.min_range_candles)
            .level_tolerance(ta_config.level_tolerance)
            .create_greedy_levels(ta_config.create_greedy_levels)
    }

    fn compute_ta_background(&self, timeframe: usize) {
        let candles = self.document.timeframes.get(timeframe).map(|tf| tf.candles.clone()).unwrap_or_default();
        let sender = self.bg_sender.clone();
        let ta_config = self.analyzer_config_for_timeframe(timeframe);

        thread::spawn(move || {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(LoadingState::ComputingTa { timeframe }));

            let mut analyzer = Analyzer::with_config(ta_config);

            for candle in &candles {
                analyzer.process_candle(*candle);
            }

            let ranges = analyzer.ranges().to_vec();
            let levels = analyzer.all_levels().to_vec();
            let trends = analyzer.all_trends().to_vec();

            let _ = sender.send(BackgroundMessage::TaComputed { timeframe, ranges, levels, trends });
        });
    }

    fn precompute_ml_ta(&self) {
        const ML_TIMEFRAME_INDICES: [usize; 4] = [2, 4, 8, 9];
        const MIN_CANDLES: usize = 100;

        let mut timeframes_to_compute: Vec<(usize, Vec<Candle>, AnalyzerConfig)> = Vec::new();

        for &tf_idx in &ML_TIMEFRAME_INDICES {
            if tf_idx >= self.document.timeframes.len() {
                continue;
            }
            if let Some(ta) = self.document.ta_data.get(tf_idx) {
                if ta.computed {
                    continue;
                }
            }
            let candles = &self.document.timeframes[tf_idx].candles;
            if candles.len() < MIN_CANDLES {
                continue;
            }
            let ta_config = self.analyzer_config_for_timeframe(tf_idx);
            timeframes_to_compute.push((tf_idx, candles.clone(), ta_config));
        }

        if timeframes_to_compute.is_empty() {
            return;
        }

        let total = timeframes_to_compute.len();
        let sender = self.bg_sender.clone();

        let _ = sender.send(BackgroundMessage::MlTaProgress { completed: 0, total });

        thread::spawn(move || {
            for (completed, (tf_idx, candles, ta_config)) in timeframes_to_compute.into_iter().enumerate() {
                let mut analyzer = Analyzer::with_config(ta_config);
                for candle in &candles {
                    analyzer.process_candle(*candle);
                }
                let ranges = analyzer.ranges().to_vec();
                let levels = analyzer.all_levels().to_vec();
                let trends = analyzer.all_trends().to_vec();

                let _ = sender.send(BackgroundMessage::TaComputed { timeframe: tf_idx, ranges, levels, trends });
                let _ = sender.send(BackgroundMessage::MlTaProgress { completed: completed + 1, total });
            }
        });
    }

    fn ensure_ta_computed(&mut self) {
        let tf_idx = self.document.current_timeframe;

        if let Some(ta) = self.document.ta_data.get(tf_idx) {
            if ta.computed {
                return;
            }
        }

        if self.interaction.loading_state.is_loading() {
            return;
        }

        if let Some(tf) = self.document.timeframes.get(tf_idx) {
            if tf.candles.is_empty() {
                return;
            }
        }

        self.compute_ta_background(tf_idx);
    }

    fn update_ta_buffers(&mut self) {
        let tf_idx = self.document.current_timeframe;
        let tf = match self.document.timeframes.get(tf_idx) {
            Some(tf) => tf,
            None => return,
        };

        // Get TA data - use replay data if in locked replay mode, otherwise use document data
        // Clone to avoid borrow issues since we need to borrow self mutably later for queue writes
        let ta: TimeframeTaData = if self.interaction.replay.is_locked() {
            self.interaction.replay.ta_data.clone().unwrap_or_else(|| {
                self.document.ta_data.get(tf_idx).cloned().unwrap_or_default()
            })
        } else {
            self.document.ta_data.get(tf_idx).cloned().unwrap_or_default()
        };

        // Convert ranges to GPU format
        let mut range_gpus: Vec<RangeGpu> = ta.ranges.iter()
            .filter(|_| self.document.ta_settings.show_ranges)
            .take(MAX_TA_RANGES)
            .map(|r| RangeGpu {
                x_start: r.start_index as f32 * CANDLE_SPACING,
                x_end: r.end_index as f32 * CANDLE_SPACING,
                y_pos: r.low,
                is_bullish: if r.direction == CandleDirection::Bullish { 1 } else { 0 },
            })
            .collect();

        let range_count = range_gpus.len() as u32;
        while range_gpus.len() < MAX_TA_RANGES {
            range_gpus.push(RangeGpu { x_start: 0.0, x_end: 0.0, y_pos: 0.0, is_bullish: 0 });
        }

        // Filter and convert levels
        let filtered_levels: Vec<&Level> = ta.levels.iter()
            .filter(|l| {
                let type_ok = match l.level_type {
                    LevelType::Hold => self.document.ta_settings.show_hold_levels,
                    LevelType::GreedyHold => self.document.ta_settings.show_greedy_levels,
                };
                let state_ok = match l.state {
                    LevelState::Inactive | LevelState::Active => self.document.ta_settings.show_active_levels,
                    LevelState::Broken => self.document.ta_settings.show_broken_levels,
                };
                type_ok && state_ok
            })
            .take(MAX_TA_LEVELS)
            .collect();

        let level_count = filtered_levels.len() as u32;
        let mut level_gpus: Vec<LevelGpu> = filtered_levels.iter()
            .map(|l| {
                let (r, g, b, a) = match (l.direction, l.state) {
                    (CandleDirection::Bullish, LevelState::Inactive) => (0.0, 0.5, 0.3, 0.4),
                    (CandleDirection::Bullish, LevelState::Active) => (0.0, 0.8, 0.4, 0.7),
                    (CandleDirection::Bullish, LevelState::Broken) => (0.0, 0.3, 0.2, 0.3),
                    (CandleDirection::Bearish, LevelState::Inactive) => (0.5, 0.15, 0.15, 0.4),
                    (CandleDirection::Bearish, LevelState::Active) => (0.8, 0.2, 0.2, 0.7),
                    (CandleDirection::Bearish, LevelState::Broken) => (0.3, 0.1, 0.1, 0.3),
                    (CandleDirection::Doji, _) => (0.5, 0.5, 0.5, 0.5),
                };
                LevelGpu {
                    y_value: l.price,
                    x_start: l.source_candle_index as f32 * CANDLE_SPACING,
                    r, g, b, a,
                    level_type: if l.level_type == LevelType::Hold { 0 } else { 1 },
                    hit_count: l.hits.len() as u32,
                }
            })
            .collect();

        while level_gpus.len() < MAX_TA_LEVELS {
            level_gpus.push(LevelGpu { y_value: 0.0, x_start: 0.0, r: 0.0, g: 0.0, b: 0.0, a: 0.0, level_type: 0, hit_count: 0 });
        }

        // Filter and convert trends
        let filtered_trends: Vec<&Trend> = ta.trends.iter()
            .filter(|t| {
                if !self.document.ta_settings.show_trends { return false; }
                match t.state {
                    TrendState::Active => self.document.ta_settings.show_active_trends,
                    TrendState::Hit => self.document.ta_settings.show_hit_trends,
                    TrendState::Broken => self.document.ta_settings.show_broken_trends,
                }
            })
            .take(MAX_TA_TRENDS)
            .collect();

        let trend_count = filtered_trends.len() as u32;
        let mut trend_gpus: Vec<TrendGpu> = filtered_trends.iter()
            .map(|t| {
                let (r, g, b, a) = match (t.direction, t.state) {
                    (CandleDirection::Bullish, TrendState::Active) => (0.0, 0.9, 0.5, 0.8),
                    (CandleDirection::Bullish, TrendState::Hit) => (0.0, 0.7, 0.4, 0.6),
                    (CandleDirection::Bullish, TrendState::Broken) => (0.0, 0.4, 0.2, 0.4),
                    (CandleDirection::Bearish, TrendState::Active) => (0.9, 0.3, 0.3, 0.8),
                    (CandleDirection::Bearish, TrendState::Hit) => (0.7, 0.2, 0.2, 0.6),
                    (CandleDirection::Bearish, TrendState::Broken) => (0.4, 0.1, 0.1, 0.4),
                    (CandleDirection::Doji, _) => (0.5, 0.5, 0.5, 0.5),
                };
                TrendGpu {
                    x_start: t.start.candle_index as f32 * CANDLE_SPACING,
                    y_start: t.start.price,
                    x_end: t.end.candle_index as f32 * CANDLE_SPACING,
                    y_end: t.end.price,
                    r, g, b, a,
                }
            })
            .collect();

        while trend_gpus.len() < MAX_TA_TRENDS {
            trend_gpus.push(TrendGpu { x_start: 0.0, y_start: 0.0, x_end: 0.0, y_end: 0.0, r: 0.0, g: 0.0, b: 0.0, a: 0.0 });
        }

        // Compute params
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        if chart_height < 1.0 || chart_width < 1.0 {
            return;
        }
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);
        let (_, visible_x_max) = self.graphics.renderer.camera.visible_x_range(aspect);
        let candle_x_max = (tf.candles.len() as f32) * CANDLE_SPACING;
        let x_max = visible_x_max.max(candle_x_max);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        let level_thickness = (1.5 * world_units_per_pixel).max(price_range * 0.0008);
        let range_thickness = (2.0 * world_units_per_pixel).max(price_range * 0.001);

        let ta_params = TaRenderParams {
            first_visible: self.graphics.renderer.visible_start,
            candle_spacing: CANDLE_SPACING,
            range_thickness,
            level_thickness,
            x_max,
            range_count,
            level_count,
            trend_count,
        };

        // Write to GPU buffers
        self.graphics.queue.write_buffer(&tf.ta_range_buffer, 0, bytemuck::cast_slice(&range_gpus));
        self.graphics.queue.write_buffer(&tf.ta_level_buffer, 0, bytemuck::cast_slice(&level_gpus));
        self.graphics.queue.write_buffer(&tf.ta_trend_buffer, 0, bytemuck::cast_slice(&trend_gpus));
        self.graphics.queue.write_buffer(&tf.ta_params_buffer, 0, bytemuck::cast_slice(&[ta_params]));
    }

    // =========================================================================
    // MACD/Indicator Methods
    // =========================================================================

    fn update_macd_params(&mut self) {
        let visible_start = self.graphics.renderer.visible_start as usize;
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        if chart_height < 1.0 || chart_width < 1.0 {
            return;
        }
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        let line_thickness = (world_units_per_pixel * 1.5).max(0.001);

        for instance in self.document.indicators.iter() {
            let buffers = match &instance.gpu_buffers {
                Some(IndicatorGpuBuffers::Macd(b)) => b,
                _ => continue,
            };

            let macd_first_visible = if visible_start > buffers.macd_start_index {
                (visible_start - buffers.macd_start_index) as u32
            } else {
                0
            };
            let signal_first_visible = if visible_start > buffers.signal_start_index {
                (visible_start - buffers.signal_start_index) as u32
            } else {
                0
            };

            let macd_params = IndicatorParams {
                first_visible: macd_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness,
                count: buffers.macd_point_count,
            };
            self.graphics.queue.write_buffer(&buffers.params_buffer, 0, bytemuck::cast_slice(&[macd_params]));

            let signal_params = IndicatorParams {
                first_visible: signal_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness,
                count: buffers.signal_point_count,
            };
            self.graphics.queue.write_buffer(&buffers.signal_params_buffer, 0, bytemuck::cast_slice(&[signal_params]));
        }
    }

    /// Add a new MACD indicator instance.
    pub fn add_macd(&mut self, config: MacdConfig) {
        let num_timeframes = self.document.timeframes.len();
        let dyn_macd = DynMacd::new(config);
        let label = dyn_macd.label();

        let id = self.document.indicators.add(dyn_macd, label, num_timeframes);

        let candles = self.document.current_candles().to_vec();
        let tf = self.document.current_timeframe;
        if let Some(instance) = self.document.indicators.get_by_id_mut(id) {
            Self::compute_macd_for_instance_internal(instance, &candles, tf);
        }

        let idx = self.document.indicators.index_of(id).unwrap();
        self.create_macd_gpu_buffers_for_instance(idx);
    }

    /// Remove a MACD indicator by index.
    pub fn remove_macd(&mut self, index: usize) {
        self.document.indicators.remove(index);
    }

    fn compute_macd_for_instance_internal(instance: &mut crate::indicators::IndicatorInstance, candles: &[Candle], timeframe: usize) {
        if candles.is_empty() {
            instance.outputs[timeframe] = None;
            instance.macd_outputs[timeframe] = None;
            return;
        }

        let output = instance.indicator.calculate(candles);
        instance.outputs[timeframe] = Some(output);

        if let Some(config) = instance.macd_config() {
            let macd = Macd::new(config.clone());
            let macd_output = macd.calculate_macd(candles);
            instance.macd_outputs[timeframe] = Some(macd_output);
        }
    }

    fn recompute_all_macd(&mut self) {
        let tf = self.document.current_timeframe;
        let candles = self.document.current_candles().to_vec();

        for instance in self.document.indicators.iter_mut() {
            if candles.is_empty() {
                instance.outputs[tf] = None;
                instance.macd_outputs[tf] = None;
                continue;
            }

            let output = instance.indicator.calculate(&candles);
            instance.outputs[tf] = Some(output);

            if let Some(config) = instance.macd_config() {
                let macd = Macd::new(config.clone());
                let macd_output = macd.calculate_macd(&candles);
                instance.macd_outputs[tf] = Some(macd_output);
            }
        }

        self.update_macd_gpu_buffers();
    }

    fn create_macd_gpu_buffers_for_instance(&mut self, instance_idx: usize) {
        let instance = match self.document.indicators.get(instance_idx) {
            Some(i) => i,
            None => return,
        };
        let tf = self.document.current_timeframe;
        let conversion = self.convert_macd_to_gpu_points_from_instance(instance, tf);

        let macd_line_buffer = self.graphics.renderer.indicator_pipeline.create_indicator_buffer(&self.graphics.device, &conversion.macd_points);
        let signal_line_buffer = self.graphics.renderer.indicator_pipeline.create_indicator_buffer(&self.graphics.device, &conversion.signal_points);
        let histogram_buffer = self.graphics.renderer.indicator_pipeline.create_indicator_buffer(&self.graphics.device, &conversion.histogram_points);
        let params_buffer = self.graphics.renderer.indicator_pipeline.create_indicator_params_buffer(&self.graphics.device);
        let signal_params_buffer = self.graphics.renderer.indicator_pipeline.create_indicator_params_buffer(&self.graphics.device);

        let macd_bind_group = self.graphics.renderer.indicator_pipeline.create_bind_group(&self.graphics.device, &macd_line_buffer, &params_buffer);
        let signal_bind_group = self.graphics.renderer.indicator_pipeline.create_bind_group(&self.graphics.device, &signal_line_buffer, &signal_params_buffer);
        let histogram_bind_group = self.graphics.renderer.indicator_pipeline.create_bind_group(&self.graphics.device, &histogram_buffer, &params_buffer);

        let buffers = MacdGpuBuffers {
            macd_line_buffer,
            signal_line_buffer,
            histogram_buffer,
            params_buffer,
            signal_params_buffer,
            macd_bind_group,
            signal_bind_group,
            histogram_bind_group,
            macd_point_count: conversion.macd_points.len() as u32,
            signal_point_count: conversion.signal_points.len() as u32,
            histogram_point_count: conversion.histogram_points.len() as u32,
            macd_start_index: conversion.macd_start_index,
            signal_start_index: conversion.signal_start_index,
        };

        if let Some(instance) = self.document.indicators.get_mut(instance_idx) {
            instance.gpu_buffers = Some(IndicatorGpuBuffers::Macd(buffers));
        }
    }

    fn convert_macd_to_gpu_points_from_instance(&self, instance: &crate::indicators::IndicatorInstance, timeframe: usize) -> MacdConversionResult {
        let output = match &instance.macd_outputs[timeframe] {
            Some(o) => o,
            None => return MacdConversionResult::empty(),
        };

        let config = match instance.macd_config() {
            Some(c) => c,
            None => return MacdConversionResult::empty(),
        };

        let macd_start_index = output.macd_line.start_index();
        let signal_start_index = output.signal_line.start_index();

        let mut macd_points: Vec<IndicatorPointGpu> = output.macd_line.iter()
            .map(|(idx, &val)| IndicatorPointGpu {
                x: idx as f32 * CANDLE_SPACING,
                y: val,
                r: config.macd_color[0],
                g: config.macd_color[1],
                b: config.macd_color[2],
                _padding: 0.0,
            })
            .collect();

        let mut signal_points: Vec<IndicatorPointGpu> = output.signal_line.iter()
            .map(|(idx, &val)| IndicatorPointGpu {
                x: idx as f32 * CANDLE_SPACING,
                y: val,
                r: config.signal_color[0],
                g: config.signal_color[1],
                b: config.signal_color[2],
                _padding: 0.0,
            })
            .collect();

        let mut histogram_points: Vec<IndicatorPointGpu> = output.histogram.iter()
            .map(|(idx, &val)| {
                let color = if val >= 0.0 { config.histogram_pos_color } else { config.histogram_neg_color };
                IndicatorPointGpu {
                    x: idx as f32 * CANDLE_SPACING,
                    y: val,
                    r: color[0],
                    g: color[1],
                    b: color[2],
                    _padding: 0.0,
                }
            })
            .collect();

        if macd_points.is_empty() {
            macd_points.push(IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 });
        }
        if signal_points.is_empty() {
            signal_points.push(IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 });
        }
        if histogram_points.is_empty() {
            histogram_points.push(IndicatorPointGpu { x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0, _padding: 0.0 });
        }

        MacdConversionResult { macd_points, signal_points, histogram_points, macd_start_index, signal_start_index }
    }

    fn update_macd_gpu_buffers(&mut self) {
        let tf = self.document.current_timeframe;
        let visible_start = self.graphics.renderer.visible_start as usize;

        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        if chart_height < 1.0 || chart_width < 1.0 {
            return;
        }
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        let line_thickness = (world_units_per_pixel * 1.5).max(0.001);

        for i in 0..self.document.indicators.len() {
            let instance = match self.document.indicators.get(i) {
                Some(inst) => inst,
                None => continue,
            };

            if instance.gpu_buffers.is_none() {
                self.create_macd_gpu_buffers_for_instance(i);
                continue;
            }

            let conversion = self.convert_macd_to_gpu_points_from_instance(instance, tf);

            let instance = self.document.indicators.get(i).unwrap();
            let buffers = match &instance.gpu_buffers {
                Some(IndicatorGpuBuffers::Macd(b)) => b,
                _ => continue,
            };

            self.graphics.queue.write_buffer(&buffers.macd_line_buffer, 0, bytemuck::cast_slice(&conversion.macd_points));
            self.graphics.queue.write_buffer(&buffers.signal_line_buffer, 0, bytemuck::cast_slice(&conversion.signal_points));
            self.graphics.queue.write_buffer(&buffers.histogram_buffer, 0, bytemuck::cast_slice(&conversion.histogram_points));

            let macd_first_visible = if visible_start > conversion.macd_start_index { (visible_start - conversion.macd_start_index) as u32 } else { 0 };
            let signal_first_visible = if visible_start > conversion.signal_start_index { (visible_start - conversion.signal_start_index) as u32 } else { 0 };

            let macd_params = IndicatorParams {
                first_visible: macd_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness,
                count: conversion.macd_points.len() as u32,
            };
            self.graphics.queue.write_buffer(&buffers.params_buffer, 0, bytemuck::cast_slice(&[macd_params]));

            let signal_params = IndicatorParams {
                first_visible: signal_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness,
                count: conversion.signal_points.len() as u32,
            };
            self.graphics.queue.write_buffer(&buffers.signal_params_buffer, 0, bytemuck::cast_slice(&[signal_params]));

            if let Some(instance) = self.document.indicators.get_mut(i)
                && let Some(IndicatorGpuBuffers::Macd(buffers)) = &mut instance.gpu_buffers
            {
                buffers.macd_start_index = conversion.macd_start_index;
                buffers.signal_start_index = conversion.signal_start_index;
                buffers.macd_point_count = conversion.macd_points.len() as u32;
                buffers.signal_point_count = conversion.signal_points.len() as u32;
                buffers.histogram_point_count = conversion.histogram_points.len() as u32;
            }
        }
    }

    // =========================================================================
    // ML Methods
    // =========================================================================

    fn calculate_rsi_for_candles(candles: &[Candle], current_idx: usize, period: usize) -> f32 {
        if current_idx < period + 1 || candles.is_empty() {
            return 0.5;
        }

        let start_idx = current_idx.saturating_sub(100.min(current_idx));
        let lookback_candles = &candles[start_idx..=current_idx];

        if lookback_candles.len() < period + 1 {
            return 0.5;
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..lookback_candles.len() {
            let change = lookback_candles[i].close - lookback_candles[i - 1].close;
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        if gains.len() < period {
            return 0.5;
        }

        let mut avg_gain: f32 = gains.iter().take(period).sum::<f32>() / period as f32;
        let mut avg_loss: f32 = losses.iter().take(period).sum::<f32>() / period as f32;

        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f32 + gains[i]) / period as f32;
            avg_loss = (avg_loss * (period - 1) as f32 + losses[i]) / period as f32;
        }

        if avg_loss == 0.0 {
            return 1.0;
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi / 100.0
    }

    fn compute_ml_prediction(&self, ml_inference: &MlInferenceHandle, replay_ts: f64, current_tf_candles: &[Candle]) -> Option<charter_ta::MlPrediction> {
        const MIN_CANDLES: usize = 100;
        const ML_TIMEFRAME_INDICES: [usize; 4] = [2, 4, 8, 9];
        const LOOKBACK_SECONDS: f64 = 21.0 * 24.0 * 3600.0;
        let lookback_start_ts = replay_ts - LOOKBACK_SECONDS;

        let current_candle = current_tf_candles.last()?;
        let current_candle_idx = current_tf_candles.len().saturating_sub(1);
        let current_price = current_candle.close;

        let mut tf_features: Vec<TimeframeFeatures> = Vec::new();

        for (feature_idx, &tf_idx) in ML_TIMEFRAME_INDICES.iter().enumerate() {
            let ta_config = self.analyzer_config_for_timeframe(tf_idx);
            if tf_idx >= self.document.timeframes.len() {
                continue;
            }
            let tf_data = &self.document.timeframes[tf_idx];
            let candles = &tf_data.candles;
            if candles.len() < MIN_CANDLES {
                continue;
            }

            let end_idx = candles
                .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|i| i.saturating_sub(1));

            if end_idx == 0 {
                continue;
            }

            let tf_candle_idx = end_idx.min(candles.len().saturating_sub(1));

            let (levels, trends, feature_candle_idx) = if let Some(ta) = self.document.ta_data.get(tf_idx) {
                if ta.computed {
                    let levels: Vec<_> = ta.levels.iter()
                        .filter(|l| {
                            if l.created_at_index > tf_candle_idx { return false; }
                            if l.state == LevelState::Broken {
                                if let Some(ref break_event) = l.break_event {
                                    return break_event.candle_index > tf_candle_idx;
                                }
                            }
                            true
                        })
                        .cloned()
                        .collect();
                    let trends: Vec<_> = ta.trends.iter()
                        .filter(|t| {
                            if t.created_at_index > tf_candle_idx { return false; }
                            if t.state == TrendState::Broken {
                                if let Some(ref break_event) = t.break_event {
                                    return break_event.candle_index > tf_candle_idx;
                                }
                            }
                            true
                        })
                        .cloned()
                        .collect();
                    (levels, trends, tf_candle_idx)
                } else {
                    let start_idx = candles
                        .binary_search_by(|c| c.timestamp.partial_cmp(&lookback_start_ts).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or_else(|i| i);

                    let mut analyzer = Analyzer::with_config(ta_config.clone());
                    for candle in &candles[start_idx..=tf_candle_idx] {
                        analyzer.process_candle(*candle);
                    }
                    let relative_idx = tf_candle_idx - start_idx;
                    (analyzer.all_levels().to_vec(), analyzer.all_trends().to_vec(), relative_idx)
                }
            } else {
                continue;
            };

            let features = TimeframeFeatures::extract(feature_idx, &levels, &trends, current_price, feature_candle_idx);
            tf_features.push(features);
        }

        if tf_features.is_empty() {
            return None;
        }

        let prev_close = if current_candle_idx > 0 {
            current_tf_candles.get(current_candle_idx - 1).map(|c| c.close).unwrap_or(current_candle.open)
        } else {
            current_candle.open
        };
        let price_change = if prev_close > 0.0 { (current_candle.close - prev_close) / prev_close } else { 0.0 };

        let body = (current_candle.close - current_candle.open).abs();
        let range = current_candle.high - current_candle.low;
        let body_ratio = if range > f32::EPSILON { body / range } else { 0.5 };

        let volume_sum: f32 = current_tf_candles.iter().rev().take(100).map(|c| c.volume).sum();
        let volume_count = current_tf_candles.len().min(100) as f32;
        let avg_volume = volume_sum / volume_count.max(1.0);
        let volume_normalized = if avg_volume > f32::EPSILON { current_candle.volume / avg_volume } else { 1.0 };

        let rsi_14 = Self::calculate_rsi_for_candles(current_tf_candles, current_candle_idx, 14);

        let ml_features = MlFeatures {
            timeframes: tf_features,
            current_price,
            current_volume_normalized: volume_normalized,
            price_change_normalized: price_change,
            body_ratio,
            is_bullish: if current_candle.close > current_candle.open { 1.0 } else { 0.0 },
            rsi_14,
        };

        let feature_count = ml_features.feature_count();
        if feature_count != 302 {
            eprintln!("ML feature count mismatch: got {} (from {} timeframes), expected 302", feature_count, ml_features.timeframes.len());
            return None;
        }

        match ml_inference.predict(&ml_features) {
            Ok(prediction) => Some(prediction),
            Err(e) => {
                static LAST_ERROR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                let last = LAST_ERROR.load(std::sync::atomic::Ordering::Relaxed);
                if now > last + 5 {
                    LAST_ERROR.store(now, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("ML inference error: {}", e);
                }
                None
            }
        }
    }

    // =========================================================================
    // Hover State Methods
    // =========================================================================

    fn get_cursor_candle_index(&self) -> usize {
        let tf = match self.document.timeframes.get(self.document.current_timeframe) {
            Some(tf) => tf,
            None => return 0,
        };
        if let Some(pos) = self.input.last_mouse_pos {
            let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
            if pos[0] < chart_width {
                let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
                if chart_height < 1.0 || chart_width < 1.0 {
                    return tf.candles.len().saturating_sub(1);
                }
                let aspect = chart_width / chart_height;
                let (x_min, _) = self.graphics.renderer.camera.visible_x_range(aspect);
                let normalized_x = pos[0] / chart_width;
                let world_x = x_min + normalized_x * (self.graphics.renderer.camera.scale[0] * aspect * 2.0);
                let idx = (world_x / CANDLE_SPACING).round() as usize;
                return idx.min(tf.candles.len().saturating_sub(1));
            }
        }
        tf.candles.len().saturating_sub(1)
    }

    fn get_cursor_world_pos(&self) -> Option<(f32, f32)> {
        let pos = self.input.last_mouse_pos?;
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);

        if pos[0] >= chart_width || pos[1] >= chart_height {
            return None;
        }

        if chart_height < 1.0 || chart_width < 1.0 {
            return None;
        }

        let aspect = chart_width / chart_height;
        let (x_min, _) = self.graphics.renderer.camera.visible_x_range(aspect);
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);

        let normalized_x = pos[0] / chart_width;
        let normalized_y = pos[1] / chart_height;

        let world_x = x_min + normalized_x * (self.graphics.renderer.camera.scale[0] * aspect * 2.0);
        let world_y = y_max - normalized_y * (y_max - y_min);

        Some((world_x, world_y))
    }

    fn find_hovered_range(&self) -> Option<usize> {
        if !self.document.ta_settings.show_ta || !self.document.ta_settings.show_ranges {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = self.document.ta_data.get(self.document.current_timeframe)?;

        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        if chart_height < 1.0 || chart_width < 1.0 {
            return None;
        }
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let tolerance = price_range * 0.01;

        for (i, range) in ta.ranges.iter().enumerate() {
            let x_start = range.start_index as f32 * CANDLE_SPACING;
            let x_end = range.end_index as f32 * CANDLE_SPACING;

            if world_x >= x_start && world_x <= x_end {
                if (world_y - range.low).abs() < tolerance {
                    return Some(i);
                }
            }
        }

        None
    }

    fn find_hovered_level(&self) -> Option<usize> {
        if !self.document.ta_settings.show_ta {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = self.document.ta_data.get(self.document.current_timeframe)?;

        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.graphics.config.width as f32 - STATS_PANEL_WIDTH;
        if chart_height < 1.0 || chart_width < 1.0 {
            return None;
        }
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let tolerance = price_range * 0.005;

        for (i, level) in ta.levels.iter().enumerate() {
            let type_visible = match level.level_type {
                LevelType::Hold => self.document.ta_settings.show_hold_levels,
                LevelType::GreedyHold => self.document.ta_settings.show_greedy_levels,
            };
            let state_visible = match level.state {
                LevelState::Inactive | LevelState::Active => self.document.ta_settings.show_active_levels,
                LevelState::Broken => self.document.ta_settings.show_broken_levels,
            };

            if !type_visible || !state_visible {
                continue;
            }

            let x_start = level.created_at_index as f32 * CANDLE_SPACING;

            if world_x >= x_start && (world_y - level.price).abs() < tolerance {
                return Some(i);
            }
        }

        None
    }

    fn find_hovered_trend(&self) -> Option<usize> {
        if !self.document.ta_settings.show_ta || !self.document.ta_settings.show_trends {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = self.document.ta_data.get(self.document.current_timeframe)?;

        let chart_height = self.graphics.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let y_scale = self.graphics.renderer.camera.scale[1] * 2.0 / chart_height;
        let tolerance = y_scale * 10.0;

        for (i, trend) in ta.trends.iter().enumerate() {
            let state_visible = match trend.state {
                TrendState::Active => self.document.ta_settings.show_active_trends,
                TrendState::Hit => self.document.ta_settings.show_hit_trends,
                TrendState::Broken => self.document.ta_settings.show_broken_trends,
            };

            if !state_visible {
                continue;
            }

            let start_x = trend.start.candle_index as f32 * CANDLE_SPACING;

            if world_x < start_x {
                continue;
            }

            let candle_pos = world_x / CANDLE_SPACING;
            let dx = trend.end.candle_index as f32 - trend.start.candle_index as f32;
            let trend_price = if dx.abs() < f32::EPSILON {
                trend.start.price
            } else {
                let slope = (trend.end.price - trend.start.price) / dx;
                let x = candle_pos - trend.start.candle_index as f32;
                trend.start.price + slope * x
            };

            if (world_y - trend_price).abs() < tolerance {
                return Some(i);
            }
        }

        None
    }

    fn update_hover_state(&mut self) {
        self.document.hovered_range = self.find_hovered_range();
        self.document.hovered_level = if self.document.hovered_range.is_none() {
            self.find_hovered_level()
        } else {
            None
        };
        self.document.hovered_trend = if self.document.hovered_range.is_none() && self.document.hovered_level.is_none() {
            self.find_hovered_trend()
        } else {
            None
        };
    }

    // =========================================================================
    // Render Method
    // =========================================================================

    /// Render a frame.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // Process any pending background messages
        self.process_background_messages();

        if !self.graphics.is_surface_configured {
            return Ok(());
        }

        let output = self.graphics.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let total_width = self.graphics.config.width as f32;
        let total_height = self.graphics.config.height as f32;
        let chart_width = (total_width - STATS_PANEL_WIDTH).floor().max(1.0);
        let chart_height = (total_height * (1.0 - VOLUME_HEIGHT_RATIO)).floor().max(1.0);
        let volume_height = (total_height - chart_height).max(1.0);

        // Get data needed for egui
        let cursor_candle_idx = self.get_cursor_candle_index();
        let candle = self.document.current_candles().get(cursor_candle_idx).copied();
        let candle_count = self.document.current_candles().len();

        // Build hovered info for TA panel
        let ta_hovered = TaHoveredInfo {
            range: self.document.hovered_range.and_then(|idx| {
                self.document.ta_data.get(self.document.current_timeframe).and_then(|ta| {
                    ta.ranges.get(idx).map(|r| (r.direction, r.candle_count, r.high, r.low, r.start_index, r.end_index))
                })
            }),
            level: self.document.hovered_level.and_then(|idx| {
                self.document.ta_data.get(self.document.current_timeframe).and_then(|ta| {
                    ta.levels.get(idx).map(|l| (l.price, l.level_type, l.direction, l.state, l.hits.len()))
                })
            }),
            trend: self.document.hovered_trend.and_then(|idx| {
                self.document.ta_data.get(self.document.current_timeframe).and_then(|ta| {
                    ta.trends.get(idx).map(|t| (t.direction, t.state, t.start.price, t.end.price, t.start.candle_index, t.end.candle_index, t.hits.len()))
                })
            }),
        };

        let show_macd = self.ui.panels.show_macd_panel;
        let should_show_symbol_picker = self.ui.panels.show_symbol_picker;
        let current_symbol = self.interaction.current_symbol.clone();
        let current_timeframe = self.document.current_timeframe;
        let ws_connected = self.interaction.ws_connected;
        let sync_enabled = self.sync_enabled;
        let sync_state = self.sync_state.clone();
        let new_timeframe = RefCell::new(None::<usize>);
        let should_toggle_sync = RefCell::new(false);

        let ta_settings = RefCell::new(self.document.ta_settings.clone());
        let symbol_picker_state = RefCell::new(self.ui.symbol_picker_state.clone());

        let ta_data = if self.interaction.replay.is_locked() {
            self.interaction.replay.ta_data.clone().unwrap_or_else(|| {
                self.document.ta_data.get(self.document.current_timeframe).cloned().unwrap_or_default()
            })
        } else {
            self.document.ta_data.get(self.document.current_timeframe).cloned().unwrap_or_default()
        };

        let indicators = &self.document.indicators;
        let loading_state = self.interaction.loading_state.clone();
        let screen_width = self.graphics.config.width as f32;
        let macd_response = RefCell::new(MacdPanelResponse::default());
        let symbol_picker_response = RefCell::new(None::<(bool, Option<String>)>);

        let drawing_tool = self.drawing.tool;
        let drawing_snap_enabled = self.drawing.snap_enabled;
        let drawing_toolbar_response = RefCell::new(DrawingToolbarResponse::default());

        let guideline_values = self.graphics.renderer.guideline_values.clone();
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);

        // Build egui UI
        let raw_input = self.ui.egui_state.take_egui_input(&self.window);
        let full_output = self.ui.egui_ctx.run(raw_input, |ctx| {
            // Draw price labels on the left side
            egui::Area::new(egui::Id::new("price_labels"))
                .fixed_pos(egui::Pos2::ZERO)
                .order(egui::Order::Foreground)
                .interactable(false)
                .show(ctx, |ui| {
                    let painter = ui.painter();
                    for &price in &guideline_values {
                        let normalized_y = (price - y_min) / (y_max - y_min);
                        let screen_y = chart_height * (1.0 - normalized_y);

                        if screen_y >= 0.0 && screen_y <= chart_height {
                            let text = if price >= 1000.0 {
                                format!("{:.0}", price)
                            } else if price >= 1.0 {
                                format!("{:.2}", price)
                            } else {
                                format!("{:.4}", price)
                            };

                            painter.text(
                                egui::Pos2::new(5.0, screen_y - 6.0),
                                egui::Align2::LEFT_CENTER,
                                text,
                                egui::FontId::proportional(11.0),
                                egui::Color32::from_rgba_unmultiplied(180, 180, 190, 200),
                            );
                        }
                    }
                });

            // Top bar with symbol, OHLC data, timeframe selector, and sync toggle
            let top_bar_response = TopBar::show(
                ctx,
                &current_symbol,
                current_timeframe,
                candle.as_ref(),
                ws_connected,
                sync_enabled,
                &sync_state,
            );
            if let Some(tf) = top_bar_response.clicked_timeframe {
                *new_timeframe.borrow_mut() = Some(tf);
            }
            if top_bar_response.toggle_sync {
                *should_toggle_sync.borrow_mut() = true;
            }

            // Drawing toolbar
            *drawing_toolbar_response.borrow_mut() = show_drawing_toolbar(ctx, drawing_tool, drawing_snap_enabled);

            // TA control panel
            {
                let ta_response = show_ta_panel(ctx, &ta_settings.borrow(), Some(&ta_data), &ta_hovered, screen_width);
                if ta_response.settings_changed
                    && let Some(new_settings) = ta_response.new_settings
                {
                    *ta_settings.borrow_mut() = new_settings;
                }
            }

            // MACD Indicators panel
            if show_macd {
                *macd_response.borrow_mut() = show_macd_panel(ctx, indicators, screen_width);
            }

            // Loading indicator overlay
            show_loading_overlay(ctx, &loading_state);

            // Symbol picker overlay
            if should_show_symbol_picker {
                let response = crate::ui::show_symbol_picker(ctx, &mut symbol_picker_state.borrow_mut(), &current_symbol);
                *symbol_picker_response.borrow_mut() = Some((response.closed, response.selected_symbol));
            }

            // Replay mode overlay and cursor line
            if self.interaction.replay.enabled {
                let (x_min, _x_max) = self.graphics.renderer.camera.visible_x_range(aspect);

                let cursor_x = if let Some(replay_ts) = self.interaction.replay.timestamp {
                    let tf_candles = self.document.current_candles();
                    if !tf_candles.is_empty() {
                        let idx = tf_candles
                            .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap())
                            .unwrap_or_else(|i| i.saturating_sub(1))
                            .min(tf_candles.len() - 1);

                        let candle_start_ts = tf_candles[idx].timestamp;
                        let candle_duration = Timeframe::all()[self.document.current_timeframe].seconds() as f32;
                        let fraction = ((replay_ts - candle_start_ts) as f32 / candle_duration).clamp(0.0, 1.0);

                        (idx as f32 + fraction) * CANDLE_SPACING
                    } else {
                        x_min
                    }
                } else if let Some(pos) = self.input.last_mouse_pos {
                    let normalized_x = pos[0] / chart_width;
                    x_min + normalized_x * (self.graphics.renderer.camera.scale[0] * aspect * 2.0)
                } else {
                    x_min
                };

                let screen_x = ((cursor_x - x_min) / (self.graphics.renderer.camera.scale[0] * aspect * 2.0)) * chart_width;

                if screen_x >= 0.0 && screen_x <= chart_width {
                    egui::Area::new(egui::Id::new("replay_cursor"))
                        .fixed_pos(egui::Pos2::ZERO)
                        .order(egui::Order::Background)
                        .show(ctx, |ui| {
                            let painter = ui.painter();
                            let (color, width) = if self.interaction.replay.index.is_some() {
                                (egui::Color32::from_rgba_unmultiplied(255, 200, 0, 100), 1.0)
                            } else {
                                (egui::Color32::from_rgba_unmultiplied(255, 255, 255, 150), 1.5)
                            };
                            painter.line_segment(
                                [
                                    egui::Pos2::new(screen_x, 0.0),
                                    egui::Pos2::new(screen_x, total_height),
                                ],
                                egui::Stroke::new(width, color),
                            );
                        });
                }

                let base_candle_info = if let Some(ts) = self.interaction.replay.timestamp {
                    let base_candles = self.document.timeframes.first().map(|tf| &tf.candles[..]).unwrap_or(&[]);
                    if !base_candles.is_empty() {
                        let base_idx = base_candles
                            .binary_search_by(|c| c.timestamp.partial_cmp(&ts).unwrap())
                            .unwrap_or_else(|i| i.saturating_sub(1));
                        Some((base_idx + 1, base_candles.len()))
                    } else {
                        None
                    }
                } else {
                    None
                };

                egui::Window::new("Replay Mode")
                    .title_bar(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_TOP, [0.0, 10.0])
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("REPLAY").color(egui::Color32::YELLOW).strong());
                            if let Some(idx) = self.interaction.replay.index {
                                ui.label(format!("Candle {}/{}", idx + 1, candle_count));
                                if let Some((base_idx, base_total)) = base_candle_info {
                                    ui.label(format!("(1m: {}/{})", base_idx, base_total));
                                }
                            } else {
                                ui.label("Click to set position");
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("Step: {}", self.interaction.replay.step_timeframe.label()));
                            ui.label("| [ ] step | ,. size | R exit");
                        });
                    });
            }
        });

        // Update TA settings if changed
        let new_ta_settings = ta_settings.into_inner();
        let ta_changed = new_ta_settings.show_ta != self.document.ta_settings.show_ta
            || new_ta_settings.show_ranges != self.document.ta_settings.show_ranges
            || new_ta_settings.show_hold_levels != self.document.ta_settings.show_hold_levels
            || new_ta_settings.show_greedy_levels != self.document.ta_settings.show_greedy_levels
            || new_ta_settings.show_active_levels != self.document.ta_settings.show_active_levels
            || new_ta_settings.show_hit_levels != self.document.ta_settings.show_hit_levels
            || new_ta_settings.show_broken_levels != self.document.ta_settings.show_broken_levels
            || new_ta_settings.show_trends != self.document.ta_settings.show_trends
            || new_ta_settings.show_active_trends != self.document.ta_settings.show_active_trends
            || new_ta_settings.show_hit_trends != self.document.ta_settings.show_hit_trends
            || new_ta_settings.show_broken_trends != self.document.ta_settings.show_broken_trends;

        if ta_changed {
            let was_ta_enabled = self.document.ta_settings.show_ta;
            self.document.ta_settings = new_ta_settings;
            if self.document.ta_settings.show_ta {
                if !was_ta_enabled {
                    self.ensure_ta_computed();
                }
                self.update_ta_buffers();
            }
        }

        // Apply MACD changes
        let macd_resp = macd_response.into_inner();
        let macd_changed = {
            let mut changed = false;

            if let Some(config) = macd_resp.add_indicator {
                self.add_macd(config);
                changed = true;
            }

            if let Some(idx) = macd_resp.remove_indicator {
                self.remove_macd(idx);
                changed = true;
            }

            for (i, new_config) in macd_resp.updated_configs {
                // Get candles before mutable borrow
                let candles = self.document.current_candles().to_vec();
                let tf = self.document.current_timeframe;

                if let Some(instance) = self.document.indicators.get_mut(i) {
                    if let Some(config) = instance.macd_config_mut() {
                        *config = new_config.clone();
                    }
                    if !candles.is_empty() {
                        let macd = Macd::new(new_config);
                        let output = macd.calculate_macd(&candles);
                        instance.macd_outputs[tf] = Some(output);
                    }
                }
                changed = true;
            }

            changed || macd_resp.config_changed
        };

        if macd_changed && !self.document.indicators.is_empty() {
            self.update_macd_gpu_buffers();
        }

        // Handle symbol picker response
        if let Some((closed, selected)) = symbol_picker_response.borrow().clone() {
            if closed {
                self.ui.panels.show_symbol_picker = false;
            }
            if let Some(symbol) = selected {
                self.ui.symbol_picker_state.add_recent(&self.interaction.current_symbol);
                self.switch_symbol(&symbol);
                self.ui.panels.show_symbol_picker = false;
            }
        }

        self.ui.symbol_picker_state = symbol_picker_state.into_inner();

        // Apply timeframe change from top bar
        if let Some(tf) = new_timeframe.into_inner() {
            self.switch_timeframe(tf);
        }

        // Apply sync toggle from top bar
        if should_toggle_sync.into_inner() {
            self.toggle_sync();
        }

        // Handle drawing toolbar response
        let drawing_resp = drawing_toolbar_response.into_inner();
        if let Some(tool) = drawing_resp.selected_tool {
            self.drawing.set_tool(tool);
        }
        if drawing_resp.toggle_snap {
            self.drawing.toggle_snap();
        }

        self.ui.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        let paint_jobs = self.ui.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.graphics.config.width, self.graphics.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.ui.egui_renderer.update_texture(&self.graphics.device, &self.graphics.queue, *id, image_delta);
        }

        let mut encoder = self.graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.ui.egui_renderer.update_buffers(&self.graphics.device, &self.graphics.queue, &mut encoder, &paint_jobs, &screen_descriptor);

        // First render pass: Clear and render charts
        {
            let tf = match self.document.timeframes.get(self.document.current_timeframe) {
                Some(tf) => tf,
                None => {
                    self.graphics.queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                    return Ok(());
                }
            };

            let use_replay_data = self.interaction.replay.enabled && self.interaction.replay.has_custom_timeframe_data();
            let replay_tf = self.interaction.replay.timeframe_data.as_ref();

            // Update render params for replay data
            if use_replay_data
                && let Some(ref rtf) = self.interaction.replay.timeframe_data
            {
                let (x_min, x_max) = self.graphics.renderer.camera.visible_x_range(aspect);
                let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);

                let visible_width = x_max - x_min;
                let x_pixel_size = visible_width / chart_width;
                let min_candle_width = 3.0 * x_pixel_size;
                let candle_width = 0.8_f32.max(min_candle_width).min(CANDLE_SPACING * 0.95);
                let wick_width = (candle_width * 0.1).clamp(1.0 * x_pixel_size, 4.0 * x_pixel_size);
                let visible_height = y_max - y_min;
                let y_pixel_size = visible_height / chart_height;
                let min_body_height = 2.0 * y_pixel_size;

                let render_params = RenderParams {
                    first_visible: 0,
                    candle_width,
                    candle_spacing: CANDLE_SPACING,
                    wick_width,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    price_min: rtf.price_normalization.price_min,
                    price_range: rtf.price_normalization.price_range,
                    min_body_height,
                    _padding: 0.0,
                };
                self.graphics.queue.write_buffer(&self.graphics.renderer.render_params_buffer, 0, bytemuck::cast_slice(&[render_params]));

                let volume_params = VolumeRenderParams {
                    first_visible: 0,
                    bar_width: candle_width,
                    bar_spacing: CANDLE_SPACING,
                    max_volume: rtf.max_volume.max(1.0),
                };
                self.graphics.queue.write_buffer(&self.graphics.renderer.volume_params_buffer, 0, bytemuck::cast_slice(&[volume_params]));
            }

            let effective_visible_count = if use_replay_data {
                replay_tf.map(|r| r.candles.len() as u32).unwrap_or(0)
            } else if self.interaction.replay.enabled {
                if let Some(replay_idx) = self.interaction.replay.index {
                    let visible_start = self.graphics.renderer.visible_start as usize;
                    let visible_count = self.graphics.renderer.visible_count as usize;
                    if replay_idx < visible_start {
                        0
                    } else if replay_idx >= visible_start + visible_count {
                        visible_count as u32
                    } else {
                        (replay_idx - visible_start + 1) as u32
                    }
                } else {
                    self.graphics.renderer.visible_count
                }
            } else {
                self.graphics.renderer.visible_count
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chart Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.04, b: 0.06, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_viewport(0.0, 0.0, chart_width, chart_height, 0.0, 1.0);

            // Render price guidelines first
            if self.graphics.renderer.guideline_count > 0 {
                render_pass.set_pipeline(&self.graphics.renderer.guideline_pipeline.pipeline);
                render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.graphics.renderer.guideline_bind_group, &[]);
                render_pass.draw(0..6, 0..self.graphics.renderer.guideline_count);
            }

            // Render candle chart
            render_pass.set_pipeline(&self.graphics.renderer.candle_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);

            let candle_bind_group = if use_replay_data {
                &replay_tf.unwrap().candle_bind_group
            } else if self.graphics.renderer.current_lod_factor == 1 {
                &tf.candle_bind_group
            } else if let Some(lod) = tf.lod_levels.iter().find(|l| l.factor == self.graphics.renderer.current_lod_factor) {
                &lod.candle_bind_group
            } else {
                &tf.candle_bind_group
            };
            render_pass.set_bind_group(1, candle_bind_group, &[]);
            render_pass.set_index_buffer(self.graphics.renderer.candle_pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..INDICES_PER_CANDLE, 0, 0..effective_visible_count);

            // Render current price line
            self.graphics.renderer.render_current_price(&mut render_pass);

            // Render TA
            if self.document.ta_settings.show_ta {
                // Get TA data references
                let replay_ta = self.interaction.replay.ta_data.as_ref();
                let doc_ta = self.document.ta_data.get(self.document.current_timeframe);
                let default_ta = TimeframeTaData::default();

                let ta: &TimeframeTaData = if self.interaction.replay.is_locked() {
                    replay_ta.or(doc_ta).unwrap_or(&default_ta)
                } else {
                    doc_ta.unwrap_or(&default_ta)
                };

                // Render ranges
                if self.document.ta_settings.show_ranges && !ta.ranges.is_empty() {
                    let range_count = ta.ranges.len().min(MAX_TA_RANGES) as u32;
                    render_pass.set_pipeline(&self.graphics.renderer.ta_pipeline.range_pipeline);
                    render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                    render_pass.draw(0..6, 0..range_count);
                }

                // Render levels
                let filtered_level_count = ta.levels.iter()
                    .filter(|l| {
                        let type_ok = match l.level_type {
                            LevelType::Hold => self.document.ta_settings.show_hold_levels,
                            LevelType::GreedyHold => self.document.ta_settings.show_greedy_levels,
                        };
                        let state_ok = match l.state {
                            LevelState::Inactive | LevelState::Active => self.document.ta_settings.show_active_levels,
                            LevelState::Broken => self.document.ta_settings.show_broken_levels,
                        };
                        type_ok && state_ok
                    })
                    .take(MAX_TA_LEVELS)
                    .count() as u32;

                if filtered_level_count > 0 {
                    render_pass.set_pipeline(&self.graphics.renderer.ta_pipeline.level_pipeline);
                    render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                    render_pass.draw(0..6, 0..filtered_level_count);
                }

                // Render trends
                if self.document.ta_settings.show_trends && !ta.trends.is_empty() {
                    let filtered_trend_count = ta.trends.iter()
                        .filter(|t| match t.state {
                            TrendState::Active => self.document.ta_settings.show_active_trends,
                            TrendState::Hit => self.document.ta_settings.show_hit_trends,
                            TrendState::Broken => self.document.ta_settings.show_broken_trends,
                        })
                        .take(MAX_TA_TRENDS)
                        .count() as u32;

                    if filtered_trend_count > 0 {
                        render_pass.set_pipeline(&self.graphics.renderer.ta_pipeline.trend_pipeline);
                        render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                        render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                        render_pass.draw(0..6, 0..filtered_trend_count);
                    }
                }
            }

            // Render user drawings
            if !self.drawing.drawings.is_empty() || self.drawing.tool != DrawingTool::None {
                let (x_min, x_max) = self.graphics.renderer.camera.visible_x_range(aspect);
                let (y_min, y_max) = self.graphics.renderer.camera.visible_y_range(aspect);

                let render_data = prepare_drawing_render_data(&self.drawing, CANDLE_SPACING);

                if !render_data.hrays.is_empty() {
                    self.graphics.queue.write_buffer(&self.drawing_hray_buffer, 0, bytemuck::cast_slice(&render_data.hrays));
                }
                if !render_data.rays.is_empty() {
                    self.graphics.queue.write_buffer(&self.drawing_ray_buffer, 0, bytemuck::cast_slice(&render_data.rays));
                }
                if !render_data.rects.is_empty() {
                    self.graphics.queue.write_buffer(&self.drawing_rect_buffer, 0, bytemuck::cast_slice(&render_data.rects));
                }
                if !render_data.anchors.is_empty() {
                    self.graphics.queue.write_buffer(&self.drawing_anchor_buffer, 0, bytemuck::cast_slice(&render_data.anchors));
                }

                let price_range = y_max - y_min;
                let visible_width = x_max - x_min;
                let y_world_per_pixel = price_range / chart_height;
                let x_world_per_pixel = visible_width / chart_width;

                let line_thickness = (y_world_per_pixel * 1.5).max(0.001);
                let x_line_thickness = (x_world_per_pixel * 1.5).max(0.001);
                // Anchor size should be about 8 pixels in each dimension
                let anchor_size = (y_world_per_pixel * 8.0).max(0.01);
                let anchor_size_x = (x_world_per_pixel * 8.0).max(0.01);

                let params = DrawingRenderParams {
                    x_min,
                    x_max,
                    line_thickness,
                    x_line_thickness,
                    anchor_size,
                    anchor_size_x,
                    hray_count: render_data.hrays.len() as u32,
                    ray_count: render_data.rays.len() as u32,
                    rect_count: render_data.rects.len() as u32,
                    anchor_count: render_data.anchors.len() as u32,
                    _padding1: 0,
                    _padding2: 0,
                };
                self.graphics.queue.write_buffer(&self.drawing_params_buffer, 0, bytemuck::cast_slice(&[params]));

                self.drawing_pipeline.render_rect_fills(&mut render_pass, &self.graphics.renderer.camera_bind_group, &self.drawing_bind_group, render_data.rects.len() as u32);
                self.drawing_pipeline.render_rect_borders(&mut render_pass, &self.graphics.renderer.camera_bind_group, &self.drawing_bind_group, render_data.rects.len() as u32);
                self.drawing_pipeline.render_hrays(&mut render_pass, &self.graphics.renderer.camera_bind_group, &self.drawing_bind_group, render_data.hrays.len() as u32);
                self.drawing_pipeline.render_rays(&mut render_pass, &self.graphics.renderer.camera_bind_group, &self.drawing_bind_group, render_data.rays.len() as u32);
                self.drawing_pipeline.render_anchors(&mut render_pass, &self.graphics.renderer.camera_bind_group, &self.drawing_bind_group, render_data.anchors.len() as u32);
            }

            // Render MACD indicators
            for instance in self.document.indicators.iter() {
                if !instance.is_enabled() {
                    continue;
                }

                let buffers = match &instance.gpu_buffers {
                    Some(IndicatorGpuBuffers::Macd(b)) => b,
                    _ => continue,
                };

                if buffers.macd_point_count > 1 {
                    render_pass.set_pipeline(&self.graphics.renderer.indicator_pipeline.pipeline);
                    render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &buffers.macd_bind_group, &[]);
                    render_pass.draw(0..6, 0..(buffers.macd_point_count - 1));
                }

                if buffers.signal_point_count > 1 {
                    render_pass.set_pipeline(&self.graphics.renderer.indicator_pipeline.pipeline);
                    render_pass.set_bind_group(0, &self.graphics.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &buffers.signal_bind_group, &[]);
                    render_pass.draw(0..6, 0..(buffers.signal_point_count - 1));
                }
            }

            // Render volume bars
            render_pass.set_viewport(0.0, chart_height, chart_width, volume_height, 0.0, 1.0);
            render_pass.set_pipeline(&self.graphics.renderer.volume_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.graphics.renderer.volume_camera_bind_group, &[]);

            let volume_bind_group = if use_replay_data {
                &replay_tf.unwrap().volume_bind_group
            } else if self.graphics.renderer.current_lod_factor == 1 {
                &tf.volume_bind_group
            } else if let Some(lod) = tf.lod_levels.iter().find(|l| l.factor == self.graphics.renderer.current_lod_factor) {
                &lod.volume_bind_group
            } else {
                &tf.volume_bind_group
            };
            render_pass.set_bind_group(1, volume_bind_group, &[]);
            render_pass.draw(0..6, 0..effective_visible_count);
        }

        self.graphics.queue.submit(std::iter::once(encoder.finish()));

        // Second encoder for egui
        let mut egui_encoder = self.graphics.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Egui Encoder"),
        });

        {
            let render_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let mut render_pass = render_pass.forget_lifetime();
            self.ui.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.ui.egui_renderer.free_texture(id);
        }

        self.graphics.queue.submit(std::iter::once(egui_encoder.finish()));
        output.present();

        // Update FPS counter
        self.frame_count += 1;
        let elapsed = self.last_frame_time.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_count = 0;
            self.last_frame_time = Instant::now();

            let tf_labels = ["1m", "3m", "5m", "30m", "1h", "3h", "5h", "10h", "1d", "1w", "3w", "1M"];
            let tf_label = tf_labels[self.document.current_timeframe];
            let candle_count = self.document.current_candles().len();

            self.window.set_title(&format!(
                "Charter [{}] - {:.1} FPS | {} candles ({} visible)",
                tf_label, self.fps, candle_count, self.graphics.renderer.visible_count
            ));
        }

        Ok(())
    }
}
