//! Application state and orchestration.

use std::cell::RefCell;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use tokio::sync::mpsc as tokio_mpsc;
use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use crate::indicators::{DynMacd, IndicatorGpuBuffers, IndicatorRegistry, MacdGpuBuffers};
use crate::input::{InputAction, InputHandler};
use crate::replay::{ReplayManager, TimeframeTaData};
use crate::ui::{
    show_loading_overlay, show_macd_panel, show_ta_panel,
    MacdPanelResponse, SymbolPickerState, TaHoveredInfo,
};
use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_data::{LiveDataEvent, LiveDataManager, MexcSource};
use charter_indicators::{Indicator, Macd, MacdConfig};
use charter_render::{
    ChartRenderer, IndicatorParams, IndicatorPointGpu, LevelGpu, RangeGpu, RenderParams,
    TaRenderParams, TimeframeData, TrendGpu, VolumeRenderParams, CANDLE_SPACING, INDICES_PER_CANDLE,
    MAX_TA_LEVELS, MAX_TA_RANGES, MAX_TA_TRENDS, STATS_PANEL_WIDTH, VOLUME_HEIGHT_RATIO,
};
use charter_ta::{
    Analyzer, AnalyzerConfig, CandleDirection, Level, LevelState, LevelType, MlFeatures, Range,
    TimeframeFeatures, Trend, TrendState, MlInferenceHandle,
};
use charter_config::Config as AppConfig;
use charter_ui::TopBar;

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
    pub fn is_loading(&self) -> bool {
        !matches!(self, LoadingState::Idle)
    }

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

/// Messages sent from background threads.
pub enum BackgroundMessage {
    /// Data loaded from file or API.
    DataLoaded(Vec<Candle>),
    /// A timeframe has been aggregated.
    TimeframeAggregated { index: usize, candles: Vec<Candle> },
    /// TA computation complete for a timeframe.
    TaComputed {
        timeframe: usize,
        ranges: Vec<Range>,
        levels: Vec<Level>,
        trends: Vec<Trend>,
    },
    /// Progress update for batch ML TA computation.
    MlTaProgress { completed: usize, total: usize },
    /// Loading state update.
    LoadingStateChanged(LoadingState),
    /// Live candle update from WebSocket.
    LiveCandleUpdate { candle: Candle, is_closed: bool },
    /// WebSocket connection status changed.
    ConnectionStatus(bool),
    /// Error occurred.
    Error(String),
}

/// Settings for TA display filtering.
#[derive(Debug, Clone)]
pub struct TaDisplaySettings {
    pub show_ta: bool,
    pub show_ranges: bool,
    pub show_hold_levels: bool,
    pub show_greedy_levels: bool,
    pub show_active_levels: bool,
    pub show_hit_levels: bool,
    pub show_broken_levels: bool,
    pub show_trends: bool,
    pub show_active_trends: bool,
    pub show_hit_trends: bool,
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

/// MACD conversion result with points and start indices.
struct MacdConversionResult {
    macd_points: Vec<IndicatorPointGpu>,
    signal_points: Vec<IndicatorPointGpu>,
    histogram_points: Vec<IndicatorPointGpu>,
    macd_start_index: usize,
    signal_start_index: usize,
}

pub struct State {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub is_surface_configured: bool,
    pub window: Arc<Window>,

    // Rendering
    pub renderer: ChartRenderer,
    pub timeframes: Vec<TimeframeData>,
    pub current_timeframe: usize,

    // Technical Analysis
    pub ta_data: Vec<TimeframeTaData>,
    pub ta_settings: TaDisplaySettings,
    pub hovered_range: Option<usize>,
    pub hovered_level: Option<usize>,
    pub hovered_trend: Option<usize>,

    // Indicators (MACD, etc.)
    pub indicators: IndicatorRegistry,
    pub show_macd_panel: bool,

    // Background loading
    pub loading_state: LoadingState,
    pub bg_receiver: Receiver<BackgroundMessage>,
    pub bg_sender: Sender<BackgroundMessage>,
    /// Candles waiting to be converted to GPU buffers (index, candles).
    pub pending_timeframes: Vec<(usize, Vec<Candle>)>,

    // Egui integration
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    // Input handling
    pub input: InputHandler,

    // FPS tracking
    pub last_frame_time: Instant,
    pub frame_count: u32,
    pub fps: f32,

    // Replay mode
    pub replay: ReplayManager,

    // ML inference (optional - loaded if model exists)
    pub ml_inference: Option<MlInferenceHandle>,

    // MEXC integration
    pub current_symbol: String,
    pub live_event_rx: Option<tokio_mpsc::Receiver<LiveDataEvent>>,
    pub ws_connected: bool,
    pub show_symbol_picker: bool,
    pub symbol_picker_state: SymbolPickerState,
    tokio_runtime: Option<tokio::runtime::Runtime>,

    // Application config
    pub app_config: AppConfig,
}

impl State {
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

        // Create tokio runtime for async MEXC operations
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        // Load application config
        let app_config = AppConfig::load_default();

        // Default symbol from config
        let default_symbol = app_config.general.default_symbol.clone();

        // Start background data loading from MEXC
        let sender = bg_sender.clone();
        let symbol = default_symbol.clone();
        tokio_runtime.spawn(async move {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                LoadingState::FetchingMexcData { symbol: symbol.clone() }
            ));

            // Load data from MEXC API
            let source = MexcSource::new(&symbol); // 90 days of history
            match source.load().await {
                Ok(base_candles) => {
                    let _ = sender.send(BackgroundMessage::DataLoaded(base_candles));
                }
                Err(e) => {
                    let _ = sender.send(BackgroundMessage::Error(format!("Failed to load data: {}", e)));
                }
            }
        });

        // Create renderer with empty candle data (will be populated later)
        let empty_candles: Vec<Candle> = vec![];
        let renderer = ChartRenderer::new(
            &device,
            &queue,
            surface_format,
            size.width,
            size.height,
            &empty_candles,
        );

        // Create placeholder timeframes (will be populated from background thread)
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

            // Create empty placeholder timeframe
            let tf_data = renderer.create_timeframe_data(&device, Vec::new(), tf.label());
            timeframes.push(tf_data);
        }

        // Egui setup
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1, false);

        // Try to load ML model if it exists
        let ml_inference = match MlInferenceHandle::load("data/charter_model.onnx") {
            Ok(handle) => {
                println!("ML model loaded successfully from data/charter_model.onnx");
                Some(handle)
            }
            Err(e) => {
                println!("ML model not available (this is OK): {}", e);
                None
            }
        };

        // Convert display config to TaDisplaySettings
        let ta_settings = TaDisplaySettings {
            show_ta: app_config.ta.display.show_ta,
            show_ranges: app_config.ta.display.show_ranges,
            show_hold_levels: app_config.ta.display.show_hold_levels,
            show_greedy_levels: app_config.ta.display.show_greedy_levels,
            show_active_levels: app_config.ta.display.show_active_levels,
            show_hit_levels: app_config.ta.display.show_hit_levels,
            show_broken_levels: app_config.ta.display.show_broken_levels,
            show_trends: app_config.ta.display.show_trends,
            show_active_trends: app_config.ta.display.show_active_trends,
            show_hit_trends: app_config.ta.display.show_hit_trends,
            show_broken_trends: app_config.ta.display.show_broken_trends,
        };

        let state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            renderer,
            timeframes,
            current_timeframe: 0,
            ta_data,
            ta_settings,
            hovered_range: None,
            hovered_level: None,
            hovered_trend: None,
            indicators: IndicatorRegistry::new(),
            show_macd_panel: false,
            loading_state: LoadingState::FetchingMexcData { symbol: default_symbol.clone() },
            bg_receiver,
            bg_sender,
            pending_timeframes: Vec::new(),
            egui_ctx,
            egui_state,
            egui_renderer,
            input: InputHandler::new(),
            last_frame_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            replay: ReplayManager::new(),
            ml_inference,
            // MEXC integration
            current_symbol: default_symbol,
            live_event_rx: None,
            ws_connected: false,
            show_symbol_picker: false,
            symbol_picker_state: SymbolPickerState::default(),
            tokio_runtime: Some(tokio_runtime),
            app_config,
        };

        // Don't update visible range yet - data is loading
        Ok(state)
    }

    fn update_visible_range(&mut self) {
        let tf = &self.timeframes[self.current_timeframe];
        self.renderer.update_visible_range(&self.queue, tf);
        if self.ta_settings.show_ta {
            self.update_ta_buffers();
        }
        // Update MACD params when view changes (first_visible depends on visible_start)
        if !self.indicators.is_empty() {
            self.update_macd_params();
        }
    }

    /// Update only the MACD params buffers (when view changes).
    /// This is more efficient than update_macd_gpu_buffers which also updates point data.
    fn update_macd_params(&mut self) {
        let visible_start = self.renderer.visible_start as usize;

        for instance in self.indicators.iter() {
            let buffers = match &instance.gpu_buffers {
                Some(IndicatorGpuBuffers::Macd(b)) => b,
                _ => continue,
            };

            // Calculate first_visible as offset into the indicator's points array
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

            // Update MACD params
            let macd_params = IndicatorParams {
                first_visible: macd_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness: 2.0,
                count: buffers.macd_point_count,
            };
            self.queue.write_buffer(
                &buffers.params_buffer,
                0,
                bytemuck::cast_slice(&[macd_params]),
            );

            // Update signal params
            let signal_params = IndicatorParams {
                first_visible: signal_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness: 2.0,
                count: buffers.signal_point_count,
            };
            self.queue.write_buffer(
                &buffers.signal_params_buffer,
                0,
                bytemuck::cast_slice(&[signal_params]),
            );
        }
    }

    /// Process any pending messages from background threads.
    /// Returns true if data was updated and view should refresh.
    pub fn process_background_messages(&mut self) -> bool {
        let mut updated = false;

        // Process all available messages (non-blocking)
        while let Ok(msg) = self.bg_receiver.try_recv() {
            match msg {
                BackgroundMessage::LoadingStateChanged(state) => {
                    self.loading_state = state;
                }
                BackgroundMessage::DataLoaded(base_candles) => {
                    // Data loaded - start aggregating timeframes in background
                    self.loading_state = LoadingState::AggregatingTimeframes {
                        current: 0,
                        total: Timeframe::all().len(),
                    };

                    // Spawn thread to aggregate all timeframes
                    let sender = self.bg_sender.clone();
                    thread::spawn(move || {
                        let timeframe_types = Timeframe::all();
                        let total = timeframe_types.len();

                        for (i, tf) in timeframe_types.iter().enumerate() {
                            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                                LoadingState::AggregatingTimeframes { current: i + 1, total },
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

                        // Signal we need to create GPU buffers
                        let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                            LoadingState::CreatingBuffers { current: 0, total },
                        ));
                    });
                }
                BackgroundMessage::TimeframeAggregated { index, candles } => {
                    // Store candles for GPU buffer creation (must happen on main thread)
                    self.pending_timeframes.push((index, candles));
                }
                BackgroundMessage::TaComputed {
                    timeframe,
                    ranges,
                    levels,
                    trends,
                } => {
                    // TA computation complete
                    self.ta_data[timeframe] = TimeframeTaData {
                        ranges,
                        levels,
                        trends,
                        computed: true,
                        prediction: None,
                    };
                    self.loading_state = LoadingState::Idle;

                    // Update TA buffers if this is the current timeframe
                    if timeframe == self.current_timeframe && self.ta_settings.show_ta {
                        self.update_ta_buffers();
                    }
                    updated = true;
                }
                BackgroundMessage::MlTaProgress { completed, total } => {
                    // Update progress for ML TA batch computation
                    if completed >= total {
                        self.loading_state = LoadingState::Idle;
                        // Recompute replay TA now that all ML timeframes are ready
                        if self.replay.is_locked() {
                            self.recompute_replay_ta();
                        }
                    } else {
                        self.loading_state = LoadingState::ComputingMlTa { completed, total };
                    }
                    updated = true;
                }
                BackgroundMessage::Error(err) => {
                    eprintln!("Background error: {}", err);
                    self.loading_state = LoadingState::Idle;
                }
                BackgroundMessage::LiveCandleUpdate { candle, is_closed } => {
                    // Skip live updates while data is being loaded (prevents wrong symbol's data)
                    if !matches!(self.loading_state, LoadingState::Idle) {
                        continue;
                    }
                    // Skip if we have no candles yet (data not loaded)
                    if self.timeframes[0].candles.is_empty() {
                        continue;
                    }
                    // Update the current 1-minute candle with live data
                    self.update_live_candle(candle, is_closed);
                    updated = true;
                }
                BackgroundMessage::ConnectionStatus(connected) => {
                    self.ws_connected = connected;
                    updated = true;
                }
            }
        }

        // Process pending timeframes (GPU buffer creation on main thread)
        if !self.pending_timeframes.is_empty() {
            let total = Timeframe::all().len();
            let pending: Vec<_> = self.pending_timeframes.drain(..).collect();

            for (index, candles) in pending {
                self.loading_state = LoadingState::CreatingBuffers {
                    current: index + 1,
                    total,
                };

                let label = Timeframe::all()[index].label();
                let tf_data = self.renderer.create_timeframe_data(&self.device, candles, label);
                self.timeframes[index] = tf_data;
            }

            // All done - set idle and fit view to show data
            self.loading_state = LoadingState::Idle;
            self.fit_view(); // Properly position camera after data is loaded

            // Start live updates automatically if not already connected
            if !self.ws_connected {
                self.start_live_updates();
            }

            updated = true;
        }

        updated
    }

    /// Update the current candle with live data from WebSocket.
    fn update_live_candle(&mut self, candle: Candle, is_closed: bool) {
        if is_closed {
            // Previous candle is closed, append the new one
            self.timeframes[0].candles.push(candle);

            // Re-aggregate all higher timeframes
            self.reaggregate_timeframes();

            // Rebuild GPU buffer for 1m timeframe
            let candles = self.timeframes[0].candles.clone();
            let new_tf_data = self.renderer.create_timeframe_data(
                &self.device,
                candles,
                "1m",
            );
            self.timeframes[0] = new_tf_data;
        } else {
            // Update the last candle in place
            if let Some(last) = self.timeframes[0].candles.last_mut() {
                last.high = last.high.max(candle.high);
                last.low = last.low.min(candle.low);
                last.close = candle.close;
                last.volume = candle.volume;
                // Keep open and timestamp unchanged
            }

            // Rebuild GPU buffer so the update is visible
            let candles = self.timeframes[0].candles.clone();
            let new_tf_data = self.renderer.create_timeframe_data(
                &self.device,
                candles,
                "1m",
            );
            self.timeframes[0] = new_tf_data;
        }

        // Update render params with new price_normalization (critical for correct rendering)
        self.update_visible_range();

        self.window.request_redraw();
    }

    /// Re-aggregate all higher timeframes from the base 1-minute data.
    fn reaggregate_timeframes(&mut self) {
        let base_candles = self.timeframes[0].candles.clone();
        let timeframe_types = Timeframe::all();

        for (i, tf) in timeframe_types.iter().enumerate().skip(1) {
            let candles = aggregate_candles(&base_candles, *tf);
            let new_tf_data = self.renderer.create_timeframe_data(
                &self.device,
                candles,
                tf.label(),
            );
            self.timeframes[i] = new_tf_data;
        }
    }

    /// Switch to a different trading symbol.
    pub fn switch_symbol(&mut self, symbol: &str) {
        if symbol.to_uppercase() == self.current_symbol {
            return; // Already on this symbol
        }

        let symbol = symbol.to_uppercase();
        self.current_symbol = symbol.clone();

        // Update loading state
        self.loading_state = LoadingState::FetchingMexcData { symbol: symbol.clone() };

        // Recreate empty TimeframeData to clear GPU buffers (just clearing Vec leaves stale GPU data)
        let timeframe_types = Timeframe::all();
        for (i, tf) in timeframe_types.iter().enumerate() {
            let empty_tf = self.renderer.create_timeframe_data(&self.device, Vec::new(), tf.label());
            self.timeframes[i] = empty_tf;
        }

        // Clear TA data
        for ta in &mut self.ta_data {
            ta.ranges.clear();
            ta.levels.clear();
            ta.trends.clear();
            ta.computed = false;
            ta.prediction = None;
        }

        // Close existing WebSocket connection
        self.live_event_rx = None;
        self.ws_connected = false;

        // Fetch new data in background
        if let Some(runtime) = &self.tokio_runtime {
            let sender = self.bg_sender.clone();
            let sym = symbol.clone();
            runtime.spawn(async move {
                let source = MexcSource::new(&sym);
                match source.load().await {
                    Ok(base_candles) => {
                        let _ = sender.send(BackgroundMessage::DataLoaded(base_candles));
                    }
                    Err(e) => {
                        let _ = sender.send(BackgroundMessage::Error(
                            format!("Failed to load {}: {}", sym, e)
                        ));
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
            let symbol = self.current_symbol.clone();

            runtime.spawn(async move {
                let mut manager = LiveDataManager::new();
                match manager.subscribe(&symbol).await {
                    Ok(mut rx) => {
                        let _ = sender.send(BackgroundMessage::ConnectionStatus(true));

                        while let Some(event) = rx.recv().await {
                            match event {
                                LiveDataEvent::CandleUpdate { candle, is_closed } => {
                                    let _ = sender.send(BackgroundMessage::LiveCandleUpdate {
                                        candle,
                                        is_closed,
                                    });
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

    /// Get AnalyzerConfig for a specific timeframe index.
    fn analyzer_config_for_timeframe(&self, tf_idx: usize) -> AnalyzerConfig {
        let tf_label = Timeframe::all()[tf_idx].label();
        let ta_config = self.app_config.ta_analysis_for_timeframe(tf_label);
        AnalyzerConfig::default()
            .doji_threshold(ta_config.doji_threshold)
            .min_range_candles(ta_config.min_range_candles)
            .level_tolerance(ta_config.level_tolerance)
            .create_greedy_levels(ta_config.create_greedy_levels)
    }

    /// Compute TA in background thread for a timeframe.
    fn compute_ta_background(&self, timeframe: usize) {
        let candles = self.timeframes[timeframe].candles.clone();
        let sender = self.bg_sender.clone();
        let ta_config = self.analyzer_config_for_timeframe(timeframe);

        thread::spawn(move || {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                LoadingState::ComputingTa { timeframe },
            ));

            let mut analyzer = Analyzer::with_config(ta_config);

            for candle in &candles {
                analyzer.process_candle(*candle);
            }

            let ranges = analyzer.ranges().to_vec();
            let levels = analyzer.all_levels().to_vec();
            let trends = analyzer.all_trends().to_vec();

            let _ = sender.send(BackgroundMessage::TaComputed {
                timeframe,
                ranges,
                levels,
                trends,
            });
        });
    }

    /// Precompute TA for all timeframes needed for ML inference.
    /// The ML model uses 4 essential timeframes: 5m, 1h, 1d, 1w
    /// These correspond to indices [2, 4, 8, 9] in Timeframe::all().
    /// This runs in background to avoid blocking the UI.
    fn precompute_ml_ta(&self) {
        // Essential timeframe indices: 5m(2), 1h(4), 1d(8), 1w(9)
        const ML_TIMEFRAME_INDICES: [usize; 4] = [2, 4, 8, 9];
        const MIN_CANDLES: usize = 100;

        // Collect timeframes that need TA computation (with their configs)
        let mut timeframes_to_compute: Vec<(usize, Vec<Candle>, AnalyzerConfig)> = Vec::new();

        for &tf_idx in &ML_TIMEFRAME_INDICES {
            if tf_idx >= self.timeframes.len() {
                continue;
            }

            // Skip if already computed
            if self.ta_data[tf_idx].computed {
                continue;
            }

            // Skip if not enough candles
            let candles = &self.timeframes[tf_idx].candles;
            if candles.len() < MIN_CANDLES {
                continue;
            }

            let ta_config = self.analyzer_config_for_timeframe(tf_idx);
            timeframes_to_compute.push((tf_idx, candles.clone(), ta_config));
        }

        if timeframes_to_compute.is_empty() {
            return; // Nothing to compute
        }

        let total = timeframes_to_compute.len();
        let sender = self.bg_sender.clone();

        // Send initial progress
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

                // Send TA data for this timeframe
                let _ = sender.send(BackgroundMessage::TaComputed {
                    timeframe: tf_idx,
                    ranges,
                    levels,
                    trends,
                });

                // Send progress update
                let _ = sender.send(BackgroundMessage::MlTaProgress {
                    completed: completed + 1,
                    total,
                });
            }
        });
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.renderer.resize(width, height);
            self.renderer.update_camera(&self.queue);
            self.update_visible_range();
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        // Don't process key events if symbol picker is open (except Escape)
        if self.show_symbol_picker && code != KeyCode::Escape {
            return;
        }

        if let Some(action) = self.input.handle_key(code, is_pressed) {
            match action {
                InputAction::Exit => {
                    if self.show_symbol_picker {
                        self.show_symbol_picker = false;
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
                // These actions are not triggered by keyboard input
                InputAction::Pan { .. }
                | InputAction::Zoom { .. }
                | InputAction::StartDrag
                | InputAction::EndDrag
                | InputAction::CursorMoved { .. }
                | InputAction::SetReplayIndex => {}
            }
        }
    }

    fn toggle_macd_panel(&mut self) {
        self.show_macd_panel = !self.show_macd_panel;
        self.window.request_redraw();
    }

    fn toggle_symbol_picker(&mut self) {
        self.show_symbol_picker = !self.show_symbol_picker;
        self.window.request_redraw();
    }

    fn toggle_ta(&mut self) {
        self.ta_settings.show_ta = !self.ta_settings.show_ta;
        if self.ta_settings.show_ta {
            self.ensure_ta_computed();
            self.update_ta_buffers();
            // If in replay mode with ML, precompute TA for all ML timeframes
            if self.replay.enabled && self.ml_inference.is_some() {
                self.precompute_ml_ta();
            }
        }
        self.window.request_redraw();
    }

    fn toggle_replay_mode(&mut self) {
        let was_enabled = self.replay.enabled;
        self.replay.toggle(self.current_timeframe);

        if !was_enabled && self.replay.enabled {
            // Entering replay mode - precompute TA for all ML timeframes if ML is available
            if self.ml_inference.is_some() && self.ta_settings.show_ta {
                self.precompute_ml_ta();
            }
        } else if was_enabled && !self.replay.enabled {
            // Refresh TA buffers with full data when exiting replay mode
            if self.ta_settings.show_ta {
                self.update_ta_buffers();
            }
        }
        self.window.request_redraw();
    }

    fn replay_step_forward(&mut self) {
        let base_candles = &self.timeframes[0].candles; // 1min candles
        if self.replay.step_forward(base_candles) {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
            self.window.request_redraw();
        }
    }

    fn replay_step_backward(&mut self) {
        let base_candles = &self.timeframes[0].candles;
        if self.replay.step_backward(base_candles) {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
            self.window.request_redraw();
        }
    }

    fn replay_increase_step_size(&mut self) {
        if self.replay.increase_step_size(self.current_timeframe) {
            self.window.request_redraw();
        }
    }

    fn replay_decrease_step_size(&mut self) {
        if self.replay.decrease_step_size() {
            self.window.request_redraw();
        }
    }

    fn set_replay_index(&mut self, index: usize) {
        let candles = &self.timeframes[self.current_timeframe].candles;
        self.replay.set_index(index, candles);
        self.recompute_replay_candles();
        self.recompute_replay_ta();
        self.window.request_redraw();
    }

    /// Recompute the replay candles from base 1min data.
    /// Re-aggregates candles up to replay_timestamp for accurate partial candle display.
    fn recompute_replay_candles(&mut self) {
        let base_candles = &self.timeframes[0].candles; // 1min candles
        let current_timeframe_idx = self.current_timeframe;
        let device = &self.device;
        let renderer = &self.renderer;

        self.replay.recompute_candles(
            base_candles,
            current_timeframe_idx,
            |candles, tf_label| renderer.create_timeframe_data(device, candles, tf_label),
        );
    }

    fn recompute_replay_ta(&mut self) {
        // Skip TA computation if TA is disabled
        if !self.ta_settings.show_ta {
            self.replay.ta_data = None;
            return;
        }

        // Get the replay timestamp
        let replay_ts = match self.replay.timestamp {
            Some(ts) => ts,
            None => {
                self.replay.ta_data = None;
                return;
            }
        };

        // Use replay_candles if available (re-aggregated data), otherwise use timeframe candles
        let current_tf_candles: Vec<Candle> = if let Some(ref replay_candles) = self.replay.candles {
            replay_candles.clone()
        } else if let Some(replay_idx) = self.replay.index {
            let tf_candles = &self.timeframes[self.current_timeframe].candles;
            if tf_candles.is_empty() || replay_idx == 0 {
                Vec::new()
            } else {
                tf_candles[..=replay_idx.min(tf_candles.len() - 1)].to_vec()
            }
        } else {
            self.replay.ta_data = None;
            return;
        };

        if current_tf_candles.is_empty() {
            self.replay.ta_data = Some(TimeframeTaData::with_data(
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
            self.update_ta_buffers();
            return;
        }

        // Get replay candle index for current timeframe
        let current_tf_idx = self.current_timeframe;
        let replay_candle_idx = current_tf_candles.len().saturating_sub(1);

        // Use precomputed TA if available, filtering to only include elements
        // that existed at the replay point. Otherwise compute on-the-fly.
        let mut ta_data = if self.ta_data[current_tf_idx].computed {
            // Filter precomputed TA data to replay point
            let ranges: Vec<_> = self.ta_data[current_tf_idx]
                .ranges
                .iter()
                .filter(|r| r.end_index <= replay_candle_idx)
                .cloned()
                .collect();

            let levels: Vec<_> = self.ta_data[current_tf_idx]
                .levels
                .iter()
                .filter(|l| {
                    if l.created_at_index > replay_candle_idx {
                        return false;
                    }
                    if l.state == LevelState::Broken {
                        if let Some(ref break_event) = l.break_event {
                            return break_event.candle_index > replay_candle_idx;
                        }
                    }
                    true
                })
                .cloned()
                .collect();

            let trends: Vec<_> = self.ta_data[current_tf_idx]
                .trends
                .iter()
                .filter(|t| {
                    if t.created_at_index > replay_candle_idx {
                        return false;
                    }
                    if t.state == TrendState::Broken {
                        if let Some(ref break_event) = t.break_event {
                            return break_event.candle_index > replay_candle_idx;
                        }
                    }
                    true
                })
                .cloned()
                .collect();

            TimeframeTaData::with_data(ranges, levels, trends)
        } else {
            // Fallback: compute TA on-the-fly (slow path)
            let ta_config = self.analyzer_config_for_timeframe(current_tf_idx);
            let mut analyzer = Analyzer::with_config(ta_config);
            for candle in &current_tf_candles {
                analyzer.process_candle(*candle);
            }
            TimeframeTaData::with_data(
                analyzer.ranges().to_vec(),
                analyzer.all_levels().to_vec(),
                analyzer.all_trends().to_vec(),
            )
        };

        // Run ML inference if model is available
        if let Some(ref ml_inference) = self.ml_inference {
            // Compute multi-timeframe features for ML
            let prediction = self.compute_ml_prediction(ml_inference, replay_ts, &current_tf_candles);
            if let Some(pred) = prediction {
                ta_data.set_prediction(pred);
            }
        }

        self.replay.ta_data = Some(ta_data);
        self.update_ta_buffers();
    }

    /// Helper function to calculate RSI for a specific candle index.
    fn calculate_rsi_for_candles(candles: &[Candle], current_idx: usize, period: usize) -> f32 {
        if current_idx < period + 1 || candles.is_empty() {
            return 0.5; // Neutral if not enough data
        }

        let start_idx = current_idx.saturating_sub(100.min(current_idx)); // Look back max 100 candles
        let lookback_candles = &candles[start_idx..=current_idx];

        if lookback_candles.len() < period + 1 {
            return 0.5;
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        // Calculate price changes
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

        // Calculate average gain and loss using Wilder's smoothing
        let mut avg_gain: f32 = gains.iter().take(period).sum::<f32>() / period as f32;
        let mut avg_loss: f32 = losses.iter().take(period).sum::<f32>() / period as f32;

        // Apply smoothing for remaining values
        for i in period..gains.len() {
            avg_gain = (avg_gain * (period - 1) as f32 + gains[i]) / period as f32;
            avg_loss = (avg_loss * (period - 1) as f32 + losses[i]) / period as f32;
        }

        // Calculate RSI
        if avg_loss == 0.0 {
            return 1.0; // Maximum RSI (100) normalized to 1.0
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        // Normalize to 0-1 range
        rsi / 100.0
    }

    /// Compute ML prediction using multi-timeframe features.
    ///
    /// The ML model uses 4 essential timeframes: 5m, 1h, 1d, 1w
    /// These correspond to indices [2, 4, 8, 9] in Timeframe::all().
    ///
    /// OPTIMIZATION: We only need TA from the last ~3 weekly ranges (21 days) to extract
    /// meaningful features. This avoids computing TA for years of history.
    fn compute_ml_prediction(
        &self,
        ml_inference: &MlInferenceHandle,
        replay_ts: f64,
        current_tf_candles: &[Candle],
    ) -> Option<charter_ta::MlPrediction> {
        // Minimum candles required per timeframe (same as training)
        const MIN_CANDLES: usize = 100;

        // Essential timeframe indices: 5m(2), 1h(4), 1d(8), 1w(9)
        const ML_TIMEFRAME_INDICES: [usize; 4] = [2, 4, 8, 9];

        // Only compute TA for the last 3 weekly ranges (~21 days)
        // This is sufficient for feature extraction and much faster than full history
        const LOOKBACK_SECONDS: f64 = 21.0 * 24.0 * 3600.0; // 21 days in seconds
        let lookback_start_ts = replay_ts - LOOKBACK_SECONDS;

        // Get current candle for global features
        let current_candle = current_tf_candles.last()?;
        let current_candle_idx = current_tf_candles.len().saturating_sub(1);
        let current_price = current_candle.close;

        // Compute features for the 6 essential timeframes (matching training)
        let mut tf_features: Vec<TimeframeFeatures> = Vec::new();

        for (feature_idx, &tf_idx) in ML_TIMEFRAME_INDICES.iter().enumerate() {
            // Get per-timeframe TA config
            let ta_config = self.analyzer_config_for_timeframe(tf_idx);
            if tf_idx >= self.timeframes.len() {
                continue;
            }
            let tf_data = &self.timeframes[tf_idx];

            // Find candles up to replay timestamp for this timeframe
            let candles = &tf_data.candles;
            if candles.len() < MIN_CANDLES {
                continue; // Skip timeframes with too few candles
            }

            // Binary search to find the last candle at or before replay_ts
            let end_idx = candles
                .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|i| i.saturating_sub(1));

            if end_idx == 0 {
                continue; // Not enough history
            }

            let tf_candle_idx = end_idx.min(candles.len().saturating_sub(1));

            // Use precomputed TA data if available for this timeframe
            let (levels, trends, feature_candle_idx) = if self.ta_data[tf_idx].computed {
                // Filter to only include TA elements that existed at this candle index
                // AND were not broken yet at this point in time
                let levels: Vec<_> = self.ta_data[tf_idx]
                    .levels
                    .iter()
                    .filter(|l| {
                        // Level must have been created before or at this candle
                        if l.created_at_index > tf_candle_idx {
                            return false;
                        }
                        // Check if level was broken at this point in time
                        // If broken, check when - if break happened after tf_candle_idx, it was still active
                        if l.state == LevelState::Broken {
                            if let Some(ref break_event) = l.break_event {
                                // Level was still active at replay time if break happened later
                                return break_event.candle_index > tf_candle_idx;
                            }
                        }
                        true // Active or hit level
                    })
                    .cloned()
                    .collect();
                let trends: Vec<_> = self.ta_data[tf_idx]
                    .trends
                    .iter()
                    .filter(|t| {
                        // Trend must have been created before or at this candle
                        if t.created_at_index > tf_candle_idx {
                            return false;
                        }
                        // Check if trend was broken at this point in time
                        if t.state == TrendState::Broken {
                            if let Some(ref break_event) = t.break_event {
                                return break_event.candle_index > tf_candle_idx;
                            }
                        }
                        true
                    })
                    .cloned()
                    .collect();
                // Use absolute index for precomputed data
                (levels, trends, tf_candle_idx)
            } else {
                // Compute TA only for the last ~3 weeks (not full history)
                // Find start index for lookback window
                let start_idx = candles
                    .binary_search_by(|c| c.timestamp.partial_cmp(&lookback_start_ts).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|i| i);

                let mut analyzer = Analyzer::with_config(ta_config.clone());
                for candle in &candles[start_idx..=tf_candle_idx] {
                    analyzer.process_candle(*candle);
                }
                // Use relative index for on-the-fly computation since analyzer
                // indices are relative to the window start
                let relative_idx = tf_candle_idx - start_idx;
                (analyzer.all_levels().to_vec(), analyzer.all_trends().to_vec(), relative_idx)
            };

            let features = TimeframeFeatures::extract(
                feature_idx, // Use sequential index 0-5 for feature extraction
                &levels,
                &trends,
                current_price,
                feature_candle_idx,
            );
            tf_features.push(features);
        }

        if tf_features.is_empty() {
            return None; // No valid timeframes
        }

        // Calculate global candle features
        let prev_close = if current_candle_idx > 0 {
            current_tf_candles.get(current_candle_idx - 1).map(|c| c.close).unwrap_or(current_candle.open)
        } else {
            current_candle.open
        };
        let price_change = if prev_close > 0.0 {
            (current_candle.close - prev_close) / prev_close
        } else {
            0.0
        };

        let body = (current_candle.close - current_candle.open).abs();
        let range = current_candle.high - current_candle.low;
        let body_ratio = if range > f32::EPSILON { body / range } else { 0.5 };

        // Normalize volume using rolling average (approximation)
        let volume_sum: f32 = current_tf_candles.iter().rev().take(100).map(|c| c.volume).sum();
        let volume_count = current_tf_candles.len().min(100) as f32;
        let avg_volume = volume_sum / volume_count.max(1.0);
        let volume_normalized = if avg_volume > f32::EPSILON {
            current_candle.volume / avg_volume
        } else {
            1.0
        };

        // Calculate RSI (14-period)
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

        // Check feature dimension matches model expectation
        // Model trained with 4 timeframes (5m, 1h, 1d, 1w): 302 features (4 × 74 + 6 with RSI)
        let feature_count = ml_features.feature_count();
        if feature_count != 302 {
            eprintln!(
                "ML feature count mismatch: got {} (from {} timeframes), expected 302 (4 timeframes × 74 + 6 with RSI)",
                feature_count,
                ml_features.timeframes.len()
            );
            return None;
        }

        // Debug: print some feature values to verify they change
        let feature_vec = ml_features.to_vec();
        let first_5: Vec<f32> = feature_vec.iter().take(5).copied().collect();
        let last_5: Vec<f32> = feature_vec.iter().rev().take(5).rev().copied().collect();
        eprintln!(
            "ML features: first 5 = {:?}, last 5 = {:?}, price = {:.2}",
            first_5, last_5, current_price
        );

        // Run inference with timing
        let inference_start = Instant::now();
        match ml_inference.predict(&ml_features) {
            Ok(prediction) => {
                let inference_time = inference_start.elapsed();
                eprintln!(
                    "ML inference: {:.2}ms, direction_up={:.1}%, level_break={:.1}%, conf={:.1}%",
                    inference_time.as_secs_f64() * 1000.0,
                    prediction.direction_up_prob * 100.0,
                    prediction.level_break_prob * 100.0,
                    prediction.confidence * 100.0
                );
                Some(prediction)
            }
            Err(e) => {
                // Log error but don't spam - only log occasionally
                static LAST_ERROR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let last = LAST_ERROR.load(std::sync::atomic::Ordering::Relaxed);
                if now > last + 5 {
                    LAST_ERROR.store(now, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("ML inference error: {}", e);
                }
                None
            }
        }
    }

    /// Compute TA for current timeframe if not already computed.
    /// Uses background thread to avoid blocking UI.
    fn ensure_ta_computed(&mut self) {
        let tf_idx = self.current_timeframe;

        // Skip if already computed or currently loading
        if self.ta_data[tf_idx].computed {
            return;
        }

        // Skip if data is still loading or no candles yet
        if self.loading_state.is_loading() || self.timeframes[tf_idx].candles.is_empty() {
            return;
        }

        // Start background TA computation
        self.compute_ta_background(tf_idx);
    }

    fn update_ta_buffers(&mut self) {
        let tf_idx = self.current_timeframe;
        let tf = &self.timeframes[tf_idx];

        // Use replay TA data if in replay mode with an index set
        let ta: &TimeframeTaData = if self.replay.is_locked() {
            self.replay.ta_data.as_ref().unwrap_or(&self.ta_data[tf_idx])
        } else {
            &self.ta_data[tf_idx]
        };

        // Convert ranges to GPU format
        let mut range_gpus: Vec<RangeGpu> = ta
            .ranges
            .iter()
            .filter(|_| self.ta_settings.show_ranges)
            .take(MAX_TA_RANGES)
            .map(|r| RangeGpu {
                x_start: r.start_index as f32 * CANDLE_SPACING,
                x_end: r.end_index as f32 * CANDLE_SPACING,
                y_pos: r.low,
                is_bullish: if r.direction == CandleDirection::Bullish { 1 } else { 0 },
            })
            .collect();

        let range_count = range_gpus.len() as u32;

        // Pad to MAX_TA_RANGES
        while range_gpus.len() < MAX_TA_RANGES {
            range_gpus.push(RangeGpu {
                x_start: 0.0,
                x_end: 0.0,
                y_pos: 0.0,
                is_bullish: 0,
            });
        }

        // Filter levels based on settings
        let filtered_levels: Vec<&Level> = ta
            .levels
            .iter()
            .filter(|l| {
                let type_ok = match l.level_type {
                    LevelType::Hold => self.ta_settings.show_hold_levels,
                    LevelType::GreedyHold => self.ta_settings.show_greedy_levels,
                };
                let state_ok = match l.state {
                    LevelState::Inactive => self.ta_settings.show_active_levels, // Inactive shown with active
                    LevelState::Active => self.ta_settings.show_active_levels,
                    LevelState::Broken => self.ta_settings.show_broken_levels,
                };
                type_ok && state_ok
            })
            .take(MAX_TA_LEVELS)
            .collect();

        let level_count = filtered_levels.len() as u32;

        // Convert levels to GPU format
        let mut level_gpus: Vec<LevelGpu> = filtered_levels
            .iter()
            .map(|l| {
                let (r, g, b, a) = match (l.direction, l.state) {
                    // Inactive levels: dimmer than active (waiting for price to close on other side)
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
                    r,
                    g,
                    b,
                    a,
                    level_type: if l.level_type == LevelType::Hold { 0 } else { 1 },
                    hit_count: l.hits.len() as u32,
                }
            })
            .collect();

        // Pad to MAX_TA_LEVELS
        while level_gpus.len() < MAX_TA_LEVELS {
            level_gpus.push(LevelGpu {
                y_value: 0.0,
                x_start: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
                level_type: 0,
                hit_count: 0,
            });
        }

        // Filter trends based on settings
        let filtered_trends: Vec<&Trend> = ta
            .trends
            .iter()
            .filter(|t| {
                if !self.ta_settings.show_trends {
                    return false;
                }
                match t.state {
                    TrendState::Active => self.ta_settings.show_active_trends,
                    TrendState::Hit => self.ta_settings.show_hit_trends,
                    TrendState::Broken => self.ta_settings.show_broken_trends,
                }
            })
            .take(MAX_TA_TRENDS)
            .collect();

        let trend_count = filtered_trends.len() as u32;

        // Convert trends to GPU format
        let mut trend_gpus: Vec<TrendGpu> = filtered_trends
            .iter()
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
                    r,
                    g,
                    b,
                    a,
                }
            })
            .collect();

        // Pad to MAX_TA_TRENDS
        while trend_gpus.len() < MAX_TA_TRENDS {
            trend_gpus.push(TrendGpu {
                x_start: 0.0,
                y_start: 0.0,
                x_end: 0.0,
                y_end: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            });
        }

        // Compute line thickness based on current view
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = (self.config.width as f32 - STATS_PANEL_WIDTH) / chart_height;
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);

        // Extend levels/trends to the right edge of visible screen (not just last candle)
        let (_, visible_x_max) = self.renderer.camera.visible_x_range(aspect);
        let candle_x_max = (tf.candles.len() as f32) * CANDLE_SPACING;
        let x_max = visible_x_max.max(candle_x_max);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        let level_thickness = (1.5 * world_units_per_pixel).max(price_range * 0.0008);
        let range_thickness = (2.0 * world_units_per_pixel).max(price_range * 0.001);

        let ta_params = TaRenderParams {
            first_visible: self.renderer.visible_start,
            candle_spacing: CANDLE_SPACING,
            range_thickness,
            level_thickness,
            x_max,
            range_count,
            level_count,
            trend_count,
        };

        // Write to GPU buffers
        self.queue.write_buffer(
            &tf.ta_range_buffer,
            0,
            bytemuck::cast_slice(&range_gpus),
        );
        self.queue.write_buffer(
            &tf.ta_level_buffer,
            0,
            bytemuck::cast_slice(&level_gpus),
        );
        self.queue.write_buffer(
            &tf.ta_trend_buffer,
            0,
            bytemuck::cast_slice(&trend_gpus),
        );
        self.queue.write_buffer(
            &tf.ta_params_buffer,
            0,
            bytemuck::cast_slice(&[ta_params]),
        );
    }

    // =========================================================================
    // Indicator Methods (using IndicatorRegistry)
    // =========================================================================

    /// Add a new MACD indicator instance with the given configuration.
    pub fn add_macd(&mut self, config: MacdConfig) {
        let num_timeframes = self.timeframes.len();
        let dyn_macd = DynMacd::new(config);
        let label = dyn_macd.label();

        // Add to registry
        let id = self.indicators.add(dyn_macd, label, num_timeframes);

        // Compute MACD for current timeframe immediately
        let candles = self.timeframes[self.current_timeframe].candles.clone();
        let tf = self.current_timeframe;
        if let Some(instance) = self.indicators.get_by_id_mut(id) {
            Self::compute_macd_for_instance_internal(instance, &candles, tf);
        }

        // Create GPU buffers for this instance
        let idx = self.indicators.index_of(id).unwrap();
        self.create_macd_gpu_buffers_for_instance(idx);
    }

    /// Remove a MACD indicator instance by index.
    pub fn remove_macd(&mut self, index: usize) {
        self.indicators.remove(index);
    }

    /// Compute MACD for a specific instance and timeframe.
    fn compute_macd_for_instance_internal(
        instance: &mut crate::indicators::IndicatorInstance,
        candles: &[Candle],
        timeframe: usize,
    ) {
        if candles.is_empty() {
            instance.outputs[timeframe] = None;
            instance.macd_outputs[timeframe] = None;
            return;
        }

        // Use the DynIndicator interface for generic calculation
        let output = instance.indicator.calculate(candles);
        instance.outputs[timeframe] = Some(output);

        // Also compute typed MACD output for GPU buffer creation
        if let Some(config) = instance.macd_config() {
            let macd = Macd::new(config.clone());
            let macd_output = macd.calculate_macd(candles);
            instance.macd_outputs[timeframe] = Some(macd_output);
        }
    }

    /// Recompute all indicators on the current timeframe.
    fn recompute_all_macd(&mut self) {
        let tf = self.current_timeframe;
        let candles = self.timeframes[tf].candles.clone();

        for instance in self.indicators.iter_mut() {
            if candles.is_empty() {
                instance.outputs[tf] = None;
                instance.macd_outputs[tf] = None;
                continue;
            }

            // Use the DynIndicator interface
            let output = instance.indicator.calculate(&candles);
            instance.outputs[tf] = Some(output);

            // Also compute typed MACD output
            if let Some(config) = instance.macd_config() {
                let macd = Macd::new(config.clone());
                let macd_output = macd.calculate_macd(&candles);
                instance.macd_outputs[tf] = Some(macd_output);
            }
        }

        // Update GPU buffers
        self.update_macd_gpu_buffers();
    }

    /// Create GPU buffers for a single indicator instance.
    fn create_macd_gpu_buffers_for_instance(&mut self, instance_idx: usize) {
        let instance = match self.indicators.get(instance_idx) {
            Some(i) => i,
            None => return,
        };
        let tf = self.current_timeframe;

        // Get the output for current timeframe
        let conversion = self.convert_macd_to_gpu_points_from_instance(instance, tf);

        // Create buffers
        let macd_line_buffer = self
            .renderer
            .indicator_pipeline
            .create_indicator_buffer(&self.device, &conversion.macd_points);
        let signal_line_buffer = self
            .renderer
            .indicator_pipeline
            .create_indicator_buffer(&self.device, &conversion.signal_points);
        let histogram_buffer = self
            .renderer
            .indicator_pipeline
            .create_indicator_buffer(&self.device, &conversion.histogram_points);

        // Create separate params buffers for MACD and signal lines (they have different start indices)
        let params_buffer = self
            .renderer
            .indicator_pipeline
            .create_indicator_params_buffer(&self.device);
        let signal_params_buffer = self
            .renderer
            .indicator_pipeline
            .create_indicator_params_buffer(&self.device);

        let macd_bind_group = self.renderer.indicator_pipeline.create_bind_group(
            &self.device,
            &macd_line_buffer,
            &params_buffer,
        );
        let signal_bind_group = self.renderer.indicator_pipeline.create_bind_group(
            &self.device,
            &signal_line_buffer,
            &signal_params_buffer,
        );
        let histogram_bind_group = self.renderer.indicator_pipeline.create_bind_group(
            &self.device,
            &histogram_buffer,
            &params_buffer,
        );

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

        // Store buffers in the instance
        if let Some(instance) = self.indicators.get_mut(instance_idx) {
            instance.gpu_buffers = Some(IndicatorGpuBuffers::Macd(buffers));
        }
    }

    /// Convert MACD output to GPU points for rendering (from IndicatorInstance).
    fn convert_macd_to_gpu_points_from_instance(
        &self,
        instance: &crate::indicators::IndicatorInstance,
        timeframe: usize,
    ) -> MacdConversionResult {
        let output = match &instance.macd_outputs[timeframe] {
            Some(o) => o,
            None => {
                return MacdConversionResult {
                    macd_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    signal_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    histogram_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    macd_start_index: 0,
                    signal_start_index: 0,
                };
            }
        };

        let config = match instance.macd_config() {
            Some(c) => c,
            None => {
                return MacdConversionResult {
                    macd_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    signal_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    histogram_points: vec![IndicatorPointGpu {
                        x: 0.0,
                        y: 0.0,
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        _padding: 0.0,
                    }],
                    macd_start_index: 0,
                    signal_start_index: 0,
                };
            }
        };

        let macd_start_index = output.macd_line.start_index();
        let signal_start_index = output.signal_line.start_index();

        // Convert MACD line
        let mut macd_points: Vec<IndicatorPointGpu> = output
            .macd_line
            .iter()
            .map(|(idx, &val)| IndicatorPointGpu {
                x: idx as f32 * CANDLE_SPACING,
                y: val,
                r: config.macd_color[0],
                g: config.macd_color[1],
                b: config.macd_color[2],
                _padding: 0.0,
            })
            .collect();

        // Convert signal line
        let mut signal_points: Vec<IndicatorPointGpu> = output
            .signal_line
            .iter()
            .map(|(idx, &val)| IndicatorPointGpu {
                x: idx as f32 * CANDLE_SPACING,
                y: val,
                r: config.signal_color[0],
                g: config.signal_color[1],
                b: config.signal_color[2],
                _padding: 0.0,
            })
            .collect();

        // Convert histogram (as points for now - could be rendered as bars)
        let mut histogram_points: Vec<IndicatorPointGpu> = output
            .histogram
            .iter()
            .map(|(idx, &val)| {
                let color = if val >= 0.0 {
                    config.histogram_pos_color
                } else {
                    config.histogram_neg_color
                };
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

        // Ensure at least one point for GPU buffers
        if macd_points.is_empty() {
            macd_points.push(IndicatorPointGpu {
                x: 0.0,
                y: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                _padding: 0.0,
            });
        }
        if signal_points.is_empty() {
            signal_points.push(IndicatorPointGpu {
                x: 0.0,
                y: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                _padding: 0.0,
            });
        }
        if histogram_points.is_empty() {
            histogram_points.push(IndicatorPointGpu {
                x: 0.0,
                y: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                _padding: 0.0,
            });
        }

        MacdConversionResult {
            macd_points,
            signal_points,
            histogram_points,
            macd_start_index,
            signal_start_index,
        }
    }

    /// Update GPU buffers for all MACD instances.
    fn update_macd_gpu_buffers(&mut self) {
        let tf = self.current_timeframe;
        let visible_start = self.renderer.visible_start as usize;

        for i in 0..self.indicators.len() {
            let instance = match self.indicators.get(i) {
                Some(inst) => inst,
                None => continue,
            };

            // Check if buffers exist, create them if not
            if instance.gpu_buffers.is_none() {
                self.create_macd_gpu_buffers_for_instance(i);
                continue;
            }

            let conversion = self.convert_macd_to_gpu_points_from_instance(instance, tf);

            // Get buffers (we need to re-borrow to avoid borrow checker issues)
            let instance = self.indicators.get(i).unwrap();
            let buffers = match &instance.gpu_buffers {
                Some(IndicatorGpuBuffers::Macd(b)) => b,
                _ => continue,
            };

            // Update buffers
            self.queue.write_buffer(
                &buffers.macd_line_buffer,
                0,
                bytemuck::cast_slice(&conversion.macd_points),
            );
            self.queue.write_buffer(
                &buffers.signal_line_buffer,
                0,
                bytemuck::cast_slice(&conversion.signal_points),
            );
            self.queue.write_buffer(
                &buffers.histogram_buffer,
                0,
                bytemuck::cast_slice(&conversion.histogram_points),
            );

            // Calculate first_visible as offset into the indicator's points array.
            let macd_first_visible = if visible_start > conversion.macd_start_index {
                (visible_start - conversion.macd_start_index) as u32
            } else {
                0
            };
            let signal_first_visible = if visible_start > conversion.signal_start_index {
                (visible_start - conversion.signal_start_index) as u32
            } else {
                0
            };

            // Update MACD params
            let macd_params = IndicatorParams {
                first_visible: macd_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness: 2.0,
                count: conversion.macd_points.len() as u32,
            };
            self.queue.write_buffer(
                &buffers.params_buffer,
                0,
                bytemuck::cast_slice(&[macd_params]),
            );

            // Update signal params (different start index)
            let signal_params = IndicatorParams {
                first_visible: signal_first_visible,
                point_spacing: CANDLE_SPACING,
                line_thickness: 2.0,
                count: conversion.signal_points.len() as u32,
            };
            self.queue.write_buffer(
                &buffers.signal_params_buffer,
                0,
                bytemuck::cast_slice(&[signal_params]),
            );

            // Update start indices and point counts in the instance
            if let Some(instance) = self.indicators.get_mut(i)
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

    fn switch_timeframe(&mut self, index: usize) {
        if index >= self.timeframes.len() || index == self.current_timeframe {
            return;
        }

        let old_candles = &self.timeframes[self.current_timeframe].candles;
        let new_candles = &self.timeframes[index].candles;

        if old_candles.is_empty() || new_candles.is_empty() {
            self.current_timeframe = index;
            self.fit_view();
            return;
        }

        let old_candle_idx =
            (self.renderer.camera.position[0] / CANDLE_SPACING).round() as usize;
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
        let new_scale_x = self.renderer.camera.scale[0] / ratio;

        self.renderer.camera.position[0] = new_x;
        self.renderer.camera.scale[0] = new_scale_x.max(5.0);

        self.current_timeframe = index;
        self.renderer.update_camera(&self.queue);

        // If in replay mode with a locked position, recompute replay data for new timeframe
        if self.replay.enabled && self.replay.is_locked() {
            self.recompute_replay_candles();
            self.recompute_replay_ta();
        }

        // Compute TA first (if enabled) before update_visible_range calls update_ta_buffers
        if self.ta_settings.show_ta {
            self.ensure_ta_computed();
        }
        // Recompute indicators for new timeframe
        if !self.indicators.is_empty() {
            self.recompute_all_macd();
        }
        self.update_visible_range();
        self.window.request_redraw();
    }

    fn fit_view(&mut self) {
        let candles = &self.timeframes[self.current_timeframe].candles;
        self.renderer.fit_view(&self.queue, candles);
        self.update_visible_range();
        self.window.request_redraw();
    }

    pub fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        let replay_index_set = self.replay.index.is_some();
        if let Some(action) =
            self.input
                .handle_mouse_input(state, button, self.replay.enabled, replay_index_set)
        {
            match action {
                InputAction::SetReplayIndex => {
                    let candle_idx = self.get_cursor_candle_index();
                    self.set_replay_index(candle_idx);
                }
                InputAction::StartDrag | InputAction::EndDrag => {
                    // Mouse state is already updated by InputHandler
                }
                // Other actions don't apply to mouse input
                _ => {}
            }
        }
    }

    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let pos = (position.x as f32, position.y as f32);

        if let Some(action) = self.input.handle_cursor_moved(pos) {
            match action {
                InputAction::Pan { dx, dy } => {
                    let aspect = self.config.width as f32 / self.config.height as f32;
                    let world_dx = -dx * (self.renderer.camera.scale[0] * aspect * 2.0)
                        / self.config.width as f32;
                    let world_dy =
                        dy * (self.renderer.camera.scale[1] * 2.0) / self.config.height as f32;

                    self.renderer.camera.position[0] += world_dx;
                    self.renderer.camera.position[1] += world_dy;

                    self.renderer.update_camera(&self.queue);
                    self.update_visible_range();
                    self.window.request_redraw();
                }
                InputAction::CursorMoved { .. } => {
                    // Update hover state when TA is visible
                    if self.ta_settings.show_ta {
                        self.update_hover_state();
                    }
                }
                _ => {}
            }
        }
    }

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
        let candles = &self.timeframes[self.current_timeframe].candles;

        // Don't zoom if there's no data
        if candles.is_empty() {
            return;
        }

        // Use chart area dimensions (excluding stats panel and volume section)
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = chart_width / chart_height;

        // Calculate cursor position in NDC relative to the chart area
        let cursor_ndc = if cursor_x < chart_width && cursor_y < chart_height {
            [
                (cursor_x / chart_width) * 2.0 - 1.0,
                1.0 - (cursor_y / chart_height) * 2.0,
            ]
        } else {
            // Cursor outside chart area, zoom from center
            [0.0, 0.0]
        };

        let world_x = self.renderer.camera.position[0]
            + cursor_ndc[0] * self.renderer.camera.scale[0] * aspect;
        let world_y =
            self.renderer.camera.position[1] + cursor_ndc[1] * self.renderer.camera.scale[1];

        let data_width = (candles.len() as f32) * CANDLE_SPACING;
        let max_x_zoom = (data_width / 2.0 / aspect).max(10.0) * 1.2; // Ensure minimum zoom

        let (min_price, max_price) = candles
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), c| {
                (min.min(c.low), max.max(c.high))
            });
        let price_range = (max_price - min_price).max(1.0); // Ensure minimum range
        let max_y_zoom = (price_range / 2.0).max(10.0) * 1.5; // Ensure minimum zoom

        if scroll_x.abs() > 0.001 {
            let zoom_factor = 1.0 - scroll_x * 0.1;
            let old_scale = self.renderer.camera.scale[0];
            self.renderer.camera.scale[0] = (old_scale * zoom_factor).clamp(5.0, max_x_zoom);
            let new_world_x = self.renderer.camera.position[0]
                + cursor_ndc[0] * self.renderer.camera.scale[0] * aspect;
            self.renderer.camera.position[0] += world_x - new_world_x;
        }

        if scroll_y.abs() > 0.001 {
            let zoom_factor = 1.0 + scroll_y * 0.1;
            let old_scale = self.renderer.camera.scale[1];
            self.renderer.camera.scale[1] = (old_scale * zoom_factor).clamp(1.0, max_y_zoom);
            let new_world_y = self.renderer.camera.position[1]
                + cursor_ndc[1] * self.renderer.camera.scale[1];
            self.renderer.camera.position[1] += world_y - new_world_y;
        }

        self.renderer.update_camera(&self.queue);
        self.update_visible_range();
        self.window.request_redraw();
    }

    pub fn update(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // Process any pending background messages
        self.process_background_messages();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let total_width = self.config.width as f32;
        let total_height = self.config.height as f32;
        let chart_width = (total_width - STATS_PANEL_WIDTH).floor().max(1.0);
        let chart_height = (total_height * (1.0 - VOLUME_HEIGHT_RATIO)).floor().max(1.0);
        let volume_height = (total_height - chart_height).max(1.0);

        // Get data needed for egui (extracted to avoid borrow conflicts)
        let cursor_candle_idx = self.get_cursor_candle_index();
        let candle = self.timeframes[self.current_timeframe]
            .candles
            .get(cursor_candle_idx)
            .copied();
        let candle_count = self.timeframes[self.current_timeframe].candles.len();

        // Build hovered info for TA panel
        let ta_hovered = TaHoveredInfo {
            range: self.hovered_range.and_then(|idx| {
                self.ta_data[self.current_timeframe].ranges.get(idx).map(|r| {
                    (
                        r.direction,
                        r.candle_count,
                        r.high,
                        r.low,
                        r.start_index,
                        r.end_index,
                    )
                })
            }),
            level: self.hovered_level.and_then(|idx| {
                self.ta_data[self.current_timeframe].levels.get(idx).map(|l| {
                    (
                        l.price,
                        l.level_type,
                        l.direction,
                        l.state,
                        l.hits.len(),
                    )
                })
            }),
            trend: self.hovered_trend.and_then(|idx| {
                self.ta_data[self.current_timeframe].trends.get(idx).map(|t| {
                    (
                        t.direction,
                        t.state,
                        t.start.price,
                        t.end.price,
                        t.start.candle_index,
                        t.end.candle_index,
                        t.hits.len(),
                    )
                })
            }),
        };

        let show_macd = self.show_macd_panel;
        let should_show_symbol_picker = self.show_symbol_picker;
        let current_symbol = self.current_symbol.clone();
        let current_timeframe = self.current_timeframe;
        let ws_connected = self.ws_connected;
        let new_timeframe = RefCell::new(None::<usize>);

        // Copy data for egui closure (will be updated after)
        let ta_settings = RefCell::new(self.ta_settings.clone());
        let symbol_picker_state = RefCell::new(self.symbol_picker_state.clone());
        // Use replay TA data if in replay mode with locked position, otherwise use precomputed TA
        let ta_data = if self.replay.is_locked() {
            self.replay.ta_data.clone().unwrap_or_else(|| self.ta_data[self.current_timeframe].clone())
        } else {
            self.ta_data[self.current_timeframe].clone()
        };
        let indicators = &self.indicators;
        let loading_state = self.loading_state.clone();
        let screen_width = self.config.width as f32;
        let macd_response = RefCell::new(MacdPanelResponse::default());
        let symbol_picker_response = RefCell::new(None::<(bool, Option<String>)>);

        // Get guideline values and visible range for price labels
        let guideline_values = self.renderer.guideline_values.clone();
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);

        // Build egui UI
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Draw price labels on the left side (using Foreground so they appear on top)
            egui::Area::new(egui::Id::new("price_labels"))
                .fixed_pos(egui::Pos2::ZERO)
                .order(egui::Order::Foreground)
                .interactable(false)
                .show(ctx, |ui| {
                    let painter = ui.painter();
                    for &price in &guideline_values {
                        // Convert world Y to screen Y
                        let normalized_y = (price - y_min) / (y_max - y_min);
                        let screen_y = chart_height * (1.0 - normalized_y);

                        // Only draw if within visible chart area
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

            // Top bar with symbol, OHLC data, and timeframe selector
            if let Some(tf) = TopBar::show(
                ctx,
                &current_symbol,
                current_timeframe,
                candle.as_ref(),
                ws_connected,
            ) {
                *new_timeframe.borrow_mut() = Some(tf);
            }

            // TA control panel (extracted to ui::ta_panel)
            {
                let ta_response = show_ta_panel(
                    ctx,
                    &ta_settings.borrow(),
                    Some(&ta_data),
                    &ta_hovered,
                    screen_width,
                );
                if ta_response.settings_changed
                    && let Some(new_settings) = ta_response.new_settings
                {
                    *ta_settings.borrow_mut() = new_settings;
                }
            }

            // MACD Indicators panel (extracted to ui::macd_panel)
            if show_macd {
                *macd_response.borrow_mut() = show_macd_panel(ctx, indicators, screen_width);
            }

            // Loading indicator overlay (extracted to ui::loading_overlay)
            show_loading_overlay(ctx, &loading_state);

            // Symbol picker overlay
            if should_show_symbol_picker {
                let response = crate::ui::show_symbol_picker(
                    ctx,
                    &mut symbol_picker_state.borrow_mut(),
                    &current_symbol,
                );
                *symbol_picker_response.borrow_mut() = Some((response.closed, response.selected_symbol));
            }

            // Replay mode overlay and cursor line
            if self.replay.enabled {
                let aspect = chart_width / chart_height;
                let (x_min, _x_max) = self.renderer.camera.visible_x_range(aspect);

                // Determine cursor X position
                let cursor_x = if let Some(replay_ts) = self.replay.timestamp {
                    // Use timestamp for sub-candle precision
                    let tf_candles = &self.timeframes[self.current_timeframe].candles;
                    if !tf_candles.is_empty() {
                        // Find the candle this timestamp falls into
                        let idx = tf_candles
                            .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap())
                            .unwrap_or_else(|i| i.saturating_sub(1))
                            .min(tf_candles.len() - 1);

                        // Calculate fractional position within the candle
                        let candle_start_ts = tf_candles[idx].timestamp;
                        let candle_duration = Timeframe::all()[self.current_timeframe].seconds() as f32;
                        let fraction = ((replay_ts - candle_start_ts) as f32 / candle_duration).clamp(0.0, 1.0);

                        (idx as f32 + fraction) * CANDLE_SPACING
                    } else {
                        x_min
                    }
                } else if let Some(pos) = self.input.last_mouse_pos {
                    // Following cursor
                    let normalized_x = pos[0] / chart_width;
                    x_min + normalized_x * (self.renderer.camera.scale[0] * aspect * 2.0)
                } else {
                    x_min
                };

                // Convert world X to screen X
                let screen_x = ((cursor_x - x_min) / (self.renderer.camera.scale[0] * aspect * 2.0)) * chart_width;

                // Draw vertical cursor line (thinner and more subtle when locked)
                if screen_x >= 0.0 && screen_x <= chart_width {
                    egui::Area::new(egui::Id::new("replay_cursor"))
                        .fixed_pos(egui::Pos2::ZERO)
                        .order(egui::Order::Background) // Behind UI elements
                        .show(ctx, |ui| {
                            let painter = ui.painter();
                            let (color, width) = if self.replay.index.is_some() {
                                // Subtle dashed appearance when locked
                                (egui::Color32::from_rgba_unmultiplied(255, 200, 0, 100), 1.0)
                            } else {
                                // More visible when following cursor
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

                // Calculate base candle progress for display
                let base_candle_info = if let Some(ts) = self.replay.timestamp {
                    let base_candles = &self.timeframes[0].candles;
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

                // Replay mode indicator
                egui::Window::new("Replay Mode")
                    .title_bar(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_TOP, [0.0, 10.0])
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("REPLAY").color(egui::Color32::YELLOW).strong());
                            if let Some(idx) = self.replay.index {
                                ui.label(format!("Candle {}/{}", idx + 1, candle_count));
                                if let Some((base_idx, base_total)) = base_candle_info {
                                    ui.label(format!("(1m: {}/{})", base_idx, base_total));
                                }
                            } else {
                                ui.label("Click to set position");
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("Step: {}", self.replay.step_timeframe.label()));
                            ui.label("| [ ] step | ,. size | R exit");
                        });
                    });
            }
        });

        // Update TA settings if changed (from extracted ta_panel)
        let new_ta_settings = ta_settings.into_inner();
        let ta_changed = new_ta_settings.show_ta != self.ta_settings.show_ta
            || new_ta_settings.show_ranges != self.ta_settings.show_ranges
            || new_ta_settings.show_hold_levels != self.ta_settings.show_hold_levels
            || new_ta_settings.show_greedy_levels != self.ta_settings.show_greedy_levels
            || new_ta_settings.show_active_levels != self.ta_settings.show_active_levels
            || new_ta_settings.show_hit_levels != self.ta_settings.show_hit_levels
            || new_ta_settings.show_broken_levels != self.ta_settings.show_broken_levels
            || new_ta_settings.show_trends != self.ta_settings.show_trends
            || new_ta_settings.show_active_trends != self.ta_settings.show_active_trends
            || new_ta_settings.show_hit_trends != self.ta_settings.show_hit_trends
            || new_ta_settings.show_broken_trends != self.ta_settings.show_broken_trends;

        if ta_changed {
            let was_ta_enabled = self.ta_settings.show_ta;
            self.ta_settings = new_ta_settings;
            if self.ta_settings.show_ta {
                // If TA was just enabled via UI, ensure it's computed
                if !was_ta_enabled {
                    self.ensure_ta_computed();
                }
                self.update_ta_buffers();
            }
        }

        // Apply MACD changes after egui run (from extracted macd_panel)
        let macd_resp = macd_response.into_inner();
        let macd_changed = {
            let mut changed = false;

            // Handle add request
            if let Some(config) = macd_resp.add_indicator {
                self.add_macd(config);
                changed = true;
            }

            // Handle remove request
            if let Some(idx) = macd_resp.remove_indicator {
                self.remove_macd(idx);
                changed = true;
            }

            // Update configs that changed
            for (i, new_config) in macd_resp.updated_configs {
                if let Some(instance) = self.indicators.get_mut(i) {
                    if let Some(config) = instance.macd_config_mut() {
                        *config = new_config.clone();
                    }
                    // Recompute MACD for this instance
                    let candles = &self.timeframes[self.current_timeframe].candles;
                    if !candles.is_empty() {
                        let macd = Macd::new(new_config);
                        let output = macd.calculate_macd(candles);
                        instance.macd_outputs[self.current_timeframe] = Some(output);
                    }
                }
                changed = true;
            }

            changed || macd_resp.config_changed
        };

        if macd_changed && !self.indicators.is_empty() {
            self.update_macd_gpu_buffers();
        }

        // Handle symbol picker response
        if let Some((closed, selected)) = symbol_picker_response.borrow().clone() {
            if closed {
                self.show_symbol_picker = false;
            }
            if let Some(symbol) = selected {
                self.symbol_picker_state.add_recent(&self.current_symbol);
                self.switch_symbol(&symbol);
                self.show_symbol_picker = false;
            }
        }

        // Update symbol picker state if it changed
        self.symbol_picker_state = symbol_picker_state.into_inner();

        // Apply timeframe change from top bar
        if let Some(tf) = new_timeframe.into_inner() {
            self.switch_timeframe(tf);
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // First render pass: Clear and render charts
        {
            let tf = &self.timeframes[self.current_timeframe];

            // Use replay timeframe data if available, otherwise use normal timeframe data
            let use_replay_data = self.replay.enabled && self.replay.has_custom_timeframe_data();
            let replay_tf = self.replay.timeframe_data.as_ref();

            // Update render params for replay data (different price normalization and first_visible)
            if use_replay_data
                && let Some(ref rtf) = self.replay.timeframe_data
            {
                let aspect = chart_width / chart_height;
                let (x_min, x_max) = self.renderer.camera.visible_x_range(aspect);
                let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);

                // Use fixed base values for replay - no LOD adjustment needed
                let visible_width = x_max - x_min;
                let x_pixel_size = visible_width / chart_width;
                let min_candle_width = 3.0 * x_pixel_size; // At least 3 pixels
                let candle_width = 0.8_f32.max(min_candle_width).min(CANDLE_SPACING * 0.95);
                let wick_width = (candle_width * 0.1).clamp(1.0 * x_pixel_size, 4.0 * x_pixel_size);
                // Use Y-axis pixel size for body height (price units per pixel)
                let visible_height = y_max - y_min;
                let y_pixel_size = visible_height / chart_height;
                let min_body_height = 2.0 * y_pixel_size;

                let render_params = RenderParams {
                    first_visible: 0, // Replay candles start at index 0
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
                self.queue.write_buffer(
                    &self.renderer.render_params_buffer,
                    0,
                    bytemuck::cast_slice(&[render_params]),
                );

                let volume_params = VolumeRenderParams {
                    first_visible: 0,
                    bar_width: candle_width,
                    bar_spacing: CANDLE_SPACING,
                    max_volume: rtf.max_volume.max(1.0),
                };
                self.queue.write_buffer(
                    &self.renderer.volume_params_buffer,
                    0,
                    bytemuck::cast_slice(&[volume_params]),
                );
            }

            // Calculate effective visible count for replay mode
            let effective_visible_count = if use_replay_data {
                // When using replay data, show all the re-aggregated candles
                replay_tf.map(|r| r.candles.len() as u32).unwrap_or(0)
            } else if self.replay.enabled {
                if let Some(replay_idx) = self.replay.index {
                    // In replay mode with locked position (no re-aggregation needed - same timeframe)
                    let visible_start = self.renderer.visible_start as usize;
                    let visible_count = self.renderer.visible_count as usize;
                    if replay_idx < visible_start {
                        // Replay point is before visible range
                        0
                    } else if replay_idx >= visible_start + visible_count {
                        // Replay point is beyond visible range - show all visible
                        visible_count as u32
                    } else {
                        // Replay point is within visible range
                        (replay_idx - visible_start + 1) as u32
                    }
                } else {
                    // Cursor following mode - show all
                    self.renderer.visible_count
                }
            } else {
                self.renderer.visible_count
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chart Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.06,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Set viewport for chart area
            render_pass.set_viewport(0.0, 0.0, chart_width, chart_height, 0.0, 1.0);

            // Render price guidelines first (behind candles)
            if self.renderer.guideline_count > 0 {
                render_pass.set_pipeline(&self.renderer.guideline_pipeline.pipeline);
                render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.renderer.guideline_bind_group, &[]);
                render_pass.draw(0..6, 0..self.renderer.guideline_count);
            }

            // Render candle chart using appropriate LOD level
            render_pass.set_pipeline(&self.renderer.candle_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);

            // Select the appropriate bind group based on replay mode and LOD
            let candle_bind_group = if use_replay_data {
                // Use replay timeframe data bind group
                &replay_tf.unwrap().candle_bind_group
            } else if self.renderer.current_lod_factor == 1 {
                &tf.candle_bind_group
            } else if let Some(lod) = tf.lod_levels.iter().find(|l| l.factor == self.renderer.current_lod_factor) {
                &lod.candle_bind_group
            } else {
                &tf.candle_bind_group
            };
            render_pass.set_bind_group(1, candle_bind_group, &[]);
            render_pass.set_index_buffer(
                self.renderer.candle_pipeline.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..INDICES_PER_CANDLE, 0, 0..effective_visible_count);

            // Render TA (ranges and levels) if enabled
            if self.ta_settings.show_ta {
                // Use replay TA data if in replay mode with locked position
                let ta: &TimeframeTaData = if self.replay.is_locked() {
                    self.replay.ta_data.as_ref().unwrap_or(&self.ta_data[self.current_timeframe])
                } else {
                    &self.ta_data[self.current_timeframe]
                };

                // Render ranges (6 vertices per quad, one per range)
                if self.ta_settings.show_ranges && !ta.ranges.is_empty() {
                    let range_count = ta.ranges.len().min(MAX_TA_RANGES) as u32;
                    render_pass.set_pipeline(&self.renderer.ta_pipeline.range_pipeline);
                    render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                    render_pass.draw(0..6, 0..range_count);
                }

                // Render levels (6 vertices per quad, one per level)
                let filtered_level_count = ta.levels.iter()
                    .filter(|l| {
                        let type_ok = match l.level_type {
                            LevelType::Hold => self.ta_settings.show_hold_levels,
                            LevelType::GreedyHold => self.ta_settings.show_greedy_levels,
                        };
                        let state_ok = match l.state {
                            LevelState::Inactive => self.ta_settings.show_active_levels,
                            LevelState::Active => self.ta_settings.show_active_levels,
                            LevelState::Broken => self.ta_settings.show_broken_levels,
                        };
                        type_ok && state_ok
                    })
                    .take(MAX_TA_LEVELS)
                    .count() as u32;

                if filtered_level_count > 0 {
                    render_pass.set_pipeline(&self.renderer.ta_pipeline.level_pipeline);
                    render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                    render_pass.draw(0..6, 0..filtered_level_count);
                }

                // Render trends (6 vertices per quad, one per trend)
                if self.ta_settings.show_trends && !ta.trends.is_empty() {
                    let filtered_trend_count = ta.trends.iter()
                        .filter(|t| {
                            match t.state {
                                TrendState::Active => self.ta_settings.show_active_trends,
                                TrendState::Hit => self.ta_settings.show_hit_trends,
                                TrendState::Broken => self.ta_settings.show_broken_trends,
                            }
                        })
                        .take(MAX_TA_TRENDS)
                        .count() as u32;

                    if filtered_trend_count > 0 {
                        render_pass.set_pipeline(&self.renderer.ta_pipeline.trend_pipeline);
                        render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                        render_pass.set_bind_group(1, &tf.ta_bind_group, &[]);
                        render_pass.draw(0..6, 0..filtered_trend_count);
                    }
                }
            }

            // Render MACD indicators if any are enabled
            // Note: MACD values are on a different scale than prices, so this renders
            // them overlaid on the price chart. For proper visualization, a separate
            // oscillator pane should be implemented.
            for instance in self.indicators.iter() {
                if !instance.is_enabled() {
                    continue;
                }

                let buffers = match &instance.gpu_buffers {
                    Some(IndicatorGpuBuffers::Macd(b)) => b,
                    _ => continue,
                };

                // Render MACD line (each segment needs 6 vertices)
                if buffers.macd_point_count > 1 {
                    render_pass.set_pipeline(&self.renderer.indicator_pipeline.pipeline);
                    render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &buffers.macd_bind_group, &[]);
                    render_pass.draw(0..6, 0..(buffers.macd_point_count - 1));
                }

                // Render signal line
                if buffers.signal_point_count > 1 {
                    render_pass.set_pipeline(&self.renderer.indicator_pipeline.pipeline);
                    render_pass.set_bind_group(0, &self.renderer.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &buffers.signal_bind_group, &[]);
                    render_pass.draw(0..6, 0..(buffers.signal_point_count - 1));
                }
            }

            // Render volume bars using appropriate LOD level
            render_pass.set_viewport(0.0, chart_height, chart_width, volume_height, 0.0, 1.0);
            render_pass.set_pipeline(&self.renderer.volume_pipeline.pipeline);
            render_pass.set_bind_group(0, &self.renderer.volume_camera_bind_group, &[]);

            let volume_bind_group = if use_replay_data {
                // Use replay timeframe data volume bind group
                &replay_tf.unwrap().volume_bind_group
            } else if self.renderer.current_lod_factor == 1 {
                &tf.volume_bind_group
            } else if let Some(lod) = tf.lod_levels.iter().find(|l| l.factor == self.renderer.current_lod_factor) {
                &lod.volume_bind_group
            } else {
                &tf.volume_bind_group
            };
            render_pass.set_bind_group(1, volume_bind_group, &[]);
            render_pass.draw(0..6, 0..effective_visible_count);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Second encoder for egui
        let mut egui_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            self.egui_renderer
                .render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(egui_encoder.finish()));
        output.present();

        // Update FPS counter
        self.frame_count += 1;
        let elapsed = self.last_frame_time.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_count = 0;
            self.last_frame_time = Instant::now();

            let tf_labels = ["1m", "3m", "5m", "30m", "1h", "3h", "5h", "10h", "1d", "1w", "3w", "1M"];
            let tf_label = tf_labels[self.current_timeframe];
            let candle_count = self.timeframes[self.current_timeframe].candles.len();

            self.window.set_title(&format!(
                "Charter [{}] - {:.1} FPS | {} candles ({} visible)",
                tf_label, self.fps, candle_count, self.renderer.visible_count
            ));
        }

        Ok(())
    }

    fn get_cursor_candle_index(&self) -> usize {
        let tf = &self.timeframes[self.current_timeframe];
        if let Some(pos) = self.input.last_mouse_pos {
            let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
            if pos[0] < chart_width {
                let aspect = chart_width / (self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO));
                let (x_min, _) = self.renderer.camera.visible_x_range(aspect);
                let normalized_x = pos[0] / chart_width;
                let world_x = x_min + normalized_x * (self.renderer.camera.scale[0] * aspect * 2.0);
                let idx = (world_x / CANDLE_SPACING).round() as usize;
                return idx.min(tf.candles.len().saturating_sub(1));
            }
        }
        tf.candles.len().saturating_sub(1)
    }

    /// Get cursor world coordinates (x, y) in chart space.
    fn get_cursor_world_pos(&self) -> Option<(f32, f32)> {
        let pos = self.input.last_mouse_pos?;
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);

        // Only return position if cursor is in chart area
        if pos[0] >= chart_width || pos[1] >= chart_height {
            return None;
        }

        let aspect = chart_width / chart_height;
        let (x_min, _) = self.renderer.camera.visible_x_range(aspect);
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);

        let normalized_x = pos[0] / chart_width;
        let normalized_y = pos[1] / chart_height;

        let world_x = x_min + normalized_x * (self.renderer.camera.scale[0] * aspect * 2.0);
        let world_y = y_max - normalized_y * (y_max - y_min); // Flip Y

        Some((world_x, world_y))
    }

    /// Find hovered range at cursor position.
    fn find_hovered_range(&self) -> Option<usize> {
        if !self.ta_settings.show_ta || !self.ta_settings.show_ranges {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = &self.ta_data[self.current_timeframe];

        // Calculate tolerance based on view scale
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let tolerance = price_range * 0.01; // 1% of visible price range

        for (i, range) in ta.ranges.iter().enumerate() {
            let x_start = range.start_index as f32 * CANDLE_SPACING;
            let x_end = range.end_index as f32 * CANDLE_SPACING;

            // Check if cursor is within range's horizontal span
            if world_x >= x_start && world_x <= x_end {
                // Check if cursor is near the range line (at range.low)
                if (world_y - range.low).abs() < tolerance {
                    return Some(i);
                }
            }
        }

        None
    }

    /// Find hovered level at cursor position.
    fn find_hovered_level(&self) -> Option<usize> {
        if !self.ta_settings.show_ta {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = &self.ta_data[self.current_timeframe];

        // Calculate tolerance based on view scale
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let tolerance = price_range * 0.005; // 0.5% of visible price range

        for (i, level) in ta.levels.iter().enumerate() {
            // Check visibility based on settings
            let type_visible = match level.level_type {
                LevelType::Hold => self.ta_settings.show_hold_levels,
                LevelType::GreedyHold => self.ta_settings.show_greedy_levels,
            };
            let state_visible = match level.state {
                LevelState::Inactive => self.ta_settings.show_active_levels,
                LevelState::Active => self.ta_settings.show_active_levels,
                LevelState::Broken => self.ta_settings.show_broken_levels,
            };

            if !type_visible || !state_visible {
                continue;
            }

            let x_start = level.created_at_index as f32 * CANDLE_SPACING;

            // Check if cursor is to the right of level start and near the level price
            if world_x >= x_start && (world_y - level.price).abs() < tolerance {
                return Some(i);
            }
        }

        None
    }

    /// Find hovered trend at cursor position.
    fn find_hovered_trend(&self) -> Option<usize> {
        if !self.ta_settings.show_ta || !self.ta_settings.show_trends {
            return None;
        }

        let (world_x, world_y) = self.get_cursor_world_pos()?;
        let ta = &self.ta_data[self.current_timeframe];

        // Calculate tolerance based on view scale
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let y_scale = self.renderer.camera.scale[1] * 2.0 / chart_height;
        let tolerance = y_scale * 10.0; // 10 pixels tolerance

        for (i, trend) in ta.trends.iter().enumerate() {
            // Check visibility based on state
            let state_visible = match trend.state {
                charter_ta::types::trend::TrendState::Active => self.ta_settings.show_active_trends,
                charter_ta::types::trend::TrendState::Hit => self.ta_settings.show_hit_trends,
                charter_ta::types::trend::TrendState::Broken => self.ta_settings.show_broken_trends,
            };

            if !state_visible {
                continue;
            }

            let start_x = trend.start.candle_index as f32 * CANDLE_SPACING;

            // Check if cursor x is within trend's x range (extended to the right)
            if world_x < start_x {
                continue;
            }

            // Calculate trend price at cursor x (using linear interpolation)
            let candle_pos = world_x / CANDLE_SPACING;
            let dx = trend.end.candle_index as f32 - trend.start.candle_index as f32;
            let trend_price = if dx.abs() < f32::EPSILON {
                trend.start.price
            } else {
                let slope = (trend.end.price - trend.start.price) / dx;
                let x = candle_pos - trend.start.candle_index as f32;
                trend.start.price + slope * x
            };

            // Check if cursor is near the trend line
            if (world_y - trend_price).abs() < tolerance {
                return Some(i);
            }
        }

        None
    }

    /// Update hovered range/level/trend state.
    fn update_hover_state(&mut self) {
        self.hovered_range = self.find_hovered_range();
        self.hovered_level = if self.hovered_range.is_none() {
            self.find_hovered_level()
        } else {
            None
        };
        self.hovered_trend = if self.hovered_range.is_none() && self.hovered_level.is_none() {
            self.find_hovered_trend()
        } else {
            None
        };
    }
}
