//! Application state and orchestration.

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

use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_data::CsvLoader;
use charter_data::DataSource;
use charter_indicators::{Indicator, Macd, MacdConfig, MacdOutput};
use charter_render::{
    ChartRenderer, IndicatorParams, IndicatorPointGpu, LevelGpu, RangeGpu, RenderParams,
    TaRenderParams, TimeframeData, TrendGpu, VolumeRenderParams, CANDLE_SPACING, MAX_TA_LEVELS,
    MAX_TA_RANGES, MAX_TA_TRENDS, STATS_PANEL_WIDTH, VERTICES_PER_CANDLE, VOLUME_HEIGHT_RATIO,
};
use charter_ta::{
    Analyzer, AnalyzerConfig, CandleDirection, Level, LevelState, LevelType, Range, Trend,
    TrendState,
};
use charter_ui::StatsPanel;

/// Loading state for background operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingState {
    /// No loading in progress.
    Idle,
    /// Loading data from file.
    LoadingData,
    /// Aggregating timeframes.
    AggregatingTimeframes { current: usize, total: usize },
    /// Creating GPU buffers.
    CreatingBuffers { current: usize, total: usize },
    /// Computing technical analysis.
    ComputingTa { timeframe: usize },
}

impl LoadingState {
    pub fn is_loading(&self) -> bool {
        !matches!(self, LoadingState::Idle)
    }

    pub fn message(&self) -> String {
        match self {
            LoadingState::Idle => String::new(),
            LoadingState::LoadingData => "Loading data...".to_string(),
            LoadingState::AggregatingTimeframes { current, total } => {
                format!("Aggregating timeframes ({}/{})", current, total)
            }
            LoadingState::CreatingBuffers { current, total } => {
                format!("Creating GPU buffers ({}/{})", current, total)
            }
            LoadingState::ComputingTa { timeframe } => {
                format!("Computing TA for timeframe {}", timeframe)
            }
        }
    }
}

/// Messages sent from background threads.
pub enum BackgroundMessage {
    /// Data loaded from file.
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
    /// Loading state update.
    LoadingStateChanged(LoadingState),
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

/// A single MACD indicator instance with configuration and computed data.
#[derive(Clone)]
pub struct MacdInstance {
    /// Unique identifier for this instance.
    pub id: usize,
    /// Configuration for this MACD.
    pub config: MacdConfig,
    /// Computed output per timeframe.
    pub outputs: Vec<Option<MacdOutput>>,
}

impl MacdInstance {
    pub fn new(id: usize, config: MacdConfig, num_timeframes: usize) -> Self {
        Self {
            id,
            config,
            outputs: vec![None; num_timeframes],
        }
    }

    pub fn label(&self) -> String {
        format!(
            "MACD({},{},{})",
            self.config.fast_period, self.config.slow_period, self.config.signal_period
        )
    }
}

/// GPU buffers for rendering MACD indicators.
pub struct MacdGpuBuffers {
    pub macd_line_buffer: wgpu::Buffer,
    pub signal_line_buffer: wgpu::Buffer,
    pub histogram_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub signal_params_buffer: wgpu::Buffer,
    pub macd_bind_group: wgpu::BindGroup,
    pub signal_bind_group: wgpu::BindGroup,
    pub histogram_bind_group: wgpu::BindGroup,
    pub macd_point_count: u32,
    pub signal_point_count: u32,
    pub histogram_point_count: u32,
    /// Candle index where MACD line data starts.
    pub macd_start_index: usize,
    /// Candle index where signal line data starts.
    pub signal_start_index: usize,
}

/// MACD conversion result with points and start indices.
struct MacdConversionResult {
    macd_points: Vec<IndicatorPointGpu>,
    signal_points: Vec<IndicatorPointGpu>,
    histogram_points: Vec<IndicatorPointGpu>,
    macd_start_index: usize,
    signal_start_index: usize,
}

/// TA data computed for a single timeframe.
pub struct TimeframeTaData {
    pub ranges: Vec<Range>,
    pub levels: Vec<Level>,
    pub trends: Vec<Trend>,
    pub computed: bool,
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

    // MACD Indicators
    pub macd_instances: Vec<MacdInstance>,
    pub macd_next_id: usize,
    pub macd_gpu_buffers: Vec<MacdGpuBuffers>,
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
    pub stats_panel: StatsPanel,

    // Mouse state
    pub mouse_pressed: bool,
    pub last_mouse_pos: Option<[f32; 2]>,

    // FPS tracking
    pub last_frame_time: Instant,
    pub frame_count: u32,
    pub fps: f32,

    // Replay mode
    pub replay_mode: bool,
    pub replay_index: Option<usize>, // None = cursor following, Some = locked to index
    pub replay_ta_data: Option<TimeframeTaData>, // TA computed for replay range
    pub replay_timestamp: Option<f64>, // Current replay position in timestamp
    pub replay_step_timeframe: Timeframe, // Step size for replay (can be finer than view timeframe)
    pub replay_candles: Option<Vec<Candle>>, // Cached partial candles for replay
    pub replay_timeframe_data: Option<TimeframeData>, // GPU data for replay candles
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

        // Start background data loading
        let sender = bg_sender.clone();
        thread::spawn(move || {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(LoadingState::LoadingData));

            // Load data
            let loader = CsvLoader::new("data/btc-new.csv");
            match loader.load() {
                Ok(base_candles) => {
                    println!("Loaded {} base candles", base_candles.len());
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

        let stats_panel = StatsPanel::default();

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
            ta_settings: TaDisplaySettings::default(),
            hovered_range: None,
            hovered_level: None,
            hovered_trend: None,
            macd_instances: Vec::new(),
            macd_next_id: 0,
            macd_gpu_buffers: Vec::new(),
            show_macd_panel: false,
            loading_state: LoadingState::LoadingData,
            bg_receiver,
            bg_sender,
            pending_timeframes: Vec::new(),
            egui_ctx,
            egui_state,
            egui_renderer,
            stats_panel,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_frame_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
            replay_mode: false,
            replay_index: None,
            replay_ta_data: None,
            replay_timestamp: None,
            replay_step_timeframe: Timeframe::Min1,
            replay_candles: None,
            replay_timeframe_data: None,
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
        if !self.macd_instances.is_empty() {
            self.update_macd_params();
        }
    }

    /// Update only the MACD params buffers (when view changes).
    /// This is more efficient than update_macd_gpu_buffers which also updates point data.
    fn update_macd_params(&mut self) {
        let visible_start = self.renderer.visible_start as usize;

        for i in 0..self.macd_gpu_buffers.len() {
            let buffers = &self.macd_gpu_buffers[i];

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
                            println!("  {} timeframe: {} candles", tf.label(), candles.len());

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
                    };
                    self.loading_state = LoadingState::Idle;

                    // Update TA buffers if this is the current timeframe
                    if timeframe == self.current_timeframe && self.ta_settings.show_ta {
                        self.update_ta_buffers();
                    }
                    updated = true;
                }
                BackgroundMessage::Error(err) => {
                    eprintln!("Background error: {}", err);
                    self.loading_state = LoadingState::Idle;
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

            // All done - set idle and update view
            self.loading_state = LoadingState::Idle;
            self.update_visible_range();
            updated = true;
        }

        updated
    }

    /// Compute TA in background thread for a timeframe.
    fn compute_ta_background(&self, timeframe: usize) {
        let candles = self.timeframes[timeframe].candles.clone();
        let sender = self.bg_sender.clone();

        thread::spawn(move || {
            let _ = sender.send(BackgroundMessage::LoadingStateChanged(
                LoadingState::ComputingTa { timeframe },
            ));

            println!("Computing TA for timeframe {} in background...", timeframe);
            let start = Instant::now();

            let ta_config = AnalyzerConfig::default();
            let mut analyzer = Analyzer::with_config(ta_config);

            for candle in &candles {
                analyzer.process_candle(*candle);
            }

            let ranges = analyzer.ranges().to_vec();
            let levels = analyzer.all_levels().to_vec();
            let trends = analyzer.all_trends().to_vec();

            println!(
                "  TA computed: {} ranges, {} levels, {} trends in {:.2}s",
                ranges.len(),
                levels.len(),
                trends.len(),
                start.elapsed().as_secs_f32()
            );

            let _ = sender.send(BackgroundMessage::TaComputed {
                timeframe,
                ranges,
                levels,
                trends,
            });
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
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyF, true) | (KeyCode::Home, true) => self.fit_view(),
            (KeyCode::KeyP, true) => self.toggle_ta(),
            (KeyCode::KeyM, true) => self.toggle_macd_panel(),
            (KeyCode::KeyR, true) => self.toggle_replay_mode(),
            (KeyCode::BracketRight, true) => self.replay_step_forward(),
            (KeyCode::BracketLeft, true) => self.replay_step_backward(),
            (KeyCode::Comma, true) => self.replay_decrease_step_size(),
            (KeyCode::Period, true) => self.replay_increase_step_size(),
            (KeyCode::Digit1, true) => self.switch_timeframe(0),
            (KeyCode::Digit2, true) => self.switch_timeframe(1),
            (KeyCode::Digit3, true) => self.switch_timeframe(2),
            (KeyCode::Digit4, true) => self.switch_timeframe(3),
            (KeyCode::Digit5, true) => self.switch_timeframe(4),
            _ => {}
        }
    }

    fn toggle_macd_panel(&mut self) {
        self.show_macd_panel = !self.show_macd_panel;
        self.window.request_redraw();
    }

    fn toggle_ta(&mut self) {
        self.ta_settings.show_ta = !self.ta_settings.show_ta;
        if self.ta_settings.show_ta {
            self.ensure_ta_computed();
            self.update_ta_buffers();
        }
        self.window.request_redraw();
    }

    fn toggle_replay_mode(&mut self) {
        self.replay_mode = !self.replay_mode;
        if self.replay_mode {
            // Entering replay mode - cursor following until click
            self.replay_index = None;
            self.replay_ta_data = None;
            self.replay_timestamp = None;
            self.replay_candles = None;
            self.replay_timeframe_data = None;
            // Default step size to current timeframe
            self.replay_step_timeframe = Timeframe::all()[self.current_timeframe];
        } else {
            // Exiting replay mode - clear replay state
            self.replay_index = None;
            self.replay_ta_data = None;
            self.replay_timestamp = None;
            self.replay_candles = None;
            self.replay_timeframe_data = None;
            // Refresh TA buffers with full data
            if self.ta_settings.show_ta {
                self.update_ta_buffers();
            }
        }
        self.window.request_redraw();
    }

    fn replay_step_forward(&mut self) {
        if !self.replay_mode || self.replay_timestamp.is_none() {
            return;
        }

        let base_candles = &self.timeframes[0].candles; // 1min candles
        if base_candles.is_empty() {
            return;
        }

        let max_timestamp = base_candles.last().map(|c| c.timestamp).unwrap_or(0.0);
        let step_seconds = self.replay_step_timeframe.seconds();

        if let Some(ts) = self.replay_timestamp {
            let new_ts = (ts + step_seconds).min(max_timestamp);
            if new_ts > ts {
                self.replay_timestamp = Some(new_ts);
                self.recompute_replay_candles();
                self.recompute_replay_ta();
                self.window.request_redraw();
            }
        }
    }

    fn replay_step_backward(&mut self) {
        if !self.replay_mode || self.replay_timestamp.is_none() {
            return;
        }

        let base_candles = &self.timeframes[0].candles;
        if base_candles.is_empty() {
            return;
        }

        let min_timestamp = base_candles.first().map(|c| c.timestamp).unwrap_or(0.0);
        let step_seconds = self.replay_step_timeframe.seconds();

        if let Some(ts) = self.replay_timestamp {
            let new_ts = (ts - step_seconds).max(min_timestamp);
            if new_ts < ts {
                self.replay_timestamp = Some(new_ts);
                self.recompute_replay_candles();
                self.recompute_replay_ta();
                self.window.request_redraw();
            }
        }
    }

    fn replay_increase_step_size(&mut self) {
        if !self.replay_mode {
            return;
        }
        let timeframes = Timeframe::all();
        let current_idx = timeframes.iter().position(|&t| t == self.replay_step_timeframe).unwrap_or(0);
        // Increase step size (up to current view timeframe)
        let max_idx = self.current_timeframe;
        if current_idx < max_idx {
            self.replay_step_timeframe = timeframes[current_idx + 1];
            self.window.request_redraw();
        }
    }

    fn replay_decrease_step_size(&mut self) {
        if !self.replay_mode {
            return;
        }
        let timeframes = Timeframe::all();
        let current_idx = timeframes.iter().position(|&t| t == self.replay_step_timeframe).unwrap_or(0);
        // Decrease step size (down to 1min)
        if current_idx > 0 {
            self.replay_step_timeframe = timeframes[current_idx - 1];
            self.window.request_redraw();
        }
    }

    fn set_replay_index(&mut self, index: usize) {
        // Convert candle index to timestamp
        let candles = &self.timeframes[self.current_timeframe].candles;
        if candles.is_empty() {
            return;
        }

        let clamped_idx = index.min(candles.len().saturating_sub(1));
        let timestamp = candles[clamped_idx].timestamp;

        // Also set replay_index for backward compatibility with candle limiting
        self.replay_index = Some(clamped_idx);
        self.replay_timestamp = Some(timestamp);

        self.recompute_replay_candles();
        self.recompute_replay_ta();
        self.window.request_redraw();
    }

    /// Recompute the replay candles from base 1min data.
    /// Re-aggregates candles up to replay_timestamp for accurate partial candle display.
    fn recompute_replay_candles(&mut self) {
        let Some(replay_ts) = self.replay_timestamp else {
            self.replay_candles = None;
            self.replay_index = None;
            self.replay_timeframe_data = None;
            return;
        };

        let base_candles = &self.timeframes[0].candles; // 1min candles
        if base_candles.is_empty() {
            self.replay_index = Some(0);
            self.replay_candles = None;
            self.replay_timeframe_data = None;
            return;
        }

        // Binary search to find the last base candle at or before replay_ts
        let base_end_idx = base_candles
            .binary_search_by(|c| c.timestamp.partial_cmp(&replay_ts).unwrap())
            .unwrap_or_else(|i| i.saturating_sub(1))
            .min(base_candles.len().saturating_sub(1));

        // Get the current view timeframe
        let current_tf = Timeframe::all()[self.current_timeframe];

        // If we're on 1min timeframe, just use the index directly
        if current_tf == Timeframe::Min1 {
            self.replay_index = Some(base_end_idx);
            self.replay_candles = None;
            self.replay_timeframe_data = None;
            return;
        }

        // Re-aggregate base candles up to replay_ts into the current timeframe
        let filtered_base = &base_candles[..=base_end_idx];
        let aggregated = aggregate_candles(filtered_base, current_tf);

        if aggregated.is_empty() {
            self.replay_index = Some(0);
            self.replay_candles = None;
            self.replay_timeframe_data = None;
            return;
        }

        // Update replay_index to the last aggregated candle
        self.replay_index = Some(aggregated.len().saturating_sub(1));

        // Create GPU buffers for the re-aggregated candles
        let tf_label = current_tf.label();
        let tf_data = self.renderer.create_timeframe_data(&self.device, aggregated.clone(), tf_label);

        self.replay_candles = Some(aggregated);
        self.replay_timeframe_data = Some(tf_data);
    }

    fn recompute_replay_ta(&mut self) {
        // Skip TA computation if TA is disabled
        if !self.ta_settings.show_ta {
            self.replay_ta_data = None;
            return;
        }

        // Use replay_candles if available (re-aggregated data), otherwise use timeframe candles
        let candles: Vec<Candle> = if let Some(ref replay_candles) = self.replay_candles {
            replay_candles.clone()
        } else if let Some(replay_idx) = self.replay_index {
            let tf_candles = &self.timeframes[self.current_timeframe].candles;
            if tf_candles.is_empty() || replay_idx == 0 {
                Vec::new()
            } else {
                tf_candles[..=replay_idx.min(tf_candles.len() - 1)].to_vec()
            }
        } else {
            self.replay_ta_data = None;
            return;
        };

        if candles.is_empty() {
            self.replay_ta_data = Some(TimeframeTaData {
                ranges: Vec::new(),
                levels: Vec::new(),
                trends: Vec::new(),
                computed: true,
            });
            self.update_ta_buffers();
            return;
        }

        let ta_config = AnalyzerConfig::default();
        let mut analyzer = Analyzer::with_config(ta_config);

        for candle in &candles {
            analyzer.process_candle(*candle);
        }

        self.replay_ta_data = Some(TimeframeTaData {
            ranges: analyzer.ranges().to_vec(),
            levels: analyzer.all_levels().to_vec(),
            trends: analyzer.all_trends().to_vec(),
            computed: true,
        });

        self.update_ta_buffers();
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
        let ta: &TimeframeTaData = if self.replay_mode && self.replay_index.is_some() {
            self.replay_ta_data.as_ref().unwrap_or(&self.ta_data[tf_idx])
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
                    LevelState::Pending => self.ta_settings.show_active_levels, // Pending shown with active
                    LevelState::Active => self.ta_settings.show_active_levels,
                    LevelState::Hit => self.ta_settings.show_hit_levels,
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
                    // Pending levels: dimmer than active (waiting for price to cross)
                    (CandleDirection::Bullish, LevelState::Pending) => (0.0, 0.5, 0.3, 0.4),
                    (CandleDirection::Bullish, LevelState::Active) => (0.0, 0.8, 0.4, 0.7),
                    (CandleDirection::Bullish, LevelState::Hit) => (0.0, 0.6, 0.3, 0.5),
                    (CandleDirection::Bullish, LevelState::Broken) => (0.0, 0.3, 0.2, 0.3),
                    (CandleDirection::Bearish, LevelState::Pending) => (0.5, 0.15, 0.15, 0.4),
                    (CandleDirection::Bearish, LevelState::Active) => (0.8, 0.2, 0.2, 0.7),
                    (CandleDirection::Bearish, LevelState::Hit) => (0.6, 0.2, 0.2, 0.5),
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

        // Calculate x_max for rendering
        let x_max = (tf.candles.len() as f32) * CANDLE_SPACING;

        // Compute line thickness based on current view
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = (self.config.width as f32 - STATS_PANEL_WIDTH) / chart_height;
        let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);
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
    // MACD Indicator Methods
    // =========================================================================

    /// Add a new MACD indicator instance with the given configuration.
    pub fn add_macd(&mut self, config: MacdConfig) {
        let num_timeframes = self.timeframes.len();
        let id = self.macd_next_id;
        self.macd_next_id += 1;

        let mut instance = MacdInstance::new(id, config, num_timeframes);

        // Compute MACD for current timeframe immediately
        self.compute_macd_for_instance(&mut instance, self.current_timeframe);

        self.macd_instances.push(instance);

        // Create GPU buffers for this instance
        self.create_macd_gpu_buffers_for_instance(self.macd_instances.len() - 1);
    }

    /// Remove a MACD indicator instance by index.
    pub fn remove_macd(&mut self, index: usize) {
        if index < self.macd_instances.len() {
            self.macd_instances.remove(index);
            self.macd_gpu_buffers.remove(index);
        }
    }

    /// Compute MACD for a specific instance and timeframe.
    fn compute_macd_for_instance(&self, instance: &mut MacdInstance, timeframe: usize) {
        let candles = &self.timeframes[timeframe].candles;
        if candles.is_empty() {
            instance.outputs[timeframe] = None;
            return;
        }

        let macd = Macd::new(instance.config.clone());
        let output = macd.calculate_macd(candles);
        instance.outputs[timeframe] = Some(output);
    }

    /// Recompute MACD for all instances on the current timeframe.
    fn recompute_all_macd(&mut self) {
        let tf = self.current_timeframe;
        for i in 0..self.macd_instances.len() {
            let candles = &self.timeframes[tf].candles;
            if candles.is_empty() {
                self.macd_instances[i].outputs[tf] = None;
                continue;
            }

            let macd = Macd::new(self.macd_instances[i].config.clone());
            let output = macd.calculate_macd(candles);
            self.macd_instances[i].outputs[tf] = Some(output);
        }

        // Update GPU buffers
        self.update_macd_gpu_buffers();
    }

    /// Create GPU buffers for a single MACD instance.
    fn create_macd_gpu_buffers_for_instance(&mut self, instance_idx: usize) {
        let instance = &self.macd_instances[instance_idx];
        let tf = self.current_timeframe;

        // Get the output for current timeframe
        let conversion = self.convert_macd_to_gpu_points(instance, tf);

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

        if instance_idx < self.macd_gpu_buffers.len() {
            self.macd_gpu_buffers[instance_idx] = buffers;
        } else {
            self.macd_gpu_buffers.push(buffers);
        }
    }

    /// Convert MACD output to GPU points for rendering.
    fn convert_macd_to_gpu_points(
        &self,
        instance: &MacdInstance,
        timeframe: usize,
    ) -> MacdConversionResult {
        let output = match &instance.outputs[timeframe] {
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

        let config = &instance.config;
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

        for i in 0..self.macd_instances.len() {
            if i >= self.macd_gpu_buffers.len() {
                self.create_macd_gpu_buffers_for_instance(i);
                continue;
            }

            let instance = &self.macd_instances[i];
            let conversion = self.convert_macd_to_gpu_points(instance, tf);

            // Update buffers
            self.queue.write_buffer(
                &self.macd_gpu_buffers[i].macd_line_buffer,
                0,
                bytemuck::cast_slice(&conversion.macd_points),
            );
            self.queue.write_buffer(
                &self.macd_gpu_buffers[i].signal_line_buffer,
                0,
                bytemuck::cast_slice(&conversion.signal_points),
            );
            self.queue.write_buffer(
                &self.macd_gpu_buffers[i].histogram_buffer,
                0,
                bytemuck::cast_slice(&conversion.histogram_points),
            );

            // Calculate first_visible as offset into the indicator's points array.
            // If visible_start is 100 and MACD starts at candle 25, we want to render
            // starting from points array index 75 (= 100 - 25).
            // If visible_start is before MACD starts, we render from the beginning (index 0).
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
                &self.macd_gpu_buffers[i].params_buffer,
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
                &self.macd_gpu_buffers[i].signal_params_buffer,
                0,
                bytemuck::cast_slice(&[signal_params]),
            );

            // Store start indices for later use
            self.macd_gpu_buffers[i].macd_start_index = conversion.macd_start_index;
            self.macd_gpu_buffers[i].signal_start_index = conversion.signal_start_index;
        }

        // Update point counts
        for i in 0..self.macd_instances.len() {
            if i < self.macd_gpu_buffers.len() {
                let instance = &self.macd_instances[i];
                let conversion = self.convert_macd_to_gpu_points(instance, tf);
                self.macd_gpu_buffers[i].macd_point_count = conversion.macd_points.len() as u32;
                self.macd_gpu_buffers[i].signal_point_count = conversion.signal_points.len() as u32;
                self.macd_gpu_buffers[i].histogram_point_count = conversion.histogram_points.len() as u32;
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
        // Compute TA first (if enabled) before update_visible_range calls update_ta_buffers
        if self.ta_settings.show_ta {
            self.ensure_ta_computed();
        }
        // Recompute MACD for new timeframe
        if !self.macd_instances.is_empty() {
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
        if button == MouseButton::Left {
            // Handle replay mode click
            if self.replay_mode && self.replay_index.is_none() && state == ElementState::Pressed {
                // Set replay index from cursor position
                let candle_idx = self.get_cursor_candle_index();
                self.set_replay_index(candle_idx);
                return;
            }

            self.mouse_pressed = state == ElementState::Pressed;
            if !self.mouse_pressed {
                self.last_mouse_pos = None;
            }
        }
    }

    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let current_pos = [position.x as f32, position.y as f32];

        if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let dx = current_pos[0] - last_pos[0];
                let dy = current_pos[1] - last_pos[1];

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
        }

        self.last_mouse_pos = Some(current_pos);

        // Update hover state when TA is visible
        if self.ta_settings.show_ta {
            self.update_hover_state();
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let (scroll_x, scroll_y) = match delta {
            MouseScrollDelta::LineDelta(x, y) => (x, y),
            MouseScrollDelta::PixelDelta(pos) => (pos.x as f32 / 50.0, pos.y as f32 / 50.0),
        };

        let candles = &self.timeframes[self.current_timeframe].candles;

        // Use chart area dimensions (excluding stats panel and volume section)
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = chart_width / chart_height;

        // Calculate cursor position in NDC relative to the chart area
        let cursor_ndc = if let Some(pos) = self.last_mouse_pos {
            // Only apply zoom-to-cursor if cursor is within the chart area
            if pos[0] < chart_width && pos[1] < chart_height {
                [
                    (pos[0] / chart_width) * 2.0 - 1.0,
                    1.0 - (pos[1] / chart_height) * 2.0,
                ]
            } else {
                // Cursor outside chart area, zoom from center
                [0.0, 0.0]
            }
        } else {
            [0.0, 0.0]
        };

        let world_x = self.renderer.camera.position[0]
            + cursor_ndc[0] * self.renderer.camera.scale[0] * aspect;
        let world_y =
            self.renderer.camera.position[1] + cursor_ndc[1] * self.renderer.camera.scale[1];

        let data_width = (candles.len() as f32) * CANDLE_SPACING;
        let max_x_zoom = (data_width / 2.0 / aspect) * 1.2;

        let (min_price, max_price) = candles
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), c| {
                (min.min(c.low), max.max(c.high))
            });
        let price_range = max_price - min_price;
        let max_y_zoom = (price_range / 2.0) * 1.5;

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
        let ta_range_count = self.ta_data[self.current_timeframe].ranges.len();
        let ta_level_count = self.ta_data[self.current_timeframe].levels.len();
        let ta_trend_count = self.ta_data[self.current_timeframe].trends.len();

        // Get hovered element info
        let hovered_range_info = self.hovered_range.and_then(|idx| {
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
        });

        let hovered_level_info = self.hovered_level.and_then(|idx| {
            self.ta_data[self.current_timeframe].levels.get(idx).map(|l| {
                (
                    l.price,
                    l.level_type,
                    l.direction,
                    l.state,
                    l.hits.len(),
                )
            })
        });

        let hovered_trend_info = self.hovered_trend.and_then(|idx| {
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
        });

        // Copy TA settings for egui (will be copied back after)
        let mut ta_settings = self.ta_settings.clone();

        // Copy MACD instances for egui (will apply changes after)
        let mut macd_configs: Vec<MacdConfig> = self.macd_instances.iter().map(|i| i.config.clone()).collect();
        let mut macd_add_requested = false;
        let mut macd_remove_index: Option<usize> = None;
        let show_macd_panel = self.show_macd_panel;

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

            self.stats_panel.show(
                ctx,
                self.current_timeframe,
                self.fps,
                candle_count,
                self.renderer.visible_count,
                self.renderer.current_lod_factor,
                candle.as_ref(),
            );

            // TA control panel
            egui::Window::new("Technical Analysis")
                .default_pos([self.config.width as f32 - STATS_PANEL_WIDTH - 220.0, 10.0])
                .default_width(200.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.checkbox(&mut ta_settings.show_ta, "Enable TA (P)");

                    if ta_settings.show_ta {
                        ui.separator();
                        ui.checkbox(&mut ta_settings.show_ranges, "Show Ranges");

                        ui.separator();
                        ui.label("Level Types:");
                        ui.checkbox(&mut ta_settings.show_hold_levels, "Hold Levels");
                        ui.checkbox(&mut ta_settings.show_greedy_levels, "Greedy Levels");

                        ui.separator();
                        ui.label("Level States:");
                        ui.checkbox(&mut ta_settings.show_active_levels, "Active");
                        ui.checkbox(&mut ta_settings.show_hit_levels, "Hit");
                        ui.checkbox(&mut ta_settings.show_broken_levels, "Broken");

                        ui.separator();
                        ui.checkbox(&mut ta_settings.show_trends, "Show Trends");
                        if ta_settings.show_trends {
                            ui.indent("trend_states", |ui| {
                                ui.checkbox(&mut ta_settings.show_active_trends, "Active");
                                ui.checkbox(&mut ta_settings.show_hit_trends, "Hit");
                                ui.checkbox(&mut ta_settings.show_broken_trends, "Broken");
                            });
                        }

                        ui.separator();
                        ui.label(format!("Ranges: {}", ta_range_count));
                        ui.label(format!("Levels: {}", ta_level_count));
                        ui.label(format!("Trends: {}", ta_trend_count));

                        // Show hovered element info
                        if let Some((dir, count, high, low, start, end)) = hovered_range_info {
                            ui.separator();
                            ui.label("Hovered Range:");
                            ui.label(format!("  Direction: {:?}", dir));
                            ui.label(format!("  Candles: {}", count));
                            ui.label(format!("  High: {:.2}", high));
                            ui.label(format!("  Low: {:.2}", low));
                            ui.label(format!("  Index: {} - {}", start, end));
                        }

                        if let Some((price, ltype, dir, state, hits)) = hovered_level_info {
                            ui.separator();
                            ui.label("Hovered Level:");
                            ui.label(format!("  Price: {:.2}", price));
                            ui.label(format!("  Type: {:?}", ltype));
                            ui.label(format!("  Direction: {:?}", dir));
                            ui.label(format!("  State: {:?}", state));
                            ui.label(format!("  Hits: {}", hits));
                        }

                        if let Some((dir, state, start_price, end_price, start_idx, end_idx, hits)) = hovered_trend_info {
                            ui.separator();
                            ui.label("Hovered Trend:");
                            ui.label(format!("  Direction: {:?}", dir));
                            ui.label(format!("  State: {:?}", state));
                            ui.label(format!("  Start: {:.2} @ idx {}", start_price, start_idx));
                            ui.label(format!("  End: {:.2} @ idx {}", end_price, end_idx));
                            ui.label(format!("  Hits: {}", hits));
                        }
                    }
                });

            // MACD Indicators panel
            if show_macd_panel {
                egui::Window::new("MACD Indicators")
                    .default_pos([self.config.width as f32 - STATS_PANEL_WIDTH - 440.0, 10.0])
                    .default_width(200.0)
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("Press M to toggle this panel");
                        ui.separator();

                        // Add new MACD button
                        if ui.button("+ Add MACD").clicked() {
                            macd_add_requested = true;
                        }

                        ui.separator();

                        // List existing MACD instances
                        for (i, config) in macd_configs.iter_mut().enumerate() {
                            ui.push_id(i, |ui| {
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(format!("MACD #{}", i + 1));
                                        if ui.small_button("X").clicked() {
                                            macd_remove_index = Some(i);
                                        }
                                    });

                                    // Enable toggle
                                    ui.checkbox(&mut config.enabled, "Enabled");

                                    // Period settings
                                    ui.horizontal(|ui| {
                                        ui.label("Fast:");
                                        let mut fast = config.fast_period as i32;
                                        if ui.add(egui::DragValue::new(&mut fast).range(1..=100)).changed() {
                                            config.fast_period = fast.max(1) as usize;
                                        }
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Slow:");
                                        let mut slow = config.slow_period as i32;
                                        if ui.add(egui::DragValue::new(&mut slow).range(1..=200)).changed() {
                                            config.slow_period = slow.max(1) as usize;
                                        }
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Signal:");
                                        let mut signal = config.signal_period as i32;
                                        if ui.add(egui::DragValue::new(&mut signal).range(1..=100)).changed() {
                                            config.signal_period = signal.max(1) as usize;
                                        }
                                    });

                                    // Color pickers (compact)
                                    ui.horizontal(|ui| {
                                        ui.label("MACD:");
                                        ui.color_edit_button_rgb(&mut config.macd_color);
                                        ui.label("Sig:");
                                        ui.color_edit_button_rgb(&mut config.signal_color);
                                    });
                                });
                            });
                        }

                        if macd_configs.is_empty() {
                            ui.label("No MACD indicators. Click '+ Add MACD' to create one.");
                        }
                    });
            }

            // Loading indicator overlay
            if self.loading_state.is_loading() {
                let screen_rect = ctx.screen_rect();
                let center = screen_rect.center();

                // Semi-transparent dark overlay
                egui::Area::new(egui::Id::new("loading_overlay"))
                    .fixed_pos(egui::Pos2::ZERO)
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        let painter = ui.painter();
                        painter.rect_filled(
                            screen_rect,
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
                        );
                    });

                // Loading window
                egui::Window::new("Loading")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .fixed_pos(center)
                    .show(ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(10.0);
                            ui.spinner();
                            ui.add_space(10.0);
                            ui.label(self.loading_state.message());
                            ui.add_space(10.0);
                        });
                    });
            }

            // Replay mode overlay and cursor line
            if self.replay_mode {
                let aspect = chart_width / chart_height;
                let (x_min, _x_max) = self.renderer.camera.visible_x_range(aspect);

                // Determine cursor X position
                let cursor_x = if let Some(replay_ts) = self.replay_timestamp {
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
                } else if let Some(pos) = self.last_mouse_pos {
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
                            let (color, width) = if self.replay_index.is_some() {
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
                let base_candle_info = if let Some(ts) = self.replay_timestamp {
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
                            if let Some(idx) = self.replay_index {
                                ui.label(format!("Candle {}/{}", idx + 1, candle_count));
                                if let Some((base_idx, base_total)) = base_candle_info {
                                    ui.label(format!("(1m: {}/{})", base_idx, base_total));
                                }
                            } else {
                                ui.label("Click to set position");
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(format!("Step: {}", self.replay_step_timeframe.label()));
                            ui.label("| [ ] step | ,. size | R exit");
                        });
                    });
            }
        });

        // Update TA settings if changed
        let ta_changed = ta_settings.show_ta != self.ta_settings.show_ta
            || ta_settings.show_ranges != self.ta_settings.show_ranges
            || ta_settings.show_hold_levels != self.ta_settings.show_hold_levels
            || ta_settings.show_greedy_levels != self.ta_settings.show_greedy_levels
            || ta_settings.show_active_levels != self.ta_settings.show_active_levels
            || ta_settings.show_hit_levels != self.ta_settings.show_hit_levels
            || ta_settings.show_broken_levels != self.ta_settings.show_broken_levels
            || ta_settings.show_trends != self.ta_settings.show_trends
            || ta_settings.show_active_trends != self.ta_settings.show_active_trends
            || ta_settings.show_hit_trends != self.ta_settings.show_hit_trends
            || ta_settings.show_broken_trends != self.ta_settings.show_broken_trends;

        if ta_changed {
            let was_ta_enabled = self.ta_settings.show_ta;
            self.ta_settings = ta_settings;
            if self.ta_settings.show_ta {
                // If TA was just enabled via UI, ensure it's computed
                if !was_ta_enabled {
                    self.ensure_ta_computed();
                }
                self.update_ta_buffers();
            }
        }

        // Apply MACD changes after egui run
        let macd_changed = {
            let mut changed = false;

            // Handle add request
            if macd_add_requested {
                self.add_macd(MacdConfig::default());
                changed = true;
            }

            // Handle remove request
            if let Some(idx) = macd_remove_index {
                self.remove_macd(idx);
                changed = true;
            }

            // Update configs that changed
            for (i, new_config) in macd_configs.into_iter().enumerate() {
                if i < self.macd_instances.len() {
                    let old_config = &self.macd_instances[i].config;
                    if old_config.fast_period != new_config.fast_period
                        || old_config.slow_period != new_config.slow_period
                        || old_config.signal_period != new_config.signal_period
                        || old_config.enabled != new_config.enabled
                        || old_config.macd_color != new_config.macd_color
                        || old_config.signal_color != new_config.signal_color
                    {
                        self.macd_instances[i].config = new_config;
                        // Recompute MACD for this instance
                        let candles = &self.timeframes[self.current_timeframe].candles;
                        if !candles.is_empty() {
                            let macd = Macd::new(self.macd_instances[i].config.clone());
                            let output = macd.calculate_macd(candles);
                            self.macd_instances[i].outputs[self.current_timeframe] = Some(output);
                        }
                        changed = true;
                    }
                }
            }

            changed
        };

        if macd_changed && !self.macd_instances.is_empty() {
            self.update_macd_gpu_buffers();
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
            let use_replay_data = self.replay_mode && self.replay_timeframe_data.is_some();
            let replay_tf = self.replay_timeframe_data.as_ref();

            // Update render params for replay data (different price normalization and first_visible)
            if use_replay_data {
                if let Some(ref rtf) = self.replay_timeframe_data {
                    let aspect = chart_width / chart_height;
                    let (x_min, x_max) = self.renderer.camera.visible_x_range(aspect);
                    let (y_min, y_max) = self.renderer.camera.visible_y_range(aspect);

                    // Use fixed base values for replay - no LOD adjustment needed
                    // BASE_CANDLE_WIDTH = 0.8, which is less than CANDLE_SPACING = 1.2
                    let candle_width = 0.8;
                    let wick_width = candle_width * 0.15;

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
                        _padding1: 0.0,
                        _padding2: 0.0,
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
            }

            // Calculate effective visible count for replay mode
            let effective_visible_count = if use_replay_data {
                // When using replay data, show all the re-aggregated candles
                replay_tf.map(|r| r.candles.len() as u32).unwrap_or(0)
            } else if self.replay_mode {
                if let Some(replay_idx) = self.replay_index {
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
            render_pass.draw(0..VERTICES_PER_CANDLE, 0..effective_visible_count);

            // Render TA (ranges and levels) if enabled
            if self.ta_settings.show_ta {
                // Use replay TA data if in replay mode with locked position
                let ta: &TimeframeTaData = if self.replay_mode && self.replay_index.is_some() {
                    self.replay_ta_data.as_ref().unwrap_or(&self.ta_data[self.current_timeframe])
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
                            LevelState::Pending => self.ta_settings.show_active_levels,
                            LevelState::Active => self.ta_settings.show_active_levels,
                            LevelState::Hit => self.ta_settings.show_hit_levels,
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
            for (i, instance) in self.macd_instances.iter().enumerate() {
                if !instance.config.enabled {
                    continue;
                }
                if i >= self.macd_gpu_buffers.len() {
                    continue;
                }

                let buffers = &self.macd_gpu_buffers[i];

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

            let tf_labels = ["1m", "15m", "1h", "1w", "1M"];
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
        if let Some(pos) = self.last_mouse_pos {
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
        let pos = self.last_mouse_pos?;
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
                LevelState::Pending => self.ta_settings.show_active_levels,
                LevelState::Active => self.ta_settings.show_active_levels,
                LevelState::Hit => self.ta_settings.show_hit_levels,
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
