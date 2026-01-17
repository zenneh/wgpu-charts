use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use egui::Color32;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// OHLCV Candle data structure (CPU side)
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub timestamp: f64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

impl Candle {
    pub fn new(timestamp: f64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Self {
        Self { timestamp, open, high, low, close, volume }
    }
}

// Timeframe enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeframe {
    Min1,    // 1 minute (base data)
    Min15,   // 15 minutes
    Hour1,   // 1 hour
    Week1,   // 1 week
    Month1,  // 1 month
}

impl Timeframe {
    fn seconds(&self) -> f64 {
        match self {
            Timeframe::Min1 => 60.0,
            Timeframe::Min15 => 60.0 * 15.0,
            Timeframe::Hour1 => 60.0 * 60.0,
            Timeframe::Week1 => 60.0 * 60.0 * 24.0 * 7.0,
            Timeframe::Month1 => 60.0 * 60.0 * 24.0 * 30.0,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Timeframe::Min1 => "1m",
            Timeframe::Min15 => "15m",
            Timeframe::Hour1 => "1h",
            Timeframe::Week1 => "1w",
            Timeframe::Month1 => "1M",
        }
    }
}

// Aggregate candles into a larger timeframe
fn aggregate_candles(candles: &[Candle], timeframe: Timeframe) -> Vec<Candle> {
    if candles.is_empty() {
        return Vec::new();
    }

    let interval = timeframe.seconds();
    let mut aggregated = Vec::new();
    let mut current_bucket: Option<Candle> = None;
    let mut current_bucket_start = 0.0;

    for candle in candles {
        let bucket_start = (candle.timestamp / interval).floor() * interval;

        if let Some(ref mut agg) = current_bucket {
            if bucket_start == current_bucket_start {
                // Same bucket - update high, low, close, accumulate volume
                agg.high = agg.high.max(candle.high);
                agg.low = agg.low.min(candle.low);
                agg.close = candle.close;
                agg.volume += candle.volume;
            } else {
                // New bucket - save current and start new
                aggregated.push(*agg);
                current_bucket = Some(Candle::new(
                    bucket_start,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ));
                current_bucket_start = bucket_start;
            }
        } else {
            // First candle
            current_bucket = Some(Candle::new(
                bucket_start,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ));
            current_bucket_start = bucket_start;
        }
    }

    // Don't forget the last bucket
    if let Some(agg) = current_bucket {
        aggregated.push(agg);
    }

    aggregated
}

// Pre-computed timeframe data with GPU buffers for candles and volume
pub struct TimeframeData {
    pub candles: Vec<Candle>,
    pub candle_buffer: wgpu::Buffer,
    pub candle_bind_group: wgpu::BindGroup,
    pub volume_buffer: wgpu::Buffer,
    pub volume_bind_group: wgpu::BindGroup,
    pub count: u32,
    pub max_volume: f32,
}

// GPU-compatible candle data for storage buffer
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CandleGpu {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

impl From<&Candle> for CandleGpu {
    fn from(c: &Candle) -> Self {
        Self {
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
        }
    }
}

// GPU-compatible volume bar data
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeGpu {
    pub volume: f32,
    pub is_bullish: u32,
    pub _padding1: f32,
    pub _padding2: f32,
}

impl VolumeGpu {
    fn from_candle(c: &Candle) -> Self {
        Self {
            volume: c.volume,
            is_bullish: if c.close >= c.open { 1 } else { 0 },
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

// Volume render parameters
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeRenderParams {
    pub first_visible: u32,
    pub bar_width: f32,
    pub bar_spacing: f32,
    pub max_volume: f32,
}

// Render parameters uniform - passed to shader for instancing
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderParams {
    pub first_visible: u32,
    pub candle_width: f32,
    pub candle_spacing: f32,
    pub wick_width: f32,
}

// Guideline GPU struct (16 bytes aligned)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GuidelineGpu {
    pub y_value: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

// Guideline render parameters
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GuidelineParams {
    pub x_min: f32,
    pub x_max: f32,
    pub line_thickness: f32,
    pub count: u32,
}

const MAX_GUIDELINES: usize = 32;

// 2D Camera for panning the chart
pub struct Camera {
    pub position: [f32; 2],
    pub scale: [f32; 2],
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
        }
    }

    pub fn build_view_projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let half_width = self.scale[0] * aspect;
        let half_height = self.scale[1];

        let left = self.position[0] - half_width;
        let right = self.position[0] + half_width;
        let bottom = self.position[1] - half_height;
        let top = self.position[1] + half_height;

        let sx = 2.0 / (right - left);
        let sy = 2.0 / (top - bottom);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);

        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, 0.0, 1.0],
        ]
    }

    // Get visible X range in world coordinates
    pub fn visible_x_range(&self, aspect: f32) -> (f32, f32) {
        let half_width = self.scale[0] * aspect;
        (self.position[0] - half_width, self.position[0] + half_width)
    }

    // Get visible Y range in world coordinates
    pub fn visible_y_range(&self, _aspect: f32) -> (f32, f32) {
        let half_height = self.scale[1];
        (self.position[1] - half_height, self.position[1] + half_height)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.build_view_projection_matrix(aspect);
    }
}

// Constants for candle rendering
const BASE_CANDLE_WIDTH: f32 = 0.8;
const CANDLE_SPACING: f32 = 1.2;
const VERTICES_PER_CANDLE: u32 = 18; // 6 triangles * 3 vertices
const MIN_CANDLE_PIXELS: f32 = 3.0; // Minimum candle width in pixels

// Layout constants
const STATS_PANEL_WIDTH: f32 = 200.0;
const VOLUME_HEIGHT_RATIO: f32 = 0.2; // Volume panel takes 20% of chart area height

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    // Candle rendering
    candle_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Volume rendering
    volume_pipeline: wgpu::RenderPipeline,
    volume_camera: Camera,
    volume_camera_uniform: CameraUniform,
    volume_camera_buffer: wgpu::Buffer,
    volume_camera_bind_group: wgpu::BindGroup,
    volume_params_buffer: wgpu::Buffer,
    // Guideline rendering
    guideline_pipeline: wgpu::RenderPipeline,
    guideline_buffer: wgpu::Buffer,
    guideline_params_buffer: wgpu::Buffer,
    guideline_bind_group: wgpu::BindGroup,
    guideline_count: u32,
    // Timeframe data (pre-computed for each timeframe)
    timeframes: Vec<TimeframeData>,
    current_timeframe: usize,
    render_params_buffer: wgpu::Buffer,
    // Visible range (for viewport culling)
    visible_start: u32,
    visible_count: u32,
    // Egui integration
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // Mouse state
    mouse_pressed: bool,
    last_mouse_pos: Option<[f32; 2]>,
    // FPS tracking
    last_frame_time: Instant,
    frame_count: u32,
    fps: f32,
}

// Parse datetime string "YYYY-MM-DD HH:MM:SS" to unix timestamp
fn parse_datetime(s: &str) -> Option<f64> {
    // Format: "2017-08-17 04:00:00"
    let parts: Vec<&str> = s.split(&['-', ' ', ':']).collect();
    if parts.len() < 6 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    let hour: u32 = parts[3].parse().ok()?;
    let min: u32 = parts[4].parse().ok()?;
    let sec: u32 = parts[5].parse().ok()?;

    // Simple timestamp calculation (not accounting for leap seconds, etc.)
    // Days since Unix epoch (1970-01-01)
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
    }
    let month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    days += month_days[month as usize - 1] as i64;
    if month > 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        days += 1;
    }
    days += (day - 1) as i64;

    let timestamp = days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64;
    Some(timestamp as f64)
}

// Load candles from CSV file and analyze for missing data
// Expected format: Open time,Open,High,Low,Close,Volume,... (comma-separated)
fn load_candles_from_csv<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Candle>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .from_path(path)?;

    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;
        // Column 0 = datetime string, 1=open, 2=high, 3=low, 4=close, 5=volume
        let datetime_str = record.get(0).unwrap_or("");
        let timestamp = parse_datetime(datetime_str).unwrap_or(0.0);
        let open: f32 = record.get(1).unwrap_or("0").parse()?;
        let high: f32 = record.get(2).unwrap_or("0").parse()?;
        let low: f32 = record.get(3).unwrap_or("0").parse()?;
        let close: f32 = record.get(4).unwrap_or("0").parse()?;
        let volume: f32 = record.get(5).unwrap_or("0").parse()?;

        candles.push(Candle::new(timestamp, open, high, low, close, volume));
    }

    // Analyze for missing data points
    let timestamps: Vec<f64> = candles.iter().map(|c| c.timestamp).collect();
    analyze_data_gaps(&timestamps);

    // Data is already in chronological order (oldest first)
    Ok(candles)
}

fn analyze_data_gaps(timestamps: &[f64]) {
    if timestamps.len() < 2 {
        println!("Not enough data points to analyze gaps");
        return;
    }

    // Detect the interval (should be 60 seconds for minute data)
    let mut intervals: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for window in timestamps.windows(2) {
        let diff = (window[1] - window[0]).round() as i64;
        *intervals.entry(diff).or_insert(0) += 1;
    }

    // Find the most common interval (expected interval)
    let expected_interval = intervals
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(interval, _)| *interval)
        .unwrap_or(60);

    // Count gaps and missing data points
    let mut total_gaps = 0;
    let mut total_missing = 0;
    let mut largest_gap = 0i64;
    let mut largest_gap_start = 0f64;

    for window in timestamps.windows(2) {
        let diff = (window[1] - window[0]).round() as i64;
        if diff > expected_interval {
            total_gaps += 1;
            let missing_in_gap = (diff / expected_interval) - 1;
            total_missing += missing_in_gap;

            if diff > largest_gap {
                largest_gap = diff;
                largest_gap_start = window[0];
            }
        }
    }

    // Calculate time span
    let first_ts = timestamps.first().unwrap_or(&0.0);
    let last_ts = timestamps.last().unwrap_or(&0.0);
    let total_span_seconds = last_ts - first_ts;
    let expected_points = (total_span_seconds / expected_interval as f64).round() as i64;
    let actual_points = timestamps.len() as i64;
    let coverage_pct = (actual_points as f64 / expected_points as f64) * 100.0;

    println!("=== Data Analysis ===");
    println!("Total data points: {}", timestamps.len());
    println!("Expected interval: {} seconds", expected_interval);
    println!("Time span: {:.1} days", total_span_seconds / 86400.0);
    println!("Expected data points: {}", expected_points);
    println!("Missing data points: {} ({:.2}% coverage)", total_missing, coverage_pct);
    println!("Number of gaps: {}", total_gaps);
    if largest_gap > 0 {
        println!("Largest gap: {} seconds ({:.1} hours) at timestamp {}",
            largest_gap, largest_gap as f64 / 3600.0, largest_gap_start);
    }
    println!("=====================");
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // Load Bitcoin candle data from CSV (1-minute base data)
        let base_candles = load_candles_from_csv("data/btc-new.csv")?;
        println!("Loaded {} base candles", base_candles.len());

        // Pre-compute all timeframes
        let timeframe_types = [
            Timeframe::Min1,
            Timeframe::Min15,
            Timeframe::Hour1,
            Timeframe::Week1,
            Timeframe::Month1,
        ];

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
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
            }, None)
            .await?;

        let candle_shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let volume_shader = device.create_shader_module(wgpu::include_wgsl!("volume.wgsl"));
        let guideline_shader = device.create_shader_module(wgpu::include_wgsl!("guidelines.wgsl"));

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

        // Camera bind group layout (shared between candle and volume pipelines)
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        // Candle render params
        let render_params = RenderParams {
            first_visible: 0,
            candle_width: BASE_CANDLE_WIDTH,
            candle_spacing: CANDLE_SPACING,
            wick_width: CANDLE_SPACING * 0.08,
        };

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::cast_slice(&[render_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Volume render params
        let volume_params = VolumeRenderParams {
            first_visible: 0,
            bar_width: BASE_CANDLE_WIDTH,
            bar_spacing: CANDLE_SPACING,
            max_volume: 1.0,
        };

        let volume_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Volume Params Buffer"),
            contents: bytemuck::cast_slice(&[volume_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Candle data bind group layout
        let candle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("candle_bind_group_layout"),
            });

        // Volume data bind group layout (same structure)
        let volume_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("volume_bind_group_layout"),
            });

        // Guideline bind group layout
        let guideline_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("guideline_bind_group_layout"),
            });

        // Pre-compute and create GPU buffers for all timeframes
        let mut timeframes = Vec::new();
        for tf in &timeframe_types {
            let candles = if *tf == Timeframe::Min1 {
                base_candles.clone()
            } else {
                aggregate_candles(&base_candles, *tf)
            };
            println!("  {} timeframe: {} candles", tf.label(), candles.len());

            // Candle GPU data
            let candles_gpu: Vec<CandleGpu> = candles.iter().map(CandleGpu::from).collect();
            let candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Candle Buffer {}", tf.label())),
                contents: bytemuck::cast_slice(&candles_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let candle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &candle_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: candle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: render_params_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("Candle Bind Group {}", tf.label())),
            });

            // Volume GPU data
            let volume_gpu: Vec<VolumeGpu> = candles.iter().map(VolumeGpu::from_candle).collect();
            let max_volume = candles.iter().map(|c| c.volume).fold(0.0f32, f32::max);
            let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Volume Buffer {}", tf.label())),
                contents: bytemuck::cast_slice(&volume_gpu),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let volume_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &volume_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: volume_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: volume_params_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("Volume Bind Group {}", tf.label())),
            });

            timeframes.push(TimeframeData {
                candles,
                candle_buffer,
                candle_bind_group,
                volume_buffer,
                volume_bind_group,
                count: candles_gpu.len() as u32,
                max_volume,
            });
        }

        // Camera setup - calculate initial view to fit first timeframe
        let initial_candles = &timeframes[0].candles;
        let mut camera = Camera::new();
        let (min_price, max_price) = initial_candles.iter().fold((f32::MAX, f32::MIN), |(min, max), c| {
            (min.min(c.low), max.max(c.high))
        });
        let price_center = (min_price + max_price) / 2.0;
        let price_range = (max_price - min_price) * 1.1;

        let x_max = (initial_candles.len() as f32) * CANDLE_SPACING;
        let x_center = x_max / 2.0;

        camera.position = [x_center, price_center];
        camera.scale = [x_max / 2.0, price_range / 2.0];

        let mut camera_uniform = CameraUniform::new();
        let aspect = config.width as f32 / config.height as f32;
        camera_uniform.update_view_proj(&camera, aspect);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Volume camera (separate for the volume viewport)
        let mut volume_camera = Camera::new();
        volume_camera.position = [x_center, 0.5];
        volume_camera.scale = [x_max / 2.0, 0.5];

        let mut volume_camera_uniform = CameraUniform::new();
        volume_camera_uniform.update_view_proj(&volume_camera, aspect);

        let volume_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Volume Camera Buffer"),
            contents: bytemuck::cast_slice(&[volume_camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let volume_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: volume_camera_buffer.as_entire_binding(),
            }],
            label: Some("volume_camera_bind_group"),
        });

        // Guideline buffers (initially empty, populated dynamically)
        let initial_guidelines: Vec<GuidelineGpu> = vec![GuidelineGpu { y_value: 0.0, r: 0.0, g: 0.0, b: 0.0 }; MAX_GUIDELINES];
        let guideline_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Guideline Buffer"),
            contents: bytemuck::cast_slice(&initial_guidelines),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let guideline_params = GuidelineParams {
            x_min: 0.0,
            x_max: x_max,
            line_thickness: price_range * 0.001,
            count: 0,
        };
        let guideline_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Guideline Params Buffer"),
            contents: bytemuck::cast_slice(&[guideline_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let guideline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &guideline_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: guideline_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: guideline_params_buffer.as_entire_binding(),
                },
            ],
            label: Some("guideline_bind_group"),
        });

        // Candle render pipeline
        let candle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Candle Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &candle_bind_group_layout],
                push_constant_ranges: &[],
            });

        let candle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Candle Pipeline"),
            layout: Some(&candle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &candle_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &candle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Volume render pipeline
        let volume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Volume Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &volume_bind_group_layout],
                push_constant_ranges: &[],
            });

        let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Volume Pipeline"),
            layout: Some(&volume_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &volume_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &volume_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Guideline render pipeline (with alpha blending for transparency)
        let guideline_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Guideline Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &guideline_bind_group_layout],
                push_constant_ranges: &[],
            });

        let guideline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Guideline Pipeline"),
            layout: Some(&guideline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &guideline_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &guideline_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

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

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            candle_pipeline,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            volume_pipeline,
            volume_camera,
            volume_camera_uniform,
            volume_camera_buffer,
            volume_camera_bind_group,
            volume_params_buffer,
            guideline_pipeline,
            guideline_buffer,
            guideline_params_buffer,
            guideline_bind_group,
            guideline_count: 0,
            timeframes,
            current_timeframe: 0,
            render_params_buffer,
            visible_start: 0,
            visible_count: 0,
            egui_ctx,
            egui_state,
            egui_renderer,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_frame_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
        };

        state.update_visible_range();
        Ok(state)
    }

    // Calculate which candles are visible and update render params
    fn update_visible_range(&mut self) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let (x_min, x_max) = self.camera.visible_x_range(aspect);

        let candle_count = self.timeframes[self.current_timeframe].candles.len();

        // Convert world X to candle index
        let first_idx = ((x_min / CANDLE_SPACING).floor() as i32 - 1).max(0) as u32;
        let last_idx = ((x_max / CANDLE_SPACING).ceil() as i32 + 1).min(candle_count as i32) as u32;

        self.visible_start = first_idx;
        self.visible_count = if last_idx > first_idx {
            last_idx - first_idx
        } else {
            0
        };

        // Calculate adaptive candle width to ensure minimum pixel visibility
        // world_units_per_pixel = visible_width / screen_width
        let visible_width = x_max - x_min;
        let world_units_per_pixel = visible_width / self.config.width as f32;
        let min_world_width = MIN_CANDLE_PIXELS * world_units_per_pixel;
        let candle_width = BASE_CANDLE_WIDTH.max(min_world_width);

        // Adaptive wick width - minimum 1 pixel
        let base_wick_width = CANDLE_SPACING * 0.08;
        let min_wick_width = 1.0 * world_units_per_pixel;
        let wick_width = base_wick_width.max(min_wick_width);

        // Update render params on GPU
        let render_params = RenderParams {
            first_visible: self.visible_start,
            candle_width,
            candle_spacing: CANDLE_SPACING,
            wick_width,
        };

        self.queue.write_buffer(
            &self.render_params_buffer,
            0,
            bytemuck::cast_slice(&[render_params]),
        );

        // Update volume params with same adaptive width and visible max volume
        let tf = &self.timeframes[self.current_timeframe];
        let visible_max_volume = tf.candles
            .iter()
            .skip(self.visible_start as usize)
            .take(self.visible_count as usize)
            .map(|c| c.volume)
            .fold(0.0f32, f32::max)
            .max(1.0); // Prevent division by zero

        let volume_params = VolumeRenderParams {
            first_visible: self.visible_start,
            bar_width: candle_width, // Use same adaptive width as candles
            bar_spacing: CANDLE_SPACING,
            max_volume: visible_max_volume,
        };
        self.queue.write_buffer(
            &self.volume_params_buffer,
            0,
            bytemuck::cast_slice(&[volume_params]),
        );

        // Update price guidelines based on visible price range
        self.update_guidelines(x_min, x_max);
    }

    // Calculate nice round price levels for guidelines
    fn update_guidelines(&mut self, x_min: f32, x_max: f32) {
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = (self.config.width as f32 - STATS_PANEL_WIDTH) / chart_height;
        let (y_min, y_max) = self.camera.visible_y_range(aspect);

        let price_range = y_max - y_min;

        // Find a nice step size (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, etc.)
        let target_lines = 8.0; // Aim for about 8 guidelines
        let raw_step = price_range / target_lines;

        // Round to a nice number
        let magnitude = 10f32.powf(raw_step.log10().floor());
        let normalized = raw_step / magnitude;
        let nice_step = if normalized < 1.5 {
            magnitude
        } else if normalized < 3.5 {
            2.0 * magnitude
        } else if normalized < 7.5 {
            5.0 * magnitude
        } else {
            10.0 * magnitude
        };

        // Generate guidelines at nice round prices
        let first_line = (y_min / nice_step).ceil() * nice_step;
        let mut guidelines = Vec::with_capacity(MAX_GUIDELINES);

        let mut y = first_line;
        while y < y_max && guidelines.len() < MAX_GUIDELINES {
            guidelines.push(GuidelineGpu {
                y_value: y,
                r: 0.3,  // Gray color
                g: 0.3,
                b: 0.35,
            });
            y += nice_step;
        }

        // Calculate adaptive line thickness (minimum 1 pixel)
        let world_units_per_pixel = price_range / chart_height;
        let line_thickness = (1.5 * world_units_per_pixel).max(price_range * 0.0005);

        self.guideline_count = guidelines.len() as u32;

        // Pad to MAX_GUIDELINES
        while guidelines.len() < MAX_GUIDELINES {
            guidelines.push(GuidelineGpu { y_value: 0.0, r: 0.0, g: 0.0, b: 0.0 });
        }

        // Update GPU buffers
        self.queue.write_buffer(
            &self.guideline_buffer,
            0,
            bytemuck::cast_slice(&guidelines),
        );

        let guideline_params = GuidelineParams {
            x_min,
            x_max,
            line_thickness,
            count: self.guideline_count,
        };
        self.queue.write_buffer(
            &self.guideline_params_buffer,
            0,
            bytemuck::cast_slice(&[guideline_params]),
        );
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.update_camera();
            self.update_visible_range();
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyF, true) | (KeyCode::Home, true) => self.fit_view(),
            // Timeframe switching with number keys 1-5
            (KeyCode::Digit1, true) => self.switch_timeframe(0),
            (KeyCode::Digit2, true) => self.switch_timeframe(1),
            (KeyCode::Digit3, true) => self.switch_timeframe(2),
            (KeyCode::Digit4, true) => self.switch_timeframe(3),
            (KeyCode::Digit5, true) => self.switch_timeframe(4),
            _ => {}
        }
    }

    // Switch to a different timeframe, preserving view position by timestamp
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

        // Find the timestamp at current camera X position
        let old_candle_idx = (self.camera.position[0] / CANDLE_SPACING).round() as usize;
        let old_candle_idx = old_candle_idx.min(old_candles.len().saturating_sub(1));
        let target_timestamp = old_candles[old_candle_idx].timestamp;

        // Binary search for the closest candle in new timeframe
        let new_candle_idx = match new_candles.binary_search_by(|c| {
            c.timestamp.partial_cmp(&target_timestamp).unwrap()
        }) {
            Ok(idx) => idx,
            Err(idx) => idx.min(new_candles.len().saturating_sub(1)),
        };

        // Calculate new camera X position
        let new_x = (new_candle_idx as f32) * CANDLE_SPACING;

        // Scale the horizontal zoom proportionally to the ratio of candle counts
        // This keeps roughly the same "time span" visible
        let ratio = old_candles.len() as f32 / new_candles.len() as f32;
        let new_scale_x = self.camera.scale[0] / ratio;

        self.camera.position[0] = new_x;
        self.camera.scale[0] = new_scale_x.max(5.0); // Maintain minimum zoom

        self.current_timeframe = index;
        self.update_camera();
        self.update_visible_range();
        self.window.request_redraw();
    }

    // Fit all candles in view (press F or Home)
    fn fit_view(&mut self) {
        let candles = &self.timeframes[self.current_timeframe].candles;
        if candles.is_empty() {
            return;
        }

        // Calculate price range
        let (min_price, max_price) = candles.iter().fold((f32::MAX, f32::MIN), |(min, max), c| {
            (min.min(c.low), max.max(c.high))
        });
        let price_center = (min_price + max_price) / 2.0;
        let price_range = (max_price - min_price) * 1.1; // 10% padding

        // Calculate x range
        let x_max = (candles.len() as f32) * CANDLE_SPACING;
        let x_center = x_max / 2.0;

        // Account for aspect ratio
        let aspect = self.config.width as f32 / self.config.height as f32;
        let x_scale = x_max / 2.0 / aspect;

        self.camera.position = [x_center, price_center];
        self.camera.scale = [x_scale, price_range / 2.0];

        self.update_camera();
        self.update_visible_range();
        self.window.request_redraw();
    }

    fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        if button == MouseButton::Left {
            self.mouse_pressed = state == ElementState::Pressed;
            if !self.mouse_pressed {
                self.last_mouse_pos = None;
            }
        }
    }

    fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let current_pos = [position.x as f32, position.y as f32];

        if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let dx = current_pos[0] - last_pos[0];
                let dy = current_pos[1] - last_pos[1];

                let aspect = self.config.width as f32 / self.config.height as f32;
                let world_dx = -dx * (self.camera.scale[0] * aspect * 2.0) / self.config.width as f32;
                let world_dy = dy * (self.camera.scale[1] * 2.0) / self.config.height as f32;

                self.camera.position[0] += world_dx;
                self.camera.position[1] += world_dy;

                self.update_camera();
                self.update_visible_range();
                self.window.request_redraw();
            }
        }

        self.last_mouse_pos = Some(current_pos);
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let (scroll_x, scroll_y) = match delta {
            MouseScrollDelta::LineDelta(x, y) => (x, y),
            MouseScrollDelta::PixelDelta(pos) => (pos.x as f32 / 50.0, pos.y as f32 / 50.0),
        };

        let candles = &self.timeframes[self.current_timeframe].candles;
        let aspect = self.config.width as f32 / self.config.height as f32;

        // Get cursor position in normalized screen coordinates (-1 to 1)
        let cursor_ndc = if let Some(pos) = self.last_mouse_pos {
            [
                (pos[0] / self.config.width as f32) * 2.0 - 1.0,
                1.0 - (pos[1] / self.config.height as f32) * 2.0, // Y is flipped
            ]
        } else {
            [0.0, 0.0] // Default to center if no cursor position
        };

        // Convert cursor to world coordinates (before zoom)
        let world_x = self.camera.position[0] + cursor_ndc[0] * self.camera.scale[0] * aspect;
        let world_y = self.camera.position[1] + cursor_ndc[1] * self.camera.scale[1];

        // Calculate zoom limits
        let data_width = (candles.len() as f32) * CANDLE_SPACING;
        let max_x_zoom = (data_width / 2.0 / aspect) * 1.2;

        let (min_price, max_price) = candles.iter().fold((f32::MAX, f32::MIN), |(min, max), c| {
            (min.min(c.low), max.max(c.high))
        });
        let price_range = max_price - min_price;
        let max_y_zoom = (price_range / 2.0) * 1.5;

        // Apply zoom
        if scroll_x.abs() > 0.001 {
            let zoom_factor = 1.0 - scroll_x * 0.1;
            let old_scale = self.camera.scale[0];
            self.camera.scale[0] = (old_scale * zoom_factor).clamp(5.0, max_x_zoom);
            // Adjust position to keep world point under cursor
            let new_world_x = self.camera.position[0] + cursor_ndc[0] * self.camera.scale[0] * aspect;
            self.camera.position[0] += world_x - new_world_x;
        }

        if scroll_y.abs() > 0.001 {
            let zoom_factor = 1.0 + scroll_y * 0.1;
            let old_scale = self.camera.scale[1];
            self.camera.scale[1] = (old_scale * zoom_factor).clamp(1.0, max_y_zoom);
            // Adjust position to keep world point under cursor
            let new_world_y = self.camera.position[1] + cursor_ndc[1] * self.camera.scale[1];
            self.camera.position[1] += world_y - new_world_y;
        }

        self.update_camera();
        self.update_visible_range();
        self.window.request_redraw();
    }

    fn update_camera(&mut self) {
        let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = chart_width / chart_height;

        // Update main candle camera
        self.camera_uniform.update_view_proj(&self.camera, aspect);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update volume camera (synced X position with main camera)
        self.volume_camera.position[0] = self.camera.position[0];
        self.volume_camera.scale[0] = self.camera.scale[0];
        let volume_aspect = chart_width / (self.config.height as f32 * VOLUME_HEIGHT_RATIO);
        self.volume_camera_uniform.update_view_proj(&self.volume_camera, volume_aspect);
        self.queue.write_buffer(
            &self.volume_camera_buffer,
            0,
            bytemuck::cast_slice(&[self.volume_camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate layout dimensions (use floor to prevent viewport exceeding render target)
        let total_width = self.config.width as f32;
        let total_height = self.config.height as f32;
        let chart_width = (total_width - STATS_PANEL_WIDTH).floor().max(1.0);
        let chart_height = (total_height * (1.0 - VOLUME_HEIGHT_RATIO)).floor().max(1.0);
        // Calculate volume_height to exactly fill remaining space (avoids float precision issues)
        let volume_height = (total_height - chart_height).max(1.0);

        // Build egui UI
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            self.build_stats_panel(ctx);
        });

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update egui buffers
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_descriptor);

        // First render pass: Clear and render charts
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chart Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let tf = &self.timeframes[self.current_timeframe];

            // Set viewport for chart area
            render_pass.set_viewport(0.0, 0.0, chart_width, chart_height, 0.0, 1.0);

            // Render price guidelines first (behind candles)
            if self.guideline_count > 0 {
                render_pass.set_pipeline(&self.guideline_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.guideline_bind_group, &[]);
                render_pass.draw(0..6, 0..self.guideline_count); // 6 vertices per guideline
            }

            // Render candle chart (top portion, left of stats panel)
            render_pass.set_pipeline(&self.candle_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &tf.candle_bind_group, &[]);
            render_pass.draw(0..VERTICES_PER_CANDLE, 0..self.visible_count);

            // Render volume bars (bottom portion, left of stats panel)
            render_pass.set_viewport(0.0, chart_height, chart_width, volume_height, 0.0, 1.0);
            render_pass.set_pipeline(&self.volume_pipeline);
            render_pass.set_bind_group(0, &self.volume_camera_bind_group, &[]);
            render_pass.set_bind_group(1, &tf.volume_bind_group, &[]);
            render_pass.draw(0..6, 0..self.visible_count); // 6 vertices per volume bar
        }

        // Submit chart rendering
        self.queue.submit(std::iter::once(encoder.finish()));

        // Second command encoder for egui (separate submission to avoid lifetime issues)
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

            // egui-wgpu 0.31 requires RenderPass<'static>, use forget_lifetime()
            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        // Free egui textures
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
                tf_label,
                self.fps,
                candle_count,
                self.visible_count
            ));
        }

        Ok(())
    }

    fn build_stats_panel(&self, ctx: &egui::Context) {
        let tf = &self.timeframes[self.current_timeframe];
        let tf_labels = ["1m", "15m", "1h", "1w", "1M"];

        // Get the candle under cursor or the last visible candle
        let cursor_candle_idx = if let Some(pos) = self.last_mouse_pos {
            let chart_width = self.config.width as f32 - STATS_PANEL_WIDTH;
            if pos[0] < chart_width {
                let aspect = chart_width / (self.config.height as f32 * (1.0 - VOLUME_HEIGHT_RATIO));
                let (x_min, _) = self.camera.visible_x_range(aspect);
                let normalized_x = pos[0] / chart_width;
                let world_x = x_min + normalized_x * (self.camera.scale[0] * aspect * 2.0);
                let idx = (world_x / CANDLE_SPACING).round() as usize;
                idx.min(tf.candles.len().saturating_sub(1))
            } else {
                tf.candles.len().saturating_sub(1)
            }
        } else {
            tf.candles.len().saturating_sub(1)
        };

        let candle = tf.candles.get(cursor_candle_idx);

        egui::SidePanel::right("stats_panel")
            .exact_width(STATS_PANEL_WIDTH)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Stats");
                ui.separator();

                // Timeframe buttons
                ui.horizontal(|ui| {
                    for (i, label) in tf_labels.iter().enumerate() {
                        let selected = i == self.current_timeframe;
                        if ui.selectable_label(selected, *label).clicked() {
                            // Note: Can't mutate self here, but buttons show selection
                        }
                    }
                });
                ui.separator();

                ui.label(format!("FPS: {:.1}", self.fps));
                ui.label(format!("Candles: {}", tf.candles.len()));
                ui.label(format!("Visible: {}", self.visible_count));
                ui.separator();

                if let Some(c) = candle {
                    ui.heading("OHLCV");
                    ui.label(format!("Open:  ${:.2}", c.open));
                    ui.label(format!("High:  ${:.2}", c.high));
                    ui.label(format!("Low:   ${:.2}", c.low));
                    ui.label(format!("Close: ${:.2}", c.close));
                    ui.separator();

                    let change = c.close - c.open;
                    let change_pct = (change / c.open) * 100.0;
                    let color = if change >= 0.0 { Color32::GREEN } else { Color32::RED };
                    ui.colored_label(color, format!("Change: {:.2} ({:.2}%)", change, change_pct));
                    ui.separator();

                    ui.label(format!("Volume: {:.4}", c.volume));
                }
            });
    }

    fn update(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        // Let egui handle the event first
        let egui_response = state.egui_state.on_window_event(&state.window, &event);
        if egui_response.consumed {
            state.window.request_redraw();
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let _ = state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                state.handle_mouse_input(btn_state, button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_wheel(delta);
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

pub fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
    }
}
