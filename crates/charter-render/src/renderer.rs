//! Chart renderer coordination.

use wgpu::util::DeviceExt;

use charter_core::Candle;

use crate::camera::{Camera, CameraUniform};
use crate::gpu_types::{
    aggregate_candles_lod, aggregate_volume_lod, CandleGpu, CurrentPriceParams, GuidelineGpu,
    GuidelineParams, LodConfig, PackedCandleGpu, PriceNormalization, RenderParams, VolumeGpu,
    VolumeRenderParams, MAX_GUIDELINES,
};
use crate::pipeline::{
    CandlePipeline, CurrentPricePipeline, GuidelinePipeline, IndicatorPipeline, TaPipeline,
    VolumePipeline,
};
use crate::{BASE_CANDLE_WIDTH, CANDLE_SPACING, MIN_CANDLE_PIXELS, STATS_PANEL_WIDTH, VOLUME_HEIGHT_RATIO};

/// LOD data for a single level.
pub struct LodData {
    pub candle_buffer: wgpu::Buffer,
    pub candle_bind_group: wgpu::BindGroup,
    pub volume_buffer: wgpu::Buffer,
    pub volume_bind_group: wgpu::BindGroup,
    pub count: u32,
    pub max_volume: f32,
    pub factor: usize,
}

/// Pre-computed timeframe data with GPU buffers for candles, volume, and TA.
pub struct TimeframeData {
    pub candles: Vec<Candle>,
    /// Full resolution candle buffer (LOD 0) - uses packed u16 format.
    pub candle_buffer: wgpu::Buffer,
    pub candle_bind_group: wgpu::BindGroup,
    pub volume_buffer: wgpu::Buffer,
    pub volume_bind_group: wgpu::BindGroup,
    pub count: u32,
    pub max_volume: f32,
    /// Price normalization parameters for packed candles.
    pub price_normalization: PriceNormalization,
    /// LOD levels for zoomed-out rendering.
    pub lod_levels: Vec<LodData>,
    /// LOD configuration.
    pub lod_config: LodConfig,
    // TA data
    pub ta_range_buffer: wgpu::Buffer,
    pub ta_level_buffer: wgpu::Buffer,
    pub ta_trend_buffer: wgpu::Buffer,
    pub ta_params_buffer: wgpu::Buffer,
    pub ta_bind_group: wgpu::BindGroup,
    pub ta_range_count: u32,
    pub ta_level_count: u32,
    pub ta_trend_count: u32,
}

impl TimeframeData {
    /// Get the LOD data for a given candles-per-pixel density.
    /// Returns (candle_bind_group, volume_bind_group, count, factor, max_volume).
    pub fn lod_for_density(&self, candles_per_pixel: f32) -> (&wgpu::BindGroup, &wgpu::BindGroup, u32, usize, f32) {
        let desired_factor = self.lod_config.factor_for_density(candles_per_pixel);

        // If full resolution is desired, return it immediately
        if desired_factor == 1 {
            return (&self.candle_bind_group, &self.volume_bind_group, self.count, 1, self.max_volume);
        }

        // Try to find exact match first
        if let Some(lod) = self.lod_levels.iter().find(|l| l.factor == desired_factor) {
            return (&lod.candle_bind_group, &lod.volume_bind_group, lod.count, lod.factor, lod.max_volume);
        }

        // Find the closest available LOD level (prefer slightly higher detail)
        let closest_lod = self.lod_levels.iter()
            .min_by_key(|lod| {
                let diff = (lod.factor as i32 - desired_factor as i32).abs();
                // Prefer lower factors (higher detail) when equally close
                if lod.factor < desired_factor {
                    diff * 10 // Penalize less detailed options less
                } else {
                    diff * 15 // Penalize more detailed options more
                }
            });

        if let Some(lod) = closest_lod {
            (&lod.candle_bind_group, &lod.volume_bind_group, lod.count, lod.factor, lod.max_volume)
        } else {
            // Fallback to full resolution if no LOD levels exist
            (&self.candle_bind_group, &self.volume_bind_group, self.count, 1, self.max_volume)
        }
    }
}

/// Coordinates all GPU rendering pipelines.
pub struct ChartRenderer {
    pub candle_pipeline: CandlePipeline,
    pub volume_pipeline: VolumePipeline,
    pub guideline_pipeline: GuidelinePipeline,
    pub indicator_pipeline: IndicatorPipeline,
    pub ta_pipeline: TaPipeline,

    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    pub volume_camera: Camera,
    pub volume_camera_uniform: CameraUniform,
    pub volume_camera_buffer: wgpu::Buffer,
    pub volume_camera_bind_group: wgpu::BindGroup,

    pub render_params_buffer: wgpu::Buffer,
    pub volume_params_buffer: wgpu::Buffer,

    pub guideline_buffer: wgpu::Buffer,
    pub guideline_params_buffer: wgpu::Buffer,
    pub guideline_bind_group: wgpu::BindGroup,
    pub guideline_count: u32,
    pub guideline_values: Vec<f32>, // Y-values for price labels

    // Current price line (dotted line at current price when ws connected)
    pub current_price_pipeline: CurrentPricePipeline,
    pub current_price_params_buffer: wgpu::Buffer,
    pub current_price_bind_group: wgpu::BindGroup,
    pub current_price: Option<f32>,
    /// Open price of the current candle (for determining bullish/bearish color)
    current_price_open: f32,
    /// X bounds for current price line
    current_price_x_min: f32,
    current_price_x_max: f32,

    pub visible_start: u32,
    pub visible_count: u32,

    /// Current LOD factor (1 = full, 10 = low, 100 = very low).
    pub current_lod_factor: usize,
    /// Candles per pixel for LOD selection.
    pub candles_per_pixel: f32,

    config_width: u32,
    config_height: u32,
}

impl ChartRenderer {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        initial_candles: &[Candle],
    ) -> Self {
        // Create pipelines
        let candle_pipeline = CandlePipeline::new(device, format);
        let volume_pipeline = VolumePipeline::new(device, format, &candle_pipeline.camera_bind_group_layout);
        let guideline_pipeline = GuidelinePipeline::new(device, format, &candle_pipeline.camera_bind_group_layout);
        let indicator_pipeline = IndicatorPipeline::new(device, format, &candle_pipeline.camera_bind_group_layout);
        let ta_pipeline = TaPipeline::new(device, format, &candle_pipeline.camera_bind_group_layout);

        // Camera setup
        let mut camera = Camera::new();
        let (min_price, max_price) = initial_candles
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), c| {
                (min.min(c.low), max.max(c.high))
            });
        let price_center = (min_price + max_price) / 2.0;
        let price_range = (max_price - min_price) * 1.1;

        let x_max = (initial_candles.len() as f32) * CANDLE_SPACING;
        let x_center = x_max / 2.0;

        camera.position = [x_center, price_center];
        camera.scale = [x_max / 2.0, price_range / 2.0];

        let mut camera_uniform = CameraUniform::new();
        let aspect = width as f32 / height as f32;
        camera_uniform.update_view_proj(&camera, aspect);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &candle_pipeline.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Volume camera
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
            layout: &candle_pipeline.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: volume_camera_buffer.as_entire_binding(),
            }],
            label: Some("volume_camera_bind_group"),
        });

        // Render params
        let render_params_buffer = candle_pipeline.create_render_params_buffer(device);
        let volume_params_buffer = volume_pipeline.create_volume_params_buffer(device);

        // Guidelines
        let guideline_buffer = guideline_pipeline.create_guideline_buffer(device);
        let guideline_params_buffer = guideline_pipeline.create_guideline_params_buffer(device, x_max, price_range);
        let guideline_bind_group = guideline_pipeline.create_bind_group(device, &guideline_buffer, &guideline_params_buffer);

        // Current price line
        let current_price_pipeline =
            CurrentPricePipeline::new(device, format, &candle_pipeline.camera_bind_group_layout);
        let current_price_params_buffer = current_price_pipeline.create_params_buffer(device);
        let current_price_bind_group =
            current_price_pipeline.create_bind_group(device, &current_price_params_buffer);

        Self {
            candle_pipeline,
            volume_pipeline,
            guideline_pipeline,
            indicator_pipeline,
            ta_pipeline,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            volume_camera,
            volume_camera_uniform,
            volume_camera_buffer,
            volume_camera_bind_group,
            render_params_buffer,
            volume_params_buffer,
            guideline_buffer,
            guideline_params_buffer,
            guideline_bind_group,
            guideline_count: 0,
            guideline_values: Vec::new(),
            current_price_pipeline,
            current_price_params_buffer,
            current_price_bind_group,
            current_price: None,
            current_price_open: 0.0,
            current_price_x_min: 0.0,
            current_price_x_max: 0.0,
            visible_start: 0,
            visible_count: 0,
            current_lod_factor: 1,
            candles_per_pixel: 1.0,
            config_width: width,
            config_height: height,
        }
    }

    pub fn create_timeframe_data(
        &self,
        device: &wgpu::Device,
        candles: Vec<Candle>,
        label: &str,
    ) -> TimeframeData {
        self.create_timeframe_data_with_config(device, candles, label, LodConfig::default())
    }

    pub fn create_timeframe_data_with_config(
        &self,
        device: &wgpu::Device,
        candles: Vec<Candle>,
        label: &str,
        lod_config: LodConfig,
    ) -> TimeframeData {
        // Compute price normalization for packed candles (50% memory savings)
        let price_normalization = PriceNormalization::from_candles(&candles);

        // Pack candles to u16 format (8 bytes per candle instead of 16)
        let packed_candles = price_normalization.pack_candles(&candles);

        // Ensure we have at least one element for GPU buffers (wgpu doesn't allow zero-sized buffers)
        let candles_for_buffer = if packed_candles.is_empty() {
            vec![PackedCandleGpu { open: 0, high: 0, low: 0, close: 0 }]
        } else {
            packed_candles
        };

        let candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Candle Buffer {}", label)),
            contents: bytemuck::cast_slice(&candles_for_buffer),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let candle_bind_group = self.candle_pipeline.create_bind_group(
            device,
            &candle_buffer,
            &self.render_params_buffer,
        );

        let volume_gpu: Vec<VolumeGpu> = candles.iter().map(VolumeGpu::from_candle).collect();
        let max_volume = candles.iter().map(|c| c.volume).fold(0.0f32, f32::max);

        // Ensure we have at least one element for GPU buffers
        let volume_for_buffer = if volume_gpu.is_empty() {
            vec![VolumeGpu { volume: 0.0, is_bullish: 1, _padding1: 0.0, _padding2: 0.0 }]
        } else {
            volume_gpu.clone()
        };

        let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Volume Buffer {}", label)),
            contents: bytemuck::cast_slice(&volume_for_buffer),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let volume_bind_group = self.volume_pipeline.create_bind_group(
            device,
            &volume_buffer,
            &self.volume_params_buffer,
        );

        // Create LOD levels using packed candles
        let candles_gpu: Vec<CandleGpu> = candles.iter().map(CandleGpu::from).collect();
        let mut lod_levels = Vec::new();

        // Generate LOD levels based on configuration
        for &factor in lod_config.factors() {
            // Only create LOD if we have enough candles (at least 5x the factor for meaningful aggregation)
            if candles_gpu.len() > factor * 5 {
                // Aggregate then pack for LOD
                let lod_candles_gpu = aggregate_candles_lod(&candles_gpu, factor);
                let lod_packed = price_normalization.pack_candles_gpu(&lod_candles_gpu);
                let lod_volume = aggregate_volume_lod(&volume_gpu, &candles_gpu, factor);
                let lod_max_volume = lod_volume.iter().map(|v| v.volume).fold(0.0f32, f32::max);

                let lod_candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Candle Buffer {} LOD {}", label, factor)),
                    contents: bytemuck::cast_slice(&lod_packed),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let lod_candle_bind_group = self.candle_pipeline.create_bind_group(
                    device,
                    &lod_candle_buffer,
                    &self.render_params_buffer,
                );

                let lod_volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Volume Buffer {} LOD {}", label, factor)),
                    contents: bytemuck::cast_slice(&lod_volume),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let lod_volume_bind_group = self.volume_pipeline.create_bind_group(
                    device,
                    &lod_volume_buffer,
                    &self.volume_params_buffer,
                );

                lod_levels.push(LodData {
                    candle_buffer: lod_candle_buffer,
                    candle_bind_group: lod_candle_bind_group,
                    volume_buffer: lod_volume_buffer,
                    volume_bind_group: lod_volume_bind_group,
                    count: lod_packed.len() as u32,
                    max_volume: lod_max_volume,
                    factor,
                });
            }
        }

        // Create TA buffers (initially empty, will be populated later)
        let ta_range_buffer = self.ta_pipeline.create_range_buffer(device);
        let ta_level_buffer = self.ta_pipeline.create_level_buffer(device);
        let ta_trend_buffer = self.ta_pipeline.create_trend_buffer(device);
        let ta_params_buffer = self.ta_pipeline.create_params_buffer(device);
        let ta_bind_group = self.ta_pipeline.create_bind_group(
            device,
            &ta_range_buffer,
            &ta_level_buffer,
            &ta_params_buffer,
            &ta_trend_buffer,
        );

        let count = candles.len() as u32;

        TimeframeData {
            candles,
            candle_buffer,
            candle_bind_group,
            volume_buffer,
            volume_bind_group,
            count,
            max_volume,
            price_normalization,
            lod_levels,
            lod_config,
            ta_range_buffer,
            ta_level_buffer,
            ta_trend_buffer,
            ta_params_buffer,
            ta_bind_group,
            ta_range_count: 0,
            ta_level_count: 0,
            ta_trend_count: 0,
        }
    }

    /// Update only the candle and volume data of a TimeframeData, preserving TA buffers.
    /// This is used for live updates where we don't want to lose computed TA data.
    pub fn update_candle_data(
        &self,
        device: &wgpu::Device,
        existing: &TimeframeData,
        candles: Vec<Candle>,
        label: &str,
    ) -> TimeframeData {
        // Compute price normalization for packed candles
        let price_normalization = PriceNormalization::from_candles(&candles);

        // Pack candles to u16 format
        let packed_candles = price_normalization.pack_candles(&candles);

        // Ensure we have at least one element for GPU buffers
        let candles_for_buffer = if packed_candles.is_empty() {
            vec![PackedCandleGpu { open: 0, high: 0, low: 0, close: 0 }]
        } else {
            packed_candles
        };

        let candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Candle Buffer {}", label)),
            contents: bytemuck::cast_slice(&candles_for_buffer),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let candle_bind_group = self.candle_pipeline.create_bind_group(
            device,
            &candle_buffer,
            &self.render_params_buffer,
        );

        let volume_gpu: Vec<VolumeGpu> = candles.iter().map(VolumeGpu::from_candle).collect();
        let max_volume = candles.iter().map(|c| c.volume).fold(0.0f32, f32::max);

        let volume_for_buffer = if volume_gpu.is_empty() {
            vec![VolumeGpu { volume: 0.0, is_bullish: 1, _padding1: 0.0, _padding2: 0.0 }]
        } else {
            volume_gpu.clone()
        };

        let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Volume Buffer {}", label)),
            contents: bytemuck::cast_slice(&volume_for_buffer),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let volume_bind_group = self.volume_pipeline.create_bind_group(
            device,
            &volume_buffer,
            &self.volume_params_buffer,
        );

        // Create LOD levels
        let candles_gpu: Vec<CandleGpu> = candles.iter().map(CandleGpu::from).collect();
        let mut lod_levels = Vec::new();
        let lod_config = existing.lod_config.clone();

        for &factor in lod_config.factors() {
            if candles_gpu.len() > factor * 5 {
                let lod_candles_gpu = aggregate_candles_lod(&candles_gpu, factor);
                let lod_packed = price_normalization.pack_candles_gpu(&lod_candles_gpu);
                let lod_volume = aggregate_volume_lod(&volume_gpu, &candles_gpu, factor);
                let lod_max_volume = lod_volume.iter().map(|v| v.volume).fold(0.0f32, f32::max);

                let lod_candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Candle Buffer {} LOD {}", label, factor)),
                    contents: bytemuck::cast_slice(&lod_packed),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let lod_candle_bind_group = self.candle_pipeline.create_bind_group(
                    device,
                    &lod_candle_buffer,
                    &self.render_params_buffer,
                );

                let lod_volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Volume Buffer {} LOD {}", label, factor)),
                    contents: bytemuck::cast_slice(&lod_volume),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let lod_volume_bind_group = self.volume_pipeline.create_bind_group(
                    device,
                    &lod_volume_buffer,
                    &self.volume_params_buffer,
                );

                lod_levels.push(LodData {
                    candle_buffer: lod_candle_buffer,
                    candle_bind_group: lod_candle_bind_group,
                    volume_buffer: lod_volume_buffer,
                    volume_bind_group: lod_volume_bind_group,
                    count: lod_packed.len() as u32,
                    max_volume: lod_max_volume,
                    factor,
                });
            }
        }

        // Recreate TA bind group with existing buffers (buffers can't be cloned, need new bind group)
        let ta_range_buffer = self.ta_pipeline.create_range_buffer(device);
        let ta_level_buffer = self.ta_pipeline.create_level_buffer(device);
        let ta_trend_buffer = self.ta_pipeline.create_trend_buffer(device);
        let ta_params_buffer = self.ta_pipeline.create_params_buffer(device);
        let ta_bind_group = self.ta_pipeline.create_bind_group(
            device,
            &ta_range_buffer,
            &ta_level_buffer,
            &ta_params_buffer,
            &ta_trend_buffer,
        );

        let count = candles.len() as u32;

        TimeframeData {
            candles,
            candle_buffer,
            candle_bind_group,
            volume_buffer,
            volume_bind_group,
            count,
            max_volume,
            price_normalization,
            lod_levels,
            lod_config,
            ta_range_buffer,
            ta_level_buffer,
            ta_trend_buffer,
            ta_params_buffer,
            ta_bind_group,
            // Preserve TA counts from existing data
            ta_range_count: existing.ta_range_count,
            ta_level_count: existing.ta_level_count,
            ta_trend_count: existing.ta_trend_count,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config_width = width;
        self.config_height = height;
    }

    pub fn update_camera(&mut self, queue: &wgpu::Queue) {
        let chart_width = self.config_width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = chart_width / chart_height;

        self.camera_uniform.update_view_proj(&self.camera, aspect);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.volume_camera.position[0] = self.camera.position[0];
        self.volume_camera.scale[0] = self.camera.scale[0];
        // Use the same aspect ratio as the chart for X alignment
        // The Y scaling is handled by the volume camera's scale[1] and viewport
        self.volume_camera_uniform.update_view_proj(&self.volume_camera, aspect);
        queue.write_buffer(
            &self.volume_camera_buffer,
            0,
            bytemuck::cast_slice(&[self.volume_camera_uniform]),
        );
    }

    pub fn update_visible_range(&mut self, queue: &wgpu::Queue, timeframe: &TimeframeData) {
        let chart_width = self.config_width as f32 - STATS_PANEL_WIDTH;
        let chart_height = self.config_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = chart_width / chart_height;
        let (x_min, x_max) = self.camera.visible_x_range(aspect);
        let (y_min, y_max) = self.camera.visible_y_range(aspect);

        let visible_width = x_max - x_min;

        // Compute candles per pixel for LOD selection
        let visible_candles_approx = visible_width / CANDLE_SPACING;
        self.candles_per_pixel = visible_candles_approx / chart_width;

        // Select appropriate LOD based on density
        let desired_factor = timeframe.lod_config.factor_for_density(self.candles_per_pixel);

        // Find the best available LOD factor
        self.current_lod_factor = if desired_factor == 1 {
            1
        } else if let Some(lod) = timeframe.lod_levels.iter().find(|l| l.factor == desired_factor) {
            lod.factor
        } else {
            // Find closest available LOD level
            timeframe.lod_levels.iter()
                .min_by_key(|lod| {
                    let diff = (lod.factor as i32 - desired_factor as i32).abs();
                    if lod.factor < desired_factor {
                        diff * 10
                    } else {
                        diff * 15
                    }
                })
                .map(|lod| lod.factor)
                .unwrap_or(1)
        };

        // Compute effective values based on LOD
        let effective_spacing = CANDLE_SPACING * self.current_lod_factor as f32;
        let effective_candle_count = (timeframe.candles.len() + self.current_lod_factor - 1) / self.current_lod_factor;

        let first_idx = ((x_min / effective_spacing).floor() as i32 - 1).max(0) as u32;
        let last_idx = ((x_max / effective_spacing).ceil() as i32 + 1).min(effective_candle_count as i32) as u32;

        self.visible_start = first_idx;
        self.visible_count = if last_idx > first_idx {
            last_idx - first_idx
        } else {
            0
        };

        let world_units_per_pixel = visible_width / chart_width;
        let min_world_width = MIN_CANDLE_PIXELS * world_units_per_pixel;
        let desired_width = (BASE_CANDLE_WIDTH * self.current_lod_factor as f32).max(min_world_width);
        // Ensure candle width never exceeds spacing to prevent overlap
        let candle_width = desired_width.min(effective_spacing * 0.95);

        // Wick width: proportional to candle, clamped between 1-4 pixels
        // Also cap at absolute maximum of 10% of candle width to stay visually thin
        let x_pixel_size = world_units_per_pixel;
        let min_wick_width = 1.0 * x_pixel_size; // Minimum 1 pixel
        let max_wick_width = 4.0 * x_pixel_size; // Maximum 4 pixels
        let proportional_wick = candle_width * 0.1; // 10% of candle width
        let wick_width = proportional_wick.clamp(min_wick_width, max_wick_width);

        // Minimum body height for doji candles - use Y-axis pixel size (price units per pixel)
        // to ensure at least 2 pixels visibility on screen
        let visible_height = y_max - y_min;
        let y_pixel_size = visible_height / chart_height;
        let min_body_height = 2.0 * y_pixel_size;

        // Pass view bounds and price normalization to shader
        let render_params = RenderParams {
            first_visible: self.visible_start,
            candle_width,
            candle_spacing: effective_spacing,
            wick_width,
            x_min,
            x_max,
            y_min,
            y_max,
            price_min: timeframe.price_normalization.price_min,
            price_range: timeframe.price_normalization.price_range,
            min_body_height,
            _padding: 0.0,
        };

        queue.write_buffer(
            &self.render_params_buffer,
            0,
            bytemuck::cast_slice(&[render_params]),
        );

        // Get the appropriate max_volume for the LOD level
        let visible_max_volume = if self.current_lod_factor == 1 {
            timeframe
                .candles
                .iter()
                .skip(self.visible_start as usize)
                .take(self.visible_count as usize)
                .map(|c| c.volume)
                .fold(0.0f32, f32::max)
                .max(1.0)
        } else if let Some(lod) = timeframe.lod_levels.iter().find(|l| l.factor == self.current_lod_factor) {
            lod.max_volume
        } else {
            timeframe.max_volume
        };

        let volume_params = VolumeRenderParams {
            first_visible: self.visible_start,
            bar_width: candle_width,
            bar_spacing: effective_spacing,
            max_volume: visible_max_volume,
        };
        queue.write_buffer(
            &self.volume_params_buffer,
            0,
            bytemuck::cast_slice(&[volume_params]),
        );

        self.update_guidelines(queue, x_min, x_max);
    }

    fn update_guidelines(&mut self, queue: &wgpu::Queue, x_min: f32, x_max: f32) {
        let chart_height = self.config_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = (self.config_width as f32 - STATS_PANEL_WIDTH) / chart_height;
        let (y_min, y_max) = self.camera.visible_y_range(aspect);

        let price_range = y_max - y_min;

        let target_lines = 8.0;
        let raw_step = price_range / target_lines;

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

        let first_line = (y_min / nice_step).ceil() * nice_step;
        let mut guidelines = Vec::with_capacity(MAX_GUIDELINES);

        let mut y = first_line;
        while y < y_max && guidelines.len() < MAX_GUIDELINES {
            guidelines.push(GuidelineGpu {
                y_value: y,
                r: 0.3,
                g: 0.3,
                b: 0.35,
            });
            y += nice_step;
        }

        let world_units_per_pixel = price_range / chart_height;
        let line_thickness = (1.5 * world_units_per_pixel).max(price_range * 0.0005);

        self.guideline_count = guidelines.len() as u32;
        self.guideline_values = guidelines.iter().map(|g| g.y_value).collect();

        while guidelines.len() < MAX_GUIDELINES {
            guidelines.push(GuidelineGpu {
                y_value: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
            });
        }

        queue.write_buffer(&self.guideline_buffer, 0, bytemuck::cast_slice(&guidelines));

        let guideline_params = GuidelineParams {
            x_min,
            x_max,
            line_thickness,
            count: self.guideline_count,
        };
        queue.write_buffer(
            &self.guideline_params_buffer,
            0,
            bytemuck::cast_slice(&[guideline_params]),
        );
    }

    pub fn fit_view(&mut self, queue: &wgpu::Queue, candles: &[Candle]) {
        if candles.is_empty() {
            return;
        }

        let (min_price, max_price) = candles
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), c| {
                (min.min(c.low), max.max(c.high))
            });
        let price_center = (min_price + max_price) / 2.0;
        let price_range = (max_price - min_price) * 1.1;

        let x_max = (candles.len() as f32) * CANDLE_SPACING;
        let x_center = x_max / 2.0;

        let aspect = self.config_width as f32 / self.config_height as f32;
        let x_scale = x_max / 2.0 / aspect;

        self.camera.position = [x_center, price_center];
        self.camera.scale = [x_scale, price_range / 2.0];

        self.update_camera(queue);
    }

    pub fn config_width(&self) -> u32 {
        self.config_width
    }

    pub fn config_height(&self) -> u32 {
        self.config_height
    }

    /// Update the current price line. Set to None to hide it.
    /// `open_price` is the open price of the current candle, used to determine color (bullish/bearish).
    pub fn set_current_price(&mut self, queue: &wgpu::Queue, price: Option<f32>, open_price: f32, x_min: f32, x_max: f32) {
        self.current_price = price;
        self.current_price_open = open_price;
        self.current_price_x_min = x_min;
        self.current_price_x_max = x_max;

        // Determine color based on candle direction (bullish = green, bearish = red)
        let (r, g, b) = if let Some(close) = price {
            if close >= open_price {
                (0.0, 0.8, 0.4)  // Green for bullish (close >= open)
            } else {
                (0.8, 0.2, 0.2)  // Red for bearish (close < open)
            }
        } else {
            (1.0, 0.843, 0.0)  // Default gold/yellow when no price
        };

        let chart_height = self.config_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let aspect = (self.config_width as f32 - STATS_PANEL_WIDTH) / chart_height;
        let (y_min, y_max) = self.camera.visible_y_range(aspect);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        // 1.0 for exactly 1 pixel thickness
        let line_thickness = world_units_per_pixel.max(price_range * 0.0005);

        let params = CurrentPriceParams {
            y_value: price.unwrap_or(0.0),
            x_min,
            x_max,
            line_thickness,
            r,
            g,
            b,
            visible: if price.is_some() { 1 } else { 0 },
            dot_spacing: 10.0,
            screen_width: self.config_width as f32,
            _padding1: 0.0,
            _padding2: 0.0,
        };

        queue.write_buffer(
            &self.current_price_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    /// Update the current price line thickness and x range based on current zoom level.
    /// Call this after camera changes (zooming, panning) to keep line at constant 1px and full width.
    pub fn update_current_price_line(&mut self, queue: &wgpu::Queue) {
        // Only update if we have a visible price line
        let Some(price) = self.current_price else {
            return;
        };

        // Determine color based on candle direction (bullish = green, bearish = red)
        let (r, g, b) = if price >= self.current_price_open {
            (0.0, 0.8, 0.4)  // Green for bullish (close >= open)
        } else {
            (0.8, 0.2, 0.2)  // Red for bearish (close < open)
        };

        let chart_height = self.config_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO);
        let chart_width = self.config_width as f32 - STATS_PANEL_WIDTH;
        let aspect = chart_width / chart_height;
        let (y_min, y_max) = self.camera.visible_y_range(aspect);
        let (x_min, x_max) = self.camera.visible_x_range(aspect);
        let price_range = y_max - y_min;
        let world_units_per_pixel = price_range / chart_height;
        // 1.0 for exactly 1 pixel thickness
        let line_thickness = world_units_per_pixel.max(price_range * 0.0005);

        // Update stored x range for full canvas width
        self.current_price_x_min = x_min;
        self.current_price_x_max = x_max;

        let params = CurrentPriceParams {
            y_value: price,
            x_min,
            x_max,
            line_thickness,
            r,
            g,
            b,
            visible: 1,
            dot_spacing: 10.0,
            screen_width: self.config_width as f32,
            _padding1: 0.0,
            _padding2: 0.0,
        };

        queue.write_buffer(
            &self.current_price_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    /// Render the current price line.
    pub fn render_current_price<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        if self.current_price.is_some() {
            self.current_price_pipeline.render_line(
                render_pass,
                &self.camera_bind_group,
                &self.current_price_bind_group,
            );
        }
    }
}
