//! Chart renderer coordination.

use wgpu::util::DeviceExt;

use charter_core::Candle;

use crate::camera::{Camera, CameraUniform};
use crate::gpu_types::{
    CandleGpu, GuidelineGpu, GuidelineParams, RenderParams, VolumeGpu, VolumeRenderParams,
    MAX_GUIDELINES,
};
use crate::pipeline::{CandlePipeline, GuidelinePipeline, IndicatorPipeline, TaPipeline, VolumePipeline};
use crate::{BASE_CANDLE_WIDTH, CANDLE_SPACING, MIN_CANDLE_PIXELS, STATS_PANEL_WIDTH, VOLUME_HEIGHT_RATIO};

/// Pre-computed timeframe data with GPU buffers for candles, volume, and TA.
pub struct TimeframeData {
    pub candles: Vec<Candle>,
    pub candle_buffer: wgpu::Buffer,
    pub candle_bind_group: wgpu::BindGroup,
    pub volume_buffer: wgpu::Buffer,
    pub volume_bind_group: wgpu::BindGroup,
    pub count: u32,
    pub max_volume: f32,
    // TA data
    pub ta_range_buffer: wgpu::Buffer,
    pub ta_level_buffer: wgpu::Buffer,
    pub ta_params_buffer: wgpu::Buffer,
    pub ta_bind_group: wgpu::BindGroup,
    pub ta_range_count: u32,
    pub ta_level_count: u32,
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

    pub visible_start: u32,
    pub visible_count: u32,

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
            visible_start: 0,
            visible_count: 0,
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
        let candles_gpu: Vec<CandleGpu> = candles.iter().map(CandleGpu::from).collect();
        let candle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Candle Buffer {}", label)),
            contents: bytemuck::cast_slice(&candles_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let candle_bind_group = self.candle_pipeline.create_bind_group(
            device,
            &candle_buffer,
            &self.render_params_buffer,
        );

        let volume_gpu: Vec<VolumeGpu> = candles.iter().map(VolumeGpu::from_candle).collect();
        let max_volume = candles.iter().map(|c| c.volume).fold(0.0f32, f32::max);
        let volume_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Volume Buffer {}", label)),
            contents: bytemuck::cast_slice(&volume_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let volume_bind_group = self.volume_pipeline.create_bind_group(
            device,
            &volume_buffer,
            &self.volume_params_buffer,
        );

        // Create TA buffers (initially empty, will be populated later)
        let ta_range_buffer = self.ta_pipeline.create_range_buffer(device);
        let ta_level_buffer = self.ta_pipeline.create_level_buffer(device);
        let ta_params_buffer = self.ta_pipeline.create_params_buffer(device);
        let ta_bind_group = self.ta_pipeline.create_bind_group(
            device,
            &ta_range_buffer,
            &ta_level_buffer,
            &ta_params_buffer,
        );

        TimeframeData {
            candles,
            candle_buffer,
            candle_bind_group,
            volume_buffer,
            volume_bind_group,
            count: candles_gpu.len() as u32,
            max_volume,
            ta_range_buffer,
            ta_level_buffer,
            ta_params_buffer,
            ta_bind_group,
            ta_range_count: 0,
            ta_level_count: 0,
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
        let volume_aspect = chart_width / (self.config_height as f32 * VOLUME_HEIGHT_RATIO);
        self.volume_camera_uniform.update_view_proj(&self.volume_camera, volume_aspect);
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

        let candle_count = timeframe.candles.len();

        let first_idx = ((x_min / CANDLE_SPACING).floor() as i32 - 1).max(0) as u32;
        let last_idx = ((x_max / CANDLE_SPACING).ceil() as i32 + 1).min(candle_count as i32) as u32;

        self.visible_start = first_idx;
        self.visible_count = if last_idx > first_idx {
            last_idx - first_idx
        } else {
            0
        };

        let visible_width = x_max - x_min;
        let world_units_per_pixel = visible_width / self.config_width as f32;
        let min_world_width = MIN_CANDLE_PIXELS * world_units_per_pixel;
        let candle_width = BASE_CANDLE_WIDTH.max(min_world_width);

        let base_wick_width = CANDLE_SPACING * 0.08;
        let min_wick_width = 1.0 * world_units_per_pixel;
        let wick_width = base_wick_width.max(min_wick_width);

        let render_params = RenderParams {
            first_visible: self.visible_start,
            candle_width,
            candle_spacing: CANDLE_SPACING,
            wick_width,
        };

        queue.write_buffer(
            &self.render_params_buffer,
            0,
            bytemuck::cast_slice(&[render_params]),
        );

        let visible_max_volume = timeframe
            .candles
            .iter()
            .skip(self.visible_start as usize)
            .take(self.visible_count as usize)
            .map(|c| c.volume)
            .fold(0.0f32, f32::max)
            .max(1.0);

        let volume_params = VolumeRenderParams {
            first_visible: self.visible_start,
            bar_width: candle_width,
            bar_spacing: CANDLE_SPACING,
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
}
