//! Technical Analysis rendering pipeline.

use crate::gpu_types::{LevelGpu, RangeGpu, TaRenderParams, TrendGpu, MAX_TA_LEVELS, MAX_TA_RANGES, MAX_TA_TRENDS};
use crate::pipeline::shared::SharedLayouts;

/// Pipeline for rendering technical analysis elements (ranges, levels, and trends).
pub struct TaPipeline {
    pub range_pipeline: wgpu::RenderPipeline,
    pub level_pipeline: wgpu::RenderPipeline,
    pub trend_pipeline: wgpu::RenderPipeline,
    pub ta_bind_group_layout: wgpu::BindGroupLayout,
}

impl TaPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ta.wgsl").into()),
        });

        let ta_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Ranges storage buffer
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    // Levels storage buffer
                    SharedLayouts::storage_entry(1, wgpu::ShaderStages::VERTEX),
                    // TA params uniform
                    SharedLayouts::uniform_entry(2, wgpu::ShaderStages::VERTEX),
                    // Trends storage buffer
                    SharedLayouts::storage_entry(3, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("ta_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TA Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &ta_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Range rendering pipeline
        let range_pipeline = SharedLayouts::create_render_pipeline(
            device,
            "TA Range Pipeline",
            &pipeline_layout,
            &shader,
            "vs_range",
            "fs_range",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        // Level rendering pipeline
        let level_pipeline = SharedLayouts::create_render_pipeline(
            device,
            "TA Level Pipeline",
            &pipeline_layout,
            &shader,
            "vs_level",
            "fs_level",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        // Trend rendering pipeline
        let trend_pipeline = SharedLayouts::create_render_pipeline(
            device,
            "TA Trend Pipeline",
            &pipeline_layout,
            &shader,
            "vs_trend",
            "fs_trend",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        Self {
            range_pipeline,
            level_pipeline,
            trend_pipeline,
            ta_bind_group_layout,
        }
    }

    pub fn create_range_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial_ranges = vec![
            RangeGpu {
                x_start: 0.0,
                x_end: 0.0,
                y_pos: 0.0,
                is_bullish: 0,
            };
            MAX_TA_RANGES
        ];
        SharedLayouts::create_storage_buffer(device, "TA Range Buffer", &initial_ranges)
    }

    pub fn create_level_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial_levels = vec![
            LevelGpu {
                y_value: 0.0,
                x_start: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
                level_type: 0,
                hit_count: 0,
            };
            MAX_TA_LEVELS
        ];
        SharedLayouts::create_storage_buffer(device, "TA Level Buffer", &initial_levels)
    }

    pub fn create_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = TaRenderParams {
            first_visible: 0,
            candle_spacing: 1.2,
            range_thickness: 2.0,
            level_thickness: 1.0,
            x_max: 1000.0,
            range_count: 0,
            level_count: 0,
            trend_count: 0,
        };
        SharedLayouts::create_uniform_buffer(device, "TA Params Buffer", &params)
    }

    pub fn create_trend_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial_trends = vec![
            TrendGpu {
                x_start: 0.0,
                y_start: 0.0,
                x_end: 0.0,
                y_end: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            };
            MAX_TA_TRENDS
        ];
        SharedLayouts::create_storage_buffer(device, "TA Trend Buffer", &initial_trends)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        range_buffer: &wgpu::Buffer,
        level_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        trend_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "TA Bind Group",
            &self.ta_bind_group_layout,
            &[range_buffer, level_buffer, params_buffer, trend_buffer],
        )
    }

    /// Renders TA ranges.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `range_count` - Number of ranges to render
    pub fn render_ranges<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        range_count: u32,
    ) {
        render_pass.set_pipeline(&self.range_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..range_count);
    }

    /// Renders TA levels (support/resistance).
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `level_count` - Number of levels to render
    pub fn render_levels<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        level_count: u32,
    ) {
        render_pass.set_pipeline(&self.level_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..level_count);
    }

    /// Renders TA trends (trendlines).
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `trend_count` - Number of trends to render
    pub fn render_trends<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        trend_count: u32,
    ) {
        render_pass.set_pipeline(&self.trend_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..trend_count);
    }
}

/// Number of vertices per TA element (2 triangles = 6 vertices).
const VERTICES_PER_TA_ELEMENT: u32 = 6;

