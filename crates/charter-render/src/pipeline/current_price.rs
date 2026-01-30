//! Current price line rendering pipeline.
//!
//! Renders a horizontal dotted line at the current price level when
//! the websocket connection is active.

use std::ops::Range;

use crate::gpu_types::CurrentPriceParams;
use crate::pipeline::shared::SharedLayouts;
use crate::pipeline::traits::Pipeline;

/// Pipeline for rendering the current price line.
pub struct CurrentPricePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CurrentPricePipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Current Price Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/current_price.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[SharedLayouts::uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            )],
            label: Some("current_price_bind_group_layout"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Current Price Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Current Price Pipeline",
            &pipeline_layout,
            &shader,
            "vs_main",
            "fs_main",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn create_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = CurrentPriceParams::default();
        SharedLayouts::create_uniform_buffer(device, "Current Price Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Current Price Bind Group",
            &self.bind_group_layout,
            &[params_buffer],
        )
    }
}

/// Number of vertices for the current price line (2 triangles = 6 vertices).
const VERTICES_PER_LINE: u32 = 6;

impl Pipeline for CurrentPricePipeline {
    type BindGroupData = wgpu::BindGroup;

    fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        vertex_range: Range<u32>,
        _instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(vertex_range, 0..1);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl CurrentPricePipeline {
    /// Render the current price line.
    pub fn render_line<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_LINE, 0..1);
    }
}
