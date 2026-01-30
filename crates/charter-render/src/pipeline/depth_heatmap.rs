//! Depth heatmap rendering pipeline.

use crate::gpu_types::{DepthHeatmapCellGpu, DepthHeatmapParams, MAX_DEPTH_LEVELS};
use crate::pipeline::shared::SharedLayouts;

/// Pipeline for rendering depth heatmap as a grid of instanced quads.
pub struct DepthHeatmapPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl DepthHeatmapPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Depth Heatmap Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/depth_heatmap.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Cells storage buffer
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    // Params uniform
                    SharedLayouts::uniform_entry(1, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("depth_heatmap_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Heatmap Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Depth Heatmap Pipeline",
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

    pub fn create_cell_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            DepthHeatmapCellGpu {
                price: 0.0,
                bar_height: 0.0,
                bid_quantity: 0.0,
                ask_quantity: 0.0,
            };
            MAX_DEPTH_LEVELS
        ];
        SharedLayouts::create_storage_buffer(device, "Depth Heatmap Cell Buffer", &initial)
    }

    pub fn create_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = DepthHeatmapParams {
            level_count: 0,
            _pad0: 0,
            max_quantity: 1.0,
            half_width: 1.0,
            _pad1: 0.0,
            _pad2: 0.0,
            x_center: 0.0,
            visible: 0,
        };
        SharedLayouts::create_uniform_buffer(device, "Depth Heatmap Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        cell_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Depth Heatmap Bind Group",
            &self.bind_group_layout,
            &[cell_buffer, params_buffer],
        )
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        cell_count: u32,
    ) {
        if cell_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(0..12, 0..cell_count);
    }
}
