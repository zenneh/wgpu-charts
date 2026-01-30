//! Guideline rendering pipeline.

use crate::gpu_types::{GuidelineGpu, GuidelineParams, MAX_GUIDELINES};
use crate::pipeline::shared::SharedLayouts;
use crate::pipeline::traits::{InstancedPipeline, Pipeline};

/// Pipeline for rendering price guidelines.
pub struct GuidelinePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub guideline_bind_group_layout: wgpu::BindGroupLayout,
}

impl GuidelinePipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Guideline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/guidelines.wgsl").into()),
        });

        let guideline_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    SharedLayouts::uniform_entry(1, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("guideline_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Guideline Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &guideline_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Guideline Pipeline",
            &pipeline_layout,
            &shader,
            "vs_main",
            "fs_main",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        Self {
            pipeline,
            guideline_bind_group_layout,
        }
    }

    pub fn create_guideline_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial_guidelines: Vec<GuidelineGpu> = vec![
            GuidelineGpu {
                y_value: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
            };
            MAX_GUIDELINES
        ];
        SharedLayouts::create_storage_buffer(device, "Guideline Buffer", &initial_guidelines)
    }

    pub fn create_guideline_params_buffer(&self, device: &wgpu::Device, x_max: f32, price_range: f32) -> wgpu::Buffer {
        let params = GuidelineParams {
            x_min: 0.0,
            x_max,
            line_thickness: price_range * 0.001,
            count: 0,
        };
        SharedLayouts::create_uniform_buffer(device, "Guideline Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        guideline_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Guideline Bind Group",
            &self.guideline_bind_group_layout,
            &[guideline_buffer, params_buffer],
        )
    }
}

/// Number of vertices per guideline (2 triangles = 6 vertices).
const VERTICES_PER_GUIDELINE: u32 = 6;

impl Pipeline for GuidelinePipeline {
    type BindGroupData = wgpu::BindGroup;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl InstancedPipeline for GuidelinePipeline {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_GUIDELINE;
}
