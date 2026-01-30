//! Indicator line rendering pipeline.

use crate::gpu_types::{IndicatorParams, IndicatorPointGpu};
use crate::pipeline::shared::SharedLayouts;
use crate::pipeline::traits::{InstancedPipeline, Pipeline};

/// Pipeline for rendering indicator lines.
pub struct IndicatorPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub indicator_bind_group_layout: wgpu::BindGroupLayout,
}

impl IndicatorPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Indicator Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/indicator.wgsl").into()),
        });

        let indicator_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    SharedLayouts::uniform_entry(1, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("indicator_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Indicator Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &indicator_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Indicator Pipeline",
            &pipeline_layout,
            &shader,
            "vs_main",
            "fs_main",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        Self {
            pipeline,
            indicator_bind_group_layout,
        }
    }

    pub fn create_indicator_buffer(
        &self,
        device: &wgpu::Device,
        points: &[IndicatorPointGpu],
    ) -> wgpu::Buffer {
        SharedLayouts::create_storage_buffer(device, "Indicator Buffer", points)
    }

    pub fn create_indicator_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = IndicatorParams {
            first_visible: 0,
            point_spacing: 1.2,
            line_thickness: 2.0,
            count: 0,
        };
        SharedLayouts::create_uniform_buffer(device, "Indicator Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        indicator_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Indicator Bind Group",
            &self.indicator_bind_group_layout,
            &[indicator_buffer, params_buffer],
        )
    }
}

/// Number of vertices per indicator line segment (2 triangles = 6 vertices).
const VERTICES_PER_INDICATOR_SEGMENT: u32 = 6;

impl Pipeline for IndicatorPipeline {
    type BindGroupData = wgpu::BindGroup;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl InstancedPipeline for IndicatorPipeline {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_INDICATOR_SEGMENT;
}
