//! Volume bar rendering pipeline.

use crate::gpu_types::{VolumeGpu, VolumeRenderParams};
use crate::pipeline::shared::SharedLayouts;
use crate::pipeline::traits::{InstancedPipeline, Pipeline};
use crate::{BASE_CANDLE_WIDTH, CANDLE_SPACING};

/// Pipeline for rendering volume bars.
pub struct VolumePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub volume_bind_group_layout: wgpu::BindGroupLayout,
}

impl VolumePipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume.wgsl").into()),
        });

        let volume_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    SharedLayouts::uniform_entry(1, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("volume_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &volume_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Volume Pipeline",
            &pipeline_layout,
            &shader,
            "vs_main",
            "fs_main",
            format,
            wgpu::BlendState::REPLACE,
        );

        Self {
            pipeline,
            volume_bind_group_layout,
        }
    }

    pub fn create_volume_buffer(&self, device: &wgpu::Device, volumes: &[VolumeGpu]) -> wgpu::Buffer {
        SharedLayouts::create_static_storage_buffer(device, "Volume Buffer", volumes)
    }

    pub fn create_volume_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = VolumeRenderParams {
            first_visible: 0,
            bar_width: BASE_CANDLE_WIDTH,
            bar_spacing: CANDLE_SPACING,
            max_volume: 1.0,
        };
        SharedLayouts::create_uniform_buffer(device, "Volume Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        volume_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Volume Bind Group",
            &self.volume_bind_group_layout,
            &[volume_buffer, params_buffer],
        )
    }
}

/// Number of vertices per volume bar (2 triangles = 6 vertices).
const VERTICES_PER_VOLUME_BAR: u32 = 6;

impl Pipeline for VolumePipeline {
    type BindGroupData = wgpu::BindGroup;

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl InstancedPipeline for VolumePipeline {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_VOLUME_BAR;
}
