//! Volume profile rendering pipeline.

use crate::gpu_types::{VolumeProfileBucketGpu, VolumeProfileParams, MAX_VOLUME_PROFILE_BUCKETS};
use crate::pipeline::shared::SharedLayouts;

/// Pipeline for rendering volume profile horizontal bars.
pub struct VolumeProfilePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl VolumeProfilePipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shared: &SharedLayouts,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Volume Profile Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/volume_profile.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Buckets storage buffer
                    SharedLayouts::storage_entry(0, wgpu::ShaderStages::VERTEX),
                    // Params uniform
                    SharedLayouts::uniform_entry(1, wgpu::ShaderStages::VERTEX),
                ],
                label: Some("volume_profile_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Volume Profile Pipeline Layout"),
            bind_group_layouts: &[&shared.camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = SharedLayouts::create_render_pipeline(
            device,
            "Volume Profile Pipeline",
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

    pub fn create_bucket_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            VolumeProfileBucketGpu {
                price: 0.0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                _padding: 0.0,
            };
            MAX_VOLUME_PROFILE_BUCKETS
        ];
        SharedLayouts::create_storage_buffer(device, "Volume Profile Bucket Buffer", &initial)
    }

    pub fn create_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = VolumeProfileParams {
            bucket_count: 0,
            max_volume: 1.0,
            profile_width: 50.0,
            y_min: 0.0,
            y_max: 100.0,
            bucket_height: 1.0,
            x_right: 0.0,
            visible: 0,
        };
        SharedLayouts::create_uniform_buffer(device, "Volume Profile Params Buffer", &params)
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        bucket_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        SharedLayouts::create_bind_group(
            device,
            "Volume Profile Bind Group",
            &self.bind_group_layout,
            &[bucket_buffer, params_buffer],
        )
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        bucket_count: u32,
    ) {
        if bucket_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(0..6, 0..bucket_count);
    }
}
