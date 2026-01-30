//! Shared GPU resource layouts and factory functions used by all pipelines.
//!
//! This module eliminates boilerplate by providing reusable helpers for
//! bind group layout entries, render pipeline creation, buffer creation,
//! and bind group creation.

use wgpu::util::DeviceExt;

use crate::camera::CameraUniform;

/// Shared GPU resource layouts used by all rendering pipelines.
///
/// Owns the camera bind group layout and provides factory functions
/// for common wgpu resource creation patterns.
pub struct SharedLayouts {
    /// Camera uniform bind group layout (slot 0 in all pipelines).
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
}

impl SharedLayouts {
    /// Create shared layouts including the camera bind group layout.
    pub fn new(device: &wgpu::Device) -> Self {
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[Self::uniform_entry(0, wgpu::ShaderStages::VERTEX)],
                label: Some("camera_bind_group_layout"),
            });
        Self {
            camera_bind_group_layout,
        }
    }

    // =========================================================================
    // Bind group layout entry factories
    // =========================================================================

    /// Read-only storage buffer bind group layout entry.
    pub fn storage_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Uniform buffer bind group layout entry.
    pub fn uniform_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    // =========================================================================
    // Render pipeline factory
    // =========================================================================

    /// Create a render pipeline with the standard configuration shared by all pipelines.
    ///
    /// All pipelines use: TriangleList topology, no depth/stencil, 1x multisample,
    /// no vertex buffers, and vary only in label, shader entry points, blend state,
    /// and format.
    pub fn create_render_pipeline(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        format: wgpu::TextureFormat,
        blend: wgpu::BlendState,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vs_entry),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
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
        })
    }

    // =========================================================================
    // Buffer factories
    // =========================================================================

    /// Create a uniform buffer with COPY_DST usage (for per-frame updates).
    pub fn create_uniform_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &T,
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&[*data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a storage buffer with COPY_DST usage (for data updated per-frame).
    pub fn create_storage_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a read-only storage buffer (for static data like candle/volume buffers).
    pub fn create_static_storage_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    // =========================================================================
    // Bind group factory
    // =========================================================================

    /// Create a bind group from a layout and a slice of buffers.
    ///
    /// Buffers are bound at sequential indices 0, 1, 2, ... in the order provided.
    pub fn create_bind_group(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: Some(label),
        })
    }

    // =========================================================================
    // Camera helpers
    // =========================================================================

    /// Create a camera buffer + bind group pair.
    pub fn create_camera_resources(
        &self,
        device: &wgpu::Device,
        label: &str,
        uniform: &CameraUniform,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = Self::create_uniform_buffer(device, &format!("{} Buffer", label), uniform);
        let bind_group = Self::create_bind_group(
            device,
            &format!("{}_bind_group", label.to_lowercase().replace(' ', "_")),
            &self.camera_bind_group_layout,
            &[&buffer],
        );
        (buffer, bind_group)
    }
}
