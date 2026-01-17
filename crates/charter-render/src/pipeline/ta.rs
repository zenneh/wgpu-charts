//! Technical Analysis rendering pipeline.

use wgpu::util::DeviceExt;

use crate::gpu_types::{LevelGpu, RangeGpu, TaRenderParams, MAX_TA_LEVELS, MAX_TA_RANGES};

/// Pipeline for rendering technical analysis elements (ranges and levels).
pub struct TaPipeline {
    pub range_pipeline: wgpu::RenderPipeline,
    pub level_pipeline: wgpu::RenderPipeline,
    pub ta_bind_group_layout: wgpu::BindGroupLayout,
}

impl TaPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ta.wgsl").into()),
        });

        let ta_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Ranges storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Levels storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // TA params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("ta_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TA Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &ta_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Range rendering pipeline
        let range_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TA Range Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_range"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_range"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        });

        // Level rendering pipeline
        let level_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TA Level Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_level"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_level"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        });

        Self {
            range_pipeline,
            level_pipeline,
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
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TA Range Buffer"),
            contents: bytemuck::cast_slice(&initial_ranges),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
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
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TA Level Buffer"),
            contents: bytemuck::cast_slice(&initial_levels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
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
            _padding: 0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TA Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        range_buffer: &wgpu::Buffer,
        level_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.ta_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: range_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: level_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("TA Bind Group"),
        })
    }
}
