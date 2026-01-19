//! Candle rendering pipeline.

use std::ops::Range;

use wgpu::util::DeviceExt;

use crate::gpu_types::{CandleGpu, RenderParams};
use crate::pipeline::traits::{InstancedPipeline, Pipeline};
use crate::{BASE_CANDLE_WIDTH, CANDLE_SPACING, INDICES_PER_CANDLE, VERTICES_PER_CANDLE};

/// Pipeline for rendering candlestick charts.
pub struct CandlePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub candle_bind_group_layout: wgpu::BindGroupLayout,
    /// Index buffer for indexed drawing (shared across all candles).
    pub index_buffer: wgpu::Buffer,
}

impl CandlePipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Candle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/candle.wgsl").into()),
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let candle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("candle_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Candle Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &candle_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Candle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
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

        // Create index buffer for indexed drawing
        // 12 unique vertices (4 per quad), 18 indices (6 per quad forming 2 triangles)
        // Quad vertex layout: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
        // Index pattern per quad: [0,1,2, 0,2,3] = bottom-left triangle + top-left triangle
        let indices: [u16; INDICES_PER_CANDLE as usize] = [
            // Body (vertices 0-3)
            0, 1, 2, 0, 2, 3,
            // Upper wick (vertices 4-7)
            4, 5, 6, 4, 6, 7,
            // Lower wick (vertices 8-11)
            8, 9, 10, 8, 10, 11,
        ];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Candle Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            camera_bind_group_layout,
            candle_bind_group_layout,
            index_buffer,
        }
    }

    pub fn create_candle_buffer(&self, device: &wgpu::Device, candles: &[CandleGpu]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Candle Buffer"),
            contents: bytemuck::cast_slice(candles),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    pub fn create_render_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = RenderParams {
            first_visible: 0,
            candle_width: BASE_CANDLE_WIDTH,
            candle_spacing: CANDLE_SPACING,
            wick_width: CANDLE_SPACING * 0.08,
            // Default view bounds (will be updated each frame)
            x_min: 0.0,
            x_max: f32::MAX,
            y_min: 0.0,
            y_max: f32::MAX,
            // Default price normalization (will be updated per timeframe)
            price_min: 0.0,
            price_range: 1.0,
            min_body_height: CANDLE_SPACING * 0.01, // Will be updated each frame
            _padding: 0.0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        candle_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.candle_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("Candle Bind Group"),
        })
    }
}

impl Pipeline for CandlePipeline {
    type BindGroupData = wgpu::BindGroup;

    fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        _vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..INDICES_PER_CANDLE, 0, instance_range);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl InstancedPipeline for CandlePipeline {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_CANDLE;
}
