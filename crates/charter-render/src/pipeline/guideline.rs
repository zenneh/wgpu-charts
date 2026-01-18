//! Guideline rendering pipeline.

use std::ops::Range;

use wgpu::util::DeviceExt;

use crate::gpu_types::{GuidelineGpu, GuidelineParams, MAX_GUIDELINES};
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
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Guideline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/guidelines.wgsl").into()),
        });

        let guideline_bind_group_layout =
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
                label: Some("guideline_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Guideline Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &guideline_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Guideline Pipeline"),
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
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Guideline Buffer"),
            contents: bytemuck::cast_slice(&initial_guidelines),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_guideline_params_buffer(&self, device: &wgpu::Device, x_max: f32, price_range: f32) -> wgpu::Buffer {
        let params = GuidelineParams {
            x_min: 0.0,
            x_max,
            line_thickness: price_range * 0.001,
            count: 0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Guideline Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        guideline_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.guideline_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: guideline_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("Guideline Bind Group"),
        })
    }
}

/// Number of vertices per guideline (2 triangles = 6 vertices).
const VERTICES_PER_GUIDELINE: u32 = 6;

impl Pipeline for GuidelinePipeline {
    type BindGroupData = wgpu::BindGroup;

    fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(vertex_range, instance_range);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

impl InstancedPipeline for GuidelinePipeline {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_GUIDELINE;
}
