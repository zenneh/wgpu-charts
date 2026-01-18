//! Technical Analysis rendering pipeline.

use std::ops::Range;

use wgpu::util::DeviceExt;

use crate::gpu_types::{LevelGpu, RangeGpu, TaRenderParams, TrendGpu, MAX_TA_LEVELS, MAX_TA_RANGES, MAX_TA_TRENDS};
use crate::pipeline::traits::InstancedPipeline;

/// Pipeline for rendering technical analysis elements (ranges, levels, and trends).
pub struct TaPipeline {
    pub range_pipeline: wgpu::RenderPipeline,
    pub level_pipeline: wgpu::RenderPipeline,
    pub trend_pipeline: wgpu::RenderPipeline,
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
                    // Trends storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        // Trend rendering pipeline
        let trend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TA Trend Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_trend"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_trend"),
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
            trend_pipeline,
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
            trend_count: 0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TA Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_trend_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial_trends = vec![
            TrendGpu {
                x_start: 0.0,
                y_start: 0.0,
                x_end: 0.0,
                y_end: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            };
            MAX_TA_TRENDS
        ];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TA Trend Buffer"),
            contents: bytemuck::cast_slice(&initial_trends),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        range_buffer: &wgpu::Buffer,
        level_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        trend_buffer: &wgpu::Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: trend_buffer.as_entire_binding(),
                },
            ],
            label: Some("TA Bind Group"),
        })
    }

    /// Renders TA ranges.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `range_count` - Number of ranges to render
    pub fn render_ranges<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        range_count: u32,
    ) {
        render_pass.set_pipeline(&self.range_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..range_count);
    }

    /// Renders TA levels (support/resistance).
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `level_count` - Number of levels to render
    pub fn render_levels<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        level_count: u32,
    ) {
        render_pass.set_pipeline(&self.level_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..level_count);
    }

    /// Renders TA trends (trendlines).
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `ta_bind_group` - The TA data bind group (slot 1)
    /// * `trend_count` - Number of trends to render
    pub fn render_trends<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        ta_bind_group: &'a wgpu::BindGroup,
        trend_count: u32,
    ) {
        render_pass.set_pipeline(&self.trend_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, ta_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_TA_ELEMENT, 0..trend_count);
    }
}

/// Number of vertices per TA element (2 triangles = 6 vertices).
const VERTICES_PER_TA_ELEMENT: u32 = 6;

/// Wrapper for rendering TA ranges through the Pipeline trait.
pub struct TaRangePipeline<'a>(pub &'a TaPipeline);

impl<'a> crate::pipeline::traits::Pipeline for TaRangePipeline<'a> {
    type BindGroupData = wgpu::BindGroup;

    fn render<'b>(
        &'b self,
        render_pass: &mut wgpu::RenderPass<'b>,
        camera_bind_group: &'b wgpu::BindGroup,
        data_bind_group: &'b wgpu::BindGroup,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.0.range_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(vertex_range, instance_range);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.0.range_pipeline
    }
}

impl<'a> InstancedPipeline for TaRangePipeline<'a> {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_TA_ELEMENT;
}

/// Wrapper for rendering TA levels through the Pipeline trait.
pub struct TaLevelPipeline<'a>(pub &'a TaPipeline);

impl<'a> crate::pipeline::traits::Pipeline for TaLevelPipeline<'a> {
    type BindGroupData = wgpu::BindGroup;

    fn render<'b>(
        &'b self,
        render_pass: &mut wgpu::RenderPass<'b>,
        camera_bind_group: &'b wgpu::BindGroup,
        data_bind_group: &'b wgpu::BindGroup,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.0.level_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(vertex_range, instance_range);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.0.level_pipeline
    }
}

impl<'a> InstancedPipeline for TaLevelPipeline<'a> {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_TA_ELEMENT;
}

/// Wrapper for rendering TA trends through the Pipeline trait.
pub struct TaTrendPipeline<'a>(pub &'a TaPipeline);

impl<'a> crate::pipeline::traits::Pipeline for TaTrendPipeline<'a> {
    type BindGroupData = wgpu::BindGroup;

    fn render<'b>(
        &'b self,
        render_pass: &mut wgpu::RenderPass<'b>,
        camera_bind_group: &'b wgpu::BindGroup,
        data_bind_group: &'b wgpu::BindGroup,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    ) {
        render_pass.set_pipeline(&self.0.trend_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, data_bind_group, &[]);
        render_pass.draw(vertex_range, instance_range);
    }

    fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.0.trend_pipeline
    }
}

impl<'a> InstancedPipeline for TaTrendPipeline<'a> {
    const VERTICES_PER_INSTANCE: u32 = VERTICES_PER_TA_ELEMENT;
}
