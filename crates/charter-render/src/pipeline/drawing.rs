//! Drawing rendering pipeline for user-created chart annotations.
//!
//! This module provides efficient GPU rendering for drawing primitives:
//! - Horizontal rays (single-point, extends to chart edge)
//! - Rays/trendlines (two-point, extends to chart edge)
//! - Rectangles (two-corner, with fill and border)
//! - Anchor handles (for selection/editing)

use wgpu::util::DeviceExt;

use crate::gpu_types::{
    AnchorGpu, DrawingHRayGpu, DrawingRayGpu, DrawingRectGpu, DrawingRenderParams,
    MAX_DRAWING_ANCHORS, MAX_DRAWING_HRAYS, MAX_DRAWING_RAYS, MAX_DRAWING_RECTS,
};

/// Number of vertices per drawing element (2 triangles = 6 vertices).
const VERTICES_PER_ELEMENT: u32 = 6;

/// Number of vertices per rectangle border (4 segments * 6 vertices).
const VERTICES_PER_RECT_BORDER: u32 = 24;

// =============================================================================
// DrawingRenderData - Efficient buffer management with dirty tracking
// =============================================================================

/// Encapsulates all drawing data needed for GPU rendering with dirty tracking.
///
/// This struct provides a clean interface between application state and GPU rendering,
/// only updating GPU buffers when the underlying data has actually changed.
///
/// # Usage
/// ```ignore
/// // Create once at startup
/// let mut render_data = DrawingRenderData::new(&device, &pipeline);
///
/// // Each frame, update only if data changed
/// render_data.update_hrays(&queue, &new_hrays);
/// render_data.update_params(&queue, &params);
///
/// // Render using cached bind group
/// pipeline.render_all(&mut render_pass, &camera_bind_group, &render_data);
/// ```
pub struct DrawingRenderData {
    // GPU buffers - pre-allocated to max capacity
    hray_buffer: wgpu::Buffer,
    ray_buffer: wgpu::Buffer,
    rect_buffer: wgpu::Buffer,
    anchor_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Bind group - created once, reused every frame
    bind_group: wgpu::BindGroup,

    // Current counts for draw calls
    hray_count: u32,
    ray_count: u32,
    rect_count: u32,
    anchor_count: u32,

    // Dirty flags - track which buffers need updating
    // These are set when data changes and cleared after GPU upload
    hrays_dirty: bool,
    rays_dirty: bool,
    rects_dirty: bool,
    anchors_dirty: bool,
    params_dirty: bool,

    // Cached data for comparison (avoids re-upload of identical data)
    cached_hrays: Vec<DrawingHRayGpu>,
    cached_rays: Vec<DrawingRayGpu>,
    cached_rects: Vec<DrawingRectGpu>,
    cached_anchors: Vec<AnchorGpu>,
    cached_params: DrawingRenderParams,

    // Render parameters that change per-frame
    x_min: f32,
    x_max: f32,
    line_thickness: f32,
    x_line_thickness: f32,
    anchor_size: f32,
    anchor_size_x: f32,
}

impl DrawingRenderData {
    /// Create new render data with pre-allocated buffers.
    pub fn new(device: &wgpu::Device, pipeline: &DrawingPipeline) -> Self {
        // Pre-allocate buffers to maximum capacity to avoid reallocation
        let hray_buffer = Self::create_storage_buffer(
            device,
            "Drawing HRay Buffer",
            std::mem::size_of::<DrawingHRayGpu>() * MAX_DRAWING_HRAYS,
        );

        let ray_buffer = Self::create_storage_buffer(
            device,
            "Drawing Ray Buffer",
            std::mem::size_of::<DrawingRayGpu>() * MAX_DRAWING_RAYS,
        );

        let rect_buffer = Self::create_storage_buffer(
            device,
            "Drawing Rect Buffer",
            std::mem::size_of::<DrawingRectGpu>() * MAX_DRAWING_RECTS,
        );

        let anchor_buffer = Self::create_storage_buffer(
            device,
            "Drawing Anchor Buffer",
            std::mem::size_of::<AnchorGpu>() * MAX_DRAWING_ANCHORS,
        );

        let default_params = DrawingRenderParams {
            x_min: 0.0,
            x_max: 1000.0,
            line_thickness: 2.0,
            x_line_thickness: 2.0,
            anchor_size: 1.0,
            anchor_size_x: 1.0,
            hray_count: 0,
            ray_count: 0,
            rect_count: 0,
            anchor_count: 0,
            _padding1: 0,
            _padding2: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing Params Buffer"),
            contents: bytemuck::cast_slice(&[default_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group once - it references the buffers which won't change
        let bind_group = pipeline.create_bind_group(
            device,
            &hray_buffer,
            &ray_buffer,
            &rect_buffer,
            &anchor_buffer,
            &params_buffer,
        );

        Self {
            hray_buffer,
            ray_buffer,
            rect_buffer,
            anchor_buffer,
            params_buffer,
            bind_group,
            hray_count: 0,
            ray_count: 0,
            rect_count: 0,
            anchor_count: 0,
            hrays_dirty: false,
            rays_dirty: false,
            rects_dirty: false,
            anchors_dirty: false,
            params_dirty: false,
            cached_hrays: Vec::with_capacity(MAX_DRAWING_HRAYS),
            cached_rays: Vec::with_capacity(MAX_DRAWING_RAYS),
            cached_rects: Vec::with_capacity(MAX_DRAWING_RECTS),
            cached_anchors: Vec::with_capacity(MAX_DRAWING_ANCHORS),
            cached_params: default_params,
            x_min: 0.0,
            x_max: 1000.0,
            line_thickness: 2.0,
            x_line_thickness: 2.0,
            anchor_size: 1.0,
            anchor_size_x: 1.0,
        }
    }

    /// Create a storage buffer with COPY_DST usage for updates.
    fn create_storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Set horizontal ray data. Only marks dirty if data actually changed.
    pub fn set_hrays(&mut self, hrays: &[DrawingHRayGpu]) {
        let count = hrays.len().min(MAX_DRAWING_HRAYS);
        let hrays = &hrays[..count];

        // Check if data actually changed
        if self.cached_hrays.len() != count || !Self::slice_eq(&self.cached_hrays, hrays) {
            self.cached_hrays.clear();
            self.cached_hrays.extend_from_slice(hrays);
            self.hray_count = count as u32;
            self.hrays_dirty = true;
        }
    }

    /// Set ray/trendline data. Only marks dirty if data actually changed.
    pub fn set_rays(&mut self, rays: &[DrawingRayGpu]) {
        let count = rays.len().min(MAX_DRAWING_RAYS);
        let rays = &rays[..count];

        if self.cached_rays.len() != count || !Self::slice_eq(&self.cached_rays, rays) {
            self.cached_rays.clear();
            self.cached_rays.extend_from_slice(rays);
            self.ray_count = count as u32;
            self.rays_dirty = true;
        }
    }

    /// Set rectangle data. Only marks dirty if data actually changed.
    pub fn set_rects(&mut self, rects: &[DrawingRectGpu]) {
        let count = rects.len().min(MAX_DRAWING_RECTS);
        let rects = &rects[..count];

        if self.cached_rects.len() != count || !Self::slice_eq(&self.cached_rects, rects) {
            self.cached_rects.clear();
            self.cached_rects.extend_from_slice(rects);
            self.rect_count = count as u32;
            self.rects_dirty = true;
        }
    }

    /// Set anchor handle data. Only marks dirty if data actually changed.
    pub fn set_anchors(&mut self, anchors: &[AnchorGpu]) {
        let count = anchors.len().min(MAX_DRAWING_ANCHORS);
        let anchors = &anchors[..count];

        if self.cached_anchors.len() != count || !Self::slice_eq(&self.cached_anchors, anchors) {
            self.cached_anchors.clear();
            self.cached_anchors.extend_from_slice(anchors);
            self.anchor_count = count as u32;
            self.anchors_dirty = true;
        }
    }

    /// Set render parameters (x_min, x_max, line_thickness, x_line_thickness, anchor_size).
    pub fn set_render_params(&mut self, x_min: f32, x_max: f32, line_thickness: f32, x_line_thickness: f32, anchor_size: f32) {
        // Check if any parameter changed
        let changed = (self.x_min - x_min).abs() > f32::EPSILON
            || (self.x_max - x_max).abs() > f32::EPSILON
            || (self.line_thickness - line_thickness).abs() > f32::EPSILON
            || (self.x_line_thickness - x_line_thickness).abs() > f32::EPSILON
            || (self.anchor_size - anchor_size).abs() > f32::EPSILON;

        if changed {
            self.x_min = x_min;
            self.x_max = x_max;
            self.line_thickness = line_thickness;
            self.x_line_thickness = x_line_thickness;
            self.anchor_size = anchor_size;
            self.params_dirty = true;
        }
    }

    /// Upload any dirty data to the GPU. Call this once per frame before rendering.
    pub fn upload_to_gpu(&mut self, queue: &wgpu::Queue) {
        if self.hrays_dirty && !self.cached_hrays.is_empty() {
            queue.write_buffer(&self.hray_buffer, 0, bytemuck::cast_slice(&self.cached_hrays));
            self.hrays_dirty = false;
        }

        if self.rays_dirty && !self.cached_rays.is_empty() {
            queue.write_buffer(&self.ray_buffer, 0, bytemuck::cast_slice(&self.cached_rays));
            self.rays_dirty = false;
        }

        if self.rects_dirty && !self.cached_rects.is_empty() {
            queue.write_buffer(&self.rect_buffer, 0, bytemuck::cast_slice(&self.cached_rects));
            self.rects_dirty = false;
        }

        if self.anchors_dirty && !self.cached_anchors.is_empty() {
            queue.write_buffer(&self.anchor_buffer, 0, bytemuck::cast_slice(&self.cached_anchors));
            self.anchors_dirty = false;
        }

        // Always update params if counts changed or params changed
        if self.params_dirty || self.needs_params_update() {
            let params = DrawingRenderParams {
                x_min: self.x_min,
                x_max: self.x_max,
                line_thickness: self.line_thickness,
                x_line_thickness: self.x_line_thickness,
                anchor_size: self.anchor_size,
                anchor_size_x: self.anchor_size_x,
                hray_count: self.hray_count,
                ray_count: self.ray_count,
                rect_count: self.rect_count,
                anchor_count: self.anchor_count,
                _padding1: 0,
                _padding2: 0,
            };

            if params != self.cached_params {
                queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
                self.cached_params = params;
            }
            self.params_dirty = false;
        }
    }

    /// Check if params buffer needs updating due to count changes.
    fn needs_params_update(&self) -> bool {
        self.cached_params.hray_count != self.hray_count
            || self.cached_params.ray_count != self.ray_count
            || self.cached_params.rect_count != self.rect_count
            || self.cached_params.anchor_count != self.anchor_count
    }

    /// Compare two slices of Pod types for equality using byte comparison.
    fn slice_eq<T: bytemuck::Pod>(a: &[T], b: &[T]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let a_bytes = bytemuck::cast_slice::<T, u8>(a);
        let b_bytes = bytemuck::cast_slice::<T, u8>(b);
        a_bytes == b_bytes
    }

    /// Get the bind group for rendering.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Get the current counts for conditional rendering.
    pub fn counts(&self) -> DrawingCounts {
        DrawingCounts {
            hrays: self.hray_count,
            rays: self.ray_count,
            rects: self.rect_count,
            anchors: self.anchor_count,
        }
    }

    /// Check if there's anything to render.
    pub fn has_content(&self) -> bool {
        self.hray_count > 0 || self.ray_count > 0 || self.rect_count > 0 || self.anchor_count > 0
    }

    /// Clear all drawing data.
    pub fn clear(&mut self) {
        if self.hray_count > 0 {
            self.hray_count = 0;
            self.cached_hrays.clear();
            self.params_dirty = true;
        }
        if self.ray_count > 0 {
            self.ray_count = 0;
            self.cached_rays.clear();
            self.params_dirty = true;
        }
        if self.rect_count > 0 {
            self.rect_count = 0;
            self.cached_rects.clear();
            self.params_dirty = true;
        }
        if self.anchor_count > 0 {
            self.anchor_count = 0;
            self.cached_anchors.clear();
            self.params_dirty = true;
        }
    }
}

/// Drawing counts for conditional rendering.
#[derive(Debug, Clone, Copy)]
pub struct DrawingCounts {
    pub hrays: u32,
    pub rays: u32,
    pub rects: u32,
    pub anchors: u32,
}

// =============================================================================
// DrawingPipeline
// =============================================================================

/// Pipeline for rendering user drawings (horizontal rays, rays, rectangles, anchors).
///
/// This pipeline uses storage buffers for instanced rendering, allowing efficient
/// drawing of many primitives with minimal draw calls.
pub struct DrawingPipeline {
    pub hray_pipeline: wgpu::RenderPipeline,
    pub ray_pipeline: wgpu::RenderPipeline,
    pub rect_fill_pipeline: wgpu::RenderPipeline,
    pub rect_border_pipeline: wgpu::RenderPipeline,
    pub anchor_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl DrawingPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Drawing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/drawing.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Horizontal rays storage buffer
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
                // Rays storage buffer
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
                // Rectangles storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Anchors storage buffer
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
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("drawing_bind_group_layout"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Drawing Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipelines for each drawing type
        let hray_pipeline = Self::create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "vs_hray",
            "fs_hray",
            "Drawing HRay Pipeline",
            format,
        );

        let ray_pipeline = Self::create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "vs_ray",
            "fs_ray",
            "Drawing Ray Pipeline",
            format,
        );

        let rect_fill_pipeline = Self::create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "vs_rect_fill",
            "fs_rect_fill",
            "Drawing Rect Fill Pipeline",
            format,
        );

        let rect_border_pipeline = Self::create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "vs_rect_border",
            "fs_rect_border",
            "Drawing Rect Border Pipeline",
            format,
        );

        let anchor_pipeline = Self::create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "vs_anchor",
            "fs_anchor",
            "Drawing Anchor Pipeline",
            format,
        );

        Self {
            hray_pipeline,
            ray_pipeline,
            rect_fill_pipeline,
            rect_border_pipeline,
            anchor_pipeline,
            bind_group_layout,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        label: &str,
        format: wgpu::TextureFormat,
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
        })
    }

    // =========================================================================
    // Legacy buffer creation methods (for backwards compatibility)
    // =========================================================================

    /// Create the horizontal ray buffer.
    pub fn create_hray_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            DrawingHRayGpu {
                x_start: 0.0,
                y_value: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
                line_style: 0,
                _padding: 0,
            };
            MAX_DRAWING_HRAYS
        ];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing HRay Buffer"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the ray buffer.
    pub fn create_ray_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            DrawingRayGpu {
                x_start: 0.0,
                y_start: 0.0,
                x_end: 0.0,
                y_end: 0.0,
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            };
            MAX_DRAWING_RAYS
        ];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing Ray Buffer"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the rectangle buffer.
    pub fn create_rect_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            DrawingRectGpu {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 0.0,
                y_max: 0.0,
                fill_r: 0.0,
                fill_g: 0.0,
                fill_b: 0.0,
                fill_a: 0.0,
                border_r: 0.0,
                border_g: 0.0,
                border_b: 0.0,
                border_a: 0.0,
            };
            MAX_DRAWING_RECTS
        ];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing Rect Buffer"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the anchor buffer.
    pub fn create_anchor_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let initial = vec![
            AnchorGpu {
                x: 0.0,
                y: 0.0,
                is_hovered: 0,
                is_selected: 0,
            };
            MAX_DRAWING_ANCHORS
        ];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing Anchor Buffer"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the params buffer.
    pub fn create_params_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let params = DrawingRenderParams {
            x_min: 0.0,
            x_max: 1000.0,
            line_thickness: 1.0,
            x_line_thickness: 1.0,
            anchor_size: 0.5,
            anchor_size_x: 0.5,
            hray_count: 0,
            ray_count: 0,
            rect_count: 0,
            anchor_count: 0,
            _padding1: 0,
            _padding2: 0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drawing Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the bind group.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        hray_buffer: &wgpu::Buffer,
        ray_buffer: &wgpu::Buffer,
        rect_buffer: &wgpu::Buffer,
        anchor_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: anchor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("Drawing Bind Group"),
        })
    }

    // =========================================================================
    // Render methods (legacy interface)
    // =========================================================================

    /// Render horizontal rays.
    pub fn render_hrays<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        drawing_bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.hray_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, drawing_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_ELEMENT, 0..count);
    }

    /// Render rays/trendlines.
    pub fn render_rays<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        drawing_bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.ray_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, drawing_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_ELEMENT, 0..count);
    }

    /// Render rectangle fills.
    pub fn render_rect_fills<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        drawing_bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.rect_fill_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, drawing_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_ELEMENT, 0..count);
    }

    /// Render rectangle borders.
    pub fn render_rect_borders<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        drawing_bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.rect_border_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, drawing_bind_group, &[]);
        // 4 segments per rectangle, 6 vertices per segment
        render_pass.draw(0..VERTICES_PER_RECT_BORDER, 0..count);
    }

    /// Render anchor handles.
    pub fn render_anchors<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        drawing_bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.anchor_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, drawing_bind_group, &[]);
        render_pass.draw(0..VERTICES_PER_ELEMENT, 0..count);
    }

    // =========================================================================
    // Optimized render methods using DrawingRenderData
    // =========================================================================

    /// Render all drawings in optimal order, minimizing state changes.
    ///
    /// Renders in this order for correct visual layering:
    /// 1. Rectangle fills (back)
    /// 2. Rectangle borders
    /// 3. Horizontal rays
    /// 4. Rays/trendlines
    /// 5. Anchor handles (front)
    ///
    /// This method sets bind groups only once at the start, avoiding redundant
    /// bind group switches between draw calls.
    pub fn render_all<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        render_data: &'a DrawingRenderData,
    ) {
        let counts = render_data.counts();

        // Early exit if nothing to render
        if counts.hrays == 0 && counts.rays == 0 && counts.rects == 0 && counts.anchors == 0 {
            return;
        }

        // Set bind groups once for all draw calls
        // Note: We still need to set bind groups per pipeline, but the drawing bind group
        // stays the same. The camera bind group at slot 0 is also consistent.

        // Rectangle fills (back layer)
        if counts.rects > 0 {
            render_pass.set_pipeline(&self.rect_fill_pipeline);
            render_pass.set_bind_group(0, camera_bind_group, &[]);
            render_pass.set_bind_group(1, render_data.bind_group(), &[]);
            render_pass.draw(0..VERTICES_PER_ELEMENT, 0..counts.rects);

            // Rectangle borders (same bind groups, different pipeline)
            render_pass.set_pipeline(&self.rect_border_pipeline);
            // Bind groups still set from rect_fill
            render_pass.draw(0..VERTICES_PER_RECT_BORDER, 0..counts.rects);
        }

        // Horizontal rays
        if counts.hrays > 0 {
            render_pass.set_pipeline(&self.hray_pipeline);
            if counts.rects == 0 {
                // Need to set bind groups if rects didn't set them
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_bind_group(1, render_data.bind_group(), &[]);
            }
            render_pass.draw(0..VERTICES_PER_ELEMENT, 0..counts.hrays);
        }

        // Rays/trendlines
        if counts.rays > 0 {
            render_pass.set_pipeline(&self.ray_pipeline);
            if counts.rects == 0 && counts.hrays == 0 {
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_bind_group(1, render_data.bind_group(), &[]);
            }
            render_pass.draw(0..VERTICES_PER_ELEMENT, 0..counts.rays);
        }

        // Anchor handles (front layer)
        if counts.anchors > 0 {
            render_pass.set_pipeline(&self.anchor_pipeline);
            if counts.rects == 0 && counts.hrays == 0 && counts.rays == 0 {
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_bind_group(1, render_data.bind_group(), &[]);
            }
            render_pass.draw(0..VERTICES_PER_ELEMENT, 0..counts.anchors);
        }
    }
}
