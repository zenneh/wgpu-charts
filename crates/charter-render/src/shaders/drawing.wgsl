// Drawing shader for user-created chart annotations
// Renders horizontal rays, rays/trendlines, rectangles, and anchor handles
//
// Optimization notes:
// - Uses vertex pulling from storage buffers for efficient instancing
// - Minimizes branching by using array lookups where possible
// - Pre-computes line extension for rays to avoid per-vertex division
// - Uses vec4 colors for better SIMD utilization

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct DrawingRenderParams {
    x_min: f32,
    x_max: f32,
    line_thickness: f32,      // For horizontal lines (Y-axis units)
    x_line_thickness: f32,    // For vertical lines (X-axis units)
    anchor_size: f32,
    hray_count: u32,
    ray_count: u32,
    rect_count: u32,
    anchor_count: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct DrawingHRayGpu {
    x_start: f32,
    y_value: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    line_style: u32,
    _padding: u32,
};

struct DrawingRayGpu {
    x_start: f32,
    y_start: f32,
    x_end: f32,
    y_end: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

struct DrawingRectGpu {
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    fill_r: f32,
    fill_g: f32,
    fill_b: f32,
    fill_a: f32,
    border_r: f32,
    border_g: f32,
    border_b: f32,
    border_a: f32,
};

struct AnchorGpu {
    x: f32,
    y: f32,
    is_hovered: u32,
    is_selected: u32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> hrays: array<DrawingHRayGpu>;

@group(1) @binding(1)
var<storage, read> rays: array<DrawingRayGpu>;

@group(1) @binding(2)
var<storage, read> rects: array<DrawingRectGpu>;

@group(1) @binding(3)
var<storage, read> anchors: array<AnchorGpu>;

@group(1) @binding(4)
var<uniform> params: DrawingRenderParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// ============================================================================
// Shared utility: quad vertex positions
// ============================================================================

// Quad vertex offsets for a 2-triangle quad (6 vertices):
// Vertex 0: (0, 1) - top-left
// Vertex 1: (1, 1) - top-right
// Vertex 2: (0, 0) - bottom-left
// Vertex 3: (0, 0) - bottom-left (duplicate)
// Vertex 4: (1, 1) - top-right (duplicate)
// Vertex 5: (1, 0) - bottom-right

// X multipliers: [0, 1, 0, 0, 1, 1] - 0 = left edge, 1 = right edge
// Y multipliers: [1, 1, 0, 0, 1, 0] - 0 = bottom edge, 1 = top edge
// But we want +/- half_thickness, so map to [-1, 1]:
// Y offsets: [1, 1, -1, -1, 1, -1]

fn get_quad_x_factor(vertex_index: u32) -> f32 {
    // Returns 0.0 for left vertices, 1.0 for right vertices
    let factors = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    return factors[vertex_index % 6u];
}

fn get_quad_y_sign(vertex_index: u32) -> f32 {
    // Returns +1.0 for top vertices, -1.0 for bottom vertices
    let signs = array<f32, 6>(1.0, 1.0, -1.0, -1.0, 1.0, -1.0);
    return signs[vertex_index % 6u];
}

// ============================================================================
// Horizontal Ray Shader
// ============================================================================

@vertex
fn vs_hray(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let hray = hrays[instance_index];
    let quad_index = vertex_index % 6u;

    let half_thickness = params.line_thickness * 0.5;
    let x_factor = get_quad_x_factor(quad_index);
    let y_sign = get_quad_y_sign(quad_index);

    // Extend horizontal ray from left visible edge (x_min) to right visible edge (x_max)
    // This ensures the ray spans the full canvas width
    let x = mix(params.x_min, params.x_max, x_factor);
    let y = hray.y_value + y_sign * half_thickness;

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(hray.r, hray.g, hray.b, hray.a);
    return output;
}

@fragment
fn fs_hray(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// ============================================================================
// Ray/Trendline Shader
// ============================================================================

@vertex
fn vs_ray(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let ray = rays[instance_index];
    let quad_index = vertex_index % 6u;

    // Calculate extended endpoint
    // The ray extends from (x_start, y_start) through (x_end, y_end) to x_max
    let dx = ray.x_end - ray.x_start;
    let dy = ray.y_end - ray.y_start;

    // Calculate y at x_max using linear interpolation
    // If dx is very small, use y_end (nearly vertical line)
    var y_extended = ray.y_end;
    if abs(dx) > 0.0001 {
        // y = y_start + (dy/dx) * (x_max - x_start)
        // Pre-compute slope to avoid division in hot path
        let slope = dy / dx;
        y_extended = ray.y_start + slope * (params.x_max - ray.x_start);
    }

    let half_thickness = params.line_thickness * 0.5;
    let x_factor = get_quad_x_factor(quad_index);
    let y_sign = get_quad_y_sign(quad_index);

    // Interpolate between start and extended end
    let x = mix(ray.x_start, params.x_max, x_factor);
    let base_y = mix(ray.y_start, y_extended, x_factor);
    let y = base_y + y_sign * half_thickness;

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(ray.r, ray.g, ray.b, ray.a);
    return output;
}

@fragment
fn fs_ray(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// ============================================================================
// Rectangle Shader (Fill)
// ============================================================================

@vertex
fn vs_rect_fill(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let rect = rects[instance_index];
    let quad_index = vertex_index % 6u;

    let x_factor = get_quad_x_factor(quad_index);
    // Map y_sign from [-1, 1] to [0, 1] for rect interpolation: (sign + 1) / 2
    let y_factor = (get_quad_y_sign(quad_index) + 1.0) * 0.5;

    let x = mix(rect.x_min, rect.x_max, x_factor);
    let y = mix(rect.y_min, rect.y_max, y_factor);

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(rect.fill_r, rect.fill_g, rect.fill_b, rect.fill_a);
    return output;
}

@fragment
fn fs_rect_fill(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// ============================================================================
// Rectangle Border Shader (4 line segments)
// ============================================================================

// Each rectangle border needs 4 line segments (top, bottom, left, right)
// 24 vertices total: segments 0-3, each with 6 vertices

@vertex
fn vs_rect_border(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let rect = rects[instance_index];

    let segment = vertex_index / 6u;
    let quad_index = vertex_index % 6u;

    // Use different thicknesses for horizontal (Y-axis) and vertical (X-axis) lines
    let y_half_thickness = params.line_thickness * 0.5;      // For horizontal lines
    let x_half_thickness = params.x_line_thickness * 0.5;    // For vertical lines
    let x_factor = get_quad_x_factor(quad_index);
    let y_sign = get_quad_y_sign(quad_index);

    var x: f32;
    var y: f32;

    // Segment 0: Top edge (horizontal line at y_max)
    // Segment 1: Bottom edge (horizontal line at y_min)
    // Segment 2: Left edge (vertical line at x_min)
    // Segment 3: Right edge (vertical line at x_max)

    switch segment {
        case 0u: {
            // Top edge: horizontal line at y_max (uses Y-axis thickness)
            x = mix(rect.x_min, rect.x_max, x_factor);
            y = rect.y_max + y_sign * y_half_thickness;
        }
        case 1u: {
            // Bottom edge: horizontal line at y_min (uses Y-axis thickness)
            x = mix(rect.x_min, rect.x_max, x_factor);
            y = rect.y_min + y_sign * y_half_thickness;
        }
        case 2u: {
            // Left edge: vertical line at x_min (uses X-axis thickness)
            x = rect.x_min + (x_factor * 2.0 - 1.0) * x_half_thickness;
            y = mix(rect.y_min, rect.y_max, (y_sign + 1.0) * 0.5);
        }
        default: {
            // Right edge: vertical line at x_max (uses X-axis thickness)
            x = rect.x_max + (x_factor * 2.0 - 1.0) * x_half_thickness;
            y = mix(rect.y_min, rect.y_max, (y_sign + 1.0) * 0.5);
        }
    }

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(rect.border_r, rect.border_g, rect.border_b, rect.border_a);
    return output;
}

@fragment
fn fs_rect_border(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// ============================================================================
// Anchor Handle Shader
// ============================================================================

// Anchor colors - pre-defined for different states
const ANCHOR_COLOR_HOVERED: vec4<f32> = vec4<f32>(1.0, 1.0, 0.0, 1.0);  // Yellow
const ANCHOR_COLOR_SELECTED: vec4<f32> = vec4<f32>(0.0, 0.8, 0.8, 1.0); // Cyan
const ANCHOR_COLOR_DEFAULT: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.8);  // Gray

@vertex
fn vs_anchor(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let anchor = anchors[instance_index];
    let quad_index = vertex_index % 6u;

    let half_size = params.anchor_size * 0.5;

    // Get position offsets using quad helper functions
    // Map x_factor from [0, 1] to [-1, 1] for symmetric anchor
    let x_offset = (get_quad_x_factor(quad_index) * 2.0 - 1.0) * half_size;
    let y_offset = get_quad_y_sign(quad_index) * half_size;

    let x = anchor.x + x_offset;
    let y = anchor.y + y_offset;

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    // Determine color based on state
    // Priority: hovered > selected > default
    var color: vec4<f32>;
    if anchor.is_hovered == 1u {
        color = ANCHOR_COLOR_HOVERED;
    } else if anchor.is_selected == 1u {
        color = ANCHOR_COLOR_SELECTED;
    } else {
        color = ANCHOR_COLOR_DEFAULT;
    }

    var output: VertexOutput;
    output.position = pos;
    output.color = color;
    return output;
}

@fragment
fn fs_anchor(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
