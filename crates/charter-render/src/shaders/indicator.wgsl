// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Indicator point data in storage buffer
struct IndicatorPoint {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    _padding: f32,
};

struct IndicatorArray {
    points: array<IndicatorPoint>,
};

@group(1) @binding(0)
var<storage, read> indicator_data: IndicatorArray;

// Render parameters
struct IndicatorParams {
    first_visible: u32,
    point_spacing: f32,
    line_thickness: f32,
    count: u32,
};

@group(1) @binding(1)
var<uniform> params: IndicatorParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Each line segment needs 6 vertices (2 triangles)
    // instance_index is the line segment index
    let segment_index = params.first_visible + instance_index;

    // Skip if we don't have enough points for this segment
    if segment_index + 1u >= params.count {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }

    let p0 = indicator_data.points[segment_index];
    let p1 = indicator_data.points[segment_index + 1u];

    // Direction vector
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let len = sqrt(dx * dx + dy * dy);

    // Handle zero-length segments to avoid division by zero
    if len < 0.0001 {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }

    // Perpendicular (normal) vector for thickness
    let nx = -dy / len * params.line_thickness / 2.0;
    let ny = dx / len * params.line_thickness / 2.0;

    // Build quad from 2 triangles (6 vertices)
    let local_vertex = vertex_index % 6u;

    var pos: vec2<f32>;
    switch local_vertex {
        case 0u: { pos = vec2<f32>(p0.x - nx, p0.y - ny); }
        case 1u: { pos = vec2<f32>(p0.x + nx, p0.y + ny); }
        case 2u: { pos = vec2<f32>(p1.x + nx, p1.y + ny); }
        case 3u: { pos = vec2<f32>(p0.x - nx, p0.y - ny); }
        case 4u: { pos = vec2<f32>(p1.x + nx, p1.y + ny); }
        case 5u: { pos = vec2<f32>(p1.x - nx, p1.y - ny); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = vec3<f32>(p0.r, p0.g, p0.b);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 0.8); // Slightly transparent
}
