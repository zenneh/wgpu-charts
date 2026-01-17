// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Guidelines data (16-byte aligned struct)
struct Guideline {
    y_value: f32,      // Price level
    r: f32,            // Red component
    g: f32,            // Green component
    b: f32,            // Blue component
};

struct GuidelinesArray {
    lines: array<Guideline>,
};

@group(1) @binding(0)
var<storage, read> guidelines: GuidelinesArray;

// Parameters for guidelines
struct GuidelineParams {
    x_min: f32,
    x_max: f32,
    line_thickness: f32,
    count: u32,
};

@group(1) @binding(1)
var<uniform> params: GuidelineParams;

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

    let guideline = guidelines.lines[instance_index];
    let y = guideline.y_value;
    let half_thick = params.line_thickness / 2.0;

    // Each guideline is a thin horizontal quad spanning x_min to x_max
    // 6 vertices: 2 triangles
    let local_vertex = vertex_index % 6u;

    var pos: vec2<f32>;
    switch local_vertex {
        case 0u: { pos = vec2<f32>(params.x_min, y - half_thick); }
        case 1u: { pos = vec2<f32>(params.x_max, y - half_thick); }
        case 2u: { pos = vec2<f32>(params.x_max, y + half_thick); }
        case 3u: { pos = vec2<f32>(params.x_min, y - half_thick); }
        case 4u: { pos = vec2<f32>(params.x_max, y + half_thick); }
        case 5u: { pos = vec2<f32>(params.x_min, y + half_thick); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = vec3<f32>(guideline.r, guideline.g, guideline.b);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 0.5); // Semi-transparent
}
