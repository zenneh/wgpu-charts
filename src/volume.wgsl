// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Volume data in storage buffer
struct VolumeBar {
    volume: f32,
    is_bullish: u32,
    _padding1: f32,
    _padding2: f32,
};

struct VolumeArray {
    bars: array<VolumeBar>,
};

@group(1) @binding(0)
var<storage, read> volume_data: VolumeArray;

// Render parameters
struct RenderParams {
    first_visible: u32,
    bar_width: f32,
    bar_spacing: f32,
    max_volume: f32,
};

@group(1) @binding(1)
var<uniform> params: RenderParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Colors (same as candles but dimmer)
const BULLISH_COLOR: vec3<f32> = vec3<f32>(0.0, 0.5, 0.3);
const BEARISH_COLOR: vec3<f32> = vec3<f32>(0.6, 0.15, 0.15);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let bar_index = params.first_visible + instance_index;
    let bar = volume_data.bars[bar_index];

    let x_center = f32(bar_index) * params.bar_spacing;
    let half_width = params.bar_width / 2.0;

    // Normalize volume to 0-1 range based on max visible volume
    let normalized_height = bar.volume / params.max_volume;
    let bar_height = normalized_height; // Height in NDC-like space (0 to 1)

    // Volume bars are simple quads from bottom (0) to normalized height
    // 6 vertices per bar: 2 triangles
    let local_vertex = vertex_index % 6u;

    var pos: vec2<f32>;
    switch local_vertex {
        case 0u: { pos = vec2<f32>(x_center - half_width, 0.0); }
        case 1u: { pos = vec2<f32>(x_center + half_width, 0.0); }
        case 2u: { pos = vec2<f32>(x_center + half_width, bar_height); }
        case 3u: { pos = vec2<f32>(x_center - half_width, 0.0); }
        case 4u: { pos = vec2<f32>(x_center + half_width, bar_height); }
        case 5u: { pos = vec2<f32>(x_center - half_width, bar_height); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = select(BEARISH_COLOR, BULLISH_COLOR, bar.is_bullish == 1u);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
