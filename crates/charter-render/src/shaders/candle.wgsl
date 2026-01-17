// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Candle data in storage buffer
struct Candle {
    open: f32,
    high: f32,
    low: f32,
    close: f32,
};

struct CandleArray {
    candles: array<Candle>,
};

@group(1) @binding(0)
var<storage, read> candle_data: CandleArray;

// Render parameters
struct RenderParams {
    first_visible: u32,
    candle_width: f32,
    candle_spacing: f32,
    wick_width: f32,
};

@group(1) @binding(1)
var<uniform> params: RenderParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Colors
const BULLISH_COLOR: vec3<f32> = vec3<f32>(0.0, 0.8, 0.4);
const BEARISH_COLOR: vec3<f32> = vec3<f32>(0.9, 0.2, 0.2);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Get the actual candle index (accounting for viewport culling offset)
    let candle_index = params.first_visible + instance_index;
    let candle = candle_data.candles[candle_index];

    // Calculate candle properties
    let x_center = f32(candle_index) * params.candle_spacing;
    let is_bullish = candle.close >= candle.open;

    // Ensure minimum body height for doji candles (open == close)
    let raw_body_top = max(candle.open, candle.close);
    let raw_body_bottom = min(candle.open, candle.close);
    let min_body_height = params.wick_width; // Use wick_width as minimum (it's already adaptive)
    let body_center = (raw_body_top + raw_body_bottom) / 2.0;
    let body_height = max(raw_body_top - raw_body_bottom, min_body_height);
    let body_top = body_center + body_height / 2.0;
    let body_bottom = body_center - body_height / 2.0;

    // Candle body width (adaptive based on zoom)
    let half_width = params.candle_width / 2.0;
    // Wick width: adaptive, passed from CPU to ensure minimum pixel visibility
    let wick_half = params.wick_width / 2.0;

    // Determine which part of the candle we're drawing
    // 0-5: body, 6-11: upper wick, 12-17: lower wick
    let part = vertex_index / 6u;
    let local_vertex = vertex_index % 6u;

    // Same color for body and wicks
    let color = select(BEARISH_COLOR, BULLISH_COLOR, is_bullish);

    var pos: vec2<f32>;

    if part == 0u {
        // Body - two triangles forming a quad
        // Triangle 1: 0, 1, 2 (bottom-left, bottom-right, top-right)
        // Triangle 2: 3, 4, 5 (bottom-left, top-right, top-left)
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - half_width, body_bottom); }
            case 1u: { pos = vec2<f32>(x_center + half_width, body_bottom); }
            case 2u: { pos = vec2<f32>(x_center + half_width, body_top); }
            case 3u: { pos = vec2<f32>(x_center - half_width, body_bottom); }
            case 4u: { pos = vec2<f32>(x_center + half_width, body_top); }
            case 5u: { pos = vec2<f32>(x_center - half_width, body_top); }
            default: { pos = vec2<f32>(0.0, 0.0); }
        }
    } else if part == 1u {
        // Upper wick (from raw body top to high) - use raw values to avoid overlap with expanded body
        let wick_top = candle.high;
        let wick_bottom = raw_body_top;
        // Only draw if there's actual wick height
        let upper_wick_top = select(wick_bottom, wick_top, wick_top > wick_bottom);
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - wick_half, wick_bottom); }
            case 1u: { pos = vec2<f32>(x_center + wick_half, wick_bottom); }
            case 2u: { pos = vec2<f32>(x_center + wick_half, upper_wick_top); }
            case 3u: { pos = vec2<f32>(x_center - wick_half, wick_bottom); }
            case 4u: { pos = vec2<f32>(x_center + wick_half, upper_wick_top); }
            case 5u: { pos = vec2<f32>(x_center - wick_half, upper_wick_top); }
            default: { pos = vec2<f32>(0.0, 0.0); }
        }
    } else {
        // Lower wick (from low to raw body bottom) - use raw values
        let wick_top = raw_body_bottom;
        let wick_bottom = candle.low;
        // Only draw if there's actual wick height
        let lower_wick_bottom = select(wick_top, wick_bottom, wick_bottom < wick_top);
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - wick_half, lower_wick_bottom); }
            case 1u: { pos = vec2<f32>(x_center + wick_half, lower_wick_bottom); }
            case 2u: { pos = vec2<f32>(x_center + wick_half, wick_top); }
            case 3u: { pos = vec2<f32>(x_center - wick_half, lower_wick_bottom); }
            case 4u: { pos = vec2<f32>(x_center + wick_half, wick_top); }
            case 5u: { pos = vec2<f32>(x_center - wick_half, wick_top); }
            default: { pos = vec2<f32>(0.0, 0.0); }
        }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
