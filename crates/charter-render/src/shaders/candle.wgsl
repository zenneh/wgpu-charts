// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Candle data in storage buffer (16 bytes per candle, full f32 precision)
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
    // View bounds for GPU-side culling
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    // Minimum body height for doji candles
    min_body_height: f32,
    _padding: f32,
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

    let open = candle.open;
    let high = candle.high;
    let low = candle.low;
    let close = candle.close;

    // Calculate candle properties
    let x_center = f32(candle_index) * params.candle_spacing;

    // GPU-side culling: Y-axis only (CPU already limits X range via instance count)
    // If completely outside Y range, output degenerate triangle
    let outside_y = high < params.y_min || low > params.y_max;

    if outside_y {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }

    let half_width = params.candle_width / 2.0;
    let is_bullish = close >= open;

    // Ensure minimum body height for doji candles (open == close)
    // Use a small minimum (~1 pixel) to keep body visible without covering wicks
    let raw_body_top = max(open, close);
    let raw_body_bottom = min(open, close);
    let body_center = (raw_body_top + raw_body_bottom) / 2.0;
    let body_height = max(raw_body_top - raw_body_bottom, params.min_body_height);
    let body_top = body_center + body_height / 2.0;
    let body_bottom = body_center - body_height / 2.0;

    // Wick width: adaptive, passed from CPU to ensure minimum pixel visibility
    let wick_half = params.wick_width / 2.0;

    // Indexed drawing: 12 unique vertices (4 per quad), index buffer forms triangles
    // Part: 0=body (vertices 0-3), 1=upper wick (4-7), 2=lower wick (8-11)
    // Local vertex: 0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
    let part = vertex_index / 4u;
    let local_vertex = vertex_index % 4u;

    // Same color for body and wicks
    let color = select(BEARISH_COLOR, BULLISH_COLOR, is_bullish);

    var pos: vec2<f32>;

    if part == 0u {
        // Body quad corners
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - half_width, body_bottom); }
            case 1u: { pos = vec2<f32>(x_center + half_width, body_bottom); }
            case 2u: { pos = vec2<f32>(x_center + half_width, body_top); }
            case 3u: { pos = vec2<f32>(x_center - half_width, body_top); }
            default: { pos = vec2<f32>(0.0, 0.0); }
        }
    } else if part == 1u {
        // Upper wick quad corners (from raw body top to high)
        let wick_top = high;
        let wick_bottom = raw_body_top;
        let upper_wick_top = select(wick_bottom, wick_top, wick_top > wick_bottom);
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - wick_half, wick_bottom); }
            case 1u: { pos = vec2<f32>(x_center + wick_half, wick_bottom); }
            case 2u: { pos = vec2<f32>(x_center + wick_half, upper_wick_top); }
            case 3u: { pos = vec2<f32>(x_center - wick_half, upper_wick_top); }
            default: { pos = vec2<f32>(0.0, 0.0); }
        }
    } else {
        // Lower wick quad corners (from low to raw body bottom)
        let wick_top = raw_body_bottom;
        let wick_bottom = low;
        let lower_wick_bottom = select(wick_top, wick_bottom, wick_bottom < wick_top);
        switch local_vertex {
            case 0u: { pos = vec2<f32>(x_center - wick_half, lower_wick_bottom); }
            case 1u: { pos = vec2<f32>(x_center + wick_half, lower_wick_bottom); }
            case 2u: { pos = vec2<f32>(x_center + wick_half, wick_top); }
            case 3u: { pos = vec2<f32>(x_center - wick_half, wick_top); }
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
