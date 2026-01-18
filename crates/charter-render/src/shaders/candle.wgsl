// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Packed candle data in storage buffer (8 bytes per candle instead of 16)
// Values are normalized u16 packed into u32:
//   open_high: open (lower 16 bits) | high (upper 16 bits)
//   low_close: low (lower 16 bits) | close (upper 16 bits)
struct PackedCandle {
    open_high: u32,
    low_close: u32,
};

struct CandleArray {
    candles: array<PackedCandle>,
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
    // Price denormalization: price = price_min + normalized * price_range
    price_min: f32,
    price_range: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(1) @binding(1)
var<uniform> params: RenderParams;

// Unpack and denormalize a packed candle
fn unpack_candle(packed: PackedCandle) -> vec4<f32> {
    // Extract u16 values from packed u32s
    let open_norm = f32(packed.open_high & 0xFFFFu) / 65535.0;
    let high_norm = f32((packed.open_high >> 16u) & 0xFFFFu) / 65535.0;
    let low_norm = f32(packed.low_close & 0xFFFFu) / 65535.0;
    let close_norm = f32((packed.low_close >> 16u) & 0xFFFFu) / 65535.0;

    // Denormalize to actual prices
    let open = params.price_min + open_norm * params.price_range;
    let high = params.price_min + high_norm * params.price_range;
    let low = params.price_min + low_norm * params.price_range;
    let close = params.price_min + close_norm * params.price_range;

    return vec4<f32>(open, high, low, close);
}

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
    let packed = candle_data.candles[candle_index];

    // Unpack and denormalize candle data (50% memory savings)
    let candle = unpack_candle(packed);
    let open = candle.x;
    let high = candle.y;
    let low = candle.z;
    let close = candle.w;

    // Calculate candle properties
    let x_center = f32(candle_index) * params.candle_spacing;

    // GPU-side culling: check if candle is outside visible range
    // If so, output degenerate triangle (all vertices at same point)
    let half_width = params.candle_width / 2.0;
    let x_left = x_center - half_width;
    let x_right = x_center + half_width;

    // Cull if completely outside X range or Y range
    let outside_x = x_right < params.x_min || x_left > params.x_max;
    let outside_y = high < params.y_min || low > params.y_max;

    if outside_x || outside_y {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }
    let is_bullish = close >= open;

    // Ensure minimum body height for doji candles (open == close)
    let raw_body_top = max(open, close);
    let raw_body_bottom = min(open, close);
    let min_body_height = params.wick_width; // Use wick_width as minimum (it's already adaptive)
    let body_center = (raw_body_top + raw_body_bottom) / 2.0;
    let body_height = max(raw_body_top - raw_body_bottom, min_body_height);
    let body_top = body_center + body_height / 2.0;
    let body_bottom = body_center - body_height / 2.0;

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
        let wick_top = high;
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
        let wick_bottom = low;
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
