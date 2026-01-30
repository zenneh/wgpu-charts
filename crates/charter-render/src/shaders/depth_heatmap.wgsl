// Depth Sidebar shader - renders a two-sided horizontal bar chart of the order book.
// Each instance = one price level with bid + ask quantities.
// Bids extend LEFT from center (green/teal), asks extend RIGHT (red/orange).
// Bar width proportional to quantity / max_quantity.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct DepthHeatmapCellGpu {
    price: f32,
    bar_height: f32,
    bid_quantity: f32,
    ask_quantity: f32,
};

struct DepthHeatmapParams {
    level_count: u32,
    _pad0: u32,
    max_quantity: f32,
    half_width: f32,
    _pad1: f32,
    _pad2: f32,
    x_center: f32,
    visible: u32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> cells: array<DepthHeatmapCellGpu>;

@group(1) @binding(1)
var<uniform> params: DepthHeatmapParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;

    if params.visible == 0u || params.max_quantity <= 0.0 {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let cell = cells[instance_index];

    // Y span centered on the exact price
    let y_start = cell.price - cell.bar_height * 0.5;
    let y_end = cell.price + cell.bar_height * 0.5;

    // Determine which side: 0-5 = bid (left), 6-11 = ask (right)
    let is_ask = vertex_index >= 6u;
    let local_idx = vertex_index % 6u;

    var x_left: f32;
    var x_right: f32;
    var bar_color: vec4<f32>;

    if is_ask {
        // Ask bar extends RIGHT from center
        let intensity = clamp(cell.ask_quantity / params.max_quantity, 0.0, 1.0);
        let bar_width = intensity * params.half_width;
        x_left = params.x_center;
        x_right = params.x_center + bar_width;
        // Red/orange with intensity-based alpha
        let r = mix(0.4, 0.85, intensity);
        let g = mix(0.12, 0.25, intensity);
        let b = mix(0.08, 0.12, intensity);
        bar_color = vec4<f32>(r, g, b, mix(0.2, 0.65, intensity));
    } else {
        // Bid bar extends LEFT from center
        let intensity = clamp(cell.bid_quantity / params.max_quantity, 0.0, 1.0);
        let bar_width = intensity * params.half_width;
        x_left = params.x_center - bar_width;
        x_right = params.x_center;
        // Green/teal with intensity-based alpha
        let r = mix(0.05, 0.12, intensity);
        let g = mix(0.25, 0.7, intensity);
        let b = mix(0.2, 0.45, intensity);
        bar_color = vec4<f32>(r, g, b, mix(0.2, 0.65, intensity));
    }

    // Skip degenerate bars
    if x_right - x_left < 0.0001 {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    // 6 vertices per quad (2 triangles)
    var x: f32;
    var y: f32;
    switch local_idx {
        case 0u: { x = x_left;  y = y_end; }
        case 1u: { x = x_right; y = y_end; }
        case 2u: { x = x_left;  y = y_start; }
        case 3u: { x = x_right; y = y_end; }
        case 4u: { x = x_right; y = y_start; }
        default: { x = x_left;  y = y_start; }
    }

    out.position = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);
    out.color = bar_color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.color.a < 0.01 {
        discard;
    }
    return in.color;
}
