// Volume Profile shader - renders horizontal bars along Y-axis (right side of chart)
// Each instance = one price bucket. Bar width proportional to volume/max_volume.
// Color: green for buy-dominant, red for sell-dominant, blended by ratio.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct VolumeProfileBucketGpu {
    price: f32,
    buy_volume: f32,
    sell_volume: f32,
    _padding: f32,
};

struct VolumeProfileParams {
    bucket_count: u32,
    max_volume: f32,
    profile_width: f32,
    y_min: f32,
    y_max: f32,
    bucket_height: f32,
    x_right: f32,
    visible: u32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> buckets: array<VolumeProfileBucketGpu>;

@group(1) @binding(1)
var<uniform> params: VolumeProfileParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;

    if params.visible == 0u || instance_index >= params.bucket_count {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let bucket = buckets[instance_index];
    let total_volume = bucket.buy_volume + bucket.sell_volume;

    if total_volume <= 0.0 || params.max_volume <= 0.0 {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let volume_ratio = total_volume / params.max_volume;
    let bar_width = params.profile_width * volume_ratio;

    // Bar extends leftward from x_right
    let x_right = params.x_right;
    let x_left = x_right - bar_width;
    let y_bottom = bucket.price - params.bucket_height * 0.5;
    let y_top = bucket.price + params.bucket_height * 0.5;

    // 6 vertices for a quad
    let quad_index = vertex_index % 6u;
    var x: f32;
    var y: f32;
    switch quad_index {
        case 0u: { x = x_left; y = y_top; }
        case 1u: { x = x_right; y = y_top; }
        case 2u: { x = x_left; y = y_bottom; }
        case 3u: { x = x_right; y = y_top; }
        case 4u: { x = x_right; y = y_bottom; }
        default: { x = x_left; y = y_bottom; }
    }

    out.position = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    // Color: green for buy-dominant, red for sell-dominant
    let buy_ratio = bucket.buy_volume / total_volume;
    let r = (1.0 - buy_ratio) * 0.8;
    let g = buy_ratio * 0.7;
    let b = 0.1;
    out.color = vec4<f32>(r, g, b, 0.6);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
