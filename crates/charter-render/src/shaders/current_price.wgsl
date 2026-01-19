// Current price line shader - renders a horizontal dotted line at the current price

// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Current price parameters
struct CurrentPriceParams {
    y_value: f32,           // Price level
    x_min: f32,             // Left edge
    x_max: f32,             // Right edge
    line_thickness: f32,    // Thickness in world units
    r: f32,                 // Red component
    g: f32,                 // Green component
    b: f32,                 // Blue component
    visible: u32,           // 1 = visible, 0 = hidden
    dot_spacing: f32,       // Spacing between dots in world units
    screen_width: f32,      // Screen width in pixels
    _padding1: f32,
    _padding2: f32,
};

@group(1) @binding(0)
var<uniform> params: CurrentPriceParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_x: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Don't render if not visible
    if params.visible == 0u {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        out.world_x = 0.0;
        return out;
    }

    let y = params.y_value;
    let half_thick = params.line_thickness / 2.0;

    // Single horizontal quad spanning x_min to x_max
    // 6 vertices: 2 triangles
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(params.x_min, y - half_thick); }
        case 1u: { pos = vec2<f32>(params.x_max, y - half_thick); }
        case 2u: { pos = vec2<f32>(params.x_max, y + half_thick); }
        case 3u: { pos = vec2<f32>(params.x_min, y - half_thick); }
        case 4u: { pos = vec2<f32>(params.x_max, y + half_thick); }
        case 5u: { pos = vec2<f32>(params.x_min, y + half_thick); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.color = vec3<f32>(params.r, params.g, params.b);
    out.world_x = pos.x;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create dotted pattern based on screen x position
    let screen_x = in.clip_position.x;

    // Use screen position for consistent dot spacing regardless of zoom
    let dot_period = 12.0; // Pixels per dot cycle
    let dot_duty = 0.5;    // 50% duty cycle (dot is half the period)

    let pattern = fract(screen_x / dot_period);

    // Discard pixels in the "gap" portion of the pattern
    if pattern > dot_duty {
        discard;
    }

    return vec4<f32>(in.color, 0.9); // Slightly transparent
}
