// Technical Analysis shader for rendering ranges, levels, and trends

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct TaRenderParams {
    first_visible: u32,
    candle_spacing: f32,
    range_thickness: f32,
    level_thickness: f32,
    x_max: f32,
    range_count: u32,
    level_count: u32,
    trend_count: u32,
};

struct RangeGpu {
    x_start: f32,
    x_end: f32,
    y_pos: f32,
    is_bullish: u32,
};

struct LevelGpu {
    y_value: f32,
    x_start: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    level_type: u32,
    hit_count: u32,
};

struct TrendGpu {
    x_start: f32,
    y_start: f32,
    x_end: f32,
    y_end: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> ranges: array<RangeGpu>;

@group(1) @binding(1)
var<storage, read> levels: array<LevelGpu>;

@group(1) @binding(2)
var<uniform> params: TaRenderParams;

@group(1) @binding(3)
var<storage, read> trends: array<TrendGpu>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Vertex shader for range underlines
// Each range is 6 vertices (2 triangles forming a rectangle)
@vertex
fn vs_range(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let range = ranges[instance_index];

    // 6 vertices for a quad (2 triangles)
    // 0--1
    // |\ |
    // | \|
    // 2--3
    // Triangles: 0-2-1, 1-2-3
    let quad_index = vertex_index % 6u;

    var x: f32;
    var y: f32;

    let half_thickness = params.range_thickness * 0.5;

    if quad_index == 0u {
        x = range.x_start;
        y = range.y_pos + half_thickness;
    } else if quad_index == 1u || quad_index == 4u {
        x = range.x_end;
        y = range.y_pos + half_thickness;
    } else if quad_index == 2u || quad_index == 3u {
        x = range.x_start;
        y = range.y_pos - half_thickness;
    } else {
        x = range.x_end;
        y = range.y_pos - half_thickness;
    }

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    // Color based on direction
    var color: vec4<f32>;
    if range.is_bullish == 1u {
        color = vec4<f32>(0.2, 0.7, 0.3, 0.8); // Green for bullish
    } else {
        color = vec4<f32>(0.8, 0.2, 0.2, 0.8); // Red for bearish
    }

    var output: VertexOutput;
    output.position = pos;
    output.color = color;
    return output;
}

@fragment
fn fs_range(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// Vertex shader for level lines
// Each level is 6 vertices (2 triangles forming a horizontal line)
@vertex
fn vs_level(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let level = levels[instance_index];

    let quad_index = vertex_index % 6u;

    var x: f32;
    var y: f32;

    let half_thickness = params.level_thickness * 0.5;
    let x_start = level.x_start;
    let x_end = params.x_max;

    if quad_index == 0u {
        x = x_start;
        y = level.y_value + half_thickness;
    } else if quad_index == 1u || quad_index == 4u {
        x = x_end;
        y = level.y_value + half_thickness;
    } else if quad_index == 2u || quad_index == 3u {
        x = x_start;
        y = level.y_value - half_thickness;
    } else {
        x = x_end;
        y = level.y_value - half_thickness;
    }

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(level.r, level.g, level.b, level.a);
    return output;
}

@fragment
fn fs_level(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// Vertex shader for trendlines
// Each trend is 6 vertices (2 triangles forming a sloped line extending to x_max)
@vertex
fn vs_trend(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let trend = trends[instance_index];

    let quad_index = vertex_index % 6u;

    // Calculate slope from the two defining points
    let dx = trend.x_end - trend.x_start;
    let dy = trend.y_end - trend.y_start;

    // Extend line to x_max using the slope
    var x_extended = params.x_max;
    var y_extended = trend.y_end; // Default to end point
    if abs(dx) > 0.0001 {
        let slope = dy / dx;
        y_extended = trend.y_start + slope * (x_extended - trend.x_start);
    }

    // Use simple Y offset for thickness (like levels)
    // This gives consistent visual thickness regardless of slope
    let half_thickness = params.level_thickness * 0.5;

    var x: f32;
    var y: f32;

    // Build a quad from start point to extended end point
    // Offset only in Y direction for consistent thickness
    // 0--1
    // |\ |
    // | \|
    // 2--3
    // Triangles: 0-2-1, 1-2-3
    if quad_index == 0u {
        x = trend.x_start;
        y = trend.y_start + half_thickness;
    } else if quad_index == 1u || quad_index == 4u {
        x = x_extended;
        y = y_extended + half_thickness;
    } else if quad_index == 2u || quad_index == 3u {
        x = trend.x_start;
        y = trend.y_start - half_thickness;
    } else {
        x = x_extended;
        y = y_extended - half_thickness;
    }

    let pos = camera.view_proj * vec4<f32>(x, y, 0.0, 1.0);

    var output: VertexOutput;
    output.position = pos;
    output.color = vec4<f32>(trend.r, trend.g, trend.b, trend.a);
    return output;
}

@fragment
fn fs_trend(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
