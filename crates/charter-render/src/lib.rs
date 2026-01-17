//! GPU rendering for charter.

pub mod camera;
pub mod gpu_types;
pub mod pipeline;
pub mod renderer;

pub use camera::{Camera, CameraUniform};
pub use gpu_types::{
    CandleGpu, GuidelineGpu, GuidelineParams, LevelGpu, RangeGpu, RenderParams, TaRenderParams,
    VolumeGpu, VolumeRenderParams, MAX_GUIDELINES, MAX_TA_LEVELS, MAX_TA_RANGES,
};
pub use pipeline::{
    CandlePipeline, GuidelinePipeline, IndicatorPipeline, TaPipeline, VolumePipeline,
};
pub use renderer::{ChartRenderer, TimeframeData};

/// Constants for candle rendering.
pub const BASE_CANDLE_WIDTH: f32 = 0.8;
pub const CANDLE_SPACING: f32 = 1.2;
pub const VERTICES_PER_CANDLE: u32 = 18; // 6 triangles * 3 vertices
pub const MIN_CANDLE_PIXELS: f32 = 3.0; // Minimum candle width in pixels

/// Layout constants.
pub const STATS_PANEL_WIDTH: f32 = 200.0;
pub const VOLUME_HEIGHT_RATIO: f32 = 0.2; // Volume panel takes 20% of chart area height
