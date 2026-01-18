//! GPU rendering for charter.

pub mod camera;
pub mod gpu_types;
pub mod pipeline;
pub mod renderer;

pub use camera::{Camera, CameraUniform};
pub use gpu_types::{
    aggregate_candles_lod, aggregate_volume_lod, CandleGpu, GuidelineGpu, GuidelineParams,
    IndicatorParams, IndicatorPointGpu, LevelGpu, LodConfig, PackedCandleGpu, PriceNormalization,
    RangeGpu, RenderParams, TaRenderParams, TrendGpu, VolumeGpu, VolumeRenderParams,
    MAX_GUIDELINES, MAX_TA_LEVELS, MAX_TA_RANGES, MAX_TA_TRENDS,
};
pub use pipeline::{
    CandlePipeline, GuidelinePipeline, IndicatorPipeline, TaPipeline, VolumePipeline,
};
pub use renderer::{ChartRenderer, LodData, TimeframeData};

/// Constants for candle rendering.
pub const BASE_CANDLE_WIDTH: f32 = 0.8;
pub const CANDLE_SPACING: f32 = 1.2;
pub const VERTICES_PER_CANDLE: u32 = 18; // 6 triangles * 3 vertices
pub const MIN_CANDLE_PIXELS: f32 = 3.0; // Minimum candle width in pixels

/// Layout constants.
pub const STATS_PANEL_WIDTH: f32 = 200.0;
pub const VOLUME_HEIGHT_RATIO: f32 = 0.2; // Volume panel takes 20% of chart area height
