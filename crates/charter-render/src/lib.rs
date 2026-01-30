//! GPU rendering for charter.

pub mod camera;
pub mod gpu_context;
pub mod gpu_types;
pub mod pipeline;
pub mod renderer;

pub use camera::{Camera, CameraUniform};
pub use gpu_context::GpuContext;
pub use gpu_types::{
    aggregate_candles_lod, aggregate_volume_lod, AnchorGpu, CandleGpu,
    DepthHeatmapCellGpu, DepthHeatmapParams, DrawingHRayGpu, DrawingRayGpu, DrawingRectGpu,
    DrawingRenderParams, GuidelineGpu, GuidelineParams, IndicatorParams, IndicatorPointGpu,
    LevelGpu, LodConfig, RangeGpu, RenderParams, TaRenderParams, TrendGpu, VolumeGpu,
    VolumeProfileBucketGpu, VolumeProfileParams, VolumeRenderParams, MAX_DRAWING_ANCHORS,
    MAX_DRAWING_HRAYS, MAX_DRAWING_RAYS, MAX_DRAWING_RECTS, MAX_GUIDELINES,
    MAX_DEPTH_LEVELS, MAX_TA_LEVELS,
    MAX_TA_RANGES, MAX_TA_TRENDS, MAX_VOLUME_PROFILE_BUCKETS,
};
pub use pipeline::{
    CandlePipeline, DepthHeatmapPipeline, DrawingCounts, DrawingPipeline, DrawingRenderData,
    GuidelinePipeline, IndicatorPipeline, InstancedPipeline, Pipeline, SharedLayouts, TaPipeline,
    VolumeProfilePipeline, VolumePipeline,
};
pub use renderer::{CameraBundle, ChartRenderer, LodData, TimeframeData};

/// Constants for candle rendering.
pub const BASE_CANDLE_WIDTH: f32 = 0.8;
pub const CANDLE_SPACING: f32 = 1.2;
pub const VERTICES_PER_CANDLE: u32 = 12; // 4 vertices per quad * 3 quads (indexed drawing)
pub const INDICES_PER_CANDLE: u32 = 18; // 6 indices per quad * 3 quads
pub const MIN_CANDLE_PIXELS: f32 = 3.0; // Minimum candle width in pixels

/// Layout constants.
pub const STATS_PANEL_WIDTH: f32 = 0.0; // Removed right sidebar
pub const TOP_BAR_HEIGHT: f32 = 24.0;
pub const VOLUME_HEIGHT_RATIO: f32 = 0.2; // Volume panel takes 20% of chart area height
