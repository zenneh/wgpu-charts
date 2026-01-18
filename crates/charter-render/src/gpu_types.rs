//! GPU-compatible data structures.

use charter_core::Candle;

/// GPU-compatible candle data for storage buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CandleGpu {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

impl From<&Candle> for CandleGpu {
    fn from(c: &Candle) -> Self {
        Self {
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
        }
    }
}

/// GPU-compatible volume bar data.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeGpu {
    pub volume: f32,
    pub is_bullish: u32,
    pub _padding1: f32,
    pub _padding2: f32,
}

impl VolumeGpu {
    pub fn from_candle(c: &Candle) -> Self {
        Self {
            volume: c.volume,
            is_bullish: if c.close >= c.open { 1 } else { 0 },
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

/// Volume render parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeRenderParams {
    pub first_visible: u32,
    pub bar_width: f32,
    pub bar_spacing: f32,
    pub max_volume: f32,
}

/// Render parameters uniform - passed to shader for instancing.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderParams {
    pub first_visible: u32,
    pub candle_width: f32,
    pub candle_spacing: f32,
    pub wick_width: f32,
    /// Visible X range for GPU-side culling.
    pub x_min: f32,
    pub x_max: f32,
    /// Visible Y range for GPU-side culling.
    pub y_min: f32,
    pub y_max: f32,
}

/// Guideline GPU struct (16 bytes aligned).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GuidelineGpu {
    pub y_value: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// Guideline render parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GuidelineParams {
    pub x_min: f32,
    pub x_max: f32,
    pub line_thickness: f32,
    pub count: u32,
}

/// Maximum number of guidelines that can be rendered.
pub const MAX_GUIDELINES: usize = 32;

/// Indicator line GPU struct (for rendering indicator output).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndicatorPointGpu {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub _padding: f32,
}

/// Indicator render parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndicatorParams {
    pub first_visible: u32,
    pub point_spacing: f32,
    pub line_thickness: f32,
    pub count: u32,
}

// ============================================================================
// Technical Analysis GPU Types
// ============================================================================

/// GPU struct for rendering a range underline.
/// Each range is rendered as a colored line segment below the candles.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RangeGpu {
    /// Start X position (in candle spacing units).
    pub x_start: f32,
    /// End X position (in candle spacing units).
    pub x_end: f32,
    /// Y position (below the lowest candle in range).
    pub y_pos: f32,
    /// 1 = bullish (green), 0 = bearish (red).
    pub is_bullish: u32,
}

/// GPU struct for rendering a level line.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LevelGpu {
    /// Y price value of the level.
    pub y_value: f32,
    /// X start position (where level was created).
    pub x_start: f32,
    /// Color red component.
    pub r: f32,
    /// Color green component.
    pub g: f32,
    /// Color blue component.
    pub b: f32,
    /// Alpha transparency.
    pub a: f32,
    /// 1 = hold level, 0 = greedy hold.
    pub level_type: u32,
    /// Number of times hit.
    pub hit_count: u32,
}

/// TA render parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TaRenderParams {
    /// First visible candle index.
    pub first_visible: u32,
    /// Candle spacing.
    pub candle_spacing: f32,
    /// Line thickness for ranges.
    pub range_thickness: f32,
    /// Line thickness for levels.
    pub level_thickness: f32,
    /// X max for level lines (right edge of chart).
    pub x_max: f32,
    /// Number of ranges to render.
    pub range_count: u32,
    /// Number of levels to render.
    pub level_count: u32,
    /// Padding.
    pub _padding: u32,
}

/// Maximum number of ranges that can be rendered at once.
pub const MAX_TA_RANGES: usize = 256;

/// Maximum number of levels that can be rendered at once.
pub const MAX_TA_LEVELS: usize = 128;

// ============================================================================
// Level of Detail (LOD) System
// ============================================================================

/// LOD levels for candle aggregation.
/// Each level aggregates N candles into 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LodLevel {
    /// Full resolution - every candle.
    Full,
    /// 10:1 aggregation (10 candles → 1).
    Low,
    /// 100:1 aggregation (100 candles → 1).
    VeryLow,
}

impl LodLevel {
    /// Returns the aggregation factor for this LOD level.
    pub fn factor(&self) -> usize {
        match self {
            LodLevel::Full => 1,
            LodLevel::Low => 10,
            LodLevel::VeryLow => 100,
        }
    }

    /// Returns all LOD levels in order.
    pub fn all() -> [LodLevel; 3] {
        [LodLevel::Full, LodLevel::Low, LodLevel::VeryLow]
    }

    /// Choose appropriate LOD based on candles per pixel.
    /// When many candles map to one pixel, use lower LOD.
    pub fn for_density(candles_per_pixel: f32) -> LodLevel {
        if candles_per_pixel > 50.0 {
            LodLevel::VeryLow
        } else if candles_per_pixel > 5.0 {
            LodLevel::Low
        } else {
            LodLevel::Full
        }
    }
}

/// Aggregates candles for LOD rendering.
pub fn aggregate_candles_lod(candles: &[CandleGpu], factor: usize) -> Vec<CandleGpu> {
    if factor <= 1 {
        return candles.to_vec();
    }

    candles
        .chunks(factor)
        .map(|chunk| {
            let open = chunk.first().map(|c| c.open).unwrap_or(0.0);
            let close = chunk.last().map(|c| c.close).unwrap_or(0.0);
            let high = chunk.iter().map(|c| c.high).fold(f32::MIN, f32::max);
            let low = chunk.iter().map(|c| c.low).fold(f32::MAX, f32::min);
            CandleGpu { open, high, low, close }
        })
        .collect()
}

/// Aggregates volume for LOD rendering.
pub fn aggregate_volume_lod(volumes: &[VolumeGpu], factor: usize) -> Vec<VolumeGpu> {
    if factor <= 1 {
        return volumes.to_vec();
    }

    volumes
        .chunks(factor)
        .map(|chunk| {
            let total_volume: f32 = chunk.iter().map(|v| v.volume).sum();
            // Use the last candle's direction for the aggregated bar
            let is_bullish = chunk.last().map(|v| v.is_bullish).unwrap_or(1);
            VolumeGpu {
                volume: total_volume,
                is_bullish,
                _padding1: 0.0,
                _padding2: 0.0,
            }
        })
        .collect()
}
