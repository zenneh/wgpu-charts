//! GPU-compatible data structures.

use charter_core::Candle;

/// GPU-compatible candle data for storage buffer (16 bytes).
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

/// Packed GPU-compatible candle data (8 bytes) - 50% memory reduction.
/// Values are normalized to u16 (0-65535) based on price_min and price_range.
/// Shader denormalizes: price = price_min + (packed_value / 65535.0) * price_range
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedCandleGpu {
    pub open: u16,
    pub high: u16,
    pub low: u16,
    pub close: u16,
}

impl PackedCandleGpu {
    /// Create a packed candle from normalized values.
    pub fn from_normalized(open: f32, high: f32, low: f32, close: f32) -> Self {
        Self {
            open: (open.clamp(0.0, 1.0) * 65535.0) as u16,
            high: (high.clamp(0.0, 1.0) * 65535.0) as u16,
            low: (low.clamp(0.0, 1.0) * 65535.0) as u16,
            close: (close.clamp(0.0, 1.0) * 65535.0) as u16,
        }
    }
}

/// Normalization parameters for packed candles.
#[derive(Debug, Clone, Copy)]
pub struct PriceNormalization {
    pub price_min: f32,
    pub price_range: f32, // price_max - price_min
}

impl PriceNormalization {
    /// Compute normalization parameters from candles.
    pub fn from_candles(candles: &[Candle]) -> Self {
        if candles.is_empty() {
            return Self {
                price_min: 0.0,
                price_range: 1.0,
            };
        }

        let mut min_price = f32::MAX;
        let mut max_price = f32::MIN;

        for c in candles {
            min_price = min_price.min(c.low);
            max_price = max_price.max(c.high);
        }

        // Add small padding to avoid edge cases
        let padding = (max_price - min_price) * 0.001;
        min_price -= padding;
        max_price += padding;

        let range = (max_price - min_price).max(0.001); // Avoid division by zero

        Self {
            price_min: min_price,
            price_range: range,
        }
    }

    /// Normalize a price value to 0.0-1.0 range.
    pub fn normalize(&self, price: f32) -> f32 {
        (price - self.price_min) / self.price_range
    }

    /// Pack candles using this normalization.
    pub fn pack_candles(&self, candles: &[Candle]) -> Vec<PackedCandleGpu> {
        candles
            .iter()
            .map(|c| {
                PackedCandleGpu::from_normalized(
                    self.normalize(c.open),
                    self.normalize(c.high),
                    self.normalize(c.low),
                    self.normalize(c.close),
                )
            })
            .collect()
    }

    /// Pack CandleGpu values using this normalization.
    pub fn pack_candles_gpu(&self, candles: &[CandleGpu]) -> Vec<PackedCandleGpu> {
        candles
            .iter()
            .map(|c| {
                PackedCandleGpu::from_normalized(
                    self.normalize(c.open),
                    self.normalize(c.high),
                    self.normalize(c.low),
                    self.normalize(c.close),
                )
            })
            .collect()
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
    /// Price denormalization: price = price_min + normalized * price_range
    pub price_min: f32,
    pub price_range: f32,
    /// Minimum body height for doji candles (should be small, ~1 pixel).
    pub min_body_height: f32,
    pub _padding: f32,
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

/// Current price line parameters - renders a dotted horizontal line at the current price.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CurrentPriceParams {
    pub y_value: f32,           // Price level
    pub x_min: f32,             // Left edge
    pub x_max: f32,             // Right edge
    pub line_thickness: f32,    // Thickness in world units
    pub r: f32,                 // Red component
    pub g: f32,                 // Green component
    pub b: f32,                 // Blue component
    pub visible: u32,           // 1 = visible, 0 = hidden
    pub dot_spacing: f32,       // Spacing between dots in world units
    pub screen_width: f32,      // Screen width in pixels
    pub _padding1: f32,
    pub _padding2: f32,
}

impl Default for CurrentPriceParams {
    fn default() -> Self {
        Self {
            y_value: 0.0,
            x_min: 0.0,
            x_max: 0.0,
            line_thickness: 0.0,
            r: 1.0,      // Yellow/gold color
            g: 0.843,
            b: 0.0,
            visible: 0,
            dot_spacing: 10.0,
            screen_width: 1920.0,
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

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
    /// X max for level/trend lines (right edge of chart).
    pub x_max: f32,
    /// Number of ranges to render.
    pub range_count: u32,
    /// Number of levels to render.
    pub level_count: u32,
    /// Number of trends to render.
    pub trend_count: u32,
}

/// Maximum number of ranges that can be rendered at once.
pub const MAX_TA_RANGES: usize = 4096;

/// Maximum number of levels that can be rendered at once.
pub const MAX_TA_LEVELS: usize = 4096;

/// Maximum number of trends that can be rendered at once.
pub const MAX_TA_TRENDS: usize = 2048;

/// GPU struct for rendering a trendline.
/// Trendlines connect two points and extend to the right.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrendGpu {
    /// Start X position (candle index * spacing).
    pub x_start: f32,
    /// Start Y position (price).
    pub y_start: f32,
    /// End X position (candle index * spacing).
    pub x_end: f32,
    /// End Y position (price).
    pub y_end: f32,
    /// Color red component.
    pub r: f32,
    /// Color green component.
    pub g: f32,
    /// Color blue component.
    pub b: f32,
    /// Alpha transparency.
    pub a: f32,
}

// ============================================================================
// Level of Detail (LOD) System
// ============================================================================

/// Configuration for LOD system.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// LOD factors to generate (e.g., [2, 5, 10, 25, 50, 100, 250, 500]).
    /// These define the aggregation ratios: factor N means N candles → 1.
    pub factors: Vec<usize>,
    /// Minimum candles per pixel thresholds for each LOD level.
    /// Should have same length as factors. If candles_per_pixel exceeds
    /// threshold[i], use factors[i].
    pub thresholds: Vec<f32>,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self::logarithmic(2, 50000, 16)
    }
}

impl LodConfig {
    /// Returns all configured LOD factors.
    pub fn factors(&self) -> &[usize] {
        &self.factors
    }

    /// Choose appropriate LOD factor based on candles per pixel density.
    pub fn factor_for_density(&self, candles_per_pixel: f32) -> usize {
        // Find the highest threshold that's exceeded
        for (i, &threshold) in self.thresholds.iter().enumerate().rev() {
            if candles_per_pixel > threshold {
                return self.factors[i];
            }
        }
        // Default to full resolution
        1
    }

    /// Create a configuration with evenly spaced factors in log scale.
    pub fn logarithmic(min_factor: usize, max_factor: usize, num_levels: usize) -> Self {
        if num_levels == 0 {
            return Self::default();
        }

        let min_log = (min_factor as f32).ln();
        let max_log = (max_factor as f32).ln();
        let step = (max_log - min_log) / (num_levels as f32);

        let mut factors = Vec::with_capacity(num_levels);
        let mut thresholds = Vec::with_capacity(num_levels);

        for i in 0..num_levels {
            let factor = (min_log + step * (i as f32 + 1.0)).exp().round() as usize;
            factors.push(factor);
            // Threshold is roughly half the factor (when you have factor/2 candles per pixel, switch to this LOD)
            thresholds.push((factor as f32) / 2.0);
        }

        Self { factors, thresholds }
    }

    /// Create a configuration with linear spacing between factors.
    pub fn linear(min_factor: usize, max_factor: usize, step: usize) -> Self {
        let mut factors = Vec::new();
        let mut thresholds = Vec::new();

        let mut factor = min_factor;
        while factor <= max_factor {
            factors.push(factor);
            thresholds.push((factor as f32) / 2.0);
            factor += step;
        }

        Self { factors, thresholds }
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
/// The is_bullish flag is determined by comparing the aggregated period's
/// first open price with the last close price (net price movement).
pub fn aggregate_volume_lod(volumes: &[VolumeGpu], candles: &[CandleGpu], factor: usize) -> Vec<VolumeGpu> {
    if factor <= 1 {
        return volumes.to_vec();
    }

    volumes
        .chunks(factor)
        .zip(candles.chunks(factor))
        .map(|(vol_chunk, candle_chunk)| {
            let total_volume: f32 = vol_chunk.iter().map(|v| v.volume).sum();

            // Calculate is_bullish based on aggregated open/close
            let open = candle_chunk.first().map(|c| c.open).unwrap_or(0.0);
            let close = candle_chunk.last().map(|c| c.close).unwrap_or(0.0);
            let is_bullish = if close >= open { 1 } else { 0 };

            VolumeGpu {
                volume: total_volume,
                is_bullish,
                _padding1: 0.0,
                _padding2: 0.0,
            }
        })
        .collect()
}

// ============================================================================
// User Drawing GPU Types
// ============================================================================

/// GPU struct for rendering a horizontal ray.
/// A horizontal line from a start X position extending to the right edge.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawingHRayGpu {
    /// Start X position (candle index * spacing).
    pub x_start: f32,
    /// Y position (price level).
    pub y_value: f32,
    /// Color RGBA.
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    /// Line style (0 = solid, 1 = dashed, 2 = dotted).
    pub line_style: u32,
    pub _padding: u32,
}

/// GPU struct for rendering a ray/trendline.
/// A line from start to end, extending to the right edge.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawingRayGpu {
    /// Start X position.
    pub x_start: f32,
    /// Start Y position.
    pub y_start: f32,
    /// End X position.
    pub x_end: f32,
    /// End Y position.
    pub y_end: f32,
    /// Color RGBA.
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// GPU struct for rendering a rectangle.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawingRectGpu {
    /// Minimum X position.
    pub x_min: f32,
    /// Minimum Y position.
    pub y_min: f32,
    /// Maximum X position.
    pub x_max: f32,
    /// Maximum Y position.
    pub y_max: f32,
    /// Fill color RGBA.
    pub fill_r: f32,
    pub fill_g: f32,
    pub fill_b: f32,
    pub fill_a: f32,
    /// Border color RGBA.
    pub border_r: f32,
    pub border_g: f32,
    pub border_b: f32,
    pub border_a: f32,
}

/// GPU struct for rendering an anchor point handle.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AnchorGpu {
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Whether this anchor is hovered (1 = yes, 0 = no).
    pub is_hovered: u32,
    /// Whether this anchor's drawing is selected (1 = yes, 0 = no).
    pub is_selected: u32,
}

/// Drawing render parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawingRenderParams {
    /// Left edge X position (for extending rays to left).
    pub x_min: f32,
    /// Right edge X position (for extending rays to right).
    pub x_max: f32,
    /// Line thickness for horizontal lines (in Y-axis world units).
    pub line_thickness: f32,
    /// Line thickness for vertical lines (in X-axis world units).
    pub x_line_thickness: f32,
    /// Anchor handle size in world units.
    pub anchor_size: f32,
    /// Number of horizontal rays.
    pub hray_count: u32,
    /// Number of rays/trendlines.
    pub ray_count: u32,
    /// Number of rectangles.
    pub rect_count: u32,
    /// Number of anchor handles.
    pub anchor_count: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Maximum number of each drawing type.
pub const MAX_DRAWING_HRAYS: usize = 256;
pub const MAX_DRAWING_RAYS: usize = 256;
pub const MAX_DRAWING_RECTS: usize = 256;
pub const MAX_DRAWING_ANCHORS: usize = 512;
