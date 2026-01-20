//! Centralized coordinate system for Charter.
//!
//! This module provides the SINGLE source of truth for all coordinate conversions
//! in the application. It defines three coordinate spaces:
//!
//! - **Screen coordinates** ([`ScreenPos`]): Pixel positions from the top-left of the window
//! - **NDC coordinates** ([`NdcPos`]): Normalized Device Coordinates (-1 to 1)
//! - **World coordinates** ([`WorldPos`]): Chart coordinates (candle index, price)
//!
//! The [`CoordinateSystem`] struct provides all conversion methods and should be
//! used instead of ad-hoc conversions scattered throughout the codebase.
//!
//! # Example
//!
//! ```ignore
//! use charter::coords::{CoordinateSystem, ScreenPos};
//!
//! let coords = CoordinateSystem::new(
//!     1920, 1080,
//!     camera_position,
//!     camera_scale,
//! );
//!
//! // Convert mouse position to world coordinates
//! let screen = ScreenPos::new(960.0, 540.0);
//! let world = coords.screen_to_world(screen);
//! println!("Mouse at candle {}, price {}", world.candle_index, world.price);
//! ```

/// Layout constants imported from charter-render.
/// These define the chart area dimensions.
pub use charter_render::{CANDLE_SPACING, STATS_PANEL_WIDTH, VOLUME_HEIGHT_RATIO};

/// Screen coordinates in pixels from the top-left corner.
///
/// The origin (0, 0) is at the top-left of the window.
/// X increases to the right, Y increases downward.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScreenPos {
    /// X position in pixels from the left edge.
    pub x: f32,
    /// Y position in pixels from the top edge.
    pub y: f32,
}

impl ScreenPos {
    /// Create a new screen position.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Create from a tuple.
    #[must_use]
    pub const fn from_tuple(pos: (f32, f32)) -> Self {
        Self { x: pos.0, y: pos.1 }
    }

    /// Create from winit PhysicalPosition.
    #[must_use]
    pub fn from_physical(pos: winit::dpi::PhysicalPosition<f64>) -> Self {
        Self {
            x: pos.x as f32,
            y: pos.y as f32,
        }
    }

    /// Convert to a tuple.
    #[must_use]
    pub const fn to_tuple(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Convert to an array.
    #[must_use]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Calculate distance to another screen position.
    #[must_use]
    pub fn distance_to(self, other: ScreenPos) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate squared distance (faster than distance_to when only comparing).
    #[must_use]
    pub fn distance_squared_to(self, other: ScreenPos) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

impl From<(f32, f32)> for ScreenPos {
    fn from(pos: (f32, f32)) -> Self {
        Self::from_tuple(pos)
    }
}

impl From<[f32; 2]> for ScreenPos {
    fn from(pos: [f32; 2]) -> Self {
        Self::new(pos[0], pos[1])
    }
}

impl From<winit::dpi::PhysicalPosition<f64>> for ScreenPos {
    fn from(pos: winit::dpi::PhysicalPosition<f64>) -> Self {
        Self::from_physical(pos)
    }
}

/// Normalized Device Coordinates.
///
/// Range is -1 to 1 for both axes.
/// - X: -1 is left edge, +1 is right edge
/// - Y: -1 is bottom edge, +1 is top edge
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NdcPos {
    /// X position in NDC (-1 to 1).
    pub x: f32,
    /// Y position in NDC (-1 to 1).
    pub y: f32,
}

impl NdcPos {
    /// Create a new NDC position.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Check if the position is within the valid NDC range.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.x >= -1.0 && self.x <= 1.0 && self.y >= -1.0 && self.y <= 1.0
    }

    /// Clamp the position to the valid NDC range.
    #[must_use]
    pub fn clamp(self) -> Self {
        Self {
            x: self.x.clamp(-1.0, 1.0),
            y: self.y.clamp(-1.0, 1.0),
        }
    }
}

/// World/Chart coordinates.
///
/// These represent positions in the chart's logical coordinate space:
/// - `candle_index`: The horizontal position in units of candle indices (can be fractional)
/// - `price`: The vertical position in price units
///
/// Note: The `candle_index` is in "candle units", not world X units.
/// To get the actual world X coordinate, multiply by `CANDLE_SPACING`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WorldPos {
    /// X position as a candle index (fractional).
    pub candle_index: f32,
    /// Y position in price units.
    pub price: f32,
}

impl WorldPos {
    /// Create a new world position.
    #[must_use]
    pub const fn new(candle_index: f32, price: f32) -> Self {
        Self { candle_index, price }
    }

    /// Get the actual world X coordinate (candle_index * CANDLE_SPACING).
    #[must_use]
    pub fn world_x(self) -> f32 {
        self.candle_index * CANDLE_SPACING
    }

    /// Create from raw world coordinates (x, y) where x is already scaled.
    #[must_use]
    pub fn from_world_xy(x: f32, y: f32) -> Self {
        Self {
            candle_index: x / CANDLE_SPACING,
            price: y,
        }
    }

    /// Convert to raw world coordinates (x, y) tuple.
    #[must_use]
    pub fn to_world_xy(self) -> (f32, f32) {
        (self.world_x(), self.price)
    }

    /// Get the integer candle index (floor).
    #[must_use]
    pub fn candle_index_floor(self) -> usize {
        self.candle_index.floor().max(0.0) as usize
    }

    /// Get the nearest integer candle index (round).
    #[must_use]
    pub fn candle_index_round(self) -> usize {
        self.candle_index.round().max(0.0) as usize
    }

    /// Calculate distance to another world position.
    #[must_use]
    pub fn distance_to(self, other: WorldPos) -> f32 {
        let dx = (self.candle_index - other.candle_index) * CANDLE_SPACING;
        let dy = self.price - other.price;
        (dx * dx + dy * dy).sqrt()
    }

    /// Translate by a delta.
    #[must_use]
    pub fn translate(self, d_candle: f32, d_price: f32) -> Self {
        Self {
            candle_index: self.candle_index + d_candle,
            price: self.price + d_price,
        }
    }
}

/// Chart bounds in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChartBounds {
    /// Minimum candle index.
    pub x_min: f32,
    /// Maximum candle index.
    pub x_max: f32,
    /// Minimum price.
    pub y_min: f32,
    /// Maximum price.
    pub y_max: f32,
}

impl ChartBounds {
    /// Create new chart bounds.
    #[must_use]
    pub const fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Get the width in candle indices.
    #[must_use]
    pub fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    /// Get the height in price units.
    #[must_use]
    pub fn height(self) -> f32 {
        self.y_max - self.y_min
    }

    /// Get the center point.
    #[must_use]
    pub fn center(self) -> WorldPos {
        WorldPos {
            candle_index: (self.x_min + self.x_max) / 2.0,
            price: (self.y_min + self.y_max) / 2.0,
        }
    }

    /// Check if a world position is within these bounds.
    #[must_use]
    pub fn contains(self, pos: WorldPos) -> bool {
        pos.candle_index >= self.x_min
            && pos.candle_index <= self.x_max
            && pos.price >= self.y_min
            && pos.price <= self.y_max
    }

    /// Expand bounds by a margin (in percentage, e.g., 0.1 = 10%).
    #[must_use]
    pub fn expand(self, margin: f32) -> Self {
        let width_margin = self.width() * margin / 2.0;
        let height_margin = self.height() * margin / 2.0;
        Self {
            x_min: self.x_min - width_margin,
            x_max: self.x_max + width_margin,
            y_min: self.y_min - height_margin,
            y_max: self.y_max + height_margin,
        }
    }
}

impl Default for ChartBounds {
    fn default() -> Self {
        Self {
            x_min: 0.0,
            x_max: 100.0,
            y_min: 0.0,
            y_max: 100.0,
        }
    }
}

/// Centralized coordinate system for all conversions.
///
/// This struct holds all the information needed to convert between
/// screen, NDC, and world coordinates. It should be constructed
/// once per frame and passed to any code that needs coordinate conversions.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateSystem {
    /// Window width in pixels.
    pub window_width: u32,
    /// Window height in pixels.
    pub window_height: u32,

    /// Chart area width (excluding stats panel).
    pub chart_width: f32,
    /// Chart area height (excluding volume section).
    pub chart_height: f32,
    /// Chart aspect ratio.
    pub aspect: f32,

    /// Camera position in world coordinates.
    pub camera_position: [f32; 2],
    /// Camera scale (half-extents in world units).
    pub camera_scale: [f32; 2],
}

impl CoordinateSystem {
    /// Create a new coordinate system.
    ///
    /// # Arguments
    ///
    /// * `window_width` - Window width in pixels
    /// * `window_height` - Window height in pixels
    /// * `camera_position` - Camera center in world coordinates [x, y]
    /// * `camera_scale` - Camera half-extents [scale_x, scale_y]
    #[must_use]
    pub fn new(
        window_width: u32,
        window_height: u32,
        camera_position: [f32; 2],
        camera_scale: [f32; 2],
    ) -> Self {
        // Use .max(1.0) to prevent division by zero when calculating aspect ratio
        let chart_width = (window_width as f32 - STATS_PANEL_WIDTH).max(1.0);
        let chart_height = (window_height as f32 * (1.0 - VOLUME_HEIGHT_RATIO)).max(1.0);
        let aspect = chart_width / chart_height;

        Self {
            window_width,
            window_height,
            chart_width,
            chart_height,
            aspect,
            camera_position,
            camera_scale,
        }
    }

    /// Update with new camera state.
    pub fn update_camera(&mut self, position: [f32; 2], scale: [f32; 2]) {
        self.camera_position = position;
        self.camera_scale = scale;
    }

    /// Update with new window size.
    pub fn update_size(&mut self, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;
        // Use .max(1.0) to prevent division by zero when calculating aspect ratio
        self.chart_width = (width as f32 - STATS_PANEL_WIDTH).max(1.0);
        self.chart_height = (height as f32 * (1.0 - VOLUME_HEIGHT_RATIO)).max(1.0);
        self.aspect = self.chart_width / self.chart_height;
    }

    // =========================================================================
    // Screen <-> NDC conversions
    // =========================================================================

    /// Convert screen coordinates to NDC.
    ///
    /// **Important:** The screen coordinates must be chart-relative, meaning:
    /// - `screen.x` should be the position within the chart area (0 to chart_width)
    /// - `screen.y` should be the position within the chart area (0 to chart_height)
    ///
    /// If you have window-relative coordinates (e.g., from mouse events), you must
    /// first subtract any offsets. For example, if the stats panel is on the left,
    /// subtract `STATS_PANEL_WIDTH` from the window x coordinate before calling.
    ///
    /// Use [`is_in_chart_area`](Self::is_in_chart_area) to check if a position
    /// is within the valid chart area before conversion.
    #[must_use]
    pub fn screen_to_ndc(&self, screen: ScreenPos) -> NdcPos {
        NdcPos {
            x: (screen.x / self.chart_width) * 2.0 - 1.0,
            y: 1.0 - (screen.y / self.chart_height) * 2.0,
        }
    }

    /// Convert NDC to screen coordinates.
    #[must_use]
    pub fn ndc_to_screen(&self, ndc: NdcPos) -> ScreenPos {
        ScreenPos {
            x: (ndc.x + 1.0) * 0.5 * self.chart_width,
            y: (1.0 - ndc.y) * 0.5 * self.chart_height,
        }
    }

    // =========================================================================
    // NDC <-> World conversions
    // =========================================================================

    /// Convert NDC to world coordinates.
    ///
    /// Note: Returns the raw world X coordinate, not the candle index.
    #[must_use]
    pub fn ndc_to_world(&self, ndc: NdcPos) -> WorldPos {
        let world_x = self.camera_position[0] + ndc.x * self.camera_scale[0] * self.aspect;
        let world_y = self.camera_position[1] + ndc.y * self.camera_scale[1];

        WorldPos::from_world_xy(world_x, world_y)
    }

    /// Convert world coordinates to NDC.
    #[must_use]
    pub fn world_to_ndc(&self, world: WorldPos) -> NdcPos {
        let world_x = world.world_x();
        NdcPos {
            x: (world_x - self.camera_position[0]) / (self.camera_scale[0] * self.aspect),
            y: (world.price - self.camera_position[1]) / self.camera_scale[1],
        }
    }

    // =========================================================================
    // Screen <-> World conversions (most common)
    // =========================================================================

    /// Convert screen coordinates to world coordinates.
    ///
    /// This is the most commonly used conversion - translating mouse positions
    /// to chart coordinates.
    #[must_use]
    pub fn screen_to_world(&self, screen: ScreenPos) -> WorldPos {
        let ndc = self.screen_to_ndc(screen);
        self.ndc_to_world(ndc)
    }

    /// Convert world coordinates to screen coordinates.
    #[must_use]
    pub fn world_to_screen(&self, world: WorldPos) -> ScreenPos {
        let ndc = self.world_to_ndc(world);
        self.ndc_to_screen(ndc)
    }

    // =========================================================================
    // Utility methods
    // =========================================================================

    /// Get the visible X range in world coordinates (as raw X values, not candle indices).
    #[must_use]
    pub fn visible_x_range(&self) -> (f32, f32) {
        let half_width = self.camera_scale[0] * self.aspect;
        (
            self.camera_position[0] - half_width,
            self.camera_position[0] + half_width,
        )
    }

    /// Get the visible Y range in world coordinates.
    #[must_use]
    pub fn visible_y_range(&self) -> (f32, f32) {
        let half_height = self.camera_scale[1];
        (
            self.camera_position[1] - half_height,
            self.camera_position[1] + half_height,
        )
    }

    /// Get the visible candle index range.
    #[must_use]
    pub fn visible_candle_range(&self) -> (f32, f32) {
        let (x_min, x_max) = self.visible_x_range();
        (x_min / CANDLE_SPACING, x_max / CANDLE_SPACING)
    }

    /// Get the visible bounds as a ChartBounds struct.
    #[must_use]
    pub fn visible_bounds(&self) -> ChartBounds {
        let (candle_min, candle_max) = self.visible_candle_range();
        let (price_min, price_max) = self.visible_y_range();
        ChartBounds::new(candle_min, candle_max, price_min, price_max)
    }

    /// Convert a screen delta to world delta.
    ///
    /// Useful for pan operations where you have a pixel delta
    /// and need to know how much to move in world space.
    #[must_use]
    pub fn screen_delta_to_world(&self, dx: f32, dy: f32) -> (f32, f32) {
        // World units per pixel
        let world_per_pixel_x = (self.camera_scale[0] * self.aspect * 2.0) / self.chart_width;
        let world_per_pixel_y = (self.camera_scale[1] * 2.0) / self.chart_height;

        // Note: Y is inverted (screen Y increases downward, world Y increases upward)
        (-dx * world_per_pixel_x, dy * world_per_pixel_y)
    }

    /// Convert a screen delta to candle/price delta.
    ///
    /// Returns (delta_candle_index, delta_price).
    #[must_use]
    pub fn screen_delta_to_candle_price(&self, dx: f32, dy: f32) -> (f32, f32) {
        let (world_dx, world_dy) = self.screen_delta_to_world(dx, dy);
        (world_dx / CANDLE_SPACING, world_dy)
    }

    /// Check if a screen position is within the chart area.
    #[must_use]
    pub fn is_in_chart_area(&self, screen: ScreenPos) -> bool {
        screen.x >= 0.0
            && screen.x < self.chart_width
            && screen.y >= 0.0
            && screen.y < self.chart_height
    }

    /// Check if a screen position is in the volume area.
    #[must_use]
    pub fn is_in_volume_area(&self, screen: ScreenPos) -> bool {
        let volume_top = self.chart_height;
        let volume_bottom = self.window_height as f32;
        screen.x >= 0.0
            && screen.x < self.chart_width
            && screen.y >= volume_top
            && screen.y < volume_bottom
    }

    /// Convert window-relative coordinates to chart-relative coordinates.
    ///
    /// This subtracts the stats panel width from the X coordinate.
    /// Use this to convert raw mouse positions from window events
    /// to the chart-relative coordinates expected by [`screen_to_ndc`](Self::screen_to_ndc).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Mouse event gives window-relative position
    /// let window_pos = ScreenPos::new(mouse_x, mouse_y);
    ///
    /// // Convert to chart-relative coordinates
    /// let chart_pos = coords.window_to_chart_pos(window_pos);
    ///
    /// // Check if in chart area and convert
    /// if coords.is_in_chart_area(chart_pos) {
    ///     let world = coords.screen_to_world(chart_pos);
    /// }
    /// ```
    #[must_use]
    pub fn window_to_chart_pos(&self, window_pos: ScreenPos) -> ScreenPos {
        ScreenPos {
            x: window_pos.x - STATS_PANEL_WIDTH,
            y: window_pos.y,
        }
    }

    /// Check if a window-relative position is within the chart area.
    ///
    /// This is a convenience method that accounts for the stats panel offset.
    /// It checks if the position falls within the chart drawing area.
    #[must_use]
    pub fn is_window_pos_in_chart_area(&self, window_pos: ScreenPos) -> bool {
        window_pos.x >= STATS_PANEL_WIDTH
            && window_pos.x < STATS_PANEL_WIDTH + self.chart_width
            && window_pos.y >= 0.0
            && window_pos.y < self.chart_height
    }

    /// Calculate the number of pixels per candle at the current zoom level.
    #[must_use]
    pub fn pixels_per_candle(&self) -> f32 {
        let (x_min, x_max) = self.visible_x_range();
        let visible_width = x_max - x_min;
        let visible_candles = visible_width / CANDLE_SPACING;
        self.chart_width / visible_candles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_coords() -> CoordinateSystem {
        CoordinateSystem::new(1920, 1080, [100.0, 50000.0], [50.0, 1000.0])
    }

    #[test]
    fn test_screen_pos_creation() {
        let pos = ScreenPos::new(100.0, 200.0);
        assert_eq!(pos.x, 100.0);
        assert_eq!(pos.y, 200.0);

        let from_tuple: ScreenPos = (100.0, 200.0).into();
        assert_eq!(from_tuple, pos);
    }

    #[test]
    fn test_screen_pos_distance() {
        let a = ScreenPos::new(0.0, 0.0);
        let b = ScreenPos::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_ndc_valid() {
        assert!(NdcPos::new(0.0, 0.0).is_valid());
        assert!(NdcPos::new(-1.0, 1.0).is_valid());
        assert!(!NdcPos::new(-1.5, 0.0).is_valid());
    }

    #[test]
    fn test_world_pos_conversion() {
        let world = WorldPos::new(10.0, 100.0);
        let (x, y) = world.to_world_xy();
        assert_eq!(x, 10.0 * CANDLE_SPACING);
        assert_eq!(y, 100.0);

        let back = WorldPos::from_world_xy(x, y);
        assert!((back.candle_index - world.candle_index).abs() < 0.001);
        assert!((back.price - world.price).abs() < 0.001);
    }

    #[test]
    fn test_chart_bounds() {
        let bounds = ChartBounds::new(0.0, 100.0, 1000.0, 2000.0);
        assert_eq!(bounds.width(), 100.0);
        assert_eq!(bounds.height(), 1000.0);

        let center = bounds.center();
        assert_eq!(center.candle_index, 50.0);
        assert_eq!(center.price, 1500.0);

        assert!(bounds.contains(WorldPos::new(50.0, 1500.0)));
        assert!(!bounds.contains(WorldPos::new(-1.0, 1500.0)));
    }

    #[test]
    fn test_screen_to_ndc_center() {
        let coords = test_coords();
        // Center of chart area should be roughly (0, 0) in NDC
        let center_screen = ScreenPos::new(coords.chart_width / 2.0, coords.chart_height / 2.0);
        let ndc = coords.screen_to_ndc(center_screen);
        assert!(ndc.x.abs() < 0.01);
        assert!(ndc.y.abs() < 0.01);
    }

    #[test]
    fn test_screen_ndc_roundtrip() {
        let coords = test_coords();
        let original = ScreenPos::new(500.0, 300.0);
        let ndc = coords.screen_to_ndc(original);
        let back = coords.ndc_to_screen(ndc);
        assert!((original.x - back.x).abs() < 0.01);
        assert!((original.y - back.y).abs() < 0.01);
    }

    #[test]
    fn test_screen_world_roundtrip() {
        let coords = test_coords();
        let original = ScreenPos::new(500.0, 300.0);
        let world = coords.screen_to_world(original);
        let back = coords.world_to_screen(world);
        assert!((original.x - back.x).abs() < 0.01);
        assert!((original.y - back.y).abs() < 0.01);
    }

    #[test]
    fn test_visible_ranges() {
        let coords = test_coords();
        let (x_min, x_max) = coords.visible_x_range();
        assert!(x_min < coords.camera_position[0]);
        assert!(x_max > coords.camera_position[0]);

        let (y_min, y_max) = coords.visible_y_range();
        assert!(y_min < coords.camera_position[1]);
        assert!(y_max > coords.camera_position[1]);
    }

    #[test]
    fn test_is_in_chart_area() {
        let coords = test_coords();
        assert!(coords.is_in_chart_area(ScreenPos::new(100.0, 100.0)));
        assert!(!coords.is_in_chart_area(ScreenPos::new(-10.0, 100.0)));
        assert!(!coords.is_in_chart_area(ScreenPos::new(100.0, coords.chart_height + 10.0)));
    }

    #[test]
    fn test_zero_size_protection() {
        // Test that zero or very small window sizes don't cause division by zero
        let coords = CoordinateSystem::new(0, 0, [100.0, 50000.0], [50.0, 1000.0]);
        assert!(coords.chart_width >= 1.0);
        assert!(coords.chart_height >= 1.0);
        assert!(coords.aspect.is_finite());

        // Test with window size smaller than stats panel width
        let small_coords = CoordinateSystem::new(100, 100, [100.0, 50000.0], [50.0, 1000.0]);
        assert!(small_coords.chart_width >= 1.0);
        assert!(small_coords.chart_height >= 1.0);
        assert!(small_coords.aspect.is_finite());
    }

    #[test]
    fn test_update_size_protection() {
        let mut coords = test_coords();
        coords.update_size(0, 0);
        assert!(coords.chart_width >= 1.0);
        assert!(coords.chart_height >= 1.0);
        assert!(coords.aspect.is_finite());
    }

    #[test]
    fn test_window_to_chart_pos() {
        let coords = test_coords();

        // Position at stats panel edge should map to chart x=0
        let window_pos = ScreenPos::new(STATS_PANEL_WIDTH, 100.0);
        let chart_pos = coords.window_to_chart_pos(window_pos);
        assert!((chart_pos.x - 0.0).abs() < 0.01);
        assert!((chart_pos.y - 100.0).abs() < 0.01);

        // Test that window_to_chart_pos correctly subtracts STATS_PANEL_WIDTH
        // When STATS_PANEL_WIDTH is 0, window and chart coordinates are the same for x
        let window_pos2 = ScreenPos::new(50.0, 100.0);
        let chart_pos2 = coords.window_to_chart_pos(window_pos2);
        assert!((chart_pos2.x - (50.0 - STATS_PANEL_WIDTH)).abs() < 0.01);
    }

    #[test]
    fn test_is_window_pos_in_chart_area() {
        let coords = test_coords();

        // Position within chart area should return true
        assert!(coords.is_window_pos_in_chart_area(ScreenPos::new(
            STATS_PANEL_WIDTH + 10.0,
            100.0
        )));

        // Position at negative x should not be in chart area
        assert!(!coords.is_window_pos_in_chart_area(ScreenPos::new(-10.0, 100.0)));

        // Position below chart (in volume area) should not be in chart area
        assert!(!coords.is_window_pos_in_chart_area(ScreenPos::new(
            STATS_PANEL_WIDTH + 10.0,
            coords.chart_height + 10.0
        )));

        // Position beyond chart width should not be in chart area
        assert!(!coords.is_window_pos_in_chart_area(ScreenPos::new(
            STATS_PANEL_WIDTH + coords.chart_width + 10.0,
            100.0
        )));
    }
}
