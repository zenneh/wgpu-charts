//! View and camera state.
//!
//! This module contains state related to the current view of the chart:
//! camera position, zoom level, and visible range calculations.

use charter_render::CANDLE_SPACING;

/// Screen coordinates (pixels from top-left).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScreenPos {
    pub x: f32,
    pub y: f32,
}

impl ScreenPos {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// World/Chart coordinates (candle index * spacing, price).
#[derive(Debug, Clone, Copy, Default)]
pub struct WorldPos {
    pub x: f32,
    pub y: f32,
}

impl WorldPos {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Get the candle index from world X coordinate.
    pub fn candle_index(&self) -> f32 {
        self.x / CANDLE_SPACING
    }

    /// Get the price from world Y coordinate.
    pub fn price(&self) -> f32 {
        self.y
    }
}

/// View state for the chart display.
///
/// This struct tracks the camera state and visible range,
/// enabling coordinate transformations between screen and world space.
#[derive(Debug, Clone, Default)]
pub struct ViewState {
    /// The current visible start index (first visible candle).
    pub visible_start: u32,

    /// The current visible count (number of visible candles).
    pub visible_count: u32,

    /// Current Level of Detail factor (1 = full detail).
    pub current_lod_factor: u32,

    /// Cached guideline values for rendering.
    pub guideline_values: Vec<f32>,

    /// Number of guidelines to render.
    pub guideline_count: u32,
}

impl ViewState {
    /// Create a new view state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the visible candle range as (start, end).
    pub fn visible_range(&self) -> (usize, usize) {
        let start = self.visible_start as usize;
        let end = start.saturating_add(self.visible_count as usize);
        (start, end)
    }

    /// Check if a candle index is within the visible range.
    pub fn is_visible(&self, index: usize) -> bool {
        let (start, end) = self.visible_range();
        index >= start && index < end
    }
}

/// Coordinate system utilities for screen/world transformations.
///
/// This provides a central place for all coordinate transformations,
/// avoiding scattered conversion logic throughout the codebase.
pub struct CoordinateSystem {
    /// Screen width in pixels.
    pub screen_width: f32,
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Chart area width (excluding stats panel).
    pub chart_width: f32,
    /// Chart area height (excluding volume section).
    pub chart_height: f32,
    /// Camera position in world coordinates.
    pub camera_position: [f32; 2],
    /// Camera scale (zoom level).
    pub camera_scale: [f32; 2],
}

impl CoordinateSystem {
    /// Create a new coordinate system from current state.
    pub fn new(
        screen_width: f32,
        screen_height: f32,
        stats_panel_width: f32,
        volume_height_ratio: f32,
        camera_position: [f32; 2],
        camera_scale: [f32; 2],
    ) -> Self {
        let chart_width = (screen_width - stats_panel_width).max(1.0);
        let chart_height = (screen_height * (1.0 - volume_height_ratio)).max(1.0);

        Self {
            screen_width,
            screen_height,
            chart_width,
            chart_height,
            camera_position,
            camera_scale,
        }
    }

    /// Get the aspect ratio of the chart area.
    pub fn aspect(&self) -> f32 {
        self.chart_width / self.chart_height
    }

    /// Convert screen coordinates to world coordinates.
    pub fn screen_to_world(&self, screen: ScreenPos) -> WorldPos {
        let aspect = self.aspect();

        // Calculate position in NDC relative to the chart area
        let ndc_x = (screen.x / self.chart_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen.y / self.chart_height) * 2.0;

        // Convert from NDC to world coordinates
        let world_x = self.camera_position[0] + ndc_x * self.camera_scale[0] * aspect;
        let world_y = self.camera_position[1] + ndc_y * self.camera_scale[1];

        WorldPos::new(world_x, world_y)
    }

    /// Convert world coordinates to screen coordinates.
    pub fn world_to_screen(&self, world: WorldPos) -> ScreenPos {
        let aspect = self.aspect();

        // Convert from world to NDC
        let ndc_x = (world.x - self.camera_position[0]) / (self.camera_scale[0] * aspect);
        let ndc_y = (world.y - self.camera_position[1]) / self.camera_scale[1];

        // Convert from NDC to screen coordinates
        let screen_x = (ndc_x + 1.0) * 0.5 * self.chart_width;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.chart_height;

        ScreenPos::new(screen_x, screen_y)
    }

    /// Get the visible X range in world coordinates.
    pub fn visible_x_range(&self) -> (f32, f32) {
        let aspect = self.aspect();
        let half_width = self.camera_scale[0] * aspect;
        let x_min = self.camera_position[0] - half_width;
        let x_max = self.camera_position[0] + half_width;
        (x_min, x_max)
    }

    /// Get the visible Y range in world coordinates.
    pub fn visible_y_range(&self) -> (f32, f32) {
        let y_min = self.camera_position[1] - self.camera_scale[1];
        let y_max = self.camera_position[1] + self.camera_scale[1];
        (y_min, y_max)
    }

    /// Check if screen position is within the chart area.
    pub fn is_in_chart_area(&self, screen: ScreenPos) -> bool {
        screen.x >= 0.0 && screen.x < self.chart_width && screen.y >= 0.0 && screen.y < self.chart_height
    }

    /// Get the candle index at a screen X position.
    pub fn candle_index_at_screen_x(&self, screen_x: f32) -> f32 {
        let world = self.screen_to_world(ScreenPos::new(screen_x, 0.0));
        world.candle_index()
    }
}
