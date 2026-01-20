//! Drawing types for interactive chart annotations.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique drawing IDs.
static NEXT_DRAWING_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrawingId(u64);

impl DrawingId {
    /// Generate a new unique drawing ID.
    pub fn new() -> Self {
        Self(NEXT_DRAWING_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for DrawingId {
    fn default() -> Self {
        Self::new()
    }
}

/// An anchor point for a drawing, positioned in chart coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnchorPoint {
    /// X position in candle index units (fractional).
    pub candle_index: f32,
    /// Y position in price units.
    pub price: f32,
    /// Whether this point was snapped to OHLC.
    pub snapped: bool,
}

impl AnchorPoint {
    /// Create a new anchor point.
    pub fn new(candle_index: f32, price: f32) -> Self {
        Self {
            candle_index,
            price,
            snapped: false,
        }
    }

    /// Create a new snapped anchor point.
    pub fn new_snapped(candle_index: f32, price: f32) -> Self {
        Self {
            candle_index,
            price,
            snapped: true,
        }
    }

    /// Convert to world coordinates given candle spacing.
    pub fn to_world(&self, candle_spacing: f32) -> (f32, f32) {
        (self.candle_index * candle_spacing, self.price)
    }
}

/// Default color for drawings (cyan/teal).
pub const DEFAULT_DRAWING_COLOR: [f32; 4] = [0.0, 0.8, 0.8, 1.0];
/// Preview color (more transparent).
pub const PREVIEW_DRAWING_COLOR: [f32; 4] = [0.0, 0.8, 0.8, 0.5];

/// A horizontal ray drawing - extends from anchor to right edge.
#[derive(Debug, Clone)]
pub struct HorizontalRay {
    pub id: DrawingId,
    pub anchor: AnchorPoint,
    pub color: [f32; 4],
}

impl HorizontalRay {
    pub fn new(anchor: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            anchor,
            color: DEFAULT_DRAWING_COLOR,
        }
    }

    pub fn preview(anchor: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            anchor,
            color: PREVIEW_DRAWING_COLOR,
        }
    }
}

/// A ray/trendline drawing - connects two points and extends to right edge.
#[derive(Debug, Clone)]
pub struct Ray {
    pub id: DrawingId,
    pub start: AnchorPoint,
    pub end: AnchorPoint,
    pub color: [f32; 4],
}

impl Ray {
    pub fn new(start: AnchorPoint, end: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            start,
            end,
            color: DEFAULT_DRAWING_COLOR,
        }
    }

    pub fn preview(start: AnchorPoint, end: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            start,
            end,
            color: PREVIEW_DRAWING_COLOR,
        }
    }
}

/// A rectangle drawing - defined by two opposite corners.
#[derive(Debug, Clone)]
pub struct Rectangle {
    pub id: DrawingId,
    pub corner1: AnchorPoint,
    pub corner2: AnchorPoint,
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
}

impl Rectangle {
    pub fn new(corner1: AnchorPoint, corner2: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            corner1,
            corner2,
            fill_color: [0.0, 0.8, 0.8, 0.15],
            border_color: DEFAULT_DRAWING_COLOR,
        }
    }

    pub fn preview(corner1: AnchorPoint, corner2: AnchorPoint) -> Self {
        Self {
            id: DrawingId::new(),
            corner1,
            corner2,
            fill_color: [0.0, 0.8, 0.8, 0.08],
            border_color: PREVIEW_DRAWING_COLOR,
        }
    }

    pub fn x_min(&self) -> f32 {
        self.corner1.candle_index.min(self.corner2.candle_index)
    }

    pub fn x_max(&self) -> f32 {
        self.corner1.candle_index.max(self.corner2.candle_index)
    }

    pub fn y_min(&self) -> f32 {
        self.corner1.price.min(self.corner2.price)
    }

    pub fn y_max(&self) -> f32 {
        self.corner1.price.max(self.corner2.price)
    }
}

/// Enum wrapping all drawing types.
#[derive(Debug, Clone)]
pub enum Drawing {
    HorizontalRay(HorizontalRay),
    Ray(Ray),
    Rectangle(Rectangle),
}

impl Drawing {
    pub fn id(&self) -> DrawingId {
        match self {
            Drawing::HorizontalRay(d) => d.id,
            Drawing::Ray(d) => d.id,
            Drawing::Rectangle(d) => d.id,
        }
    }

    /// Get all anchor points for this drawing.
    pub fn anchors(&self) -> Vec<&AnchorPoint> {
        match self {
            Drawing::HorizontalRay(d) => vec![&d.anchor],
            Drawing::Ray(d) => vec![&d.start, &d.end],
            Drawing::Rectangle(d) => vec![&d.corner1, &d.corner2],
        }
    }

    /// Get mutable anchor by index.
    pub fn anchor_mut(&mut self, index: usize) -> Option<&mut AnchorPoint> {
        match self {
            Drawing::HorizontalRay(d) => {
                if index == 0 {
                    Some(&mut d.anchor)
                } else {
                    None
                }
            }
            Drawing::Ray(d) => match index {
                0 => Some(&mut d.start),
                1 => Some(&mut d.end),
                _ => None,
            },
            Drawing::Rectangle(d) => match index {
                0 => Some(&mut d.corner1),
                1 => Some(&mut d.corner2),
                _ => None,
            },
        }
    }

    /// Move all anchors by the given delta.
    pub fn translate(&mut self, dx: f32, dy: f32) {
        match self {
            Drawing::HorizontalRay(d) => {
                d.anchor.candle_index += dx;
                d.anchor.price += dy;
            }
            Drawing::Ray(d) => {
                d.start.candle_index += dx;
                d.start.price += dy;
                d.end.candle_index += dx;
                d.end.price += dy;
            }
            Drawing::Rectangle(d) => {
                d.corner1.candle_index += dx;
                d.corner1.price += dy;
                d.corner2.candle_index += dx;
                d.corner2.price += dy;
            }
        }
    }

    /// Get the color of this drawing.
    pub fn color(&self) -> [f32; 4] {
        match self {
            Drawing::HorizontalRay(d) => d.color,
            Drawing::Ray(d) => d.color,
            Drawing::Rectangle(d) => d.border_color,
        }
    }
}
