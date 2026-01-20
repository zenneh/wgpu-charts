//! Drawing state management.

use charter_core::Candle;

use super::types::{AnchorPoint, Drawing, DrawingId, HorizontalRay, Ray, Rectangle};

/// Available drawing tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DrawingTool {
    #[default]
    None,
    Select,
    HorizontalRay,
    Ray,
    Rectangle,
}

impl DrawingTool {
    /// Get the display name for this tool.
    pub fn name(&self) -> &'static str {
        match self {
            DrawingTool::None => "None",
            DrawingTool::Select => "Select",
            DrawingTool::HorizontalRay => "H-Ray",
            DrawingTool::Ray => "Ray",
            DrawingTool::Rectangle => "Box",
        }
    }

    /// Get the keyboard shortcut for this tool.
    pub fn shortcut(&self) -> &'static str {
        match self {
            DrawingTool::None => "",
            DrawingTool::Select => "V",
            DrawingTool::HorizontalRay => "H",
            DrawingTool::Ray => "T",
            DrawingTool::Rectangle => "B",
        }
    }

    /// Check if this tool creates drawings.
    pub fn is_drawing_tool(&self) -> bool {
        matches!(self, DrawingTool::HorizontalRay | DrawingTool::Ray | DrawingTool::Rectangle)
    }

    /// Get all available tools.
    pub fn all() -> &'static [DrawingTool] {
        &[
            DrawingTool::Select,
            DrawingTool::HorizontalRay,
            DrawingTool::Ray,
            DrawingTool::Rectangle,
        ]
    }
}

/// Current interaction state.
#[derive(Debug, Clone, Default)]
pub enum InteractionState {
    #[default]
    Idle,
    /// Placing anchors for a new drawing.
    Drawing {
        first_anchor: Option<AnchorPoint>,
    },
    /// Dragging a single anchor point.
    DraggingAnchor {
        drawing_id: DrawingId,
        anchor_index: usize,
    },
    /// Dragging an entire drawing.
    DraggingDrawing {
        drawing_id: DrawingId,
        last_pos: AnchorPoint,
    },
}

/// Index of hovered anchor.
pub type HoveredAnchor = (DrawingId, usize);

/// Manager for all drawing state and operations.
#[derive(Debug, Default)]
pub struct DrawingManager {
    /// Currently selected drawing tool.
    pub tool: DrawingTool,
    /// Current interaction state.
    pub interaction: InteractionState,
    /// Whether snap to OHLC is enabled.
    pub snap_enabled: bool,
    /// All completed drawings.
    pub drawings: Vec<Drawing>,
    /// Currently selected drawing.
    pub selected: Option<DrawingId>,
    /// Currently hovered anchor.
    pub hovered_anchor: Option<HoveredAnchor>,
    /// Current cursor position in chart coordinates.
    pub cursor_pos: Option<AnchorPoint>,
}

impl DrawingManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the active tool.
    pub fn set_tool(&mut self, tool: DrawingTool) {
        // Clear any in-progress drawing
        self.interaction = InteractionState::Idle;
        self.tool = tool;

        // Clear selection when switching away from Select tool
        if tool != DrawingTool::Select {
            self.selected = None;
            self.hovered_anchor = None;
        }
    }

    /// Toggle snap to OHLC.
    pub fn toggle_snap(&mut self) {
        self.snap_enabled = !self.snap_enabled;
    }

    /// Cancel the current operation.
    pub fn cancel(&mut self) {
        self.interaction = InteractionState::Idle;
        self.selected = None;
    }

    /// Delete the selected drawing.
    pub fn delete_selected(&mut self) {
        if let Some(id) = self.selected.take() {
            self.drawings.retain(|d| d.id() != id);
        }
    }

    /// Compute snap target for a world position.
    pub fn snap_target(&self, world_x: f32, world_y: f32, candles: &[Candle], candle_spacing: f32) -> AnchorPoint {
        let candle_index = world_x / candle_spacing;

        // Early return for negative indices or when snap is disabled
        if !self.snap_enabled || candle_index < 0.0 {
            return AnchorPoint::new(candle_index, world_y);
        }

        // Round to nearest candle (not floor) for better UX when cursor is between candles
        let idx = candle_index.round() as usize;
        if let Some(candle) = candles.get(idx) {
            let ohlc = [candle.open, candle.high, candle.low, candle.close];
            let closest = ohlc.iter()
                .min_by(|a, b| ((*a - world_y).abs()).partial_cmp(&((*b - world_y).abs())).unwrap())
                .copied()
                .unwrap_or(world_y);
            // Snap to candle center (candle N has center at x = N * spacing)
            AnchorPoint::new_snapped(idx as f32, closest)
        } else {
            AnchorPoint::new(candle_index, world_y)
        }
    }

    /// Update cursor position and hover state.
    pub fn update_cursor(&mut self, world_x: f32, world_y: f32, candles: &[Candle], candle_spacing: f32) {
        self.cursor_pos = Some(self.snap_target(world_x, world_y, candles, candle_spacing));

        // Update hover state only in Select mode when not dragging
        if self.tool == DrawingTool::Select {
            if let InteractionState::Idle = self.interaction {
                self.update_hover(world_x, world_y, candle_spacing);
            }
        } else {
            self.hovered_anchor = None;
        }
    }

    /// Find if cursor is hovering over an anchor.
    fn update_hover(&mut self, world_x: f32, world_y: f32, candle_spacing: f32) {
        const ANCHOR_HIT_RADIUS_X: f32 = 0.8;
        const ANCHOR_HIT_RADIUS_Y: f32 = 15.0;

        self.hovered_anchor = None;

        // Only check selected drawing's anchors, or all if nothing selected
        let drawings_to_check: Vec<_> = if let Some(selected_id) = self.selected {
            self.drawings.iter().filter(|d| d.id() == selected_id).collect()
        } else {
            self.drawings.iter().collect()
        };

        for drawing in drawings_to_check {
            for (idx, anchor) in drawing.anchors().iter().enumerate() {
                let ax = anchor.candle_index * candle_spacing;
                let ay = anchor.price;
                let dx = (world_x - ax).abs();
                let dy = (world_y - ay).abs();

                if dx < ANCHOR_HIT_RADIUS_X * candle_spacing && dy < ANCHOR_HIT_RADIUS_Y {
                    self.hovered_anchor = Some((drawing.id(), idx));
                    return;
                }
            }
        }
    }

    /// Handle mouse press.
    pub fn handle_press(&mut self, world_x: f32, world_y: f32, candles: &[Candle], candle_spacing: f32) -> bool {
        let anchor = self.snap_target(world_x, world_y, candles, candle_spacing);

        match self.tool {
            DrawingTool::None => false,

            DrawingTool::Select => {
                // Check if clicking on a hovered anchor
                if let Some((drawing_id, anchor_idx)) = self.hovered_anchor {
                    self.selected = Some(drawing_id);
                    self.interaction = InteractionState::DraggingAnchor {
                        drawing_id,
                        anchor_index: anchor_idx,
                    };
                    return true;
                }

                // Check if clicking on a drawing body
                if let Some(drawing_id) = self.find_drawing_at(world_x, world_y, candle_spacing) {
                    self.selected = Some(drawing_id);
                    self.interaction = InteractionState::DraggingDrawing {
                        drawing_id,
                        last_pos: anchor,
                    };
                    return true;
                }

                // Clicked on nothing - deselect
                self.selected = None;
                false
            }

            DrawingTool::HorizontalRay => {
                // Single-click placement
                let drawing = Drawing::HorizontalRay(HorizontalRay::new(anchor));
                self.drawings.push(drawing);
                true
            }

            DrawingTool::Ray | DrawingTool::Rectangle => {
                match &self.interaction {
                    InteractionState::Idle | InteractionState::Drawing { first_anchor: None } => {
                        // First click - start drawing
                        self.interaction = InteractionState::Drawing {
                            first_anchor: Some(anchor),
                        };
                        true
                    }
                    InteractionState::Drawing { first_anchor: Some(first) } => {
                        // Second click - complete drawing
                        let first = *first;

                        // Validate minimum drawing size to prevent degenerate drawings
                        const MIN_DRAWING_SIZE: f32 = 0.1;
                        let dx = (first.candle_index - anchor.candle_index).abs();
                        let dy = (first.price - anchor.price).abs();

                        if dx < MIN_DRAWING_SIZE && dy < MIN_DRAWING_SIZE {
                            // Too small - cancel the drawing
                            self.interaction = InteractionState::Idle;
                            return false;
                        }

                        let drawing = match self.tool {
                            DrawingTool::Ray => Drawing::Ray(Ray::new(first, anchor)),
                            DrawingTool::Rectangle => Drawing::Rectangle(Rectangle::new(first, anchor)),
                            _ => unreachable!(),
                        };
                        self.drawings.push(drawing);
                        self.interaction = InteractionState::Idle;
                        true
                    }
                    _ => false,
                }
            }
        }
    }

    /// Handle mouse release.
    pub fn handle_release(&mut self) {
        match &self.interaction {
            InteractionState::DraggingAnchor { .. } | InteractionState::DraggingDrawing { .. } => {
                self.interaction = InteractionState::Idle;
            }
            _ => {}
        }
    }

    /// Handle mouse drag.
    pub fn handle_drag(&mut self, world_x: f32, world_y: f32, candles: &[Candle], candle_spacing: f32) -> bool {
        let anchor = self.snap_target(world_x, world_y, candles, candle_spacing);

        match &self.interaction {
            InteractionState::DraggingAnchor { drawing_id, anchor_index } => {
                let drawing_id = *drawing_id;
                let anchor_index = *anchor_index;

                if let Some(drawing) = self.drawings.iter_mut().find(|d| d.id() == drawing_id) {
                    if let Some(a) = drawing.anchor_mut(anchor_index) {
                        *a = anchor;
                        return true;
                    }
                }
            }
            InteractionState::DraggingDrawing { drawing_id, last_pos } => {
                let drawing_id = *drawing_id;
                let dx = anchor.candle_index - last_pos.candle_index;
                let dy = anchor.price - last_pos.price;

                if let Some(drawing) = self.drawings.iter_mut().find(|d| d.id() == drawing_id) {
                    drawing.translate(dx, dy);
                    self.interaction = InteractionState::DraggingDrawing {
                        drawing_id,
                        last_pos: anchor,
                    };
                    return true;
                }
            }
            _ => {}
        }

        false
    }

    /// Find a drawing at the given position.
    fn find_drawing_at(&self, world_x: f32, world_y: f32, candle_spacing: f32) -> Option<DrawingId> {
        const HIT_TOLERANCE: f32 = 8.0;

        for drawing in self.drawings.iter().rev() {
            match drawing {
                Drawing::HorizontalRay(hr) => {
                    let line_x = hr.anchor.candle_index * candle_spacing;
                    if world_x >= line_x && (world_y - hr.anchor.price).abs() < HIT_TOLERANCE {
                        return Some(hr.id);
                    }
                }
                Drawing::Ray(ray) => {
                    let x1 = ray.start.candle_index * candle_spacing;
                    let x2 = ray.end.candle_index * candle_spacing;
                    let y1 = ray.start.price;
                    let y2 = ray.end.price;
                    let x_min = x1.min(x2);
                    let x_max = x1.max(x2);
                    let y_min = y1.min(y2) - HIT_TOLERANCE;
                    let y_max = y1.max(y2) + HIT_TOLERANCE;

                    if world_x >= x_min && world_x <= x_max && world_y >= y_min && world_y <= y_max {
                        return Some(ray.id);
                    }
                }
                Drawing::Rectangle(rect) => {
                    let x_min = rect.x_min() * candle_spacing;
                    let x_max = rect.x_max() * candle_spacing;
                    let y_min = rect.y_min();
                    let y_max = rect.y_max();

                    if world_x >= x_min && world_x <= x_max && world_y >= y_min && world_y <= y_max {
                        return Some(rect.id);
                    }
                }
            }
        }

        None
    }

    /// Check if currently in a drawing/dragging operation.
    pub fn is_interacting(&self) -> bool {
        !matches!(self.interaction, InteractionState::Idle)
    }

    /// Get the preview drawing (if any) for the current drawing operation.
    pub fn preview_drawing(&self) -> Option<Drawing> {
        let cursor = self.cursor_pos?;

        match (&self.tool, &self.interaction) {
            // Show horizontal ray preview at cursor
            (DrawingTool::HorizontalRay, InteractionState::Idle) => {
                Some(Drawing::HorizontalRay(HorizontalRay::preview(cursor)))
            }

            // Show ray/rect preview from first anchor to cursor
            (DrawingTool::Ray, InteractionState::Drawing { first_anchor: Some(first) }) => {
                Some(Drawing::Ray(Ray::preview(*first, cursor)))
            }
            (DrawingTool::Rectangle, InteractionState::Drawing { first_anchor: Some(first) }) => {
                Some(Drawing::Rectangle(Rectangle::preview(*first, cursor)))
            }

            // Show first point indicator for ray/rect before first click
            (DrawingTool::Ray, InteractionState::Idle | InteractionState::Drawing { first_anchor: None }) => {
                Some(Drawing::HorizontalRay(HorizontalRay::preview(cursor)))
            }
            (DrawingTool::Rectangle, InteractionState::Idle | InteractionState::Drawing { first_anchor: None }) => {
                Some(Drawing::HorizontalRay(HorizontalRay::preview(cursor)))
            }

            _ => None,
        }
    }
}
