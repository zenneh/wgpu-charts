//! Event and command type definitions.
//!
//! This module defines the core event types used throughout the Charter application:
//! - [`InputEvent`] - Raw input events from the windowing system
//! - [`InputMode`] - Input modes that affect event interpretation
//! - [`AppEvent`] - Semantic application-level events
//! - [`Command`] - State mutation commands

use crate::drawing::{Drawing, DrawingId, DrawingTool};

/// Input modes that change how keyboard and mouse events are interpreted.
///
/// Different modes interpret the same input differently:
/// - In **Normal** mode, 'H' enters HorizontalRay drawing mode
/// - In **Drawing** mode, 'H' switches to HorizontalRay tool
/// - In **Replay** mode, 'H' is ignored
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    /// Default viewing and panning mode.
    #[default]
    Normal,

    /// Drawing tool is active, placing a new annotation.
    Drawing(DrawingTool),

    /// Selection mode for editing existing drawings.
    Selecting,

    /// Replay mode - historical playback controls are active.
    Replay,

    /// Symbol picker modal dialog is open.
    SymbolPicker,
}

impl InputMode {
    /// Check if this mode is a drawing mode.
    #[must_use]
    pub fn is_drawing(&self) -> bool {
        matches!(self, InputMode::Drawing(_))
    }

    /// Check if this mode blocks panning.
    ///
    /// Some modes (like Drawing) may allow panning, while others (like SymbolPicker)
    /// should block all chart interaction.
    #[must_use]
    pub fn blocks_panning(&self) -> bool {
        matches!(self, InputMode::SymbolPicker)
    }

    /// Check if this mode consumes mouse clicks.
    ///
    /// When true, mouse clicks should be handled by the mode-specific handler
    /// rather than the default panning behavior.
    #[must_use]
    pub fn consumes_clicks(&self) -> bool {
        matches!(
            self,
            InputMode::Drawing(_) | InputMode::Selecting | InputMode::SymbolPicker
        )
    }

    /// Get the drawing tool if in drawing mode.
    #[must_use]
    pub fn drawing_tool(&self) -> Option<DrawingTool> {
        match self {
            InputMode::Drawing(tool) => Some(*tool),
            _ => None,
        }
    }

    /// Get a human-readable name for this mode.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            InputMode::Normal => "Normal",
            InputMode::Drawing(tool) => tool.name(),
            InputMode::Selecting => "Select",
            InputMode::Replay => "Replay",
            InputMode::SymbolPicker => "Symbol Picker",
        }
    }
}

// Re-export coordinate types for event payloads
pub use crate::coords::{ScreenPos, WorldPos};

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl From<winit::event::MouseButton> for MouseButton {
    fn from(button: winit::event::MouseButton) -> Self {
        match button {
            winit::event::MouseButton::Left => MouseButton::Left,
            winit::event::MouseButton::Right => MouseButton::Right,
            winit::event::MouseButton::Middle => MouseButton::Middle,
            _ => MouseButton::Left, // Default fallback
        }
    }
}

/// Keyboard modifier state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

impl Modifiers {
    /// Check if any modifier is pressed.
    #[must_use]
    pub fn any(&self) -> bool {
        self.shift || self.ctrl || self.alt || self.meta
    }

    /// Check if no modifiers are pressed.
    #[must_use]
    pub fn none(&self) -> bool {
        !self.any()
    }
}

impl From<winit::keyboard::ModifiersState> for Modifiers {
    fn from(state: winit::keyboard::ModifiersState) -> Self {
        Self {
            shift: state.shift_key(),
            ctrl: state.control_key(),
            alt: state.alt_key(),
            meta: state.super_key(),
        }
    }
}

/// Raw input events from the windowing system.
///
/// These events represent low-level input that hasn't been interpreted
/// in the context of the current application mode.
#[derive(Debug, Clone)]
pub enum InputEvent {
    /// A key was pressed.
    KeyPressed {
        key: winit::keyboard::KeyCode,
        modifiers: Modifiers,
    },

    /// A key was released.
    KeyReleased {
        key: winit::keyboard::KeyCode,
        modifiers: Modifiers,
    },

    /// A mouse button was pressed.
    MousePressed {
        button: MouseButton,
        position: ScreenPos,
        modifiers: Modifiers,
    },

    /// A mouse button was released.
    MouseReleased {
        button: MouseButton,
        position: ScreenPos,
        modifiers: Modifiers,
    },

    /// The mouse cursor moved.
    MouseMoved {
        position: ScreenPos,
        modifiers: Modifiers,
    },

    /// The mouse wheel was scrolled.
    MouseWheel {
        delta_x: f32,
        delta_y: f32,
        position: ScreenPos,
        modifiers: Modifiers,
    },

    /// The mouse cursor entered the window.
    CursorEntered,

    /// The mouse cursor left the window.
    CursorLeft,
}

/// UI panel identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Panel {
    /// Technical analysis overlay panel.
    TaOverlay,
    /// MACD indicator panel.
    MacdPanel,
    /// Symbol picker modal.
    SymbolPicker,
    /// Drawing toolbar.
    DrawingToolbar,
}

/// View-related events (camera, zoom, pan).
#[derive(Debug, Clone)]
pub enum ViewEvent {
    /// Fit the view to show all data.
    FitView,

    /// Pan the view by screen delta.
    Pan {
        delta_x: f32,
        delta_y: f32,
    },

    /// Zoom the view.
    Zoom {
        /// Zoom factor (positive = zoom in, negative = zoom out).
        factor_x: f32,
        factor_y: f32,
        /// Center point for zoom in screen coordinates.
        center: ScreenPos,
    },

    /// Reset zoom to default level.
    ResetZoom,
}

/// Mode change events.
#[derive(Debug, Clone)]
pub enum ModeEvent {
    /// Enter a new input mode.
    Enter(InputMode),

    /// Exit the current mode and return to Normal.
    Exit,

    /// Toggle between modes (e.g., Normal <-> Replay).
    Toggle(InputMode),
}

/// Drawing-related events.
#[derive(Debug, Clone)]
pub enum DrawingEvent {
    /// Select a drawing tool.
    ToolSelected(DrawingTool),

    /// Toggle snap-to-OHLC.
    ToggleSnap,

    /// A drawing operation has started.
    Started {
        tool: DrawingTool,
        position: WorldPos,
    },

    /// A drawing in progress was updated (mouse moved).
    Updated {
        position: WorldPos,
    },

    /// A drawing was completed successfully.
    Completed,

    /// A drawing operation was cancelled.
    Cancelled,

    /// A drawing was selected.
    Selected(Option<DrawingId>),

    /// A drawing was deleted.
    Deleted(DrawingId),

    /// Request to delete the currently selected drawing.
    DeleteSelected,

    /// Anchor drag started.
    AnchorDragStarted {
        drawing_id: DrawingId,
        anchor_index: usize,
    },

    /// Drawing body drag started.
    BodyDragStarted {
        drawing_id: DrawingId,
    },

    /// Drag operation ended.
    DragEnded,
}

/// Data-related events.
#[derive(Debug, Clone)]
pub enum DataEvent {
    /// Switch to a different timeframe by index.
    TimeframeChanged(usize),

    /// Request to change the symbol.
    SymbolChangeRequested(String),

    /// Symbol was successfully changed.
    SymbolChanged(String),

    /// Start live data updates.
    StartLiveUpdates,

    /// Stop live data updates.
    StopLiveUpdates,

    /// Live connection status changed.
    ConnectionStatusChanged(bool),

    /// New candle data received.
    CandleReceived {
        is_closed: bool,
    },
}

/// Replay mode events.
#[derive(Debug, Clone)]
pub enum ReplayEvent {
    /// Toggle replay mode on/off.
    Toggle,

    /// Step forward by the current step size.
    StepForward,

    /// Step backward by the current step size.
    StepBackward,

    /// Increase the step size.
    IncreaseStep,

    /// Decrease the step size.
    DecreaseStep,

    /// Set the replay index directly (e.g., from a click).
    SetIndex(usize),

    /// Clear the replay index (return to cursor-following).
    ClearIndex,
}

/// UI-related events.
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// Toggle a panel's visibility.
    PanelToggled(Panel),

    /// Request a UI refresh.
    Refresh,
}

/// Semantic application events.
///
/// These events represent high-level actions that have been interpreted
/// from raw input based on the current application mode and context.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// View-related events (camera, zoom, pan).
    View(ViewEvent),

    /// Mode change events.
    Mode(ModeEvent),

    /// Drawing-related events.
    Drawing(DrawingEvent),

    /// Data-related events.
    Data(DataEvent),

    /// Replay mode events.
    Replay(ReplayEvent),

    /// UI-related events.
    Ui(UiEvent),

    /// Request application exit.
    Quit,

    /// No-op event (useful for escape in normal mode).
    None,
}

// Convenience constructors for AppEvent
impl AppEvent {
    /// Create a FitView event.
    #[must_use]
    pub fn fit_view() -> Self {
        AppEvent::View(ViewEvent::FitView)
    }

    /// Create a Pan event.
    #[must_use]
    pub fn pan(delta_x: f32, delta_y: f32) -> Self {
        AppEvent::View(ViewEvent::Pan { delta_x, delta_y })
    }

    /// Create a Zoom event.
    #[must_use]
    pub fn zoom(factor_x: f32, factor_y: f32, center: ScreenPos) -> Self {
        AppEvent::View(ViewEvent::Zoom {
            factor_x,
            factor_y,
            center,
        })
    }

    /// Create a mode enter event.
    #[must_use]
    pub fn enter_mode(mode: InputMode) -> Self {
        AppEvent::Mode(ModeEvent::Enter(mode))
    }

    /// Create a mode exit event.
    #[must_use]
    pub fn exit_mode() -> Self {
        AppEvent::Mode(ModeEvent::Exit)
    }

    /// Create a tool selection event.
    #[must_use]
    pub fn select_tool(tool: DrawingTool) -> Self {
        AppEvent::Drawing(DrawingEvent::ToolSelected(tool))
    }

    /// Create a drawing cancelled event.
    #[must_use]
    pub fn cancel_drawing() -> Self {
        AppEvent::Drawing(DrawingEvent::Cancelled)
    }

    /// Create a panel toggle event.
    #[must_use]
    pub fn toggle_panel(panel: Panel) -> Self {
        AppEvent::Ui(UiEvent::PanelToggled(panel))
    }

    /// Create a timeframe change event.
    #[must_use]
    pub fn switch_timeframe(index: usize) -> Self {
        AppEvent::Data(DataEvent::TimeframeChanged(index))
    }

    /// Create a replay toggle event.
    #[must_use]
    pub fn toggle_replay() -> Self {
        AppEvent::Replay(ReplayEvent::Toggle)
    }
}

/// Camera update specification.
#[derive(Debug, Clone)]
pub enum CameraUpdate {
    /// Set camera position directly.
    SetPosition { x: f32, y: f32 },

    /// Adjust camera position by delta.
    AdjustPosition { dx: f32, dy: f32 },

    /// Set camera scale directly.
    SetScale { x: f32, y: f32 },

    /// Adjust camera scale by factor.
    AdjustScale {
        factor_x: f32,
        factor_y: f32,
        /// Optional center point for zoom (in world coordinates).
        center: Option<WorldPos>,
    },

    /// Fit the camera to show all data.
    FitToData,

    /// Fit the camera to a specific range.
    FitToRange {
        x_min: f32,
        x_max: f32,
        y_min: f32,
        y_max: f32,
    },
}

/// Update specification for a drawing.
#[derive(Debug, Clone)]
pub enum DrawingUpdate {
    /// Move the entire drawing by a delta.
    Translate { dx: f32, dy: f32 },

    /// Move a specific anchor to a new position.
    MoveAnchor {
        anchor_index: usize,
        new_position: WorldPos,
    },

    /// Change the drawing's color.
    SetColor { color: [f32; 4] },
}

/// Replay state specification.
#[derive(Debug, Clone)]
pub struct ReplayState {
    pub enabled: bool,
    pub index: Option<usize>,
    pub timestamp: Option<f64>,
}

/// Commands that modify application state.
///
/// Commands are the final step in the event pipeline. They represent
/// concrete state mutations that should be applied atomically.
#[derive(Debug, Clone)]
pub enum Command {
    /// Update the camera (position, scale, etc.).
    UpdateCamera(CameraUpdate),

    /// Set the active drawing tool.
    SetDrawingTool(DrawingTool),

    /// Add a new drawing.
    AddDrawing(Drawing),

    /// Remove a drawing by ID.
    RemoveDrawing(DrawingId),

    /// Update an existing drawing.
    UpdateDrawing {
        id: DrawingId,
        update: DrawingUpdate,
    },

    /// Select a drawing (or deselect with None).
    SelectDrawing(Option<DrawingId>),

    /// Set the replay state.
    SetReplayState(ReplayState),

    /// Switch to a timeframe by index.
    SwitchTimeframe(usize),

    /// Request a redraw of the window.
    RequestRedraw,

    /// Update the visible range based on current camera.
    UpdateVisibleRange,

    /// Recompute technical analysis data.
    RecomputeTa,

    /// Show an error message to the user.
    ShowError(String),

    /// Batch multiple commands together.
    Batch(Vec<Command>),
}

impl Command {
    /// Create a batch of commands.
    #[must_use]
    pub fn batch(commands: impl IntoIterator<Item = Command>) -> Self {
        Command::Batch(commands.into_iter().collect())
    }

    /// Check if this command requires a redraw.
    #[must_use]
    pub fn requires_redraw(&self) -> bool {
        match self {
            Command::UpdateCamera(_) => true,
            Command::SetDrawingTool(_) => true,
            Command::AddDrawing(_) => true,
            Command::RemoveDrawing(_) => true,
            Command::UpdateDrawing { .. } => true,
            Command::SelectDrawing(_) => true,
            Command::SetReplayState(_) => true,
            Command::SwitchTimeframe(_) => true,
            Command::RequestRedraw => true,
            Command::UpdateVisibleRange => true,
            Command::RecomputeTa => true,
            Command::ShowError(_) => false,
            Command::Batch(cmds) => cmds.iter().any(|c| c.requires_redraw()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modifiers_any() {
        let empty = Modifiers::default();
        assert!(!empty.any());
        assert!(empty.none());

        let with_shift = Modifiers {
            shift: true,
            ..Default::default()
        };
        assert!(with_shift.any());
        assert!(!with_shift.none());
    }

    #[test]
    fn test_app_event_constructors() {
        let fit = AppEvent::fit_view();
        assert!(matches!(fit, AppEvent::View(ViewEvent::FitView)));

        let pan = AppEvent::pan(10.0, 20.0);
        assert!(matches!(
            pan,
            AppEvent::View(ViewEvent::Pan {
                delta_x: 10.0,
                delta_y: 20.0
            })
        ));
    }

    #[test]
    fn test_command_requires_redraw() {
        assert!(Command::RequestRedraw.requires_redraw());
        assert!(!Command::ShowError("test".into()).requires_redraw());

        let batch = Command::batch([Command::RequestRedraw, Command::ShowError("test".into())]);
        assert!(batch.requires_redraw());

        let empty_batch = Command::batch([Command::ShowError("a".into())]);
        assert!(!empty_batch.requires_redraw());
    }
}
