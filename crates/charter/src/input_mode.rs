//! Input mode system for mode-aware keyboard and mouse handling.
//!
//! This module provides the [`InputContext`] struct for mode-aware key mapping,
//! converting raw key codes into semantic [`AppEvent`]s based on the current mode.
//!
//! The [`InputMode`] enum is defined in the events module and re-exported here
//! for convenience.
//!
//! # Example
//!
//! ```ignore
//! use charter::input_mode::{InputMode, InputContext};
//! use winit::keyboard::KeyCode;
//!
//! let mut ctx = InputContext::new();
//! ctx.set_mode(InputMode::Normal);
//!
//! // In Normal mode, 'F' triggers FitView
//! if let Some(event) = ctx.map_key(KeyCode::KeyF) {
//!     // Handle FitView event
//! }
//!
//! // In Drawing mode, 'F' would be ignored
//! ctx.set_mode(InputMode::Drawing(DrawingTool::HorizontalRay));
//! assert!(ctx.map_key(KeyCode::KeyF).is_none());
//! ```

use winit::keyboard::KeyCode;

use crate::coords::WorldPos;
use crate::drawing::DrawingTool;
use crate::events::{AppEvent, DataEvent, DrawingEvent, ModeEvent, Panel, ReplayEvent, UiEvent, ViewEvent};

// Re-export key types for convenience
pub use crate::events::{InputMode, Modifiers};

/// Context for mode-aware input handling.
///
/// This struct maintains the current input mode and provides methods for
/// mapping raw input events to semantic application events.
#[derive(Debug, Clone)]
pub struct InputContext {
    /// Current input mode.
    pub mode: InputMode,
    /// Current keyboard modifiers.
    pub modifiers: Modifiers,
    /// Current mouse position in world coordinates (if known).
    pub mouse_world_pos: Option<WorldPos>,
    /// Whether a drag operation is in progress.
    pub is_dragging: bool,
    /// Whether the mouse button is currently pressed.
    pub mouse_pressed: bool,
}

impl Default for InputContext {
    fn default() -> Self {
        Self::new()
    }
}

impl InputContext {
    /// Create a new input context in Normal mode.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: InputMode::Normal,
            modifiers: Modifiers::default(),
            mouse_world_pos: None,
            is_dragging: false,
            mouse_pressed: false,
        }
    }

    /// Set the current input mode.
    pub fn set_mode(&mut self, mode: InputMode) {
        // Clean up previous mode state
        self.is_dragging = false;

        // If exiting modal modes, reset mouse state
        if matches!(self.mode, InputMode::SymbolPicker | InputMode::Drawing(_)) {
            self.mouse_pressed = false;
        }

        self.mode = mode;
    }

    /// Update keyboard modifiers.
    pub fn set_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }

    /// Update mouse world position.
    pub fn set_mouse_world_pos(&mut self, pos: Option<WorldPos>) {
        self.mouse_world_pos = pos;
    }

    /// Set drag state.
    pub fn set_dragging(&mut self, dragging: bool) {
        self.is_dragging = dragging;
    }

    /// Set mouse pressed state.
    pub fn set_mouse_pressed(&mut self, pressed: bool) {
        self.mouse_pressed = pressed;
    }

    /// Map a key press to an application event based on the current mode.
    ///
    /// Returns `Some(event)` if the key should trigger an action, or `None`
    /// if the key should be ignored in the current mode.
    #[must_use]
    pub fn map_key(&self, key: KeyCode) -> Option<AppEvent> {
        // Global shortcuts (work in any mode)
        if key == KeyCode::Escape {
            return Some(self.escape_action());
        }

        // Mode-specific shortcuts
        match self.mode {
            InputMode::Normal => self.map_key_normal(key),
            InputMode::Drawing(_) => self.map_key_drawing(key),
            InputMode::Selecting => self.map_key_selecting(key),
            InputMode::Replay => self.map_key_replay(key),
            InputMode::SymbolPicker => self.map_key_symbol_picker(key),
        }
    }

    /// Get the action for the Escape key based on current mode.
    fn escape_action(&self) -> AppEvent {
        match self.mode {
            InputMode::SymbolPicker => AppEvent::Ui(UiEvent::PanelToggled(Panel::SymbolPicker)),
            InputMode::Drawing(_) => AppEvent::Drawing(DrawingEvent::Cancelled),
            InputMode::Selecting => AppEvent::Mode(ModeEvent::Exit),
            InputMode::Replay => AppEvent::Replay(ReplayEvent::Toggle),
            InputMode::Normal => AppEvent::None, // No-op in normal mode
        }
    }

    /// Map keys in Normal mode.
    fn map_key_normal(&self, key: KeyCode) -> Option<AppEvent> {
        match key {
            // View controls
            KeyCode::KeyF | KeyCode::Home => Some(AppEvent::View(ViewEvent::FitView)),

            // Panels
            KeyCode::KeyP => Some(AppEvent::Ui(UiEvent::PanelToggled(Panel::TaOverlay))),
            KeyCode::KeyM => Some(AppEvent::Ui(UiEvent::PanelToggled(Panel::MacdPanel))),
            KeyCode::KeyS | KeyCode::Slash => {
                Some(AppEvent::Ui(UiEvent::PanelToggled(Panel::SymbolPicker)))
            }

            // Mode changes
            KeyCode::KeyR => Some(AppEvent::Replay(ReplayEvent::Toggle)),
            KeyCode::KeyL => Some(AppEvent::Data(DataEvent::StartLiveUpdates)),

            // Drawing tools - these enter Drawing mode
            KeyCode::KeyH => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::HorizontalRay,
            )))),
            KeyCode::KeyT => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::Ray,
            )))),
            KeyCode::KeyB => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::Rectangle,
            )))),
            KeyCode::KeyV => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Selecting))),

            // Timeframe switching (digits 1-0)
            KeyCode::Digit1 => Some(AppEvent::Data(DataEvent::TimeframeChanged(0))),
            KeyCode::Digit2 => Some(AppEvent::Data(DataEvent::TimeframeChanged(1))),
            KeyCode::Digit3 => Some(AppEvent::Data(DataEvent::TimeframeChanged(2))),
            KeyCode::Digit4 => Some(AppEvent::Data(DataEvent::TimeframeChanged(3))),
            KeyCode::Digit5 => Some(AppEvent::Data(DataEvent::TimeframeChanged(4))),
            KeyCode::Digit6 => Some(AppEvent::Data(DataEvent::TimeframeChanged(5))),
            KeyCode::Digit7 => Some(AppEvent::Data(DataEvent::TimeframeChanged(6))),
            KeyCode::Digit8 => Some(AppEvent::Data(DataEvent::TimeframeChanged(7))),
            KeyCode::Digit9 => Some(AppEvent::Data(DataEvent::TimeframeChanged(8))),
            KeyCode::Digit0 => Some(AppEvent::Data(DataEvent::TimeframeChanged(9))),

            _ => None,
        }
    }

    /// Map keys in Drawing mode.
    fn map_key_drawing(&self, key: KeyCode) -> Option<AppEvent> {
        match key {
            // Toggle snap to OHLC
            KeyCode::KeyG => Some(AppEvent::Drawing(DrawingEvent::ToggleSnap)),

            // Quick tool switching (stay in drawing mode but change tool)
            KeyCode::KeyH => Some(AppEvent::Drawing(DrawingEvent::ToolSelected(
                DrawingTool::HorizontalRay,
            ))),
            KeyCode::KeyT => Some(AppEvent::Drawing(DrawingEvent::ToolSelected(DrawingTool::Ray))),
            KeyCode::KeyB => Some(AppEvent::Drawing(DrawingEvent::ToolSelected(
                DrawingTool::Rectangle,
            ))),

            // Switch to select mode
            KeyCode::KeyV => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Selecting))),

            _ => None,
        }
    }

    /// Map keys in Selecting mode.
    fn map_key_selecting(&self, key: KeyCode) -> Option<AppEvent> {
        match key {
            // Delete selected drawing
            KeyCode::Delete | KeyCode::Backspace => {
                Some(AppEvent::Drawing(DrawingEvent::DeleteSelected))
            }

            // Toggle snap
            KeyCode::KeyG => Some(AppEvent::Drawing(DrawingEvent::ToggleSnap)),

            // Quick tool switching (enter drawing mode)
            KeyCode::KeyH => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::HorizontalRay,
            )))),
            KeyCode::KeyT => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::Ray,
            )))),
            KeyCode::KeyB => Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::Rectangle,
            )))),

            _ => None,
        }
    }

    /// Map keys in Replay mode.
    fn map_key_replay(&self, key: KeyCode) -> Option<AppEvent> {
        match key {
            // Replay navigation
            KeyCode::BracketRight => Some(AppEvent::Replay(ReplayEvent::StepForward)),
            KeyCode::BracketLeft => Some(AppEvent::Replay(ReplayEvent::StepBackward)),
            KeyCode::Period => Some(AppEvent::Replay(ReplayEvent::IncreaseStep)),
            KeyCode::Comma => Some(AppEvent::Replay(ReplayEvent::DecreaseStep)),

            // View controls still work in replay mode
            KeyCode::KeyF | KeyCode::Home => Some(AppEvent::View(ViewEvent::FitView)),

            // Timeframe switching still works
            KeyCode::Digit1 => Some(AppEvent::Data(DataEvent::TimeframeChanged(0))),
            KeyCode::Digit2 => Some(AppEvent::Data(DataEvent::TimeframeChanged(1))),
            KeyCode::Digit3 => Some(AppEvent::Data(DataEvent::TimeframeChanged(2))),
            KeyCode::Digit4 => Some(AppEvent::Data(DataEvent::TimeframeChanged(3))),
            KeyCode::Digit5 => Some(AppEvent::Data(DataEvent::TimeframeChanged(4))),
            KeyCode::Digit6 => Some(AppEvent::Data(DataEvent::TimeframeChanged(5))),
            KeyCode::Digit7 => Some(AppEvent::Data(DataEvent::TimeframeChanged(6))),
            KeyCode::Digit8 => Some(AppEvent::Data(DataEvent::TimeframeChanged(7))),
            KeyCode::Digit9 => Some(AppEvent::Data(DataEvent::TimeframeChanged(8))),
            KeyCode::Digit0 => Some(AppEvent::Data(DataEvent::TimeframeChanged(9))),

            _ => None,
        }
    }

    /// Map keys in SymbolPicker mode.
    ///
    /// Most keys are consumed by the text input, so we only handle special cases.
    fn map_key_symbol_picker(&self, _key: KeyCode) -> Option<AppEvent> {
        // The symbol picker handles its own input through egui.
        // We only handle Escape (which is handled globally) to close it.
        None
    }

    /// Check if the current mode allows panning with mouse drag.
    #[must_use]
    pub fn allows_panning(&self) -> bool {
        match self.mode {
            InputMode::Normal => true,
            InputMode::Drawing(_) => !self.is_dragging, // Allow pan unless actively drawing
            InputMode::Selecting => !self.is_dragging,  // Allow pan unless dragging a drawing
            InputMode::Replay => true,
            InputMode::SymbolPicker => false,
        }
    }

    /// Check if the current mode should handle a left click.
    ///
    /// Returns true if the mode has special click handling (drawing, selection, etc.)
    #[must_use]
    pub fn handles_click(&self) -> bool {
        matches!(
            self.mode,
            InputMode::Drawing(_) | InputMode::Selecting | InputMode::Replay
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_mode_properties() {
        assert!(!InputMode::Normal.is_drawing());
        assert!(InputMode::Drawing(DrawingTool::HorizontalRay).is_drawing());

        assert!(!InputMode::Normal.blocks_panning());
        assert!(InputMode::SymbolPicker.blocks_panning());

        assert!(!InputMode::Normal.consumes_clicks());
        assert!(InputMode::Drawing(DrawingTool::Ray).consumes_clicks());
    }

    #[test]
    fn test_input_mode_drawing_tool() {
        assert_eq!(InputMode::Normal.drawing_tool(), None);
        assert_eq!(
            InputMode::Drawing(DrawingTool::Rectangle).drawing_tool(),
            Some(DrawingTool::Rectangle)
        );
    }

    #[test]
    fn test_modifiers() {
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
    fn test_input_context_new() {
        let ctx = InputContext::new();
        assert_eq!(ctx.mode, InputMode::Normal);
        assert!(ctx.modifiers.none());
        assert!(ctx.mouse_world_pos.is_none());
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_escape_action_by_mode() {
        let mut ctx = InputContext::new();

        // Normal mode - escape does nothing
        ctx.set_mode(InputMode::Normal);
        assert!(matches!(ctx.map_key(KeyCode::Escape), Some(AppEvent::None)));

        // Drawing mode - escape cancels
        ctx.set_mode(InputMode::Drawing(DrawingTool::HorizontalRay));
        assert!(matches!(
            ctx.map_key(KeyCode::Escape),
            Some(AppEvent::Drawing(DrawingEvent::Cancelled))
        ));

        // Replay mode - escape toggles off
        ctx.set_mode(InputMode::Replay);
        assert!(matches!(
            ctx.map_key(KeyCode::Escape),
            Some(AppEvent::Replay(ReplayEvent::Toggle))
        ));

        // Symbol picker - escape closes
        ctx.set_mode(InputMode::SymbolPicker);
        assert!(matches!(
            ctx.map_key(KeyCode::Escape),
            Some(AppEvent::Ui(UiEvent::PanelToggled(Panel::SymbolPicker)))
        ));
    }

    #[test]
    fn test_normal_mode_shortcuts() {
        let ctx = InputContext::new();

        // FitView
        assert!(matches!(
            ctx.map_key(KeyCode::KeyF),
            Some(AppEvent::View(ViewEvent::FitView))
        ));

        // TA panel toggle
        assert!(matches!(
            ctx.map_key(KeyCode::KeyP),
            Some(AppEvent::Ui(UiEvent::PanelToggled(Panel::TaOverlay)))
        ));

        // Drawing tool selection
        assert!(matches!(
            ctx.map_key(KeyCode::KeyH),
            Some(AppEvent::Mode(ModeEvent::Enter(InputMode::Drawing(
                DrawingTool::HorizontalRay
            ))))
        ));

        // Timeframe switch
        assert!(matches!(
            ctx.map_key(KeyCode::Digit1),
            Some(AppEvent::Data(DataEvent::TimeframeChanged(0)))
        ));
    }

    #[test]
    fn test_drawing_mode_shortcuts() {
        let mut ctx = InputContext::new();
        ctx.set_mode(InputMode::Drawing(DrawingTool::HorizontalRay));

        // Toggle snap
        assert!(matches!(
            ctx.map_key(KeyCode::KeyG),
            Some(AppEvent::Drawing(DrawingEvent::ToggleSnap))
        ));

        // Tool switching within drawing mode
        assert!(matches!(
            ctx.map_key(KeyCode::KeyT),
            Some(AppEvent::Drawing(DrawingEvent::ToolSelected(DrawingTool::Ray)))
        ));

        // FitView should NOT work in drawing mode
        assert!(ctx.map_key(KeyCode::KeyF).is_none());
    }

    #[test]
    fn test_replay_mode_shortcuts() {
        let mut ctx = InputContext::new();
        ctx.set_mode(InputMode::Replay);

        // Replay navigation
        assert!(matches!(
            ctx.map_key(KeyCode::BracketRight),
            Some(AppEvent::Replay(ReplayEvent::StepForward))
        ));
        assert!(matches!(
            ctx.map_key(KeyCode::BracketLeft),
            Some(AppEvent::Replay(ReplayEvent::StepBackward))
        ));

        // FitView still works in replay
        assert!(matches!(
            ctx.map_key(KeyCode::KeyF),
            Some(AppEvent::View(ViewEvent::FitView))
        ));
    }

    #[test]
    fn test_allows_panning() {
        let mut ctx = InputContext::new();

        // Normal mode always allows panning
        ctx.set_mode(InputMode::Normal);
        assert!(ctx.allows_panning());

        // Drawing mode allows panning when not dragging
        ctx.set_mode(InputMode::Drawing(DrawingTool::Ray));
        ctx.set_dragging(false);
        assert!(ctx.allows_panning());

        ctx.set_dragging(true);
        assert!(!ctx.allows_panning());

        // Symbol picker blocks panning
        ctx.set_mode(InputMode::SymbolPicker);
        ctx.set_dragging(false);
        assert!(!ctx.allows_panning());
    }

    #[test]
    fn test_handles_click() {
        let mut ctx = InputContext::new();

        ctx.set_mode(InputMode::Normal);
        assert!(!ctx.handles_click());

        ctx.set_mode(InputMode::Drawing(DrawingTool::HorizontalRay));
        assert!(ctx.handles_click());

        ctx.set_mode(InputMode::Selecting);
        assert!(ctx.handles_click());

        ctx.set_mode(InputMode::Replay);
        assert!(ctx.handles_click());
    }
}
