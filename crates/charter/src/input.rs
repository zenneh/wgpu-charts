//! Input handling for the Charter application.
//!
//! This module provides a centralized input handler that converts raw input events
//! (keyboard, mouse, scroll) into semantic [`InputAction`]s that can be processed
//! by the application state.

use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

/// Semantic actions that can be triggered by user input.
///
/// These actions represent high-level operations that the application can perform,
/// decoupled from the specific input events that trigger them.
#[derive(Debug, Clone, PartialEq)]
pub enum InputAction {
    /// Exit the application.
    Exit,
    /// Fit the view to show all data.
    FitView,
    /// Toggle technical analysis overlay visibility.
    ToggleTa,
    /// Toggle the MACD panel visibility.
    ToggleMacdPanel,
    /// Toggle replay mode on/off.
    ToggleReplayMode,
    /// Step forward in replay mode.
    ReplayStepForward,
    /// Step backward in replay mode.
    ReplayStepBackward,
    /// Decrease the replay step size.
    ReplayDecreaseStep,
    /// Increase the replay step size.
    ReplayIncreaseStep,
    /// Switch to a different timeframe by index.
    SwitchTimeframe(usize),
    /// Pan the view by the given delta in screen coordinates.
    Pan { dx: f32, dy: f32 },
    /// Zoom the view.
    Zoom {
        /// Horizontal scroll delta for X-axis zoom.
        delta_x: f32,
        /// Vertical scroll delta for Y-axis zoom.
        delta_y: f32,
        /// Cursor X position for zoom-to-cursor behavior.
        cursor_x: f32,
        /// Cursor Y position for zoom-to-cursor behavior.
        cursor_y: f32,
    },
    /// Start a drag operation (mouse pressed).
    StartDrag,
    /// End a drag operation (mouse released).
    EndDrag,
    /// Update cursor position (for hover effects and replay mode).
    CursorMoved { x: f32, y: f32 },
    /// Set replay index from click position.
    SetReplayIndex,
    /// Toggle the symbol picker.
    ToggleSymbolPicker,
    /// Start live data updates.
    StartLiveUpdates,
    /// Cancel the current drawing operation.
    DrawingCancel,
    /// Select a drawing tool.
    SelectDrawingTool(crate::drawing::DrawingTool),
    /// Toggle snap mode for drawing tools.
    ToggleSnap,
    /// Delete the currently selected drawing.
    DrawingDelete,
}

/// Handles input events and converts them to semantic actions.
///
/// The `InputHandler` maintains mouse state (pressed, last position) to enable
/// drag-based panning and zoom-to-cursor behavior.
///
/// # Example
///
/// ```ignore
/// let mut handler = InputHandler::new();
///
/// // Handle a key press
/// if let Some(action) = handler.handle_key(KeyCode::KeyF, true) {
///     match action {
///         InputAction::FitView => state.fit_view(),
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct InputHandler {
    /// Whether the left mouse button is currently pressed.
    pub mouse_pressed: bool,
    /// Last known cursor position in screen coordinates.
    pub last_mouse_pos: Option<[f32; 2]>,
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl InputHandler {
    /// Create a new input handler with default state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    /// Handle a keyboard event.
    ///
    /// Returns `Some(InputAction)` if the key event should trigger an action,
    /// or `None` if the event should be ignored.
    ///
    /// # Arguments
    ///
    /// * `code` - The physical key code that was pressed or released.
    /// * `is_pressed` - Whether the key is being pressed (true) or released (false).
    #[must_use]
    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> Option<InputAction> {
        // Only respond to key press events, not releases
        if !is_pressed {
            return None;
        }

        match code {
            KeyCode::Escape => Some(InputAction::Exit),
            KeyCode::KeyF | KeyCode::Home => Some(InputAction::FitView),
            KeyCode::KeyP => Some(InputAction::ToggleTa),
            KeyCode::KeyM => Some(InputAction::ToggleMacdPanel),
            KeyCode::KeyR => Some(InputAction::ToggleReplayMode),
            KeyCode::KeyS | KeyCode::Slash => Some(InputAction::ToggleSymbolPicker),
            KeyCode::KeyL => Some(InputAction::StartLiveUpdates),
            KeyCode::BracketRight => Some(InputAction::ReplayStepForward),
            KeyCode::BracketLeft => Some(InputAction::ReplayStepBackward),
            KeyCode::Comma => Some(InputAction::ReplayDecreaseStep),
            KeyCode::Period => Some(InputAction::ReplayIncreaseStep),
            KeyCode::Digit1 => Some(InputAction::SwitchTimeframe(0)),
            KeyCode::Digit2 => Some(InputAction::SwitchTimeframe(1)),
            KeyCode::Digit3 => Some(InputAction::SwitchTimeframe(2)),
            KeyCode::Digit4 => Some(InputAction::SwitchTimeframe(3)),
            KeyCode::Digit5 => Some(InputAction::SwitchTimeframe(4)),
            KeyCode::Digit6 => Some(InputAction::SwitchTimeframe(5)),
            KeyCode::Digit7 => Some(InputAction::SwitchTimeframe(6)),
            KeyCode::Digit8 => Some(InputAction::SwitchTimeframe(7)),
            KeyCode::Digit9 => Some(InputAction::SwitchTimeframe(8)),
            KeyCode::Digit0 => Some(InputAction::SwitchTimeframe(9)),

            // Drawing tools
            KeyCode::KeyH => Some(InputAction::SelectDrawingTool(crate::drawing::DrawingTool::HorizontalRay)),
            KeyCode::KeyT => Some(InputAction::SelectDrawingTool(crate::drawing::DrawingTool::Ray)),
            KeyCode::KeyB => Some(InputAction::SelectDrawingTool(crate::drawing::DrawingTool::Rectangle)),
            KeyCode::KeyV => Some(InputAction::SelectDrawingTool(crate::drawing::DrawingTool::Select)),

            // Delete selected drawing
            KeyCode::Backspace | KeyCode::Delete => Some(InputAction::DrawingDelete),

            _ => None,
        }
    }

    /// Handle a mouse button event.
    ///
    /// Returns `Some(InputAction)` if the mouse event should trigger an action,
    /// or `None` if the event should be ignored.
    ///
    /// # Arguments
    ///
    /// * `state` - Whether the button is being pressed or released.
    /// * `button` - Which mouse button was pressed.
    /// * `in_replay_mode` - Whether the application is currently in replay mode.
    /// * `replay_index_set` - Whether a replay index has already been set.
    #[must_use]
    pub fn handle_mouse_input(
        &mut self,
        state: ElementState,
        button: MouseButton,
        in_replay_mode: bool,
        replay_index_set: bool,
    ) -> Option<InputAction> {
        if button != MouseButton::Left {
            return None;
        }

        let is_pressed = state == ElementState::Pressed;

        // Handle replay mode click to set index
        if in_replay_mode && !replay_index_set && is_pressed {
            return Some(InputAction::SetReplayIndex);
        }

        // Update mouse pressed state and return appropriate action
        if is_pressed {
            self.mouse_pressed = true;
            Some(InputAction::StartDrag)
        } else {
            self.mouse_pressed = false;
            self.last_mouse_pos = None;
            Some(InputAction::EndDrag)
        }
    }

    /// Handle a cursor movement event.
    ///
    /// Returns `Some(InputAction)` if the cursor movement should trigger an action.
    /// This will return a `Pan` action if the mouse is pressed and being dragged,
    /// or a `CursorMoved` action for hover tracking.
    ///
    /// # Arguments
    ///
    /// * `position` - The new cursor position in physical (screen) coordinates.
    #[must_use]
    pub fn handle_cursor_moved(&mut self, position: (f32, f32)) -> Option<InputAction> {
        let current_pos = [position.0, position.1];

        let action = if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let dx = current_pos[0] - last_pos[0];
                let dy = current_pos[1] - last_pos[1];
                Some(InputAction::Pan { dx, dy })
            } else {
                // First movement after mouse press - just track position
                None
            }
        } else {
            // Not dragging - just report cursor position for hover effects
            Some(InputAction::CursorMoved {
                x: current_pos[0],
                y: current_pos[1],
            })
        };

        self.last_mouse_pos = Some(current_pos);
        action
    }

    /// Handle a mouse wheel event.
    ///
    /// Returns a `Zoom` action with the scroll deltas and current cursor position.
    ///
    /// # Arguments
    ///
    /// * `delta` - The scroll delta from the mouse wheel.
    #[must_use]
    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) -> Option<InputAction> {
        let (scroll_x, scroll_y) = match delta {
            MouseScrollDelta::LineDelta(x, y) => (x, y),
            MouseScrollDelta::PixelDelta(pos) => (pos.x as f32 / 50.0, pos.y as f32 / 50.0),
        };

        // Get cursor position for zoom-to-cursor, defaulting to center
        let (cursor_x, cursor_y) = self
            .last_mouse_pos
            .map(|p| (p[0], p[1]))
            .unwrap_or((0.0, 0.0));

        Some(InputAction::Zoom {
            delta_x: scroll_x,
            delta_y: scroll_y,
            cursor_x,
            cursor_y,
        })
    }

    /// Get the last known cursor position.
    #[must_use]
    #[allow(dead_code)]
    pub fn cursor_position(&self) -> Option<[f32; 2]> {
        self.last_mouse_pos
    }

    /// Check if the mouse button is currently pressed.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_mouse_pressed(&self) -> bool {
        self.mouse_pressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_input_handler() {
        let handler = InputHandler::new();
        assert!(!handler.mouse_pressed);
        assert!(handler.last_mouse_pos.is_none());
    }

    #[test]
    fn test_key_handling() {
        let mut handler = InputHandler::new();

        // Key press should return action
        assert_eq!(
            handler.handle_key(KeyCode::Escape, true),
            Some(InputAction::Exit)
        );
        assert_eq!(
            handler.handle_key(KeyCode::KeyF, true),
            Some(InputAction::FitView)
        );
        assert_eq!(
            handler.handle_key(KeyCode::Digit1, true),
            Some(InputAction::SwitchTimeframe(0))
        );

        // Key release should return None
        assert_eq!(handler.handle_key(KeyCode::Escape, false), None);

        // Unknown key should return None
        assert_eq!(handler.handle_key(KeyCode::KeyZ, true), None);
    }

    #[test]
    fn test_mouse_input() {
        let mut handler = InputHandler::new();

        // Press should start drag
        let action =
            handler.handle_mouse_input(ElementState::Pressed, MouseButton::Left, false, false);
        assert_eq!(action, Some(InputAction::StartDrag));
        assert!(handler.mouse_pressed);

        // Release should end drag
        let action =
            handler.handle_mouse_input(ElementState::Released, MouseButton::Left, false, false);
        assert_eq!(action, Some(InputAction::EndDrag));
        assert!(!handler.mouse_pressed);
    }

    #[test]
    fn test_replay_mode_click() {
        let mut handler = InputHandler::new();

        // In replay mode without index set, click should set replay index
        let action =
            handler.handle_mouse_input(ElementState::Pressed, MouseButton::Left, true, false);
        assert_eq!(action, Some(InputAction::SetReplayIndex));
    }

    #[test]
    fn test_cursor_movement_dragging() {
        let mut handler = InputHandler::new();
        handler.mouse_pressed = true;

        // First movement just sets position
        let action = handler.handle_cursor_moved((100.0, 100.0));
        assert!(action.is_none());

        // Second movement generates pan
        let action = handler.handle_cursor_moved((110.0, 105.0));
        assert_eq!(
            action,
            Some(InputAction::Pan {
                dx: 10.0,
                dy: 5.0
            })
        );
    }

    #[test]
    fn test_cursor_movement_not_dragging() {
        let mut handler = InputHandler::new();

        // Without mouse pressed, should report cursor position
        let action = handler.handle_cursor_moved((100.0, 200.0));
        assert_eq!(
            action,
            Some(InputAction::CursorMoved {
                x: 100.0,
                y: 200.0
            })
        );
    }

    #[test]
    fn test_mouse_wheel() {
        let mut handler = InputHandler::new();
        handler.last_mouse_pos = Some([150.0, 250.0]);

        let action = handler.handle_mouse_wheel(MouseScrollDelta::LineDelta(1.0, -2.0));
        assert_eq!(
            action,
            Some(InputAction::Zoom {
                delta_x: 1.0,
                delta_y: -2.0,
                cursor_x: 150.0,
                cursor_y: 250.0,
            })
        );
    }
}
