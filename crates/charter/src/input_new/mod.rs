//! Input handling for the Charter application.
//!
//! This module provides:
//! - [`InputMode`] - Different input modes (Normal, Drawing, Selecting, Replay, SymbolPicker)
//! - [`InputContext`] - Mode-aware input handling with key mapping
//! - [`Modifiers`] - Keyboard modifier state tracking
//! - [`InputHandler`] - Legacy input handler for raw event processing
//! - [`InputAction`] - Legacy semantic input actions
//!
//! # Architecture
//!
//! The input system is designed around the concept of *modes*. Different modes
//! interpret the same input differently:
//!
//! - In **Normal** mode, 'H' enters HorizontalRay drawing mode
//! - In **Drawing** mode, 'H' switches to HorizontalRay tool
//! - In **Replay** mode, 'H' is ignored
//!
//! The [`InputContext`] struct provides the `map_key()` method that handles
//! this mode-aware mapping automatically.
//!
//! # Migration Note
//!
//! The `handler` module contains the legacy [`InputHandler`] and [`InputAction`]
//! types. New code should prefer using [`InputContext`] and the event system.
//! The legacy types are kept for backward compatibility during the refactoring.
//!
//! # Example
//!
//! ```ignore
//! use charter::input::{InputContext, InputMode};
//! use winit::keyboard::KeyCode;
//!
//! let mut ctx = InputContext::new();
//!
//! // Start in normal mode
//! if let Some(event) = ctx.map_key(KeyCode::KeyH) {
//!     // This will be ModeEvent::Enter(InputMode::Drawing(HorizontalRay))
//!     handle_event(event);
//! }
//!
//! // Now in drawing mode
//! ctx.set_mode(InputMode::Drawing(DrawingTool::HorizontalRay));
//!
//! // Same key, different result
//! if let Some(event) = ctx.map_key(KeyCode::KeyH) {
//!     // This will be DrawingEvent::ToolSelected(HorizontalRay)
//!     handle_event(event);
//! }
//! ```

mod handler;
pub mod mode;

// Re-export new types for the event-based architecture
pub use mode::{InputContext, InputMode, Modifiers};

// Re-export legacy types for backward compatibility
pub use handler::{InputAction, InputHandler};
