//! Event system for the Charter application.
//!
//! This module provides a clean event-based architecture with:
//! - [`InputEvent`] - Raw input events from winit (keyboard, mouse, scroll)
//! - [`AppEvent`] - Semantic application events (view, mode, drawing, data, replay, UI)
//! - [`Command`] - State mutation commands
//! - [`EventBus`] - Simple event queue and command dispatch system
//!
//! # Architecture
//!
//! The event system follows a unidirectional data flow:
//!
//! ```text
//! WindowEvent -> InputEvent -> AppEvent -> Command -> State Mutation
//! ```
//!
//! This separation allows:
//! - Mode-aware input handling (same key can do different things in different modes)
//! - Decoupled event processing
//! - Easy testing of event handlers
//!
//! # Example
//!
//! ```ignore
//! use charter::events::{EventBus, AppEvent, Command};
//!
//! let mut bus = EventBus::new();
//!
//! // Emit semantic events
//! bus.emit(AppEvent::FitView);
//!
//! // Process events and dispatch commands
//! for event in bus.drain_events() {
//!     match event {
//!         AppEvent::FitView => {
//!             bus.dispatch(Command::RequestRedraw);
//!         }
//!         _ => {}
//!     }
//! }
//!
//! // Execute commands
//! for cmd in bus.drain_commands() {
//!     // Apply command to state
//! }
//! ```

mod bus;
mod types;

pub use bus::EventBus;
pub use types::{
    AppEvent, CameraUpdate, Command, DataEvent, DrawingEvent, InputEvent, InputMode, ModeEvent,
    Modifiers, MouseButton, Panel, ReplayEvent, UiEvent, ViewEvent,
};
