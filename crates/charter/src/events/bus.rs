//! Event bus for queuing and dispatching events and commands.
//!
//! The [`EventBus`] provides a simple mechanism for collecting events
//! during input processing and then draining them for handling.

use std::collections::VecDeque;

use super::types::{AppEvent, Command};

/// A simple event bus that queues events and commands for processing.
///
/// The event bus maintains two separate queues:
/// - Events: Semantic application events that need to be processed
/// - Commands: State mutation commands that need to be executed
///
/// # Usage Pattern
///
/// ```ignore
/// let mut bus = EventBus::new();
///
/// // During input handling, emit events
/// bus.emit(AppEvent::View(ViewEvent::FitView));
///
/// // During event processing, dispatch commands
/// for event in bus.drain_events() {
///     match event {
///         AppEvent::View(ViewEvent::FitView) => {
///             bus.dispatch(Command::UpdateCamera(CameraUpdate::FitToData));
///             bus.dispatch(Command::RequestRedraw);
///         }
///         _ => {}
///     }
/// }
///
/// // Execute all pending commands
/// for cmd in bus.drain_commands() {
///     state.execute(cmd);
/// }
/// ```
#[derive(Debug, Default)]
pub struct EventBus {
    /// Queue of pending application events.
    events: VecDeque<AppEvent>,
    /// Queue of pending commands.
    commands: VecDeque<Command>,
}

impl EventBus {
    /// Create a new empty event bus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            commands: VecDeque::new(),
        }
    }

    /// Create an event bus with pre-allocated capacity.
    ///
    /// Useful when you know approximately how many events/commands
    /// will be processed per frame.
    #[must_use]
    pub fn with_capacity(event_capacity: usize, command_capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(event_capacity),
            commands: VecDeque::with_capacity(command_capacity),
        }
    }

    /// Emit an application event to be processed.
    ///
    /// Events are added to the end of the queue and will be processed
    /// in FIFO order when `drain_events` is called.
    pub fn emit(&mut self, event: AppEvent) {
        self.events.push_back(event);
    }

    /// Emit multiple events at once.
    pub fn emit_all(&mut self, events: impl IntoIterator<Item = AppEvent>) {
        self.events.extend(events);
    }

    /// Dispatch a command to be executed.
    ///
    /// Commands are added to the end of the queue and will be executed
    /// in FIFO order when `drain_commands` is called.
    pub fn dispatch(&mut self, cmd: Command) {
        self.commands.push_back(cmd);
    }

    /// Dispatch multiple commands at once.
    pub fn dispatch_all(&mut self, commands: impl IntoIterator<Item = Command>) {
        self.commands.extend(commands);
    }

    /// Drain all pending events.
    ///
    /// Returns an iterator that removes and yields all queued events.
    /// After this call completes, the event queue will be empty.
    pub fn drain_events(&mut self) -> impl Iterator<Item = AppEvent> + '_ {
        self.events.drain(..)
    }

    /// Drain all pending commands.
    ///
    /// Returns an iterator that removes and yields all queued commands.
    /// After this call completes, the command queue will be empty.
    pub fn drain_commands(&mut self) -> impl Iterator<Item = Command> + '_ {
        self.commands.drain(..)
    }

    /// Take all pending events, leaving the queue empty.
    ///
    /// Unlike `drain_events`, this returns an owned `Vec` that can be
    /// stored or passed around.
    #[must_use]
    pub fn take_events(&mut self) -> Vec<AppEvent> {
        std::mem::take(&mut self.events).into_iter().collect()
    }

    /// Take all pending commands, leaving the queue empty.
    ///
    /// Unlike `drain_commands`, this returns an owned `Vec` that can be
    /// stored or passed around.
    #[must_use]
    pub fn take_commands(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.commands).into_iter().collect()
    }

    /// Check if there are any pending events.
    #[must_use]
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Check if there are any pending commands.
    #[must_use]
    pub fn has_commands(&self) -> bool {
        !self.commands.is_empty()
    }

    /// Get the number of pending events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get the number of pending commands.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Clear all pending events and commands.
    pub fn clear(&mut self) {
        self.events.clear();
        self.commands.clear();
    }

    /// Clear only pending events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Clear only pending commands.
    pub fn clear_commands(&mut self) {
        self.commands.clear();
    }

    /// Peek at the next event without removing it.
    #[must_use]
    pub fn peek_event(&self) -> Option<&AppEvent> {
        self.events.front()
    }

    /// Peek at the next command without removing it.
    #[must_use]
    pub fn peek_command(&self) -> Option<&Command> {
        self.commands.front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::types::ViewEvent;

    #[test]
    fn test_new_bus_is_empty() {
        let bus = EventBus::new();
        assert!(!bus.has_events());
        assert!(!bus.has_commands());
        assert_eq!(bus.event_count(), 0);
        assert_eq!(bus.command_count(), 0);
    }

    #[test]
    fn test_emit_and_drain_events() {
        let mut bus = EventBus::new();

        bus.emit(AppEvent::View(ViewEvent::FitView));
        bus.emit(AppEvent::View(ViewEvent::ResetZoom));

        assert!(bus.has_events());
        assert_eq!(bus.event_count(), 2);

        let events: Vec<_> = bus.drain_events().collect();
        assert_eq!(events.len(), 2);
        assert!(!bus.has_events());
    }

    #[test]
    fn test_dispatch_and_drain_commands() {
        let mut bus = EventBus::new();

        bus.dispatch(Command::RequestRedraw);
        bus.dispatch(Command::UpdateVisibleRange);

        assert!(bus.has_commands());
        assert_eq!(bus.command_count(), 2);

        let commands: Vec<_> = bus.drain_commands().collect();
        assert_eq!(commands.len(), 2);
        assert!(!bus.has_commands());
    }

    #[test]
    fn test_emit_all() {
        let mut bus = EventBus::new();

        bus.emit_all([
            AppEvent::View(ViewEvent::FitView),
            AppEvent::View(ViewEvent::ResetZoom),
            AppEvent::None,
        ]);

        assert_eq!(bus.event_count(), 3);
    }

    #[test]
    fn test_dispatch_all() {
        let mut bus = EventBus::new();

        bus.dispatch_all([
            Command::RequestRedraw,
            Command::UpdateVisibleRange,
            Command::RecomputeTa,
        ]);

        assert_eq!(bus.command_count(), 3);
    }

    #[test]
    fn test_take_events() {
        let mut bus = EventBus::new();
        bus.emit(AppEvent::View(ViewEvent::FitView));
        bus.emit(AppEvent::None);

        let events = bus.take_events();
        assert_eq!(events.len(), 2);
        assert!(!bus.has_events());
    }

    #[test]
    fn test_clear() {
        let mut bus = EventBus::new();
        bus.emit(AppEvent::None);
        bus.dispatch(Command::RequestRedraw);

        bus.clear();

        assert!(!bus.has_events());
        assert!(!bus.has_commands());
    }

    #[test]
    fn test_peek() {
        let mut bus = EventBus::new();

        assert!(bus.peek_event().is_none());
        assert!(bus.peek_command().is_none());

        bus.emit(AppEvent::View(ViewEvent::FitView));
        bus.dispatch(Command::RequestRedraw);

        assert!(bus.peek_event().is_some());
        assert!(bus.peek_command().is_some());

        // Peek doesn't consume
        assert!(bus.has_events());
        assert!(bus.has_commands());
    }

    #[test]
    fn test_fifo_order() {
        let mut bus = EventBus::new();

        bus.emit(AppEvent::View(ViewEvent::FitView));
        bus.emit(AppEvent::View(ViewEvent::ResetZoom));
        bus.emit(AppEvent::None);

        let mut events = bus.drain_events();

        // First event should be FitView
        assert!(matches!(events.next(), Some(AppEvent::View(ViewEvent::FitView))));
        // Second should be ResetZoom
        assert!(matches!(events.next(), Some(AppEvent::View(ViewEvent::ResetZoom))));
        // Third should be None
        assert!(matches!(events.next(), Some(AppEvent::None)));
        // No more events
        assert!(events.next().is_none());
    }
}
