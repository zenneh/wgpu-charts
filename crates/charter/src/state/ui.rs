//! UI framework state.
//!
//! This module contains state for the UI rendering framework (egui)
//! and panel visibility/state.

use crate::ui::SymbolPickerState;

/// Panel visibility state.
#[derive(Debug, Clone, Default)]
pub struct PanelState {
    /// Whether the MACD indicator panel is visible.
    pub show_macd_panel: bool,

    /// Whether the symbol picker is visible.
    pub show_symbol_picker: bool,
}

impl PanelState {
    /// Create new panel state with all panels hidden.
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle MACD panel visibility.
    pub fn toggle_macd_panel(&mut self) {
        self.show_macd_panel = !self.show_macd_panel;
    }

    /// Toggle symbol picker visibility.
    pub fn toggle_symbol_picker(&mut self) {
        self.show_symbol_picker = !self.show_symbol_picker;
    }
}

/// UI state for the application.
///
/// This struct owns the egui context and renderer, as well as
/// any UI-specific state like panel visibility and picker state.
pub struct UiState {
    /// The egui context for UI rendering.
    pub egui_ctx: egui::Context,

    /// The egui-winit state for input handling.
    pub egui_state: egui_winit::State,

    /// The egui-wgpu renderer for GPU rendering.
    pub egui_renderer: egui_wgpu::Renderer,

    /// Panel visibility state.
    pub panels: PanelState,

    /// Symbol picker state.
    pub symbol_picker_state: SymbolPickerState,
}

impl UiState {
    /// Create new UI state with the given egui components.
    pub fn new(
        egui_ctx: egui::Context,
        egui_state: egui_winit::State,
        egui_renderer: egui_wgpu::Renderer,
    ) -> Self {
        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            panels: PanelState::new(),
            symbol_picker_state: SymbolPickerState::default(),
        }
    }

    /// Check if the symbol picker is visible.
    pub fn is_symbol_picker_open(&self) -> bool {
        self.panels.show_symbol_picker
    }

    /// Check if any modal dialog is open.
    pub fn is_modal_open(&self) -> bool {
        self.panels.show_symbol_picker
    }

    /// Toggle the MACD panel.
    pub fn toggle_macd_panel(&mut self) {
        self.panels.toggle_macd_panel();
    }

    /// Toggle the symbol picker.
    pub fn toggle_symbol_picker(&mut self) {
        self.panels.toggle_symbol_picker();
    }

    /// Close the symbol picker.
    pub fn close_symbol_picker(&mut self) {
        self.panels.show_symbol_picker = false;
    }

    /// Add a symbol to recent history.
    pub fn add_recent_symbol(&mut self, symbol: &str) {
        self.symbol_picker_state.add_recent(symbol);
    }

    /// Check if egui wants keyboard input.
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.wants_keyboard_input()
    }

    /// Check if egui wants pointer input.
    pub fn wants_pointer_input(&self) -> bool {
        self.egui_ctx.wants_pointer_input()
    }
}
