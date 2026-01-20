//! UI module for extracted egui panels and overlays.
//!
//! This module contains the UI components extracted from state.rs to improve
//! code organization and maintainability. Each component is designed to:
//!
//! 1. Take ownership of what it needs to display/modify via parameters
//! 2. Return response structs describing what actions should be taken
//! 3. Not directly modify application state - that responsibility stays in state.rs

mod drawing_toolbar;
mod loading_overlay;
mod macd_panel;
mod symbol_picker;
mod ta_panel;

pub use drawing_toolbar::{show_drawing_toolbar, DrawingToolbarResponse};
pub use loading_overlay::show_loading_overlay;
pub use macd_panel::{show_macd_panel, MacdPanelResponse};
pub use symbol_picker::{show_symbol_picker, SymbolPickerState};
pub use ta_panel::{show_ta_panel, TaHoveredInfo};
