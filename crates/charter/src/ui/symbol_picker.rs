//! Symbol picker UI component for selecting trading pairs.

use egui::{Context, Key, Window};

/// Popular symbols for quick selection.
const POPULAR_SYMBOLS: &[&str] = &[
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "LTCUSDT",
];

/// Response from the symbol picker UI.
#[derive(Debug, Clone, Default)]
pub struct SymbolPickerResponse {
    /// Whether the picker was closed.
    pub closed: bool,
    /// The symbol that was selected, if any.
    pub selected_symbol: Option<String>,
}

/// State for the symbol picker.
#[derive(Clone)]
pub struct SymbolPickerState {
    /// Current search query.
    pub search_query: String,
    /// Recently used symbols.
    pub recent_symbols: Vec<String>,
}

impl Default for SymbolPickerState {
    fn default() -> Self {
        Self {
            search_query: String::new(),
            recent_symbols: Vec::new(),
        }
    }
}

impl SymbolPickerState {
    /// Add a symbol to recent history.
    pub fn add_recent(&mut self, symbol: &str) {
        let symbol = symbol.to_uppercase();
        // Remove if already exists
        self.recent_symbols.retain(|s| s != &symbol);
        // Add to front
        self.recent_symbols.insert(0, symbol);
        // Keep only last 5
        self.recent_symbols.truncate(5);
    }
}

/// Show the symbol picker UI.
///
/// Returns a `SymbolPickerResponse` indicating what action to take.
pub fn show_symbol_picker(
    ctx: &Context,
    state: &mut SymbolPickerState,
    current_symbol: &str,
) -> SymbolPickerResponse {
    let mut response = SymbolPickerResponse::default();

    // Handle escape to close
    if ctx.input(|i| i.key_pressed(Key::Escape)) {
        response.closed = true;
        return response;
    }

    Window::new("Select Symbol")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_min_width(300.0);

            // Current symbol display
            ui.horizontal(|ui| {
                ui.label("Current:");
                ui.strong(current_symbol);
            });

            ui.add_space(8.0);

            // Search box
            ui.horizontal(|ui| {
                ui.label("Search:");
                let text_edit = ui.text_edit_singleline(&mut state.search_query);
                if text_edit.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                    if !state.search_query.is_empty() {
                        response.selected_symbol = Some(state.search_query.to_uppercase());
                        response.closed = true;
                    }
                }
                // Focus the search box on open
                text_edit.request_focus();
            });

            ui.add_space(8.0);

            // Filter symbols based on search query
            let query = state.search_query.to_uppercase();
            let filtered_popular: Vec<_> = POPULAR_SYMBOLS
                .iter()
                .filter(|s| s.contains(&query))
                .collect();

            // Recent symbols section
            if !state.recent_symbols.is_empty() && query.is_empty() {
                ui.separator();
                ui.label("Recent:");
                ui.horizontal_wrapped(|ui| {
                    for symbol in &state.recent_symbols {
                        if ui.button(symbol).clicked() {
                            response.selected_symbol = Some(symbol.clone());
                            response.closed = true;
                        }
                    }
                });
            }

            // Popular symbols section
            ui.separator();
            ui.label("Popular:");
            ui.horizontal_wrapped(|ui| {
                for symbol in filtered_popular {
                    let is_current = *symbol == current_symbol;
                    let button = egui::Button::new(*symbol)
                        .selected(is_current);
                    if ui.add(button).clicked() && !is_current {
                        response.selected_symbol = Some(symbol.to_string());
                        response.closed = true;
                    }
                }
            });

            ui.add_space(8.0);

            // Close button
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    response.closed = true;
                }
                ui.add_space(8.0);
                ui.label("(Press Escape to close)");
            });
        });

    response
}
