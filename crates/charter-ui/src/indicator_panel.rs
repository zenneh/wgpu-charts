//! Indicator configuration panel.

use egui::{Context, Ui};

/// Indicator panel for configuring technical indicators.
pub struct IndicatorPanel {
    enabled_indicators: Vec<String>,
}

impl IndicatorPanel {
    pub fn new() -> Self {
        Self {
            enabled_indicators: Vec::new(),
        }
    }

    pub fn enabled_indicators(&self) -> &[String] {
        &self.enabled_indicators
    }

    pub fn show(&mut self, ctx: &Context) {
        egui::Window::new("Indicators")
            .collapsible(true)
            .resizable(true)
            .show(ctx, |ui| {
                self.build_content(ui);
            });
    }

    fn build_content(&mut self, ui: &mut Ui) {
        ui.label("Available Indicators:");
        ui.separator();

        // Placeholder for indicator list - users will implement their own indicators
        ui.label("No indicators loaded.");
        ui.label("Implement the Indicator trait to add custom indicators.");
    }
}

impl Default for IndicatorPanel {
    fn default() -> Self {
        Self::new()
    }
}
