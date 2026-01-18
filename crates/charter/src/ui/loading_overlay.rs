//! Loading overlay UI.
//!
//! This module provides the loading overlay that is displayed during
//! background operations like data loading, aggregation, and TA computation.

use crate::state::LoadingState;

/// Shows the loading overlay when loading is in progress.
///
/// This displays a semi-transparent dark overlay over the entire screen
/// with a centered loading indicator and status message.
///
/// # Arguments
///
/// * `ctx` - The egui context
/// * `state` - The current loading state
///
/// # Notes
///
/// This function does nothing if `state` is `LoadingState::Idle`.
pub fn show_loading_overlay(ctx: &egui::Context, state: &LoadingState) {
    if !state.is_loading() {
        return;
    }

    let screen_rect = ctx.screen_rect();
    let center = screen_rect.center();

    // Semi-transparent dark overlay
    egui::Area::new(egui::Id::new("loading_overlay"))
        .fixed_pos(egui::Pos2::ZERO)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let painter = ui.painter();
            painter.rect_filled(
                screen_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
            );
        });

    // Loading window
    egui::Window::new("Loading")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_pos(center)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(10.0);
                ui.spinner();
                ui.add_space(10.0);
                ui.label(state.message());
                ui.add_space(10.0);
            });
        });
}
