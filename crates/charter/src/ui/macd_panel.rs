//! MACD indicators control panel UI.
//!
//! This module provides the MACD panel that allows users to add, remove,
//! and configure MACD indicators.

use crate::indicators::IndicatorRegistry;
use charter_indicators::MacdConfig;
use charter_render::STATS_PANEL_WIDTH;

/// Response from the MACD panel indicating what actions should be taken.
#[derive(Debug, Clone, Default)]
pub struct MacdPanelResponse {
    /// Request to add a new MACD indicator with this config.
    pub add_indicator: Option<MacdConfig>,
    /// Request to remove the MACD at this index.
    pub remove_indicator: Option<usize>,
    /// Whether any config was changed (requires recomputation).
    pub config_changed: bool,
    /// Updated configs (indexed by registry position).
    pub updated_configs: Vec<(usize, MacdConfig)>,
}

/// Shows the MACD Indicators control panel.
///
/// # Arguments
///
/// * `ctx` - The egui context
/// * `registry` - The indicator registry (read-only, used to get current configs)
/// * `screen_width` - Screen width for positioning the panel
///
/// # Returns
///
/// A `MacdPanelResponse` describing the actions to take (add/remove/update indicators).
pub fn show_macd_panel(
    ctx: &egui::Context,
    registry: &IndicatorRegistry,
    screen_width: f32,
) -> MacdPanelResponse {
    let mut response = MacdPanelResponse::default();

    // Copy MACD configs for editing
    let mut macd_configs: Vec<MacdConfig> = registry
        .iter()
        .filter_map(|i| i.macd_config().cloned())
        .collect();

    let mut add_requested = false;
    let mut remove_index: Option<usize> = None;

    egui::Window::new("MACD Indicators")
        .default_pos([screen_width - STATS_PANEL_WIDTH - 440.0, 10.0])
        .default_width(200.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label("Press M to toggle this panel");
            ui.separator();

            // Add new MACD button
            if ui.button("+ Add MACD").clicked() {
                add_requested = true;
            }

            ui.separator();

            // List existing MACD instances
            for (i, config) in macd_configs.iter_mut().enumerate() {
                ui.push_id(i, |ui| {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("MACD #{}", i + 1));
                            if ui.small_button("X").clicked() {
                                remove_index = Some(i);
                            }
                        });

                        // Enable toggle
                        ui.checkbox(&mut config.enabled, "Enabled");

                        // Period settings
                        ui.horizontal(|ui| {
                            ui.label("Fast:");
                            let mut fast = config.fast_period as i32;
                            if ui
                                .add(egui::DragValue::new(&mut fast).range(1..=100))
                                .changed()
                            {
                                config.fast_period = fast.max(1) as usize;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Slow:");
                            let mut slow = config.slow_period as i32;
                            if ui
                                .add(egui::DragValue::new(&mut slow).range(1..=200))
                                .changed()
                            {
                                config.slow_period = slow.max(1) as usize;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Signal:");
                            let mut signal = config.signal_period as i32;
                            if ui
                                .add(egui::DragValue::new(&mut signal).range(1..=100))
                                .changed()
                            {
                                config.signal_period = signal.max(1) as usize;
                            }
                        });

                        // Color pickers (compact)
                        ui.horizontal(|ui| {
                            ui.label("MACD:");
                            ui.color_edit_button_rgb(&mut config.macd_color);
                            ui.label("Sig:");
                            ui.color_edit_button_rgb(&mut config.signal_color);
                        });
                    });
                });
            }

            if macd_configs.is_empty() {
                ui.label("No MACD indicators. Click '+ Add MACD' to create one.");
            }
        });

    // Process add request
    if add_requested {
        response.add_indicator = Some(MacdConfig::default());
    }

    // Process remove request
    response.remove_indicator = remove_index;

    // Check for config changes and collect updated configs
    for (i, new_config) in macd_configs.into_iter().enumerate() {
        if let Some(instance) = registry.get(i)
            && let Some(old_config) = instance.macd_config()
        {
            let changed = old_config.fast_period != new_config.fast_period
                || old_config.slow_period != new_config.slow_period
                || old_config.signal_period != new_config.signal_period
                || old_config.enabled != new_config.enabled
                || old_config.macd_color != new_config.macd_color
                || old_config.signal_color != new_config.signal_color;

            if changed {
                response.config_changed = true;
                response.updated_configs.push((i, new_config));
            }
        }
    }

    response
}
