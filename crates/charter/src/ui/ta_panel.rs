//! Technical Analysis control panel UI.
//!
//! This module provides the TA settings panel that allows users to configure
//! which TA elements (ranges, levels, trends) are displayed on the chart.

use crate::replay::TimeframeTaData;
use crate::state::TaDisplaySettings;
use charter_render::STATS_PANEL_WIDTH;
use charter_ta::{CandleDirection, LevelState, LevelType, TrendState};

/// Information about hovered TA elements to display in the panel.
#[derive(Debug, Clone, Default)]
pub struct TaHoveredInfo {
    /// Hovered range info: (direction, candle_count, high, low, start_index, end_index)
    pub range: Option<(CandleDirection, usize, f32, f32, usize, usize)>,
    /// Hovered level info: (price, level_type, direction, state, hit_count)
    pub level: Option<(f32, LevelType, CandleDirection, LevelState, usize)>,
    /// Hovered trend info: (direction, state, start_price, end_price, start_idx, end_idx, hit_count)
    pub trend: Option<(CandleDirection, TrendState, f32, f32, usize, usize, usize)>,
}

/// Response from the TA panel indicating what actions should be taken.
#[derive(Debug, Clone, Default)]
pub struct TaPanelResponse {
    /// Whether settings were changed and need to be applied.
    pub settings_changed: bool,
    /// The new settings (if changed).
    pub new_settings: Option<TaDisplaySettings>,
}

/// Shows the Technical Analysis control panel.
///
/// # Arguments
///
/// * `ctx` - The egui context
/// * `settings` - Current TA display settings (will be cloned and modified)
/// * `ta_data` - Optional TA data for the current timeframe (used for counts)
/// * `hovered` - Information about currently hovered TA elements
/// * `screen_width` - Screen width for positioning the panel
///
/// # Returns
///
/// A `TaPanelResponse` indicating whether settings were changed and the new values.
pub fn show_ta_panel(
    ctx: &egui::Context,
    settings: &TaDisplaySettings,
    ta_data: Option<&TimeframeTaData>,
    hovered: &TaHoveredInfo,
    screen_width: f32,
) -> TaPanelResponse {
    let mut response = TaPanelResponse::default();
    let mut ta_settings = settings.clone();

    // Get counts from TA data
    let (range_count, level_count, trend_count) = ta_data
        .map(|ta| (ta.ranges.len(), ta.levels.len(), ta.trends.len()))
        .unwrap_or((0, 0, 0));

    egui::Window::new("Technical Analysis")
        .default_pos([screen_width - STATS_PANEL_WIDTH - 220.0, 10.0])
        .default_width(200.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.checkbox(&mut ta_settings.show_ta, "Enable TA (P)");

            if ta_settings.show_ta {
                ui.separator();
                ui.checkbox(&mut ta_settings.show_ranges, "Show Ranges");

                ui.separator();
                ui.label("Level Types:");
                ui.checkbox(&mut ta_settings.show_hold_levels, "Hold Levels");
                ui.checkbox(&mut ta_settings.show_greedy_levels, "Greedy Levels");

                ui.separator();
                ui.label("Level States:");
                ui.checkbox(&mut ta_settings.show_active_levels, "Active");
                ui.checkbox(&mut ta_settings.show_hit_levels, "Hit");
                ui.checkbox(&mut ta_settings.show_broken_levels, "Broken");

                ui.separator();
                ui.checkbox(&mut ta_settings.show_trends, "Show Trends");
                if ta_settings.show_trends {
                    ui.indent("trend_states", |ui| {
                        ui.checkbox(&mut ta_settings.show_active_trends, "Active");
                        ui.checkbox(&mut ta_settings.show_hit_trends, "Hit");
                        ui.checkbox(&mut ta_settings.show_broken_trends, "Broken");
                    });
                }

                ui.separator();
                ui.label(format!("Ranges: {}", range_count));
                ui.label(format!("Levels: {}", level_count));
                ui.label(format!("Trends: {}", trend_count));

                // Show hovered element info
                if let Some((dir, count, high, low, start, end)) = hovered.range {
                    ui.separator();
                    ui.label("Hovered Range:");
                    ui.label(format!("  Direction: {:?}", dir));
                    ui.label(format!("  Candles: {}", count));
                    ui.label(format!("  High: {:.2}", high));
                    ui.label(format!("  Low: {:.2}", low));
                    ui.label(format!("  Index: {} - {}", start, end));
                }

                if let Some((price, ltype, dir, state, hits)) = hovered.level {
                    ui.separator();
                    ui.label("Hovered Level:");
                    ui.label(format!("  Price: {:.2}", price));
                    ui.label(format!("  Type: {:?}", ltype));
                    ui.label(format!("  Direction: {:?}", dir));
                    ui.label(format!("  State: {:?}", state));
                    ui.label(format!("  Hits: {}", hits));
                }

                if let Some((dir, state, start_price, end_price, start_idx, end_idx, hits)) =
                    hovered.trend
                {
                    ui.separator();
                    ui.label("Hovered Trend:");
                    ui.label(format!("  Direction: {:?}", dir));
                    ui.label(format!("  State: {:?}", state));
                    ui.label(format!("  Start: {:.2} @ idx {}", start_price, start_idx));
                    ui.label(format!("  End: {:.2} @ idx {}", end_price, end_idx));
                    ui.label(format!("  Hits: {}", hits));
                }
            }
        });

    // Check if settings changed
    let changed = ta_settings.show_ta != settings.show_ta
        || ta_settings.show_ranges != settings.show_ranges
        || ta_settings.show_hold_levels != settings.show_hold_levels
        || ta_settings.show_greedy_levels != settings.show_greedy_levels
        || ta_settings.show_active_levels != settings.show_active_levels
        || ta_settings.show_hit_levels != settings.show_hit_levels
        || ta_settings.show_broken_levels != settings.show_broken_levels
        || ta_settings.show_trends != settings.show_trends
        || ta_settings.show_active_trends != settings.show_active_trends
        || ta_settings.show_hit_trends != settings.show_hit_trends
        || ta_settings.show_broken_trends != settings.show_broken_trends;

    if changed {
        response.settings_changed = true;
        response.new_settings = Some(ta_settings);
    }

    response
}
