//! Technical Analysis control panel UI.
//!
//! This module provides the TA settings panel that allows users to configure
//! which TA elements (ranges, levels) are displayed on the chart.

use crate::replay::TimeframeTaData;
use crate::state::{MarketDataDisplaySettings, TaDisplaySettings};
use charter_render::STATS_PANEL_WIDTH;
use charter_ta::{CandleDirection, LevelState, LevelType};

/// Statistics computed from TA data.
#[derive(Debug, Clone, Default)]
pub struct TaStats {
    // Level stats
    pub total_levels: usize,
    pub active_levels: usize,
    pub broken_levels: usize,
    pub levels_with_hits: usize,
    pub total_level_hits: usize,
    pub respected_hits: usize,
    pub avg_hits_before_break: f32,

    // Level stats by type
    pub hold_levels: usize,
    pub hold_broken: usize,
    pub greedy_levels: usize,
    pub greedy_broken: usize,
}

impl TaStats {
    /// Compute statistics from TA data.
    pub fn from_ta_data(ta_data: &TimeframeTaData) -> Self {
        let mut stats = Self::default();

        // Level statistics
        stats.total_levels = ta_data.levels.len();
        for level in &ta_data.levels {
            if level.state == LevelState::Broken {
                stats.broken_levels += 1;
            } else {
                stats.active_levels += 1;
            }

            if !level.hits.is_empty() {
                stats.levels_with_hits += 1;
            }
            stats.total_level_hits += level.hits.len();
            stats.respected_hits += level.hits.iter().filter(|h| h.respected).count();

            match level.level_type {
                LevelType::Hold => {
                    stats.hold_levels += 1;
                    if level.state == LevelState::Broken {
                        stats.hold_broken += 1;
                    }
                }
                LevelType::GreedyHold => {
                    stats.greedy_levels += 1;
                    if level.state == LevelState::Broken {
                        stats.greedy_broken += 1;
                    }
                }
            }
        }

        // Average hits before break (only for broken levels)
        let broken_level_hits: usize = ta_data
            .levels
            .iter()
            .filter(|l| l.state == LevelState::Broken)
            .map(|l| l.hits.len())
            .sum();
        if stats.broken_levels > 0 {
            stats.avg_hits_before_break = broken_level_hits as f32 / stats.broken_levels as f32;
        }

        stats
    }

    /// Percentage of levels that are still active (not broken).
    pub fn level_hold_rate(&self) -> f32 {
        if self.total_levels == 0 {
            0.0
        } else {
            (self.active_levels as f32 / self.total_levels as f32) * 100.0
        }
    }

    /// Percentage of hits that were respected (wick only, body held).
    pub fn hit_respect_rate(&self) -> f32 {
        if self.total_level_hits == 0 {
            0.0
        } else {
            (self.respected_hits as f32 / self.total_level_hits as f32) * 100.0
        }
    }

    /// Hold level break rate.
    pub fn hold_break_rate(&self) -> f32 {
        if self.hold_levels == 0 {
            0.0
        } else {
            (self.hold_broken as f32 / self.hold_levels as f32) * 100.0
        }
    }

    /// Greedy level break rate.
    pub fn greedy_break_rate(&self) -> f32 {
        if self.greedy_levels == 0 {
            0.0
        } else {
            (self.greedy_broken as f32 / self.greedy_levels as f32) * 100.0
        }
    }
}

/// Information about hovered TA elements to display in the panel.
#[derive(Debug, Clone, Default)]
pub struct TaHoveredInfo {
    /// Hovered range info: (direction, candle_count, high, low, start_index, end_index)
    pub range: Option<(CandleDirection, usize, f32, f32, usize, usize)>,
    /// Hovered level info: (price, level_type, direction, state, hit_count)
    pub level: Option<(f32, LevelType, CandleDirection, LevelState, usize)>,
}

/// Response from the TA panel indicating what actions should be taken.
#[derive(Debug, Clone, Default)]
pub struct TaPanelResponse {
    /// Whether settings were changed and need to be applied.
    pub settings_changed: bool,
    /// The new settings (if changed).
    pub new_settings: Option<TaDisplaySettings>,
    /// Whether market data settings were changed.
    pub market_data_changed: bool,
    /// The new market data settings (if changed).
    pub new_market_data_settings: Option<MarketDataDisplaySettings>,
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
    market_data_settings: &MarketDataDisplaySettings,
) -> TaPanelResponse {
    let mut response = TaPanelResponse::default();
    let mut ta_settings = settings.clone();
    let mut md_settings = market_data_settings.clone();

    // Get counts from TA data
    let (range_count, level_count) = ta_data
        .map(|ta| (ta.ranges.len(), ta.levels.len()))
        .unwrap_or((0, 0));

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
                    ui.label("Trend States:");
                    ui.checkbox(&mut ta_settings.show_active_trends, "Active");
                    ui.checkbox(&mut ta_settings.show_hit_trends, "Hit");
                    ui.checkbox(&mut ta_settings.show_broken_trends, "Broken");
                }

                ui.separator();
                ui.label(format!("Ranges: {}", range_count));
                ui.label(format!("Levels: {}", level_count));

                // Show TA statistics
                if let Some(ta) = ta_data {
                    let stats = TaStats::from_ta_data(ta);

                    ui.separator();
                    ui.label("Level Statistics:");
                    ui.label(format!(
                        "  Active: {} / Broken: {}",
                        stats.active_levels, stats.broken_levels
                    ));
                    ui.label(format!("  Hold rate: {:.1}%", stats.level_hold_rate()));
                    if stats.total_level_hits > 0 {
                        ui.label(format!(
                            "  Total hits: {} ({:.1}% respected)",
                            stats.total_level_hits,
                            stats.hit_respect_rate()
                        ));
                    }
                    if stats.broken_levels > 0 {
                        ui.label(format!(
                            "  Avg hits before break: {:.1}",
                            stats.avg_hits_before_break
                        ));
                    }

                    // By level type
                    if stats.hold_levels > 0 || stats.greedy_levels > 0 {
                        ui.label(format!(
                            "  Hold: {} ({:.0}% broken)",
                            stats.hold_levels,
                            stats.hold_break_rate()
                        ));
                        ui.label(format!(
                            "  Greedy: {} ({:.0}% broken)",
                            stats.greedy_levels,
                            stats.greedy_break_rate()
                        ));
                    }
                }

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
            }

            // Market Data section
            ui.separator();
            ui.label("Market Data:");
            ui.checkbox(&mut md_settings.show_volume_profile, "Volume Profile");
            ui.checkbox(&mut md_settings.show_depth_heatmap, "Depth Heatmap");
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

    // Check if market data settings changed
    let md_changed = md_settings.show_volume_profile != market_data_settings.show_volume_profile
        || md_settings.show_depth_heatmap != market_data_settings.show_depth_heatmap;

    if md_changed {
        response.market_data_changed = true;
        response.new_market_data_settings = Some(md_settings);
    }

    response
}
