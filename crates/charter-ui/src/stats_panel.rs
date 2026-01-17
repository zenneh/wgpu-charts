//! Stats panel showing OHLCV data.

use charter_core::Candle;
use egui::{Color32, Context, Ui};

/// Stats panel configuration.
pub struct StatsPanelConfig {
    pub width: f32,
    pub timeframe_labels: &'static [&'static str],
}

impl Default for StatsPanelConfig {
    fn default() -> Self {
        Self {
            width: 200.0,
            timeframe_labels: &["1m", "15m", "1h", "1w", "1M"],
        }
    }
}

/// Stats panel UI component.
pub struct StatsPanel {
    config: StatsPanelConfig,
}

impl StatsPanel {
    pub fn new(config: StatsPanelConfig) -> Self {
        Self { config }
    }

    pub fn width(&self) -> f32 {
        self.config.width
    }

    pub fn show(
        &self,
        ctx: &Context,
        current_timeframe: usize,
        fps: f32,
        candle_count: usize,
        visible_count: u32,
        candle: Option<&Candle>,
    ) {
        egui::SidePanel::right("stats_panel")
            .exact_width(self.config.width)
            .resizable(false)
            .show(ctx, |ui| {
                self.build_content(ui, current_timeframe, fps, candle_count, visible_count, candle);
            });
    }

    fn build_content(
        &self,
        ui: &mut Ui,
        current_timeframe: usize,
        fps: f32,
        candle_count: usize,
        visible_count: u32,
        candle: Option<&Candle>,
    ) {
        ui.heading("Stats");
        ui.separator();

        // Timeframe buttons
        ui.horizontal(|ui| {
            for (i, label) in self.config.timeframe_labels.iter().enumerate() {
                let selected = i == current_timeframe;
                let _ = ui.selectable_label(selected, *label);
            }
        });
        ui.separator();

        ui.label(format!("FPS: {:.1}", fps));
        ui.label(format!("Candles: {}", candle_count));
        ui.label(format!("Visible: {}", visible_count));
        ui.separator();

        if let Some(c) = candle {
            ui.heading("OHLCV");
            ui.label(format!("Open:  ${:.2}", c.open));
            ui.label(format!("High:  ${:.2}", c.high));
            ui.label(format!("Low:   ${:.2}", c.low));
            ui.label(format!("Close: ${:.2}", c.close));
            ui.separator();

            let change = c.close - c.open;
            let change_pct = (change / c.open) * 100.0;
            let color = if change >= 0.0 {
                Color32::GREEN
            } else {
                Color32::RED
            };
            ui.colored_label(color, format!("Change: {:.2} ({:.2}%)", change, change_pct));
            ui.separator();

            ui.label(format!("Volume: {:.4}", c.volume));
        }
    }
}

impl Default for StatsPanel {
    fn default() -> Self {
        Self::new(StatsPanelConfig::default())
    }
}
