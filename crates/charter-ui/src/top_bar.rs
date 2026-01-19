//! Top bar UI component (dwm-style).

use charter_core::Candle;
use egui::{Color32, Context, Frame, RichText, Ui};

/// Top bar height in pixels.
pub const TOP_BAR_HEIGHT: f32 = 24.0;

/// Timeframe labels.
pub const TIMEFRAME_LABELS: &[&str] = &["1m", "3m", "5m", "30m", "1h", "3h", "5h", "10h", "1d", "1w", "3w", "1M"];

/// Top bar UI component.
pub struct TopBar;

impl TopBar {
    /// Show the top bar.
    /// Returns Some(timeframe_index) if a timeframe was clicked.
    pub fn show(
        ctx: &Context,
        symbol: &str,
        current_timeframe: usize,
        candle: Option<&Candle>,
        ws_connected: bool,
    ) -> Option<usize> {
        let mut clicked_timeframe = None;

        egui::TopBottomPanel::top("top_bar")
            .exact_height(TOP_BAR_HEIGHT)
            .frame(Frame::new().fill(Color32::from_rgb(30, 30, 30)))
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;

                    // Symbol section
                    Self::section(ui, Color32::from_rgb(40, 40, 40), |ui| {
                        let symbol_color = if ws_connected {
                            Color32::from_rgb(100, 200, 100)
                        } else {
                            Color32::from_rgb(200, 200, 200)
                        };
                        ui.label(RichText::new(symbol).color(symbol_color).strong());
                    });

                    // OHLC section
                    if let Some(c) = candle {
                        let is_bullish = c.close >= c.open;
                        let price_color = if is_bullish {
                            Color32::from_rgb(100, 200, 100)
                        } else {
                            Color32::from_rgb(200, 100, 100)
                        };

                        Self::section(ui, Color32::from_rgb(35, 35, 35), |ui| {
                            ui.label(RichText::new("O").color(Color32::GRAY).small());
                            ui.label(RichText::new(format!("{:.2}", c.open)).color(Color32::WHITE));
                        });

                        Self::section(ui, Color32::from_rgb(40, 40, 40), |ui| {
                            ui.label(RichText::new("H").color(Color32::GRAY).small());
                            ui.label(RichText::new(format!("{:.2}", c.high)).color(Color32::from_rgb(100, 200, 100)));
                        });

                        Self::section(ui, Color32::from_rgb(35, 35, 35), |ui| {
                            ui.label(RichText::new("L").color(Color32::GRAY).small());
                            ui.label(RichText::new(format!("{:.2}", c.low)).color(Color32::from_rgb(200, 100, 100)));
                        });

                        Self::section(ui, Color32::from_rgb(40, 40, 40), |ui| {
                            ui.label(RichText::new("C").color(Color32::GRAY).small());
                            ui.label(RichText::new(format!("{:.2}", c.close)).color(price_color));
                        });

                        // Change percentage
                        let change = c.close - c.open;
                        let change_pct = if c.open != 0.0 { (change / c.open) * 100.0 } else { 0.0 };
                        Self::section(ui, Color32::from_rgb(35, 35, 35), |ui| {
                            let sign = if change_pct >= 0.0 { "+" } else { "" };
                            ui.label(RichText::new(format!("{}{:.2}%", sign, change_pct)).color(price_color));
                        });
                    }

                    // Spacer to push timeframes to the right
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;

                        // Timeframes (reversed order for right-to-left layout)
                        for (i, label) in TIMEFRAME_LABELS.iter().enumerate().rev() {
                            let is_active = i == current_timeframe;
                            let bg_color = if is_active {
                                Color32::from_rgb(60, 80, 100)
                            } else {
                                Color32::from_rgb(40, 40, 40)
                            };
                            let text_color = if is_active {
                                Color32::WHITE
                            } else {
                                Color32::from_rgb(150, 150, 150)
                            };

                            let response = Self::section_clickable(ui, bg_color, |ui| {
                                ui.label(RichText::new(*label).color(text_color));
                            });

                            if response.clicked() {
                                clicked_timeframe = Some(i);
                            }
                        }
                    });
                });
            });

        clicked_timeframe
    }

    fn section(ui: &mut Ui, bg_color: Color32, content: impl FnOnce(&mut Ui)) {
        Frame::new()
            .fill(bg_color)
            .inner_margin(egui::Margin::symmetric(8, 2))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    content(ui);
                });
            });
    }

    fn section_clickable(ui: &mut Ui, bg_color: Color32, content: impl FnOnce(&mut Ui)) -> egui::Response {
        Frame::new()
            .fill(bg_color)
            .inner_margin(egui::Margin::symmetric(6, 2))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    content(ui);
                });
            })
            .response
            .interact(egui::Sense::click())
    }
}
