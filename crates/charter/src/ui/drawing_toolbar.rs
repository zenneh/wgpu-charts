//! Drawing toolbar UI component (dwm-style).

use crate::drawing::DrawingTool;
use egui::{Color32, Context, Frame, RichText, Ui};

/// Drawing toolbar height in pixels.
pub const DRAWING_TOOLBAR_HEIGHT: f32 = 22.0;

/// Response from the drawing toolbar.
#[derive(Default)]
pub struct DrawingToolbarResponse {
    /// New tool selected (if any).
    pub selected_tool: Option<DrawingTool>,
    /// Whether snap toggle was clicked.
    pub toggle_snap: bool,
}

/// Show the drawing toolbar (dwm-style, below main top bar).
pub fn show_drawing_toolbar(
    ctx: &Context,
    current_tool: DrawingTool,
    snap_enabled: bool,
) -> DrawingToolbarResponse {
    let mut response = DrawingToolbarResponse::default();

    egui::TopBottomPanel::top("drawing_toolbar")
        .exact_height(DRAWING_TOOLBAR_HEIGHT)
        .frame(Frame::new().fill(Color32::from_rgb(25, 25, 28)))
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                // Tool buttons
                for tool in DrawingTool::all() {
                    let is_selected = current_tool == *tool;
                    let shortcut = tool.shortcut();

                    let bg_color = if is_selected {
                        Color32::from_rgb(50, 70, 90)
                    } else {
                        Color32::from_rgb(35, 35, 38)
                    };
                    let text_color = if is_selected {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(140, 140, 150)
                    };

                    let tool_response = section_clickable(ui, bg_color, |ui| {
                        ui.label(RichText::new(tool.name()).color(text_color));
                        ui.label(RichText::new(format!("[{}]", shortcut)).color(Color32::from_rgb(80, 80, 90)).small());
                    });

                    if tool_response.clicked() {
                        response.selected_tool = Some(*tool);
                    }
                }

                // Separator
                section(ui, Color32::from_rgb(30, 30, 33), |ui| {
                    ui.label(RichText::new("|").color(Color32::from_rgb(50, 50, 55)));
                });

                // Snap toggle
                let snap_bg = if snap_enabled {
                    Color32::from_rgb(40, 60, 45)
                } else {
                    Color32::from_rgb(35, 35, 38)
                };
                let snap_text_color = if snap_enabled {
                    Color32::from_rgb(100, 200, 100)
                } else {
                    Color32::from_rgb(140, 140, 150)
                };

                let snap_response = section_clickable(ui, snap_bg, |ui| {
                    let icon = if snap_enabled { "\u{2713}" } else { "\u{2717}" }; // ✓ or ✗
                    ui.label(RichText::new(icon).color(snap_text_color));
                    ui.label(RichText::new("Snap").color(snap_text_color));
                    ui.label(RichText::new("[G]").color(Color32::from_rgb(80, 80, 90)).small());
                });

                if snap_response.clicked() {
                    response.toggle_snap = true;
                }

                // Spacer and help text on right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;

                    section(ui, Color32::from_rgb(30, 30, 33), |ui| {
                        ui.label(RichText::new("Del").color(Color32::from_rgb(100, 100, 110)).small());
                        ui.label(RichText::new("delete").color(Color32::from_rgb(70, 70, 80)).small());
                    });

                    section(ui, Color32::from_rgb(35, 35, 38), |ui| {
                        ui.label(RichText::new("Esc").color(Color32::from_rgb(100, 100, 110)).small());
                        ui.label(RichText::new("cancel").color(Color32::from_rgb(70, 70, 80)).small());
                    });
                });
            });
        });

    response
}

fn section(ui: &mut Ui, bg_color: Color32, content: impl FnOnce(&mut Ui)) {
    Frame::new()
        .fill(bg_color)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                content(ui);
            });
        });
}

fn section_clickable(ui: &mut Ui, bg_color: Color32, content: impl FnOnce(&mut Ui)) -> egui::Response {
    Frame::new()
        .fill(bg_color)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                content(ui);
            });
        })
        .response
        .interact(egui::Sense::click())
}
