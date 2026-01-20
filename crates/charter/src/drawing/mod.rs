//! Interactive drawing tools for chart annotations.

mod state;
mod types;

pub use state::{DrawingManager, DrawingTool};
pub use types::{Drawing, DrawingId};

use charter_render::{AnchorGpu, DrawingHRayGpu, DrawingRayGpu, DrawingRectGpu};

/// Data needed for rendering drawings.
pub struct DrawingRenderData {
    pub hrays: Vec<DrawingHRayGpu>,
    pub rays: Vec<DrawingRayGpu>,
    pub rects: Vec<DrawingRectGpu>,
    pub anchors: Vec<AnchorGpu>,
}

/// Convert drawings to GPU format for rendering.
pub fn prepare_drawing_render_data(manager: &DrawingManager, candle_spacing: f32) -> DrawingRenderData {
    let mut hrays = Vec::new();
    let mut rays = Vec::new();
    let mut rects = Vec::new();
    let mut anchors = Vec::new();

    // Add all completed drawings
    for drawing in &manager.drawings {
        let is_selected = manager.selected.map_or(false, |id| id == drawing.id());
        add_drawing_to_buffers(
            drawing,
            is_selected,
            manager.hovered_anchor,
            candle_spacing,
            &mut hrays,
            &mut rays,
            &mut rects,
            &mut anchors,
        );
    }

    // Add preview drawing (if any)
    if let Some(preview) = manager.preview_drawing() {
        add_drawing_to_buffers(
            &preview,
            false,
            None,
            candle_spacing,
            &mut hrays,
            &mut rays,
            &mut rects,
            &mut anchors,
        );
    }

    DrawingRenderData { hrays, rays, rects, anchors }
}

fn add_drawing_to_buffers(
    drawing: &Drawing,
    is_selected: bool,
    hovered_anchor: Option<(DrawingId, usize)>,
    candle_spacing: f32,
    hrays: &mut Vec<DrawingHRayGpu>,
    rays: &mut Vec<DrawingRayGpu>,
    rects: &mut Vec<DrawingRectGpu>,
    anchors: &mut Vec<AnchorGpu>,
) {
    match drawing {
        Drawing::HorizontalRay(hr) => {
            hrays.push(DrawingHRayGpu {
                x_start: hr.anchor.candle_index * candle_spacing,
                y_value: hr.anchor.price,
                r: hr.color[0],
                g: hr.color[1],
                b: hr.color[2],
                a: hr.color[3],
                line_style: 0,
                _padding: 0,
            });

            if is_selected {
                let is_hovered = hovered_anchor.map_or(false, |(id, idx)| id == hr.id && idx == 0);
                anchors.push(AnchorGpu {
                    x: hr.anchor.candle_index * candle_spacing,
                    y: hr.anchor.price,
                    is_hovered: u32::from(is_hovered),
                    is_selected: 1,
                });
            }
        }
        Drawing::Ray(ray) => {
            rays.push(DrawingRayGpu {
                x_start: ray.start.candle_index * candle_spacing,
                y_start: ray.start.price,
                x_end: ray.end.candle_index * candle_spacing,
                y_end: ray.end.price,
                r: ray.color[0],
                g: ray.color[1],
                b: ray.color[2],
                a: ray.color[3],
            });

            if is_selected {
                for (idx, anchor) in [&ray.start, &ray.end].iter().enumerate() {
                    let is_hovered = hovered_anchor.map_or(false, |(id, i)| id == ray.id && i == idx);
                    anchors.push(AnchorGpu {
                        x: anchor.candle_index * candle_spacing,
                        y: anchor.price,
                        is_hovered: u32::from(is_hovered),
                        is_selected: 1,
                    });
                }
            }
        }
        Drawing::Rectangle(rect) => {
            rects.push(DrawingRectGpu {
                x_min: rect.x_min() * candle_spacing,
                y_min: rect.y_min(),
                x_max: rect.x_max() * candle_spacing,
                y_max: rect.y_max(),
                fill_r: rect.fill_color[0],
                fill_g: rect.fill_color[1],
                fill_b: rect.fill_color[2],
                fill_a: rect.fill_color[3],
                border_r: rect.border_color[0],
                border_g: rect.border_color[1],
                border_b: rect.border_color[2],
                border_a: rect.border_color[3],
            });

            if is_selected {
                for (idx, anchor) in [&rect.corner1, &rect.corner2].iter().enumerate() {
                    let is_hovered = hovered_anchor.map_or(false, |(id, i)| id == rect.id && i == idx);
                    anchors.push(AnchorGpu {
                        x: anchor.candle_index * candle_spacing,
                        y: anchor.price,
                        is_hovered: u32::from(is_hovered),
                        is_selected: 1,
                    });
                }
            }
        }
    }
}
