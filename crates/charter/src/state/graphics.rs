//! GPU and rendering resources.
//!
//! This module contains all GPU-related state including the wgpu surface,
//! device, queue, and the chart renderer.

use charter_render::ChartRenderer;

/// Graphics state containing all GPU resources.
///
/// This struct owns the core wgpu objects needed for rendering:
/// - Surface for presenting to the window
/// - Device for creating GPU resources
/// - Queue for submitting commands
/// - Surface configuration
/// - The main chart renderer
pub struct GraphicsState {
    /// The wgpu surface for presenting frames.
    pub surface: wgpu::Surface<'static>,

    /// The wgpu device for creating resources.
    pub device: wgpu::Device,

    /// The wgpu queue for submitting commands.
    pub queue: wgpu::Queue,

    /// Current surface configuration.
    pub config: wgpu::SurfaceConfiguration,

    /// Whether the surface has been configured at least once.
    pub is_surface_configured: bool,

    /// The main chart renderer.
    pub renderer: ChartRenderer,
}

impl GraphicsState {
    /// Create new graphics state with the given wgpu objects.
    pub fn new(
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        renderer: ChartRenderer,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            renderer,
        }
    }

    /// Resize the surface and update the renderer.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.renderer.resize(width, height);
            self.renderer.update_camera(&self.queue);
        }
    }

    /// Get the current surface dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Get the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }
}
