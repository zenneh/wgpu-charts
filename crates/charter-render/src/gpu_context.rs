//! GPU context management for wgpu-based rendering.
//!
//! This module provides the [`GpuContext`] struct which encapsulates the core wgpu
//! resources needed for GPU rendering: device, queue, surface, and configuration.

/// Encapsulates the core wgpu resources needed for GPU rendering.
///
/// `GpuContext` owns the GPU device, command queue, render surface, and surface
/// configuration. It provides methods for common operations like resizing the
/// surface and acquiring the current frame texture.
///
/// # Example
///
/// ```ignore
/// let gpu_context = GpuContext::new(device, queue, surface, config);
/// gpu_context.resize(1920, 1080);
/// let texture = gpu_context.get_current_texture()?;
/// ```
pub struct GpuContext {
    /// The wgpu device for creating GPU resources.
    pub device: wgpu::Device,
    /// The command queue for submitting GPU commands.
    pub queue: wgpu::Queue,
    /// The render surface (typically backed by a window).
    pub surface: wgpu::Surface<'static>,
    /// Configuration for the surface (format, size, present mode, etc.).
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// Creates a new `GpuContext` from existing wgpu resources.
    ///
    /// This constructor takes ownership of the provided resources. The surface
    /// should already be configured with the provided configuration before
    /// calling this constructor, or `configure_surface` should be called
    /// immediately after.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The command queue
    /// * `surface` - The render surface
    /// * `config` - The surface configuration
    #[must_use]
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        Self {
            device,
            queue,
            surface,
            config,
        }
    }

    /// Resizes the surface to the given dimensions.
    ///
    /// This method updates the surface configuration and reconfigures the
    /// surface with the new size. Dimensions are clamped to a minimum of 1
    /// to avoid zero-sized surfaces.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels
    /// * `height` - New height in pixels
    pub fn resize(&mut self, width: u32, height: u32) {
        // Ensure dimensions are at least 1 to avoid zero-sized surfaces
        let width = width.max(1);
        let height = height.max(1);

        if self.config.width != width || self.config.height != height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Configures the surface with the current configuration.
    ///
    /// This is useful after creating a `GpuContext` with `new()` if the surface
    /// was not already configured, or if the surface needs to be reconfigured
    /// (e.g., after the window was recreated).
    pub fn configure_surface(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Gets the current texture from the surface for rendering.
    ///
    /// This acquires the next frame's texture from the surface. The returned
    /// `SurfaceTexture` should be presented after rendering is complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the surface texture cannot be acquired (e.g., if
    /// the surface is lost or outdated).
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Returns the surface texture format.
    #[must_use]
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns the current surface width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Returns the current surface height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Returns the current surface dimensions as (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Returns the aspect ratio (width / height) of the surface.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }
}

#[cfg(test)]
mod tests {
    // Note: Most tests for GpuContext require a real GPU and window,
    // which makes them difficult to run in CI. The struct is primarily
    // tested through integration tests.
}
