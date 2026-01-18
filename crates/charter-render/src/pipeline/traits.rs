//! Traits for GPU rendering pipelines.
//!
//! This module defines the [`Pipeline`] trait which provides a common interface
//! for all rendering pipelines in the charter-render crate.

use std::ops::Range;

/// A trait for GPU rendering pipelines.
///
/// This trait provides a common interface for rendering with wgpu pipelines.
/// All chart element pipelines (candles, volume, guidelines, indicators, TA)
/// implement this trait to enable uniform rendering patterns.
///
/// # Type Parameters
///
/// * `BindGroupData` - Associated type representing the data needed to create
///   or reference the pipeline's bind group. This allows each pipeline to
///   specify what data it needs for rendering.
///
/// # Example
///
/// ```ignore
/// impl Pipeline for CandlePipeline {
///     type BindGroupData = wgpu::BindGroup;
///
///     fn render<'a>(
///         &'a self,
///         render_pass: &mut wgpu::RenderPass<'a>,
///         camera_bind_group: &'a wgpu::BindGroup,
///         data_bind_group: &'a wgpu::BindGroup,
///         vertex_range: Range<u32>,
///         instance_range: Range<u32>,
///     ) {
///         render_pass.set_pipeline(&self.pipeline);
///         render_pass.set_bind_group(0, camera_bind_group, &[]);
///         render_pass.set_bind_group(1, data_bind_group, &[]);
///         render_pass.draw(vertex_range, instance_range);
///     }
/// }
/// ```
pub trait Pipeline {
    /// The type of bind group data this pipeline uses.
    ///
    /// This is typically `wgpu::BindGroup` but can be customized for pipelines
    /// that need additional data or different bind group configurations.
    type BindGroupData;

    /// Renders using this pipeline.
    ///
    /// This method sets up the pipeline, binds the necessary bind groups, and
    /// issues the draw call. The camera bind group is always at slot 0, and
    /// the data-specific bind group is at slot 1.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `data_bind_group` - The pipeline-specific data bind group (slot 1)
    /// * `vertex_range` - The range of vertices to draw
    /// * `instance_range` - The range of instances to draw
    ///
    /// # Lifetimes
    ///
    /// The `'a` lifetime ensures that all bind groups and the pipeline itself
    /// live at least as long as the render pass encoding.
    fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        vertex_range: Range<u32>,
        instance_range: Range<u32>,
    );

    /// Returns a reference to the underlying wgpu render pipeline.
    ///
    /// This is useful for cases where direct access to the pipeline is needed,
    /// such as when setting up multiple draw calls with different bind groups.
    fn pipeline(&self) -> &wgpu::RenderPipeline;
}

/// Extension trait for pipelines that support instanced rendering with a
/// fixed number of vertices per instance.
///
/// Many chart elements (candles, volume bars, guidelines) use a fixed number
/// of vertices per instance (e.g., 6 vertices for a quad, 18 for a candle).
/// This trait provides a convenience method that automatically uses the
/// correct vertex count.
pub trait InstancedPipeline: Pipeline {
    /// The number of vertices per instance for this pipeline.
    const VERTICES_PER_INSTANCE: u32;

    /// Renders instances using the pipeline's fixed vertex count.
    ///
    /// This is a convenience method that calls `render` with the appropriate
    /// vertex range based on `VERTICES_PER_INSTANCE`.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass to record commands into
    /// * `camera_bind_group` - The camera/view-projection bind group (slot 0)
    /// * `data_bind_group` - The pipeline-specific data bind group (slot 1)
    /// * `instance_count` - The number of instances to draw
    fn render_instances<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        data_bind_group: &'a wgpu::BindGroup,
        instance_count: u32,
    ) {
        self.render(
            render_pass,
            camera_bind_group,
            data_bind_group,
            0..Self::VERTICES_PER_INSTANCE,
            0..instance_count,
        );
    }
}

#[cfg(test)]
mod tests {
    // Trait tests would require GPU resources, so they are tested through
    // integration tests with the concrete pipeline implementations.
}
