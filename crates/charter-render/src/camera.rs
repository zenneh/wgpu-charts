//! Camera types for 2D chart rendering.

/// 2D Camera for panning and zooming the chart.
pub struct Camera {
    pub position: [f32; 2],
    pub scale: [f32; 2],
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
        }
    }

    pub fn build_view_projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let half_width = self.scale[0] * aspect;
        let half_height = self.scale[1];

        let left = self.position[0] - half_width;
        let right = self.position[0] + half_width;
        let bottom = self.position[1] - half_height;
        let top = self.position[1] + half_height;

        let sx = 2.0 / (right - left);
        let sy = 2.0 / (top - bottom);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);

        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, 0.0, 1.0],
        ]
    }

    /// Get visible X range in world coordinates.
    pub fn visible_x_range(&self, aspect: f32) -> (f32, f32) {
        let half_width = self.scale[0] * aspect;
        (self.position[0] - half_width, self.position[0] + half_width)
    }

    /// Get visible Y range in world coordinates.
    pub fn visible_y_range(&self, _aspect: f32) -> (f32, f32) {
        let half_height = self.scale[1];
        (self.position[1] - half_height, self.position[1] + half_height)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-compatible camera uniform.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.build_view_projection_matrix(aspect);
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}
