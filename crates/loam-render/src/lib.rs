//! Thin wgpu wrapper plus a tiny render-graph harness.

pub mod composite;
pub mod depth;
pub mod device;
pub mod gizmo;
pub mod gpu_timer;
pub mod graph;
pub mod hypergimbal;
pub mod lattice;
pub mod line_raster;
pub mod line_raster_static_r4;
pub mod point_raster;
pub mod raymarch;
pub mod triangle_raster;

pub use depth::DepthBuffer;
pub use lattice::Viewport;
pub use line_raster::{LineRasterNode, LineRasterUniforms};
pub use line_raster_static_r4::{LineRasterStaticR4Node, LineRasterStaticR4Uniforms};
pub use point_raster::{PointRasterNode, PointRasterUniforms};
pub use raymarch::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
pub use triangle_raster::{
    FragmentShading, TriangleRasterNode, TriangleRasterUniforms, TriangleVertex,
};

/// How a rasterizer pipeline interacts with depth. Three states only -- avoids the
/// invalid-combination problem an `Option<TextureFormat> + bool depth_write` API would
/// have. Shared by [`LineRasterNode`] and [`TriangleRasterNode`].
#[derive(Copy, Clone, Debug)]
pub enum DepthMode {
    Off,
    ReadWrite { format: wgpu::TextureFormat },
    ReadOnly { format: wgpu::TextureFormat },
}

impl DepthMode {
    /// Format of the depth attachment, if any.
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        match self {
            DepthMode::Off => None,
            DepthMode::ReadWrite { format } | DepthMode::ReadOnly { format } => Some(*format),
        }
    }

    /// `true` for any depth-aware mode (read-only or read-write).
    pub fn is_active(&self) -> bool {
        !matches!(self, DepthMode::Off)
    }

    /// Whether the pipeline should write depth.
    pub fn writes(&self) -> bool {
        matches!(self, DepthMode::ReadWrite { .. })
    }
}
