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
pub mod sky_ground;
pub mod triangle_raster;

pub use depth::DepthBuffer;
pub use lattice::Viewport;
pub use line_raster::{LineRasterNode, LineRasterUniforms};
pub use line_raster_static_r4::{LineRasterStaticR4Node, LineRasterStaticR4Uniforms};
pub use point_raster::{PointRasterNode, PointRasterUniforms};
pub use raymarch::{GeodesicRayMarchNode, RayMarchNode, RayMarchUniforms};
pub use sky_ground::{Ground, SkyGroundNode, SkyGroundUniforms};
pub use triangle_raster::{
    FragmentShading, TriangleRasterNode, TriangleRasterUniforms, TriangleVertex,
};

#[derive(Copy, Clone, Debug)]
pub enum DepthMode {
    Off,
    ReadWrite { format: wgpu::TextureFormat },
    ReadOnly { format: wgpu::TextureFormat },
}

impl DepthMode {
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        match self {
            DepthMode::Off => None,
            DepthMode::ReadWrite { format } | DepthMode::ReadOnly { format } => Some(*format),
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, DepthMode::Off)
    }

    pub fn writes(&self) -> bool {
        matches!(self, DepthMode::ReadWrite { .. })
    }
}
