use loam_math::Projection;
use loam_render::raymarch::RaymarchShape;
use loam_shape::polytope::Polytope4;

use crate::catalog::ShapeEntry;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum SurfaceMode {
    /// Exact, much faster on the 120/600-cell than the SDF, and sidesteps the
    /// cell120/600 face-plane BUG in `loam_physics::euclidean_r4`.
    #[default]
    Raster,
    /// Kept for visual comparison; slower on the 120/600-cell and carries the
    /// face-plane BUG.
    Sdf,
    Off,
}

impl SurfaceMode {
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        match token {
            "raster" => Some(SurfaceMode::Raster),
            "sdf" => Some(SurfaceMode::Sdf),
            "off" => Some(SurfaceMode::Off),
            _ => None,
        }
    }

    pub(crate) fn uses_sdf_for_polychora(self) -> bool {
        matches!(self, SurfaceMode::Sdf)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct SectionLayer {
    pub(crate) perimeter: bool,
    /// Per-fragment fill alpha in `[0, 1]`. `0.0` skips the fill; `(0, 1)` goes
    /// through the depth-write-disabled translucent pipeline; `1.0` is opaque
    /// with depth-write.
    pub(crate) surface_alpha: f32,
}

impl SectionLayer {
    pub(crate) fn fill_visible(self) -> bool {
        self.surface_alpha > 0.0
    }
}

const CROSS_SECTION_DEFAULT_ALPHA: f32 = 1.0;

impl SectionLayer {
    pub(crate) const CROSS_SECTION_DEFAULT: SectionLayer = SectionLayer {
        perimeter: true,
        surface_alpha: CROSS_SECTION_DEFAULT_ALPHA,
    };
    pub(crate) const PROJECTED_CAP_DEFAULT: SectionLayer = SectionLayer {
        perimeter: false,
        surface_alpha: 0.0,
    };
}

// The honest cross-section is ALWAYS drop-w
// ([`loam_math::Projection::Identity`]); the slice is a 3-flat and drop-w is
// its undistorted view, matching the SDF raymarch. The projected cap follows
// the active wireframe `projection`.
pub(crate) fn section_layer_projection(
    is_cross_section: bool,
    projection: Projection<4>,
) -> Projection<4> {
    if is_cross_section {
        Projection::Identity
    } else {
        projection
    }
}

// Blocked from SDF for TWO reasons, either sufficient: their SDFs (120 / 600
// face hyperplanes each, against the per-pixel Wolfe-greedy projection) overrun
// the browser WebGPU shader budget and crash the tab; and their
// `cell{120,600}_face_planes` are the known-wrong dual-vertex approximation
// (see `loam_shape::polytope_geom`). Do NOT re-enable on a perf fix alone.
pub(crate) fn row_blocks_sdf(row: &[ShapeEntry]) -> bool {
    row.iter().any(|e| {
        matches!(
            e.shape,
            RaymarchShape::Polytope(Polytope4::Cell120 | Polytope4::Cell600)
        )
    })
}
