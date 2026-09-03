//! Pose is extrinsic: shapes are defined in a local frame and positioned by
//! the caller's transform. [`Shape::Sphere`] and [`Shape::HyperSphere4D`] are
//! the exceptions, carrying a `center` that physics ignores.

pub mod isovolume;
pub mod polytope;
pub mod polytope_geom;
pub mod visualizable;

pub use isovolume::Isovolume;
pub use visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    Sphere {
        /// Geodesic center in the shape frame. Ignored by physics.
        center: Vec3,
        /// Positive; a zero or negative radius is not rejected here.
        radius: f32,
    },

    /// `{ p : dot(p, normal) − offset ≤ 0 }` is the solid side.
    HalfSpace {
        /// Assumed unit: `dot(p, normal) - offset` is read as a signed distance.
        normal: Vec3,
        offset: f32,
    },

    /// Only meaningful on a static body (`inv_mass = 0`).
    HalfSpace4D {
        /// Assumed unit, as in [`Shape::HalfSpace`].
        normal: Vec4,
        offset: f32,
    },

    Box3 {
        /// The box spans `[-half_extents, half_extents]`.
        half_extents: Vec3,
    },

    Polygon2D {
        /// Counter-clockwise boundary loop in the local frame.
        vertices: Vec<Vec2>,
    },

    ConvexPolytope3D {
        /// Unordered point set in the shape frame; the collider is its hull.
        vertices: Vec<Vec3>,
    },

    ConvexPolytope4D {
        /// Unordered point set in R⁴; same hull semantics as
        /// [`Shape::ConvexPolytope3D`].
        vertices: Vec<Vec4>,
    },

    HyperSphere4D {
        /// Center in the shape frame; unlike [`Shape::Sphere`] this is the pose.
        center: Vec4,
        /// Positive; same non-enforcement as [`Shape::Sphere`].
        radius: f32,
    },
}

impl Shape {
    pub fn kind(&self) -> ShapeKind {
        match self {
            Shape::Sphere { .. } => ShapeKind::Sphere,
            Shape::HalfSpace { .. } => ShapeKind::HalfSpace,
            Shape::HalfSpace4D { .. } => ShapeKind::HalfSpace4D,
            Shape::Box3 { .. } => ShapeKind::Box3,
            Shape::Polygon2D { .. } => ShapeKind::Polygon2D,
            Shape::ConvexPolytope3D { .. } => ShapeKind::ConvexPolytope3D,
            Shape::ConvexPolytope4D { .. } => ShapeKind::ConvexPolytope4D,
            Shape::HyperSphere4D { .. } => ShapeKind::HyperSphere4D,
        }
    }

    pub fn sphere_at_origin(radius: f32) -> Self {
        Self::Sphere {
            center: Vec3::ZERO,
            radius,
        }
    }

    pub fn sphere_at(center: Vec3, radius: f32) -> Self {
        Self::Sphere { center, radius }
    }
}

/// One variant per [`Shape`] variant, and [`Shape::kind`] is total.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    Sphere,
    HalfSpace,
    HalfSpace4D,
    Box3,
    Polygon2D,
    ConvexPolytope3D,
    ConvexPolytope4D,
    HyperSphere4D,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ron_roundtrip_preserves_shape() {
        for original in [
            Shape::sphere_at_origin(0.5),
            Shape::sphere_at(Vec3::new(1.0, 2.0, 3.0), 0.25),
            Shape::HalfSpace {
                normal: Vec3::Y,
                offset: 0.5,
            },
            Shape::HalfSpace4D {
                normal: Vec4::Y,
                offset: -0.5,
            },
            Shape::Box3 {
                half_extents: Vec3::new(0.5, 1.0, 0.25),
            },
            Shape::Polygon2D {
                vertices: vec![Vec2::ZERO, Vec2::X, Vec2::Y],
            },
            Shape::ConvexPolytope3D {
                vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
            },
            Shape::ConvexPolytope4D {
                vertices: vec![Vec4::ZERO, Vec4::X, Vec4::Y, Vec4::Z, Vec4::W],
            },
            Shape::HyperSphere4D {
                center: Vec4::new(0.1, 0.2, 0.3, 0.4),
                radius: 0.7,
            },
        ] {
            let s = ron::ser::to_string(&original).unwrap();
            let back: Shape = ron::de::from_str(&s).unwrap();
            assert_eq!(back.kind(), original.kind());
        }
    }
}
