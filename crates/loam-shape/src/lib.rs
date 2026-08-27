//! `loam-shape`: the canonical geometric-primitive data model.
//!
//! Pose is extrinsic. Most shapes (Sphere, Box3, the polytopes) are defined in a local
//! "shape frame" and positioned by the caller's transform: the physics body's
//! `position`+`orientation`, or an SDF scene node's transform. The one exception is
//! [`Shape::Sphere`], which carries a `center` field so SDF scenes can place spheres without a
//! transform combinator. Physics ignores that field (it always uses the body's position), the
//! physics sphere constructors set `center = Vec3::ZERO`.

#![warn(missing_docs)]

pub mod isovolume;
pub mod polytope;
pub mod polytope_geom;
pub mod visualizable;

pub use isovolume::Isovolume;
pub use visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};

use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

/// A geometric primitive. Used by both SDF rendering and physics collision; which subset of
/// variants each role supports is documented on the per-role trait in each consumer crate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    /// Sphere with a local center and radius.
    Sphere {
        /// Geodesic center in the shape frame. Ignored by physics.
        center: Vec3,
        /// Positive; a zero or negative radius is not rejected here and
        /// yields a degenerate SDF and no contact manifold.
        radius: f32,
    },

    /// A half-space `{ p : dot(p, normal) − offset ≤ 0 }`, equivalent to a totally-geodesic plane
    /// with the "solid" side picked by sign convention.
    HalfSpace {
        /// Assumed unit: `dot(p, normal) - offset` is read directly as a
        /// signed distance, which a non-unit normal rescales.
        normal: Vec3,
        /// Signed distance from the origin to the plane along `normal`.
        offset: f32,
    },

    /// Only meaningful on a static body
    /// (`inv_mass = 0`); a dynamic half-space isn't physically sensible.
    HalfSpace4D {
        /// Assumed unit, as in [`Shape::HalfSpace`].
        normal: Vec4,
        /// Signed distance from the origin to the 3-flat along `normal`.
        offset: f32,
    },

    /// Axis-aligned 3D box, centered at the origin of its local frame.
    Box3 {
        /// Per-axis distance from the local origin to each face, so the box
        /// spans `[-half_extents, half_extents]`.
        half_extents: Vec3,
    },

    /// Convex 2D polygon, counter-clockwise vertices in the local frame.
    Polygon2D {
        /// Boundary loop in the local frame. Convexity is a precondition
        /// SAT cannot detect the violation of; fewer than three vertices
        /// yields no contact at all rather than an error.
        vertices: Vec<Vec2>,
    },

    /// Convex 3D polytope, arbitrary vertex list, assumed convex.
    ConvexPolytope3D {
        /// Unordered point set in the shape frame. The collider is its
        /// convex hull, so a non-convex list silently collides as the hull
        /// and interior points only cost support-function time.
        vertices: Vec<Vec3>,
    },

    /// Convex 4D polytope.
    ConvexPolytope4D {
        /// Unordered point set in R⁴; same hull semantics as
        /// [`Shape::ConvexPolytope3D`].
        vertices: Vec<Vec4>,
    },

    /// 4D ball with a local centre and radius, the 4D analogue of [`Shape::Sphere`].
    HyperSphere4D {
        /// Center in the shape frame; unlike [`Shape::Sphere`] this is the
        /// pose, since `Scene4` has no transform combinator to carry it.
        center: Vec4,
        /// Positive; same non-enforcement as [`Shape::Sphere`].
        radius: f32,
    },
}

impl Shape {
    /// Runtime discriminant, used by physics narrowphase dispatch and by any consumer that needs
    /// to route on shape type without pattern-matching on the enum.
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

    /// Convenience constructor: a sphere at the origin of its local frame.
    pub fn sphere_at_origin(radius: f32) -> Self {
        Self::Sphere {
            center: Vec3::ZERO,
            radius,
        }
    }

    /// Convenience constructor: an SDF-scene sphere placed at an arbitrary `center`.
    pub fn sphere_at(center: Vec3, radius: f32) -> Self {
        Self::Sphere { center, radius }
    }
}

/// Runtime discriminant of [`Shape`]. Keyed into dispatch tables by physics narrowphase and
/// (eventually) any other consumer that needs O(1) variant routing.
///
/// One variant per `Shape` variant and [`Shape::kind`] is total, so a dispatch table indexed
/// by this enum is exhaustive over shapes by construction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShapeKind {
    /// Selects [`Shape::Sphere`].
    Sphere,
    /// Selects [`Shape::HalfSpace`].
    HalfSpace,
    /// Selects [`Shape::HalfSpace4D`].
    HalfSpace4D,
    /// Selects [`Shape::Box3`].
    Box3,
    /// Selects [`Shape::Polygon2D`].
    Polygon2D,
    /// Selects [`Shape::ConvexPolytope3D`].
    ConvexPolytope3D,
    /// Selects [`Shape::ConvexPolytope4D`].
    ConvexPolytope4D,
    /// Selects [`Shape::HyperSphere4D`].
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
