//! `loam-shape`: the canonical geometric-primitive data model.
//!
//! Before this crate existed, `loam_scene::PrimitiveKind` (for rendering) and
//! `loam_physics::Collider` (for collision) each defined their own parallel enum of shape
//! types. Adding a new shape, say a horosphere for H³, meant touching both, keeping their
//! variant lists in sync by hand, and inventing new conversion glue. This crate is the single
//! source of truth they both now alias to.
//!
//! ## Design
//!
//! - **One enum, all variants.** A `Shape` carries every shape either role needs. Variants that
//!   don't apply to a particular role; e.g. [`Shape::Polygon2D`] has no 3D SDF emission,
//!   [`Shape::Box3`] has no dedicated physics narrowphase today; are simply not implemented by
//!   that role's trait and return `None` / no-op.
//! - **Pose is extrinsic.** Most shapes (Sphere, Box3, the polytopes) are defined in a local
//!   "shape frame" and positioned by the caller's transform: the physics body's
//!   `position`+`orientation`, or an SDF scene node's transform. The one exception is
//!   [`Shape::Sphere`], which carries a `center` field so SDF scenes can place spheres without a
//!   transform combinator. Physics ignores that field (it always uses the body's position), the
//!   physics sphere constructors set `center = Vec3::ZERO`.
//! - **No behavior, but interfaces are OK.** This crate defines the [`Shape`] data and the
//!   [`Visualizable`] trait *interface*. Trait definitions count as data-shape interfaces, not
//!   behavior; they add zero dependencies on application-level code. Role impls live in
//!   the role crates (`loam-scene` for `Primitive` (SDF); `loam-physics` for `Collider`).
//!   [`Visualizable`] follows the data instead of the role, so [`polytope::Polytope4`]'s impl
//!   sits in this crate: it reads only the topology this crate owns, which lets the renderer
//!   draw a polychoron without depending on the simulation layer. The dep graph stays a tree.

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
    /// Sphere with a local center and radius. In SDF scenes `center` is the geodesic center; in
    /// physics `center` is ignored (body position is the center) and conventionally set to
    /// [`Vec3::ZERO`].
    Sphere {
        /// Geodesic center in the shape frame. Ignored by physics.
        center: Vec3,
        /// Positive; a zero or negative radius is not rejected here and
        /// yields a degenerate SDF and no contact manifold.
        radius: f32,
    },

    /// A half-space `{ p : dot(p, normal) − offset ≤ 0 }`, equivalent to a totally-geodesic plane
    /// with the "solid" side picked by sign convention. Unifies SDF's `Plane` and physics's
    /// `HalfSpace`.
    HalfSpace {
        /// Assumed unit: `dot(p, normal) - offset` is read directly as a
        /// signed distance, which a non-unit normal rescales.
        normal: Vec3,
        /// Signed distance from the origin to the plane along `normal`.
        offset: f32,
    },

    /// 4D half-space: same convention as [`Shape::HalfSpace`] but with a `Vec4` normal, used by
    /// the 4D physics ground in the pentatope-falls demo. Only meaningful on a static body
    /// (`inv_mass = 0`); a dynamic half-space isn't physically sensible.
    HalfSpace4D {
        /// Assumed unit, as in [`Shape::HalfSpace`].
        normal: Vec4,
        /// Signed distance from the origin to the 3-flat along `normal`.
        offset: f32,
    },

    /// Axis-aligned 3D box, centered at the origin of its local frame. SDF emits the standard
    /// Euclidean-box formula; physics prefers the equivalent 8-vertex [`Shape::ConvexPolytope3D`]
    /// today but may grow a dedicated narrowphase later.
    Box3 {
        /// Per-axis distance from the local origin to each face, so the box
        /// spans `[-half_extents, half_extents]`.
        half_extents: Vec3,
    },

    /// Convex 2D polygon, counter-clockwise vertices in the local frame. Physics 2D narrowphase
    /// uses SAT on this.
    Polygon2D {
        /// Boundary loop in the local frame. Convexity is a precondition
        /// SAT cannot detect the violation of; fewer than three vertices
        /// yields no contact at all rather than an error.
        vertices: Vec<Vec2>,
    },

    /// Convex 3D polytope, arbitrary vertex list, assumed convex. Physics 3D narrowphase uses
    /// GJK+EPA; SDF has no emission for this variant today.
    ConvexPolytope3D {
        /// Unordered point set in the shape frame. The collider is its
        /// convex hull, so a non-convex list silently collides as the hull
        /// and interior points only cost support-function time.
        vertices: Vec<Vec3>,
    },

    /// Convex 4D polytope. Physics 4D narrowphase uses 4D GJK+EPA; SDF emission via
    /// `loam_scene::Primitive4` (max-of-half-spaces).
    ConvexPolytope4D {
        /// Unordered point set in R⁴; same hull semantics as
        /// [`Shape::ConvexPolytope3D`].
        vertices: Vec<Vec4>,
    },

    /// 4D ball with a local centre and radius, the 4D analogue of [`Shape::Sphere`]. SDF:
    /// `length(p - center) - radius` in `vec4`. Physics narrowphase reuses the `Sphere` path
    /// with a `Vec4` centre via the body position; this variant is for SDF scene authoring
    /// (`Scene4`) where pose is encoded in the shape rather than a transform combinator.
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

    /// Convenience constructor: a sphere at the origin of its local frame. The physics
    /// convention, where the body's `position` is the sphere's center, always constructs spheres
    /// this way.
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
    fn kind_matches_variant() {
        assert_eq!(Shape::sphere_at_origin(0.5).kind(), ShapeKind::Sphere);
        assert_eq!(
            Shape::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0
            }
            .kind(),
            ShapeKind::HalfSpace
        );
        assert_eq!(
            Shape::Box3 {
                half_extents: Vec3::splat(1.0)
            }
            .kind(),
            ShapeKind::Box3
        );
        assert_eq!(
            Shape::Polygon2D { vertices: vec![] }.kind(),
            ShapeKind::Polygon2D
        );
        assert_eq!(
            Shape::ConvexPolytope3D { vertices: vec![] }.kind(),
            ShapeKind::ConvexPolytope3D
        );
        assert_eq!(
            Shape::ConvexPolytope4D { vertices: vec![] }.kind(),
            ShapeKind::ConvexPolytope4D
        );
        assert_eq!(
            Shape::HalfSpace4D {
                normal: Vec4::Y,
                offset: 0.0
            }
            .kind(),
            ShapeKind::HalfSpace4D
        );
        assert_eq!(
            Shape::HyperSphere4D {
                center: Vec4::ZERO,
                radius: 1.0
            }
            .kind(),
            ShapeKind::HyperSphere4D
        );
    }

    #[test]
    fn ron_roundtrip_preserves_shape() {
        // Sanity: the derived serde impls work on every variant. Scenes and pair-cache files
        // lean on this. Covers all 8 variants so adding a new one without thinking about serde
        // surfaces here, not at runtime when the scene file fails to parse.
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
