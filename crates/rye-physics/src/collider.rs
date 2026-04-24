//! Collider shapes.
//!
//! The variants of [`Collider`] are space-agnostic data — just the raw
//! shape parameters. Their *behavior* under a given space lives in
//! narrowphase functions registered with [`crate::Narrowphase`].
//!
//! New collider types are added by:
//! 1. Adding a variant here.
//! 2. Adding a [`ColliderKind`] discriminant.
//! 3. Registering narrowphase functions in `Narrowphase::register(kind_a,
//!    kind_b, fn)`.
//!
//! Nothing in the solver or `World` changes.

use glam::{Vec2, Vec3};

/// All collider shapes supported by Rye's physics. Space-agnostic data —
/// the same `Polygon2D` can live in `EuclideanR2` or (in principle) in
/// a curved 2D space, provided a narrowphase function has been
/// registered for it in that space.
#[derive(Clone, Debug)]
pub enum Collider {
    /// Sphere in any N-dim space. Radius only; the body's position is
    /// the center.
    Sphere { radius: f32 },

    /// Convex polygon in 2D Euclidean space. Vertices in local body
    /// frame (orientation-relative), ordered counter-clockwise.
    Polygon2D { vertices: Vec<Vec2> },

    /// Half-space `{ p : dot(p, normal) ≥ offset }` in any dimension
    /// whose `Vector` is `Vec3`. The complement is the "solid" side.
    /// Used for infinite floors/walls until polyhedron SAT ships; also
    /// useful long-term for static terrain planes.
    ///
    /// Only meaningful on a static body (`inv_mass == 0`) — dynamic
    /// half-spaces are nonsensical.
    HalfSpace { normal: Vec3, offset: f32 },

    /// Convex polytope in 3D — arbitrary vertex list, assumed convex.
    /// GJK only needs the vertex list (support function returns the
    /// vertex with max dot against the query direction); EPA reuses
    /// the same support function for penetration depth.
    ///
    /// Vertices are in body-local coordinates. The body's `position`
    /// and `orientation` transform them to world space per query.
    /// Winding and face structure aren't required for GJK or EPA.
    ConvexPolytope3D { vertices: Vec<Vec3> },

    // Future:
    // Horosphere { point_at_inf: Vec3, offset: f32 },    // H³-only
    // ConvexPolytope4D { vertices: Vec<Vec4> },          // for Simplex 4D
}

impl Collider {
    pub fn kind(&self) -> ColliderKind {
        match self {
            Collider::Sphere { .. } => ColliderKind::Sphere,
            Collider::Polygon2D { .. } => ColliderKind::Polygon2D,
            Collider::HalfSpace { .. } => ColliderKind::HalfSpace,
            Collider::ConvexPolytope3D { .. } => ColliderKind::ConvexPolytope3D,
        }
    }
}

/// Discriminant used as the key for [`crate::Narrowphase`] dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColliderKind {
    Sphere,
    Polygon2D,
    HalfSpace,
    ConvexPolytope3D,
}
