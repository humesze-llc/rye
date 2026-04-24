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

    /// Convex polyhedron in 3D Euclidean space. Vertices + face normals
    /// in local body frame.
    Polyhedron3D {
        vertices: Vec<Vec3>,
        face_normals: Vec<Vec3>,
    },
    // Future:
    // Horosphere { point_at_inf: Vec3, offset: f32 },    // H³-only
    // Polytope4D { vertices: Vec<[f32; 4]>, ... },       // for Simplex 4D
}

impl Collider {
    pub fn kind(&self) -> ColliderKind {
        match self {
            Collider::Sphere { .. } => ColliderKind::Sphere,
            Collider::Polygon2D { .. } => ColliderKind::Polygon2D,
            Collider::Polyhedron3D { .. } => ColliderKind::Polyhedron3D,
        }
    }
}

/// Discriminant used as the key for [`crate::Narrowphase`] dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ColliderKind {
    Sphere,
    Polygon2D,
    Polyhedron3D,
}
