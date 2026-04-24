//! Space-agnostic collision algorithms shared across Euclidean
//! narrowphases.
//!
//! - [`vector_ops`] — the vector-algebra trait shared between 3D and
//!   4D (and beyond), keeping GJK and most of EPA dimension-generic.
//! - [`gjk`] — the Gilbert-Johnson-Keerthi containment test for convex
//!   shapes via their Minkowski difference. Used for
//!   polytope-polytope narrowphase where analytical solutions
//!   (sphere-sphere, sphere-halfspace) don't apply.
//! - *(upcoming)* `epa` — Expanding Polytope Algorithm for penetration
//!   depth and contact normal from a GJK-terminating simplex.
//! - *(upcoming)* `manifold` — persistent contact cache for stable
//!   stacking over multiple frames.

pub mod epa;
pub mod gjk;
pub mod vector_ops;

pub use epa::{epa, ContactInfo};
pub use gjk::{gjk_intersect, ConvexHull, GjkResult, MinkowskiPoint, Sphere, SupportFn};
pub use vector_ops::VectorOps;
