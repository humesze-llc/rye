//! The persistent contact cache lives in `crate::manifold`, not here, because it depends on
//! `PhysicsSpace` (S::Vector, etc.) and these algorithms are deliberately Space-free.

pub mod epa;
pub mod epa_r4;
pub mod gjk;
pub mod gjk_r4;
pub mod simplex_r4;
pub mod vector_ops;

pub use epa::{epa, ContactInfo};
pub use epa_r4::{epa_r4, ContactInfo4};
pub use gjk::{gjk_intersect, ConvexHull, GjkResult, MinkowskiPoint, Sphere, SupportFn};
pub use gjk_r4::{
    gjk_intersect_r4, ConvexHull4, GjkResult4, MinkowskiPoint4, PosedHull4, Sphere4, SupportFn4,
};
pub use simplex_r4::{closest_to_origin as closest_to_origin_r4, Closest as ClosestR4};
pub use vector_ops::VectorOps;
