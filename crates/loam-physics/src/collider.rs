//! The `Collider::Sphere` variant carries an extra `center: Vec3` field that's unused by physics
//! (the body's position is the center). Construct physics spheres via
//! [`loam_shape::Shape::sphere_at_origin`] or pattern-match with `Shape::Sphere { radius, .. }`
//! to ignore it.

pub use loam_shape::{Shape as Collider, ShapeKind as ColliderKind};
