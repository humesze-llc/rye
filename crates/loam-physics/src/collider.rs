//! `Collider::Sphere::center` is ignored: the body position is the centre.
//! Build physics spheres with [`loam_shape::Shape::sphere_at_origin`].

pub use loam_shape::{Shape as Collider, ShapeKind as ColliderKind};
