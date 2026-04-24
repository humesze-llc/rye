//! Contacts and impulse-based response.
//!
//! A [`Contact`] is the output of narrowphase collision detection. The
//! solver applies an instantaneous impulse to both bodies along the
//! contact normal, with magnitude scaled by combined restitution and
//! inverse mass.
//!
//! Linear response only at the skeleton stage — angular impulse (torque
//! from off-center contacts) lands once the first collider type that
//! rotates non-trivially (polygon, polyhedron) is wired in.

use crate::body::RigidBody;
use crate::integrator::PhysicsSpace;

/// Result of narrowphase collision detection between two bodies.
pub struct Contact<S: PhysicsSpace> {
    /// Unit vector from body A toward body B, in A's tangent space.
    /// Points outward from the contact — pushing along this normal
    /// separates the bodies.
    pub normal: S::Vector,

    /// How far the bodies overlap. Positive when they do.
    pub penetration: f32,

    /// Combined coefficient of restitution for this pair.
    pub restitution: f32,
}

/// Apply a linear impulse along the contact normal to two bodies.
/// Standard 1-D collision formula:
///
///   j = −(1 + e)·(v_rel · n) / (m⁻¹_a + m⁻¹_b)
///
/// where `v_rel = v_b − v_a` is relative velocity in A's frame. The
/// impulse is then added to B's velocity and subtracted from A's.
pub fn apply_impulse<S>(a: &mut RigidBody<S>, b: &mut RigidBody<S>, contact: &Contact<S>)
where
    S: PhysicsSpace,
    S::Vector: Copy
        + std::ops::Add<Output = S::Vector>
        + std::ops::Sub<Output = S::Vector>
        + std::ops::Mul<f32, Output = S::Vector>,
    S::Vector: DotProduct,
{
    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum <= 0.0 {
        // Two static bodies. Nothing to resolve.
        return;
    }

    let v_rel = b.velocity - a.velocity;
    let v_along_normal = v_rel.dot(contact.normal);

    if v_along_normal >= 0.0 {
        // Bodies already separating — no impulse.
        return;
    }

    let j = -(1.0 + contact.restitution) * v_along_normal / inv_mass_sum;
    let impulse = contact.normal * j;

    a.velocity = a.velocity - impulse * a.inv_mass;
    b.velocity = b.velocity + impulse * b.inv_mass;
}

/// Positional correction (Baumgarte): after velocities are resolved,
/// directly shift bodies apart by a fraction of the remaining
/// penetration. Prevents the slow sinking that impulse-only solvers
/// exhibit under persistent forces like gravity.
///
/// - `SLOP`: small penetration we tolerate without correction (avoids
///   jitter at rest).
/// - `PERCENT`: fraction of the over-slop penetration to resolve each
///   frame (smaller = smoother but slower convergence).
pub fn correct_position<S>(a: &mut RigidBody<S>, b: &mut RigidBody<S>, contact: &Contact<S>, space: &S)
where
    S: PhysicsSpace,
    S::Vector: Copy + std::ops::Mul<f32, Output = S::Vector>,
{
    const SLOP: f32 = 0.005;
    const PERCENT: f32 = 0.4;

    let inv_mass_sum = a.inv_mass + b.inv_mass;
    if inv_mass_sum <= 0.0 {
        return;
    }

    let magnitude = (contact.penetration - SLOP).max(0.0) * PERCENT / inv_mass_sum;
    if magnitude <= 0.0 {
        return;
    }
    let correction = contact.normal * magnitude;

    // A moves against the normal (away from B), B moves along it.
    a.position = space.exp(a.position, correction * (-a.inv_mass));
    b.position = space.exp(b.position, correction * b.inv_mass);
}

/// Trait for the dot product on a vector type. Concrete impls for
/// `glam::Vec2`, `Vec3`, `Vec4` below. Allows `apply_impulse` to be
/// generic over the space's vector type.
pub trait DotProduct: Copy {
    fn dot(self, rhs: Self) -> f32;
}

impl DotProduct for glam::Vec2 {
    fn dot(self, rhs: Self) -> f32 {
        glam::Vec2::dot(self, rhs)
    }
}

impl DotProduct for glam::Vec3 {
    fn dot(self, rhs: Self) -> f32 {
        glam::Vec3::dot(self, rhs)
    }
}

impl DotProduct for glam::Vec4 {
    fn dot(self, rhs: Self) -> f32 {
        glam::Vec4::dot(self, rhs)
    }
}
