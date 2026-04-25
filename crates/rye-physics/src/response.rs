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

// Note: the per-contact velocity-impulse solver lives on the
// `PhysicsSpace` trait (see `integrator.rs`), decomposed into building
// blocks (`velocity_at_point`, `effective_mass_inv`, `apply_contact_impulse`)
// because the angular-velocity / inertia types differ per space (scalar
// in 2D, bivector in 3D, eigendecomposed bivector in 4D). This module
// owns the space-agnostic positional correction and the shared
// friction coefficient used by both the legacy single-pass solve and
// the new PGS loop.

/// Coulomb friction coefficient applied uniformly across spaces. 0.35
/// reads as "moderate grip" — shapes roll under gravity rather than
/// slide indefinitely. A per-material-pair coefficient is a future
/// extension (track on `RigidBody` and combine via geometric mean).
pub const FRICTION_COEFF: f32 = 0.35;

/// Result of narrowphase collision detection between two bodies.
pub struct Contact<S: PhysicsSpace> {
    /// Unit vector from body A toward body B, in A's tangent space.
    /// Points outward from the contact — pushing along this normal
    /// separates the bodies.
    pub normal: S::Vector,

    /// The world-space point at which the contact is applied. Needed
    /// by the solver to compute angular response (torque = r × j).
    pub point: S::Point,

    /// How far the bodies overlap. Positive when they do.
    pub penetration: f32,

    /// Combined coefficient of restitution for this pair.
    pub restitution: f32,
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
pub fn correct_position<S>(
    a: &mut RigidBody<S>,
    b: &mut RigidBody<S>,
    contact: &Contact<S>,
    space: &S,
) where
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
