//! The [`PhysicsSpace`] trait and the generic integration function.
//!
//! `PhysicsSpace` extends [`rye_math::Space`] with what physics needs on
//! top of kinematics: an angular-velocity type, an inertia type, and an
//! orientation-integration rule. Everything else — position integration,
//! velocity transport, gravity, collision — is written against the
//! `Space` trait and works unchanged across E², E³, H³, S³, etc.

use std::ops::Mul;

use rye_math::{Bivector, Space};

use crate::body::RigidBody;
use crate::response::Contact;

/// A [`Space`] equipped with the rotation-dynamics machinery physics
/// needs: angular velocity, inertia, and a way to integrate orientation
/// over a timestep.
///
/// New spaces opt into physics by implementing this trait. Sphere-sphere
/// collision works immediately via [`rye_math::Space::distance`] and
/// [`rye_math::Space::log`]; polygon/polyhedron collision requires per-
/// space narrowphase functions registered in [`crate::Narrowphase`].
pub trait PhysicsSpace: Space {
    /// Bivector representing angular velocity (e.g. [`rye_math::Bivector2`]
    /// for 2D, [`rye_math::Bivector3`] for 3D).
    type AngVel: Bivector;

    /// Inertia representation. Scalar in 2D; a 3×3 symmetric matrix in
    /// 3D; a 6×6 bivector-to-bivector map in 4D. Kept opaque here — the
    /// implementor decides the layout.
    type Inertia: Copy;

    /// Integrate orientation by angular velocity over a timestep.
    /// Returns the new orientation.
    fn integrate_orientation(&self, iso: Self::Iso, omega: Self::AngVel, dt: f32) -> Self::Iso;

    /// Apply the inverse inertia to a torque-bivector. Used by the
    /// solver for `ω ← ω + I⁻¹τ dt`.
    fn apply_inv_inertia(&self, inertia: Self::Inertia, torque: Self::AngVel) -> Self::AngVel;

    /// Apply a velocity impulse (linear + angular) for one contact.
    ///
    /// The generic-solver version lives on the trait because the
    /// calculation depends on the specific shapes of `AngVel` and
    /// `Inertia` — a 2D scalar, a 3D bivector, a 4D bivector — that
    /// don't share arithmetic bounds cleanly. Each space's impl
    /// computes relative velocity at the contact point, the full
    /// linear+angular impulse denominator, and friction.
    ///
    /// Positional correction (Baumgarte) is applied separately by
    /// [`crate::correct_position`].
    fn resolve_contact(
        &self,
        a: &mut RigidBody<Self>,
        b: &mut RigidBody<Self>,
        contact: &Contact<Self>,
    ) where
        Self: Sized;
}

/// Default integration step: advance position along the geodesic,
/// parallel-transport velocity to the new tangent space, and integrate
/// orientation.
///
/// This is the Space-generic integration loop — it calls only
/// [`rye_math::Space::exp`], [`rye_math::Space::parallel_transport`],
/// and [`PhysicsSpace::integrate_orientation`].
pub fn integrate_body<S>(space: &S, body: &mut RigidBody<S>, dt: f32)
where
    S: PhysicsSpace,
    S::Vector: Mul<f32, Output = S::Vector>,
{
    if body.inv_mass == 0.0 {
        // Static body. Zero velocity, don't integrate.
        return;
    }

    let p_old = body.position;
    let v_dt = body.velocity * dt;
    let p_new = space.exp(p_old, v_dt);
    body.velocity = space.parallel_transport(p_old, p_new, body.velocity);
    body.position = p_new;
    body.orientation = space.integrate_orientation(body.orientation, body.angular_velocity, dt);
}
