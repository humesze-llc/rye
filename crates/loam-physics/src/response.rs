use crate::integrator::PhysicsSpace;

/// Coulomb friction coefficient applied uniformly across spaces. 0.35 reads as "moderate grip":
/// shapes roll under gravity rather than slide indefinitely.
pub const FRICTION_COEFF: f32 = 0.35;

pub struct Contact<S: PhysicsSpace> {
    /// Unit vector from body A toward body B, in A's tangent space.
    pub normal: S::Vector,

    /// The world-space point at which the contact is applied.
    pub point: S::Point,

    /// How far the bodies overlap. Positive when they do.
    pub penetration: f32,

    pub restitution: f32,
}
