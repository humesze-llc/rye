use crate::integrator::PhysicsSpace;

/// Uniform across spaces. 0.35 reads as moderate grip: shapes roll under
/// gravity rather than slide indefinitely.
pub const FRICTION_COEFF: f32 = 0.35;

pub struct Contact<S: PhysicsSpace> {
    /// Unit, from body A toward body B, in A's tangent space.
    pub normal: S::Vector,

    pub point: S::Point,

    /// Positive when the bodies overlap.
    pub penetration: f32,

    pub restitution: f32,
}
