use crate::body::RigidBody;
use crate::integrator::PhysicsSpace;

/// Forces are pure functions of body state and time.
pub trait ForceField<S: PhysicsSpace>: Send + Sync {
    fn force_at(&self, body: &RigidBody<S>, t: f32) -> S::Vector;
}

pub struct Gravity<S: PhysicsSpace> {
    pub acceleration: S::Vector,
}

impl<S: PhysicsSpace> Gravity<S> {
    pub fn new(acceleration: S::Vector) -> Self {
        Self { acceleration }
    }
}

impl<S: PhysicsSpace> ForceField<S> for Gravity<S>
where
    S::Vector: Copy + std::ops::Mul<f32, Output = S::Vector>,
{
    fn force_at(&self, body: &RigidBody<S>, _t: f32) -> S::Vector {
        // F = m·a, and the solver divides by mass again, so all bodies fall at
        // the same rate.
        self.acceleration * body.mass
    }
}
