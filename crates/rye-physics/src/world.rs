//! [`World<S>`] — top-level container. Owns bodies, force fields, and the
//! narrowphase dispatch table; runs one simulation tick per [`World::step`].
//!
//! ## Step pipeline
//!
//! Each tick runs phases in a fixed order:
//!
//! 1. **Apply forces** — sample every [`ForceField`] at each body's
//!    position, accumulate `F·dt/m` into velocities.
//! 2. **Integrate** — advance position and orientation via
//!    [`crate::integrate_body`]. Position uses `space.exp`, velocity
//!    parallel-transports, orientation integrates per the space's rule.
//! 3. **Broadphase** — O(n²) all-pairs for now. Grid / BVH come in when
//!    body counts demand it.
//! 4. **Narrowphase** — dispatch through [`crate::Narrowphase`] for
//!    each candidate pair.
//! 5. **Solve contacts** — apply impulses via [`crate::apply_impulse`].
//!
//! Each phase is exposed as a method so games / test harnesses can
//! substitute individual phases without forking the step loop.

use std::ops::Mul;

use crate::body::RigidBody;
use crate::field::ForceField;
use crate::integrator::{integrate_body, PhysicsSpace};
use crate::narrowphase::Narrowphase;
use crate::response::correct_position;

pub struct World<S: PhysicsSpace> {
    pub space: S,
    pub bodies: Vec<RigidBody<S>>,
    pub fields: Vec<Box<dyn ForceField<S>>>,
    pub narrowphase: Narrowphase<S>,
    pub time: f32,
}

impl<S: PhysicsSpace> World<S> {
    pub fn new(space: S) -> Self {
        Self {
            space,
            bodies: Vec::new(),
            fields: Vec::new(),
            narrowphase: Narrowphase::new(),
            time: 0.0,
        }
    }

    /// Add a body to the world; returns its index.
    pub fn push_body(&mut self, body: RigidBody<S>) -> usize {
        let id = self.bodies.len();
        self.bodies.push(body);
        id
    }

    /// Add a force field to the world.
    pub fn push_field(&mut self, field: Box<dyn ForceField<S>>) {
        self.fields.push(field);
    }

    /// Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32)
    where
        S::Vector: Copy
            + std::ops::Add<Output = S::Vector>
            + Mul<f32, Output = S::Vector>,
    {
        self.apply_forces(dt);
        self.integrate(dt);

        let pairs = self.broadphase();
        for (i, j) in pairs {
            // Disjoint mut borrows via split_at_mut.
            let (left, right) = self.bodies.split_at_mut(j);
            let a = &mut left[i];
            let b = &mut right[0];
            if let Some(contact) = self.narrowphase.test(a, b, &self.space) {
                self.space.resolve_contact(a, b, &contact);
                correct_position(a, b, &contact, &self.space);
            }
        }

        self.time += dt;
    }

    fn apply_forces(&mut self, dt: f32)
    where
        S::Vector: Copy + std::ops::Add<Output = S::Vector> + Mul<f32, Output = S::Vector>,
    {
        for body in &mut self.bodies {
            if body.inv_mass == 0.0 {
                continue;
            }
            for field in &self.fields {
                let f = field.force_at(body, self.time);
                // v += F·(dt/m) = (F·dt)·inv_mass
                body.velocity = body.velocity + f * (dt * body.inv_mass);
            }
        }
    }

    fn integrate(&mut self, dt: f32)
    where
        S::Vector: Mul<f32, Output = S::Vector>,
    {
        for body in &mut self.bodies {
            integrate_body(&self.space, body, dt);
        }
    }

    /// All-pairs broadphase. Returns `(i, j)` pairs with `i < j`.
    /// Replace with a grid / BVH when body counts demand it.
    fn broadphase(&self) -> Vec<(usize, usize)> {
        let n = self.bodies.len();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                // Skip static-static pairs — they can't generate useful
                // contacts since neither can move.
                if self.bodies[i].inv_mass == 0.0 && self.bodies[j].inv_mass == 0.0 {
                    continue;
                }
                pairs.push((i, j));
            }
        }
        pairs
    }
}
