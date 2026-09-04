use std::collections::HashMap;

use crate::body::RigidBody;
use crate::collider::ColliderKind;
use crate::integrator::PhysicsSpace;
use crate::response::Contact;

/// Always called with `a.kind()` matching the key's first component.
pub type NarrowphaseFn<S> = fn(a: &RigidBody<S>, b: &RigidBody<S>, space: &S) -> Option<Contact<S>>;

pub struct Narrowphase<S: PhysicsSpace> {
    dispatch: HashMap<(ColliderKind, ColliderKind), NarrowphaseFn<S>>,
}

impl<S: PhysicsSpace> Default for Narrowphase<S> {
    fn default() -> Self {
        Self {
            dispatch: HashMap::new(),
        }
    }
}

impl<S: PhysicsSpace> Narrowphase<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, a: ColliderKind, b: ColliderKind, f: NarrowphaseFn<S>) {
        self.dispatch.insert((a, b), f);
    }

    pub fn test(&self, a: &RigidBody<S>, b: &RigidBody<S>, space: &S) -> Option<Contact<S>>
    where
        S::Vector: std::ops::Mul<f32, Output = S::Vector>,
    {
        let key = (a.collider.kind(), b.collider.kind());
        if let Some(&f) = self.dispatch.get(&key) {
            return f(a, b, space);
        }
        let reversed = (b.collider.kind(), a.collider.kind());
        if let Some(&f) = self.dispatch.get(&reversed) {
            return f(b, a, space).map(|c| Contact {
                normal: c.normal * -1.0,
                point: c.point,
                penetration: c.penetration,
                restitution: c.restitution,
            });
        }
        None
    }
}
