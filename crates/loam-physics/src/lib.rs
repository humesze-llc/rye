//! The integration loop is written against [`loam_math::Space`] operations
//! only, so it does not assume flat space: `exp`, `parallel_transport`,
//! `distance`.

pub mod body;
pub mod collider;
pub mod field;
pub mod integrator;
pub mod manifold;
pub mod narrowphase;
pub mod response;
pub mod world;

pub mod collision;
#[cfg(test)]
mod determinism_fixture;
pub mod euclidean_r2;
pub mod euclidean_r3;
pub mod euclidean_r4;

pub use body::{BodyArena, BodyId, RigidBody};
pub use collider::{Collider, ColliderKind};
pub use field::{ForceField, Gravity};
pub use integrator::{integrate_body, PhysicsSpace};
pub use manifold::{ContactPoint, Manifold};
pub use narrowphase::{Narrowphase, NarrowphaseFn};
pub use response::{Contact, FRICTION_COEFF};
pub use world::{Island, OrderPolicy, Schedule, SchedulePhase, World};
