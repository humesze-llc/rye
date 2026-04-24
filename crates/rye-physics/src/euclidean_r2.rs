//! `impl PhysicsSpace for EuclideanR2` — 2D Euclidean rigid-body physics.
//!
//! Angular velocity is a scalar ([`Bivector2`]); inertia is the scalar
//! moment of inertia. Orientation integrates by composing a rotor
//! derived from `ω·dt`.

use glam::Vec2;

use rye_math::{Bivector, Bivector2, EuclideanR2, Iso2};

use crate::body::RigidBody;
use crate::collider::{Collider, ColliderKind};
use crate::integrator::PhysicsSpace;
use crate::narrowphase::Narrowphase;
use crate::response::Contact;

impl PhysicsSpace for EuclideanR2 {
    type AngVel = Bivector2;
    /// Scalar moment of inertia I about the body's center.
    type Inertia = f32;

    fn integrate_orientation(&self, iso: Iso2, omega: Bivector2, dt: f32) -> Iso2 {
        // θ_new = θ_old + ω·dt. Compose existing rotor with the rotor
        // built from the incremental bivector.
        let delta = (omega * dt).exp();
        Iso2 {
            rotation: iso.rotation * delta,
            translation: iso.translation,
        }
    }

    fn apply_inv_inertia(&self, inertia: f32, torque: Bivector2) -> Bivector2 {
        if inertia > 0.0 {
            torque * (1.0 / inertia)
        } else {
            Bivector2::zero()
        }
    }
}

/// Moment of inertia for a solid disk of radius `r` and mass `m`:
/// `I = ½·m·r²`. Used as the default for `Collider::Sphere` in 2D.
pub fn disk_inertia(mass: f32, radius: f32) -> f32 {
    0.5 * mass * radius * radius
}

/// Convenience: build a dynamic circular body in R².
pub fn sphere_body(
    position: Vec2,
    velocity: Vec2,
    radius: f32,
    mass: f32,
) -> RigidBody<EuclideanR2> {
    RigidBody::new(
        position,
        velocity,
        Collider::Sphere { radius },
        mass,
        disk_inertia(mass, radius),
        &EuclideanR2,
    )
}

/// Register the narrowphase functions that are meaningful in R².
/// Currently: sphere-sphere. Polygon-polygon and polygon-sphere are
/// added when 2D SAT lands.
pub fn register_default_narrowphase(np: &mut Narrowphase<EuclideanR2>) {
    np.register(ColliderKind::Sphere, ColliderKind::Sphere, sphere_sphere_r2);
}

fn sphere_sphere_r2(
    a: &RigidBody<EuclideanR2>,
    b: &RigidBody<EuclideanR2>,
    space: &EuclideanR2,
) -> Option<Contact<EuclideanR2>> {
    let Collider::Sphere { radius: ra } = a.collider else {
        return None;
    };
    let Collider::Sphere { radius: rb } = b.collider else {
        return None;
    };

    use rye_math::Space;
    let d = space.distance(a.position, b.position);
    let combined = ra + rb;
    if d >= combined {
        return None;
    }
    let log = space.log(a.position, b.position);
    let len = log.length();
    let normal = if len > 1e-8 { log / len } else { Vec2::Y };

    Some(Contact {
        normal,
        penetration: combined - d,
        restitution: (a.restitution + b.restitution) * 0.5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gravity;
    use crate::world::World;

    #[test]
    fn falling_body_accelerates_under_gravity() {
        let mut world = World::new(EuclideanR2);
        let id = world.push_body(sphere_body(Vec2::new(0.0, 5.0), Vec2::ZERO, 0.5, 1.0));
        world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

        // One tick of dt = 1/60.
        world.step(1.0 / 60.0);

        let body = &world.bodies[id];
        // After one tick: v_y ≈ −9.8/60 ≈ −0.163.
        assert!(body.velocity.y < -0.1 && body.velocity.y > -0.2);
        // Position moved down by v·dt (velocity sampled post-gravity):
        // y ≈ 5 + (−0.163)·(1/60) ≈ 4.9973.
        assert!(body.position.y < 5.0 && body.position.y > 4.99);
    }

    #[test]
    fn static_body_ignores_gravity() {
        let mut world = World::new(EuclideanR2);
        let id = world.push_body(RigidBody::fixed(
            Vec2::new(0.0, 0.0),
            Collider::Sphere { radius: 1.0 },
            disk_inertia(0.0, 1.0),
            &EuclideanR2,
        ));
        world.push_field(Box::new(Gravity::new(Vec2::new(0.0, -9.8))));

        for _ in 0..10 {
            world.step(1.0 / 60.0);
        }

        let body = &world.bodies[id];
        assert_eq!(body.position, Vec2::ZERO);
        assert_eq!(body.velocity, Vec2::ZERO);
    }

    #[test]
    fn sphere_sphere_contact_detected() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let a = sphere_body(Vec2::ZERO, Vec2::ZERO, 1.0, 1.0);
        let b = sphere_body(Vec2::new(1.5, 0.0), Vec2::ZERO, 1.0, 1.0);
        let contact = np.test(&a, &b, &EuclideanR2).expect("should collide");
        assert!((contact.normal - Vec2::X).length() < 1e-5);
        assert!((contact.penetration - 0.5).abs() < 1e-5);
    }

    #[test]
    fn separating_spheres_produce_no_contact() {
        let mut np = Narrowphase::<EuclideanR2>::new();
        register_default_narrowphase(&mut np);

        let a = sphere_body(Vec2::ZERO, Vec2::ZERO, 0.4, 1.0);
        let b = sphere_body(Vec2::new(2.0, 0.0), Vec2::ZERO, 0.4, 1.0);
        assert!(np.test(&a, &b, &EuclideanR2).is_none());
    }

    #[test]
    fn head_on_collision_separates_spheres() {
        let mut world = World::new(EuclideanR2);
        register_default_narrowphase(&mut world.narrowphase);

        world.push_body(sphere_body(
            Vec2::new(-1.0, 0.0),
            Vec2::new(2.0, 0.0),
            0.5,
            1.0,
        ));
        world.push_body(sphere_body(
            Vec2::new(1.0, 0.0),
            Vec2::new(-2.0, 0.0),
            0.5,
            1.0,
        ));

        // Step a few times so they meet.
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }

        // Velocities should have their sign reversed in the x direction
        // for an elastic-ish bounce.
        assert!(
            world.bodies[0].velocity.x < 0.0,
            "body 0 should bounce back"
        );
        assert!(
            world.bodies[1].velocity.x > 0.0,
            "body 1 should bounce back"
        );
    }
}
