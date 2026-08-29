//! The four Shapes-view render paths (SDF upload, section caps, wireframe
//! overlay, point sprites) source their pose here, so they cannot disagree
//! about where a body is.

use glam::{Vec2, Vec3, Vec4};
use loam_math::{Bivector4, EuclideanR4, Rotor, Rotor4};
use loam_physics::euclidean_r4::{
    ball4_inertia, register_default_narrowphase, regular_polytope4_inertia, sphere_body_r4,
};
use loam_physics::{Collider, World};
use loam_render::raymarch::RaymarchShape;

use crate::catalog::ShapeEntry;
use crate::spins::SlotSpins;
use crate::state::body_position;

// Must match the app's fixed sim tick rate.
const PHYSICS_DT: f32 = 1.0 / 60.0;

// Density is unmodelled: a 5-cell hull masses the same as a 24-cell.
const BODY_MASS: f32 = 1.0;

// Fastest launch the R⁴ step resolves against a thin static wall: the recorded
// `RECORDED_R4` bound from `loam_physics::world`'s tunneling gate, 0.150 per
// step, spent at 90% because the record is a scanned floor at 0.0025
// resolution rather than a two-sided pin. Nothing in this scene gives a body
// speed since the throw gesture moved to the toybox, so this bounds the
// fixtures below and nothing else.
#[cfg(test)]
pub(crate) const MAX_RESOLVED_SPEED: f32 = 0.9 * 0.150 / PHYSICS_DT;

// Time constant of the velocity decay, in seconds. Zero-g and frictionless,
// a body given velocity would otherwise never re-enter the exact-zero fixpoint
// [`PlaygroundPhysics::at_rest`] tests for, so the step's skip would never
// re-engage and the body would leave the chamber for good. Travel is bounded
// by `speed · TAU`, which at [`MAX_RESOLVED_SPEED`] is 4.9 units: under the
// width of a full eight-slot row, so nothing leaves the frame.
const VELOCITY_DECAY_TAU: f32 = 0.6;

// Speeds under which a decaying body is snapped to exact rest. Exponential
// decay approaches zero without reaching it, and `at_rest` compares against
// exact zero; these are the thresholds that close the gap. Sized well under
// one pixel of motion per second at the demo's framing.
const REST_SPEED: f32 = 0.02;
const REST_ANGULAR_SPEED: f32 = 0.02;

// y-up NDC: the inverse of what `loam_camera::Camera::ray_from_ndc` consumes.
pub(crate) fn ndc_from_pixels(pixels: Vec2, viewport: (u32, u32)) -> Vec2 {
    let (width, height) = (viewport.0 as f32, viewport.1 as f32);
    Vec2::new(2.0 * pixels.x / width - 1.0, 1.0 - 2.0 * pixels.y / height)
}

// `Rotor4` multiplies left-first (`apply(a * b, v) == apply(b, apply(a, v))`),
// so the world-frame physics rotor is the right factor.
pub(crate) fn composed_rotor(spin: Rotor4, orientation: Rotor4) -> Rotor4 {
    spin * orientation
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct BodyPose {
    pub(crate) position: Vec4,
    pub(crate) rotor: Rotor4,
}

impl BodyPose {
    // The R³ translation the raster paths apply AFTER projection, so a
    // Perspective4D divide never scales the body's x-position.
    pub(crate) fn position_r3(&self) -> Vec3 {
        self.position.truncate()
    }

    // The `w` offset moves the frame off the origin, so the body's vertices
    // stop sharing a radius about it. No caller may read an endpoint's
    // `length()` as its circumradius.
    pub(crate) fn body_local(&self, canonical: Vec4, size: f32) -> Vec4 {
        size * self.rotor.apply(canonical) + Vec4::W * self.position.w
    }
}

#[derive(Copy, Clone, PartialEq)]
struct SyncedSlot {
    shape: RaymarchShape,
    spin: Rotor4,
}

pub(crate) struct PlaygroundPhysics {
    pub(crate) world: World<EuclideanR4>,
    synced: Vec<SyncedSlot>,
    synced_size: f32,
}

impl PlaygroundPhysics {
    pub(crate) fn new(slots: usize, radius: f32) -> Self {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let mut physics = Self {
            world,
            synced: Vec::new(),
            synced_size: radius,
        };
        physics.respawn(slots, radius);
        physics
    }

    pub(crate) fn respawn(&mut self, slots: usize, radius: f32) {
        // Despawn rather than replace the arena: a fresh arena restarts
        // generations at 0, so a handle held across a respawn would alias
        // whichever body lands in its slot next.
        while let Some(last) = self.world.bodies.len().checked_sub(1) {
            let id = self.world.bodies.id_at(last);
            self.world.bodies.despawn(id);
        }
        self.world.manifolds.clear();
        self.synced.clear();
        for slot in 0..slots {
            let position = Vec4::from_array(body_position(slot, slots));
            self.world
                .push_body(sphere_body_r4(position, Vec4::ZERO, radius, BODY_MASS));
        }
    }

    // The slot's UI spin is BAKED into the hull's vertex list, because
    // `world_vertices4_into` applies `body.orientation.rotation` alone. The
    // spin's rim velocity stays unmodelled: 16% of MAX_RESOLVED_SPEED by default.
    pub(crate) fn sync(&mut self, row: &[ShapeEntry], spins: &SlotSpins, size: f32) {
        if self.world.bodies.len() != row.len() {
            self.respawn(row.len(), size);
        }
        let unchanged = self.synced_size == size
            && self.synced.len() == row.len()
            && (self.synced.iter().enumerate()).all(|(slot, synced)| {
                synced.shape == row[slot].shape && synced.spin == spins.rotor(slot)
            });
        if unchanged {
            return;
        }
        self.synced_size = size;
        self.synced.clear();
        for (slot, entry) in row.iter().enumerate() {
            let spin = spins.rotor(slot);
            self.synced.push(SyncedSlot {
                shape: entry.shape,
                spin,
            });
            let body = &mut self.world.bodies[slot];
            let hull = entry
                .collider_polytope()
                .map(|p| (p, regular_polytope4_inertia(p, body.mass, size)));
            let Some((polytope, inertia)) = hull else {
                body.collider = Collider::sphere_at_origin(size);
                body.inertia = ball4_inertia(body.mass, size);
                continue;
            };
            let mut vertices =
                match std::mem::replace(&mut body.collider, Collider::sphere_at_origin(size)) {
                    Collider::ConvexPolytope4D { vertices } => vertices,
                    _ => Vec::new(),
                };
            vertices.clear();
            vertices.extend((polytope.topology().vertices.iter()).map(|v| size * spin.apply(*v)));
            body.collider = Collider::ConvexPolytope4D { vertices };
            body.inertia = inertia;
        }
    }

    pub(crate) fn at_rest(&self) -> bool {
        self.world
            .bodies
            .iter()
            .all(|b| b.velocity == Vec4::ZERO && b.angular_velocity.magnitude_squared() == 0.0)
    }

    // The skip is load-bearing, not an optimization: `surface scale` past
    // `BODY_X_SPACING / (2 · BODY_SIZE)` overlaps neighbouring bounding
    // spheres, and solving that would push a row nobody threw off its layout.
    pub(crate) fn step(&mut self, ticks: usize) {
        if self.at_rest() {
            return;
        }
        let decay = (-PHYSICS_DT / VELOCITY_DECAY_TAU).exp();
        for _ in 0..ticks {
            self.world.step(PHYSICS_DT);
            self.damp(decay);
        }
    }

    fn damp(&mut self, decay: f32) {
        for body in self.world.bodies.iter_mut() {
            body.velocity *= decay;
            if body.velocity.length_squared() < REST_SPEED * REST_SPEED {
                body.velocity = Vec4::ZERO;
            }
            body.angular_velocity = body.angular_velocity * decay;
            if body.angular_velocity.magnitude_squared() < REST_ANGULAR_SPEED * REST_ANGULAR_SPEED {
                body.angular_velocity = Bivector4::ZERO;
            }
        }
    }

    pub(crate) fn pose(&self, slot: usize, slots: usize, spin: Rotor4) -> BodyPose {
        assert_eq!(
            self.world.bodies.len(),
            slots,
            "physics world not synced to the rendered row"
        );
        let body = &self.world.bodies[slot];
        BodyPose {
            position: body.position,
            rotor: composed_rotor(spin, body.orientation.rotation),
        }
    }

    pub(crate) fn body_frame(
        &self,
        slot: usize,
        slots: usize,
        spin: Rotor4,
        canonical: &[Vec4],
        size: f32,
        out: &mut Vec<Vec4>,
    ) -> Vec3 {
        let pose = self.pose(slot, slots, spin);
        out.clear();
        out.extend(canonical.iter().map(|v| pose.body_local(*v, size)));
        pose.position_r3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};
    use loam_shape::polytope::Polytope4;

    const RADIUS: f32 = crate::consts::BODY_SIZE;

    // Moment per unit mass at unit circumradius: half the mean radius squared
    // that `regular_polytope4_inertia` documents for each solid.
    const HULL_SHAPES: [(Polytope4, f32); 6] = [
        (Polytope4::Pentatope, 1.0 / 12.0),
        (Polytope4::Tesseract, 1.0 / 6.0),
        (Polytope4::Cell16, 2.0 / 15.0),
        (Polytope4::Cell24, 13.0 / 60.0),
        (Polytope4::Cell600, 0.295_136_73),
        (Polytope4::Cell120, 0.307_740_58),
    ];

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    fn row_of(shape: RaymarchShape, slots: usize) -> Vec<ShapeEntry> {
        let entry = *crate::catalog::SHAPE_CATALOG
            .iter()
            .find(|e| e.shape == shape)
            .expect("every RaymarchShape has a catalog entry");
        vec![entry; slots]
    }

    fn synced_row(
        shape: RaymarchShape,
        slots: usize,
        size: f32,
        spin: Rotor4,
    ) -> (PlaygroundPhysics, Vec<ShapeEntry>, SlotSpins) {
        let row = row_of(shape, slots);
        let spins = SlotSpins::uniform(slots, spin);
        let mut physics = PlaygroundPhysics::new(slots, size);
        physics.sync(&row, &spins, size);
        (physics, row, spins)
    }

    fn sweep_spins() -> [Rotor4; 4] {
        [
            Rotor4::IDENTITY,
            rotor_at(Plane4::Xz, 1.1),
            rotor_at(Plane4::Xw, 0.6),
            rotor_at(Plane4::Xy, 0.7) * rotor_at(Plane4::Zw, 0.4),
        ]
    }

    #[test]
    fn at_rest_world_holds_the_static_layout() {
        let slots = 4;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        assert!(physics.at_rest());
        physics.step(600);
        for slot in 0..slots {
            let pose = physics.pose(slot, slots, Rotor4::IDENTITY);
            assert_eq!(pose.position.to_array(), body_position(slot, slots));
            assert_eq!(pose.rotor, Rotor4::IDENTITY);
        }
    }

    #[test]
    fn overlapping_layout_at_rest_is_never_pushed_apart() {
        let slots = 4;
        let mut physics = PlaygroundPhysics::new(slots, crate::consts::BODY_X_SPACING);
        physics.step(120);
        for slot in 0..slots {
            assert_eq!(
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
                body_position(slot, slots)
            );
        }
    }

    #[test]
    fn idle_orientation_leaves_the_spin_rotor_exact() {
        let physics = PlaygroundPhysics::new(3, RADIUS);
        for plane in Plane4::ALL {
            for &angle in &[0.3_f32, 1.7, -2.4] {
                let spin = rotor_at(plane, angle);
                for slot in 0..3 {
                    assert_eq!(
                        physics.pose(slot, 3, spin).rotor,
                        spin,
                        "{plane:?} at {angle} rad perturbed the spin rotor"
                    );
                }
            }
        }
    }

    #[test]
    fn body_local_carries_the_body_w_into_the_slice_frame() {
        let v = Vec4::new(0.5, -0.25, 0.125, 0.75);
        let flat = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.0),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(flat.body_local(v, RADIUS), RADIUS * v);
        assert_eq!(flat.position_r3(), Vec3::new(1.0, 0.9, 0.0));

        let lifted = BodyPose {
            position: Vec4::new(1.0, 0.9, 0.0, 0.25),
            rotor: Rotor4::IDENTITY,
        };
        assert_eq!(
            lifted.body_local(v, RADIUS),
            RADIUS * v + Vec4::new(0.0, 0.0, 0.0, 0.25)
        );
    }

    #[test]
    fn an_impulse_drives_its_own_slot_and_only_that_slot() {
        let slots = 3;
        let ticks = 30;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let impulse = Vec4::new(0.0, 0.0, 0.0, 2.0);
        physics.world.bodies[1].apply_impulse(impulse);
        assert!(!physics.at_rest());
        physics.step(ticks);

        let decay = (-PHYSICS_DT / VELOCITY_DECAY_TAU).exp();
        let travel = PHYSICS_DT * (1.0 - decay.powi(ticks as i32)) / (1.0 - decay);
        let expected = Vec4::from_array(body_position(1, slots)) + impulse * travel;
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position;
        assert!(
            (moved - expected).length() < 1e-5,
            "struck pose {moved} away from {expected}"
        );
        for slot in [0, 2] {
            assert_eq!(
                physics
                    .pose(slot, slots, Rotor4::IDENTITY)
                    .position
                    .to_array(),
                body_position(slot, slots),
                "untouched slot {slot} moved"
            );
        }
    }

    #[test]
    fn angular_impulse_composes_after_the_ui_spin() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let layout = Vec4::from_array(body_position(0, 1));
        physics.world.bodies[0].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            layout + Vec4::W * 0.5,
        );
        physics.step(10);

        let orientation = physics.world.bodies[0].orientation.rotation;
        assert_ne!(
            orientation,
            Rotor4::IDENTITY,
            "off-centre impulse produced no rotation"
        );

        let spin = rotor_at(Plane4::Xy, 0.9);
        let composed = physics.pose(0, 1, spin).rotor;
        let v = Vec4::new(0.3, -0.2, 0.9, 0.1);
        let staged = orientation.apply(spin.apply(v));
        assert!(
            (composed.apply(v) - staged).length() < 1e-5,
            "composition order is not spin-then-physics"
        );
    }

    #[test]
    fn sync_respawns_only_when_the_slot_count_changes() {
        let shape = RaymarchShape::Polytope(Polytope4::Tesseract);
        let (mut physics, row, spins) = synced_row(shape, 3, RADIUS, Rotor4::IDENTITY);
        physics.world.bodies[0].apply_impulse(Vec4::new(0.0, 0.0, 0.0, 1.0));
        physics.step(10);
        let in_flight = physics.pose(0, 3, Rotor4::IDENTITY).position;

        physics.sync(&row, &spins, RADIUS);
        assert_eq!(
            physics.pose(0, 3, Rotor4::IDENTITY).position,
            in_flight,
            "same-count sync cancelled an impulse"
        );

        physics.sync(
            &row_of(shape, 4),
            &SlotSpins::uniform(4, Rotor4::IDENTITY),
            RADIUS,
        );
        assert!(physics.at_rest(), "respawn left motion behind");
        for slot in 0..4 {
            assert_eq!(
                physics.pose(slot, 4, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, 4)
            );
        }
    }

    #[test]
    fn every_polychoron_collides_as_its_own_hull_and_the_smooth_solids_do_not() {
        for entry in crate::catalog::SHAPE_CATALOG {
            let (physics, ..) = synced_row(entry.shape, 1, RADIUS, Rotor4::IDENTITY);
            let expected_hull = HULL_SHAPES
                .iter()
                .any(|(p, _)| entry.shape == RaymarchShape::Polytope(*p));
            let got_hull = matches!(
                physics.world.bodies[0].collider,
                Collider::ConvexPolytope4D { .. }
            );
            assert_eq!(
                got_hull, expected_hull,
                "{} collided as {:?}",
                entry.label, physics.world.bodies[0].collider
            );
        }
    }

    #[test]
    fn hull_bodies_carry_the_exact_moment_and_everything_else_the_bounding_ball() {
        for size in [RADIUS, 0.4, 1.3] {
            let ball = ball4_inertia(BODY_MASS, size);
            for (polytope, moment_over_mr2) in HULL_SHAPES {
                let (physics, ..) =
                    synced_row(RaymarchShape::Polytope(polytope), 1, size, Rotor4::IDENTITY);
                let inertia = physics.world.bodies[0].inertia;
                let expected = BODY_MASS * size * size * moment_over_mr2;
                assert!(
                    (inertia - expected).abs() < 1e-6 * expected.max(1.0),
                    "{polytope:?} at size {size} carries {inertia}, not {expected}"
                );
                assert!(inertia < ball, "{polytope:?} is no lighter than the ball");
            }
            for shape in [RaymarchShape::ThreeSphere, RaymarchShape::CliffordTorus] {
                let (physics, ..) = synced_row(shape, 1, size, Rotor4::IDENTITY);
                assert_eq!(physics.world.bodies[0].inertia, ball, "{shape:?}");
            }
        }
    }

    #[test]
    fn the_hull_collider_is_the_shape_the_row_draws_under_its_ui_spin() {
        let orientation = rotor_at(Plane4::Yw, 0.8);
        for spin in sweep_spins() {
            for (polytope, _) in HULL_SHAPES {
                let (mut physics, ..) =
                    synced_row(RaymarchShape::Polytope(polytope), 1, RADIUS, spin);
                physics.world.bodies[0].orientation.rotation = orientation;
                let pose = physics.pose(0, 1, spin);
                let Collider::ConvexPolytope4D { vertices } = &physics.world.bodies[0].collider
                else {
                    panic!("{polytope:?} lost its hull");
                };
                let canonical = polytope.topology().vertices;
                assert_eq!(vertices.len(), canonical.len());
                for (local, v) in vertices.iter().zip(canonical) {
                    let collided = orientation.apply(*local);
                    let drawn = pose.body_local(*v, RADIUS);
                    assert!(
                        (collided - drawn).length() < 1e-5,
                        "{polytope:?} collides at {collided} and draws at {drawn}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_spinning_row_refills_its_hull_in_place_and_skips_an_unchanged_one() {
        let shape = RaymarchShape::Polytope(Polytope4::Cell24);
        let (mut physics, row, _) = synced_row(shape, 2, RADIUS, Rotor4::IDENTITY);
        let buffer_at = |physics: &PlaygroundPhysics, slot: usize| {
            let Collider::ConvexPolytope4D { vertices } = &physics.world.bodies[slot].collider
            else {
                panic!("slot {slot} lost its hull");
            };
            (vertices.as_ptr(), vertices.capacity())
        };
        let before = [buffer_at(&physics, 0), buffer_at(&physics, 1)];
        for step in 1..200 {
            let spins = SlotSpins::uniform(2, rotor_at(Plane4::Xw, step as f32 * 0.03));
            physics.sync(&row, &spins, RADIUS);
        }
        assert_eq!(
            [buffer_at(&physics, 0), buffer_at(&physics, 1)],
            before,
            "the spin reallocated a hull vertex buffer"
        );

        let spins = SlotSpins::uniform(2, rotor_at(Plane4::Xw, 199.0 * 0.03));
        physics.world.bodies[0].inertia = 0.0;
        physics.sync(&row, &spins, RADIUS);
        assert_eq!(
            physics.world.bodies[0].inertia, 0.0,
            "unchanged row resynced"
        );
        physics.sync(&row, &spins, RADIUS * 1.5);
        assert!(
            physics.world.bodies[0].inertia > 0.0,
            "a size edit was skipped"
        );
    }

    fn facing_pair(
        shape: RaymarchShape,
        spin: Rotor4,
        separation: f32,
        lateral: f32,
    ) -> PlaygroundPhysics {
        let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
        let origin = physics.world.bodies[0].position;
        physics.world.bodies[1].position = origin + Vec4::new(separation, lateral, 0.0, 0.0);
        physics
    }

    fn normal_impulse_lever(physics: &PlaygroundPhysics) -> Option<f32> {
        let (a, b) = (&physics.world.bodies[0], &physics.world.bodies[1]);
        let contact = physics.world.narrowphase.test(a, b, &EuclideanR4)?;
        Some(Bivector4::wedge(contact.point - a.position, contact.normal).magnitude())
    }

    #[test]
    fn only_the_hull_pair_puts_a_lever_on_its_normal_impulse() {
        const SEPARATION: f32 = RADIUS;
        for lateral in [0.0_f32, 0.1, 0.3, 0.5, 0.7] {
            let ball = facing_pair(
                RaymarchShape::ThreeSphere,
                Rotor4::IDENTITY,
                SEPARATION,
                lateral,
            );
            let lever = normal_impulse_lever(&ball).expect("overlapping balls");
            assert!(
                lever < 1e-6,
                "a ball pair offset by {lateral} carried a lever of {lever}"
            );
        }

        for (polytope, _) in HULL_SHAPES {
            let shape = RaymarchShape::Polytope(polytope);
            let mut best = 0.0_f32;
            for spin in sweep_spins() {
                for lateral in [0.0_f32, 0.1, 0.3, 0.5] {
                    let pair = facing_pair(shape, spin, SEPARATION, lateral);
                    best = best.max(normal_impulse_lever(&pair).unwrap_or(0.0));
                }
            }
            assert!(
                best > 1e-2,
                "{polytope:?} never produced a normal impulse with a lever \
                 (best {best}), so its contacts cannot spin a body either"
            );
        }
    }

    fn peak_struck_spin(shape: RaymarchShape, spin: Rotor4) -> f32 {
        let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        let mut peak = 0.0_f32;
        for _ in 0..120 {
            physics.step(1);
            peak = peak.max(physics.world.bodies[1].angular_velocity.magnitude());
        }
        peak
    }

    #[test]
    fn a_head_on_hull_collision_spins_the_struck_body_where_a_ball_pair_cannot() {
        let spin = rotor_at(Plane4::Xz, 1.1);
        for (polytope, _) in HULL_SHAPES {
            let peak = peak_struck_spin(RaymarchShape::Polytope(polytope), spin);
            assert!(
                peak > 0.0,
                "{polytope:?} left the body it struck at |ω| = {peak}"
            );
        }
        // Roundness costs lever arm. The 120-cell and 600-cell have 120 and
        // 600 cells over the same circumradius, so a head-on hit lands much
        // closer to the line of centres than it does on a 5-cell's corner, and
        // the spin it can impart falls accordingly. This is the property that
        // separates a hull from the ball it approaches, so it is worth pinning
        // in the direction of the inequality rather than at a value.
        let angular = peak_struck_spin(RaymarchShape::Polytope(Polytope4::Pentatope), spin);
        for round in [Polytope4::Cell120, Polytope4::Cell600] {
            let peak = peak_struck_spin(RaymarchShape::Polytope(round), spin);
            assert!(
                peak < angular,
                "{round:?} spun the struck body by {peak}, at or past the 5-cell's {angular}"
            );
        }
        assert_eq!(
            peak_struck_spin(RaymarchShape::ThreeSphere, spin),
            0.0,
            "the smooth solids keep the ball collider, which has no lever to spin on"
        );
    }

    #[test]
    fn a_hull_collision_pushes_the_struck_body_off_the_w_zero_slice() {
        for (polytope, _) in HULL_SHAPES {
            let shape = RaymarchShape::Polytope(polytope);
            let mut leaked = 0.0_f32;
            for spin in sweep_spins() {
                let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
                physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
                assert_eq!(
                    physics.world.bodies[0].velocity.w, 0.0,
                    "the impulse itself left the slice"
                );
                for _ in 0..120 {
                    physics.step(1);
                    leaked = leaked.max(physics.world.bodies[1].position.w.abs());
                }
            }
            assert!(
                leaked > 1e-3,
                "no spin in the sweep moved a struck {polytope:?} off the \
                 slice (best |w| = {leaked})"
            );
        }

        let (mut physics, ..) = synced_row(RaymarchShape::ThreeSphere, 2, RADIUS, Rotor4::IDENTITY);
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        physics.step(120);
        assert_eq!(physics.world.bodies[1].position.w, 0.0);
    }

    fn contact_width(shape: RaymarchShape, spin: Rotor4) -> f32 {
        const RUNG: f32 = 0.005;
        let mut width = 0.0_f32;
        let mut separation = RUNG;
        while separation < 4.0 * RADIUS {
            if normal_impulse_lever(&facing_pair(shape, spin, separation, 0.0)).is_some() {
                width = separation;
            }
            separation += RUNG;
        }
        width
    }

    #[test]
    fn overlapped_hulls_reach_the_at_rest_fixpoint_in_a_bounded_step_count() {
        const BUDGET: usize = 400;
        for (polytope, _) in HULL_SHAPES {
            let shape = RaymarchShape::Polytope(polytope);
            for spin in sweep_spins() {
                let width = contact_width(shape, spin);
                let mut physics = facing_pair(shape, spin, 0.75 * width, 0.0);
                physics.world.bodies[1].apply_impulse(flick(0.0625, RIGHT));
                assert!(!physics.at_rest(), "the fixture started in the fixpoint");

                let mut touched = false;
                let mut settled = None;
                for step in 0..BUDGET {
                    physics.step(1);
                    touched |= !physics.world.manifolds.is_empty();
                    if physics.at_rest() {
                        settled = Some(step);
                        break;
                    }
                }
                let settled = settled.unwrap_or_else(|| {
                    panic!("{polytope:?} at {spin:?} never came to rest in {BUDGET} steps")
                });
                assert!(touched, "{polytope:?} settled without ever contacting");

                let separation =
                    (physics.world.bodies[1].position - physics.world.bodies[0].position).x;
                assert!(
                    separation >= width,
                    "{polytope:?} came to rest {separation} apart, inside the {width} \
                     it presents: the overlap resolved by passing one hull through \
                     the other rather than by separating them"
                );

                let resting: Vec<Vec4> = physics.world.bodies.iter().map(|b| b.position).collect();
                physics.step(600);
                let after: Vec<Vec4> = physics.world.bodies.iter().map(|b| b.position).collect();
                assert_eq!(
                    resting, after,
                    "{polytope:?} kept drifting after reaching rest at step {settled}"
                );
            }
        }
    }

    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.4, 0.0, 0.0, 1.2),
            layout + Vec4::W * 0.5,
        );
        physics.step(24);
        physics
    }

    #[test]
    fn body_frame_reports_the_live_pose_not_the_authored_spin() {
        let slots = 3;
        let physics = tumbling(slots);
        let spin = rotor_at(Plane4::Xy, 0.7);
        let size = 0.4;
        let canonical = [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.6, -0.3, 0.2),
        ];

        let mut out = Vec::new();
        let origin = physics.body_frame(1, slots, spin, &canonical, size, &mut out);

        let body = &physics.world.bodies[1];
        let composed = composed_rotor(spin, body.orientation.rotation);
        assert_ne!(
            body.orientation.rotation,
            Rotor4::IDENTITY,
            "the impulse produced no rotation, so the pin below is vacuous"
        );
        assert_eq!(origin, body.position.truncate());
        assert_ne!(
            origin,
            Vec4::from_array(body_position(1, slots)).truncate(),
            "R³ translate still reads the static layout"
        );
        for (i, v) in canonical.iter().enumerate() {
            assert_eq!(
                out[i],
                size * composed.apply(*v) + Vec4::W * body.position.w
            );
            assert_ne!(
                out[i],
                size * spin.apply(*v),
                "frame vertex {i} still reads the authored spin alone"
            );
        }
    }

    #[test]
    fn body_frame_of_an_untouched_slot_is_the_authored_spin_exactly() {
        let slots = 3;
        let physics = tumbling(slots);
        let spin = rotor_at(Plane4::Zw, -1.1);
        let size = 0.4;
        let canonical = [Vec4::new(0.2, -0.7, 0.5, 0.1)];

        let mut out = Vec::new();
        let origin = physics.body_frame(2, slots, spin, &canonical, size, &mut out);
        assert_eq!(out[0], size * spin.apply(canonical[0]));
        assert_eq!(origin, Vec4::from_array(body_position(2, slots)).truncate());
    }

    #[test]
    fn body_frame_refills_the_scratch_buffer() {
        let physics = PlaygroundPhysics::new(2, RADIUS);
        let canonical = [Vec4::X, Vec4::Y, Vec4::Z];
        let mut out = vec![Vec4::ONE; 7];
        physics.body_frame(0, 2, Rotor4::IDENTITY, &canonical, 1.0, &mut out);
        assert_eq!(out.len(), canonical.len());
    }

    #[test]
    #[should_panic(expected = "physics world not synced to the rendered row")]
    fn pose_rejects_a_row_the_world_was_not_synced_to() {
        let physics = PlaygroundPhysics::new(3, RADIUS);
        physics.pose(0, 4, Rotor4::IDENTITY);
    }

    const RIGHT: Vec3 = Vec3::X;
    const UP: Vec3 = Vec3::Y;

    // Impulse carrying `fraction` of [`MAX_RESOLVED_SPEED`] along `direction`.
    // `m · speed · direction`, because `apply_impulse` divides by the same mass.
    fn flick(fraction: f32, direction: Vec3) -> Vec4 {
        (direction * (fraction * MAX_RESOLVED_SPEED * BODY_MASS)).extend(0.0)
    }

    #[test]
    fn an_impulse_advances_the_world_and_returns_it_to_the_at_rest_fixpoint() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let layout = Vec4::from_array(body_position(0, 1));
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        assert!(!physics.at_rest(), "an impulse left the world at rest");

        physics.step(6);
        let moved = physics.pose(0, 1, Rotor4::IDENTITY).position;
        assert!(
            (moved - layout).length() > 0.1,
            "six ticks of a full-power impulse moved the body only {}",
            (moved - layout).length()
        );

        // 0.6 s time constant from MAX_RESOLVED_SPEED down to REST_SPEED needs
        // ~3.7 s; ten seconds of ticks is comfortably past it.
        physics.step(600);
        assert!(physics.at_rest(), "the throw never decayed back to rest");
        let settled = physics.pose(0, 1, Rotor4::IDENTITY).position;
        physics.step(600);
        assert_eq!(
            physics.pose(0, 1, Rotor4::IDENTITY).position,
            settled,
            "a settled body kept drifting"
        );
    }

    #[test]
    fn a_body_that_has_come_to_rest_takes_a_second_impulse() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        physics.world.bodies[0].apply_impulse(flick(0.8, RIGHT));
        physics.step(600);
        assert!(physics.at_rest());
        let settled = physics.pose(0, 1, Rotor4::IDENTITY).position;

        physics.world.bodies[0].apply_impulse(flick(0.8, UP));
        assert!(
            !physics.at_rest(),
            "the second impulse did not wake the row"
        );
        physics.step(6);
        let after = physics.pose(0, 1, Rotor4::IDENTITY).position;
        assert!(
            after.y - settled.y > 0.1,
            "the second impulse moved the body {} in y",
            after.y - settled.y
        );
    }

    #[test]
    fn a_full_speed_impulse_transfers_momentum_to_the_neighbour_it_hits() {
        let slots = 2;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let target_layout = Vec4::from_array(body_position(1, slots));
        physics.world.bodies[0].apply_impulse(flick(1.0, RIGHT));
        physics.step(12);

        let thrower = physics.world.bodies[0].velocity;
        let target = physics.world.bodies[1].velocity;
        assert!(
            target.x > 1.0,
            "the neighbour was left at {target}: the impulse passed through it"
        );
        assert!(
            target.x > thrower.x,
            "the thrower kept more speed ({thrower}) than the body it hit ({target})"
        );
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position - target_layout;
        assert!(moved.x > 0.0, "the neighbour never left its layout");
    }

    #[test]
    fn ndc_from_pixels_centres_the_viewport_and_flips_y() {
        let viewport = (800, 600);
        assert_eq!(
            ndc_from_pixels(Vec2::new(400.0, 300.0), viewport),
            Vec2::ZERO
        );
        assert_eq!(
            ndc_from_pixels(Vec2::ZERO, viewport),
            Vec2::new(-1.0, 1.0),
            "window top-left is NDC (-1, +1)"
        );
        assert_eq!(
            ndc_from_pixels(Vec2::new(800.0, 600.0), viewport),
            Vec2::new(1.0, -1.0)
        );
    }
}
