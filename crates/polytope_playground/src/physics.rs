//! The playground's rigid-body layer: a [`World<EuclideanR4>`] holding one
//! dynamic body per rendered row slot. The four Shapes-view render paths (SDF
//! upload, section caps, wireframe overlay, point sprites) source their pose
//! here, so they cannot disagree about where a body is.
//!
//! Filmstrip is outside the seam: its cells are a w/t sweep of a single
//! subject drawn at a fixed centre from the UI spin rotor alone, with no body
//! behind them.
//!
//! The chamber is zero-g and empty of static geometry: no [`ForceField`] is
//! registered, so a body only moves once a flick throws it (see
//! [`Demo::update_throw`]) and nothing but [`VELOCITY_DECAY_TAU`] slows it
//! down again.
//!
//! [`ForceField`]: loam_physics::ForceField

use glam::{Vec2, Vec3, Vec4};
use loam_app::Input;
use loam_camera::Ray;
use loam_math::{Bivector4, EuclideanR4, Rotor, Rotor4};
use loam_physics::euclidean_r4::{ball4_inertia, register_default_narrowphase, sphere_body_r4};
use loam_physics::{Collider, World};

use crate::state::{body_position, CameraMode, Demo};

/// Physics tick, matching the app's fixed 60 Hz sim tick. The world advances
/// `FrameCtx::n_ticks` steps of this rather than one step of the frame's wall
/// time, so a trajectory is reproducible across frame rates.
const PHYSICS_DT: f32 = 1.0 / 60.0;

/// Uniform body mass. Row members differ in vertex count, which is not a
/// quantity the bounding-sphere collider prices, so a per-shape mass would be
/// a number with nothing behind it.
const BODY_MASS: f32 = 1.0;

/// Largest per-step displacement the R⁴ step still resolves against a thin
/// static wall, as measured by the tunneling gate in `loam_physics::world`
/// (`RECORDED_R4`, scanned over 64 launch alignments). The bound is geometric
/// (`wall_half_thickness + body_radius` for the fixture that recorded it), so
/// no impulse magnitude or solver iteration count moves it: the only way to
/// stay inside it is to bound the speed a throw can leave a body at.
const MAX_PER_STEP_DISPLACEMENT: f32 = 0.150;

/// Fraction of [`MAX_PER_STEP_DISPLACEMENT`] a throw is allowed to use. The
/// recorded bound is a scanned FLOOR at 0.0025 resolution, not a two-sided
/// pin, and it was measured against a 0.1-radius projectile rather than this
/// row's bodies; a throw sized exactly at it would have no margin for either.
const TUNNELING_MARGIN: f32 = 0.9;

/// Speed ceiling for a thrown body: the usable share of the tunneling
/// displacement spread over one [`PHYSICS_DT`]. Enforced on the post-impulse
/// velocity rather than on the impulse, so repeated flicks at a body already
/// in flight cannot sum past it.
pub(crate) const MAX_THROW_SPEED: f32 = TUNNELING_MARGIN * MAX_PER_STEP_DISPLACEMENT / PHYSICS_DT;

/// Cursor travel, in physical pixels, that a flick needs to reach
/// [`MAX_THROW_SPEED`]. Roughly a quarter of a 1080p window's height, so a
/// full-power throw is a deliberate gesture and an idle click is nearly zero.
const FULL_SCALE_DRAG_PIXELS: f32 = 240.0;

/// Time constant of the velocity decay, in seconds. Zero-g and frictionless,
/// a thrown body would otherwise never re-enter the exact-zero fixpoint
/// [`PlaygroundPhysics::at_rest`] tests for, so the step's skip would never
/// re-engage and the body would leave the chamber for good. Travel from a
/// throw is bounded by `speed · TAU`, which at [`MAX_THROW_SPEED`] is 4.9
/// units: under the width of a full eight-slot row, so a flick stays in frame.
const VELOCITY_DECAY_TAU: f32 = 0.6;

/// Speeds under which a decaying body is snapped to exact rest. Exponential
/// decay approaches zero without reaching it, and `at_rest` compares against
/// exact zero; these are the thresholds that close the gap. Sized well under
/// one pixel of motion per second at the demo's framing.
const REST_SPEED: f32 = 0.02;
const REST_ANGULAR_SPEED: f32 = 0.02;

/// Impulse for a flick of `drag_pixels` under the camera's screen basis.
///
/// The mapping, in one line: **direction is the drag projected onto the camera
/// plane, speed is linear in drag length and saturates at
/// [`FULL_SCALE_DRAG_PIXELS`] pixels = [`MAX_THROW_SPEED`]**. Window
/// coordinates are y-down, so the vertical term negates `up`; the impulse is
/// `m · speed · direction` because [`loam_physics::RigidBody::apply_impulse`]
/// divides by the same mass.
///
/// The result stays in the `w = 0` slice the row lives on. A flick with a `w`
/// component would be the more 4D gesture, but it has no drag axis to come
/// from and it would move bodies off the slice, which the stereographic arc
/// path documented on [`BodyPose::body_local`] assumes it can rely on.
pub(crate) fn throw_impulse(drag_pixels: Vec2, right: Vec3, up: Vec3) -> Vec4 {
    // `right` and `up` are orthonormal, so this has length `drag_pixels`
    // and a zero drag is the only input `try_normalize` has to reject.
    let Some(direction) = (right * drag_pixels.x - up * drag_pixels.y).try_normalize() else {
        return Vec4::ZERO;
    };
    let speed = MAX_THROW_SPEED * (drag_pixels.length() / FULL_SCALE_DRAG_PIXELS).min(1.0);
    (direction * (speed * BODY_MASS)).extend(0.0)
}

/// Normalised device coordinates of a window-relative pixel position, y up.
/// The inverse of what [`loam_camera::Camera::ray_from_ndc`] consumes.
pub(crate) fn ndc_from_pixels(pixels: Vec2, viewport: (u32, u32)) -> Vec2 {
    let (width, height) = (viewport.0 as f32, viewport.1 as f32);
    Vec2::new(2.0 * pixels.x / width - 1.0, 1.0 - 2.0 * pixels.y / height)
}

/// Ray parameter at which `ray` first enters the sphere `(centre, radius)`, or
/// `None` when it misses. A ray starting inside returns the exit distance, so
/// a click from within a body still picks it.
///
/// Ericson, *Real-Time Collision Detection* (2005), §5.3.2; `ray.direction` is
/// unit, which is what drops the quadratic's leading coefficient.
fn ray_sphere_distance(ray: &Ray, centre: Vec3, radius: f32) -> Option<f32> {
    let offset = ray.origin - centre;
    let along = offset.dot(ray.direction);
    let outside = offset.length_squared() - radius * radius;
    // Pointing away from a sphere it is already outside of: no root can be
    // positive, and the discriminant would not say so.
    if outside > 0.0 && along > 0.0 {
        return None;
    }
    let discriminant = along * along - outside;
    if discriminant < 0.0 {
        return None;
    }
    let near = -along - discriminant.sqrt();
    Some(near.max(0.0))
}

/// Rendered orientation for a body: the UI spin applied first, then the
/// body's physics orientation. [`Rotor4`] multiplies left-first
/// (`apply(a * b, v) == apply(b, apply(a, v))`), so the world-frame physics
/// rotor is the right factor.
///
/// An identity physics orientation returns `spin` component-for-component:
/// every product against the identity's zeros vanishes exactly, which is what
/// leaves the rotation UI untouched while nothing has been thrown.
pub(crate) fn composed_rotor(spin: Rotor4, orientation: Rotor4) -> Rotor4 {
    spin * orientation
}

/// A rendered body's world pose.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BodyPose {
    pub(crate) position: Vec4,
    pub(crate) rotor: Rotor4,
}

impl BodyPose {
    /// The R³ translation the raster paths apply AFTER projection, so a
    /// Perspective4D divide never scales the body's x-position.
    pub(crate) fn position_r3(&self) -> Vec3 {
        self.position.truncate()
    }

    /// A canonical vertex in the body's own 4D frame: oriented, scaled by
    /// `size`, then offset by the body's `w`. The `w` offset is what keeps the
    /// world `w_slice` cutting the body where physics put it instead of always
    /// through its centre; it is exactly zero for a body on the layout.
    ///
    /// Precondition of the wireframe's S³ arc path (`blend > 0` in
    /// [`crate::wireframe_geom::push_blended_edge`]): `position.w == 0`. That
    /// path reads each endpoint's `length()` as its circumradius, which holds
    /// only while the frame is origin-centred; the `w` offset moves the body
    /// off the origin, so the endpoints stop sharing a radius and the interior
    /// bows onto a sphere the body is not on. Dormant until something throws a
    /// body off the slice, and the fix is to arc in the body's own centred
    /// frame rather than to drop the offset (the section cut needs it).
    pub(crate) fn body_local(&self, canonical: Vec4, size: f32) -> Vec4 {
        size * self.rotor.apply(canonical) + Vec4::W * self.position.w
    }
}

pub(crate) struct PlaygroundPhysics {
    pub(crate) world: World<EuclideanR4>,
}

impl PlaygroundPhysics {
    pub(crate) fn new(slots: usize, radius: f32) -> Self {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        let mut physics = Self { world };
        physics.respawn(slots, radius);
        physics
    }

    /// Drop every body back onto the static layout at rest. Manifolds go with
    /// them: their keys name the despawned handles, so every surviving entry
    /// is unreachable warm-start state the next step would walk for nothing.
    pub(crate) fn respawn(&mut self, slots: usize, radius: f32) {
        // Despawn rather than replace the arena: a fresh arena restarts
        // generations at 0, so a handle held across a respawn would alias
        // whichever body lands in its slot next, which is the exact aliasing
        // the generation counter exists to prevent.
        while let Some(last) = self.world.bodies.len().checked_sub(1) {
            let id = self.world.bodies.id_at(last);
            self.world.bodies.despawn(id);
        }
        self.world.manifolds.clear();
        for slot in 0..slots {
            let position = Vec4::from_array(body_position(slot, slots));
            self.world
                .push_body(sphere_body_r4(position, Vec4::ZERO, radius, BODY_MASS));
        }
    }

    /// Reconcile with a row of `slots` bodies. A slot-count change respawns
    /// the row, because the layout position is a function of the count and so
    /// every body moves. A same-count call only refreshes the collider, which
    /// is what makes this safe to run on any frame: a throw in flight survives.
    pub(crate) fn sync(&mut self, slots: usize, radius: f32) {
        if self.world.bodies.len() != slots {
            self.respawn(slots, radius);
            return;
        }
        for body in self.world.bodies.iter_mut() {
            body.collider = Collider::sphere_at_origin(radius);
            body.inertia = ball4_inertia(body.mass, radius);
        }
    }

    /// True while no body carries motion. Exact zero rather than a sleep
    /// threshold, which the decay in [`Self::damp`] is what makes reachable:
    /// a resting row is an exact fixpoint of the integrator, so this reads as
    /// "nothing is moving right now". It is not a record of whether anything
    /// was ever thrown; a throw that has decayed away reads at rest again.
    pub(crate) fn at_rest(&self) -> bool {
        self.world
            .bodies
            .iter()
            .all(|b| b.velocity == Vec4::ZERO && b.angular_velocity.magnitude_squared() == 0.0)
    }

    /// Advance `ticks` fixed steps, skipped entirely while at rest. The skip
    /// is load-bearing rather than an optimization: `surface scale` past
    /// `BODY_X_SPACING / (2 · BODY_SIZE)` overlaps neighbouring bounding
    /// spheres, and solving that overlap would push a row nobody threw off its
    /// layout.
    pub(crate) fn step(&mut self, ticks: usize) {
        if self.at_rest() {
            return;
        }
        // One `exp` per call rather than per tick; the exponent is a pair of
        // constants, so this is the same number every frame.
        let decay = (-PHYSICS_DT / VELOCITY_DECAY_TAU).exp();
        for _ in 0..ticks {
            self.world.step(PHYSICS_DT);
            self.damp(decay);
        }
    }

    /// Scale every body's velocity by `decay` and snap the ones under
    /// [`REST_SPEED`] / [`REST_ANGULAR_SPEED`] to exact zero, which is what
    /// returns a thrown row to the fixpoint [`Self::at_rest`] tests for.
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

    /// Slot whose bounding sphere `ray` enters first, or `None` when it enters
    /// none. `slots` and `radius` are the rendered row's, so a pick agrees
    /// with what the render paths drew rather than with the static layout.
    pub(crate) fn pick(&self, ray: &Ray, slots: usize, radius: f32) -> Option<usize> {
        let mut nearest: Option<(usize, f32)> = None;
        for slot in 0..slots {
            let centre = self.pose(slot, slots, Rotor4::IDENTITY).position_r3();
            let Some(distance) = ray_sphere_distance(ray, centre, radius) else {
                continue;
            };
            if nearest.is_none_or(|(_, best)| distance < best) {
                nearest = Some((slot, distance));
            }
        }
        nearest.map(|(slot, _)| slot)
    }

    /// Throw `slot`: apply `impulse` and clamp the resulting speed to
    /// [`MAX_THROW_SPEED`], which is the tunneling bound expressed as a
    /// velocity. Out-of-range slots are ignored, because a row edit can
    /// retire the slot a drag started on.
    pub(crate) fn throw(&mut self, slot: usize, impulse: Vec4) {
        if slot >= self.world.bodies.len() {
            return;
        }
        let body = &mut self.world.bodies[slot];
        body.apply_impulse(impulse);
        let speed = body.velocity.length();
        if speed > MAX_THROW_SPEED {
            body.velocity *= MAX_THROW_SPEED / speed;
        }
    }

    /// Pose of `slot` in a rendered row of `slots` bodies, under the UI spin
    /// rotor `spin`.
    ///
    /// `slots` is the caller's own row length and is checked, not trusted: the
    /// layout is frozen into each body at [`Self::respawn`] time, so a world
    /// that missed a row edit would draw every body at another slot's layout
    /// position and index past the end on the tail. [`Self::sync`] is the
    /// reconciliation point, and the body upload runs it at every row and size
    /// edit, before any render path reads a pose.
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

    /// Carry `canonical` into `slot`'s live body frame (writing `out`) and
    /// return the R³ translate the raster paths apply AFTER projection. `out`
    /// is cleared and refilled so a caller's scratch keeps its capacity.
    ///
    /// The single seam between the world and the raster passes: points,
    /// section caps, and the wireframe take all of their per-body geometry
    /// from here, which is what stops a pass from quietly falling back to the
    /// authored spin rotor over the static layout.
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

/// A flick in progress: the slot the press ray picked and the drag it has
/// travelled so far, both in window-relative physical pixels. Held across
/// frames because the release edge is the only frame that knows the gesture
/// is finished, and `FrameInput` drops its press anchor at that edge.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ThrowDrag {
    pub(crate) slot: usize,
    pub(crate) press_px: Vec2,
    pub(crate) cursor_px: Vec2,
}

impl ThrowDrag {
    pub(crate) fn drag_pixels(&self) -> Vec2 {
        self.cursor_px - self.press_px
    }

    /// Fraction of [`MAX_THROW_SPEED`] this drag has wound up, in `[0, 1]`.
    /// The aim overlay's only reading of the mapping.
    pub(crate) fn charge(&self) -> f32 {
        (self.drag_pixels().length() / FULL_SCALE_DRAG_PIXELS).min(1.0)
    }
}

impl Demo {
    /// Drive one frame of the pick / aim / release cycle and report whether a
    /// flick currently owns the left button.
    ///
    /// `viewport` is in physical pixels, matching `FrameInput::cursor_pos`.
    /// Freecam holds the cursor, so it has no position a ray can be built
    /// from; the caller passes `enabled = false` there and while egui has the
    /// pointer, and the button edges are still tracked so the next viewport
    /// press is a press and not the tail of a click that landed elsewhere.
    pub(crate) fn update_throw(
        &mut self,
        enabled: bool,
        input: &Input,
        viewport: (u32, u32),
    ) -> bool {
        let down = input.buttons.left.down;
        let pressed = down && !self.left_was_down;
        let released = !down && self.left_was_down;
        self.left_was_down = down;

        if !enabled {
            self.throw_drag = None;
            return false;
        }

        if pressed {
            // A press whose anchor is unknown (the cursor position was
            // invalidated before it arrived) has nothing to aim from.
            self.throw_drag = input.buttons.left.press_pos.and_then(|press_px| {
                let ray = self
                    .camera
                    .ray_from_ndc(ndc_from_pixels(press_px, viewport));
                let slots = self.render_row().len();
                self.physics
                    .pick(&ray, slots, self.effective_body_size())
                    .map(|slot| ThrowDrag {
                        slot,
                        press_px,
                        cursor_px: press_px,
                    })
            });
        } else if let (Some(drag), Some(cursor_px)) = (self.throw_drag.as_mut(), input.cursor_pos) {
            drag.cursor_px = cursor_px;
        }

        if released {
            if let Some(drag) = self.throw_drag.take() {
                let view = self.camera.view();
                let impulse = throw_impulse(drag.drag_pixels(), view.right, view.up);
                self.physics.throw(drag.slot, impulse);
            }
        }
        self.throw_drag.is_some()
    }

    /// Whether the flick gesture may run this frame. Orbit is the only camera
    /// mode with a free cursor, and egui owning the pointer means the press
    /// belongs to a widget.
    pub(crate) fn throw_enabled(&self, ui_has_focus: bool) -> bool {
        !ui_has_focus && self.camera_mode == CameraMode::Orbit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};

    const RADIUS: f32 = crate::consts::BODY_SIZE;

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    /// A world nothing has thrown holds the static layout exactly, however
    /// long it runs: the demo's boot state is a fixpoint, not a slow drift.
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

    /// Overlapping bounding spheres, which `surface scale` past
    /// `BODY_X_SPACING / (2 · BODY_SIZE)` produces, must not push a row nobody
    /// threw off its layout. Pins the at-rest skip in [`PlaygroundPhysics::step`]:
    /// without it the contact solver separates the row on its own.
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

    /// The rendered rotor equals the UI spin component-for-component while the
    /// physics orientation is identity. This is the rotation-UI half of "idle
    /// physics changes nothing": a tolerance here would let a real drift hide.
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

    /// A body at the layout `w = 0` maps canonical vertices exactly as the
    /// pre-physics `size * rotor.apply(v)` did; a body physics moved off the
    /// slice carries its `w` into the frame the cut is taken in.
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

    /// An impulse moves the thrown body's rendered pose along the decaying
    /// trajectory and leaves every other slot on the layout: poses follow the
    /// bodies, and only the bodies that were thrown.
    ///
    /// The closed form is the geometric sum of the per-step decay: `damp` runs
    /// after each `world.step`, so step `k` integrates `v₀·decayᵏ` and
    /// `x_n = x₀ + dt·v₀·(1 − decayⁿ)/(1 − decay)`.
    #[test]
    fn impulse_drives_the_thrown_slot_and_only_that_slot() {
        let slots = 3;
        let ticks = 30;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        // Thrown along +w: the one axis with no R³ analogue, and one that
        // cannot bring the body into contact with its neighbours.
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
            "thrown pose {moved} away from {expected}"
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

    /// An off-centre impulse spins the body, and the rendered rotor applies
    /// the UI spin FIRST and the physics orientation second. Reversing the
    /// factors would rotate the body's own animation into the world frame.
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

        // The impulse's lever `∧` force lands in the xw plane, so the spin
        // must share an index with it: absolutely orthogonal rotors commute
        // and could not tell the two composition orders apart.
        let spin = rotor_at(Plane4::Xy, 0.9);
        let composed = physics.pose(0, 1, spin).rotor;
        let v = Vec4::new(0.3, -0.2, 0.9, 0.1);
        let staged = orientation.apply(spin.apply(v));
        assert!(
            (composed.apply(v) - staged).length() < 1e-5,
            "composition order is not spin-then-physics"
        );
    }

    /// `sync` respawns on a slot-count change (the layout is a function of the
    /// count) and leaves poses alone otherwise, which is what lets the render
    /// path call it every frame without cancelling a throw.
    #[test]
    fn sync_respawns_only_when_the_slot_count_changes() {
        let mut physics = PlaygroundPhysics::new(3, RADIUS);
        physics.world.bodies[0].apply_impulse(Vec4::new(0.0, 0.0, 0.0, 1.0));
        physics.step(10);
        let in_flight = physics.pose(0, 3, Rotor4::IDENTITY).position;

        physics.sync(3, RADIUS);
        assert_eq!(
            physics.pose(0, 3, Rotor4::IDENTITY).position,
            in_flight,
            "same-count sync cancelled a throw"
        );

        physics.sync(4, RADIUS);
        assert!(physics.at_rest(), "respawn left motion behind");
        for slot in 0..4 {
            assert_eq!(
                physics.pose(slot, 4, Rotor4::IDENTITY).position.to_array(),
                body_position(slot, 4)
            );
        }
    }

    /// Throw slot 1 off-centre so it carries BOTH a linear and an angular
    /// velocity, then run it far enough that its pose cannot be confused with
    /// the layout.
    fn tumbling(slots: usize) -> PlaygroundPhysics {
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let layout = Vec4::from_array(body_position(1, slots));
        // The +w lever puts the torque in the xw plane. The push is mostly +w,
        // the axis on which the body cannot reach a neighbour, with enough +x
        // to move its R³ translate off the layout as well; 0.16 of travel over
        // these ticks against a 0.4 surface gap keeps the row a clean control
        // group.
        physics.world.bodies[1].apply_impulse_at_point(
            &EuclideanR4,
            Vec4::new(0.4, 0.0, 0.0, 1.2),
            layout + Vec4::W * 0.5,
        );
        physics.step(24);
        physics
    }

    /// [`PlaygroundPhysics::body_frame`] is the seam every raster pass reads
    /// its per-body geometry through, so it must report the LIVE pose: the
    /// composed rotor and the body's own `w` in the frame vertices, the body's
    /// live centre in the R³ translate. Reverting it to the authored
    /// `size * spin.apply(v)` over the static layout, which is what unwiring a
    /// raster pass from physics means, fails here.
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
            "throw produced no rotation, so the pin below is vacuous"
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

    /// An untouched slot's frame is byte-identical to the pre-physics
    /// `size * spin.apply(v)`: the seam adds no drift to a body nobody threw,
    /// which is what lets the pin above use exact equality.
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

    /// `out` is refilled, not appended to, so a caller passing a per-frame
    /// scratch buffer gets exactly one body's vertices.
    #[test]
    fn body_frame_refills_the_scratch_buffer() {
        let physics = PlaygroundPhysics::new(2, RADIUS);
        let canonical = [Vec4::X, Vec4::Y, Vec4::Z];
        let mut out = vec![Vec4::ONE; 7];
        physics.body_frame(0, 2, Rotor4::IDENTITY, &canonical, 1.0, &mut out);
        assert_eq!(out.len(), canonical.len());
    }

    /// A render path reading a world the row edit never reached is a bug, not
    /// a rendering: the slot count is checked at the seam rather than left to
    /// index out of bounds on the tail or, worse, silently draw a body at
    /// another slot's layout position.
    #[test]
    #[should_panic(expected = "physics world not synced to the rendered row")]
    fn pose_rejects_a_row_the_world_was_not_synced_to() {
        let physics = PlaygroundPhysics::new(3, RADIUS);
        physics.pose(0, 4, Rotor4::IDENTITY);
    }

    // ---- the flick ------------------------------------------------------

    /// Camera basis for the drag tests: the demo's boot framing looks down
    /// −Z, so screen right is +X and screen up is +Y.
    const RIGHT: Vec3 = Vec3::X;
    const UP: Vec3 = Vec3::Y;

    /// The contract every throw is sized against: whatever the drag, the
    /// resulting per-step displacement stays inside the tunneling bound the
    /// physics gate recorded for R⁴. Sweeps drags an order of magnitude past
    /// full scale and stacks flicks on a body already at the ceiling, which
    /// is the case a clamp on the IMPULSE rather than on the velocity misses.
    #[test]
    fn throw_speed_never_exceeds_the_measured_tunneling_bound() {
        let mut physics = PlaygroundPhysics::new(2, RADIUS);
        for drag in [
            Vec2::ZERO,
            Vec2::new(30.0, 0.0),
            Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0),
            Vec2::new(0.0, -4000.0),
            Vec2::new(9000.0, -9000.0),
        ] {
            for _ in 0..8 {
                physics.throw(0, throw_impulse(drag, RIGHT, UP));
                let displacement = physics.world.bodies[0].velocity.length() * PHYSICS_DT;
                assert!(
                    displacement <= MAX_PER_STEP_DISPLACEMENT,
                    "drag {drag} left {displacement} of travel per step, past the \
                     recorded {MAX_PER_STEP_DISPLACEMENT}"
                );
            }
        }
    }

    /// The half of "throw feels proportional to drag" a test can hold: speed
    /// is linear in drag length below full scale and flat above it. A mapping
    /// that squared the drag, or that ignored its length entirely, fails here.
    #[test]
    fn throw_speed_is_linear_in_drag_length_until_it_saturates() {
        for fraction in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let drag = Vec2::new(fraction * FULL_SCALE_DRAG_PIXELS, 0.0);
            let speed = throw_impulse(drag, RIGHT, UP).length() / BODY_MASS;
            assert!(
                (speed - fraction * MAX_THROW_SPEED).abs() < 1e-4,
                "drag at {fraction} of full scale gave {speed}, not \
                 {} of the ceiling",
                fraction
            );
        }
        for over in [1.5_f32, 4.0, 40.0] {
            let drag = Vec2::new(over * FULL_SCALE_DRAG_PIXELS, 0.0);
            let speed = throw_impulse(drag, RIGHT, UP).length() / BODY_MASS;
            assert!(
                (speed - MAX_THROW_SPEED).abs() < 1e-4,
                "drag at {over}x full scale gave {speed}, not the ceiling"
            );
        }
    }

    /// Direction is the drag carried into the camera plane, with the y-down
    /// window convention inverted, and it never leaves the `w = 0` slice.
    /// Catches a dropped negation (a flick that throws the wrong way
    /// vertically) and a `w` component leaking into the throw.
    #[test]
    fn throw_direction_is_the_drag_in_the_camera_plane_on_the_w_zero_slice() {
        let cases = [
            (Vec2::new(100.0, 0.0), RIGHT),
            (Vec2::new(-100.0, 0.0), -RIGHT),
            // Window y grows downward, so a downward drag throws down.
            (Vec2::new(0.0, 100.0), -UP),
            (Vec2::new(0.0, -100.0), UP),
        ];
        for (drag, expected) in cases {
            let impulse = throw_impulse(drag, RIGHT, UP);
            assert_eq!(impulse.w, 0.0, "drag {drag} threw off the slice");
            let direction = impulse.truncate().normalize();
            assert!(
                (direction - expected).length() < 1e-5,
                "drag {drag} threw toward {direction}, not {expected}"
            );
        }
        // A press with no travel is not a throw.
        assert_eq!(throw_impulse(Vec2::ZERO, RIGHT, UP), Vec4::ZERO);
    }

    /// A screen ray picks the first body it enters and nothing else: the slot
    /// nearest the eye when several line up, and `None` through empty space.
    /// A pick that returned the last hit instead would throw the body behind
    /// the one the user clicked.
    #[test]
    fn screen_ray_picks_the_nearest_body_it_enters_and_nothing_else() {
        let slots = 3;
        let physics = PlaygroundPhysics::new(slots, RADIUS);
        let centre = |slot: usize| Vec4::from_array(body_position(slot, slots)).truncate();

        for slot in 0..slots {
            let ray = Ray {
                origin: centre(slot) + Vec3::Z * 10.0,
                direction: -Vec3::Z,
            };
            assert_eq!(physics.pick(&ray, slots, RADIUS), Some(slot));
        }

        // Down the row: three bodies on one line, nearest wins.
        let along_row = Ray {
            origin: centre(0) - Vec3::X * 10.0,
            direction: Vec3::X,
        };
        assert_eq!(physics.pick(&along_row, slots, RADIUS), Some(0));
        let reversed = Ray {
            origin: centre(2) + Vec3::X * 10.0,
            direction: -Vec3::X,
        };
        assert_eq!(physics.pick(&reversed, slots, RADIUS), Some(2));

        // Clear of every bounding sphere, and pointing away from all of them.
        let sky = Ray {
            origin: centre(1) + Vec3::Y * 6.0,
            direction: -Vec3::Z,
        };
        assert_eq!(physics.pick(&sky, slots, RADIUS), None);
        let behind = Ray {
            origin: centre(1) + Vec3::Z * 10.0,
            direction: Vec3::Z,
        };
        assert_eq!(physics.pick(&behind, slots, RADIUS), None);
    }

    /// The whole point of the node, as one property: a thrown body leaves the
    /// at-rest fixpoint, actually advances through `World::step` (so the sim
    /// tick stops reading as the early return), and decays back into the
    /// fixpoint so the skip re-engages.
    #[test]
    fn a_thrown_body_advances_the_world_and_returns_to_the_at_rest_fixpoint() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let layout = Vec4::from_array(body_position(0, 1));
        physics.throw(0, throw_impulse(Vec2::new(400.0, 0.0), RIGHT, UP));
        assert!(!physics.at_rest(), "a throw left the world at rest");

        physics.step(6);
        let moved = physics.pose(0, 1, Rotor4::IDENTITY).position;
        assert!(
            (moved - layout).length() > 0.1,
            "six ticks of a full-power throw moved the body only {}",
            (moved - layout).length()
        );

        // 0.6 s time constant from MAX_THROW_SPEED down to REST_SPEED needs
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

    /// A body that has come to rest is throwable again. The step's at-rest
    /// skip returns before touching the world, so a throw applied to a
    /// sleeping row has to wake it or the second flick is inert.
    #[test]
    fn a_body_that_has_come_to_rest_is_throwable_again() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        physics.throw(0, throw_impulse(Vec2::new(200.0, 0.0), RIGHT, UP));
        physics.step(600);
        assert!(physics.at_rest());
        let settled = physics.pose(0, 1, Rotor4::IDENTITY).position;

        physics.throw(0, throw_impulse(Vec2::new(0.0, -200.0), RIGHT, UP));
        assert!(!physics.at_rest(), "the second flick did not wake the row");
        physics.step(6);
        let after = physics.pose(0, 1, Rotor4::IDENTITY).position;
        assert!(
            after.y - settled.y > 0.1,
            "the second flick moved the body {} in y",
            after.y - settled.y
        );
    }

    /// The impact: a body thrown at the ceiling speed down the row reaches
    /// its neighbour and drives it, rather than passing through it. The
    /// neighbour ends up faster than the thrower, which is what an
    /// equal-mass collision at restitution 0.2 produces.
    #[test]
    fn a_full_speed_throw_transfers_momentum_to_the_neighbour_it_hits() {
        let slots = 2;
        let mut physics = PlaygroundPhysics::new(slots, RADIUS);
        let target_layout = Vec4::from_array(body_position(1, slots));
        physics.throw(
            0,
            throw_impulse(Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0), RIGHT, UP),
        );
        // The 0.4 surface gap closes in three ticks at the ceiling speed;
        // twelve leaves the contact fully resolved and the target moving.
        physics.step(12);

        let thrower = physics.world.bodies[0].velocity;
        let target = physics.world.bodies[1].velocity;
        assert!(
            target.x > 1.0,
            "the neighbour was left at {target}: the throw passed through it"
        );
        assert!(
            target.x > thrower.x,
            "the thrower kept more speed ({thrower}) than the body it hit ({target})"
        );
        let moved = physics.pose(1, slots, Rotor4::IDENTITY).position - target_layout;
        assert!(moved.x > 0.0, "the neighbour never left its layout");
    }

    /// Pixel-to-NDC is the exact inverse the ray builder expects: centre maps
    /// to the origin, and the y flip puts window-top at NDC +1.
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
