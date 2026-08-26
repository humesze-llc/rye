//! The playground's rigid-body layer: a [`World<EuclideanR4>`] holding one
//! dynamic body per rendered row slot. The four Shapes-view render paths (SDF
//! upload, section caps, wireframe overlay, point sprites) source their pose
//! here, so they cannot disagree about where a body is.
//!
//! Filmstrip is outside the seam: its cells are a w/t sweep of a single
//! subject drawn at a fixed centre from the selected slot's UI rotation alone
//! (see [`crate::spins`]), with no body behind them.
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
use loam_physics::euclidean_r4::{
    ball4_inertia, register_default_narrowphase, regular_polytope4_inertia, sphere_body_r4,
};
use loam_physics::{Collider, World};
use loam_render::raymarch::RaymarchShape;

use crate::catalog::ShapeEntry;
use crate::spins::SlotSpins;
use crate::state::{body_position, CameraMode, Demo};

/// Physics tick, matching the app's fixed 60 Hz sim tick. The world advances
/// `FrameCtx::n_ticks` steps of this rather than one step of the frame's wall
/// time, so a trajectory is reproducible across frame rates.
const PHYSICS_DT: f32 = 1.0 / 60.0;

/// Uniform body mass. Density is unmodelled, so the four hull colliders do
/// not carry the volume ratios their solids have: a 5-cell massing the same
/// as a 24-cell is the chamber's choice, not an oversight. A per-shape mass
/// would have to pick a density, and nothing in the demo sets one.
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

/// Per-step displacement the R⁴ narrowphase still resolves BODY against BODY
/// at [`crate::consts::BODY_SIZE`]. The chamber holds no static geometry, so
/// this, not the wall figure above, is the band a spinning body's rim has to
/// stay inside. It is the 8-cell pair's number rather than the sphere pair's,
/// because a convex polychoron presents twice its own width along the launch
/// axis where a bounding ball presents twice the circumradius, and the tighter
/// of the two is what a ceiling is worth deriving against.
///
/// `the_body_tunneling_band_is_one_the_narrowphase_resolves` scans every
/// collider the row can install, rather than the two the number was set from,
/// and still finds a contact at this displacement per step at all sixteen
/// launch alignments it samples.
///
/// What it is NOT is a lower bound on the overlap window. Scanning centre
/// separation at 0.0025 puts the tightest same-collider window across ball and
/// hulls at the 8-cell's 1.395 at the unrotated pose, under this number, so a
/// step of exactly one band can in principle fall either side of an 8-cell
/// pair. What keeps [`MAX_ANGULAR_SPEED`] sound is that a body only ever
/// spends `TUNNELING_MARGIN` of the band, 1.2668, which clears that window and
/// clears the 1.335 the tightest mixed-collider pair in the same scan
/// presented. Tightening the band onto the measured window is the honest form
/// and costs angular ceiling, so it is a decision, not a repair.
const BODY_TUNNELING_BAND: f32 = 1.4075;

/// Angular speed ceiling for a thrown body, derived against the same budget as
/// [`MAX_THROW_SPEED`] and in the same units. The fastest material point on a
/// body of radius `R` covers `(|v| + |ω|·R) · PHYSICS_DT` per step, and the
/// linear ceiling already spends `MAX_THROW_SPEED · PHYSICS_DT` of that, so
/// the rotation gets what is left of `TUNNELING_MARGIN · BODY_TUNNELING_BAND`:
/// 97.0 rad/s at `R = BODY_SIZE`. Without it, a full-scale flick landing at
/// the bounding sphere's rim turns its whole impulse into spin at a rate the
/// body's inertia sets and the clamp on linear speed cannot see; at a quarter
/// of `ball4_inertia` that is 138.9 rad/s, 1.62 of rim travel per step from
/// the rotation alone, outside the band.
///
/// `R` is the authored [`crate::consts::BODY_SIZE`] and not the live
/// `effective_body_size`: `surface scale` multiplies the band and the rim
/// radius alike, so the rotation's share of the band is scale-invariant at
/// 80%, and the linear term is the only one whose share moves.
const MAX_ANGULAR_SPEED: f32 = (TUNNELING_MARGIN * BODY_TUNNELING_BAND
    - MAX_THROW_SPEED * PHYSICS_DT)
    / (crate::consts::BODY_SIZE * PHYSICS_DT);

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
/// The result stays in the `w = 0` slice the row lives on: a flick with a `w`
/// component would be the more 4D gesture, but it has no drag axis to come
/// from. Keeping the throw on the slice no longer keeps the BODIES on it,
/// because a hull contact moves one off (see [`BodyPose::body_local`]); it is
/// now only a statement about the gesture.
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

/// Rendered orientation for a body: that slot's UI rotation applied first,
/// then the body's physics orientation. [`Rotor4`] multiplies left-first
/// (`apply(a * b, v) == apply(b, apply(a, v))`), so the world-frame physics
/// rotor is the right factor.
///
/// An identity physics orientation returns `spin` component-for-component:
/// every product against the identity's zeros vanishes exactly, which is what
/// leaves the rotation UI untouched while nothing has been thrown.
pub(crate) fn composed_rotor(spin: Rotor4, orientation: Rotor4) -> Rotor4 {
    spin * orientation
}

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
    /// The `w` offset moves the frame off the origin, so the body's vertices
    /// stop sharing a radius about it. The wireframe's S³ arc path
    /// ([`crate::wireframe_geom::push_blended_edge`]) takes no precondition on
    /// that: it arcs in the body's own centred frame, about the point
    /// `body_local(Vec4::ZERO, size)` returns. No caller may read an endpoint's
    /// `length()` as its circumradius.
    pub(crate) fn body_local(&self, canonical: Vec4, size: f32) -> Vec4 {
        size * self.rotor.apply(canonical) + Vec4::W * self.position.w
    }
}

/// What a slot's collider was last built from. Together with the body size
/// these are the whole input of [`PlaygroundPhysics::sync`], so a row whose
/// spin is paused re-derives nothing.
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
        self.synced.clear();
        for slot in 0..slots {
            let position = Vec4::from_array(body_position(slot, slots));
            self.world
                .push_body(sphere_body_r4(position, Vec4::ZERO, radius, BODY_MASS));
        }
    }

    /// Reconcile with the rendered `row`. A slot-count change respawns the
    /// row, because the layout position is a function of the count and so
    /// every body moves. Otherwise only the collider and its inertia are
    /// refreshed, which is what makes this safe to run on any frame: a throw
    /// in flight survives.
    ///
    /// The polychora that fit under the narrowphase's vertex cap collide as
    /// their own hull, with the slot's UI spin BAKED into the vertex list
    /// rather than carried as a second rotor. `world_vertices4_into` applies
    /// `body.orientation.rotation` alone, so baking is what makes the
    /// collider the shape on screen: the narrowphase reconstructs
    /// [`composed_rotor`] to f32 rounding. What the solver still does not see
    /// is that the spin is MOVING the rim; that velocity is unmodelled, and
    /// at the default rate it is 16% of [`MAX_THROW_SPEED`].
    ///
    /// Everything else, the two polychora that overflow the cap and the four
    /// smooth solids, keeps the bounding ball. See
    /// [`crate::catalog::ShapeEntry::collider_polytope`] and
    /// [`loam_physics::euclidean_r4::regular_polytope4_inertia`].
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
            // One gate, not two: a shape earns a hull only if its vertex list
            // fits the narrowphase cap AND its uniform-solid moment is
            // derived, and the same two polychora fail both.
            let hull = entry
                .collider_polytope()
                .and_then(|p| regular_polytope4_inertia(p, body.mass, size).map(|i| (p, i)));
            let Some((polytope, inertia)) = hull else {
                body.collider = Collider::sphere_at_origin(size);
                body.inertia = ball4_inertia(body.mass, size);
                continue;
            };
            // Take the vertex buffer out and refill it rather than assigning a
            // fresh `Vec`: the spin rewrites it on every animating frame and
            // the count is fixed per shape, so `clear` keeps an allocation a
            // new one would repeat once per body per frame.
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

    /// Throw `slot`: apply `impulse` and clamp what the body is left carrying
    /// to the tunneling budget. Out-of-range slots are ignored, because a row
    /// edit can retire the slot a drag started on.
    pub(crate) fn throw(&mut self, slot: usize, impulse: Vec4) {
        if slot >= self.world.bodies.len() {
            return;
        }
        self.world.bodies[slot].apply_impulse(impulse);
        self.clamp_to_tunneling_budget(slot);
    }

    /// Clamp `slot`'s linear speed to [`MAX_THROW_SPEED`] and its angular
    /// speed to [`MAX_ANGULAR_SPEED`], the two halves of the per-step rim
    /// travel the narrowphase can still resolve. Enforced on the resulting
    /// velocities rather than on an impulse, so repeated flicks at a body
    /// already in motion cannot sum past either.
    ///
    /// Clamping `Bivector4::magnitude` is conservative for a double rotation:
    /// it is `sqrt(θ₁² + θ₂²)` while the fastest material point turns at
    /// `max(θ₁, θ₂)`, so an isoclinic spin is held to `1/sqrt(2)` of the rim
    /// speed a simple one gets. The alternative is the invariant
    /// decomposition per throw, which buys nothing a user could perceive.
    fn clamp_to_tunneling_budget(&mut self, slot: usize) {
        let body = &mut self.world.bodies[slot];
        let speed = body.velocity.length();
        if speed > MAX_THROW_SPEED {
            body.velocity *= MAX_THROW_SPEED / speed;
        }
        let angular_speed = body.angular_velocity.magnitude();
        if angular_speed > MAX_ANGULAR_SPEED {
            body.angular_velocity = body.angular_velocity * (MAX_ANGULAR_SPEED / angular_speed);
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
            let press = input.buttons.left.press_pos.map(|press_px| {
                let ray = self
                    .camera
                    .ray_from_ndc(ndc_from_pixels(press_px, viewport));
                let slots = self.render_row().len();
                (
                    press_px,
                    self.physics.pick(&ray, slots, self.effective_body_size()),
                )
            });
            // Selecting and aiming are one gesture: the press that picks a
            // body to flick is also the press that points the rotation
            // controls and the hypergimbal at it, so there is no second
            // click and no mode to be in. A press that entered no body
            // leaves the selection alone.
            self.spins.select_picked(press.and_then(|(_, slot)| slot));
            self.throw_drag = press.and_then(|(press_px, slot)| {
                slot.map(|slot| ThrowDrag {
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
    pub(crate) fn throw_enabled(&self, pointer_capture: bool) -> bool {
        !pointer_capture && self.camera_mode == CameraMode::Orbit
    }

    /// Throw `slot` as if a flick had dragged `drag_pixels`, and report the
    /// speed and per-step displacement it produced. The mouse path's own
    /// [`throw_impulse`] and [`PlaygroundPhysics::throw`], so a scripted throw
    /// is the same throw; the `throw` console command is the caller, and it is
    /// what makes a trajectory reproducible without a hand on the mouse.
    pub(crate) fn throw_slot(&mut self, slot: usize, drag_pixels: Vec2) -> anyhow::Result<String> {
        let slots = self.render_row().len();
        if slot >= slots {
            anyhow::bail!("slot {slot} is outside the rendered row of {slots}");
        }
        let view = self.camera.view();
        self.physics
            .throw(slot, throw_impulse(drag_pixels, view.right, view.up));
        let speed = self.physics.world.bodies[slot].velocity.length();
        Ok(format!(
            "throw: slot {slot} at {speed:.2} u/s ({:.4} per step, bound {MAX_PER_STEP_DISPLACEMENT})",
            speed * PHYSICS_DT
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::{Bivector, Plane4};
    use loam_physics::RigidBody;
    use loam_shape::polytope::Polytope4;

    const RADIUS: f32 = crate::consts::BODY_SIZE;

    /// The four polychora that collide as their own hull, with the closed-form
    /// `I/(m·r²)` each carries. Every test that sweeps "the shapes the swap
    /// reaches" iterates this, so adding a fifth cannot leave one behind.
    const HULL_SHAPES: [(Polytope4, f32); 4] = [
        (Polytope4::Pentatope, 1.0 / 12.0),
        (Polytope4::Tesseract, 1.0 / 6.0),
        (Polytope4::Cell16, 2.0 / 15.0),
        (Polytope4::Cell24, 13.0 / 60.0),
    ];

    fn rotor_at(plane: Plane4, angle: f32) -> Rotor4 {
        (plane.unit_bivector() * angle).exp().normalize()
    }

    /// A row of `slots` copies of one catalog entry, which is what the chamber
    /// holds after `shapes=x,x,...`.
    fn row_of(shape: RaymarchShape, slots: usize) -> Vec<ShapeEntry> {
        let entry = *crate::catalog::SHAPE_CATALOG
            .iter()
            .find(|e| e.shape == shape)
            .expect("every RaymarchShape has a catalog entry");
        vec![entry; slots]
    }

    /// A chamber holding `slots` copies of `shape`, synced through the real
    /// [`PlaygroundPhysics::sync`] at a row-wide UI spin, which is the only
    /// path that installs a collider.
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

    /// Spins the collider sweeps run under: the unrotated pose plus generic
    /// rotors in a coordinate plane, a `w`-mixing plane, and two planes at
    /// once. The row's UI spin is an arbitrary [`Rotor4`], so a property that
    /// only holds at identity is not a property of the chamber.
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
            "same-count sync cancelled a throw"
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
    fn exactly_the_polychora_under_the_vertex_cap_collide_as_their_own_hull() {
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
            if let Collider::ConvexPolytope4D { vertices } = &physics.world.bodies[0].collider {
                assert!(
                    vertices.len() <= loam_physics::euclidean_r4::MAX_POLYTOPE4_VERTICES,
                    "{} handed the narrowphase {} vertices",
                    entry.label,
                    vertices.len()
                );
            }
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
            for shape in [
                RaymarchShape::Polytope(Polytope4::Cell120),
                RaymarchShape::Polytope(Polytope4::Cell600),
                RaymarchShape::ThreeSphere,
                RaymarchShape::CliffordTorus,
            ] {
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
                    // What `world_vertices4_into` will do to the stored vertex.
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

        // A hand-set inertia survives a sync whose inputs did not move, and
        // only that: the exit is on the inputs, not on a dirty flag nothing
        // clears.
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

    /// Two synced bodies of `shape`, body 1 placed `separation` along `+x`
    /// from body 0 and `lateral` along `+y`. The geometry every contact pin
    /// below varies.
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

    /// The lever the contact's NORMAL impulse turns body 0 through:
    /// `apply_contact_impulse` applies `ω += I⁻¹·(ra ∧ direction·magnitude)`,
    /// so this wedge is what decides whether a hit can spin a body at all.
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

    /// Peak `|ω|` reached by the STRUCK body when slot 0 is flicked at full
    /// scale down the row into slot 1. Both bodies carry the same collider and
    /// the same spin, which is what the chamber holds.
    fn peak_struck_spin(shape: RaymarchShape, spin: Rotor4) -> f32 {
        let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
        physics.throw(
            0,
            throw_impulse(Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0), RIGHT, UP),
        );
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
                peak > 0.5,
                "{polytope:?} left the body it struck at |ω| = {peak}"
            );
        }
        for shape in [
            RaymarchShape::ThreeSphere,
            RaymarchShape::Polytope(Polytope4::Cell120),
        ] {
            assert_eq!(
                peak_struck_spin(shape, spin),
                0.0,
                "{shape:?} keeps the ball collider, which has no lever to spin on"
            );
        }
    }

    #[test]
    fn a_hull_collision_pushes_the_struck_body_off_the_w_zero_slice() {
        for (polytope, _) in HULL_SHAPES {
            let shape = RaymarchShape::Polytope(polytope);
            let mut leaked = 0.0_f32;
            for spin in sweep_spins() {
                let (mut physics, ..) = synced_row(shape, 2, RADIUS, spin);
                physics.throw(
                    0,
                    throw_impulse(Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0), RIGHT, UP),
                );
                assert_eq!(
                    physics.world.bodies[0].velocity.w, 0.0,
                    "the throw itself left the slice"
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

        // The control: a ball pair cannot leave the slice, which is why an
        // off-slice body frame was unreachable before the swap.
        let (mut physics, ..) = synced_row(RaymarchShape::ThreeSphere, 2, RADIUS, Rotor4::IDENTITY);
        physics.throw(
            0,
            throw_impulse(Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0), RIGHT, UP),
        );
        physics.step(120);
        assert_eq!(physics.world.bodies[1].position.w, 0.0);
    }

    /// Full width the pair presents along `+x` at this spin: the largest
    /// separation at which the narrowphase still reports a contact. Scanned
    /// rather than derived, because the support radius of a hull along a fixed
    /// axis is a function of the spin, and every shape in the sweep presents a
    /// different one.
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
                physics.throw(1, throw_impulse(Vec2::new(15.0, 0.0), RIGHT, UP));
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

    /// Camera basis for the drag tests: the demo's boot framing looks down
    /// −Z, so screen right is +X and screen up is +Y.
    const RIGHT: Vec3 = Vec3::X;
    const UP: Vec3 = Vec3::Y;

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

    /// Travel per step of the fastest material point on the bounding sphere's
    /// rim, which is the quantity the tunneling band bounds: the body's own
    /// speed plus what the spin adds at radius [`RADIUS`].
    fn rim_travel_per_step(body: &RigidBody<EuclideanR4>) -> f32 {
        (body.velocity.length() + body.angular_velocity.magnitude() * RADIUS) * PHYSICS_DT
    }

    /// Launch alignments per scan. Whether a sample lands where the pair
    /// overlaps is a function of where the fixed step lattice falls relative
    /// to the band, so one launch measures its own alignment and not the
    /// reach.
    const LAUNCH_PHASES: u32 = 16;

    /// Fire a body along +x at a static twin of itself at the origin, at
    /// `displacement` per step from a start `phase` further back, and report
    /// whether the narrowphase produced a contact at any point in the flight.
    /// The body-versus-body form of the R⁴ tunneling gate in
    /// `loam_physics::world`, whose fixture is a sphere against a thin wall.
    fn contact_is_detected(collider: Collider, displacement: f32, phase: f32) -> bool {
        // Clear of the widest overlap band either collider presents, in front
        // and behind, so the flight is decided by the sampling and not by
        // where it started or stopped.
        const CLEARANCE: f32 = 2.0;
        let inertia = ball4_inertia(BODY_MASS, RADIUS);
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_body(RigidBody::fixed(
            Vec4::ZERO,
            collider.clone(),
            inertia,
            &EuclideanR4,
        ));
        world.push_body(RigidBody::new(
            Vec4::new(-(CLEARANCE + phase), 0.0, 0.0, 0.0),
            Vec4::new(displacement / PHYSICS_DT, 0.0, 0.0, 0.0),
            collider,
            BODY_MASS,
            inertia,
            &EuclideanR4,
        ));
        let steps = ((2.0 * CLEARANCE + phase) / displacement).ceil() as usize + 1;
        (0..steps).any(|_| {
            world.step(PHYSICS_DT);
            !world.manifolds.is_empty()
        })
    }

    #[test]
    fn the_body_tunneling_band_is_one_the_narrowphase_resolves() {
        let mut cases = vec![(
            "ball, any spin".to_string(),
            Collider::sphere_at_origin(RADIUS),
        )];
        for (polytope, _) in HULL_SHAPES {
            for (i, spin) in sweep_spins().into_iter().enumerate() {
                let (physics, ..) = synced_row(RaymarchShape::Polytope(polytope), 1, RADIUS, spin);
                cases.push((
                    format!("{polytope:?} at sweep spin {i}"),
                    physics.world.bodies[0].collider.clone(),
                ));
            }
        }
        for (pair, collider) in cases {
            for phase in 0..LAUNCH_PHASES {
                let offset = BODY_TUNNELING_BAND * phase as f32 / LAUNCH_PHASES as f32;
                assert!(
                    contact_is_detected(collider.clone(), BODY_TUNNELING_BAND, offset),
                    "the {pair} pair missed each other at {BODY_TUNNELING_BAND} per step, \
                     launch phase {phase} of {LAUNCH_PHASES}: the band is wider than the \
                     reach MAX_ANGULAR_SPEED is derived against"
                );
            }
        }
    }

    #[test]
    fn the_throw_ceilings_together_spend_exactly_the_usable_band() {
        let spent = (MAX_THROW_SPEED + MAX_ANGULAR_SPEED * RADIUS) * PHYSICS_DT;
        let usable = TUNNELING_MARGIN * BODY_TUNNELING_BAND;
        assert!(
            (spent - usable).abs() < 1e-6,
            "the two ceilings spend {spent} of the band per step, not the \
             {usable} they are derived from"
        );
    }

    #[test]
    fn a_full_speed_off_centre_flick_stays_inside_the_body_tunneling_band() {
        let layout = Vec4::from_array(body_position(0, 1));
        let impulse = throw_impulse(Vec2::new(FULL_SCALE_DRAG_PIXELS, 0.0), RIGHT, UP);
        let ball = ball4_inertia(BODY_MASS, RADIUS);
        let usable = TUNNELING_MARGIN * BODY_TUNNELING_BAND;
        let mut worst_unclamped = 0.0_f32;
        for inertia in [ball, ball / 4.0, ball / 16.0, ball / 64.0] {
            // Levers off the impulse axis, so each carries a nonzero torque;
            // the diagonal one puts it in two planes at once.
            for offset in [Vec4::W, Vec4::Y, (Vec4::Y + Vec4::W).normalize()] {
                let mut physics = PlaygroundPhysics::new(1, RADIUS);
                physics.world.bodies[0].inertia = inertia;
                physics.world.bodies[0].apply_impulse_at_point(
                    &EuclideanR4,
                    impulse,
                    layout + offset * RADIUS,
                );
                worst_unclamped =
                    worst_unclamped.max(rim_travel_per_step(&physics.world.bodies[0]));
                physics.clamp_to_tunneling_budget(0);
                let travel = rim_travel_per_step(&physics.world.bodies[0]);
                assert!(
                    travel <= usable + 1e-5,
                    "a rim flick at inertia {inertia} on lever {offset} left {travel} \
                     of rim travel per step, past the {usable} the band allows"
                );
            }
        }
        assert!(
            worst_unclamped > BODY_TUNNELING_BAND,
            "no case in the sweep outran the band unclamped ({worst_unclamped}), \
             so the clamp above was never exercised"
        );
    }

    #[test]
    fn clamping_the_spin_preserves_its_plane() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let spin = Bivector4::new(1.0, -2.0, 0.5, 3.0, -1.5, 0.25);
        let over = spin * (10.0 * MAX_ANGULAR_SPEED / spin.magnitude());
        physics.world.bodies[0].angular_velocity = over;
        physics.throw(0, Vec4::ZERO);

        let clamped = physics.world.bodies[0].angular_velocity;
        assert!(
            (clamped.magnitude() - MAX_ANGULAR_SPEED).abs() < 1e-3,
            "clamped to {}, not the ceiling",
            clamped.magnitude()
        );
        let expected = spin * (MAX_ANGULAR_SPEED / spin.magnitude());
        assert!(
            (clamped + expected * -1.0).magnitude() < 1e-3,
            "the clamp turned the spin from {expected:?} to {clamped:?}"
        );
    }

    #[test]
    fn a_spin_inside_the_ceiling_is_untouched() {
        let mut physics = PlaygroundPhysics::new(1, RADIUS);
        let spin = Bivector4::new(0.0, 0.0, 0.5 * MAX_ANGULAR_SPEED, 0.0, 0.0, 0.0);
        physics.world.bodies[0].angular_velocity = spin;
        for _ in 0..8 {
            physics.throw(0, Vec4::ZERO);
        }
        assert_eq!(physics.world.bodies[0].angular_velocity, spin);
    }

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
