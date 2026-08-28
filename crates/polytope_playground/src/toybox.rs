//! The floor's normal is pure `y`, so its NORMAL impulse carries no `w`. Its
//! friction does: the tangent space of a contact in R⁴ is three-dimensional and
//! `w` is in it, so the impulse that arrests a landing hull also slides it off
//! the slice. The scene answers that at the spawn, with a drop shallow enough
//! and a pose flat enough to keep the landing inside [`SETTLED_W_BAND`], rather
//! than by damping `w`: the `w` a body-body contact imparts is the point. For
//! the same reason [`ANGULAR_DAMPING`] has no linear counterpart: a linear
//! damper is exactly what would erase that drift.

use std::borrow::Cow;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_camera::Ray;
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector, Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4, WPlane};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::{BodyId, Gravity, World};
use loam_render::{
    DepthBuffer, DepthMode, LineRasterNode, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode,
    Viewport,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::{LineMesh, TriangleMesh};

use crate::consts::W_SCRUB_RATE;
use crate::environment::{register_ground_command, Environment};
use crate::physics::ndc_from_pixels;

const TICK_HZ: u32 = 60;

const TICK_DT: f32 = 1.0 / TICK_HZ as f32;

// A hull settling on a half-space needs 240 Hz, not 120: `polytope_halfspace_r4`
// reports one deepest vertex per step where the resting manifold holds four, so
// halving the rate halves the constraints the solver has by the time a body
// rocks over. Four sub-steps reach that rate without moving the host clock.
const BASE_SUBSTEPS: usize = 4;

// The sub-step count a tick may rise to when something is moving fast. The
// resolvable travel per step is fixed, so the speed a body may carry without
// tunneling is linear in this: `MAX_RELEASE_SPEED` below is 4.05 u/s per
// sub-step. Sixty-four buys 259 u/s, which is far past what a cursor produces,
// and five bodies make the extra steps cheap on the ticks that need them.
const MAX_SUBSTEPS: usize = 64;

// Fixed, never derived from wall time. A resting tick runs at
// `TICK_DT / BASE_SUBSTEPS`; a tick carrying a fast body divides further.
const MIN_SOLVER_DT: f32 = TICK_DT / MAX_SUBSTEPS as f32;

const GRAVITY: f32 = -9.8;

// Read by the physics half-space and by the background's analytic ground, so
// the drawn floor cannot drift from the one the bodies land on.
const FLOOR_Y: f32 = 0.0;

// Slice every cross-section is cut at when the scene boots. A body's distance
// from the live slice is what [`section_alpha`] fades on.
const W_SLICE: f32 = 0.0;

// Half-range the slice scrubs over. A toy is `BODY_SIZE` across in `w`, so this
// carries the slice about three body widths either side of the pile: far enough
// to lose every cross-section, near enough that the way back is one held key.
const W_SLICE_RANGE: f32 = 1.5;

const BODY_MASS: f32 = 1.0;

// Circumradius.
const BODY_SIZE: f32 = 0.45;

// Near-inelastic: a bouncing toy leaves the frame before it settles, and the
// interesting motion here is the `w` drift a contact imparts, not the rebound.
const RESTITUTION: f32 = 0.05;

const TOYS: [Polytope4; 5] = [
    Polytope4::Cell24,
    Polytope4::Tesseract,
    Polytope4::Pentatope,
    Polytope4::Cell16,
    Polytope4::Tesseract,
];

// Wider than two circumradii, so each toy reaches the floor on its own rather
// than landing on a neighbour: the scene settles before anyone touches it.
const SPAWN_SPACING: f32 = 1.4;

// Height of the lowest posed hull vertex above the floor at spawn, in world
// units. Measured against `w` drift and not chosen for the look of the fall:
// the friction cone at a contact in R⁴ has a `w` tangent, so the impulse that
// arrests a landing hull also slides it off the slice. The landing is not
// monotone in drop height, so this is a sampled value: over 24 seeds it settles
// every toy inside [`SETTLED_W_BAND`], where 0.05 and 0.35 both leave a
// pentatope a quarter of a unit off the slice with no cross-section left.
const SPAWN_CLEARANCE: f32 = 0.20;

// Worst `|w|` a toy reaches from its landing alone, over the 24 seeds
// `the_floor_alone_leaves_every_toy_inside_the_slice_band` scans, plus margin:
// the measured worst is 0.047. It is not zero, because the friction that stops
// the landing acts in a tangent space that includes `w`.
const SETTLED_W_BAND: f32 = 0.06;

// Per-plane angle, in radians, of the extra turn a toy is posed with about the
// axis it lands on. The turn fixes `y`, so it varies the pose without taking
// the chosen cell off the floor: a hull that lands on a corner rocks for
// seconds, and every rock slides it further off the slice.
const SPAWN_YAW: f32 = std::f32::consts::PI;

// |wedge| of two unit vectors is the sine of their angle, so this is the band
// in which the plane they span is not determined.
const ROTOR_PLANE_EPS: f32 = 1e-6;

pub(crate) const DEFAULT_SEED: u64 = 0x7061_7930;

// Per-step travel the R⁴ narrowphase still resolves against a thin static wall,
// as recorded by `loam_physics::world`'s tunneling gate. The floor is a
// half-space and cannot be tunneled, so this bounds body-against-body only.
const RESOLVABLE_STEP_TRAVEL: f32 = 0.150;

// The recorded figure is a scanned floor, not a two-sided pin, so a release
// sized exactly at it would have no margin.
const TRAVEL_MARGIN: f32 = 0.9;

// The fastest material point of a body of radius `R` covers
// `(|v| + |ω|·R)·SOLVER_DT` per step. The linear and angular ceilings split the
// usable budget evenly, so together they spend exactly
// `TRAVEL_MARGIN · RESOLVABLE_STEP_TRAVEL`.
const MAX_RELEASE_SPEED: f32 = 0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / MIN_SOLVER_DT;

const MAX_ANGULAR_SPEED: f32 =
    0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / (BODY_SIZE * MIN_SOLVER_DT);

// Travel one sub-step may cover before the narrowphase stops resolving it.
// `substeps_for` divides a tick until the fastest material point fits.
const STEP_TRAVEL_BUDGET: f32 = TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL;

// A release throws at the cursor's own speed, clamped only by
// [`MAX_RELEASE_SPEED`]. An earlier flat cap made every flick above it land on
// one speed, which flattened the whole top of the range: a fast mouse threw no
// harder than a medium one. The walls are what make an unclamped throw safe to
// offer, since a hard throw now bounces rather than leaving.

// Fastest the hold drive carries a body under the cursor. Separate from the
// release ceiling: a throw may leave at any speed the narrowphase resolves,
// but a carried body chasing a jumped cursor should not cross the container in
// a tick.
const MAX_CARRY_SPEED: f32 = 30.0;

// Most the grab may change a held body's velocity per tick. Finite so a
// contact impulse can win: the drive used to assign velocity outright, which
// discarded whatever the solver had just done to push the body out of a wall
// and drove it straight back in for as long as the cursor stayed there.
const MAX_GRAB_ACCEL: f32 = 400.0;

const _: () = assert!(MAX_CARRY_SPEED < MAX_RELEASE_SPEED);

// Per-second exponential decay of a free-flight tumble, applied per solver
// substep. A body knocked off the ground keeps no contact to brake its spin,
// so without this it tumbles until something catches it; at 1.2 the spin is
// down to a tenth in two seconds. LINEAR velocity is deliberately undamped:
// the `w` a body-body contact imparts is positional drift, and a linear damper
// is exactly what would suppress it.
const ANGULAR_DAMPING: f32 = 1.2;

// Rate, in 1/s, at which a held body closes the gap to the cursor's plane
// point. One tick moves it `GRAB_STIFFNESS/TICK_HZ` of the way, so the body
// arrives in about three ticks and cannot overshoot: the drive re-reads the
// error every tick rather than integrating a force.
const GRAB_STIFFNESS: f32 = 20.0;

// Samples of the grab target kept for the release estimate.
const GRAB_TRAIL: usize = 8;

// Seconds of cursor history the release velocity is measured over. Short
// enough that a drag which stalls before the release throws nothing, which is
// the whole difference between a flick and a slingshot.
const RELEASE_WINDOW: f32 = 0.08;

// World units of `w` per world unit of cursor rise while the modifier is held.
const W_PER_RISE: f32 = 1.0;

// A camera ray and the camera forward are never more than the half-diagonal
// field of view apart, so this rejects only a ray parallel to the grab plane,
// which the camera cannot produce.
const PLANE_MIN_COS: f32 = 1e-3;

// Cross-section half-width at which a cap is drawn opaque. The widest section a
// body presents is its circumradius; below a third of that the cap is already a
// sliver, and the sliver is what used to blink out.
const FADE_EXTENT: f32 = 0.35 * BODY_SIZE;

// A resting contact is a Baumgarte limit cycle, not a fixpoint: the bias pushes
// a sunk body out over a step and gravity puts it back, forever. In R⁴ that
// leaves the contact point with a tangential velocity whose tangent space
// includes `w`, so Coulomb friction brakes along `w` and a body parked on the
// floor creeps off the slice at about 0.011 per second. The latch below snaps a
// body that has gone nowhere to exact rest, which is what stops the creep
// without damping the `w` a real contact imparts.
//
// Travel, in world units, that a body may cover over [`REST_WINDOW`] ticks and
// still count as parked. The measured limit cycle covers 0.006 of it. A parked
// body is restored to its anchor pose rather than merely stopped: zeroing the
// velocity alone leaves gravity to sink it a little further every tick, and the
// normal impulse answering that eventually ejects the body on its own.
const REST_TRAVEL: f32 = 0.02;

const REST_WINDOW: u32 = 30;

// Speed and angular speed above which a body is awake whatever its travel, so
// an impulse into a parked body is not swallowed by the latch. Both sit well
// above the limit cycle's measured residual and far below any throw.
const REST_SPEED: f32 = 0.15;
const REST_ANGULAR_SPEED: f32 = 0.3;

// A toy that only ever landed carries no readout; anything past the band it
// settles in got there by being touched.
const W_LABEL_MIN: f32 = SETTLED_W_BAND;

// Half-extent of the container, in `x` and in `z`. The outermost toy's hull
// reaches `2 * SPAWN_SPACING + BODY_SIZE` = 3.25, and the boot camera frames
// the whole floor rectangle (`the_boot_camera_frames_the_whole_arena`). Both
// wall normals are pure `x` or pure `z`, so a wall's NORMAL impulse carries no
// `w` any more than the floor's does.
const ARENA_HALF_EXTENT: f32 = 3.6;

// Contact restitution is the mean of the two bodies', so a wall at 0.4 against
// a toy at 0.05 rebounds at 0.225: a hard throw comes back rather than
// stopping dead against an invisible plane.
const WALL_RESTITUTION: f32 = 0.4;

const ARENA_OUTLINE_COLOR: [f32; 4] = [0.55, 0.60, 0.68, 1.0];

const ARENA_OUTLINE_WIDTH_PX: f32 = 1.5;

// Determinant magnitude below which the ray lies in the triangle's plane. It is
// twice the triangle area times the cosine to the ray; a cap triangle at this
// body size is ~1e-2 across, so this rejects only genuinely edge-on hits.
const RAY_PARALLEL_EPS: f32 = 1e-8;

const TOY_COLOR_FALLBACK: [f32; 3] = [0.8, 0.8, 0.8];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum GrabAxis {
    /// Cursor rise moves the body up the screen and leaves `w` alone.
    Slice,
    /// Cursor rise moves the body along `w` instead, so a grab can push a body
    /// off the slice on purpose.
    Through,
}

pub(crate) struct ToyBody {
    body: BodyId,
    polytope: Polytope4,
    color: [f32; 3],
    /// Pose the rest latch last saw the body move to.
    rest_anchor: Vec4,
    rest_rotor: Rotor4,
    rest_ticks: u32,
}

// Ring of recent grab targets. Fixed size and no allocation: the estimate runs
// on every frame of a held grab.
#[derive(Copy, Clone, Debug)]
struct GrabTrail {
    points: [Vec4; GRAB_TRAIL],
    // Seconds from the previous sample to this one.
    spans: [f32; GRAB_TRAIL],
    head: usize,
    len: usize,
}

impl GrabTrail {
    fn seeded(point: Vec4) -> Self {
        let mut trail = Self {
            points: [point; GRAB_TRAIL],
            spans: [0.0; GRAB_TRAIL],
            head: 0,
            len: 1,
        };
        trail.points[0] = point;
        trail
    }

    fn push(&mut self, point: Vec4, dt: f32) {
        self.head = (self.head + 1) % GRAB_TRAIL;
        self.points[self.head] = point;
        self.spans[self.head] = dt;
        self.len = (self.len + 1).min(GRAB_TRAIL);
    }

    /// Mean velocity over the last [`RELEASE_WINDOW`] of samples, or over the
    /// whole trail when it is shorter than that.
    fn velocity(&self) -> Vec4 {
        let mut span = 0.0;
        let mut index = self.head;
        for _ in 0..self.len - 1 {
            let previous = (index + GRAB_TRAIL - 1) % GRAB_TRAIL;
            span += self.spans[index];
            index = previous;
            if span >= RELEASE_WINDOW {
                break;
            }
        }
        if span <= 0.0 {
            return Vec4::ZERO;
        }
        (self.points[self.head] - self.points[index]) / span
    }
}

struct Grab {
    toy: usize,
    /// Depth along the camera forward the body is held at.
    depth: f32,
    /// Offset from the centre of mass to the grabbed point, in the body frame,
    /// so a body that turns while held keeps the same handle.
    lever_local: Vec4,
    /// Where the drive is pulling the body, clamped into the container every
    /// frame so it can never run away past a wall.
    target: Vec4,
    /// The same motion without the clamp. Only its DERIVATIVE is read, by the
    /// release: a flick toward a wall pins `target` and would otherwise throw
    /// nothing, because the trail would see a target that had stopped moving.
    intent: Vec4,
    /// Last cursor point on the grab plane; the target advances by its delta.
    plane_point: Vec3,
    trail: GrabTrail,
}

pub(crate) struct Toybox {
    world: World<EuclideanR4>,
    toys: Vec<ToyBody>,
    grab: Option<Grab>,
    /// The `w` every cross-section is cut at. Scene state, not a constant: the
    /// held arrow keys and the panel slider both move it.
    slice: f32,
    tick: u64,
    local_vertices: Vec<Vec4>,
    section_scratch: SectionScratch,
    cap: TriangleMesh<3>,
}

impl Toybox {
    pub(crate) fn new(seed: u64) -> Self {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY, 0.0, 0.0))));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, FLOOR_Y));
        world.bodies[floor].restitution = RESTITUTION;
        for normal in [Vec4::X, -Vec4::X, Vec4::Z, -Vec4::Z] {
            // `dot(p, n) >= offset` is the solid side, so a wall facing inward
            // from `+ARENA_HALF_EXTENT` takes the negative offset.
            let wall = world.push_body(halfspace4_body_r4(normal, -ARENA_HALF_EXTENT));
            world.bodies[wall].restitution = WALL_RESTITUTION;
        }

        // Mixing in an odd constant keeps seed 0 usable: xorshift64* has a
        // fixed point at zero.
        let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut toys = Vec::with_capacity(TOYS.len());
        for (index, polytope) in TOYS.into_iter().enumerate() {
            let x = (index as f32 - (TOYS.len() as f32 - 1.0) * 0.5) * SPAWN_SPACING;
            let cell = (draw(&mut rng) % polytope.topology().cells.len() as u64) as usize;
            let yaw = Bivector4::new(
                0.0,
                SPAWN_YAW * signed_unit(draw(&mut rng)),
                0.0,
                0.0,
                0.0,
                SPAWN_YAW * signed_unit(draw(&mut rng)),
            )
            .exp()
            .normalize();
            let pose = face_down_pose(polytope, cell) * yaw;
            let vertices: Vec<Vec4> = polytope
                .topology()
                .vertices
                .iter()
                .map(|v| BODY_SIZE * *v)
                .collect();
            // Every toy is dropped from the same clearance whatever its pose,
            // so the impact speed does not depend on which face it fell for.
            let lowest = (vertices.iter()).fold(f32::INFINITY, |m, v| m.min(pose.apply(*v).y));
            let id = world.push_body(polytope_body_r4(
                Vec4::new(x, SPAWN_CLEARANCE - lowest, 0.0, W_SLICE),
                Vec4::ZERO,
                vertices,
                BODY_MASS,
            ));
            let body = &mut world.bodies[id];
            body.restitution = RESTITUTION;
            body.orientation.rotation = pose;
            // The exact uniform-solid moment rather than `polytope_body_r4`'s
            // bounding ball, which the lever arm of a grab reads directly.
            if let Some(inertia) = regular_polytope4_inertia(polytope, BODY_MASS, BODY_SIZE) {
                body.inertia = inertia;
            }
            toys.push(ToyBody {
                body: id,
                polytope,
                color: toy_color(polytope),
                rest_anchor: world.bodies[id].position,
                rest_rotor: world.bodies[id].orientation.rotation,
                rest_ticks: 0,
            });
        }

        Self {
            world,
            toys,
            grab: None,
            slice: W_SLICE,
            tick: 0,
            local_vertices: Vec::new(),
            section_scratch: SectionScratch::default(),
            cap: TriangleMesh::<3>::default(),
        }
    }

    pub(crate) fn slice(&self) -> f32 {
        self.slice
    }

    pub(crate) fn set_slice(&mut self, slice: f32) {
        self.slice = slice.clamp(-W_SLICE_RANGE, W_SLICE_RANGE);
    }

    /// One frame of the held Up/Down scrub. `dir` is `+1` up, `-1` down, and
    /// `dt` the sim time the frame covers, so the scrub replays with the tick
    /// stream rather than the wall clock.
    pub(crate) fn scrub_slice(&mut self, dir: f32, dt: f32) {
        self.set_slice(self.slice + dir * W_SCRUB_RATE * dt);
    }

    pub(crate) fn position(&self, toy: usize) -> Vec4 {
        self.world.bodies[self.toys[toy].body].position
    }

    /// Offset from the live slice, per toy, read from the live body.
    pub(crate) fn w_offsets(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        let slice = self.slice;
        (self.toys.iter().enumerate())
            .map(move |(index, toy)| (index, self.world.bodies[toy.body].position.w - slice))
    }

    /// Sub-steps this tick needs so the fastest material point still moves less
    /// than [`STEP_TRAVEL_BUDGET`] per step. Derived from body state alone, so a
    /// seeded replay divides its ticks identically.
    fn substeps_for_current_speed(&self) -> usize {
        let fastest = self
            .toys
            .iter()
            .map(|toy| {
                let body = &self.world.bodies[toy.body];
                body.velocity.length() + body.angular_velocity.magnitude() * BODY_SIZE
            })
            .fold(0.0f32, f32::max);
        let needed = (fastest * TICK_DT / STEP_TRAVEL_BUDGET).ceil();
        (needed as usize).clamp(BASE_SUBSTEPS, MAX_SUBSTEPS)
    }

    pub(crate) fn tick(&mut self) {
        let substeps = self.substeps_for_current_speed();
        let dt = TICK_DT / substeps as f32;
        let decay = (-ANGULAR_DAMPING * dt).exp();
        for _ in 0..substeps {
            self.world.step(dt);
            for toy in &self.toys {
                let body = &mut self.world.bodies[toy.body];
                body.angular_velocity = body.angular_velocity * decay;
            }
        }
        self.latch_parked_bodies();
        self.tick += 1;
    }

    fn latch_parked_bodies(&mut self) {
        for toy in &mut self.toys {
            let body = &mut self.world.bodies[toy.body];
            let travelled = (body.position - toy.rest_anchor).length();
            let moving = body.velocity.length() > REST_SPEED
                || body.angular_velocity.magnitude() > REST_ANGULAR_SPEED;
            if travelled > REST_TRAVEL || moving {
                toy.rest_anchor = body.position;
                toy.rest_rotor = body.orientation.rotation;
                toy.rest_ticks = 0;
                continue;
            }
            toy.rest_ticks += 1;
            if toy.rest_ticks >= REST_WINDOW {
                body.position = toy.rest_anchor;
                body.orientation.rotation = toy.rest_rotor;
                body.velocity = Vec4::ZERO;
                body.angular_velocity = Bivector4::ZERO;
            }
        }
    }

    fn wake(&mut self, toy: usize) {
        let body = &self.world.bodies[self.toys[toy].body];
        self.toys[toy].rest_anchor = body.position;
        self.toys[toy].rest_rotor = body.orientation.rotation;
        self.toys[toy].rest_ticks = 0;
    }

    /// Grabs the nearest body whose drawn cross-section the ray enters.
    /// A miss leaves any previous grab released.
    pub(crate) fn press(&mut self, ray: &Ray, forward: Vec3) -> bool {
        self.grab = None;
        let Some((toy, hit)) = self.pick(ray) else {
            return false;
        };
        self.wake(toy);
        let body = &mut self.world.bodies[self.toys[toy].body];
        // The hand holds the body still: the tumble it arrived with would
        // otherwise turn the lever arm under the grab.
        body.velocity = Vec4::ZERO;
        body.angular_velocity = Bivector4::ZERO;
        // The hull is grabbed where the ray met it in R³ and at the body's own
        // `w`: a body off the slice would otherwise be handled by a point
        // outside it, and the lever arm would swing it as the drive pulled.
        let grabbed = Vec4::new(hit.x, hit.y, hit.z, body.position.w);
        let lever_local = body
            .orientation
            .rotation
            .inverse()
            .apply(grabbed - body.position);
        let depth = (hit - ray.origin).dot(forward);
        self.grab = Some(Grab {
            toy,
            depth,
            lever_local,
            target: body.position,
            intent: body.position,
            plane_point: hit,
            trail: GrabTrail::seeded(body.position),
        });
        true
    }

    /// Advances the held body's target by the cursor's motion in the grab
    /// plane and drives the body at it. `dt` is the sim time the frame covers.
    pub(crate) fn hold(&mut self, ray: &Ray, forward: Vec3, axis: GrabAxis, dt: f32) {
        let Some(grab) = self.grab.as_mut() else {
            return;
        };
        if let Some(point) = plane_point(ray, forward, grab.depth) {
            let delta = point - grab.plane_point;
            grab.target = clamp_target_to_arena(advance_target(grab.target, delta, axis));
            grab.intent = advance_target(grab.intent, delta, axis);
            grab.plane_point = point;
        }
        grab.trail.push(grab.intent, dt);
        let body = &mut self.world.bodies[self.toys[grab.toy].body];
        let desired = clamp_length(
            (grab.target - body.position) * GRAB_STIFFNESS,
            MAX_CARRY_SPEED,
        );
        body.velocity += clamp_length(desired - body.velocity, MAX_GRAB_ACCEL * TICK_DT);
    }

    /// Throws the held body along its recent target velocity, through the point
    /// it was grabbed at, so an off-centre grab imparts spin.
    pub(crate) fn release(&mut self) {
        let Some(grab) = self.grab.take() else {
            return;
        };
        let velocity = clamp_length(grab.trail.velocity(), MAX_RELEASE_SPEED);
        self.wake(grab.toy);
        let body = &mut self.world.bodies[self.toys[grab.toy].body];
        // Cleared first so the drive velocity the hold left behind does not add
        // to the throw: the impulse below is the whole release.
        body.velocity = Vec4::ZERO;
        let point = body.position + body.orientation.rotation.apply(grab.lever_local);
        body.apply_impulse_at_point(&EuclideanR4, velocity * body.mass, point);
        let angular = body.angular_velocity.magnitude();
        if angular > MAX_ANGULAR_SPEED {
            body.angular_velocity = body.angular_velocity * (MAX_ANGULAR_SPEED / angular);
        }
    }

    /// Nearest toy whose drawn cross-section the ray enters, with the world
    /// point it enters at. The cross-section and not the bounding ball: the
    /// ball of a tesseract reaches well past the shape on screen.
    ///
    /// A body the slice cannot see draws nothing to aim at, and pushing one
    /// off the slice is a gesture the scene offers. Those bodies alone fall
    /// back to their bounding ball, and only when no cross-section was hit, so
    /// a visible cap always wins.
    pub(crate) fn pick(&mut self, ray: &Ray) -> Option<(usize, Vec3)> {
        let mut nearest: Option<(usize, f32)> = None;
        let mut invisible = [false; TOYS.len()];
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        for (toy, hidden) in invisible.iter_mut().enumerate() {
            let extent = append_cap(
                &self.world,
                &self.toys[toy],
                self.slice,
                &mut local,
                &mut self.section_scratch,
                &mut cap,
            );
            *hidden = extent <= 0.0;
            for triangle in &cap.indices {
                let [a, b, c] = triangle.map(|i| Vec3::from_array(cap.vertices[i as usize]));
                let Some(distance) = ray_triangle_distance(ray, a, b, c) else {
                    continue;
                };
                if nearest.is_none_or(|(_, best)| distance < best) {
                    nearest = Some((toy, distance));
                }
            }
        }
        self.cap = cap;
        self.local_vertices = local;
        if nearest.is_none() {
            for (toy, hidden) in invisible.iter().enumerate() {
                if !hidden {
                    continue;
                }
                let centre = self.world.bodies[self.toys[toy].body].position.truncate();
                let Some(distance) = ray_ball_distance(ray, centre, BODY_SIZE) else {
                    continue;
                };
                if nearest.is_none_or(|(_, best)| distance < best) {
                    nearest = Some((toy, distance));
                }
            }
        }
        nearest.map(|(toy, distance)| (toy, ray.origin + ray.direction * distance))
    }

    /// Splits the frame's cross-section caps by whether they are opaque, so the
    /// faded ones can be drawn depth-read-only after the solid ones. A faded
    /// cap that wrote depth would punch a hole rather than fade.
    pub(crate) fn build_frame_meshes(
        &mut self,
        opaque: &mut TriangleMesh<3>,
        faded: &mut TriangleMesh<3>,
    ) {
        clear_mesh(opaque);
        clear_mesh(faded);
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        for toy in &self.toys {
            let extent = append_cap(
                &self.world,
                toy,
                self.slice,
                &mut local,
                &mut self.section_scratch,
                &mut cap,
            );
            let alpha = section_alpha(extent);
            for color in &mut cap.colors {
                color[3] = alpha;
            }
            append_mesh(if alpha >= 1.0 { opaque } else { faded }, &cap);
        }
        self.cap = cap;
        self.local_vertices = local;
    }

    fn build_overlay_mesh(&self, overlay: &PhysicsOverlay, mesh: &mut LineMesh<3>) {
        build_physics_overlay_mesh(&self.world, BODY_SIZE, overlay, mesh);
    }
}

#[cfg(test)]
impl Toybox {
    fn tick_index(&self) -> u64 {
        self.tick
    }

    fn toys(&self) -> &[ToyBody] {
        &self.toys
    }

    fn velocity(&self, toy: usize) -> Vec4 {
        self.world.bodies[self.toys[toy].body].velocity
    }

    fn angular_velocity(&self, toy: usize) -> Bivector4 {
        self.world.bodies[self.toys[toy].body].angular_velocity
    }

    fn cap_stats(&mut self, toy: usize) -> (f32, usize) {
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        let extent = append_cap(
            &self.world,
            &self.toys[toy],
            self.slice,
            &mut local,
            &mut self.section_scratch,
            &mut cap,
        );
        let stats = (section_alpha(extent), cap.vertices.len());
        self.cap = cap;
        self.local_vertices = local;
        stats
    }

    fn run(&mut self, ticks: usize) {
        for _ in 0..ticks {
            self.tick();
        }
    }

    fn fastest_step_travel(&self) -> f32 {
        let dt = TICK_DT / self.substeps_for_current_speed() as f32;
        (self.world.bodies.iter())
            .map(|b| (b.velocity.length() + b.angular_velocity.magnitude() * BODY_SIZE) * dt)
            .fold(0.0, f32::max)
    }

    fn deepest_point(&self) -> f32 {
        let mut deepest = f32::INFINITY;
        for body in self.world.bodies.iter() {
            let loam_physics::Collider::ConvexPolytope4D { vertices } = &body.collider else {
                continue;
            };
            for v in vertices {
                deepest = deepest.min(body.orientation.rotation.apply(*v).y + body.position.y);
            }
        }
        deepest
    }
}

// xorshift64* (Vigna 2016, *An experimental exploration of Marsaglia's xorshift
// generators, scrambled*, §4).
fn draw(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

// The top 24 bits convert to `f32` exactly and the scale is a power of two, so
// the value is identical on any host that rounds IEEE-754.
fn unit(value: u64) -> f32 {
    ((value >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

fn signed_unit(value: u64) -> f32 {
    2.0 * unit(value) - 1.0
}

/// Rotor putting the outward normal of `cell` on `-y`, so the toy is dropped
/// flat on that cell rather than on a corner. A regular polychoron is centred
/// at its centroid, so the cell's own centroid is along its outward normal.
fn face_down_pose(polytope: Polytope4, cell: usize) -> Rotor4 {
    let topology = polytope.topology();
    let indices = topology.cells[cell];
    let mut centroid = Vec4::ZERO;
    for index in indices {
        centroid += topology.vertices[*index as usize];
    }
    match (centroid / indices.len() as f32).try_normalize() {
        Some(normal) => rotor_onto(normal, -Vec4::Y),
        None => Rotor4::IDENTITY,
    }
}

/// Rotor turning the unit vector `from` onto the unit vector `to` in the plane
/// they span. The chord form `2·asin(|a−b|/2)` and not `acos(a·b)`, which
/// loses half its digits as the two align.
fn rotor_onto(from: Vec4, to: Vec4) -> Rotor4 {
    let plane = Bivector4::wedge(from, to);
    let magnitude = plane.magnitude();
    if magnitude < ROTOR_PLANE_EPS {
        if from.dot(to) > 0.0 {
            return Rotor4::IDENTITY;
        }
        // Antiparallel: a half turn in any plane through `from`.
        let axis = orthogonal_to(from);
        return (Bivector4::wedge(from, axis) * std::f32::consts::PI)
            .exp()
            .normalize();
    }
    let angle = 2.0 * (0.5 * (to - from).length()).clamp(0.0, 1.0).asin();
    (plane * (angle / magnitude)).exp().normalize()
}

/// Unit vector orthogonal to `unit`, taken off the coordinate axis it leans on
/// least so the rejection cannot underflow.
fn orthogonal_to(unit: Vec4) -> Vec4 {
    let axes = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    let least = (0..4).fold(0, |best, i| {
        if unit[i].abs() < unit[best].abs() {
            i
        } else {
            best
        }
    });
    (axes[least] - unit * unit[least]).normalize()
}

fn toy_color(polytope: Polytope4) -> [f32; 3] {
    (crate::catalog::SHAPE_CATALOG.iter())
        .find(|entry| entry.shape.polytope4() == Some(polytope))
        .map(|entry| entry.body_color)
        .unwrap_or(TOY_COLOR_FALLBACK)
}

// Keeps a grab target inside the container, inset by the body's circumradius,
// so a held body is never ASKED to occupy a wall or the floor. `w` is left
// free: it is the axis the modifier exists to drive, and no static geometry
// bounds it.
fn clamp_target_to_arena(target: Vec4) -> Vec4 {
    let reach = ARENA_HALF_EXTENT - BODY_SIZE;
    Vec4::new(
        target.x.clamp(-reach, reach),
        target.y.max(FLOOR_Y + BODY_SIZE),
        target.z.clamp(-reach, reach),
        target.w,
    )
}

fn clamp_length(v: Vec4, ceiling: f32) -> Vec4 {
    let length = v.length();
    if length > ceiling {
        v * (ceiling / length)
    } else {
        v
    }
}

/// Point where the ray meets the plane at `depth` along `forward`. `None` when
/// the ray runs parallel to that plane.
pub(crate) fn plane_point(ray: &Ray, forward: Vec3, depth: f32) -> Option<Vec3> {
    let along = ray.direction.dot(forward);
    if along.abs() < PLANE_MIN_COS {
        return None;
    }
    Some(ray.origin + ray.direction * (depth / along))
}

/// Routes one frame of cursor motion in the grab plane onto the target. The
/// modifier trades the plane's rise for `w`, which is why it can be pressed and
/// released mid-drag without the body jumping: the target integrates deltas.
pub(crate) fn advance_target(target: Vec4, delta: Vec3, axis: GrabAxis) -> Vec4 {
    match axis {
        GrabAxis::Slice => target + Vec4::new(delta.x, delta.y, delta.z, 0.0),
        GrabAxis::Through => target + Vec4::new(delta.x, 0.0, delta.z, delta.y * W_PER_RISE),
    }
}

/// Opacity of a cross-section of half-width `extent`. A body drifting off the
/// slice shrinks to nothing; without this its last sliver is still fully opaque
/// on the frame before it disappears.
pub(crate) fn section_alpha(extent: f32) -> f32 {
    (extent / FADE_EXTENT).clamp(0.0, 1.0)
}

/// Möller and Trumbore, *Fast, Minimum Storage Ray/Triangle Intersection*,
/// JGT 2(1), 1997. Two-sided: a cap fan faces either way about its centroid.
fn ray_triangle_distance(ray: &Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = ray.direction.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < RAY_PARALLEL_EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = ray.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(qvec) * inv_det;
    (distance > 0.0).then_some(distance)
}

/// Nearest positive root of `|o + t·d − centre|² = radius²` for a unit `d`,
/// which drops the quadratic's leading coefficient. `−b − √D` is the entry
/// root and cancels only when `b > 0`, where the ray points away from the ball
/// and the entry root is behind the eye anyway; the returned value is then the
/// exit root, computed by the well-conditioned sum.
fn ray_ball_distance(ray: &Ray, centre: Vec3, radius: f32) -> Option<f32> {
    let to_centre = ray.origin - centre;
    let b = to_centre.dot(ray.direction);
    let discriminant = b * b - (to_centre.length_squared() - radius * radius);
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let exit = -b + root;
    if exit <= 0.0 {
        return None;
    }
    let entry = -b - root;
    Some(if entry > 0.0 { entry } else { exit })
}

/// The four floor edges where the container's walls meet `y = FLOOR_Y`. The
/// walls are half-spaces and so unbounded upward; this is the footprint, which
/// is what a throw is aimed inside of.
fn append_arena_outline(mesh: &mut LineMesh<3>) {
    let e = ARENA_HALF_EXTENT;
    let corners = [
        Vec3::new(-e, FLOOR_Y, -e),
        Vec3::new(e, FLOOR_Y, -e),
        Vec3::new(e, FLOOR_Y, e),
        Vec3::new(-e, FLOOR_Y, e),
    ];
    for (index, from) in corners.iter().enumerate() {
        push_overlay_segment(
            mesh,
            *from,
            corners[(index + 1) % corners.len()],
            ARENA_OUTLINE_COLOR,
            ARENA_OUTLINE_COLOR,
            ARENA_OUTLINE_WIDTH_PX,
        );
    }
}

fn clear_mesh(mesh: &mut TriangleMesh<3>) {
    mesh.vertices.clear();
    mesh.colors.clear();
    mesh.indices.clear();
}

fn append_mesh(dst: &mut TriangleMesh<3>, src: &TriangleMesh<3>) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend_from_slice(&src.vertices);
    dst.colors.extend_from_slice(&src.colors);
    dst.indices
        .extend(src.indices.iter().map(|t| t.map(|i| i + base)));
}

/// Rebuilds `cap` as one toy's cross-section at `slice`, in world R³, and
/// returns the cap's half-width about its own centroid.
fn append_cap(
    world: &World<EuclideanR4>,
    toy: &ToyBody,
    slice: f32,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    cap: &mut TriangleMesh<3>,
) -> f32 {
    clear_mesh(cap);
    let body = &world.bodies[toy.body];
    let topology = toy.polytope.topology();
    local.clear();
    local.extend(
        (topology.vertices.iter())
            .map(|v| BODY_SIZE * body.orientation.rotation.apply(*v) + Vec4::W * body.position.w),
    );
    let [r, g, b] = toy.color;
    polytope_section_faces_append(
        topology.edges,
        topology.cells,
        local,
        WPlane::new(slice),
        [r, g, b, 1.0],
        scratch,
        cap,
    );
    let extent = cap_extent(&cap.vertices);
    let translate = body.position.truncate();
    for v in &mut cap.vertices {
        v[0] += translate.x;
        v[1] += translate.y;
        v[2] += translate.z;
    }
    extent
}

fn cap_extent(vertices: &[[f32; 3]]) -> f32 {
    if vertices.is_empty() {
        return 0.0;
    }
    let mut centroid = Vec3::ZERO;
    for v in vertices {
        centroid += Vec3::from_array(*v);
    }
    centroid /= vertices.len() as f32;
    (vertices.iter())
        .map(|v| (Vec3::from_array(*v) - centroid).length())
        .fold(0.0, f32::max)
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct PhysicsOverlay {
    contacts: bool,
    normals: bool,
    impulses: bool,
    islands: bool,
    impulse_scale: f32,
    width_px: f32,
}

// Bar length per unit impulse. Measured, not derived: a hard flick,
// [`MAX_CARRY_SPEED`] head-on into a resting neighbour, peaks at 5.48 units of
// accumulated impulse in one contact, which this scale draws at 0.34 of a body
// radius. The calibration deliberately uses a hard flick rather than
// [`MAX_RELEASE_SPEED`], which is a tunneling ceiling an order of magnitude
// above anything a cursor produces. The peak is far under the momentum the
// throw carries in because the solver is not elastic and a hull manifold
// splits the blow over several points.
const DEFAULT_IMPULSE_SCALE: f32 = 0.028;

// Small enough that a full four-point manifold reads as four marks, not a blob.
const CONTACT_CROSS_FRACTION: f32 = 0.15;

const NORMAL_LEN_FRACTION: f32 = 0.9;

const ISLAND_CROSS_FRACTION: f32 = 1.0;

const CONTACT_COLOR: [f32; 4] = [1.00, 0.95, 0.35, 1.0];
const NORMAL_TAIL_COLOR: [f32; 4] = [0.06, 0.24, 0.42, 1.0];
const NORMAL_TIP_COLOR: [f32; 4] = [0.40, 0.95, 1.00, 1.0];
const NORMAL_IMPULSE_COLOR: [f32; 4] = [1.00, 0.30, 0.22, 1.0];
const TANGENT_IMPULSE_COLOR: [f32; 4] = [0.70, 0.40, 1.00, 1.0];

// Six hues: a scene with more than six simultaneous islands repeats a colour,
// so the count is the ceiling on what this layer can claim.
const ISLAND_PALETTE: [[f32; 4]; 6] = [
    [0.35, 0.85, 0.45, 1.0],
    [0.95, 0.55, 0.20, 1.0],
    [0.45, 0.60, 1.00, 1.0],
    [0.95, 0.40, 0.75, 1.0],
    [0.90, 0.90, 0.35, 1.0],
    [0.35, 0.90, 0.90, 1.0],
];

impl Default for PhysicsOverlay {
    fn default() -> Self {
        Self {
            contacts: false,
            normals: false,
            impulses: false,
            islands: false,
            impulse_scale: DEFAULT_IMPULSE_SCALE,
            width_px: 2.0,
        }
    }
}

impl PhysicsOverlay {
    fn any_layer(self) -> bool {
        self.contacts || self.normals || self.impulses || self.islands
    }
}

fn push_overlay_segment(
    mesh: &mut LineMesh<3>,
    from: Vec3,
    to: Vec3,
    from_color: [f32; 4],
    to_color: [f32; 4],
    width: f32,
) {
    mesh.segments.push((from.to_array(), to.to_array()));
    mesh.colors.push((from_color, to_color));
    mesh.widths.push(width);
}

fn push_axis_cross(mesh: &mut LineMesh<3>, centre: Vec3, half: f32, color: [f32; 4], width: f32) {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        push_overlay_segment(
            mesh,
            centre - axis * half,
            centre + axis * half,
            color,
            color,
            width,
        );
    }
}

fn build_physics_overlay_mesh(
    world: &World<EuclideanR4>,
    radius: f32,
    overlay: &PhysicsOverlay,
    mesh: &mut LineMesh<3>,
) {
    mesh.segments.clear();
    mesh.colors.clear();
    mesh.widths.clear();
    if !overlay.any_layer() {
        return;
    }

    let width = overlay.width_px;

    if overlay.contacts || overlay.normals || overlay.impulses {
        for manifold in world.manifolds.values() {
            for cp in &manifold.points {
                let point = cp.world_point.truncate();
                let normal = cp.normal.truncate();
                if overlay.contacts {
                    push_axis_cross(
                        mesh,
                        point,
                        CONTACT_CROSS_FRACTION * radius,
                        CONTACT_COLOR,
                        width,
                    );
                }
                if overlay.normals {
                    push_overlay_segment(
                        mesh,
                        point,
                        point + normal * (NORMAL_LEN_FRACTION * radius),
                        NORMAL_TAIL_COLOR,
                        NORMAL_TIP_COLOR,
                        width,
                    );
                }
                if overlay.impulses {
                    push_overlay_segment(
                        mesh,
                        point,
                        point - normal * (cp.normal_impulse * overlay.impulse_scale),
                        NORMAL_IMPULSE_COLOR,
                        NORMAL_IMPULSE_COLOR,
                        width,
                    );
                    // The two bars name opposite bodies and do not compose:
                    // −tangent_dir is the impulse on B, not on A.
                    push_overlay_segment(
                        mesh,
                        point,
                        point
                            - cp.tangent_dir.truncate()
                                * (cp.tangent_impulse * overlay.impulse_scale),
                        TANGENT_IMPULSE_COLOR,
                        TANGENT_IMPULSE_COLOR,
                        width,
                    );
                }
            }
        }
    }

    if overlay.islands {
        for (ordinal, island) in world.islands().iter().enumerate() {
            let color = ISLAND_PALETTE[ordinal % ISLAND_PALETTE.len()];
            for &id in &island.bodies {
                push_axis_cross(
                    mesh,
                    world.bodies[id].position.truncate(),
                    ISLAND_CROSS_FRACTION * radius,
                    color,
                    width,
                );
            }
            for &(a, b) in &island.constraints {
                // A pair with a static side absorbs no impulse and merges
                // nothing (`World::fill_islands`), so a bar from a landed toy
                // to the floor body's origin would draw a coupling the solver
                // does not have.
                if world.bodies[a].inv_mass == 0.0 || world.bodies[b].inv_mass == 0.0 {
                    continue;
                }
                push_overlay_segment(
                    mesh,
                    world.bodies[a].position.truncate(),
                    world.bodies[b].position.truncate(),
                    color,
                    color,
                    width,
                );
            }
        }
    }
}

/// Everything the toybox console writes to. The overlay is one field of it so
/// the `physics` verb keeps its own vocabulary while `ground` and `wlabels`
/// reach the scene's other live settings.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub(crate) struct ToyboxControls {
    overlay: PhysicsOverlay,
    /// Per-body `w` readout painted over each toy. Off by default: the fade
    /// with cross-section size already says a body is leaving the slice.
    w_labels: bool,
    environment: Environment,
}

fn register_toybox_commands(console: &mut Console<ToyboxControls>) {
    register_ground_command(console, |c| &mut c.environment);
    console.register(
        loam_egui::cmd::<ToyboxControls, _>(
            "wlabels",
            "per-body w offset labels over each toy (on | off; bare flips)",
            |args, controls, out| {
                let next = match args.first().copied() {
                    None => !controls.w_labels,
                    Some("on") => true,
                    Some("off") => false,
                    Some(other) => {
                        return Err(anyhow!("wlabels: unknown arg `{other}` (try on|off)"));
                    }
                };
                controls.w_labels = next;
                out.line(format!("wlabels: {}", if next { "on" } else { "off" }));
                Ok(())
            },
        )
        .with_args(&[&["on", "off"]]),
    );
    console.register(
        loam_egui::subcommands::<ToyboxControls>(
            "physics",
            "solver debug overlay (bare flips all four layers)",
        )
        .on_bare(|c: &mut ToyboxControls| {
            let o = &mut c.overlay;
            let on = !o.any_layer();
            o.contacts = on;
            o.normals = on;
            o.impulses = on;
            o.islands = on;
            Ok(())
        })
        .toggle(
            "contacts",
            "axis cross at each contact point (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.contacts = v.unwrap_or(!o.contacts);
                Ok(())
            },
        )
        .toggle(
            "normals",
            "contact normal, drawn dark-to-bright along the A-toward-B direction (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.normals = v.unwrap_or(!o.normals);
                Ok(())
            },
        )
        .toggle(
            "impulses",
            "accumulated normal + tangent impulse bars (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.impulses = v.unwrap_or(!o.impulses);
                Ok(())
            },
        )
        .toggle(
            "islands",
            "colour each island's bodies and coupling constraints (bare flips)",
            |c: &mut ToyboxControls, v| {
                let o = &mut c.overlay;
                o.islands = v.unwrap_or(!o.islands);
                Ok(())
            },
        )
        .custom(
            "impulse-scale",
            "world units of bar length per unit of accumulated impulse (default 0.04)",
            &[&[]],
            &[],
            |c: &mut ToyboxControls, args, out| {
                let o = &mut c.overlay;
                match args.first().copied() {
                    None => out.line(format!("physics impulse-scale: {:.4}", o.impulse_scale)),
                    Some(token) => {
                        let s: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid impulse scale `{token}`: {e}"))?;
                        if !(s.is_finite() && s > 0.0 && s <= 100.0) {
                            return Err(anyhow!(
                                "impulse scale {s} out of range; expected a float in (0, 100]"
                            ));
                        }
                        o.impulse_scale = s;
                        out.line(format!("physics impulse-scale: set to {s:.4}"));
                    }
                }
                Ok(())
            },
        )
        .custom(
            "width",
            "overlay line thickness in pixels (default 2.0)",
            &[&[]],
            &[],
            |c: &mut ToyboxControls, args, out| {
                let o = &mut c.overlay;
                match args.first().copied() {
                    None => out.line(format!("physics width: {:.2} px", o.width_px)),
                    Some(token) => {
                        let w: f32 = token
                            .parse()
                            .map_err(|e| anyhow!("invalid width `{token}`: {e}"))?;
                        if !(w > 0.0 && w <= 16.0) {
                            return Err(anyhow!(
                                "physics width {w} out of range; expected a float in (0, 16]"
                            ));
                        }
                        o.width_px = w;
                        out.line(format!("physics width: set to {w:.2} px"));
                    }
                }
                Ok(())
            },
        ),
    );
}

// 24-bit depth cracks the thin, densely stacked caps of a tumbling 24-cell.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// Far enough that the whole arena floor is inside the frustum at 4:3 and
// wider, which `the_boot_camera_frames_the_whole_arena` pins.
const BOOT_ORBIT_DISTANCE: f32 = 9.0;
const BOOT_ORBIT_PITCH: f32 = -0.16;
const BOOT_TARGET_HEIGHT: f32 = 0.7;

const CAMERA_FOV_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.05;
const CAMERA_FAR: f32 = 200.0;

fn boot_orbit() -> OrbitController<EuclideanR3> {
    let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
    orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);
    orbit.target.y = BOOT_TARGET_HEIGHT;
    orbit
}

fn build_caps(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
    depth: DepthMode,
) -> TriangleRasterNode {
    TriangleRasterNode::new(
        device,
        format,
        depth,
        loam_render::triangle_raster::FragmentShading::FaceNormalLambert,
        samples,
    )
}

pub(crate) struct ToyboxScene {
    toybox: Toybox,
    seed: u64,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    console: Console<ToyboxControls>,
    caps: TriangleRasterNode,
    faded_caps: TriangleRasterNode,
    sky_ground: SkyGroundNode,
    depth: Option<DepthBuffer>,
    opaque_mesh: TriangleMesh<3>,
    faded_mesh: TriangleMesh<3>,
    controls: ToyboxControls,
    line_node: LineRasterNode,
    line_mesh: LineMesh<3>,
    left_was_down: bool,
    slice_up_held: bool,
    slice_down_held: bool,
    paused: bool,
}

impl ToyboxScene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let mut console = Console::<ToyboxControls>::new();
        loam_app::shell::register_command::<ToyboxControls, crate::shell::Playground>(&mut console);
        register_toybox_commands(&mut console);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.0, BOOT_ORBIT_DISTANCE);
        camera.near = 0.05;
        let orbit = boot_orbit();

        Ok(Self {
            toybox: Toybox::new(DEFAULT_SEED),
            seed: DEFAULT_SEED,
            camera,
            orbit,
            console,
            caps: build_caps(
                &ctx.rd.device,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
                DepthMode::ReadWrite {
                    format: DEPTH_FORMAT,
                },
            ),
            faded_caps: build_caps(
                &ctx.rd.device,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
                DepthMode::ReadOnly {
                    format: DEPTH_FORMAT,
                },
            ),
            sky_ground: SkyGroundNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DEPTH_FORMAT,
                ctx.rd.sample_count(),
            ),
            depth: None,
            opaque_mesh: TriangleMesh::<3>::default(),
            faded_mesh: TriangleMesh::<3>::default(),
            controls: ToyboxControls::default(),
            // No depth attachment: the overlay names where the solver put a
            // contact, which is behind the hull that owns it.
            line_node: LineRasterNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DepthMode::Off,
                ctx.rd.sample_count(),
            ),
            line_mesh: LineMesh::<3>::default(),
            left_was_down: false,
            slice_up_held: false,
            slice_down_held: false,
            paused: false,
        })
    }

    fn respawn(&mut self, seed: u64) {
        self.toybox = Toybox::new(seed);
        self.seed = seed;
    }

    fn panel(&mut self, ctx: &egui::Context) {
        let mut respawn: Option<u64> = None;
        egui::Window::new("Toybox")
            .id(egui::Id::new("toybox-scene-controls"))
            .default_pos(egui::pos2(16.0, 48.0))
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("left-drag a shape to carry it; let go to throw it");
                ui.label("hold Shift while dragging to pull it through w");
                ui.label("right-drag to orbit the camera");
                ui.separator();
                let mut slice = self.toybox.slice();
                if ui
                    .add(
                        egui::Slider::new(&mut slice, -W_SLICE_RANGE..=W_SLICE_RANGE)
                            .text("w slice (Up/Down)"),
                    )
                    .changed()
                {
                    self.toybox.set_slice(slice);
                }
                ui.separator();
                for (index, w) in self.toybox.w_offsets() {
                    ui.monospace(format!("#{index}  w {w:+.3}"));
                }
                ui.separator();
                ui.checkbox(&mut self.controls.w_labels, "w labels (wlabels)");
                ui.checkbox(&mut self.paused, "pause (Space)");
                if ui.button("respawn (R)").clicked() {
                    respawn = Some(self.seed);
                }
                if ui.button("next seed (N)").clicked() {
                    respawn = Some(self.seed.wrapping_add(1));
                }
                ui.label(
                    egui::RichText::new(format!("seed {:#x}", self.seed))
                        .small()
                        .weak(),
                );
            });
        if let Some(seed) = respawn {
            self.respawn(seed);
        }
    }

    /// One line pass for the frame: the container footprint, then whichever
    /// solver overlay layers are on. One node and one upload, because a second
    /// `write_buffer` in a frame lands before the whole command buffer and
    /// would feed both passes.
    fn record_lines(&mut self, ctx: &mut RenderCtx<'_>, view_proj: Mat4) {
        let rd = &ctx.rd;
        let cfg = &rd.surface_bundle.config;
        let mut mesh = std::mem::take(&mut self.line_mesh);
        self.toybox
            .build_overlay_mesh(&self.controls.overlay, &mut mesh);
        append_arena_outline(&mut mesh);
        self.line_node.set_camera(
            &rd.queue,
            view_proj,
            Vec2::new(cfg.width as f32, cfg.height as f32),
        );
        self.line_node.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &mesh,
            &Projection::Identity,
            1,
        );
        self.line_mesh = mesh;
        self.line_node.record(ctx.encoder, ctx.view, None, None);
    }

    // A body off the slice draws a shrinking cap or none at all, so the offset
    // is the only thing left saying it is still there.
    fn w_readouts(&self, ctx: &egui::Context, frame: &FrameCtx<'_>) {
        if !self.controls.w_labels {
            return;
        }
        let ppp = ctx.pixels_per_point();
        let cfg = &frame.rd.surface_bundle.config;
        let viewport = (
            (cfg.width as f32 / ppp).round() as u32,
            (cfg.height as f32 / ppp).round() as u32,
        );
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("toybox-w-readout"),
        ));
        for (index, w) in self.toybox.w_offsets() {
            if w.abs() < W_LABEL_MIN {
                continue;
            }
            let world = self.toybox.position(index).truncate();
            let Some(anchor) =
                loam_egui::world_to_screen(&self.camera, world, viewport, &EuclideanR3)
            else {
                continue;
            };
            painter.text(
                anchor,
                egui::Align2::CENTER_CENTER,
                format!("w {w:+.2}"),
                egui::FontId::monospace(12.0),
                egui::Color32::from_rgb(232, 198, 120),
            );
        }
    }
}

impl loam_app::shell::Scene for ToyboxScene {
    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> Result<()> {
        self.console
            .dispatch(&cmd.name, &cmd.arg_refs(), &mut self.controls);
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        let cfg = &ctx.rd.surface_bundle.config;
        let viewport = (cfg.width, cfg.height);
        self.camera.aspect = viewport.0 as f32 / viewport.1.max(1) as f32;

        let down = ctx.input.buttons.left.down;
        let pressed = down && !self.left_was_down;
        let released = !down && self.left_was_down;
        self.left_was_down = down;

        let forward = self.camera.view().forward;
        let grabbing = !ctx.ui_capture.pointer;
        if !grabbing {
            self.toybox.release();
        } else if pressed {
            if let Some(px) = ctx.input.buttons.left.press_pos {
                let ray = self.camera.ray_from_ndc(ndc_from_pixels(px, viewport));
                self.toybox.press(&ray, forward);
            }
        } else if released {
            self.toybox.release();
        } else if let Some(px) = ctx.input.cursor_pos {
            let axis = if ctx.input.modifiers.shift {
                GrabAxis::Through
            } else {
                GrabAxis::Slice
            };
            let ray = self.camera.ray_from_ndc(ndc_from_pixels(px, viewport));
            // The trail is measured in sim time, not wall time, so the release
            // speed replays with the tick stream.
            let dt = ctx.n_ticks as f32 / TICK_HZ as f32;
            self.toybox.hold(&ray, forward, axis, dt);
        }

        let dir = (self.slice_up_held as i32 - self.slice_down_held as i32) as f32;
        if dir != 0.0 {
            self.toybox
                .scrub_slice(dir, ctx.n_ticks as f32 / TICK_HZ as f32);
        }

        if !self.paused {
            for _ in 0..ctx.n_ticks {
                self.toybox.tick();
            }
        }

        if !ctx.ui_capture.pointer {
            // `OrbitController` drives off the left button. The left button is
            // the grab here, so the right one is handed to it under that name
            // and neither path has to know about the other.
            let mut input = ctx.input;
            input.left_mouse_down = input.buttons.right.down;
            input.buttons.left = input.buttons.right;
            self.orbit
                .advance(input, &mut self.camera, &EuclideanR3, ctx.dt);
        }
    }

    fn ui(&mut self, ctx: &egui::Context, frame: &mut FrameCtx<'_>) {
        self.panel(ctx);
        self.w_readouts(ctx, frame);
        loam_app::log::pump_into(&mut self.console);
        loam_app::command::pump_into(&mut self.console);
        self.console.ui(ctx);
        loam_app::command::forward_pending(&mut self.console);
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        // A release always clears the held flag, even when the console took
        // the key down: otherwise the slice scrubs on forever.
        let pressed = state == ElementState::Pressed && !ctx.ui_capture.keyboard;
        match code {
            KeyCode::ArrowUp => self.slice_up_held = pressed,
            KeyCode::ArrowDown => self.slice_down_held = pressed,
            KeyCode::Space if pressed => self.paused = !self.paused,
            KeyCode::KeyR if pressed => self.respawn(self.seed),
            KeyCode::KeyN if pressed => self.respawn(self.seed.wrapping_add(1)),
            _ => {}
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let rd = &ctx.rd;
        let cfg = &rd.surface_bundle.config;

        DepthBuffer::ensure(
            &mut self.depth,
            &rd.device,
            DEPTH_FORMAT,
            (cfg.width, cfg.height),
            rd.sample_count(),
        );
        let depth = self.depth.as_ref().expect("ensure() guarantees Some"); // ok: DepthBuffer::ensure fills the slot on the line above

        let view = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view_mat = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj_mat =
            Mat4::perspective_rh(CAMERA_FOV_DEG.to_radians(), aspect, CAMERA_NEAR, CAMERA_FAR);
        let view_proj = proj_mat * view_mat;

        // This scene owns the frame's clear; both cap passes load.
        self.sky_ground.set_uniforms(
            &rd.queue,
            &SkyGroundUniforms::new(
                view_proj,
                Viewport::full([cfg.width, cfg.height]),
                self.controls.environment.ground(FLOOR_Y, true),
            ),
        );
        self.sky_ground
            .record(ctx.encoder, ctx.view, &depth.view, None);

        self.toybox
            .build_frame_meshes(&mut self.opaque_mesh, &mut self.faded_mesh);
        for (node, mesh) in [
            (&mut self.caps, &self.opaque_mesh),
            (&mut self.faded_caps, &self.faded_mesh),
        ] {
            node.upload::<EuclideanR3, 3>(&rd.device, &rd.queue, mesh, &Projection::Identity);
            node.set_camera(&rd.queue, view_proj);
            node.record(ctx.encoder, ctx.view, Some(&depth.view), None);
        }
        self.record_lines(ctx, view_proj);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - toybox")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_probe;
    use loam_math::Plane4;
    use loam_physics::euclidean_r4::sphere_body_r4;
    use loam_physics::manifold::PENETRATION_SLOP;

    // The drop takes 12 ticks and the rest latch 30 more; the rest is margin.
    const SETTLE_TICKS: usize = 120;

    // A throw at [`MAX_RELEASE_SPEED`] closes the gap between two
    // [`SPAWN_SPACING`] neighbours' bounding spheres in two ticks; the rest is
    // the solver working the impulse up to its peak.
    const FLICK_TICKS: usize = 10;

    const SEEDS: u64 = 24;

    // Camera looking down -z from above the row, the boot framing's axis.
    const EYE: Vec3 = Vec3::new(0.0, 1.0, 8.0);
    const FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

    fn scene() -> Toybox {
        Toybox::new(DEFAULT_SEED)
    }

    fn settled() -> Toybox {
        let mut toybox = scene();
        toybox.run(SETTLE_TICKS);
        toybox
    }

    fn ray_through(target: Vec3) -> Ray {
        Ray {
            origin: EYE,
            direction: (target - EYE).normalize(),
        }
    }

    // One frame of a held drag: move the cursor to `target`, then step.
    fn drag_frame(toybox: &mut Toybox, target: Vec3, axis: GrabAxis) {
        toybox.hold(&ray_through(target), FORWARD, axis, 1.0 / TICK_HZ as f32);
        toybox.tick();
    }

    fn grab_centre(toybox: &mut Toybox, toy: usize) -> bool {
        let centre = toybox.position(toy).truncate();
        toybox.press(&ray_through(centre), FORWARD)
    }

    fn peak_w(toybox: &Toybox) -> f32 {
        toybox.w_offsets().map(|(_, w)| w.abs()).fold(0.0, f32::max)
    }

    // Camera at the boot framing and a given aspect, built the way `record`
    // builds it, so the framing pin cannot drift from the frame.
    fn boot_view_proj(aspect: f32) -> Mat4 {
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.aspect = aspect;
        boot_orbit().advance(
            loam_app::Input::default(),
            &mut camera,
            &EuclideanR3,
            1.0 / 60.0,
        );
        let view = camera.view();
        Mat4::perspective_rh(CAMERA_FOV_DEG.to_radians(), aspect, CAMERA_NEAR, CAMERA_FAR)
            * Mat4::look_to_rh(view.position, view.forward, view.up)
    }

    fn hull_reach(toybox: &Toybox) -> (f32, f32) {
        let (mut x, mut z) = (0.0_f32, 0.0_f32);
        for body in toybox.world.bodies.iter() {
            let loam_physics::Collider::ConvexPolytope4D { vertices } = &body.collider else {
                continue;
            };
            for v in vertices {
                let world = body.orientation.rotation.apply(*v) + body.position;
                x = x.max(world.x.abs());
                z = z.max(world.z.abs());
            }
        }
        (x, z)
    }

    #[test]
    fn held_arrows_and_the_panel_slider_move_the_slice_and_clamp_at_its_range() {
        let mut toybox = scene();
        assert_eq!(toybox.slice(), W_SLICE, "the scene boots off its own slice");

        // One second of held Up at the scrub rate the rotate scene uses.
        for _ in 0..TICK_HZ {
            toybox.scrub_slice(1.0, 1.0 / TICK_HZ as f32);
        }
        assert!(
            (toybox.slice() - W_SCRUB_RATE).abs() < 1e-4,
            "a second of held Up moved the slice to {}",
            toybox.slice()
        );

        for _ in 0..TICK_HZ * 10 {
            toybox.scrub_slice(1.0, 1.0 / TICK_HZ as f32);
        }
        assert_eq!(
            toybox.slice(),
            W_SLICE_RANGE,
            "the scrub ran past its range"
        );
        for _ in 0..TICK_HZ * 20 {
            toybox.scrub_slice(-1.0, 1.0 / TICK_HZ as f32);
        }
        assert_eq!(toybox.slice(), -W_SLICE_RANGE);

        // The panel slider writes through `set_slice`, which is the same clamp.
        toybox.set_slice(100.0);
        assert_eq!(toybox.slice(), W_SLICE_RANGE);
        toybox.set_slice(-100.0);
        assert_eq!(toybox.slice(), -W_SLICE_RANGE);
        toybox.set_slice(0.25);
        assert_eq!(toybox.slice(), 0.25);
    }

    #[test]
    fn moving_the_slice_moves_which_cross_sections_are_drawn() {
        let mut toybox = settled();
        let (mut opaque, mut faded) = (TriangleMesh::default(), TriangleMesh::default());
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        let at_home = opaque.vertices.len();
        assert!(
            at_home > 0,
            "the settled pile drew nothing at its own slice"
        );

        toybox.set_slice(W_SLICE_RANGE);
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert!(
            opaque.vertices.is_empty() && faded.vertices.is_empty(),
            "a slice {W_SLICE_RANGE} away from the pile still cut it"
        );
        for toy in 0..toybox.toys().len() {
            assert_eq!(toybox.cap_stats(toy), (0.0, 0), "toy {toy} still has a cap");
        }

        toybox.set_slice(W_SLICE);
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert_eq!(
            opaque.vertices.len(),
            at_home,
            "the slice came home to a different pile"
        );
    }

    #[test]
    fn a_tick_takes_only_the_substeps_its_fastest_body_needs() {
        let mut toybox = settled();
        assert_eq!(
            toybox.substeps_for_current_speed(),
            BASE_SUBSTEPS,
            "a settled pile should cost the settling rate and nothing more"
        );

        let thrown = toybox.toys[0].body;
        toybox.world.bodies[thrown].velocity = Vec4::new(MAX_RELEASE_SPEED, 0.0, 0.0, 0.0);
        let fast = toybox.substeps_for_current_speed();
        assert!(
            fast > BASE_SUBSTEPS && fast <= MAX_SUBSTEPS,
            "a body at the ceiling asked for {fast} substeps"
        );
        assert!(
            toybox.fastest_step_travel() <= STEP_TRAVEL_BUDGET,
            "the chosen substep count still lets a step outrun the narrowphase"
        );
    }

    #[test]
    fn the_substep_count_comes_from_body_state_and_not_the_clock() {
        // Determinism: two runs of the same seed must divide their ticks the
        // same way, or a replay is not a replay.
        fn counts(seed: u64) -> Vec<usize> {
            let mut toybox = Toybox::new(seed);
            (0..120)
                .map(|_| {
                    let n = toybox.substeps_for_current_speed();
                    toybox.tick();
                    n
                })
                .collect()
        }
        assert_eq!(counts(DEFAULT_SEED), counts(DEFAULT_SEED));
    }

    #[test]
    fn a_held_body_is_never_driven_into_a_wall_or_the_floor() {
        let reach = ARENA_HALF_EXTENT - BODY_SIZE;
        for corner in [
            Vec4::new(99.0, -99.0, 99.0, 0.0),
            Vec4::new(-99.0, -5.0, -99.0, 2.0),
        ] {
            let held = clamp_target_to_arena(corner);
            assert!(held.x.abs() <= reach + 1e-6 && held.z.abs() <= reach + 1e-6);
            assert!(held.y >= FLOOR_Y + BODY_SIZE - 1e-6);
            assert_eq!(held.w, corner.w, "the clamp must leave w alone");
        }
    }

    #[test]
    fn a_flick_into_a_wall_still_throws() {
        // The clamp bounds where the body is DRIVEN. If it also bounded what the
        // release reads, a flick toward a wall would pin the target and throw
        // nothing, which is the bug the intent accumulator exists to prevent.
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0));
        let mut cursor = toybox.position(0).truncate();
        for _ in 0..12 {
            cursor += Vec3::new(3.0, 0.0, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        toybox.release();
        assert!(
            toybox.velocity(0).length() > 1.0,
            "a hard flick past the wall threw {}",
            toybox.velocity(0).length()
        );
    }

    #[test]
    fn a_faster_drag_throws_harder_all_the_way_to_the_ceiling() {
        // The property a flat cap destroyed: every flick above it landed on one
        // speed, so a fast mouse threw no harder than a medium one.
        fn thrown_at(units_per_frame: f32) -> f32 {
            let mut toybox = settled();
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..12 {
                cursor += Vec3::new(units_per_frame, 0.0, 0.0);
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            toybox.release();
            toybox.velocity(0).length()
        }

        let gentle = thrown_at(0.05);
        let medium = thrown_at(0.5);
        let hard = thrown_at(4.0);
        assert!(
            gentle < medium && medium < hard,
            "throws did not order with cursor speed: {gentle}, {medium}, {hard}"
        );
        assert!(
            hard > 4.0 * medium,
            "a ten-times-faster drag threw only {hard} against {medium}, so the              top of the range is still flattened"
        );
        assert!(
            hard <= MAX_RELEASE_SPEED,
            "a throw at {hard} passed the narrowphase ceiling {MAX_RELEASE_SPEED}"
        );
    }
    #[test]
    fn the_walls_hold_a_throw_at_the_narrowphase_ceiling_for_its_whole_flight() {
        // The ceiling, not the feel speed: the walls have to hold the fastest
        // thing the solver will carry, not the fastest thing a hand throws.
        let reach = ARENA_HALF_EXTENT + 8.0 * PENETRATION_SLOP + BODY_SIZE;
        for direction in [
            Vec4::X,
            -Vec4::X,
            Vec4::Z,
            -Vec4::Z,
            Vec4::new(1.0, 0.4, 1.0, 0.0).normalize(),
            Vec4::new(-1.0, 0.6, 0.7, 0.0).normalize(),
        ] {
            let mut toybox = settled();
            let thrown = toybox.toys[0].body;
            toybox.wake(0);
            toybox.world.bodies[thrown].velocity = direction * MAX_RELEASE_SPEED;
            for tick in 0..600 {
                toybox.tick();
                let (x, z) = hull_reach(&toybox);
                assert!(
                    x <= reach && z <= reach,
                    "a throw along {direction} reached x {x}, z {z} at tick {tick}, \
                     outside the container's {ARENA_HALF_EXTENT}"
                );
            }
        }
    }

    #[test]
    fn the_boot_camera_frames_the_whole_arena() {
        let e = ARENA_HALF_EXTENT;
        // 4:3 is the narrowest aspect the framing has to hold; 16:9 is wider.
        for aspect in [4.0 / 3.0, 16.0 / 9.0] {
            let view_proj = boot_view_proj(aspect);
            for corner in [
                Vec3::new(-e, FLOOR_Y, -e),
                Vec3::new(e, FLOOR_Y, -e),
                Vec3::new(e, FLOOR_Y, e),
                Vec3::new(-e, FLOOR_Y, e),
            ] {
                let clip = view_proj * corner.extend(1.0);
                assert!(clip.w > 0.0, "corner {corner} is behind the boot camera");
                let ndc = clip.truncate() / clip.w;
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                    "aspect {aspect}: arena corner {corner} lands at ndc {ndc}, \
                     outside the boot frame"
                );
            }
        }
    }

    #[test]
    fn a_body_the_slice_cannot_see_is_still_pickable_through_its_bounding_ball() {
        let mut toybox = settled();
        let ray = ray_through(toybox.position(0).truncate());
        assert_eq!(toybox.pick(&ray).map(|(toy, _)| toy), Some(0));

        // Push toy 0 clean off the slice, so it draws nothing to aim at.
        let id = toybox.toys[0].body;
        toybox.world.bodies[id].position.w += 4.0 * BODY_SIZE;
        assert_eq!(toybox.cap_stats(0), (0.0, 0), "the toy still has a cap");

        let (picked, hit) = toybox
            .pick(&ray)
            .expect("the off-slice body is unreachable");
        assert_eq!(picked, 0, "the fallback grabbed the wrong body");
        assert!(
            (hit - toybox.position(0).truncate()).length() <= BODY_SIZE + 1e-4,
            "the fallback reported a hit {hit} away from the body's ball"
        );
        assert!(
            toybox.press(&ray, FORWARD),
            "the off-slice body cannot be grabbed"
        );
        assert_eq!(toybox.grab.as_ref().map(|g| g.toy), Some(0));
    }

    #[test]
    fn the_bounding_ball_fallback_never_fires_for_a_body_the_slice_can_see() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        // Inside every toy's bounding ball and outside every cap: the fallback
        // would answer here if a drawn cross-section did not veto it.
        let mut inside_ball = None;
        for step in 1..40 {
            let radius = BODY_SIZE * step as f32 / 40.0;
            let probe = centre + Vec3::new(radius, radius, 0.0);
            if (probe - centre).length() < BODY_SIZE && toybox.pick(&ray_through(probe)).is_none() {
                inside_ball = Some(probe);
                break;
            }
        }
        let probe = inside_ball.expect("no ray inside the ball missed every cap");
        let ray = ray_through(probe);
        assert!(
            ray_ball_distance(&ray, centre, BODY_SIZE).is_some(),
            "the probe ray misses the ball, so the veto pin is vacuous"
        );
        assert_eq!(toybox.pick(&ray), None);
    }

    #[test]
    fn the_ball_pick_takes_the_near_hit_and_rejects_a_ball_behind_the_eye() {
        let centre = Vec3::new(0.0, 0.0, -4.0);
        let ray = Ray {
            origin: Vec3::ZERO,
            direction: -Vec3::Z,
        };
        let hit = ray_ball_distance(&ray, centre, 1.0).expect("a ray down the axis hits");
        assert!(
            (hit - 3.0).abs() < 1e-5,
            "the ball entered at {hit}, not 3.0"
        );

        // Origin inside: the only positive root is the exit.
        let inside = ray_ball_distance(&ray, Vec3::new(0.0, 0.0, -0.5), 1.0).expect("inside hits");
        assert!(
            (inside - 1.5).abs() < 1e-5,
            "an inside ray left at {inside}"
        );

        assert_eq!(ray_ball_distance(&ray, Vec3::new(0.0, 0.0, 4.0), 1.0), None);
        assert_eq!(
            ray_ball_distance(&ray, Vec3::new(3.0, 0.0, -4.0), 1.0),
            None
        );
    }

    #[test]
    fn angular_damping_decays_a_free_flight_spin_and_leaves_the_contact_w_drift() {
        let mut toybox = settled();
        let id = toybox.toys[0].body;
        // Well clear of the floor, so nothing but the damper touches the spin.
        toybox.world.bodies[id].position += Vec4::Y * 6.0;
        toybox.wake(0);
        toybox.world.bodies[id].angular_velocity = Bivector4::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.6);
        let launched = toybox.angular_velocity(0).magnitude();

        let mut last = launched;
        for _ in 0..TICK_HZ {
            toybox.tick();
            let now = toybox.angular_velocity(0).magnitude();
            assert!(now <= last + 1e-6, "the spin grew in free flight");
            last = now;
        }
        let expected = launched * (-ANGULAR_DAMPING).exp();
        assert!(
            (last - expected).abs() < 1e-3 * launched,
            "a second of free flight left {last} of spin, not the {expected} the \
             damping coefficient prescribes"
        );
        assert!(
            last < 0.4 * launched,
            "a knocked body still tumbles at {} of its launch spin after a second",
            last / launched
        );

        // The same damping in place, a hull-against-hull throw still drives a
        // neighbour past the band the floor alone keeps every toy inside.
        let mut toybox = settled();
        assert!(peak_w(&toybox) < SETTLED_W_BAND);
        assert!(grab_centre(&mut toybox, 0));
        let target = toybox.position(1).truncate();
        let mut cursor = toybox.position(0).truncate();
        let step = (target - cursor) / 12.0;
        for _ in 0..12 {
            cursor += step;
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        toybox.release();
        let mut worst = 0.0_f32;
        for _ in 0..180 {
            toybox.tick();
            worst = worst.max(peak_w(&toybox));
        }
        assert!(
            worst > SETTLED_W_BAND,
            "angular damping suppressed the w drift a contact imparts (best {worst})"
        );
    }

    #[test]
    fn the_toybox_console_carries_the_w_label_and_ground_verbs_with_the_label_off() {
        let mut console = Console::<ToyboxControls>::new();
        register_toybox_commands(&mut console);
        for verb in ["wlabels", "ground", "physics"] {
            assert!(console.has_command(verb), "the console lost `{verb}`");
        }

        let mut controls = ToyboxControls::default();
        assert!(!controls.w_labels, "the w readout ships on");
        assert!(!controls.overlay.any_layer());
        assert_eq!(controls.environment, Environment::default());

        console.dispatch("wlabels", &[], &mut controls);
        assert!(controls.w_labels, "a bare wlabels did not flip the label");
        console.dispatch("wlabels", &["off"], &mut controls);
        assert!(!controls.w_labels);
        console.dispatch("wlabels", &["on"], &mut controls);
        assert!(controls.w_labels);

        console.dispatch("ground", &["fog", "0.09"], &mut controls);
        assert_eq!(controls.environment.fog_per_unit, 0.09);
        assert!(controls.w_labels, "the ground verb reached the wrong field");
    }

    #[test]
    fn the_container_outline_traces_the_walls_the_solver_holds() {
        let mut mesh = LineMesh::<3>::default();
        append_arena_outline(&mut mesh);
        assert_eq!(mesh.segments.len(), 4, "the footprint is a four-sided loop");
        assert_eq!(mesh.colors.len(), 4);
        assert_eq!(mesh.widths.len(), 4);
        for &(from, to) in &mesh.segments {
            for point in [Vec3::from_array(from), Vec3::from_array(to)] {
                assert_eq!(point.y, FLOOR_Y);
                assert!(
                    (point.x.abs() - ARENA_HALF_EXTENT).abs() < 1e-6
                        || (point.z.abs() - ARENA_HALF_EXTENT).abs() < 1e-6,
                    "outline corner {point} is off the wall planes"
                );
            }
        }
        for pair in mesh.segments.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert_eq!(mesh.segments[3].1, mesh.segments[0].0);
    }

    #[test]
    fn every_wall_normal_is_free_of_w() {
        let toybox = scene();
        let mut walls = 0;
        for body in toybox.world.bodies.iter() {
            let loam_physics::Collider::HalfSpace4D { normal, .. } = body.collider else {
                continue;
            };
            walls += 1;
            assert_eq!(
                normal.w, 0.0,
                "a static plane's normal carries w, so its normal impulse would \
                 push the pile off the slice"
            );
        }
        assert_eq!(walls, 5, "the scene lost a wall or the floor");
    }

    #[test]
    fn every_toy_falls_to_the_floor_and_settles_there() {
        let mut toybox = scene();
        assert!(
            toybox.deepest_point() > SPAWN_CLEARANCE - 1e-5,
            "a toy spawned inside the floor"
        );
        let spawned: Vec<f32> = (0..toybox.toys().len())
            .map(|i| toybox.position(i).y)
            .collect();

        toybox.run(SETTLE_TICKS);
        for (toy, start) in spawned.iter().enumerate() {
            assert!(toybox.position(toy).y < *start, "toy {toy} never fell");
            assert_eq!(
                toybox.velocity(toy),
                Vec4::ZERO,
                "toy {toy} never came to rest"
            );
        }
        let deepest = toybox.deepest_point();
        assert!(
            deepest > -2.0 * PENETRATION_SLOP,
            "a toy sank to {deepest} through the floor"
        );
        assert!(
            deepest < 0.02,
            "the pile came to rest {deepest} above the floor rather than on it"
        );

        let resting: Vec<Vec4> = (0..toybox.toys().len())
            .map(|i| toybox.position(i))
            .collect();
        toybox.run(600);
        let after: Vec<Vec4> = (0..toybox.toys().len())
            .map(|i| toybox.position(i))
            .collect();
        assert_eq!(resting, after, "a settled pile kept creeping");
    }

    #[test]
    fn nothing_crosses_the_floor_or_outruns_the_resolvable_step() {
        let mut toybox = scene();
        for _ in 0..SETTLE_TICKS {
            toybox.tick();
            let deepest = toybox.deepest_point();
            assert!(
                deepest > -8.0 * PENETRATION_SLOP,
                "a toy reached {deepest} below the floor at tick {}",
                toybox.tick_index()
            );
            let travel = toybox.fastest_step_travel();
            assert!(
                travel < RESOLVABLE_STEP_TRAVEL,
                "a toy covered {travel} in one step at tick {}",
                toybox.tick_index()
            );
        }
    }

    #[test]
    fn the_floor_alone_leaves_every_toy_inside_the_slice_band() {
        for step in 0..SEEDS {
            let mut toybox = Toybox::new(DEFAULT_SEED.wrapping_add(step));
            toybox.run(SETTLE_TICKS);
            for (index, w) in toybox.w_offsets() {
                assert!(
                    w.abs() < SETTLED_W_BAND,
                    "seed {step}: landing alone put toy {index} at w {w}, outside \
                     the band the readout treats as still in the slice"
                );
            }
            for toy in 0..toybox.toys().len() {
                let (alpha, vertices) = toybox.cap_stats(toy);
                assert!(
                    vertices > 0 && alpha >= 1.0,
                    "seed {step}: toy {toy} landed already faded ({vertices} cap \
                     vertices at alpha {alpha}), so the pile boots incomplete"
                );
            }
        }
    }

    #[test]
    fn the_pile_replays_bit_for_bit_from_its_seed() {
        let trace = |seed: u64| {
            let mut toybox = Toybox::new(seed);
            let mut samples = Vec::with_capacity(SETTLE_TICKS);
            for _ in 0..SETTLE_TICKS {
                toybox.tick();
                samples.push(
                    (0..toybox.toys().len())
                        .map(|i| toybox.position(i).to_array())
                        .collect::<Vec<_>>(),
                );
            }
            samples
        };
        assert_eq!(trace(DEFAULT_SEED), trace(DEFAULT_SEED));
        assert_ne!(
            trace(DEFAULT_SEED),
            trace(DEFAULT_SEED ^ 0xdead_beef),
            "two seeds spawned the same pile, so the seed is decoration"
        );
    }

    #[test]
    fn the_spawn_pose_lands_a_whole_cell_on_the_floor() {
        for polytope in [
            Polytope4::Pentatope,
            Polytope4::Tesseract,
            Polytope4::Cell16,
            Polytope4::Cell24,
        ] {
            let topology = polytope.topology();
            for cell in [0, topology.cells.len() / 2, topology.cells.len() - 1] {
                let pose = face_down_pose(polytope, cell);
                let posed: Vec<Vec4> = (topology.vertices.iter()).map(|v| pose.apply(*v)).collect();
                let lowest = posed.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
                for index in topology.cells[cell] {
                    let y = posed[*index as usize].y;
                    assert!(
                        (y - lowest).abs() < 1e-4,
                        "{polytope:?} cell {cell} vertex {index} sits {} above the \
                         lowest point, so the toy is dropped on a corner",
                        y - lowest
                    );
                }
            }
        }
    }

    #[test]
    fn rotor_onto_turns_one_unit_vector_onto_the_other() {
        let cases = [
            (Vec4::X, Vec4::Y),
            (Vec4::W, -Vec4::Y),
            (Vec4::Y, Vec4::Y),
            (Vec4::Y, -Vec4::Y),
            (Vec4::new(0.5, 0.5, 0.5, 0.5), -Vec4::Y),
            (
                Vec4::new(0.5, -0.5, 0.5, -0.5),
                Vec4::new(1.0, 0.0, 0.0, 0.0),
            ),
        ];
        for (from, to) in cases {
            let turned = rotor_onto(from, to).apply(from);
            assert!(
                (turned - to).length() < 1e-5,
                "{from} turned to {turned}, not {to}"
            );
        }
    }

    #[test]
    fn a_held_body_tracks_the_cursor_instead_of_staying_put() {
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0), "the grab missed toy 0");
        let start = toybox.position(0);
        let mut cursor = start.truncate();
        for _ in 0..30 {
            cursor += Vec3::new(0.02, 0.02, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        let held = toybox.position(0);
        assert!(
            (held.truncate() - cursor).length() < 0.15,
            "the body sat {} from the cursor it was dragged by",
            (held.truncate() - cursor).length()
        );
        assert!(
            (held - start).length() > 0.3,
            "the body never left the place it was grabbed at"
        );
    }

    #[test]
    fn the_release_reads_recent_cursor_speed_and_not_the_total_drag() {
        // One long drag, released twice: once in motion, once after the cursor
        // has stood still for longer than the release window.
        let throw = |stall: usize| {
            let mut toybox = settled();
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..30 {
                cursor += Vec3::new(0.05, 0.0, 0.0);
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            for _ in 0..stall {
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            toybox.release();
            toybox.velocity(0)
        };
        let flicked = throw(0);
        let stalled = throw(12);
        assert!(
            flicked.x > 2.0,
            "a 3 u/s drag released in motion threw at {flicked}"
        );
        assert!(
            stalled.length() < 0.1 * flicked.length(),
            "a drag that stopped before the release still threw at {stalled}: the \
             release is reading the total drag, not the recent motion"
        );
    }

    #[test]
    fn a_grab_away_from_the_centre_of_mass_throws_with_spin() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        let handle = centre + Vec3::new(0.0, 0.22, 0.0);
        assert!(
            toybox.press(&ray_through(handle), FORWARD),
            "the grab missed"
        );
        let id = toybox.toys[0].body;
        let lever_local = toybox.grab.as_ref().expect("held").lever_local;
        assert!(
            lever_local.length() > 0.05,
            "the grab stored a handle at the centre of mass"
        );

        let mut cursor = handle;
        for _ in 0..20 {
            cursor += Vec3::new(0.03, 0.0, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        let rotation = toybox.world.bodies[id].orientation.rotation;
        let inertia = toybox.world.bodies[id].inertia;
        let before = toybox.angular_velocity(0);
        toybox.release();

        let after = toybox.angular_velocity(0);
        assert!(
            after.magnitude() > 0.5,
            "an off-centre grab left the body spinning at {}",
            after.magnitude()
        );
        // The change in angular velocity is `I⁻¹(r ∧ J)` about the point that was
        // grabbed. An impulse through the centre of mass has no lever arm and
        // would leave the spin exactly where it was.
        let lever = rotation.apply(lever_local);
        let expected = Bivector4::wedge(lever, toybox.velocity(0) * BODY_MASS) * (1.0 / inertia);
        let got = after + before * -1.0;
        assert!(
            (got + expected * -1.0).magnitude() < 1e-3 * expected.magnitude().max(1.0),
            "the release torqued by {got:?}, not the grabbed point's {expected:?}"
        );
    }

    #[test]
    fn only_the_modifier_puts_w_into_a_release() {
        let throw = |axis: GrabAxis| {
            let mut toybox = settled();
            let parked = toybox.position(0).w;
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..24 {
                cursor += Vec3::new(0.0, 0.04, 0.0);
                drag_frame(&mut toybox, cursor, axis);
            }
            toybox.release();
            (parked, toybox.position(0), toybox.velocity(0))
        };
        let (parked, position, velocity) = throw(GrabAxis::Slice);
        assert_eq!(velocity.w, 0.0, "an unmodified grab threw off the slice");
        assert_eq!(
            position.w, parked,
            "an unmodified grab carried the body off the slice"
        );

        let (parked, position, velocity) = throw(GrabAxis::Through);
        assert!(
            velocity.w > 0.5,
            "the modifier released at w velocity {}",
            velocity.w
        );
        assert!(
            position.w - parked > 0.5,
            "the modifier never carried the body off the slice (w {})",
            position.w - parked
        );
        assert!(
            velocity.y.abs() < 1e-5,
            "the modifier kept the screen rise as well as trading it for w"
        );
    }

    #[test]
    fn a_release_never_outruns_the_resolvable_step() {
        for reach in [0.05_f32, 0.5, 4.0, 40.0] {
            let mut toybox = settled();
            assert!(grab_centre(&mut toybox, 0));
            let mut cursor = toybox.position(0).truncate();
            for _ in 0..12 {
                cursor += Vec3::new(reach, 0.0, 0.0);
                drag_frame(&mut toybox, cursor, GrabAxis::Slice);
            }
            toybox.release();
            let travel = toybox.fastest_step_travel();
            assert!(
                travel <= TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL + 1e-5,
                "a {reach} u/frame drag released at {travel} of travel per step"
            );
        }
    }

    #[test]
    fn a_thrown_toy_pushes_a_neighbour_further_off_the_slice_than_the_floor_ever_does() {
        let mut toybox = settled();
        assert!(peak_w(&toybox) < SETTLED_W_BAND);
        assert!(grab_centre(&mut toybox, 0));
        let target = toybox.position(1).truncate();
        let mut cursor = toybox.position(0).truncate();
        let step = (target - cursor) / 12.0;
        for _ in 0..12 {
            cursor += step;
            drag_frame(&mut toybox, cursor, GrabAxis::Slice);
        }
        toybox.release();
        let mut worst = 0.0_f32;
        for _ in 0..180 {
            toybox.tick();
            worst = worst.max(peak_w(&toybox));
        }
        assert!(
            worst > SETTLED_W_BAND,
            "hull against hull left every body inside the band the floor alone \
             keeps them in (best |w| = {worst})"
        );
    }

    #[test]
    fn the_w_readout_follows_the_live_body() {
        let mut toybox = settled();
        assert!(grab_centre(&mut toybox, 0));
        let mut cursor = toybox.position(0).truncate();
        for _ in 0..20 {
            cursor += Vec3::new(0.0, 0.03, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Through);
        }
        let reported: Vec<(usize, f32)> = toybox.w_offsets().collect();
        assert_eq!(reported.len(), toybox.toys().len());
        for (index, w) in reported {
            assert_eq!(w, toybox.position(index).w - W_SLICE);
        }
        assert!(
            toybox.w_offsets().any(|(_, w)| w.abs() > W_LABEL_MIN),
            "the drag through w produced nothing to read out"
        );
    }

    #[test]
    fn a_cap_fades_as_it_shrinks_and_reaches_zero_before_it_disappears() {
        assert_eq!(section_alpha(0.0), 0.0);
        assert_eq!(section_alpha(FADE_EXTENT), 1.0);
        assert_eq!(section_alpha(10.0 * FADE_EXTENT), 1.0);
        let mut last = 0.0;
        for step in 0..=16 {
            let alpha = section_alpha(FADE_EXTENT * step as f32 / 16.0);
            assert!(alpha >= last, "the fade is not monotone in cap width");
            last = alpha;
        }
        assert_eq!(last, 1.0);
    }

    #[test]
    fn a_body_leaving_the_slice_fades_out_and_moves_to_the_depth_read_pass() {
        let mut toybox = settled();
        let (mut opaque, mut faded) = (TriangleMesh::default(), TriangleMesh::default());
        toybox.build_frame_meshes(&mut opaque, &mut faded);
        let solid = opaque.vertices.len();
        assert!(solid > 0, "nothing was drawn in the slice");
        assert!(faded.vertices.is_empty(), "a settled pile drew a faded cap");
        for color in &opaque.colors {
            assert_eq!(color[3], 1.0, "the opaque pass carries a translucent cap");
        }

        assert!(grab_centre(&mut toybox, 0));
        let mut cursor = toybox.position(0).truncate();
        let mut alphas = Vec::new();
        for _ in 0..24 {
            cursor += Vec3::new(0.0, 0.02, 0.0);
            drag_frame(&mut toybox, cursor, GrabAxis::Through);
            alphas.push(toybox.cap_stats(0).0);
        }
        assert_eq!(alphas[0], 1.0, "the body started already faded");
        assert!(
            alphas.windows(2).all(|w| w[1] <= w[0] + 1e-6),
            "the fade did not fall as the cap shrank: {alphas:?}"
        );
        assert!(
            *alphas.last().expect("non-empty") < 0.5,
            "the cap was still at alpha {} when it left the slice",
            alphas.last().expect("non-empty")
        );

        toybox.build_frame_meshes(&mut opaque, &mut faded);
        assert!(
            opaque.vertices.len() < solid,
            "the drifting body is still drawn in the opaque pass"
        );
        let expected: usize = (0..toybox.toys().len())
            .map(|toy| {
                let (alpha, vertices) = toybox.cap_stats(toy);
                if alpha >= 1.0 {
                    vertices
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            opaque.vertices.len(),
            expected,
            "a cap landed in the wrong pass for its alpha"
        );
    }

    #[test]
    fn a_pick_reads_the_cross_section_and_not_the_bounding_ball() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        assert!(
            toybox.pick(&ray_through(centre)).is_some(),
            "a ray at the centre missed the body"
        );

        // Inside the bounding ball and outside every cap: the ball test this
        // replaces would have returned a hit here.
        let mut off_hull = None;
        for step in 1..40 {
            let radius = BODY_SIZE * step as f32 / 40.0;
            let probe = centre + Vec3::new(radius, radius, 0.0);
            if (probe - centre).length() < BODY_SIZE && toybox.pick(&ray_through(probe)).is_none() {
                off_hull = Some(probe);
                break;
            }
        }
        let probe = off_hull.expect(
            "no ray inside the bounding ball missed the cap, so the pin has nothing to catch",
        );
        assert!((probe - centre).length() < BODY_SIZE);
        assert_eq!(toybox.pick(&ray_through(probe)), None);
    }

    #[test]
    fn a_pick_takes_the_nearest_cap_the_ray_enters() {
        let mut toybox = settled();
        let ray = ray_through(toybox.position(0).truncate());
        assert_eq!(toybox.pick(&ray).map(|(toy, _)| toy), Some(0));

        // Park a second toy between the eye and the first, on the same ray.
        let blocker = toybox.toys[1].body;
        let far = toybox.position(0);
        toybox.world.bodies[blocker].position =
            Vec4::from((ray.origin + ray.direction * 2.0, far.w));
        assert_eq!(
            toybox.pick(&ray).map(|(toy, _)| toy),
            Some(1),
            "the pick reached past the nearer body"
        );
    }

    #[test]
    fn a_ray_that_reaches_no_body_grabs_nothing() {
        let mut toybox = settled();
        let sky = Ray {
            origin: EYE,
            direction: Vec3::new(0.0, 1.0, 0.0),
        };
        assert_eq!(toybox.pick(&sky), None);
        assert!(!toybox.press(&sky, FORWARD));
        assert!(toybox.grab.is_none());
        toybox.release();
        for toy in 0..toybox.toys().len() {
            assert_eq!(
                toybox.velocity(toy),
                Vec4::ZERO,
                "a missed press threw {toy}"
            );
        }
    }

    #[test]
    fn the_modifier_trades_screen_rise_for_w_and_leaves_the_rest_alone() {
        let start = Vec4::new(1.0, 2.0, 3.0, 0.25);
        let delta = Vec3::new(0.4, -0.7, 0.1);
        assert_eq!(
            advance_target(start, delta, GrabAxis::Slice),
            start + Vec4::new(0.4, -0.7, 0.1, 0.0)
        );
        assert_eq!(
            advance_target(start, delta, GrabAxis::Through),
            start + Vec4::new(0.4, 0.0, 0.1, -0.7 * W_PER_RISE)
        );
        // The target integrates deltas, so the modifier can be pressed and
        // released mid-drag without the body jumping.
        for axis in [GrabAxis::Slice, GrabAxis::Through] {
            assert_eq!(
                advance_target(advance_target(start, delta, axis), -delta, axis),
                start
            );
        }
    }

    #[test]
    fn the_grab_plane_holds_the_pick_depth_across_the_view() {
        let depth = 6.0;
        for ndc in [(0.0_f32, 0.0_f32), (0.7, 0.4), (-0.9, -0.6)] {
            let direction = (FORWARD + Vec3::X * ndc.0 * 0.8 + Vec3::Y * ndc.1 * 0.5).normalize();
            let ray = Ray {
                origin: EYE,
                direction,
            };
            let point = plane_point(&ray, FORWARD, depth).expect("a camera ray meets the plane");
            assert!(
                ((point - EYE).dot(FORWARD) - depth).abs() < 1e-4,
                "ndc {ndc:?} landed at depth {}",
                (point - EYE).dot(FORWARD)
            );
        }
        let parallel = Ray {
            origin: EYE,
            direction: Vec3::X,
        };
        assert_eq!(plane_point(&parallel, FORWARD, depth), None);
    }

    #[test]
    fn the_two_release_ceilings_spend_exactly_the_usable_step() {
        let spent = (MAX_RELEASE_SPEED + MAX_ANGULAR_SPEED * BODY_SIZE) * MIN_SOLVER_DT;
        let usable = TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL;
        assert!(
            (spent - usable).abs() < 1e-6,
            "the ceilings spend {spent} of the step budget, not the {usable} they \
             are derived from"
        );
    }

    #[test]
    fn every_toy_collides_as_its_own_hull_with_the_exact_moment() {
        let toybox = scene();
        for (index, toy) in toybox.toys().iter().enumerate() {
            let body = &toybox.world.bodies[toy.body];
            let loam_physics::Collider::ConvexPolytope4D { vertices } = &body.collider else {
                panic!("toy {index} collides as {:?}", body.collider.kind());
            };
            assert_eq!(vertices.len(), toy.polytope.topology().vertices.len());
            let exact = regular_polytope4_inertia(toy.polytope, BODY_MASS, BODY_SIZE)
                .expect("every toy shape has a derived moment");
            assert_eq!(body.inertia, exact, "toy {index} kept the bounding ball");
        }
    }

    #[test]
    fn the_grab_handle_is_stored_in_the_body_frame() {
        let mut toybox = settled();
        let centre = toybox.position(0).truncate();
        assert!(toybox.press(&ray_through(centre + Vec3::new(0.0, 0.2, 0.0)), FORWARD));
        let lever = toybox.grab.as_ref().expect("held").lever_local;
        assert!(lever.length() > 0.05, "an off-centre grab stored no lever");

        let id = toybox.toys[0].body;
        let before = toybox.world.bodies[id].orientation.rotation.apply(lever);
        let spun = (Plane4::Xy.unit_bivector() * 0.9).exp().normalize();
        toybox.world.bodies[id].orientation.rotation = spun;
        let after = toybox.world.bodies[id].orientation.rotation.apply(lever);
        assert!(
            (before - after).length() > 0.05,
            "the handle stayed put in world space while the body turned under it"
        );
        assert!(
            (before.length() - after.length()).abs() < 1e-5,
            "the handle changed length under a rotation"
        );
    }

    const TRANSLATE_TOL: f32 = 1e-5;

    const FIXTURE_DT: f32 = 1.0 / 60.0;

    // Past `PENETRATION_SLOP`, so the narrowphase reports rather than grazes.
    const FIXTURE_OVERLAP: f32 = 0.2;

    // Far past the sum of two bounding radii, so the broadphase cannot couple
    // two pairs into one island.
    const FIXTURE_GROUP_GAP: f32 = 20.0;

    // Fast enough that the Coulomb clamp `|jt| <= mu*jn` binds on the first
    // step, so the tangent accumulator is the cap rather than a rounding
    // artefact.
    const FIXTURE_SLIDE_SPEED: f32 = 3.0;

    // Balls, not the scene's hulls, and no floor: one contact point per
    // manifold is what makes the per-layer segment counts below exact.
    fn overlapping_pairs(pairs: usize) -> World<EuclideanR4> {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        for pair in 0..pairs {
            let base = Vec4::X * (FIXTURE_GROUP_GAP * pair as f32);
            world.push_body(sphere_body_r4(base, Vec4::ZERO, BODY_SIZE, BODY_MASS));
            world.push_body(sphere_body_r4(
                base + Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP),
                Vec4::ZERO,
                BODY_SIZE,
                BODY_MASS,
            ));
        }
        world.step(FIXTURE_DT);
        assert_eq!(
            world.manifolds.len(),
            pairs,
            "the fixture layout did not produce one manifold per pair"
        );
        world
    }

    fn every_layer() -> PhysicsOverlay {
        PhysicsOverlay {
            contacts: true,
            normals: true,
            impulses: true,
            islands: true,
            ..PhysicsOverlay::default()
        }
    }

    fn only(layer: fn(&mut PhysicsOverlay)) -> PhysicsOverlay {
        let mut overlay = PhysicsOverlay::default();
        layer(&mut overlay);
        overlay
    }

    fn overlay_mesh(world: &World<EuclideanR4>, overlay: &PhysicsOverlay) -> LineMesh<3> {
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(world, BODY_SIZE, overlay, &mut mesh);
        assert_eq!(mesh.colors.len(), mesh.segments.len());
        assert_eq!(mesh.widths.len(), mesh.segments.len());
        mesh
    }

    fn contact_count(world: &World<EuclideanR4>) -> usize {
        world.manifolds.values().map(|m| m.points.len()).sum()
    }

    #[test]
    fn a_landing_fills_the_manifolds_the_overlay_draws_from() {
        let toybox = settled();
        assert!(
            !toybox.world.manifolds.is_empty(),
            "the pile settled with no manifold, so every layer would draw nothing"
        );
        let peak = (toybox.world.manifolds.values())
            .flat_map(|m| m.points.iter())
            .fold(0.0f32, |m, cp| m.max(cp.normal_impulse));
        assert!(peak > 0.0, "a resting pile accumulated no normal impulse");

        let mut mesh = LineMesh::<3>::default();
        for (name, overlay) in [
            ("contacts", only(|o| o.contacts = true)),
            ("normals", only(|o| o.normals = true)),
            ("impulses", only(|o| o.impulses = true)),
            ("islands", only(|o| o.islands = true)),
        ] {
            toybox.build_overlay_mesh(&overlay, &mut mesh);
            assert!(
                !mesh.segments.is_empty(),
                "the {name} layer drew nothing over a settled pile"
            );
        }
    }

    #[test]
    fn the_islands_layer_draws_no_coupling_to_the_static_floor() {
        let toybox = settled();
        let islands = toybox.world.islands();
        let floor_pairs: usize = (islands.iter())
            .flat_map(|i| i.constraints.iter())
            .filter(|&&(a, b)| {
                toybox.world.bodies[a].inv_mass == 0.0 || toybox.world.bodies[b].inv_mass == 0.0
            })
            .count();
        assert!(
            floor_pairs > 0,
            "no toy is resting on the floor, so the skip is vacuous"
        );

        let mut mesh = LineMesh::<3>::default();
        toybox.build_overlay_mesh(&only(|o| o.islands = true), &mut mesh);
        let crosses: usize = islands.iter().map(|i| 3 * i.bodies.len()).sum();
        let couplings: usize = (islands.iter())
            .flat_map(|i| i.constraints.iter())
            .filter(|&&(a, b)| {
                toybox.world.bodies[a].inv_mass != 0.0 && toybox.world.bodies[b].inv_mass != 0.0
            })
            .count();
        assert_eq!(
            mesh.segments.len(),
            crosses + couplings,
            "the islands layer drew a bar for a pair the solver couples nothing through"
        );
    }

    #[test]
    fn every_contact_emits_one_normal_along_the_stored_direction() {
        let world = overlapping_pairs(2);
        let mesh = overlay_mesh(&world, &only(|o| o.normals = true));

        let contacts = contact_count(&world);
        assert!(contacts > 0, "fixture produced no contacts");
        assert_eq!(
            mesh.segments.len(),
            contacts,
            "the normals layer emitted {} segments for {contacts} contacts",
            mesh.segments.len()
        );

        let mut segments = mesh.segments.iter();
        for (&(a, b), manifold) in world.manifolds.iter() {
            let separation = (world.bodies[b].position - world.bodies[a].position).truncate();
            for cp in &manifold.points {
                let &(from, to) = segments.next().expect("one segment per contact");
                let point = cp.world_point.truncate();
                assert!(
                    (Vec3::from_array(from) - point).length() < TRANSLATE_TOL,
                    "normal starts at {from:?}, not at the contact point {point:?}"
                );
                let drawn = Vec3::from_array(to) - Vec3::from_array(from);
                let expected = cp.normal.truncate() * (NORMAL_LEN_FRACTION * BODY_SIZE);
                assert!(
                    (drawn - expected).length() < TRANSLATE_TOL,
                    "normal drawn as {drawn:?}, not {expected:?}"
                );
                assert!(
                    drawn.dot(separation) > 0.0,
                    "normal runs from A toward B; the layer would misreport a flipped normal"
                );
            }
        }
    }

    #[test]
    fn impulse_bar_length_tracks_the_accumulated_impulse() {
        let world = overlapping_pairs(1);
        let contacts = contact_count(&world);
        let solved: f32 = (world.manifolds.values())
            .flat_map(|m| m.points.iter())
            .map(|cp| cp.normal_impulse)
            .sum();
        assert!(
            solved > 0.0,
            "fixture accumulated no normal impulse, so the length pin is vacuous"
        );

        let base = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&world, &base);
        assert_eq!(
            mesh.segments.len(),
            2 * contacts,
            "the impulses layer emits a normal and a tangent bar per contact"
        );

        let doubled = overlay_mesh(
            &world,
            &PhysicsOverlay {
                impulse_scale: base.impulse_scale * 2.0,
                ..base
            },
        );
        for (i, (&(from, to), &(from2, to2))) in
            mesh.segments.iter().zip(&doubled.segments).enumerate()
        {
            let short = Vec3::from_array(to) - Vec3::from_array(from);
            let long = Vec3::from_array(to2) - Vec3::from_array(from2);
            assert!(
                (long - short * 2.0).length() < TRANSLATE_TOL,
                "bar {i} did not scale with impulse_scale: {short:?} then {long:?}"
            );
        }

        let points = world.manifolds.values().flat_map(|m| m.points.iter());
        for (cp, chunk) in points.zip(mesh.segments.chunks_exact(2)) {
            let bar = Vec3::from_array(chunk[0].1) - Vec3::from_array(chunk[0].0);
            let expected = cp.normal.truncate() * (-cp.normal_impulse * base.impulse_scale);
            assert!(
                (bar - expected).length() < TRANSLATE_TOL,
                "normal-impulse bar drawn as {bar:?}, not {expected:?}"
            );
        }
    }

    #[test]
    fn a_sliding_pair_draws_its_friction_bar_against_the_slide() {
        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_body(sphere_body_r4(
            Vec4::ZERO,
            Vec4::Y * FIXTURE_SLIDE_SPEED,
            BODY_SIZE,
            BODY_MASS,
        ));
        world.push_body(sphere_body_r4(
            Vec4::X * (2.0 * BODY_SIZE - FIXTURE_OVERLAP),
            Vec4::ZERO,
            BODY_SIZE,
            BODY_MASS,
        ));
        world.step(FIXTURE_DT);

        let overlay = only(|o| o.impulses = true);
        let mesh = overlay_mesh(&world, &overlay);
        let contacts = contact_count(&world);
        assert_eq!(mesh.segments.len(), 2 * contacts);

        let mut chunks = mesh.segments.chunks_exact(2);
        let mut braked = 0;
        for (&(a, b), manifold) in world.manifolds.iter() {
            let lead = (world.bodies[a].velocity - world.bodies[b].velocity).truncate();
            for cp in &manifold.points {
                let chunk = chunks.next().expect("two bars per contact");
                let bar = Vec3::from_array(chunk[1].1) - Vec3::from_array(chunk[1].0);
                let expected = cp.tangent_impulse * overlay.impulse_scale;
                assert!(
                    (bar.length() - expected).abs() < TRANSLATE_TOL,
                    "friction bar runs {} world units for a {} accumulator",
                    bar.length(),
                    cp.tangent_impulse
                );
                if cp.tangent_impulse > 0.0 {
                    braked += 1;
                    assert!(
                        bar.dot(lead) > 0.0,
                        "friction bar {bar:?} runs with the slide {lead:?} it should brake"
                    );
                }
            }
        }
        assert!(
            braked > 0,
            "no contact accumulated friction, so the sign pin is vacuous"
        );
    }

    #[test]
    fn a_full_speed_flick_draws_its_bar_at_a_third_of_a_body_radius() {
        let mut toybox = settled();
        let (from, to) = (toybox.position(0), toybox.position(1));
        let thrown = toybox.toys[0].body;
        toybox.world.bodies[thrown].velocity = (to - from).normalize() * MAX_CARRY_SPEED;

        let mut peak = 0.0f32;
        for _ in 0..FLICK_TICKS {
            toybox.tick();
            for manifold in toybox.world.manifolds.values() {
                for cp in &manifold.points {
                    peak = peak.max(cp.normal_impulse);
                }
            }
        }

        assert!(
            (5.0..6.0).contains(&peak),
            "peak normal impulse {peak}: the prose at DEFAULT_IMPULSE_SCALE is now stale, so recompute the bar length before touching this bound"
        );
        let bar = peak * DEFAULT_IMPULSE_SCALE;
        assert!(
            (0.30..0.42).contains(&(bar / BODY_SIZE)),
            "bar runs {bar} world units, {} of a body radius",
            bar / BODY_SIZE
        );
    }

    #[test]
    fn one_island_marks_its_bodies_in_one_colour() {
        let world = overlapping_pairs(2);
        let islands = world.islands();
        assert_eq!(islands.len(), 2, "fixture did not split into two islands");

        let mesh = overlay_mesh(&world, &only(|o| o.islands = true));
        let expected: usize = islands
            .iter()
            .map(|i| 3 * i.bodies.len() + i.constraints.len())
            .sum();
        assert_eq!(mesh.segments.len(), expected);

        let mut per_island: Vec<[u32; 4]> = Vec::new();
        for island in &islands {
            let mut colors = std::collections::BTreeSet::new();
            for &id in &island.bodies {
                let centre = world.bodies[id].position.truncate();
                let mut arms = 0;
                for (&(from, to), &(color, _)) in mesh.segments.iter().zip(&mesh.colors) {
                    let mid = (Vec3::from_array(from) + Vec3::from_array(to)) * 0.5;
                    if (mid - centre).length() < TRANSLATE_TOL
                        && Vec3::from_array(from) != Vec3::from_array(to)
                    {
                        arms += 1;
                        colors.insert(color.map(f32::to_bits));
                    }
                }
                assert_eq!(arms, 3, "body {id:?} is not marked by a three-arm cross");
            }
            assert_eq!(
                colors.len(),
                1,
                "island {:?} marked its bodies in {} colours",
                island.id,
                colors.len()
            );
            let color = *colors.iter().next().expect("one colour");
            assert!(
                !per_island.contains(&color),
                "two islands share a colour, so the partition cannot be read off the overlay"
            );
            per_island.push(color);
        }
    }

    #[test]
    fn a_world_at_rest_draws_no_physics_overlay() {
        let toybox = scene();
        assert!(toybox.world.bodies.iter().all(|b| b.velocity == Vec4::ZERO));
        assert!(toybox.world.manifolds.is_empty());
        let mut mesh = LineMesh::<3>::default();
        toybox.build_overlay_mesh(&every_layer(), &mut mesh);
        assert!(
            mesh.segments.is_empty(),
            "a resting world emitted {} segments",
            mesh.segments.len()
        );
    }

    #[test]
    fn each_overlay_layer_draws_only_its_own_geometry() {
        let world = overlapping_pairs(2);
        let contacts = contact_count(&world);

        let layers: [(&str, PhysicsOverlay, usize); 4] = [
            ("contacts", only(|o| o.contacts = true), 3 * contacts),
            ("normals", only(|o| o.normals = true), contacts),
            ("impulses", only(|o| o.impulses = true), 2 * contacts),
            ("islands", only(|o| o.islands = true), {
                let islands = world.islands();
                islands
                    .iter()
                    .map(|i| 3 * i.bodies.len() + i.constraints.len())
                    .sum()
            }),
        ];

        let all = overlay_mesh(&world, &every_layer());
        let mut total = 0;
        for (name, overlay, expected) in layers {
            let mesh = overlay_mesh(&world, &overlay);
            assert_eq!(
                mesh.segments.len(),
                expected,
                "the {name} layer alone emitted {} segments, expected {expected}",
                mesh.segments.len()
            );
            for segment in &mesh.segments {
                assert!(
                    all.segments.contains(segment),
                    "the {name} layer's {segment:?} is missing when every layer is on"
                );
            }
            total += expected;
        }
        assert_eq!(
            all.segments.len(),
            total,
            "the all-layers build is not exactly the four layers"
        );
    }

    #[test]
    fn a_hidden_physics_overlay_reaches_the_allocator_zero_times() {
        let world = overlapping_pairs(2);
        let mut mesh = LineMesh::<3>::default();

        build_physics_overlay_mesh(&world, BODY_SIZE, &every_layer(), &mut mesh);
        assert!(!mesh.segments.is_empty(), "the fixture emitted nothing");
        let warm = mesh.segments.capacity();

        let hidden = PhysicsOverlay::default();
        assert!(!hidden.any_layer(), "the overlay ships with a layer on");
        let bytes = alloc_probe::bytes_allocated_by(|| {
            build_physics_overlay_mesh(&world, BODY_SIZE, &hidden, &mut mesh)
        });
        assert_eq!(
            bytes, 0,
            "a hidden overlay asked the allocator for {bytes} bytes"
        );
        assert!(mesh.segments.is_empty());
        assert_eq!(
            mesh.segments.capacity(),
            warm,
            "the hidden path dropped the buffer it will need again"
        );
    }

    #[test]
    fn the_contact_overlay_layers_reach_the_allocator_zero_times() {
        let world = overlapping_pairs(2);
        let overlay = PhysicsOverlay {
            contacts: true,
            normals: true,
            impulses: true,
            ..PhysicsOverlay::default()
        };
        let mut mesh = LineMesh::<3>::default();
        build_physics_overlay_mesh(&world, BODY_SIZE, &overlay, &mut mesh);
        assert!(!mesh.segments.is_empty(), "the fixture emitted nothing");

        let warm = alloc_probe::bytes_allocated_by(|| {
            build_physics_overlay_mesh(&world, BODY_SIZE, &overlay, &mut mesh)
        });
        assert_eq!(
            warm, 0,
            "a warm contact overlay asked the allocator for {warm} bytes"
        );
    }
}
