//! The floor's normal is pure `y`, so its NORMAL impulse carries no `w`. Its
//! friction does: the tangent space of a contact in R⁴ is three-dimensional and
//! `w` is in it, so the impulse that arrests a landing hull also slides it off
//! the slice. The scene answers that at the spawn, with a drop shallow enough
//! and a pose flat enough to keep the landing inside [`SETTLED_W_BAND`], rather
//! than by damping `w`: the `w` a body-body contact imparts is the point.

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_camera::Ray;
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector, Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4, WPlane};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::{BodyId, Gravity, World};
use loam_render::{
    DepthBuffer, DepthMode, Ground, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode, Viewport,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::TriangleMesh;

use crate::physics::ndc_from_pixels;

const TICK_HZ: u32 = 60;

// A hull settling on a half-space needs 240 Hz, not 120: `polytope_halfspace_r4`
// reports one deepest vertex per step where the resting manifold holds four, so
// halving the rate halves the constraints the solver has by the time a body
// rocks over. Four sub-steps reach that rate without moving the host clock.
const SUBSTEPS_PER_TICK: usize = 4;

// Fixed, never derived from wall time.
const SOLVER_DT: f32 = 1.0 / (TICK_HZ as f32 * SUBSTEPS_PER_TICK as f32);

const GRAVITY: f32 = -9.8;

// Read by the physics half-space and by the background's analytic ground, so
// the drawn floor cannot drift from the one the bodies land on.
const FLOOR_Y: f32 = 0.0;

// The slice every cross-section is cut at. A body's distance from it is what
// [`section_alpha`] fades on.
const W_SLICE: f32 = 0.0;

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
const MAX_RELEASE_SPEED: f32 = 0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / SOLVER_DT;

const MAX_ANGULAR_SPEED: f32 =
    0.5 * TRAVEL_MARGIN * RESOLVABLE_STEP_TRAVEL / (BODY_SIZE * SOLVER_DT);

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
    /// Where the drive is pulling the body.
    target: Vec4,
    /// Last cursor point on the grab plane; the target advances by its delta.
    plane_point: Vec3,
    trail: GrabTrail,
}

pub(crate) struct Toybox {
    world: World<EuclideanR4>,
    toys: Vec<ToyBody>,
    grab: Option<Grab>,
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
            tick: 0,
            local_vertices: Vec::new(),
            section_scratch: SectionScratch::default(),
            cap: TriangleMesh::<3>::default(),
        }
    }

    pub(crate) fn held(&self) -> Option<usize> {
        self.grab.as_ref().map(|grab| grab.toy)
    }

    pub(crate) fn position(&self, toy: usize) -> Vec4 {
        self.world.bodies[self.toys[toy].body].position
    }

    /// Offset from [`W_SLICE`], per toy, read from the live body.
    pub(crate) fn w_offsets(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        (self.toys.iter().enumerate())
            .map(|(index, toy)| (index, self.world.bodies[toy.body].position.w - W_SLICE))
    }

    pub(crate) fn tick(&mut self) {
        for _ in 0..SUBSTEPS_PER_TICK {
            self.world.step(SOLVER_DT);
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
        let grabbed = Vec4::new(hit.x, hit.y, hit.z, W_SLICE);
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
            grab.target = advance_target(grab.target, point - grab.plane_point, axis);
            grab.plane_point = point;
        }
        grab.trail.push(grab.target, dt);
        let body = &mut self.world.bodies[self.toys[grab.toy].body];
        body.velocity = clamp_length(
            (grab.target - body.position) * GRAB_STIFFNESS,
            MAX_RELEASE_SPEED,
        );
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
    pub(crate) fn pick(&mut self, ray: &Ray) -> Option<(usize, Vec3)> {
        let mut nearest: Option<(usize, f32)> = None;
        let mut cap = std::mem::take(&mut self.cap);
        let mut local = std::mem::take(&mut self.local_vertices);
        for toy in 0..self.toys.len() {
            append_cap(
                &self.world,
                &self.toys[toy],
                &mut local,
                &mut self.section_scratch,
                &mut cap,
            );
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
        (self.world.bodies.iter())
            .map(|b| (b.velocity.length() + b.angular_velocity.magnitude() * BODY_SIZE) * SOLVER_DT)
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

/// Rebuilds `cap` as one toy's cross-section at [`W_SLICE`], in world R³, and
/// returns the cap's half-width about its own centroid.
fn append_cap(
    world: &World<EuclideanR4>,
    toy: &ToyBody,
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
        WPlane::new(W_SLICE),
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

const GRASS_DARK: [f32; 3] = [0.145, 0.205, 0.130];
const GRASS_LIGHT: [f32; 3] = [0.170, 0.235, 0.150];

// 24-bit depth cracks the thin, densely stacked caps of a tumbling 24-cell.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const BOOT_ORBIT_DISTANCE: f32 = 7.0;
const BOOT_ORBIT_PITCH: f32 = -0.16;
const BOOT_TARGET_HEIGHT: f32 = 0.7;

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
    console: Console<()>,
    caps: TriangleRasterNode,
    faded_caps: TriangleRasterNode,
    sky_ground: SkyGroundNode,
    depth: Option<DepthBuffer>,
    opaque_mesh: TriangleMesh<3>,
    faded_mesh: TriangleMesh<3>,
    left_was_down: bool,
    paused: bool,
}

impl ToyboxScene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let mut console = Console::<()>::new();
        loam_app::shell::register_command::<(), crate::shell::Playground>(&mut console);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 2.0, BOOT_ORBIT_DISTANCE);
        camera.near = 0.05;
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);
        orbit.target.y = BOOT_TARGET_HEIGHT;

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
            left_was_down: false,
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
                ui.label("drag a shape to carry it; let go to throw it");
                ui.label("hold Shift while dragging to pull it through w");
                ui.separator();
                for (index, w) in self.toybox.w_offsets() {
                    ui.monospace(format!("#{index}  w {w:+.3}"));
                }
                ui.separator();
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

    // A body off the slice draws a shrinking cap or none at all, so the offset
    // is the only thing left saying it is still there.
    fn w_readouts(&self, ctx: &egui::Context, frame: &FrameCtx<'_>) {
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
        self.console.dispatch(&cmd.name, &cmd.arg_refs(), &mut ());
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

        let held = self.toybox.held().is_some();
        if !self.paused {
            for _ in 0..ctx.n_ticks {
                self.toybox.tick();
            }
        }

        if !ctx.ui_capture.pointer {
            let mut input = ctx.input;
            input.left_mouse_down &= !held;
            input.buttons.left.down &= !held;
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
        if ctx.ui_capture.keyboard || state != ElementState::Pressed {
            return;
        }
        match code {
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::KeyR => self.respawn(self.seed),
            KeyCode::KeyN => self.respawn(self.seed.wrapping_add(1)),
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
        let depth = self.depth.as_ref().expect("ensure() guarantees Some");

        let view = self.camera.view();
        let aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        let view_mat = Mat4::look_to_rh(view.position, view.forward, view.up);
        let proj_mat = Mat4::perspective_rh(55.0_f32.to_radians(), aspect, 0.05, 200.0);
        let view_proj = proj_mat * view_mat;

        // This scene owns the frame's clear; both cap passes load.
        self.sky_ground.set_uniforms(
            &rd.queue,
            &SkyGroundUniforms::new(
                view_proj,
                Viewport::full([cfg.width, cfg.height]),
                Ground {
                    y: FLOOR_Y,
                    dark: GRASS_DARK,
                    light: GRASS_LIGHT,
                    visible: true,
                },
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
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - toybox")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_math::Plane4;
    use loam_physics::manifold::PENETRATION_SLOP;

    // The drop takes 12 ticks and the rest latch 30 more; the rest is margin.
    const SETTLE_TICKS: usize = 120;

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
        assert_eq!(toybox.held(), None);
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
        let spent = (MAX_RELEASE_SPEED + MAX_ANGULAR_SPEED * BODY_SIZE) * SOLVER_DT;
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
}
