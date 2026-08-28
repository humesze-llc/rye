//! Falling letters take [`GlyphSolid::rigid_hull_4d`], the one convex prism per
//! letter, and not [`GlyphSolid::colliders_4d`]'s faithful box cover: a moving
//! faithful cover would need a compound collider `loam-physics` does not have.

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4, WPlane};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::{BodyId, Gravity, World};
use loam_render::{
    DepthBuffer, DepthMode, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode, Viewport,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::{Shape, TriangleMesh, Visualizable};
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};
use loam_time::director::{BodyTrack, Director, Drive, Ease, Timeline, Track};

use crate::environment::{register_ground_command, Environment};

const WORD: &str = "LOAM";

const TICK_HZ: u32 = 60;

// `glyph-letter-bodies` measured that a letter hull dropped on a half-space
// settles at 240 Hz and does NOT at 120 Hz, where the same drops skid 0.45 to
// 0.61 em and end tipped on a corner: `polytope_halfspace_r4` reports one
// deepest vertex per step, a manifold holds four, and halving the rate halves
// the constraints the solver has accumulated by the time a letter rocks over.
// Four sub-steps of a 60 Hz tick is that rate, reached without moving the host
// clock.
const SUBSTEPS_PER_TICK: usize = 4;

// Fixed and never derived from wall time.
const SOLVER_DT: f32 = 1.0 / (TICK_HZ as f32 * SUBSTEPS_PER_TICK as f32);

const GRAVITY: f32 = -9.8;

const ASSEMBLE_TICKS: u32 = 90;

// The last letter must still finish inside [`ASSEMBLE_TICKS`], which
// `the_assembly_slides_every_letter_onto_its_mark_before_the_release` pins.
const LETTER_STAGGER_TICKS: u32 = 12;

const LETTER_SLIDE_TICKS: u32 = 36;

const ENTRY_SPAN: f32 = 3.2;

// Height of the lowest hull vertex above the floor at release, in em. The fall
// is short by design: `glyph-letter-bodies` sampled drop clearance from 0.01 to
// 0.17 em and found the landing is not monotone in height, because a level 4D
// landing has more tied deepest corners than the one contact per step the
// narrowphase reports. This value is inside the band that settles; widening it
// is a multi-contact narrowphase in `loam-physics`, not a change here.
const RELEASE_CLEARANCE: f32 = 0.05;

// Every letter is at rest well inside this, which
// `every_letter_is_at_rest_on_the_floor_before_the_rain_starts` pins at the
// last tick before the spawn.
const SETTLE_TICKS: u32 = 120;

const RAIN_START_TICK: u32 = ASSEMBLE_TICKS + SETTLE_TICKS;

pub(crate) const SEQUENCE_TICKS: u32 = RAIN_START_TICK + 360;

const RAIN_COUNT: usize = 14;

const RAIN_INTERVAL_TICKS: u32 = 7;
const RAIN_INTERVAL_JITTER: u32 = 4;

// Written as the difference of the two above so a jitter wider than the
// interval is a compile error rather than a wrap to a spawn tick four billion
// ticks away.
const RAIN_INTERVAL_MIN: u32 = RAIN_INTERVAL_TICKS - RAIN_INTERVAL_JITTER;

const RAIN_SIZE: f32 = 0.30;

const LETTER_MASS: f32 = 1.0;

const RAIN_MASS: f32 = 1.5;

// The lower bound clears the tops of the letters, so nothing is ever spawned
// inside one.
const RAIN_HEIGHT: (f32, f32) = (2.6, 3.6);

// Small: most of the impact speed is the fall, and a spawn speed large enough
// to matter would also be the one that tunnels.
const RAIN_ENTRY_SPEED: f32 = 1.0;

// A letter prism spans only the glyph slab in `w` (0.15 em), so a drop whose
// centre strays much further than its own circumradius would pass the letters
// by in the fourth dimension rather than hit them.
const RAIN_W_SPREAD: f32 = 0.10;

const RAIN_Z_SPREAD: f32 = 0.20;

// Per-plane ceiling on a drop's spawn tumble, rad/s. At [`RAIN_SIZE`] the
// fastest material point then travels `6 · 0.30 · SOLVER_DT = 0.0075` em per
// step, two orders inside the band the narrowphase resolves.
const RAIN_TUMBLE: f32 = 6.0;

const RESTITUTION: f32 = 0.0;

pub(crate) const DEFAULT_SEED: u64 = 0x10a3_5eed;

const RAIN_SHAPES: [Polytope4; 4] = [
    Polytope4::Cell24,
    Polytope4::Pentatope,
    Polytope4::Cell16,
    Polytope4::Tesseract,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Assemble,
    Fall,
    Rain,
}

pub(crate) struct HeroLetter {
    mesh: TriangleMesh<3>,
    hull: Vec<Vec4>,
    mark: Vec4,
    entry: Vec4,
    track: String,
    body: Option<BodyId>,
}

pub(crate) struct HeroDrop {
    body: BodyId,
    polytope: Polytope4,
    color: [f32; 3],
}

impl HeroDrop {
    pub(crate) fn polytope(&self) -> Polytope4 {
        self.polytope
    }

    pub(crate) fn color(&self) -> [f32; 3] {
        self.color
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct HeroPose {
    pub(crate) position: Vec4,
    pub(crate) rotor: Rotor4,
}

impl HeroPose {
    pub(crate) fn position_r3(&self) -> Vec3 {
        self.position.truncate()
    }
}

pub(crate) struct HeroSequence {
    world: World<EuclideanR4>,
    director: Director,
    letters: Vec<HeroLetter>,
    drops: Vec<HeroDrop>,
    // Advanced only by a spawn, so the draw a given drop gets does not depend
    // on how many ticks passed before it.
    rng: u64,
    tick: u32,
    next_spawn_tick: u32,
}

impl HeroSequence {
    pub(crate) fn new(font_bytes: &[u8], seed: u64) -> Result<Self> {
        let font = ab_glyph::FontRef::try_from_slice(font_bytes)?;
        let solids = layout_word(&font, WORD, &GlyphParams::default())?;

        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY, 0.0, 0.0))));
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, FLOOR_Y));
        world.bodies[floor].restitution = RESTITUTION;

        let letters: Vec<HeroLetter> = solids
            .iter()
            .filter(|solid| !solid.is_blank())
            .enumerate()
            .map(|(index, solid)| letter_from(solid, index))
            .collect::<Result<_>>()?;

        let director = Director::new(assembly_timeline(&letters))?;

        Ok(Self {
            world,
            director,
            letters,
            drops: Vec::with_capacity(RAIN_COUNT),
            // xorshift64* has a fixed point at zero, so a zero seed would
            // produce a constant stream rather than a degenerate one nobody
            // notices. Mixing in an odd constant keeps seed 0 usable.
            rng: seed ^ 0x9e37_79b9_7f4a_7c15,
            tick: 0,
            next_spawn_tick: RAIN_START_TICK,
        })
    }

    pub(crate) fn tick(&mut self) {
        if self.tick < ASSEMBLE_TICKS {
            self.director.advance();
        } else {
            if self.tick >= self.next_spawn_tick {
                self.spawn_drop();
            }
            for _ in 0..SUBSTEPS_PER_TICK {
                self.world.step(SOLVER_DT);
            }
        }
        self.tick += 1;
        // The handover closes the assemble phase rather than opening the fall
        // one, so `Self::phase` and the world's body count never disagree about
        // who owns a letter at a tick boundary.
        if self.tick == ASSEMBLE_TICKS {
            self.release_letters();
        }
    }

    pub(crate) fn tick_index(&self) -> u32 {
        self.tick
    }

    pub(crate) fn finished(&self) -> bool {
        self.tick >= SEQUENCE_TICKS
    }

    pub(crate) fn phase(&self) -> Phase {
        if self.tick < ASSEMBLE_TICKS {
            Phase::Assemble
        } else if self.tick < RAIN_START_TICK {
            Phase::Fall
        } else {
            Phase::Rain
        }
    }

    pub(crate) fn letters(&self) -> &[HeroLetter] {
        &self.letters
    }

    pub(crate) fn drops(&self) -> &[HeroDrop] {
        &self.drops
    }

    pub(crate) fn letter_pose(&self, index: usize) -> HeroPose {
        let letter = &self.letters[index];
        match letter.body {
            Some(id) => self.body_pose(id),
            None => HeroPose {
                position: match self.director.position(&letter.track) {
                    Drive::Directed(p) => p,
                    Drive::Host => letter.entry,
                },
                rotor: Rotor4::IDENTITY,
            },
        }
    }

    pub(crate) fn drop_pose(&self, index: usize) -> HeroPose {
        self.body_pose(self.drops[index].body)
    }

    fn body_pose(&self, id: BodyId) -> HeroPose {
        let body = &self.world.bodies[id];
        HeroPose {
            position: body.position,
            rotor: body.orientation.rotation,
        }
    }

    fn release_letters(&mut self) {
        for letter in &mut self.letters {
            debug_assert!(letter.body.is_none(), "released twice");
            let id = self.world.push_body(polytope_body_r4(
                letter.mark,
                Vec4::ZERO,
                letter.hull.clone(),
                LETTER_MASS,
            ));
            self.world.bodies[id].restitution = RESTITUTION;
            letter.body = Some(id);
        }
    }

    fn spawn_drop(&mut self) {
        if self.drops.len() >= RAIN_COUNT {
            return;
        }
        let index = self.drops.len();
        let polytope = RAIN_SHAPES[index % RAIN_SHAPES.len()];
        let span = word_span(&self.letters);
        // Rust evaluates argument lists left to right, so the draws below
        // consume the stream in written order and the seed pins each
        // coordinate to a particular position in it.
        let position = Vec4::new(
            lerp(span.0, span.1, unit(self.draw())),
            lerp(RAIN_HEIGHT.0, RAIN_HEIGHT.1, unit(self.draw())),
            RAIN_Z_SPREAD * signed_unit(self.draw()),
            RAIN_W_SPREAD * signed_unit(self.draw()),
        );
        let tumble = Bivector4::new(
            RAIN_TUMBLE * signed_unit(self.draw()),
            RAIN_TUMBLE * signed_unit(self.draw()),
            RAIN_TUMBLE * signed_unit(self.draw()),
            RAIN_TUMBLE * signed_unit(self.draw()),
            RAIN_TUMBLE * signed_unit(self.draw()),
            RAIN_TUMBLE * signed_unit(self.draw()),
        );
        let vertices: Vec<Vec4> = polytope
            .topology()
            .vertices
            .iter()
            .map(|v| RAIN_SIZE * *v)
            .collect();
        let id = self.world.push_body(polytope_body_r4(
            position,
            Vec4::new(0.0, -RAIN_ENTRY_SPEED, 0.0, 0.0),
            vertices,
            RAIN_MASS,
        ));
        let body = &mut self.world.bodies[id];
        body.restitution = RESTITUTION;
        body.angular_velocity = tumble;
        // The exact uniform-solid moment, not `polytope_body_r4`'s bounding
        // ball: every one of these symmetry groups acts irreducibly on R⁴, so
        // the scalar inertia slot is exact rather than an approximation.
        if let Some(inertia) = regular_polytope4_inertia(polytope, RAIN_MASS, RAIN_SIZE) {
            body.inertia = inertia;
        }
        self.drops.push(HeroDrop {
            body: id,
            polytope,
            color: drop_color(polytope),
        });
        let jitter = (self.draw() % (2 * RAIN_INTERVAL_JITTER as u64 + 1)) as u32;
        self.next_spawn_tick = self.tick + RAIN_INTERVAL_MIN + jitter;
    }

    // xorshift64* (Vigna 2016, *An experimental exploration of Marsaglia's
    // xorshift generators, scrambled*, §4).
    fn draw(&mut self) -> u64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

#[cfg(test)]
impl HeroSequence {
    // Backwards is a no-op: the solver has no inverse.
    fn run_to(&mut self, tick: u32) {
        while self.tick < tick {
            self.tick();
        }
    }

    fn fastest_body_speed(&self) -> f32 {
        self.world
            .bodies
            .iter()
            .map(|b| b.velocity.length())
            .fold(0.0, f32::max)
    }

    fn letter_deepest_y(&self, index: usize) -> f32 {
        let letter = &self.letters[index];
        let pose = self.letter_pose(index);
        letter
            .hull
            .iter()
            .map(|v| pose.rotor.apply(*v).y + pose.position.y)
            .fold(f32::INFINITY, f32::min)
    }

    fn deepest_dynamic_point(&self) -> f32 {
        let mut deepest = f32::INFINITY;
        for body in self.world.bodies.iter() {
            let Shape::ConvexPolytope4D { vertices } = &body.collider else {
                continue;
            };
            for v in vertices {
                deepest = deepest.min(body.orientation.rotation.apply(*v).y + body.position.y);
            }
        }
        deepest
    }
}

// Draw in `[0, 1)`. The top 24 bits convert to `f32` exactly and the scale is
// a power of two, so the value is identical on any host that rounds
// IEEE-754.
fn unit(draw: u64) -> f32 {
    ((draw >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

// Draw in `[-1, 1)`, same exactness argument as the unit draw above.
fn signed_unit(draw: u64) -> f32 {
    2.0 * unit(draw) - 1.0
}

fn lerp(a: f32, b: f32, u: f32) -> f32 {
    a + (b - a) * u
}

fn letter_from(solid: &GlyphSolid, index: usize) -> Result<HeroLetter> {
    let (centre, shape) = solid
        .rigid_hull_4d()
        .ok_or_else(|| anyhow::anyhow!("{:?} has no rigid hull", solid.ch()))?;
    let Shape::ConvexPolytope4D { vertices } = shape else {
        anyhow::bail!("{:?} hulls to a non-convex collider", solid.ch());
    };
    let mut mesh = Visualizable::<3>::to_triangles(solid)
        .map_err(|e| anyhow::anyhow!("{:?} has no render mesh: {e:?}", solid.ch()))?;
    for v in &mut mesh.vertices {
        v[0] -= centre.x;
        v[1] -= centre.y;
        v[2] -= centre.z;
    }
    let lowest = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
    let mark = Vec4::new(centre.x, RELEASE_CLEARANCE - lowest, centre.z, centre.w);
    let side = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
    Ok(HeroLetter {
        mesh,
        hull: vertices,
        mark,
        entry: mark + Vec4::new(side * ENTRY_SPAN, 0.0, 0.0, 0.0),
        track: format!("letter{index}"),
        body: None,
    })
}

fn word_span(letters: &[HeroLetter]) -> (f32, f32) {
    let lo = letters.iter().fold(f32::INFINITY, |m, l| m.min(l.mark.x));
    let hi = letters
        .iter()
        .fold(f32::NEG_INFINITY, |m, l| m.max(l.mark.x));
    (lo - RAIN_SIZE, hi + RAIN_SIZE)
}

fn assembly_timeline(letters: &[HeroLetter]) -> Timeline {
    let seconds = |ticks: u32| ticks as f32 / TICK_HZ as f32;
    Timeline {
        fps: TICK_HZ,
        frames: ASSEMBLE_TICKS + 1,
        w_slice: None,
        bodies: letters
            .iter()
            .enumerate()
            .map(|(index, letter)| {
                let start = LETTER_STAGGER_TICKS * index as u32;
                BodyTrack {
                    name: letter.track.clone(),
                    position: Some(
                        Track::new()
                            .key(seconds(start), letter.entry, Ease::Linear)
                            .key(
                                seconds(start + LETTER_SLIDE_TICKS),
                                letter.mark,
                                Ease::InOutCubic,
                            ),
                    ),
                    orientation: None,
                }
            })
            .collect(),
    }
}

const LETTER_COLOR: [f32; 4] = [0.92, 0.90, 0.86, 1.0];

// Read by the physics half-space the letters land on and by the background's
// analytic ground, so the drawn floor cannot drift from the one they hit.
const FLOOR_Y: f32 = 0.0;

// The letters live in a slab about `w = 0`, so this is where their neighbours
// have to be cut to share a scene with them.
const W_SLICE: f32 = 0.0;

// 32-bit float: the caps of a tumbling 24-cell are thin and densely stacked,
// and 24-bit depth cracks them.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const BOOT_ORBIT_DISTANCE: f32 = 7.5;
const BOOT_ORBIT_PITCH: f32 = -0.12;
const BOOT_EYE_HEIGHT: f32 = 1.4;

// Hack Regular, the same face the HUD bakes its atlas from and for the same
// reason: a system font would make the letters, and therefore the whole
// trajectory, differ between two machines running the same binary.
pub(crate) fn hero_font_bytes() -> &'static [u8] {
    epaint_default_fonts::HACK_REGULAR
}

// The scene's whole GPU footprint, in one call so a rebuild's cost is
// measurable from a device with no surface behind it.
pub(crate) fn build_triangles(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> TriangleRasterNode {
    TriangleRasterNode::new(
        device,
        format,
        DepthMode::ReadWrite {
            format: DEPTH_FORMAT,
        },
        loam_render::triangle_raster::FragmentShading::FaceNormalLambert,
        samples,
    )
}

fn drop_color(polytope: Polytope4) -> [f32; 3] {
    crate::catalog::SHAPE_CATALOG
        .iter()
        .find(|entry| entry.shape.polytope4() == Some(polytope))
        .map(|entry| entry.body_color)
        .unwrap_or([1.0, 1.0, 1.0])
}

// A drop is SLICED at [`W_SLICE`], so a tumble through a `w` plane visibly
// changes its cross-section. A letter is PROJECTED: its solid is a product with
// the glyph slab, so its true slice is the same cross-section at every `w` the
// slab covers, and slicing it would cost a re-extraction per frame.
fn build_frame_mesh(
    sequence: &HeroSequence,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    mesh: &mut TriangleMesh<3>,
) {
    mesh.vertices.clear();
    mesh.colors.clear();
    mesh.indices.clear();
    push_letters(sequence, mesh);
    push_drop_caps(sequence, local, scratch, mesh);
}

fn push_letters(sequence: &HeroSequence, mesh: &mut TriangleMesh<3>) {
    for (index, letter) in sequence.letters().iter().enumerate() {
        let pose = sequence.letter_pose(index);
        let base = mesh.vertices.len() as u32;
        let translate = pose.position_r3();
        for v in &letter.mesh.vertices {
            let posed = pose.rotor.apply(Vec4::new(v[0], v[1], v[2], 0.0));
            mesh.vertices
                .push((posed.truncate() + translate).to_array());
            mesh.colors.push(LETTER_COLOR);
        }
        mesh.indices
            .extend((letter.mesh.indices.iter()).map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    }
}

fn push_drop_caps(
    sequence: &HeroSequence,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    mesh: &mut TriangleMesh<3>,
) {
    for (index, drop) in sequence.drops().iter().enumerate() {
        let pose = sequence.drop_pose(index);
        let topo = drop.polytope().topology();
        local.clear();
        local.extend(
            (topo.vertices.iter())
                .map(|v| RAIN_SIZE * pose.rotor.apply(*v) + Vec4::W * pose.position.w),
        );
        let [r, g, b] = drop.color();
        let start = mesh.vertices.len();
        polytope_section_faces_append(
            topo.edges,
            topo.cells,
            local,
            WPlane::new(W_SLICE),
            [r, g, b, 1.0],
            scratch,
            mesh,
        );
        let translate = pose.position_r3();
        for v in &mut mesh.vertices[start..] {
            v[0] += translate.x;
            v[1] += translate.y;
            v[2] += translate.z;
        }
    }
}

pub(crate) struct HeroScene {
    sequence: HeroSequence,
    seed: u64,
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    console: Console<Environment>,
    environment: Environment,
    triangles: TriangleRasterNode,
    sky_ground: SkyGroundNode,
    depth: Option<DepthBuffer>,
    mesh: TriangleMesh<3>,
    local_vertices: Vec<Vec4>,
    section_scratch: SectionScratch,
    hold_at_end: bool,
    paused: bool,
}

impl HeroScene {
    fn build_console() -> Console<Environment> {
        let mut console = Console::<Environment>::new();
        loam_app::shell::register_command::<Environment, crate::shell::Playground>(&mut console);
        register_ground_command(&mut console, |env| env);
        console
    }

    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let console = Self::build_console();

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, BOOT_EYE_HEIGHT, BOOT_ORBIT_DISTANCE);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);

        Ok(Self {
            sequence: HeroSequence::new(hero_font_bytes(), DEFAULT_SEED)?,
            seed: DEFAULT_SEED,
            camera,
            orbit,
            console,
            environment: Environment::grass(),
            triangles: build_triangles(
                &ctx.rd.device,
                ctx.rd.target_format(),
                ctx.rd.sample_count(),
            ),
            sky_ground: SkyGroundNode::new(
                &ctx.rd.device,
                ctx.rd.target_format(),
                DEPTH_FORMAT,
                ctx.rd.sample_count(),
            ),
            depth: None,
            mesh: TriangleMesh::<3>::default(),
            local_vertices: Vec::new(),
            section_scratch: SectionScratch::default(),
            hold_at_end: true,
            paused: false,
        })
    }

    fn replay(&mut self, seed: u64) {
        match HeroSequence::new(hero_font_bytes(), seed) {
            Ok(sequence) => {
                self.sequence = sequence;
                self.seed = seed;
            }
            Err(error) => tracing::error!("hero: could not rebuild at seed {seed:#x}: {error:#}"),
        }
    }

    fn panel(&mut self, ctx: &egui::Context) {
        let mut replay: Option<u64> = None;
        egui::Window::new("Hero")
            .id(egui::Id::new("hero-scene-controls"))
            .default_pos(egui::pos2(16.0, 48.0))
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "tick {} / {SEQUENCE_TICKS}   {:?}",
                    self.sequence.tick_index(),
                    self.sequence.phase()
                ));
                ui.label(format!(
                    "{} of {RAIN_COUNT} polychora fallen",
                    self.sequence.drops().len()
                ));
                ui.checkbox(&mut self.paused, "pause (Space)");
                ui.checkbox(&mut self.hold_at_end, "hold on the last frame");
                if ui.button("replay this seed").clicked() {
                    replay = Some(self.seed);
                }
                if ui.button("next seed (N)").clicked() {
                    replay = Some(self.seed.wrapping_add(1));
                }
                ui.label(
                    egui::RichText::new(format!("seed {:#x}", self.seed))
                        .small()
                        .weak(),
                );
            });
        if let Some(seed) = replay {
            self.replay(seed);
        }
    }
}

impl loam_app::shell::Scene for HeroScene {
    fn apply_command(
        &mut self,
        cmd: &loam_app::command::CommandLine,
        _ctx: &mut loam_app::command::CommandCtx<'_>,
    ) -> Result<()> {
        self.console
            .dispatch(&cmd.name, &cmd.arg_refs(), &mut self.environment);
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if !self.paused {
            for _ in 0..ctx.n_ticks {
                if self.hold_at_end && self.sequence.finished() {
                    break;
                }
                self.sequence.tick();
            }
        }
        let cfg = &ctx.rd.surface_bundle.config;
        self.camera.aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        if !ctx.ui_capture.pointer {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt);
        }
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        self.panel(ctx);
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
            KeyCode::KeyN => self.replay(self.seed.wrapping_add(1)),
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

        // This scene owns the clear: nothing runs before it in the frame's
        // encoder, and the raster node loads rather than clears.
        self.sky_ground.set_uniforms(
            &rd.queue,
            &SkyGroundUniforms::new(
                view_proj,
                Viewport::full([cfg.width, cfg.height]),
                self.environment.ground(FLOOR_Y, true),
            ),
        );
        self.sky_ground
            .record(ctx.encoder, ctx.view, &depth.view, None);

        build_frame_mesh(
            &self.sequence,
            &mut self.local_vertices,
            &mut self.section_scratch,
            &mut self.mesh,
        );
        self.triangles.upload::<EuclideanR3, 3>(
            &rd.device,
            &rd.queue,
            &self.mesh,
            &Projection::Identity,
        );
        self.triangles.set_camera(&rd.queue, view_proj);
        self.triangles
            .record(ctx.encoder, ctx.view, Some(&depth.view), None);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - hero")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_physics::manifold::PENETRATION_SLOP;

    const SEED: u64 = 0x10a3_5eed;

    // Ticks over which a settled letter must hold still, and the position
    // spread allowed across them. The bound is `glyph-letter-bodies`'s, and the
    // measured worst here is 0.0063 em on `L`. What is left at rest is the
    // Baumgarte limit cycle: gravity sinks the letter over a step and the
    // positional bias pushes it back out, forever, at the slop's amplitude.
    const REST_WINDOW_TICKS: u32 = 60;
    const REST_SPREAD: f32 = 0.02;

    // Displacement, in em, that counts as a letter having been knocked aside
    // rather than nudged. Measured at this seed by
    // `the_quoted_figures_are_the_ones_the_scene_produces`: the rain moves `L`
    // 1.801, `O` 1.298, `A` 0.775 and `M` 0.620 em, so the criterion has 2.5x
    // of margin on the least-moved letter; the threshold is not the
    // measurement.
    const SCATTER_THRESHOLD: f32 = 0.25;

    // A tunneling bound, not a contact bound: the resting overlap the solver
    // holds is `PENETRATION_SLOP`, and an impact transiently reaches 0.034, so
    // a body 0.075 under would be one that the contact never caught.
    const TUNNEL_DEPTH: f32 = 0.075;

    // 1.5x the measured 21342 at the fullest tick: the mesh is rebuilt and
    // re-uploaded every frame, so its size is a per-frame bandwidth cost and
    // not a one-time bake.
    const MESH_VERTEX_BUDGET: u32 = 32_000;

    // Per-step travel, in em, the R⁴ narrowphase still resolves against a thin
    // static wall, as recorded by `loam_physics::world`'s tunneling gate.
    // Sampling the floor test once per tick is sound only if no body can cross
    // and return inside a tick, which is what bounding the travel buys.
    const RESOLVABLE_STEP_TRAVEL: f32 = 0.150;

    fn scene() -> HeroSequence {
        HeroSequence::new(hero_font_bytes(), SEED).expect("hero scene")
    }

    fn label(index: usize) -> char {
        WORD.chars()
            .nth(index)
            .expect("letter index is in the word")
    }

    fn letter_positions(scene: &HeroSequence) -> Vec<Vec4> {
        (0..scene.letters().len())
            .map(|i| scene.letter_pose(i).position)
            .collect()
    }

    #[test]
    fn the_whole_sequence_is_a_function_of_its_seed_and_replays_bit_for_bit() {
        let trace = || {
            let mut scene = scene();
            let mut samples = Vec::with_capacity(SEQUENCE_TICKS as usize);
            while !scene.finished() {
                scene.tick();
                let mut frame: Vec<[f32; 4]> = letter_positions(&scene)
                    .iter()
                    .map(Vec4::to_array)
                    .collect();
                frame.extend(
                    (0..scene.drops().len()).map(|i| scene.drop_pose(i).position.to_array()),
                );
                samples.push(frame);
            }
            samples
        };
        let first = trace();
        assert_eq!(first.len(), SEQUENCE_TICKS as usize);
        assert_eq!(first, trace());
    }

    #[test]
    fn the_seed_moves_the_rain_and_leaves_the_letters_landing_untouched() {
        let settled = |seed: u64| {
            let mut scene = HeroSequence::new(hero_font_bytes(), seed).expect("hero scene");
            scene.run_to(RAIN_START_TICK);
            let letters = letter_positions(&scene);
            scene.run_to(SEQUENCE_TICKS);
            let drops: Vec<Vec4> = (0..scene.drops().len())
                .map(|i| scene.drop_pose(i).position)
                .collect();
            (letters, drops)
        };
        let (letters_a, drops_a) = settled(SEED);
        let (letters_b, drops_b) = settled(SEED ^ 0xdead_beef);
        assert_eq!(
            letters_a, letters_b,
            "the seed reached the letters' landing"
        );
        assert_eq!(drops_a.len(), drops_b.len());
        assert!(
            drops_a.iter().zip(&drops_b).any(|(a, b)| a != b),
            "two seeds rained identically, so the seed is decoration"
        );
    }

    #[test]
    fn the_director_owns_a_letter_until_it_becomes_a_body_and_never_after() {
        let mut scene = scene();
        assert_eq!(scene.world.bodies.iter().count(), 1, "floor only");
        scene.run_to(ASSEMBLE_TICKS - 1);
        assert_eq!(scene.phase(), Phase::Assemble);
        assert_eq!(
            scene.world.bodies.iter().count(),
            1,
            "a letter became a body while the director still owned it"
        );

        let directed = letter_positions(&scene);
        let frame_at_release = scene.director.frame() + 1;
        scene.tick();
        assert_eq!(scene.phase(), Phase::Fall);
        assert_eq!(
            scene.world.bodies.iter().count(),
            scene.letters().len() + 1,
            "the release did not hand every letter to the solver"
        );
        // Each body starts at the pose the director's last key put its letter
        // in. The slides have all finished by then, so the last two director
        // frames are the same pose and the equality is exact.
        for (index, before) in directed.iter().enumerate() {
            assert_eq!(scene.letter_pose(index).position, *before);
        }

        scene.run_to(RAIN_START_TICK);
        assert_eq!(
            scene.director.frame(),
            frame_at_release,
            "the director advanced after handing its letters over"
        );
    }

    #[test]
    fn every_letter_is_at_rest_on_the_floor_before_the_rain_starts() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK - REST_WINDOW_TICKS);
        let mut window: Vec<Vec<Vec4>> = Vec::with_capacity(REST_WINDOW_TICKS as usize);
        while scene.tick_index() < RAIN_START_TICK {
            scene.tick();
            window.push(letter_positions(&scene));
        }
        assert!(scene.drops().is_empty(), "a drop spawned during the settle");

        for index in 0..scene.letters().len() {
            let spread = window
                .iter()
                .map(|frame| frame[index].distance(window[0][index]))
                .fold(0.0f32, f32::max);
            assert!(
                spread < REST_SPREAD,
                "{:?} still moves {spread} in the second before the rain",
                label(index)
            );
            let deepest = scene.letter_deepest_y(index);
            assert!(
                deepest <= 1.0e-4,
                "{:?} floats with its lowest point at {deepest}",
                label(index)
            );
            assert!(
                deepest >= -2.0 * PENETRATION_SLOP,
                "{:?} sank to {deepest} through the floor",
                label(index)
            );
        }
    }

    #[test]
    fn the_quoted_figures_are_the_ones_the_scene_produces() {
        let mut s = scene();
        s.run_to(RAIN_START_TICK);
        let settled = letter_positions(&s);
        let mut last_spawn = RAIN_START_TICK;
        let mut seen = s.drops().len();
        while !s.finished() {
            s.tick();
            if s.drops().len() > seen {
                seen = s.drops().len();
                last_spawn = s.tick_index();
            }
        }
        let scattered = letter_positions(&s);
        let measured: Vec<f32> = settled
            .iter()
            .zip(&scattered)
            .map(|(b, a)| b.distance(*a))
            .collect();
        for (got, want) in measured.iter().zip([1.801f32, 1.298, 0.775, 0.620]) {
            assert!(
                (got - want).abs() < 5e-3,
                "SCATTER_THRESHOLD's doc quotes {want} em; the scene produces {got}"
            );
        }
        assert_eq!(
            last_spawn, 308,
            "SEQUENCE_TICKS' doc quotes the last spawn at tick 308"
        );
    }

    #[test]
    fn the_rain_knocks_at_least_one_letter_past_the_scatter_threshold() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK);
        let settled = letter_positions(&scene);
        scene.run_to(SEQUENCE_TICKS);
        assert_eq!(scene.drops().len(), RAIN_COUNT, "the rain did not all fall");

        let scattered = letter_positions(&scene);
        let worst = settled
            .iter()
            .zip(&scattered)
            .map(|(before, after)| before.distance(*after))
            .fold(0.0f32, f32::max);
        assert!(
            worst > SCATTER_THRESHOLD,
            "the rain moved the worst-hit letter only {worst}"
        );
    }

    #[test]
    fn nothing_passes_through_the_floor_at_any_tick_of_the_run() {
        let mut scene = scene();
        while !scene.finished() {
            scene.tick();
            let deepest = scene.deepest_dynamic_point();
            assert!(
                deepest > -TUNNEL_DEPTH,
                "a body reached {deepest} below the floor at tick {}",
                scene.tick_index()
            );
            let travel = scene.fastest_body_speed() * SOLVER_DT;
            assert!(
                travel < RESOLVABLE_STEP_TRAVEL,
                "a body travelled {travel} in one step at tick {}",
                scene.tick_index()
            );
        }
        for index in 0..scene.letters().len() {
            assert!(scene.letter_pose(index).position.y > 0.0);
        }
        for index in 0..scene.drops().len() {
            assert!(scene.drop_pose(index).position.y > 0.0);
        }
    }

    #[test]
    fn every_raining_shape_collides_as_its_own_hull_rather_than_a_bounding_ball() {
        let mut scene = scene();
        scene.run_to(SEQUENCE_TICKS);
        assert_eq!(scene.drops().len(), RAIN_COUNT);
        for (index, drop) in scene.drops().iter().enumerate() {
            let body = &scene.world.bodies[drop.body];
            let Shape::ConvexPolytope4D { vertices } = &body.collider else {
                panic!(
                    "{:?} rains as a {:?}",
                    drop.polytope(),
                    body.collider.kind()
                );
            };
            assert_eq!(vertices.len(), drop.polytope().topology().vertices.len());
            let exact = regular_polytope4_inertia(drop.polytope(), RAIN_MASS, RAIN_SIZE)
                .expect("a raining shape has a derived moment");
            assert_eq!(body.inertia, exact, "drop {index} kept the ball's moment");
        }
        for excluded in [Polytope4::Cell120, Polytope4::Cell600] {
            assert!(
                !RAIN_SHAPES.contains(&excluded),
                "{excluded:?} rains but has no hull"
            );
            assert!(regular_polytope4_inertia(excluded, 1.0, 1.0).is_none());
        }
    }

    #[test]
    fn a_falling_letter_is_one_convex_prism_and_not_its_static_cover() {
        let mut scene = scene();
        scene.run_to(ASSEMBLE_TICKS + 1);
        assert_eq!(scene.world.bodies.iter().count(), scene.letters().len() + 1);

        let font = ab_glyph::FontRef::try_from_slice(hero_font_bytes()).expect("font");
        let solids = layout_word(&font, WORD, &GlyphParams::default()).expect("layout");
        let cover: usize = solids.iter().map(GlyphSolid::collider_count).sum();
        assert!(
            cover > 8 * scene.letters().len(),
            "{cover} cover boxes is not enough to make the hull a cut"
        );
        for (index, letter) in scene.letters().iter().enumerate() {
            let body = &scene.world.bodies[letter.body.expect("released")];
            let Shape::ConvexPolytope4D { vertices } = &body.collider else {
                panic!("{:?} falls as a {:?}", label(index), body.collider.kind());
            };
            assert_eq!(vertices.len() % 4, 0);
            assert!(vertices.len() <= 32, "{:?} overflows the cap", label(index));
        }
    }

    #[test]
    fn the_assembly_slides_every_letter_onto_its_mark_before_the_release() {
        let mut scene = scene();
        let entries = letter_positions(&scene);
        for (index, letter) in scene.letters().iter().enumerate() {
            assert_eq!(entries[index], letter.entry);
            assert!(
                (letter.entry.x - letter.mark.x).abs() > 1.0,
                "{:?} starts on top of its mark",
                label(index)
            );
        }
        scene.run_to(LETTER_STAGGER_TICKS);
        assert_ne!(letter_positions(&scene), entries);

        scene.run_to(ASSEMBLE_TICKS);
        for (index, letter) in scene.letters().iter().enumerate() {
            assert_eq!(
                scene.letter_pose(index).position,
                letter.mark,
                "{:?} did not reach its mark",
                label(index)
            );
        }
    }

    #[test]
    fn the_frame_mesh_is_well_formed_and_inside_the_upload_budget() {
        let mut scene = scene();
        scene.run_to(SEQUENCE_TICKS);
        let mut mesh = TriangleMesh::<3>::default();
        let (mut local, mut scratch) = (Vec::new(), SectionScratch::default());
        build_frame_mesh(&scene, &mut local, &mut scratch, &mut mesh);

        assert_eq!(mesh.colors.len(), mesh.vertices.len());
        let count = mesh.vertices.len() as u32;
        assert!(count > 4, "the mesh holds the floor and nothing else");
        for tri in &mesh.indices {
            for i in tri {
                assert!(*i < count, "index {i} past {count} vertices");
            }
        }
        for v in &mesh.vertices {
            assert!(v.iter().all(|c| c.is_finite()), "non-finite vertex {v:?}");
        }
        assert!(
            count < MESH_VERTEX_BUDGET,
            "the frame mesh grew to {count} vertices"
        );

        let repeat = mesh.vertices.clone();
        build_frame_mesh(&scene, &mut local, &mut scratch, &mut mesh);
        assert_eq!(mesh.vertices, repeat);
    }

    #[test]
    fn a_settled_letter_is_drawn_standing_on_the_floor_within_the_covers_margin() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK);
        let font = ab_glyph::FontRef::try_from_slice(hero_font_bytes()).expect("font");
        let solids = layout_word(&font, WORD, &GlyphParams::default()).expect("layout");
        let margin = solids
            .iter()
            .map(GlyphSolid::collider_margin)
            .fold(0.0f32, f32::max);

        let mut mesh = TriangleMesh::<3>::default();
        push_letters(&scene, &mut mesh);
        assert_eq!(
            mesh.vertices.len(),
            scene
                .letters()
                .iter()
                .map(|l| l.mesh.vertices.len())
                .sum::<usize>()
        );

        let mut offset = 0;
        for (index, letter) in scene.letters().iter().enumerate() {
            let span = &mesh.vertices[offset..offset + letter.mesh.vertices.len()];
            offset += letter.mesh.vertices.len();
            let drawn = span.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
            assert!(
                drawn > -2.0 * PENETRATION_SLOP,
                "{:?} is drawn {drawn} below the floor",
                label(index)
            );
            assert!(
                drawn < margin + 2.0 * PENETRATION_SLOP,
                "{:?} floats {drawn} above the floor, past the {margin} margin",
                label(index)
            );
        }
    }

    #[test]
    fn a_drops_cross_section_is_cut_at_the_slice_and_changes_as_it_tumbles() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK + 60);
        assert!(!scene.drops().is_empty(), "no drop to slice");

        let caps = |scene: &HeroSequence| {
            let mut mesh = TriangleMesh::<3>::default();
            let (mut local, mut scratch) = (Vec::new(), SectionScratch::default());
            push_drop_caps(scene, &mut local, &mut scratch, &mut mesh);
            mesh.vertices
        };
        let before = caps(&scene);
        assert!(!before.is_empty(), "every drop missed the slice");

        let rotors: Vec<Rotor4> = (0..scene.drops().len())
            .map(|i| scene.drop_pose(i).rotor)
            .collect();
        scene.run_to(RAIN_START_TICK + 70);
        assert!(
            (0..rotors.len()).any(|i| scene.drop_pose(i).rotor != rotors[i]),
            "nothing tumbled, so the cap has no reason to change"
        );
        assert_ne!(before, caps(&scene), "the cut did not follow the tumble");
    }

    #[test]
    fn the_assembled_word_is_in_reading_order_on_the_baseline() {
        let mut scene = scene();
        scene.run_to(ASSEMBLE_TICKS);
        let placed = letter_positions(&scene);
        let font = ab_glyph::FontRef::try_from_slice(hero_font_bytes()).expect("font");
        let solids = layout_word(&font, WORD, &GlyphParams::default()).expect("layout");
        let inked: Vec<char> = solids
            .iter()
            .filter(|solid| !solid.is_blank())
            .map(GlyphSolid::ch)
            .collect();
        assert_eq!(inked, WORD.chars().collect::<Vec<_>>());
        assert_eq!(inked.len(), placed.len());
        for pair in placed.windows(2) {
            assert!(pair[1].x > pair[0].x, "the word came out of order");
        }
        let heights: Vec<f32> = (0..scene.letters().len())
            .map(|i| scene.letter_deepest_y(i))
            .collect();
        for h in &heights {
            assert!((h - RELEASE_CLEARANCE).abs() < 1.0e-5, "released at {h}");
        }
    }

    #[test]
    fn the_console_carries_the_live_ground_controls() {
        assert!(HeroScene::build_console().has_command("ground"));
        let mut env = Environment::default();
        HeroScene::build_console().dispatch("ground", &["fog", "0.04"], &mut env);
        assert_eq!(env.fog_per_unit, 0.04);
    }
}
