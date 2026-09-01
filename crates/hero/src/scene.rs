//! Falling letters take [`GlyphSolid::rigid_hull_4d`], the one convex prism per
//! letter, and not [`GlyphSolid::colliders_4d`]'s faithful box cover: a moving
//! faithful cover would need a compound collider `loam-physics` does not have.

use std::borrow::Cow;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_egui::{Console, ConsoleUi};
use loam_math::{Bivector4, EuclideanR3, EuclideanR4, Projection, Rotor, Rotor4, WPlane};
use loam_physics::body::MASK_ALL;
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::{BodyId, Gravity, World};
use loam_render::{
    DepthBuffer, DepthMode, SkyGroundNode, SkyGroundUniforms, TriangleRasterNode, Viewport,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
#[cfg(test)]
use loam_shape::Visualizable;
use loam_shape::{Shape, TriangleMesh};
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};
use loam_time::director::{BodyTrack, Director, Drive, Ease, Timeline, Track};

use loam_app::capture::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};
use loam_app::environment::{register_floor_command, register_ground_command, Environment};

const WORD: &str = "LOAM";

const TICK_HZ: u32 = 60;

// A letter hull dropped on a half-space settles at 240 Hz and does NOT at 120,
// where the same drops skid 0.45 to 0.61 em and end tipped on a corner:
// `polytope_halfspace_r4` reports one deepest vertex per step, a manifold holds
// four, and halving the rate halves
// the constraints the solver has accumulated by the time a letter rocks over.
// Four sub-steps of a 60 Hz tick is that rate, reached without moving the host
// clock.
const SUBSTEPS_PER_TICK: usize = 4;

// Fixed and never derived from wall time.
const SOLVER_DT: f32 = 1.0 / (TICK_HZ as f32 * SUBSTEPS_PER_TICK as f32);

// One captured frame per tick. Every captured frame is a full image in the
// APNG and is held in memory until the stop, so the rate is the file-size and
// footprint dial; the delays keep playback at the sequence's own speed at any
// rate, and halving this halves both costs.
const RECORD_FPS: u16 = TICK_HZ as u16;

const GRAVITY: f32 = -9.8;

const PILE_PGS_ITERS: usize = 20;

// Collision groups. A drop still falling passes through the other drops still
// falling: mid-air collisions turn the rain into a scatter of ricochets before
// it ever reaches the word, and the thing worth watching is what the rain does
// TO the letters. It collides with everything the moment it touches anything,
// so a landed pile still stacks.
const GROUP_SCENERY: u32 = 1 << 0;
const GROUP_FALLING: u32 = 1 << 1;
const GROUP_LANDED: u32 = 1 << 2;
const MASK_FALLING: u32 = GROUP_SCENERY | GROUP_LANDED;

const ASSEMBLE_TICKS: u32 = 90;

// The last letter must still finish inside [`ASSEMBLE_TICKS`], which
// `the_assembly_slides_every_letter_onto_its_mark_before_the_release` pins.
const LETTER_STAGGER_TICKS: u32 = 12;

const LETTER_SLIDE_TICKS: u32 = 36;

// How far out in `w` a letter starts: two letterforms, so the entrance sweeps
// its section through both neighbours before settling on its own.
const W_ENTRY_SPAN: f32 = 0.6;

// Height of the lowest hull vertex above the floor at release, in em. The
// settle holds at 0.15 and 0.25 either side of this; the narrowphase still
// reports one contact per step, so above 0.25 is untested.
const RELEASE_CLEARANCE: f32 = 0.20;

// Every letter is at rest well inside this, which
// `every_letter_is_at_rest_on_the_floor_before_the_rain_starts` pins at the
// last tick before the spawn.
const SETTLE_TICKS: u32 = 120;

const RAIN_START_TICK: u32 = ASSEMBLE_TICKS + SETTLE_TICKS;

// The sim runs until here and is then FROZEN for the `w` sweep. Sweeping while
// the pile is still rolling reads as two unrelated motions at once; frozen, the
// only thing moving is the slice, which is the point of the sweep.
const PHYSICS_TICKS: u32 = 360;
const PHYSICS_PAUSE_TICK: u32 = RAIN_START_TICK + PHYSICS_TICKS;

// One full sweep out to `SLICE_SWEEP_RANGE` and back.
const SWEEP_TICKS: u32 = 300;

pub(crate) const SEQUENCE_TICKS: u32 = PHYSICS_PAUSE_TICK + SWEEP_TICKS;

// A cap on the drops the scene will hold, not a target: the rain spawns until
// the sim freezes, and this is the allocation and the point past which a
// spawn is skipped rather than a schedule.
const RAIN_CAP: usize = 64;

const RAIN_INTERVAL_TICKS: u32 = 7;
const RAIN_INTERVAL_JITTER: u32 = 4;

// Written as the difference of the two above so a jitter wider than the
// interval is a compile error rather than a wrap to a spawn tick four billion
// ticks away.
const RAIN_INTERVAL_MIN: u32 = RAIN_INTERVAL_TICKS - RAIN_INTERVAL_JITTER;

const RAIN_SIZE: f32 = 0.30;

const LETTER_MASS: f32 = 1.0;

// Lighter than a letter, which a 0.30 em drop should be. 48 drops heavier than
// this drive a body past the 0.075 em tunnelling bound; heavier rain is also
// what scatters the letters, so this is the most the floor will hold.
const RAIN_MASS: f32 = 0.75;

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

const RAIN_SHAPES: [Polytope4; 6] = [
    Polytope4::Cell24,
    Polytope4::Pentatope,
    Polytope4::Cell600,
    Polytope4::Cell16,
    Polytope4::Tesseract,
    Polytope4::Cell120,
];

pub(crate) struct HeroLetter {
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
    morph: MorphField,
    world: World<EuclideanR4>,
    director: Director,
    letters: Vec<HeroLetter>,
    drops: Vec<HeroDrop>,
    // Advanced only by a spawn, so the draw a given drop gets does not depend
    // on how many ticks passed before it.
    rng: u64,
    tick: u32,
    next_spawn_tick: u32,
    /// Reused across sub-steps; a fresh `Vec` here would allocate four times a
    /// tick for four floats.
    letter_w_scratch: Vec<f32>,
}

impl HeroSequence {
    pub(crate) fn new(font_bytes: &[u8], seed: u64) -> Result<Self> {
        let font = ab_glyph::FontRef::try_from_slice(font_bytes)?;
        let solids = layout_word(&font, WORD, &GlyphParams::default())?;

        let mut world = World::new(EuclideanR4);
        register_default_narrowphase(&mut world.narrowphase);
        world.push_field(Box::new(Gravity::new(Vec4::new(0.0, GRAVITY, 0.0, 0.0))));
        // A pile this deep needs more sweeps than the default 8: the load path
        // from the top of the rain down to the floor is many contacts long,
        // and PGS propagates one contact per iteration.
        world.pgs_iters = PILE_PGS_ITERS;
        let floor = world.push_body(halfspace4_body_r4(Vec4::Y, FLOOR_Y));
        world.bodies[floor].restitution = RESTITUTION;

        let mut letters: Vec<HeroLetter> = solids
            .iter()
            .filter(|solid| !solid.is_blank())
            .enumerate()
            .map(|(index, solid)| letter_from(solid, index))
            .collect::<Result<_>>()?;
        centre_word_on_origin(&mut letters);

        let director = Director::new(assembly_timeline(&letters))?;
        let cell = GlyphParams::default().em_size / GlyphParams::default().resolution as f32;
        let morph = MorphField::new(&solids, cell)
            .ok_or_else(|| anyhow::anyhow!("{WORD} laid out with no ink to morph"))?;

        Ok(Self {
            morph,
            world,
            director,
            letters,
            drops: Vec::with_capacity(RAIN_CAP),
            // xorshift64* has a fixed point at zero, so a zero seed would
            // produce a constant stream rather than a degenerate one nobody
            // notices. Mixing in an odd constant keeps seed 0 usable.
            rng: seed ^ 0x9e37_79b9_7f4a_7c15,
            tick: 0,
            next_spawn_tick: RAIN_START_TICK,
            letter_w_scratch: Vec::new(),
        })
    }

    pub(crate) fn tick(&mut self) {
        if self.tick < ASSEMBLE_TICKS {
            self.director.advance();
        } else if self.tick < PHYSICS_PAUSE_TICK {
            if self.tick >= self.next_spawn_tick {
                self.spawn_drop();
            }
            let mut before_w = std::mem::take(&mut self.letter_w_scratch);
            for _ in 0..SUBSTEPS_PER_TICK {
                self.letter_w_velocities(&mut before_w);
                self.world.step(SOLVER_DT);
                self.hold_letters_in_the_slice();
                self.keep_scenery_from_moving_letters_in_w(&before_w);
                self.land_touched_drops();
            }
            self.letter_w_scratch = before_w;
        }
        self.tick += 1;
        // Closes the assemble tick rather than opening the fall one, so the
        // director and the world never both own a letter at a tick boundary.
        if self.tick == ASSEMBLE_TICKS {
            self.release_letters();
        }
    }

    /// Promotes a drop out of [`GROUP_FALLING`] the first step it touches
    /// anything, so it stops being transparent to the rest of the rain. Read
    /// off the manifolds rather than a height, because "has landed" is exactly
    /// "has a contact" and a height would have to guess at the pile.
    fn land_touched_drops(&mut self) {
        for index in 0..self.drops.len() {
            let id = self.drops[index].body;
            if self.world.bodies[id].collision_group != GROUP_FALLING {
                continue;
            }
            let touched = (self.world.manifolds.iter())
                .any(|(key, manifold)| (key.0 == id || key.1 == id) && !manifold.points.is_empty());
            if touched {
                let body = &mut self.world.bodies[id];
                body.collision_group = GROUP_LANDED;
                body.collision_mask = MASK_ALL;
            }
        }
    }

    /// A DELIBERATE DEVIATION: letters spin only inside the slice. The draw
    /// poses a section with the rotor and drops `w`, and a 4D rotation's 3x3
    /// block is not a rotation, so a letter tumbling into a `w` plane flattens
    /// as that block goes singular. Held here the block is a rotation and the
    /// draw is exact; the letters read as 4D through `w` translation instead,
    /// and the rain, which is sliced properly, still tumbles anywhere.
    ///
    /// The three kept planes are closed under the Lie bracket, so the rotor
    /// stays in SO(3): a projection, not a fight with the integrator.
    fn hold_letters_in_the_slice(&mut self) {
        for letter in &self.letters {
            let Some(body) = letter.body else { continue };
            let spin = &mut self.world.bodies[body].angular_velocity;
            spin.xw = 0.0;
            spin.yw = 0.0;
            spin.zw = 0.0;
        }
    }

    /// Each letter's `w` velocity, to be handed back to
    /// [`Self::keep_scenery_from_moving_letters_in_w`] after the solve.
    fn letter_w_velocities(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.letters.iter().map(|letter| {
            letter
                .body
                .map_or(0.0, |body| self.world.bodies[body].velocity.w)
        }));
    }

    /// A DELIBERATE DEVIATION: scenery may slow a letter along `w` but never
    /// speed one up. Narrow three ways, so a drop striking a letter still
    /// drives it off the slice: scenery contacts only, `w` only, and only the
    /// direction that adds speed.
    ///
    /// Friction, not the normal impulse: the floor's normal is pure `y`, but a
    /// contact's tangent space in R⁴ contains `w`, so a letter's `w`-plane
    /// spin reads as `w` motion at the contact and friction turns it into
    /// linear `w`. One-sided because braking must survive; without it a
    /// rain-pushed letter slides in `w` with nothing to stop it.
    fn keep_scenery_from_moving_letters_in_w(&mut self, before: &[f32]) {
        for (index, letter) in self.letters.iter().enumerate() {
            let Some(body) = letter.body else { continue };
            let only_scenery = (self.world.manifolds.iter())
                .filter(|(key, manifold)| {
                    !manifold.points.is_empty() && (key.0 == body || key.1 == body)
                })
                .all(|(key, _)| {
                    let other = if key.0 == body { key.1 } else { key.0 };
                    self.world.bodies[other].inv_mass == 0.0
                });
            if only_scenery {
                let w = &mut self.world.bodies[body].velocity.w;
                if w.abs() > before[index].abs() {
                    *w = before[index];
                }
            }
        }
    }

    pub(crate) fn finished(&self) -> bool {
        self.tick >= SEQUENCE_TICKS
    }

    /// Where the scene is sliced this tick. Pinned at [`W_SLICE`] for as long
    /// as the sim runs, then swept once everything has frozen: a slice moving
    /// under a pile that is still rolling reads as two unrelated motions, and
    /// the sweep is worth watching precisely because nothing else moves.
    pub(crate) fn slice(&self) -> f32 {
        let Some(since) = self.tick.checked_sub(PHYSICS_PAUSE_TICK) else {
            return W_SLICE;
        };
        let phase = std::f32::consts::TAU * since as f32 / SWEEP_TICKS as f32;
        W_SLICE + SLICE_SWEEP_RANGE * phase.sin()
    }

    /// Centre of the assembled word's bounding box, which is what the orbit
    /// target is set to: the controller aims the camera at its target, so the
    /// point named here is the point that lands at the centre of the frame.
    pub(crate) fn word_centre(&self) -> Vec3 {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for letter in &self.letters {
            for v in &letter.hull {
                let world = letter.mark.truncate() + v.truncate();
                lo = lo.min(world);
                hi = hi.max(world);
            }
        }
        0.5 * (lo + hi)
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
        if self.drops.len() >= RAIN_CAP {
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
        body.collision_group = GROUP_FALLING;
        body.collision_mask = MASK_FALLING;
        // The exact uniform-solid moment, not `polytope_body_r4`'s bounding
        // ball: every one of these symmetry groups acts irreducibly on R⁴, so
        // the scalar inertia slot is exact rather than an approximation.
        body.inertia = regular_polytope4_inertia(polytope, RAIN_MASS, RAIN_SIZE);
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

/// Every letter of the word resampled onto one shared grid, so a blend between
/// two of them is elementwise. This is what makes the wordmark a single 4D
/// solid rather than four: letter `k` is the same field read at a `w` offset
/// of `k`, so sliding a letter through `w` sweeps its cross-section through
/// the other letterforms and lands on its own.
pub(crate) struct MorphField {
    origin: glam::Vec2,
    cell: f32,
    counts: (usize, usize),
    /// One resampled grid per letter, in word order.
    letters: Vec<Vec<f32>>,
    blended: Vec<f32>,
}

// A letter's own field is baked over its own bounding box, so the shared grid
// has to cover the widest letter with the padding the blend needs: a shape
// growing out of a neighbour reaches past both outlines on the way.
const MORPH_PAD_EM: f32 = 0.25;

// How much `w` separates one letterform from the next. Sized against
// `W_ENTRY_SPAN` so the entry sweeps about two letterforms, which reads as a
// morph rather than as a full cycle of the word.
const W_PER_LETTERFORM: f32 = 0.3;

impl MorphField {
    /// `None` if the word has no ink, which cannot happen for a real font but
    /// is the honest result for one that hands back only blanks.
    fn new(solids: &[GlyphSolid], cell: f32) -> Option<Self> {
        let inked: Vec<&GlyphSolid> = solids.iter().filter(|s| !s.is_blank()).collect();
        let mut half = glam::Vec2::ZERO;
        for solid in &inked {
            let field = solid.field()?;
            let (nx, ny) = field.sample_counts();
            let lo = field.sample_position(0, 0);
            let hi = field.sample_position(nx - 1, ny - 1);
            let centre = 0.5 * (lo + hi);
            half = half.max((hi - centre).abs());
        }
        half += glam::Vec2::splat(MORPH_PAD_EM);
        let counts = (
            (2.0 * half.x / cell).ceil() as usize + 1,
            (2.0 * half.y / cell).ceil() as usize + 1,
        );
        let origin = -half;

        // Each letter is resampled about its OWN centre, so the blend morphs a
        // letter in place instead of sliding it across the shared grid.
        let letters = inked
            .iter()
            .map(|solid| {
                let field = solid.field().expect("checked above");
                // The SAME centre `rigid_hull_4d` puts the collider on, which
                // is the hull ring's centroid and not the field's bounding-box
                // centre. Resampling about the box centre instead offsets the
                // drawn section from the body it belongs to, which shows up as
                // the word sinking through the floor.
                let centre = solid
                    .rigid_hull_4d()
                    .map(|(c, _)| glam::Vec2::new(c.x, c.y))
                    .unwrap_or(glam::Vec2::ZERO);
                let mut grid = Vec::with_capacity(counts.0 * counts.1);
                for j in 0..counts.1 {
                    for i in 0..counts.0 {
                        let p = origin + glam::Vec2::new(i as f32, j as f32) * cell;
                        grid.push(field.sample(p + centre));
                    }
                }
                grid
            })
            .collect::<Vec<_>>();
        (!letters.is_empty()).then(|| Self {
            origin,
            cell,
            counts,
            blended: vec![0.0; counts.0 * counts.1],
            letters,
        })
    }

    /// The field `u` letterforms along the word, wrapping, so the sequence is a
    /// loop and a letter approaching from either side of the slice finds a
    /// neighbour rather than an edge.
    fn blend_at(&mut self, u: f32) -> Option<loam_text::glyph::DistanceField2D> {
        let n = self.letters.len();
        let wrapped = u.rem_euclid(n as f32);
        let lo = wrapped.floor() as usize % n;
        let t = wrapped - wrapped.floor();
        let (a, b) = (&self.letters[lo], &self.letters[(lo + 1) % n]);
        for (out, (x, y)) in self.blended.iter_mut().zip(a.iter().zip(b.iter())) {
            *out = x + (y - x) * t;
        }
        loam_text::glyph::DistanceField2D::from_samples(
            self.origin,
            self.cell,
            self.counts.0,
            self.counts.1,
            self.blended.clone(),
        )
    }
}

fn letter_from(solid: &GlyphSolid, index: usize) -> Result<HeroLetter> {
    let (centre, shape) = solid
        .rigid_hull_4d()
        .ok_or_else(|| anyhow::anyhow!("{:?} has no rigid hull", solid.ch()))?;
    let Shape::ConvexPolytope4D { vertices } = shape else {
        anyhow::bail!("{:?} hulls to a non-convex collider", solid.ch());
    };
    let lowest = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
    let mark = Vec4::new(centre.x, RELEASE_CLEARANCE - lowest, centre.z, centre.w);
    let side = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
    Ok(HeroLetter {
        hull: vertices,
        mark,
        entry: mark + Vec4::new(0.0, 0.0, 0.0, side * W_ENTRY_SPAN),
        track: format!("letter{index}"),
        body: None,
    })
}

/// Shifts every mark so the assembled word straddles the origin in `x`. The
/// layout pen starts at `x = 0` and advances right, so an unshifted word sits
/// entirely to one side of the orbit target and reads off-centre in frame.
fn centre_word_on_origin(letters: &mut [HeroLetter]) {
    // The INK, not the marks: a mark is its glyph's own centre, and the
    // letters differ in width, so the midpoint of the marks is not the
    // midpoint of what is drawn.
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for letter in letters.iter() {
        for v in &letter.hull {
            lo = lo.min(letter.mark.x + v.x);
            hi = hi.max(letter.mark.x + v.x);
        }
    }
    let shift = -0.5 * (lo + hi);
    for letter in letters {
        letter.mark.x += shift;
        letter.entry.x += shift;
    }
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
// The slice the scene is viewed at while nothing has been disturbed. Letters
// rest here, so at rest each one cuts its own letterform.
const W_SLICE: f32 = 0.0;

// How far the slice travels once the sim has frozen. A letterform every
// `W_PER_LETTERFORM`, so a sweep of the whole word carries every letter
// through every other letterform and back to its own.
const SLICE_SWEEP_RANGE: f32 = 4.0 * W_PER_LETTERFORM;

// 32-bit float: the caps of a tumbling 24-cell are thin and densely stacked,
// and 24-bit depth cracks them.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// The 3.488 em word fills 42% of a `d * tan(30 deg) * 16/9` half-frame here.
// The bound is the rain, not the word: its +/-2.04 em spread fits the 4.11 em
// half-frame, and its 2.6 to 3.6 em spawn height sits above the 2.73 em top of
// view, so a drop falls into frame rather than appearing inside it.
const BOOT_ORBIT_DISTANCE: f32 = 4.0;
const BOOT_ORBIT_PITCH: f32 = -0.12;
const BOOT_EYE_HEIGHT: f32 = 1.4;

/// Latin Modern Roman 10 Bold, GUST's OpenType cut of Knuth's Computer
/// Modern, under the GUST Font License beside it. Vendored unmodified, so the
/// licence's rename request, which binds derived works, does not apply.
///
/// Bundled rather than a system face so the typeset word, and with it the
/// trajectory, is identical on any machine running the same binary.
pub(crate) fn hero_font_bytes() -> &'static [u8] {
    include_bytes!("../fonts/lmroman10-bold.otf")
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

// The playground's catalogue colours, inlined: this crate draws six shapes and
// does not otherwise need a shape table.
fn drop_color(polytope: Polytope4) -> [f32; 3] {
    match polytope {
        Polytope4::Pentatope => [0.95, 0.55, 0.30],
        Polytope4::Tesseract => [0.30, 0.55, 0.95],
        Polytope4::Cell16 => [0.55, 0.95, 0.40],
        Polytope4::Cell24 => [0.95, 0.45, 0.85],
        Polytope4::Cell120 => [0.40, 0.85, 0.85],
        Polytope4::Cell600 => [0.95, 0.85, 0.40],
    }
}

fn build_frame_mesh(
    sequence: &mut HeroSequence,
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

// The morph field IS a letter's 4D solid, so it is read for the letter's whole
// life: one knocked along `w` sweeps its section through the neighbouring
// letterforms. Exact while a letter's own `w` axis stays parallel to the
// world's, which covers all of its `w` translation; once a hit tumbles it into
// a `w` plane the true section would need marching a 3D implicit surface, and
// the morph parameter is taken at the body centre instead.
//
// 3.31 ms a frame while morphing against 0.62 ms settled: a fifth of a 60 Hz
// budget, and it would not fit at 240 Hz.
fn push_letters(sequence: &mut HeroSequence, mesh: &mut TriangleMesh<3>) {
    let half_depth = 0.5 * GlyphParams::default().depth;
    let slice = sequence.slice();
    for index in 0..sequence.letters().len() {
        let pose = sequence.letter_pose(index);
        let translate = pose.position_r3();
        let base = mesh.vertices.len() as u32;

        // `w` runs the other way from the letterform index so a letter
        // approaching from negative `w` arrives through the letters BEFORE it
        // in the word, which reads as the word assembling rather than
        // unwinding.
        let u = index as f32 - (pose.position.w - slice) / W_PER_LETTERFORM;
        let Some(field) = sequence.morph.blend_at(u) else {
            continue;
        };
        let mut section = TriangleMesh::<3>::default();
        if !loam_text::glyph::append_field_prism(&field, half_depth, LETTER_COLOR, &mut section) {
            continue;
        }
        for v in &section.vertices {
            let posed = pose.rotor.apply(Vec4::new(v[0], v[1], v[2], 0.0));
            mesh.vertices
                .push((posed.truncate() + translate).to_array());
            mesh.colors.push(LETTER_COLOR);
        }
        mesh.indices
            .extend((section.indices.iter()).map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    }
}

fn push_drop_caps(
    sequence: &HeroSequence,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    mesh: &mut TriangleMesh<3>,
) {
    let slice = sequence.slice();
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
            WPlane::new(slice),
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

/// Where `--record` writes its frames. `None` takes the shell's capture
/// directory, the same one the capture panel uses.
pub(crate) struct RecordRequest {
    pub(crate) dir: Option<std::path::PathBuf>,
}

// The runner drains capture requests AFTER `update` and taps the swapchain
// after that, so a stop enqueued on the tick the sequence ends would discard
// that tick's frame, which is the pose the whole sequence builds to. One frame
// of lag records it and then stops.
enum Recording {
    Running,
    LastFrameQueued,
    Stopped,
}

impl Recording {
    fn advance(&mut self, finished: bool) {
        match self {
            Recording::Running if finished => *self = Recording::LastFrameQueued,
            Recording::LastFrameQueued => {
                loam_app::capture::enqueue(CaptureRequest::Stop);
                loam_app::script::request_exit();
                *self = Recording::Stopped;
            }
            _ => {}
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
    recording: Option<Recording>,
}

impl HeroScene {
    fn build_console() -> Console<Environment> {
        let mut console = Console::<Environment>::new();
        loam_app::shell::register_shell_commands::<Environment, crate::Hero>(
            &mut console,
            loam_app::build_info!(),
        );
        register_ground_command(&mut console, |env| env);
        register_floor_command(&mut console, |env| env);
        console
    }

    pub(crate) fn new(ctx: &mut SetupCtx<'_>, record: Option<RecordRequest>) -> Result<Self> {
        let console = Self::build_console();
        let recording = record.map(|record| {
            // The sequence advances on ticks and the tap fires on rendered
            // frames, so the recording only runs at the sequence's own speed
            // where the two are locked: uncapped, this machine rendered 1760
            // frames for 870 ticks.
            loam_app::frame_pacing::set_target_fps(TICK_HZ as f32);
            // APNG rather than a PNG sequence: one lossless file and no
            // external encoder. The cost is that the worker holds every frame
            // in memory until the stop, which is what bounds a run to this
            // length. Pre-egui, so the console never lands in a frame.
            loam_app::capture::enqueue(CaptureRequest::StartSequence {
                format: CaptureFormat::Apng,
                stage: CaptureStage::Pre,
                dir: record.dir,
                name: Some("hero".to_string()),
                fps: Some(RECORD_FPS),
                scale: None,
                palette: PaletteMode::default(),
            });
            Recording::Running
        });

        let sequence = HeroSequence::new(hero_font_bytes(), DEFAULT_SEED)?;
        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, BOOT_EYE_HEIGHT, BOOT_ORBIT_DISTANCE);
        let mut orbit: OrbitController<EuclideanR3> =
            OrbitController::around(sequence.word_centre());
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);

        Ok(Self {
            sequence,
            seed: DEFAULT_SEED,
            camera,
            orbit,
            console,
            environment: Environment::default(),
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
            recording,
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
        if let Some(recording) = &mut self.recording {
            recording.advance(self.sequence.finished());
        }
        let cfg = &ctx.rd.surface_bundle.config;
        self.camera.aspect = cfg.width as f32 / cfg.height.max(1) as f32;
        if !ctx.ui_capture.pointer {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt);
        }
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
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
                self.environment
                    .ground(FLOOR_Y, self.environment.floor_visible),
            ),
        );
        self.sky_ground
            .record(ctx.encoder, ctx.view, &depth.view, None);

        build_frame_mesh(
            &mut self.sequence,
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
        Cow::Borrowed("loam")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_physics::manifold::PENETRATION_SLOP;

    const SEED: u64 = 0x10a3_5eed;

    // Ticks over which a settled letter must hold still, and the position
    // spread allowed across them. The measured worst is 0.0063 em on `L`, and
    // what is left at rest is the Baumgarte limit cycle: gravity sinks the
    // letter over a step and the positional bias pushes it back out, forever,
    // at the slop's amplitude.
    const REST_WINDOW_TICKS: u32 = 60;
    const REST_SPREAD: f32 = 0.02;

    // Displacement, in em, that reads as knocked aside rather than nudged.
    const SCATTER_THRESHOLD: f32 = 0.25;

    // A tunneling bound, not a contact bound: the resting overlap the solver
    // holds is `PENETRATION_SLOP`, and an impact transiently reaches 0.034, so
    // a body 0.075 under would be one that the contact never caught.
    const TUNNEL_DEPTH: f32 = 0.075;

    // 1.5x the measured peak. The mesh is rebuilt and re-uploaded every frame,
    // so its size is a per-frame bandwidth cost rather than a one-time bake.
    const MESH_VERTEX_BUDGET: u32 = 42_000;

    // Per-step travel, in em, the R⁴ narrowphase still resolves against a thin
    // static wall, as recorded by `loam_physics::world`'s tunneling gate.
    // Sampling the floor test once per tick is sound only if no body can cross
    // and return inside a tick, which is what bounding the travel buys.
    const RESOLVABLE_STEP_TRAVEL: f32 = 0.150;

    fn scene() -> HeroSequence {
        HeroSequence::new(hero_font_bytes(), SEED).expect("hero scene")
    }

    // Indexes the INKED solids, which is what every caller enumerates, so a
    // word carrying a blank would otherwise mislabel every message downstream.
    fn label(index: usize) -> char {
        WORD.chars()
            .filter(|c| !c.is_whitespace())
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
        // Counts differ by seed now: the rain spawns on a jittered interval
        // until the freeze rather than to a fixed total, so the seed reaches
        // how many fell as well as where.
        assert!(
            drops_a.len() != drops_b.len() || drops_a.iter().zip(&drops_b).any(|(a, b)| a != b),
            "two seeds rained identically, so the seed is decoration"
        );
    }

    #[test]
    fn the_director_owns_a_letter_until_it_becomes_a_body_and_never_after() {
        let mut scene = scene();
        assert_eq!(scene.world.bodies.iter().count(), 1, "floor only");
        scene.run_to(ASSEMBLE_TICKS - 1);
        assert_eq!(
            scene.world.bodies.iter().count(),
            1,
            "a letter became a body while the director still owned it"
        );

        let directed = letter_positions(&scene);
        let frame_at_release = scene.director.frame() + 1;
        scene.tick();
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
        while scene.tick < RAIN_START_TICK {
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
    fn every_letter_is_moved_by_the_rain_and_none_is_launched_out_of_the_word() {
        let mut s = scene();
        s.run_to(RAIN_START_TICK);
        let settled = letter_positions(&s);
        s.run_to(SEQUENCE_TICKS);

        // A band, not the four measured displacements: the pile is chaotic, so
        // exact figures re-record on any solver or constant change and pin
        // nothing. Both edges are regressions that happened. Under the floor,
        // the rain stopped reaching the letters at all; over it, scenery
        // friction stopped braking `w` and a letter slid 4.4 em out of a
        // 3.5 em word.
        for (index, (before, after)) in settled.iter().zip(&letter_positions(&s)).enumerate() {
            let moved = before.distance(*after);
            assert!(
                (0.1..2.0).contains(&moved),
                "{:?} moved {moved} em, outside the band the rain should leave it in",
                label(index)
            );
        }
    }

    #[test]
    fn the_rain_knocks_at_least_one_letter_past_the_scatter_threshold() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK);
        let settled = letter_positions(&scene);
        scene.run_to(SEQUENCE_TICKS);
        assert!(
            scene.drops().len() > 20,
            "only {} drops spawned before the freeze",
            scene.drops().len()
        );

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
                scene.tick
            );
            let travel = scene.fastest_body_speed() * SOLVER_DT;
            assert!(
                travel < RESOLVABLE_STEP_TRAVEL,
                "a body travelled {travel} in one step at tick {}",
                scene.tick
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
        assert!(scene.drops().len() > 20);
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
            let exact = regular_polytope4_inertia(drop.polytope(), RAIN_MASS, RAIN_SIZE);
            assert_eq!(body.inertia, exact, "drop {index} kept the ball's moment");
        }
        // The two large polychora were spheres until the narrowphase stopped
        // materialising world vertices through a fixed buffer. They are the
        // reason that cap is gone, so the rain is where their absence would
        // show first.
        for large in [Polytope4::Cell120, Polytope4::Cell600] {
            assert!(
                RAIN_SHAPES.contains(&large),
                "{large:?} dropped out of the rain"
            );
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
    fn the_assembled_word_is_centred_on_what_the_camera_aims_at() {
        let scene = HeroSequence::new(hero_font_bytes(), DEFAULT_SEED).expect("scene");
        let centre = scene.word_centre();
        // The orbit controller aims the camera at its target, so a target on
        // the word's bounding-box centre IS the word centred in frame. The
        // regression this catches is the layout pen: it starts at x = 0 and
        // advances right, so an unshifted word sits wholly to one side.
        assert!(
            centre.x.abs() < 1e-5,
            "the word centres at x {}, so it hangs off one side of frame",
            centre.x
        );
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for letter in scene.letters() {
            for v in &letter.hull {
                lo = lo.min(letter.mark.x + v.x);
                hi = hi.max(letter.mark.x + v.x);
            }
        }
        assert!(
            (lo + hi).abs() < 1e-5,
            "the word spans [{lo}, {hi}] em, which is not symmetric about the aim"
        );
        assert!(
            centre.y > 0.0 && centre.y < 1.0,
            "the word centres at y {}, outside a one-em letter's own height",
            centre.y
        );
    }

    fn bounds_of(mesh: &TriangleMesh<3>) -> Option<(Vec3, Vec3)> {
        mesh.vertices
            .iter()
            .map(|v| (Vec3::from_array(*v), Vec3::from_array(*v)))
            .reduce(|(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
    }

    // Bounds of everything the frame drew, which is how a section's identity
    // is compared without pinning vertex positions a re-bake would move.
    fn letter_section_bounds(scene: &mut HeroSequence, index: usize) -> (Vec3, Vec3) {
        let pose = scene.letter_pose(index);
        let u = index as f32 - (pose.position.w - W_SLICE) / W_PER_LETTERFORM;
        let field = scene.morph.blend_at(u).expect("the blend has a grid");
        let mut mesh = TriangleMesh::<3>::default();
        assert!(
            loam_text::glyph::append_field_prism(
                &field,
                0.5 * GlyphParams::default().depth,
                LETTER_COLOR,
                &mut mesh,
            ),
            "letter {index} cut an empty section"
        );
        bounds_of(&mesh).expect("a non-empty section has bounds")
    }

    #[test]
    fn the_slice_holds_while_the_sim_runs_and_sweeps_once_it_freezes() {
        let mut scene = scene();
        for tick in 0..PHYSICS_PAUSE_TICK {
            scene.run_to(tick);
            assert!(
                (scene.slice() - W_SLICE).abs() < 1e-6,
                "the slice moved to {} at tick {tick}, while the pile was still rolling",
                scene.slice()
            );
        }

        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for tick in PHYSICS_PAUSE_TICK..SEQUENCE_TICKS {
            scene.run_to(tick);
            lo = lo.min(scene.slice());
            hi = hi.max(scene.slice());
        }
        // A full sweep either side, which is what carries every letter through
        // every other letterform.
        assert!(
            hi > 0.9 * SLICE_SWEEP_RANGE && lo < -0.9 * SLICE_SWEEP_RANGE,
            "the sweep only reached [{lo}, {hi}] of +/-{SLICE_SWEEP_RANGE}"
        );
    }

    #[test]
    fn nothing_moves_once_the_sim_is_frozen_except_the_slice() {
        let mut scene = scene();
        scene.run_to(PHYSICS_PAUSE_TICK);
        let poses: Vec<Vec4> = (0..scene.drops().len())
            .map(|i| scene.drop_pose(i).position)
            .collect();
        let letters: Vec<Vec4> = (0..scene.letters().len())
            .map(|i| scene.letter_pose(i).position)
            .collect();

        // Checked on the way through rather than at the end: `run_to` only
        // goes forwards, and a whole sweep lands back on the resting slice by
        // construction, so the end is exactly where the sweep is invisible.
        let mut swept = 0.0f32;
        for tick in PHYSICS_PAUSE_TICK..SEQUENCE_TICKS {
            scene.run_to(tick);
            swept = swept.max((scene.slice() - W_SLICE).abs());
            for (i, was) in poses.iter().enumerate() {
                assert_eq!(
                    scene.drop_pose(i).position,
                    *was,
                    "drop {i} moved at tick {tick}, after the freeze"
                );
            }
            for (i, was) in letters.iter().enumerate() {
                assert_eq!(
                    scene.letter_pose(i).position,
                    *was,
                    "letter {i} moved at tick {tick}, after the freeze"
                );
            }
        }
        assert!(
            swept > 0.9 * SLICE_SWEEP_RANGE,
            "the slice froze along with everything else, reaching only {swept}"
        );
    }

    #[test]
    fn a_letters_pose_stays_a_rotation_of_the_slice_however_hard_it_is_hit() {
        let mut scene = scene();
        let mut worst = 1.0f32;
        let mut worst_tick = 0;
        for tick in ASSEMBLE_TICKS..SEQUENCE_TICKS {
            scene.run_to(tick);
            for index in 0..scene.letters().len() {
                let r = scene.letter_pose(index).rotor;
                // The block the draw actually uses: the rotor applied to each
                // local axis, truncated. A 4D rotation's 3x3 block is NOT a
                // rotation, and when the letter tumbles into a `w` plane it
                // goes singular and the letter flattens to a sheet. Holding
                // the letters' spin inside the slice is what keeps this at 1.
                let det = r.apply(Vec4::X).truncate().dot(
                    r.apply(Vec4::Y)
                        .truncate()
                        .cross(r.apply(Vec4::Z).truncate()),
                );
                if (det - 1.0).abs() > (worst - 1.0).abs() {
                    worst = det;
                    worst_tick = tick;
                }
            }
        }
        assert!(
            (worst - 1.0).abs() < 1e-3,
            "a letter's drawn basis had determinant {worst} at tick {worst_tick}, so it is being scaled rather than rotated"
        );
    }

    #[test]
    fn the_rain_still_tumbles_through_the_w_planes_the_letters_are_held_out_of() {
        let mut scene = scene();
        scene.run_to(SEQUENCE_TICKS);
        // The deviation is the LETTERS', not the scene's: a drop is sliced
        // properly, so it can tumble anywhere and show it honestly.
        let spun = (0..scene.drops().len())
            .map(|i| {
                let spin = scene.world.bodies[scene.drops()[i].body].angular_velocity;
                spin.xw.abs() + spin.yw.abs() + spin.zw.abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            spun > 0.1,
            "the rain lost its w-plane tumble too, at {spun}"
        );

        for letter in scene.letters() {
            let body = letter.body.expect("released");
            let spin = scene.world.bodies[body].angular_velocity;
            assert_eq!((spin.xw, spin.yw, spin.zw), (0.0, 0.0, 0.0));
        }
    }

    #[test]
    fn the_rain_falls_through_itself_and_stacks_only_once_it_has_landed() {
        let mut scene = scene();
        scene.run_to(RAIN_START_TICK + 20);
        let falling: Vec<BodyId> = (scene.drops().iter())
            .map(|d| d.body)
            .filter(|id| scene.world.bodies[*id].collision_group == GROUP_FALLING)
            .collect();
        assert!(
            !falling.is_empty(),
            "nothing was in the air 20 ticks into the rain"
        );

        // No manifold ever holds two bodies that are both still falling.
        for tick in RAIN_START_TICK..PHYSICS_PAUSE_TICK {
            scene.run_to(tick);
            for (key, manifold) in &scene.world.manifolds {
                if manifold.points.is_empty() {
                    continue;
                }
                let air = |id: BodyId| scene.world.bodies[id].collision_group == GROUP_FALLING;
                assert!(
                    !(air(key.0) && air(key.1)),
                    "two airborne drops met at tick {tick}"
                );
            }
        }

        // And landing is not a one-way trip into never colliding: by the
        // freeze most of the rain has joined the pile.
        let landed = (scene.drops().iter())
            .filter(|d| scene.world.bodies[d.body].collision_group == GROUP_LANDED)
            .count();
        assert!(
            landed * 2 > scene.drops().len(),
            "only {landed} of {} drops ever touched anything",
            scene.drops().len()
        );
    }

    #[test]
    fn the_floor_never_slides_a_letter_along_w_but_the_rain_still_does() {
        let mut scene = scene();
        // The whole settle, where the only thing touching a letter is scenery.
        for tick in ASSEMBLE_TICKS..RAIN_START_TICK {
            scene.run_to(tick);
            for (index, letter) in scene.letters().iter().enumerate() {
                let body = letter.body.expect("released");
                let w = scene.world.bodies[body].position.w;
                assert!(
                    (w - letter.mark.w).abs() < 1e-6,
                    "{:?} slid to w {w} on the floor alone at tick {tick}",
                    label(index)
                );
            }
        }

        // And the rain still drives them off the slice, which is the effect
        // the deviation above exists to protect rather than to suppress.
        scene.run_to(PHYSICS_PAUSE_TICK);
        let pushed = (scene.letters().iter())
            .map(|l| {
                let body = l.body.expect("released");
                (scene.world.bodies[body].position.w - l.mark.w).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            pushed > 0.05,
            "the rain moved no letter further than {pushed} in w, so the letters never morph"
        );
    }

    #[test]
    fn at_its_mark_every_letter_cuts_its_own_letterform() {
        let mut scene = scene();
        scene.run_to(ASSEMBLE_TICKS);
        for index in 0..scene.letters().len() {
            let (lo, hi) = letter_section_bounds(&mut scene, index);
            // Oracle built independently of the scene: the glyph's own mesh,
            // straight from the font, so the blend is checked against the
            // letterform and not against another copy of itself.
            let font = ab_glyph::FontRef::try_from_slice(hero_font_bytes()).expect("font");
            let solids = layout_word(&font, WORD, &GlyphParams::default()).expect("layout");
            let solid = solids
                .iter()
                .filter(|s| !s.is_blank())
                .nth(index)
                .expect("a solid per letter");
            let own = bounds_of(&Visualizable::<3>::to_triangles(solid).expect("glyph mesh"))
                .expect("baked mesh");
            // The morph grid is centred on each glyph and resampled at the
            // same pitch it was baked at, so at its own letterform the section
            // reproduces the glyph to within a cell.
            let cell = GlyphParams::default().em_size / GlyphParams::default().resolution as f32;
            let (want, got) = (own.1 - own.0, hi - lo);
            assert!(
                (want.x - got.x).abs() < 2.0 * cell && (want.y - got.y).abs() < 2.0 * cell,
                "{:?} settled on a section {got:?} against its own {want:?}",
                label(index)
            );
        }
    }

    #[test]
    fn a_letter_mid_approach_is_a_different_letterform_and_not_a_scaled_copy() {
        let mut scene = scene();
        scene.run_to(ASSEMBLE_TICKS);
        let settled: Vec<(Vec3, Vec3)> = (0..scene.letters().len())
            .map(|i| letter_section_bounds(&mut scene, i))
            .collect();

        let mut approaching = HeroSequence::new(hero_font_bytes(), DEFAULT_SEED).expect("scene");
        approaching.run_to(LETTER_SLIDE_TICKS / 2);
        let mid = letter_section_bounds(&mut approaching, 0);
        let own = settled[0];

        // Not a scaled copy: a pure scale about a point keeps the aspect
        // ratio, so a changed ratio is the signature of a genuine morph. This
        // is exactly what the bicone could not do, and what an opacity fade
        // stood in for before that.
        let ratio = |(lo, hi): (Vec3, Vec3)| (hi.x - lo.x) / (hi.y - lo.y);
        assert!(
            (ratio(mid) - ratio(own)).abs() > 0.05,
            "mid-approach aspect {} matches its own {}, so the section is only scaling",
            ratio(mid),
            ratio(own)
        );
        // And it is never empty on the way in, which the taper was.
        assert!(mid.1.y - mid.0.y > 0.1, "the approach drew almost nothing");
    }

    #[test]
    fn the_letterform_sequence_wraps_so_neither_direction_runs_off_the_word() {
        let mut scene = scene();
        let count = scene.letters().len() as f32;
        // Far outside `[0, count)` in both directions: the word is a loop, so
        // every `u` names a blend and none falls off an end.
        for step in -20..=20 {
            let u = step as f32 * 0.37 * count;
            assert!(
                scene.morph.blend_at(u).is_some(),
                "the blend at u = {u} has no field"
            );
        }
    }

    #[test]
    fn the_assembly_slides_every_letter_onto_its_mark_before_the_release() {
        let mut scene = scene();
        let entries = letter_positions(&scene);
        for (index, letter) in scene.letters().iter().enumerate() {
            assert_eq!(entries[index], letter.entry);
            // The whole point of the entrance: a letter arrives by crossing
            // into the slice, not by sliding across it. A 3D translation here
            // is the regression this pins.
            let offset = letter.entry + letter.mark * -1.0;
            assert!(
                offset.w.abs() > W_PER_LETTERFORM,
                "{:?} starts less than a letterform away, so it never morphs",
                label(index)
            );
            assert!(
                offset.truncate().length() < 1e-6,
                "{:?} enters by a 3D translation of {:?}",
                label(index),
                offset.truncate()
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
        build_frame_mesh(&mut scene, &mut local, &mut scratch, &mut mesh);

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
        build_frame_mesh(&mut scene, &mut local, &mut scratch, &mut mesh);
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

        // Per-letter spans are gone: a morph section has its own vertex
        // count per letter and per frame, so the bound is taken over the
        // whole drawn word instead. It still fails if any single letter sinks
        // or floats, which is the property.
        let mut mesh = TriangleMesh::<3>::default();
        push_letters(&mut scene, &mut mesh);
        assert!(!mesh.vertices.is_empty(), "the settled word drew nothing");
        let drawn = mesh.vertices.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
        assert!(
            drawn > -2.0 * PENETRATION_SLOP,
            "the word is drawn {drawn} below the floor"
        );
        assert!(
            drawn < margin + 2.0 * PENETRATION_SLOP,
            "the word floats {drawn} above the floor, past the {margin} margin"
        );
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
}
