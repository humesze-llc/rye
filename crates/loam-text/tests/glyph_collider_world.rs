//! A laid-out word given physics bodies: the acceptance the glyph collider
//! path exists for.
//!
//! The word is spawned as static geometry, per `GlyphSolid::colliders_4d`'s
//! static-only contract, and probed by dynamic spheres falling along `-z` onto
//! the letters' front faces. Both halves of a faithful silhouette are checked:
//! a sphere over ink lands on it, and a sphere over the counter of `O` falls
//! through. A per-letter convex hull, or a bounding volume, passes the first
//! and fails the second.
//!
//! Fonts are not vendored, so every test here skips cleanly when the machine
//! has none.

use ab_glyph::FontRef;
use glam::{Vec2, Vec4};

use loam_math::EuclideanR4;
use loam_physics::euclidean_r4::{register_default_narrowphase, sphere_body_r4};
use loam_physics::{BodyId, Gravity, RigidBody, World};
use loam_shape::Visualizable;
use loam_text::glyph::{layout_word, GlyphParams, GlyphSolid};

const WORD: &str = "LOAM";
const DT: f32 = 1.0 / 120.0;
const BALL_RADIUS: f32 = 0.03;
const DROP_Z: f32 = 0.25;
/// Steps before the ball is expected to be resting: the fall from [`DROP_Z`]
/// to the front face takes 21 of them under `g = 9.8`.
const IMPACT_STEPS: usize = 30;
/// Measured budget at [`GlyphParams::default`], with the same ~12% headroom the
/// in-crate pin uses. A regression past this is a solver cost, not a detail.
const COLLIDER_BUDGET: usize = 108;

fn system_font() -> Option<Vec<u8>> {
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ];
    CANDIDATES
        .iter()
        .find_map(|path| std::fs::read(path).ok())
        .or_else(|| {
            eprintln!("skip: no system font found in {CANDIDATES:?}");
            None
        })
}

fn word() -> Option<Vec<GlyphSolid>> {
    let bytes = system_font()?;
    let font = FontRef::try_from_slice(&bytes).expect("parse font");
    Some(layout_word(&font, WORD, &GlyphParams::default()).expect("layout"))
}

/// The word as static bodies under gravity along `-z`, i.e. into the letters'
/// back faces, so a falling body meets a front face rather than a silhouette
/// edge one slab thick.
fn word_world(letters: &[GlyphSolid]) -> World<EuclideanR4> {
    let mut world = World::new(EuclideanR4);
    register_default_narrowphase(&mut world.narrowphase);
    world.push_field(Box::new(Gravity::new(Vec4::new(0.0, 0.0, -9.8, 0.0))));
    for letter in letters {
        for (centre, hull) in letter.colliders_4d() {
            world.push_body(RigidBody::fixed(centre, hull, 1.0, &EuclideanR4));
        }
    }
    world
}

fn drop_ball(world: &mut World<EuclideanR4>, at: Vec2) -> BodyId {
    world.push_body(sphere_body_r4(
        Vec4::new(at.x, at.y, DROP_Z, 0.0),
        Vec4::ZERO,
        BALL_RADIUS,
        1.0,
    ))
}

/// Mid-height of a letter's ink, taken from the render mesh so the drop points
/// do not depend on the cover under test.
fn ink_mid_y(letter: &GlyphSolid) -> f32 {
    let mesh = Visualizable::<3>::to_triangles(letter).expect("mesh");
    let lo = mesh.vertices.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
    let hi = mesh
        .vertices
        .iter()
        .fold(f32::NEG_INFINITY, |m, v| m.max(v[1]));
    0.5 * (lo + hi)
}

/// Runs of `x` the baked field calls ink along `y = mid`, as `(start, end)`.
/// A fixed lattice, so the runs are reproducible.
fn ink_runs(letter: &GlyphSolid, mid: f32) -> Vec<(f32, f32)> {
    const PROBES: usize = 2048;
    let x0 = letter.pen_origin().x - 0.25 * letter.advance();
    let width = 1.5 * letter.advance();
    let mut runs = Vec::new();
    let mut start = None;
    for k in 0..=PROBES {
        let x = x0 + width * k as f32 / PROBES as f32;
        let inside = letter.distance_2d(Vec2::new(x, mid)) <= 0.0;
        match (inside, start) {
            (true, None) => start = Some(x),
            (false, Some(from)) => {
                runs.push((from, x));
                start = None;
            }
            _ => {}
        }
    }
    runs
}

/// Centre of the widest stroke a letter presents at mid height, i.e. the
/// sturdiest place to land something on it.
fn widest_stroke(letter: &GlyphSolid) -> Vec2 {
    let mid = ink_mid_y(letter);
    let run = ink_runs(letter, mid)
        .into_iter()
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
        .expect("letter has ink at mid height");
    Vec2::new(0.5 * (run.0 + run.1), mid)
}

/// The count criterion, asserted where the count is actually paid: a laid-out
/// word becomes this many static bodies in a `World`, and the world steps.
#[test]
fn a_laid_out_word_spawns_a_solver_sized_set_of_static_bodies() {
    let Some(letters) = word() else { return };
    let boxes: usize = letters.iter().map(GlyphSolid::collider_count).sum();
    assert!(boxes > 0);
    assert!(boxes <= COLLIDER_BUDGET, "{WORD} emits {boxes} colliders");

    let mut world = word_world(&letters);
    assert_eq!(world.bodies.iter().count(), boxes);

    let ball = drop_ball(&mut world, widest_stroke(&letters[0]));
    for _ in 0..IMPACT_STEPS * 2 {
        world.step(DT);
        let position = world.bodies.get(ball).unwrap().position;
        assert!(position.is_finite(), "simulation diverged to {position}");
    }
}

/// A body dropped on ink is held up by the letter's front face, in the band a
/// sphere resting on a conservative cover of that face can occupy.
///
/// The window closes at twice the impact time on purpose. Nothing in
/// `loam-physics` resists rolling, so a sphere keeps whatever tangential
/// velocity the contact gave it and eventually walks off a stroke a tenth of
/// an em wide; that is a solver property, not a property of this
/// decomposition, and asserting on it would pin the wrong thing.
#[test]
fn a_body_dropped_on_a_letter_is_held_up_by_its_front_face() {
    let Some(letters) = word() else { return };
    let front = 0.5 * GlyphParams::default().depth;

    for letter in &letters {
        let mut world = word_world(&letters);
        let ball = drop_ball(&mut world, widest_stroke(letter));
        for step in 0..IMPACT_STEPS * 2 {
            world.step(DT);
            if step < IMPACT_STEPS {
                continue;
            }
            let z = world.bodies.get(ball).unwrap().position.z;
            assert!(
                z > front,
                "{:?}: ball sank to z = {z}, past the front face at {front}",
                letter.ch()
            );
            assert!(
                z < front + 2.0 * BALL_RADIUS,
                "{:?}: ball is floating at z = {z}",
                letter.ch()
            );
        }
        let body = world.bodies.get(ball).unwrap();
        assert!(
            body.velocity.z.abs() < 0.5,
            "{:?}: ball is still moving vertically at {}",
            letter.ch(),
            body.velocity.z
        );
    }
}

/// The counter of `O` is a hole a body falls through. This is what the cover
/// buys over a per-letter convex hull, and what an enclosure test alone cannot
/// see: a hull encloses the letter perfectly well and plugs the counter.
#[test]
fn a_body_dropped_down_the_counter_of_o_falls_through() {
    let Some(letters) = word() else { return };
    let o = letters.iter().find(|l| l.ch() == 'O').expect("O");
    let mid = ink_mid_y(o);
    let runs = ink_runs(o, mid);
    assert_eq!(runs.len(), 2, "'O' is not two strokes at mid height");

    let counter = Vec2::new(0.5 * (runs[0].1 + runs[1].0), mid);
    // Read off the field, not off the cover: the drop point is clear of the
    // ink by more than the cover is allowed to intrude, plus the ball.
    assert!(
        o.distance_2d(counter) > o.collider_margin() + BALL_RADIUS,
        "counter probe {counter} is not clear of the ink"
    );

    let mut world = word_world(&letters);
    let ball = drop_ball(&mut world, counter);
    for _ in 0..600 {
        world.step(DT);
    }
    let z = world.bodies.get(ball).unwrap().position.z;
    assert!(z < -DROP_Z, "ball stopped at z = {z} instead of falling");
}

/// Same word, same colliders, same trajectory bit for bit. The cover is a
/// fixed-order scan and the solver visits bodies in spawn order, so a
/// hundred-body static set must not introduce an order dependence.
#[test]
fn two_runs_over_a_word_agree_bit_for_bit() {
    let Some(letters) = word() else { return };
    let at = widest_stroke(&letters[3]);
    let trajectory = || {
        let mut world = word_world(&letters);
        let ball = drop_ball(&mut world, at);
        let mut samples = Vec::with_capacity(200);
        for _ in 0..200 {
            world.step(DT);
            samples.push(world.bodies.get(ball).unwrap().position.to_array());
        }
        samples
    };
    assert_eq!(trajectory(), trajectory());
}
